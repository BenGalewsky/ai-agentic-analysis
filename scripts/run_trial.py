#!/usr/bin/env python
"""Run Claude Code trials over an MLflow evaluation dataset of questions and report to MLflow.

A trial:
  1. Pulls a prompt (default: ``IRIS-HEP``) from the MLflow Prompt Registry.
  2. Loads the questions from an MLflow evaluation dataset (default:
     ``hep-data-llm-questions``, written by ``register_questions.py``).
  3. For each question, renders the prompt with the record's inputs, stages a clean
     workspace containing a snapshot of ``skills/`` and runs ``claude -p`` in it as a
     subprocess, streaming JSON events.
  4. Logs params, metrics, artifacts and an MLflow trace per question, each as a child
     run of one parent run for the whole trial.

Example:
    uv run scripts/run_trial.py
    uv run scripts/run_trial.py --question JetPtAll
    uv run scripts/run_trial.py --prompt IRIS-HEP --prompt-version 1 --model opus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from grader import grade, metrics_from_events
from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    SpanStatusCode,
    SpanType,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import get_dataset

PROJECT_ROOT = Path(".")
SKILLS_DIR = PROJECT_ROOT / "skills"
DEFAULT_TRIALS_DIR = Path(
    os.environ.get("TRIAL_WORKSPACE_ROOT")
    or Path.home() / ".cache" / "hep-agent-trials"
)

DEFAULT_EXPERIMENT = "hep-plot-agent"
DEFAULT_PROMPT = "IRIS-HEP"
DEFAULT_DATASET = "hep-data-llm-questions"
DEFAULT_ALLOWED_TOOLS = (
    "Bash Read Write Edit Glob Grep Skill WebFetch WebSearch TodoWrite mcp__af"
)
DEFAULT_MCP_CONFIG = Path(__file__).resolve().parent / "mcp.json"

SCRIPT_SUFFIXES = (".py",)
PLOT_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".svg")


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #
def resolve_prompt(name: str, version: str | int | None):
    """Load a prompt version; ``None`` means the highest registered version."""
    if version is None:
        versions = [int(v.version) for v in mlflow.MlflowClient().search_prompt_versions(name)]
        if not versions:
            raise SystemExit(f"No versions registered for prompt {name!r}")
        version = max(versions)
    return mlflow.genai.load_prompt(f"prompts:/{name}/{version}")


def render_prompt(prompt, variables: dict[str, str]) -> str:
    if not prompt.variables:
        return prompt.template
    missing = set(prompt.variables) - set(variables)
    if missing:
        raise SystemExit(f"Prompt {prompt.name!r} needs --var for: {', '.join(sorted(missing))}")
    return prompt.format(**variables)


# --------------------------------------------------------------------------- #
# Question dataset
# --------------------------------------------------------------------------- #
def load_questions(name: str) -> tuple[Any, list[dict[str, Any]]]:
    """Load the dataset and its records, in the order of their ``question_index`` tag."""
    dataset = get_dataset(name=name)
    records = dataset.to_df().to_dict("records")
    if not records:
        raise SystemExit(f"Dataset {name!r} has no records")
    records.sort(key=lambda r: int((r["tags"] or {}).get("question_index", 0)))
    return dataset, records


def question_name(record: dict[str, Any]) -> str:
    return (record["tags"] or {}).get("name") or record["dataset_record_id"]


def select_question(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Find a record by its ``name`` tag or its ``question_index`` tag."""
    for record in records:
        tags = record["tags"] or {}
        if key in (tags.get("name"), tags.get("question_index")):
            return record
    choices = ", ".join(question_name(r) for r in records)
    raise SystemExit(f"No question {key!r} in the dataset; choose a name or index from: {choices}")


# --------------------------------------------------------------------------- #
# Workspace staging
# --------------------------------------------------------------------------- #
def hash_skills(skills_dir: Path) -> str:
    """Content hash over every skill file, so a run pins the exact skill revision."""
    digest = hashlib.sha256()
    for path in sorted(p for p in skills_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(skills_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def stage_workspace(root: Path) -> Path:
    """Create an empty workspace with ``skills/`` copied in as project skills."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    shutil.copytree(SKILLS_DIR, workspace / ".claude" / "skills")
    return workspace


def find_deliverables(outputs: Path) -> dict[str, Path]:
    """Pick the final script and plot out of everything the agent wrote.

    "Final" is the most recently modified file of each kind, which is what the
    agent lands on after any iteration.
    """

    def newest(suffixes: tuple[str, ...]) -> Path | None:
        candidates = [
            path
            for path in outputs.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        return max(candidates, key=lambda path: path.stat().st_mtime, default=None)

    found = {"script": newest(SCRIPT_SUFFIXES), "plot": newest(PLOT_SUFFIXES)}
    return {kind: path for kind, path in found.items() if path is not None}


# --------------------------------------------------------------------------- #
# Claude Code subprocess
# --------------------------------------------------------------------------- #
def build_command(args: argparse.Namespace) -> list[str]:
    """Build the CLI invocation. The prompt goes over stdin, not argv: the
    ``--allowed-tools <tools...>`` option is variadic and would swallow a
    trailing positional prompt.
    """
    cmd = [
        args.claude_bin,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        args.permission_mode,
        "--no-session-persistence",
    ]
    for config in args.mcp_config:
        # The agent runs in a staged workspace, so relative paths from the repo
        # root would not resolve; inline JSON strings are passed through as-is.
        value = config if config.lstrip().startswith("{") else str(Path(config).resolve())
        cmd += ["--mcp-config", value]
    if args.strict_mcp_config:
        cmd.append("--strict-mcp-config")
    if args.allowed_tools:
        cmd += ["--allowed-tools", args.allowed_tools]
    if args.model:
        cmd += ["--model", args.model]
    if args.max_budget_usd:
        cmd += ["--max-budget-usd", str(args.max_budget_usd)]
    return cmd


def read_mcp_config(config: str) -> str:
    return config if config.lstrip().startswith("{") else Path(config).read_text()


def mcp_server_names(configs: list[str]) -> list[str]:
    """Server names declared across the MCP configs, for logging/provenance."""
    names: list[str] = []
    for config in configs:
        names += list(json.loads(read_mcp_config(config)).get("mcpServers", {}))
    return sorted(set(names))


def check_mcp_env(configs: list[str]) -> None:
    """Fail before launch on an unset ``${VAR}`` in a config.

    Claude Code passes an unexpanded placeholder through verbatim, so a missing
    token surfaces only as an authentication failure once the trial is running.
    """
    missing = {
        name
        for config in configs
        for name, default in re.findall(r"\$\{(\w+)(:-[^}]*)?\}", read_mcp_config(config))
        if not default and name not in os.environ
    }
    if missing:
        raise SystemExit(
            f"MCP config needs unset environment variable(s): {', '.join(sorted(missing))} "
            "(expected in .env)"
        )


def run_claude(
    cmd: list[str], prompt_text: str, workspace: Path, stream_path: Path, timeout: int
) -> tuple[list[dict[str, Any]], int, str]:
    """Run Claude Code, tee the event stream to disk, and return parsed events."""
    events: list[dict[str, Any]] = []
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}

    with stream_path.open("w") as stream_file:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(prompt_text)
        proc.stdin.close()
        try:
            for line in proc.stdout:
                stream_file.write(line)
                stream_file.flush()
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(event)
                log_event(event)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        stderr = proc.stderr.read() if proc.stderr else ""

    return events, proc.returncode, stderr


def log_event(event: dict[str, Any]) -> None:
    """Minimal console trace so a long trial is watchable."""
    kind = event.get("type")
    if kind == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                print(f"  [assistant] {block['text'].strip()[:160]}")
            elif block.get("type") == "tool_use":
                print(f"  [tool] {block.get('name')}")
    elif kind == "result":
        print(f"  [result] {event.get('subtype')} in {event.get('duration_ms', 0) / 1000:.1f}s")


# --------------------------------------------------------------------------- #
# Stream -> MLflow trace
# --------------------------------------------------------------------------- #
def tool_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ``tool_use`` blocks and pair them with their results."""
    results: dict[str, Any] = {}
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block.get("content")

    calls = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if block.get("type") == "tool_use":
                calls.append(
                    {
                        "name": block.get("name"),
                        "input": block.get("input"),
                        "output": results.get(block.get("id")),
                    }
                )
    return calls


def final_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("result") or ""
    return ""


def log_trace(prompt_text: str, events: list[dict[str, Any]], result: dict[str, Any]) -> str:
    """Record the trial as one MLflow trace with a child span per tool call; return its ID."""
    root = mlflow.start_span_no_context(
        name="claude_code_trial",
        span_type=SpanType.AGENT,
        inputs={"prompt": prompt_text},
    )
    try:
        for call in tool_blocks(events):
            child = mlflow.start_span_no_context(
                name=call["name"] or "tool",
                span_type=SpanType.TOOL,
                parent_span=root,
                inputs=call["input"],
            )
            child.end(outputs={"result": call["output"]})

        root.set_attribute("num_turns", result.get("num_turns"))
        root.set_attribute("total_cost_usd", result.get("total_cost_usd"))
        root.end(
            outputs={"result": final_text(events)},
            status=SpanStatusCode.ERROR if result.get("is_error") else SpanStatusCode.OK,
        )
        return root.trace_id
    except Exception:
        root.end(status=SpanStatusCode.ERROR)
        raise


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Registered prompt name")
    parser.add_argument("--prompt-version", default=None, help="Prompt version (default: latest)")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="MLflow evaluation dataset whose records' inputs fill the prompt",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        metavar="NAME_OR_INDEX",
        help="Run only this question, by its name tag (e.g. JetPtAll) or question_index "
        "(repeatable; default: every question in the dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N questions, by question_index (applied after --question)",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Value for a prompt template variable (repeatable; overrides the dataset's inputs)",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default=None, help="Model alias passed to claude, e.g. opus")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument(
        "--permission-mode",
        default="bypassPermissions",
        choices=["acceptEdits", "auto", "bypassPermissions", "dontAsk", "plan"],
        help="Claude Code permission mode for the trial (default lets the agent run unattended)",
    )
    parser.add_argument(
        "--allowed-tools",
        default=DEFAULT_ALLOWED_TOOLS,
        help="Tools the agent may use without prompting (space-separated)",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=DEFAULT_TRIALS_DIR,
        help="Where trial workspaces are staged (default: $TRIAL_WORKSPACE_ROOT or ~/.cache/hep-agent-trials)",
    )
    parser.add_argument(
        "--mcp-config",
        action="append",
        default=None,
        metavar="PATH_OR_JSON",
        help=f"MCP server config file or inline JSON (repeatable, default: {DEFAULT_MCP_CONFIG})",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Run without any --mcp-config (the default config is skipped)",
    )
    parser.add_argument(
        "--strict-mcp-config",
        action="store_true",
        help="Ignore user/project MCP settings, so only --mcp-config servers are loaded",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Subprocess timeout in seconds")
    parser.add_argument("--max-budget-usd", type=float, default=None)
    return parser.parse_args()


def run_question(
    args: argparse.Namespace,
    prompt,
    record: dict[str, Any],
    variables: dict[str, str],
    trial_dir: Path,
    common_params: dict[str, Any],
    tracking_uri: str,
) -> dict[str, Any]:
    """Run one question as a child run of the active trial run and return its summary."""
    name = question_name(record)
    tags = record["tags"] or {}
    prompt_text = render_prompt(prompt, {**record["inputs"], **variables})
    question_dir = trial_dir / name
    workspace = stage_workspace(question_dir)
    print(f"workspace: {workspace}")

    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.log_params(
            {
                **common_params,
                "question_name": name,
                "question_index": tags.get("question_index", ""),
                "dataset_record_id": record["dataset_record_id"],
            }
        )
        mlflow.set_tags({"question": name, "datasets": tags.get("datasets", "")})
        mlflow.log_text(prompt_text, "prompt.txt")
        mlflow.log_dict(record["inputs"], "inputs.json")
        if record.get("expectations"):
            mlflow.log_dict(record["expectations"], "expectations.json")

        stream_path = question_dir / "claude_stream.jsonl"
        started = time.time()
        try:
            events, returncode, stderr = run_claude(
                build_command(args), prompt_text, workspace, stream_path, args.timeout
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            events, returncode, stderr, timed_out = [], -1, "timeout", True

        wall_seconds = time.time() - started
        result = next((e for e in reversed(events) if e.get("type") == "result"), {})
        usage = result.get("usage", {}) or {}
        succeeded = returncode == 0 and not result.get("is_error") and not timed_out

        mlflow.log_metrics(
            {
                "wall_seconds": wall_seconds,
                "duration_ms": result.get("duration_ms", 0),
                "api_duration_ms": result.get("duration_api_ms", 0),
                "num_turns": result.get("num_turns", 0),
                "total_cost_usd": result.get("total_cost_usd", 0.0),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "num_tool_calls": len(tool_blocks(events)),
                "completed": int(succeeded),
            }
        )
        failure_reason = result.get("subtype") or ("timeout" if timed_out else f"exit-{returncode}")
        mlflow.set_tags(
            {
                "status": "success" if succeeded else "failure",
                "failure_reason": "" if succeeded else failure_reason,
                "claude_session_id": result.get("session_id", ""),
            }
        )

        if stream_path.exists():
            mlflow.log_artifact(str(stream_path))
        if result:
            mlflow.log_dict(result, "result.json")
        if stderr:
            mlflow.log_text(stderr, "stderr.txt")
        mlflow.log_text(final_text(events), "final_output.md")

        # The agent's own output: everything it wrote, minus the staged skills.
        outputs = question_dir / "outputs"
        shutil.copytree(workspace, outputs, ignore=shutil.ignore_patterns(".claude"))
        if any(outputs.rglob("*")):
            mlflow.log_artifacts(str(outputs), artifact_path="outputs")

        # Promote the two deliverables that matter to a predictable artifact path.
        deliverables = find_deliverables(outputs)
        for kind, path in deliverables.items():
            mlflow.log_artifact(str(path), artifact_path="final")
            print(f"{kind:9}: {path.relative_to(outputs)}")
        mlflow.set_tags(
            {
                f"final_{kind}": deliverables[kind].name if kind in deliverables else ""
                for kind in ("script", "plot")
            }
        )
        mlflow.log_metrics(
            {
                "produced_script": int("script" in deliverables),
                "produced_plot": int("plot" in deliverables),
            }
        )

        # Score the plots against the record's reference values.
        graded = grade(record.get("expectations"), metrics_from_events(events))
        mlflow.log_metrics(graded.mlflow_metrics())
        mlflow.set_tags({"grade": "pass" if graded.passed else "fail", "grade_message": graded.message})
        mlflow.log_dict(graded.to_dict(), "grade.json")

        if events:
            trace_id = log_trace(prompt_text, events, result)
            # Traces are exported in the background, and the server's auth layer
            # answers 403 for a trace it has not stored yet - wait for the export.
            mlflow.flush_trace_async_logging()
            try:
                mlflow.log_feedback(
                    trace_id=trace_id,
                    name="metrics_match",
                    value=graded.passed,
                    rationale=graded.message,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.CODE, source_id="grader.py"
                    ),
                    metadata={"tolerance": str(graded.tolerance)},
                )
            except MlflowException as e:
                # The grade is already on the run; don't lose the sweep over the trace copy.
                print(f"warning  : could not attach grade to trace {trace_id}: {e.message}")

        cost = result.get("total_cost_usd", 0) or 0
        print(f"run      : {run.info.run_id}  ({'success' if succeeded else 'FAILURE'})")
        print(f"cost     : ${cost:.4f} over {result.get('num_turns', 0)} turns")
        print(f"grade    : {'PASS' if graded.passed else 'FAIL'} - {graded.message}")
        print(f"ui       : {run_url(tracking_uri, run)}")

    return {
        "name": name,
        "succeeded": succeeded,
        "cost": cost,
        "wall_seconds": wall_seconds,
        "produced_plot": "plot" in deliverables,
        "correct": graded.passed,
    }


def run_url(tracking_uri: str, run) -> str:
    return (
        f"{tracking_uri.split('@')[-1]}/#/experiments/{run.info.experiment_id}"
        f"/runs/{run.info.run_id}"
    )


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set (expected in .env)")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    if args.no_mcp:
        args.mcp_config = []
    elif args.mcp_config is None:
        args.mcp_config = [str(DEFAULT_MCP_CONFIG)] if DEFAULT_MCP_CONFIG.exists() else []

    check_mcp_env(args.mcp_config)

    variables = dict(v.split("=", 1) for v in args.var)
    prompt = resolve_prompt(args.prompt, args.prompt_version)
    dataset, records = load_questions(args.dataset)
    if args.question:
        records = [select_question(records, key) for key in args.question]
    if args.limit is not None:
        records = records[: args.limit]
    # Fail on a template variable the dataset cannot fill before anything is launched.
    for record in records:
        render_prompt(prompt, {**record["inputs"], **variables})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trial_dir = args.trials_dir / f"{stamp}-{prompt.name}-v{prompt.version}"
    skills_hash = hash_skills(SKILLS_DIR)
    skill_names = sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir())

    print(f"prompt   : {prompt.name} v{prompt.version}")
    print(f"dataset  : {args.dataset} ({len(records)} question{'s' if len(records) != 1 else ''})")
    print(f"skills   : {', '.join(skill_names)} ({skills_hash})")
    if args.mcp_config:
        print(f"mcp      : {', '.join(mcp_server_names(args.mcp_config))}")
    print(f"trial dir: {trial_dir}")

    # Shared by the trial run and every question run, so child runs compare on their own.
    common_params = {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_uri": prompt.uri,
        "dataset_name": args.dataset,
        "dataset_id": dataset.dataset_id,
        "model": args.model or "default",
        "permission_mode": args.permission_mode,
        "allowed_tools": args.allowed_tools,
        "mcp_config": ",".join(args.mcp_config),
        "mcp_servers": ",".join(mcp_server_names(args.mcp_config)),
        "strict_mcp_config": args.strict_mcp_config,
        "skills_hash": skills_hash,
        "skills": ",".join(skill_names),
        "num_skills": len(skill_names),
    }

    run_name = args.run_name or f"{prompt.name}-v{prompt.version}-{stamp}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                **common_params,
                "question_filter": ",".join(args.question),
                "num_questions": len(records),
            }
        )
        mlflow.log_text(prompt.template, "prompt_template.txt")
        for i, config in enumerate(args.mcp_config):
            mlflow.log_text(read_mcp_config(config), f"mcp_config_{i}.json" if i else "mcp_config.json")
        mlflow.log_artifacts(str(SKILLS_DIR), artifact_path="skills")

        summaries = []
        for i, record in enumerate(records, 1):
            print(f"\n[{i}/{len(records)}] {question_name(record)}")
            summaries.append(
                run_question(
                    args, prompt, record, variables, trial_dir, common_params, tracking_uri
                )
            )

        num_completed = sum(s["succeeded"] for s in summaries)
        num_correct = sum(s["correct"] for s in summaries)
        total_cost = sum(s["cost"] for s in summaries)
        mlflow.log_metrics(
            {
                "num_completed": num_completed,
                "completion_rate": num_completed / len(summaries),
                "num_produced_plot": sum(s["produced_plot"] for s in summaries),
                "num_correct": num_correct,
                "accuracy": num_correct / len(summaries),
                "total_cost_usd": total_cost,
                "wall_seconds": sum(s["wall_seconds"] for s in summaries),
            }
        )
        all_succeeded = num_completed == len(summaries)
        mlflow.set_tag("status", "success" if all_succeeded else "failure")

        print(
            f"\ntrial    : {run.info.run_id}  ({num_completed}/{len(summaries)} completed, "
            f"{num_correct}/{len(summaries)} correct)"
        )
        for s in summaries:
            status = "ok  " if s["succeeded"] else "FAIL"
            verdict = "pass" if s["correct"] else "fail"
            print(f"  {status} {verdict} {s['name']:24} ${s['cost']:.4f}")
        print(f"cost     : ${total_cost:.4f}")
        print(f"ui       : {run_url(tracking_uri, run)}")

    return 0 if all_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
