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
    uv run scripts/run_trial.py --question JetPtAll --repeats 5
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
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


def skill_calls(calls: list[dict[str, Any]]) -> Counter[str]:
    """How many times the agent invoked each skill through the ``Skill`` tool."""
    return Counter(
        (call["input"] or {}).get("skill", "unknown") for call in calls if call["name"] == "Skill"
    )


def skill_file_reads(calls: list[dict[str, Any]]) -> int:
    """``Read`` calls on a staged skill's files, e.g. a reference doc beyond ``SKILL.md``."""
    return sum(
        call["name"] == "Read" and ".claude/skills/" in str((call["input"] or {}).get("file_path", ""))
        for call in calls
    )


def tool_errors(events: list[dict[str, Any]]) -> int:
    """Tool results flagged as errors: failed commands, bad edits, denied calls."""
    return sum(
        isinstance(block, dict) and block.get("type") == "tool_result" and bool(block.get("is_error"))
        for event in events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", []) or []
    )


def final_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("result") or ""
    return ""


def log_trace(
    prompt_text: str,
    events: list[dict[str, Any]],
    succeeded: bool,
    metrics: dict[str, float],
    tags: dict[str, str],
) -> str:
    """Record one repeat as an MLflow trace with a child span per tool call; return its ID.

    A repeat that produced no events (a timeout, a failed launch) still gets an
    error trace, so every repeat of a question is accounted for among its traces.
    """
    root = mlflow.start_span_no_context(
        name="claude_code_trial",
        span_type=SpanType.AGENT,
        inputs={"prompt": prompt_text},
        tags=tags,
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

        root.set_attributes(metrics)
        root.end(
            outputs={"result": final_text(events)},
            status=SpanStatusCode.OK if succeeded else SpanStatusCode.ERROR,
        )
        return root.trace_id
    except Exception:
        root.end(status=SpanStatusCode.ERROR)
        raise


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


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
        "--repeats",
        type=positive_int,
        default=1,
        metavar="N",
        help="Run each question N times, to measure how consistent the agent is (default: 1)",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Value for a prompt template variable (repeatable; overrides the dataset's inputs)",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the parent trial run (default: <prompt>-v<version>-<timestamp>)",
    )
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


# Question metrics summed over the repeats as `<key>_total`; a sum of the others
# (rates, relative errors) means nothing.
ADDITIVE_METRICS = (
    "wall_seconds",
    "duration_ms",
    "api_duration_ms",
    "num_turns",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "num_tool_calls",
    "num_tool_errors",
    "num_skill_calls",
)


def run_repeat(
    args: argparse.Namespace,
    prompt_text: str,
    record: dict[str, Any],
    repeat: int,
    question_dir: Path,
) -> dict[str, Any]:
    """Run one repeat of a question inside its active question run.

    Logs the repeat's artifacts (under ``r<k>/`` when there are several repeats)
    and its trace, tagged and annotated with its own metrics, and returns its
    summary; the question run's metrics are rolled up from these.
    """
    name = question_name(record)
    repeated = args.repeats > 1
    repeat_dir = question_dir / f"r{repeat}" if repeated else question_dir
    prefix = f"r{repeat}/" if repeated else ""
    workspace = stage_workspace(repeat_dir)
    print(f"workspace: {workspace}")

    stream_path = repeat_dir / "claude_stream.jsonl"
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
    failure_reason = result.get("subtype") or ("timeout" if timed_out else f"exit-{returncode}")
    calls = tool_blocks(events)
    skills_used = skill_calls(calls)

    if stream_path.exists():
        mlflow.log_artifact(str(stream_path), artifact_path=prefix.rstrip("/") or None)
    if result:
        mlflow.log_dict(result, f"{prefix}result.json")
    if stderr:
        mlflow.log_text(stderr, f"{prefix}stderr.txt")
    mlflow.log_text(final_text(events), f"{prefix}final_output.md")

    # The agent's own output: everything it wrote, minus the staged skills.
    outputs = repeat_dir / "outputs"
    shutil.copytree(workspace, outputs, ignore=shutil.ignore_patterns(".claude"))
    if any(outputs.rglob("*")):
        mlflow.log_artifacts(str(outputs), artifact_path=f"{prefix}outputs")

    # Promote the two deliverables that matter to a predictable artifact path.
    deliverables = find_deliverables(outputs)
    for kind, path in deliverables.items():
        mlflow.log_artifact(str(path), artifact_path=f"{prefix}final")
        print(f"{kind:9}: {path.relative_to(outputs)}")

    # Score the plots against the record's reference values.
    graded = grade(record.get("expectations"), metrics_from_events(events))
    plots_matched = sum(plot.passed for plot in graded.plots)
    mlflow.log_dict(graded.to_dict(), f"{prefix}grade.json")

    metrics = {
        "wall_seconds": wall_seconds,
        "duration_ms": result.get("duration_ms", 0) or 0,
        "api_duration_ms": result.get("duration_api_ms", 0) or 0,
        "num_turns": result.get("num_turns", 0) or 0,
        "cost_usd": result.get("total_cost_usd", 0.0) or 0.0,
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
        "num_tool_calls": len(calls),
        "num_tool_errors": tool_errors(events),
        "num_skill_calls": skills_used.total(),
        "num_skill_file_reads": skill_file_reads(calls),
        "completed": int(succeeded),
        "produced_script": int("script" in deliverables),
        "produced_plot": int("plot" in deliverables),
        **graded.mlflow_metrics(),
        "num_plots_expected": len(graded.plots),
        "num_plots_matched": plots_matched,
    }
    tags = {
        "question": name,
        "repeat": str(repeat),
        "status": "success" if succeeded else "failure",
        "failure_reason": "" if succeeded else failure_reason,
        "claude_session_id": result.get("session_id", ""),
        "grade": "pass" if graded.passed else "fail",
        "grade_message": graded.message,
        **{
            f"final_{kind}": deliverables[kind].name if kind in deliverables else ""
            for kind in ("script", "plot")
        },
    }

    trace_id = log_trace(prompt_text, events, succeeded, metrics, tags)
    # Traces are exported in the background, and the server's auth layer
    # answers 403 for a trace it has not stored yet - wait for the export.
    mlflow.flush_trace_async_logging()
    try:
        mlflow.log_feedback(
            trace_id=trace_id,
            name="metrics_match",
            value=graded.passed,
            rationale=graded.message,
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="grader.py"),
            metadata={"tolerance": str(graded.tolerance)},
        )
    except MlflowException as e:
        # The grade is already on the run; don't lose the sweep over the trace copy.
        print(f"warning  : could not attach grade to trace {trace_id}: {e.message}")

    print(f"status   : {'success' if succeeded else 'FAILURE'}")
    print(f"cost     : ${metrics['cost_usd']:.4f} over {metrics['num_turns']} turns")
    print(f"grade    : {'PASS' if graded.passed else 'FAIL'} - {graded.message}")

    return {
        "name": name,
        "repeat": repeat,
        "trace_id": trace_id,
        "succeeded": succeeded,
        "correct": graded.passed,
        "produced_script": "script" in deliverables,
        "produced_plot": "plot" in deliverables,
        "plots_expected": len(graded.plots),
        "plots_matched": plots_matched,
        "cost": metrics["cost_usd"],
        "wall_seconds": wall_seconds,
        "duration_ms": metrics["duration_ms"],
        "api_duration_ms": metrics["api_duration_ms"],
        "num_turns": metrics["num_turns"],
        "num_tool_calls": len(calls),
        "num_tool_errors": metrics["num_tool_errors"],
        "input_tokens": metrics["input_tokens"],
        "output_tokens": metrics["output_tokens"],
        "cache_read_tokens": metrics["cache_read_tokens"],
        "cache_creation_tokens": metrics["cache_creation_tokens"],
        "skill_calls": skills_used,
        "num_skill_file_reads": metrics["num_skill_file_reads"],
        "metrics": metrics,
        "tags": tags,
    }


def question_metrics(repeats: list[dict[str, Any]], skill_names: list[str]) -> dict[str, float]:
    """Roll a question's repeats up into its run's metrics.

    Each key holds the mean over the repeats - so a 0/1 outcome like
    ``metrics_match`` becomes the share of repeats that passed - and, with more
    than one repeat, ``<key>_std`` beside it and ``<key>_total`` for the additive
    keys. A single repeat logs exactly its own values. A key a repeat did not log,
    such as a plot's relative error when no METRIC line matched it, is averaged
    over the repeats that did.
    """
    # Every repeat counts every skill any repeat used, so a skill's mean is over all of them.
    skills = sorted({*skill_names, *(skill for r in repeats for skill in r["skill_calls"])})
    for summary in repeats:
        for skill in skills:
            summary["metrics"][f"skill_calls_{skill}"] = summary["skill_calls"][skill]

    metrics: dict[str, float] = {}
    for key in dict.fromkeys(key for r in repeats for key in r["metrics"]):
        values = [r["metrics"][key] for r in repeats if key in r["metrics"]]
        metrics[key] = fmean(values)
        if len(repeats) > 1:
            if len(values) > 1:
                metrics[f"{key}_std"] = stdev(values)
            if key in ADDITIVE_METRICS:
                metrics[f"{key}_total"] = sum(values)
    return metrics


def question_tags(repeats: list[dict[str, Any]]) -> dict[str, str]:
    """A question run's tags: its only repeat's, or a summary across several."""
    if len(repeats) == 1:
        return {k: v for k, v in repeats[0]["tags"].items() if k not in ("question", "repeat")}
    num_passed = sum(r["correct"] for r in repeats)
    return {
        "status": "success" if all(r["succeeded"] for r in repeats) else "failure",
        "failure_reason": ",".join(
            sorted({r["tags"]["failure_reason"] for r in repeats if not r["succeeded"]})
        ),
        "grade": "pass" if num_passed == len(repeats) else "fail",
        "grade_message": f"{num_passed}/{len(repeats)} repeats passed",
    }


def run_question(
    args: argparse.Namespace,
    prompt,
    record: dict[str, Any],
    variables: dict[str, str],
    trial_dir: Path,
    common_params: dict[str, Any],
    skill_names: list[str],
    tracking_uri: str,
) -> list[dict[str, Any]]:
    """Run a question's repeats as one child run of the active trial run.

    Each repeat is a trace on the run; returns the repeats' summaries.
    """
    name = question_name(record)
    tags = record["tags"] or {}
    prompt_text = render_prompt(prompt, {**record["inputs"], **variables})
    question_dir = trial_dir / name

    with mlflow.start_run(run_name=name, nested=True, tags={"run_type": "question"}) as run:
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

        repeats = []
        for repeat in range(1, args.repeats + 1):
            if args.repeats > 1:
                print(f"-- repeat {repeat}/{args.repeats}")
            repeats.append(run_repeat(args, prompt_text, record, repeat, question_dir))

        mlflow.log_metrics(question_metrics(repeats, skill_names))
        mlflow.set_tags(question_tags(repeats))
        print(f"run      : {run.info.run_id}")
        print(f"ui       : {run_url(tracking_uri, run)}")

    for summary in repeats:
        summary["run_id"] = run.info.run_id
    return repeats


def link_traces(trace_ids: list[str], run_id: str) -> None:
    """Link the question traces to the trial run too, so the trial's traces can be
    compared across trials. Each trace stays on its question run as well.
    """
    client = mlflow.MlflowClient()
    for start in range(0, len(trace_ids), 100):  # the API takes at most 100 per call
        chunk = trace_ids[start : start + 100]
        try:
            client.link_traces_to_run(chunk, run_id)
        except MlflowException as e:
            print(f"warning  : could not link {len(chunk)} trace(s) to the trial run: {e.message}")


def trial_metrics(summaries: list[dict[str, Any]], skill_names: list[str]) -> dict[str, float]:
    """Roll the question summaries up into the parent run's metrics.

    Rates and means are what compare across trials of different sizes, e.g. a
    skills revision run over a subset of the questions; totals are kept for cost.
    A rollup never reuses a question run's metric key - it is prefixed `total_` or
    `mean_` - so a chart of a key holds one kind of value, whichever runs are shown.
    """
    n = len(summaries)

    def total(key: str) -> float:
        return sum(s[key] for s in summaries)

    def mean(key: str) -> float:
        return fmean(s[key] for s in summaries)

    num_correct = total("correct")
    plots_expected = total("plots_expected")
    tool_calls = total("num_tool_calls")
    skills_used = sum((s["skill_calls"] for s in summaries), Counter())
    # Each question's grades over its repeats, for consistency across repeats.
    grades_by_question: dict[str, list[bool]] = {}
    for s in summaries:
        grades_by_question.setdefault(s["name"], []).append(s["correct"])

    metrics = {
        # Outcomes
        "total_completed": total("succeeded"),
        "completion_rate": mean("succeeded"),
        "total_correct": num_correct,
        "accuracy": mean("correct"),
        "pass_at_k": fmean(any(grades) for grades in grades_by_question.values()),
        "pass_all_k": fmean(all(grades) for grades in grades_by_question.values()),
        "total_produced_script": total("produced_script"),
        "script_rate": mean("produced_script"),
        "total_produced_plot": total("produced_plot"),
        "plot_rate": mean("produced_plot"),
        "total_plots_expected": plots_expected,
        "total_plots_matched": total("plots_matched"),
        # Cost and effort
        "total_cost_usd": total("cost"),
        "mean_cost_usd": mean("cost"),
        "total_wall_seconds": total("wall_seconds"),
        "mean_wall_seconds": mean("wall_seconds"),
        "total_duration_ms": total("duration_ms"),
        "mean_duration_ms": mean("duration_ms"),
        "total_api_duration_ms": total("api_duration_ms"),
        "mean_api_duration_ms": mean("api_duration_ms"),
        "total_turns": total("num_turns"),
        "mean_turns": mean("num_turns"),
        "total_tool_calls": tool_calls,
        "mean_tool_calls": mean("num_tool_calls"),
        "total_tool_errors": total("num_tool_errors"),
        "mean_tool_errors": mean("num_tool_errors"),
        "total_input_tokens": total("input_tokens"),
        "total_output_tokens": total("output_tokens"),
        "total_cache_read_tokens": total("cache_read_tokens"),
        "total_cache_creation_tokens": total("cache_creation_tokens"),
        # Skill usage
        "total_skill_calls": skills_used.total(),
        "mean_skill_calls": skills_used.total() / n,
        "skill_usage_rate": fmean(bool(s["skill_calls"]) for s in summaries),
        "num_distinct_skills_used": len(skills_used),
        "total_skill_file_reads": total("num_skill_file_reads"),
        **{
            f"total_skill_calls_{skill}": skills_used[skill]
            for skill in sorted({*skill_names, *skills_used})
        },
    }
    # Only defined when there is something to divide by, rather than a misleading 0.
    if plots_expected:
        metrics["plot_accuracy"] = total("plots_matched") / plots_expected
    if num_correct:
        metrics["cost_per_correct_usd"] = total("cost") / num_correct
    if tool_calls:
        metrics["tool_error_rate"] = total("num_tool_errors") / tool_calls
    return metrics


def summary_table(summaries: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """One row per repeat of each question, for side-by-side comparison of trials in the UI."""
    columns = [key for key in summaries[0] if key not in ("skill_calls", "metrics", "tags")]
    table = {key: [s[key] for s in summaries] for key in columns}
    table["skills_invoked"] = [
        ",".join(f"{skill}:{count}" for skill, count in sorted(s["skill_calls"].items()))
        for s in summaries
    ]
    return table


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
    if args.repeats > 1:
        print(f"repeats  : {args.repeats} per question ({len(records) * args.repeats} in all)")
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
        "num_repeats": args.repeats,
    }

    run_name = args.run_name or f"{prompt.name}-v{prompt.version}-{stamp}"
    # Tagged so the trial runs filter out with `tags.run_type = 'trial'`.
    with mlflow.start_run(run_name=run_name, tags={"run_type": "trial"}) as run:
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
            summaries += run_question(
                args, prompt, record, variables, trial_dir, common_params, skill_names, tracking_uri
            )

        link_traces([s["trace_id"] for s in summaries if s["trace_id"]], run.info.run_id)
        metrics = trial_metrics(summaries, skill_names)
        mlflow.log_metrics(metrics)
        mlflow.log_table(summary_table(summaries), artifact_file="questions.json")
        num_completed, num_correct = metrics["total_completed"], metrics["total_correct"]
        total_cost = metrics["total_cost_usd"]
        all_succeeded = num_completed == len(summaries)
        mlflow.set_tag("status", "success" if all_succeeded else "failure")

        print(
            f"\ntrial    : {run.info.run_id}  ({num_completed}/{len(summaries)} completed, "
            f"{num_correct}/{len(summaries)} correct)"
        )
        if args.repeats > 1:
            print(
                f"repeats  : pass@{args.repeats} {metrics['pass_at_k']:.0%}, "
                f"all {args.repeats} passed {metrics['pass_all_k']:.0%} of questions"
            )
        print(
            f"skills   : {metrics['total_skill_calls']} call(s), used in "
            f"{metrics['skill_usage_rate']:.0%} of questions"
        )
        for s in summaries:
            status = "ok  " if s["succeeded"] else "FAIL"
            verdict = "pass" if s["correct"] else "fail"
            skills = ",".join(sorted(s["skill_calls"])) or "-"
            name = f"{s['name']}-r{s['repeat']}" if args.repeats > 1 else s["name"]
            print(f"  {status} {verdict} {name:24} ${s['cost']:.4f}  {skills}")
        print(f"cost     : ${total_cost:.4f}")
        print(f"ui       : {run_url(tracking_uri, run)}")

    return 0 if all_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
