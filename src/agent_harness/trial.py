"""Run a trial: every question's repeats, each logged to MLflow under one parent run."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException

from .claude import build_command, run_claude
from .config import SKILLS_DIR
from .grader import grade, metrics_from_events
from .mcp_config import mcp_server_names, read_mcp_config
from .prompts import render_prompt
from .questions import question_name
from .rollup import question_metrics, question_tags, summary_table, trial_metrics
from .stream import final_text, skill_calls, skill_file_reads, tool_blocks, tool_errors
from .tracing import link_traces, log_trace
from .workspace import find_deliverables, hash_skills, stage_workspace


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


def run_url(tracking_uri: str, run) -> str:
    return (
        f"{tracking_uri.split('@')[-1]}/#/experiments/{run.info.experiment_id}"
        f"/runs/{run.info.run_id}"
    )


def run_trial(
    args: argparse.Namespace,
    prompt,
    dataset,
    records: list[dict[str, Any]],
    variables: dict[str, str],
    tracking_uri: str,
) -> bool:
    """Run the questions as child runs of one parent trial run; return whether every repeat completed."""
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

    return all_succeeded
