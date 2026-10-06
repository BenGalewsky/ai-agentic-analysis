"""Run a trial: every question's repeats, each a trace on one MLflow evaluation run."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_GENAI_EVALUATE

from .config import SKILLS_DIR
from .grader import grade, metrics_from_calls
from .harnesses import Harness
from .mcp_config import mcp_server_names, read_mcp_config
from .prompts import render_prompt
from .questions import question_name
from .rollup import summary_table, trial_metrics
from .tracing import log_trace
from .workspace import find_deliverables, hash_skills, stage_workspace


def run_repeat(
    args: argparse.Namespace,
    harness: Harness,
    prompt_text: str,
    record: dict[str, Any],
    repeat: int,
    question_dir: Path,
) -> dict[str, Any]:
    """Run one repeat of a question inside the active trial run.

    Logs the repeat's artifacts under ``<question>/`` (``<question>/r<k>/`` when
    there are several repeats) and its trace, tagged and annotated with its own
    metrics, its grade and the record's expectations, and returns its summary;
    the trial run's metrics are rolled up from these.
    """
    name = question_name(record)
    repeated = args.repeats > 1
    repeat_dir = question_dir / f"r{repeat}" if repeated else question_dir
    prefix = f"{name}/r{repeat}/" if repeated else f"{name}/"
    workspace = stage_workspace(repeat_dir, harness.skills_dir)
    print(f"workspace: {workspace}")

    stream_path = repeat_dir / harness.stream_file
    started = time.time()
    events, returncode, stderr, timed_out = harness.run(
        prompt_text, workspace, stream_path, args.timeout
    )
    wall_seconds = time.time() - started
    extra_files = harness.after_run(events, repeat_dir, workspace)
    summary = harness.summarize(events, repeat_dir)
    succeeded = returncode == 0 and not summary.is_error and not timed_out
    failure_reason = (
        "timeout" if timed_out else summary.failure_reason or f"exit-{returncode}"
    )
    calls = summary.calls
    skills_used = summary.skill_calls

    for path in [stream_path, *extra_files]:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=prefix.rstrip("/"))
    if summary.result:
        mlflow.log_dict(summary.result, f"{prefix}result.json")
    if stderr:
        mlflow.log_text(stderr, f"{prefix}stderr.txt")
    mlflow.log_text(summary.final_text, f"{prefix}final_output.md")

    # The agent's own output: everything it wrote, minus the staged skills and
    # the harness's own state.
    outputs = repeat_dir / "outputs"
    shutil.copytree(workspace, outputs, ignore=shutil.ignore_patterns(".claude", ".opencode"))
    if any(outputs.rglob("*")):
        mlflow.log_artifacts(str(outputs), artifact_path=f"{prefix}outputs")

    # Promote the two deliverables that matter to a predictable artifact path.
    deliverables = find_deliverables(outputs)
    for kind, path in deliverables.items():
        mlflow.log_artifact(str(path), artifact_path=f"{prefix}final")
        print(f"{kind:9}: {path.relative_to(outputs)}")

    # Score the plots against the record's reference values.
    graded = grade(record.get("expectations"), metrics_from_calls(calls))
    plots_matched = sum(plot.passed for plot in graded.plots)
    mlflow.log_dict(graded.to_dict(), f"{prefix}grade.json")

    metrics = {
        "wall_seconds": wall_seconds,
        "duration_ms": summary.duration_ms,
        "api_duration_ms": summary.api_duration_ms,
        "num_turns": summary.num_turns,
        "cost_usd": summary.cost_usd,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "cache_read_tokens": summary.cache_read_tokens,
        "cache_creation_tokens": summary.cache_creation_tokens,
        "num_tool_calls": len(calls),
        "num_tool_errors": summary.num_tool_errors,
        "num_skill_calls": skills_used.total(),
        "num_skill_file_reads": summary.num_skill_file_reads,
        "num_mcp_calls": summary.num_mcp_calls,
        "completed": int(succeeded),
        "produced_script": int("script" in deliverables),
        "produced_plot": int("plot" in deliverables),
        **graded.mlflow_metrics(),
        "num_plots_expected": len(graded.plots),
        "num_plots_matched": plots_matched,
    }
    # A value the harness does not report is left out rather than logged as 0.
    metrics = {key: value for key, value in metrics.items() if value is not None}
    record_tags = record["tags"] or {}
    tags = {
        "question": name,
        "question_index": record_tags.get("question_index", ""),
        "datasets": record_tags.get("datasets", ""),
        "dataset_record_id": record["dataset_record_id"],
        "repeat": str(repeat),
        "status": "success" if succeeded else "failure",
        "failure_reason": "" if succeeded else failure_reason,
        "session_id": summary.session_id,
        "grade": "pass" if graded.passed else "fail",
        "grade_message": graded.message,
        **{
            f"final_{kind}": deliverables[kind].name if kind in deliverables else ""
            for kind in ("script", "plot")
        },
    }

    trace_id = log_trace(
        harness.name, record["inputs"], prompt_text, summary, succeeded, metrics, tags
    )
    # Traces are exported in the background, and the server's auth layer
    # answers 403 for a trace it has not stored yet - wait for the export.
    mlflow.flush_trace_async_logging()
    try:
        # As mlflow.genai.evaluate does: the grade is feedback from this run, and
        # the record's expectations sit beside it on the trace.
        mlflow.log_feedback(
            trace_id=trace_id,
            name="metrics_match",
            value=graded.passed,
            rationale=graded.message,
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="grader.py"),
            metadata={
                "tolerance": str(graded.tolerance),
                AssessmentMetadataKey.SOURCE_RUN_ID: mlflow.active_run().info.run_id,
            },
        )
        for key, value in (record.get("expectations") or {}).items():
            mlflow.log_expectation(trace_id=trace_id, name=key, value=value)
    except MlflowException as e:
        # The grade is already in grade.json; don't lose the sweep over the trace copy.
        print(f"warning  : could not attach grade to trace {trace_id}: {e.message}")

    print(f"status   : {'success' if succeeded else 'FAILURE'}")
    if not succeeded:
        last_stderr = next((line for line in reversed(stderr.splitlines()) if line.strip()), "")
        print(f"reason   : {failure_reason}" + (f" - {last_stderr.strip()[:200]}" if last_stderr else ""))
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
        "duration_ms": summary.duration_ms,
        "api_duration_ms": summary.api_duration_ms,
        "num_turns": metrics["num_turns"],
        "num_tool_calls": len(calls),
        "num_tool_errors": metrics["num_tool_errors"],
        "input_tokens": metrics["input_tokens"],
        "output_tokens": metrics["output_tokens"],
        "cache_read_tokens": metrics["cache_read_tokens"],
        "cache_creation_tokens": metrics["cache_creation_tokens"],
        "skill_calls": skills_used,
        "num_skill_file_reads": metrics["num_skill_file_reads"],
        "num_mcp_calls": metrics["num_mcp_calls"],
        "metrics": metrics,
        "tags": tags,
    }


def run_question(
    args: argparse.Namespace,
    harness: Harness,
    prompt,
    record: dict[str, Any],
    variables: dict[str, str],
    trial_dir: Path,
) -> list[dict[str, Any]]:
    """Run a question's repeats in the active trial run, each repeat a trace on it.

    The question's own artifacts go under ``<question>/``; returns the repeats' summaries.
    """
    name = question_name(record)
    prompt_text = render_prompt(prompt, {**record["inputs"], **variables})
    question_dir = trial_dir / name

    mlflow.log_text(prompt_text, f"{name}/prompt.txt")
    mlflow.log_dict(record["inputs"], f"{name}/inputs.json")
    if record.get("expectations"):
        mlflow.log_dict(record["expectations"], f"{name}/expectations.json")

    repeats = []
    for repeat in range(1, args.repeats + 1):
        if args.repeats > 1:
            print(f"-- repeat {repeat}/{args.repeats}")
        repeats.append(run_repeat(args, harness, prompt_text, record, repeat, question_dir))
    return repeats


def run_url(tracking_uri: str, run) -> str:
    return (
        f"{tracking_uri.split('@')[-1]}/#/experiments/{run.info.experiment_id}"
        f"/runs/{run.info.run_id}"
    )


def run_trial(
    args: argparse.Namespace,
    harness: Harness,
    prompt,
    dataset,
    records: list[dict[str, Any]],
    variables: dict[str, str],
    tracking_uri: str,
) -> bool:
    """Run the questions in one trial run, an MLflow evaluation run with a trace per
    repeat of each question; return whether every repeat completed.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trial_dir = args.trials_dir / f"{stamp}-{prompt.name}-v{prompt.version}-{harness.name}"
    skills_hash = hash_skills(SKILLS_DIR)
    skill_names = sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir())
    harness_version = harness.version

    print(f"harness  : {harness.name} ({harness_version})")
    print(f"prompt   : {prompt.name} v{prompt.version}")
    print(f"dataset  : {args.dataset} ({len(records)} question{'s' if len(records) != 1 else ''})")
    if args.repeats > 1:
        print(f"repeats  : {args.repeats} per question ({len(records) * args.repeats} in all)")
    print(f"skills   : {', '.join(skill_names)} ({skills_hash})")
    if args.mcp_config:
        print(f"mcp      : {', '.join(mcp_server_names(args.mcp_config))}")
    print(f"trial dir: {trial_dir}")

    params = {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_uri": prompt.uri,
        "dataset_name": args.dataset,
        "dataset_id": dataset.dataset_id,
        "harness": harness.name,
        "harness_version": harness_version,
        "model": args.model or "default",
        **harness.params(),
        "mcp_config": ",".join(args.mcp_config),
        "mcp_servers": ",".join(mcp_server_names(args.mcp_config)),
        "skills_hash": skills_hash,
        "skills": ",".join(skill_names),
        "num_skills": len(skill_names),
        "num_repeats": args.repeats,
        "question_filter": ",".join(args.question),
        "num_questions": len(records),
    }

    run_name = args.run_name or f"{prompt.name}-v{prompt.version}-{harness.name}-{stamp}"
    # Tagged so the trial runs filter out with `tags.run_type = 'trial'`, and as an
    # evaluation run - with the dataset as its input - like mlflow.genai.evaluate's.
    with mlflow.start_run(
        run_name=run_name,
        tags={
            "run_type": "trial",
            "harness": harness.name,
            MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_GENAI_EVALUATE,
        },
    ) as run:
        mlflow.log_params(params)
        mlflow.log_input(dataset)
        mlflow.log_text(prompt.template, "prompt_template.txt")
        for i, config in enumerate(args.mcp_config):
            mlflow.log_text(read_mcp_config(config), f"mcp_config_{i}.json" if i else "mcp_config.json")
        for name, text in harness.config_artifacts().items():
            mlflow.log_text(text, name)
        mlflow.log_artifacts(str(SKILLS_DIR), artifact_path="skills")

        summaries = []
        for i, record in enumerate(records, 1):
            print(f"\n[{i}/{len(records)}] {question_name(record)}")
            summaries += run_question(args, harness, prompt, record, variables, trial_dir)

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
        if args.mcp_config:
            print(
                f"mcp      : {metrics['total_mcp_calls']} call(s), used in "
                f"{metrics['mcp_usage_rate']:.0%} of questions"
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
