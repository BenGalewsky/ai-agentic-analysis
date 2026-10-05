"""Roll repeat summaries up into question-run and trial-run metrics."""

from __future__ import annotations

from collections import Counter
from statistics import fmean, stdev
from typing import Any

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
    "num_mcp_calls",
)


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


def trial_metrics(summaries: list[dict[str, Any]], skill_names: list[str]) -> dict[str, float]:
    """Roll the question summaries up into the parent run's metrics.

    Rates and means are what compare across trials of different sizes, e.g. a
    skills revision run over a subset of the questions; totals are kept for cost.
    A rollup never reuses a question run's metric key - it is prefixed `total_` or
    `mean_` - so a chart of a key holds one kind of value, whichever runs are shown.
    A value a harness does not report (``None``, e.g. opencode's API duration) is
    skipped, and its rollups left out when no repeat reported it.
    """
    n = len(summaries)

    def reported(key: str) -> list[float]:
        return [s[key] for s in summaries if s[key] is not None]

    def total(key: str) -> float | None:
        values = reported(key)
        return sum(values) if values else None

    def mean(key: str) -> float | None:
        values = reported(key)
        return fmean(values) if values else None

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
        # MCP usage
        "total_mcp_calls": total("num_mcp_calls"),
        "mean_mcp_calls": mean("num_mcp_calls"),
        "mcp_usage_rate": fmean(bool(s["num_mcp_calls"]) for s in summaries),
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
    return {key: value for key, value in metrics.items() if value is not None}


def summary_table(summaries: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """One row per repeat of each question, for side-by-side comparison of trials in the UI."""
    columns = [key for key in summaries[0] if key not in ("skill_calls", "metrics", "tags")]
    table = {key: [s[key] for s in summaries] for key in columns}
    table["skills_invoked"] = [
        ",".join(f"{skill}:{count}" for skill, count in sorted(s["skill_calls"].items()))
        for s in summaries
    ]
    return table
