#!/usr/bin/env python
"""Grade a trial's plots against the expectations of its ``hep-data-llm-questions`` record.

The prompt has the agent's script print one line per plot describing the values it
filled the histogram with, in the format hep-data-llm uses:

    METRIC: avg_entries_per_event=<N> mean=<M>

A record's expectations hold the reference values for each plot:

    {"plots": [{"avg_entries_per_event": 1.0, "mean": 16.45}], "n_plots": 1}

The METRIC lines are read from the output of the last tool call that printed any,
which is the agent's final run of its script. A trial passes when it printed exactly
``n_plots`` lines and each reference plot is matched, one-to-one and in any order,
by a line whose ``mean`` is within a relative tolerance (default 1%). This follows
hep-data-llm, which leaves ``avg_entries_per_event`` ungated because there are too
many valid ways to count entries for it to be comparable; its relative error is
still reported, and ``--check-avg-entries`` gates on it too.

Regrade trials already on disk, taking expectations from the dataset:

    uv run scripts/grader.py ~/.cache/hep-agent-trials/20261005T071912Z-IRIS-HEP-v1
    uv run scripts/grader.py <trial_dir>/JetPtAll --tolerance 0.005

A trial run with ``--repeats`` nests each repeat as ``<question>/r<k>/``; those are
found under a trial or question directory and graded against ``<question>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import scorer

DEFAULT_TOLERANCE = 0.01
DEFAULT_DATASET = "hep-data-llm-questions"

# Same grammar as hep-data-llm's ``extract_metrics``, so the two harnesses agree on
# what counts as a METRIC line. An f-string template such as
# ``avg_entries_per_event={n}`` (the agent cat-ing its own script) never matches.
_FLOAT = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+"
METRIC_RE = re.compile(
    rf"METRIC:\s*avg_entries_per_event=(?P<avg_entries_per_event>{_FLOAT})"
    rf"\s+mean=(?P<mean>{_FLOAT})"
)

Metric = tuple[float, float]  # (avg_entries_per_event, mean)


# --------------------------------------------------------------------------- #
# Extracting METRIC lines
# --------------------------------------------------------------------------- #
def parse_metrics(text: str) -> list[Metric]:
    return [
        (float(m.group("avg_entries_per_event")), float(m.group("mean")))
        for m in METRIC_RE.finditer(text)
    ]


def _as_text(content: Any) -> str:
    """Flatten a tool result, which is a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_as_text(b.get("text", "") if isinstance(b, dict) else b) for b in content)
    if isinstance(content, dict):
        return _as_text(content.get("result", content.get("text", "")))
    return "" if content is None else str(content)


def last_metrics(outputs: list[Any]) -> list[Metric]:
    """METRIC lines from the last tool output that printed any."""
    for output in reversed(outputs):
        if metrics := parse_metrics(_as_text(output)):
            return metrics
    return []


def metrics_from_events(events: list[dict[str, Any]]) -> list[Metric]:
    """METRIC lines from a Claude Code ``stream-json`` event list."""
    outputs = [
        block.get("content")
        for event in events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", []) or []
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    return last_metrics(outputs)


def metrics_from_stream(path: Path) -> list[Metric]:
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return metrics_from_events(events)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def relative_error(reference: float | None, observed: float) -> float | None:
    """``None`` when there is no reference; a zero reference must be matched exactly."""
    if reference is None:
        return None
    if reference == 0:
        return 0.0 if observed == 0 else math.inf
    return abs(observed - reference) / abs(reference)


@dataclass
class PlotGrade:
    index: int
    expected: dict[str, float | None]
    observed: dict[str, float] | None = None
    mean_rel_err: float | None = None
    avg_entries_rel_err: float | None = None
    passed: bool = False


@dataclass
class Grade:
    passed: bool
    message: str
    tolerance: float
    observed: list[Metric] = field(default_factory=list)
    plots: list[PlotGrade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: an infinite relative error is written as ``null``."""

        def finite(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: finite(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite(v) for v in value]
            return value

        return finite(asdict(self))

    def mlflow_metrics(self) -> dict[str, float]:
        metrics = {"metrics_match": float(self.passed), "num_metric_lines": float(len(self.observed))}
        for plot in self.plots:
            for name in ("mean_rel_err", "avg_entries_rel_err"):
                value = getattr(plot, name)
                if value is not None and math.isfinite(value):
                    metrics[f"plot_{plot.index}_{name}"] = value
        return metrics


def grade(
    expectations: dict[str, Any] | None,
    observed: list[Metric],
    tolerance: float = DEFAULT_TOLERANCE,
    check_avg_entries: bool = False,
) -> Grade:
    """Match observed METRIC lines one-to-one against the expected plots."""
    references = (expectations or {}).get("plots") or []
    if not references:
        return Grade(True, "No reference metrics supplied.", tolerance, observed)

    def plot_grade(index: int, ref: dict[str, Any], obs: Metric | None) -> PlotGrade:
        plot = PlotGrade(
            index=index,
            expected={k: ref.get(k) for k in ("avg_entries_per_event", "mean")},
        )
        if obs is None:
            return plot
        plot.observed = {"avg_entries_per_event": obs[0], "mean": obs[1]}
        plot.mean_rel_err = relative_error(ref.get("mean"), obs[1])
        plot.avg_entries_rel_err = relative_error(ref.get("avg_entries_per_event"), obs[0])
        gated = [plot.mean_rel_err] + ([plot.avg_entries_rel_err] if check_avg_entries else [])
        plot.passed = all(err is None or err <= tolerance for err in gated)
        return plot

    # Pair references and lines nearest-first by the error on the mean, so each
    # plot reports against its closest line even when the trial fails overall.
    candidates = sorted(
        (relative_error(ref.get("mean"), obs[1]) or 0.0, r, o)
        for r, ref in enumerate(references)
        for o, obs in enumerate(observed)
    )
    pairs: dict[int, int] = {}
    for _, r, o in candidates:
        if r not in pairs and o not in pairs.values():
            pairs[r] = o
    plots = [
        plot_grade(r, ref, observed[pairs[r]] if r in pairs else None)
        for r, ref in enumerate(references)
    ]

    failed = [p.index for p in plots if not p.passed]
    if not observed:
        message = "No METRIC lines were captured from the run."
    elif len(observed) != len(references):
        message = f"Expected {len(references)} METRIC line(s) but found {len(observed)}."
    elif failed:
        message = f"No METRIC line matched reference plot(s) {', '.join(map(str, failed))}."
    else:
        message = "All METRIC lines matched the supplied references."
    passed = len(observed) == len(references) and not failed
    return Grade(passed, message, tolerance, observed, plots)


# --------------------------------------------------------------------------- #
# MLflow scorer
# --------------------------------------------------------------------------- #
@scorer
def metrics_match(trace, expectations) -> Feedback:
    """``mlflow.genai.evaluate`` scorer over traces logged by ``run_trial.py``.

    Reads the METRIC lines from the trace's tool spans, so stored trials can be
    rescored without rerunning the agent.
    """
    spans = sorted(trace.search_spans(span_type=SpanType.TOOL), key=lambda s: s.start_time_ns)
    result = grade(expectations, last_metrics([s.outputs for s in spans]))
    return Feedback(
        value=result.passed,
        rationale=result.message,
        metadata={"grade": json.dumps(result.to_dict())},
    )


# --------------------------------------------------------------------------- #
# Regrading trials on disk
# --------------------------------------------------------------------------- #
REPEAT_DIR = re.compile(r"r\d+")


def question_dirs(paths: list[Path]) -> list[Path]:
    """Accept question or repeat directories, or trial directories holding several."""
    found = []
    for path in paths:
        if (path / "claude_stream.jsonl").exists():
            found.append(path)
        else:
            streams = [*path.glob("*/claude_stream.jsonl"), *path.glob("*/*/claude_stream.jsonl")]
            found += sorted(p.parent for p in streams)
    return found


def question_of(qdir: Path) -> str:
    """The record name a directory holds a run of: its own, or its parent's for a repeat."""
    return qdir.parent.name if REPEAT_DIR.fullmatch(qdir.name) else qdir.name


def run_label(qdir: Path) -> str:
    return f"{qdir.parent.name}-{qdir.name}" if REPEAT_DIR.fullmatch(qdir.name) else qdir.name


def load_expectations(dataset_name: str) -> dict[str, dict[str, Any]]:
    from mlflow.genai.datasets import get_dataset

    records = get_dataset(name=dataset_name).to_df().to_dict("records")
    return {(r["tags"] or {}).get("name"): r["expectations"] for r in records}


def format_grade(name: str, result: Grade) -> str:
    lines = [f"{'PASS' if result.passed else 'FAIL'} {name:24} {result.message}"]
    for plot in result.plots:
        expected = plot.expected["mean"]
        if plot.observed is None:
            lines.append(f"       plot {plot.index}: mean expected {expected}, no line")
            continue
        err = plot.mean_rel_err
        err_text = "n/a" if err is None else f"{err:.2%}"
        lines.append(
            f"       plot {plot.index}: mean {plot.observed['mean']:.6g} vs {expected} "
            f"({err_text}), avg_entries {plot.observed['avg_entries_per_event']:.6g} "
            f"vs {plot.expected['avg_entries_per_event']}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="Trial or question directories")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Relative tolerance")
    ap.add_argument("--check-avg-entries", action="store_true", help="Also gate on avg_entries_per_event")
    ap.add_argument("--json", action="store_true", help="Print grades as JSON")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set (expected in .env)")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    expectations = load_expectations(args.dataset)

    dirs = question_dirs(args.paths)
    if not dirs:
        raise SystemExit("No claude_stream.jsonl found under the given paths")

    grades = {}
    for qdir in dirs:
        name = question_of(qdir)
        if name not in expectations:
            print(f"SKIP {run_label(qdir):24} no record of that name in {args.dataset}", file=sys.stderr)
            continue
        grades[run_label(qdir)] = grade(
            expectations[name],
            metrics_from_stream(qdir / "claude_stream.jsonl"),
            args.tolerance,
            args.check_avg_entries,
        )

    if args.json:
        print(json.dumps({name: g.to_dict() for name, g in grades.items()}, indent=2))
    else:
        for name, result in grades.items():
            print(format_grade(name, result))
        print(f"\n{sum(g.passed for g in grades.values())}/{len(grades)} passed")
    return 0 if grades and all(g.passed for g in grades.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
