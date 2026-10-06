"""Record trials as MLflow traces."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import SpanStatusCode, SpanType

from .harnesses import RunSummary


def log_trace(
    harness: str,
    inputs: dict[str, Any],
    prompt_text: str,
    summary: RunSummary,
    succeeded: bool,
    metrics: dict[str, float],
    tags: dict[str, str],
) -> str:
    """Record one repeat as an MLflow trace with a child span per tool call; return its ID.

    A repeat that produced no events (a timeout, a failed launch) still gets an
    error trace, so every repeat of a question is accounted for among its traces.

    The trace's inputs are the dataset record's, so traces of the same question
    line up across trials - even ones run with different prompt versions - and the
    rendered prompt is an attribute of the agent span. Started inside the trial
    run, the trace belongs to it.
    """
    root = mlflow.start_span_no_context(
        name=f"{harness}_trial",
        span_type=SpanType.AGENT,
        inputs=inputs,
        attributes={"prompt": prompt_text},
        tags=tags,
    )
    try:
        for call in summary.calls:
            child = mlflow.start_span_no_context(
                name=call.name or "tool",
                span_type=SpanType.TOOL,
                parent_span=root,
                inputs=call.input,
            )
            child.end(
                outputs={"result": call.output},
                status=SpanStatusCode.ERROR if call.is_error else SpanStatusCode.OK,
            )

        root.set_attributes(metrics)
        root.end(
            outputs={"result": summary.final_text},
            status=SpanStatusCode.OK if succeeded else SpanStatusCode.ERROR,
        )
        return root.trace_id
    except Exception:
        root.end(status=SpanStatusCode.ERROR)
        raise

