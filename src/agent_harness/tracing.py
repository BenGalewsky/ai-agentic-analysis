"""Record trials as MLflow traces."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import SpanStatusCode, SpanType
from mlflow.exceptions import MlflowException

from .stream import final_text, tool_blocks


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
