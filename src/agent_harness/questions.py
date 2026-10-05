"""Load and select the questions of an MLflow evaluation dataset."""

from __future__ import annotations

from typing import Any

from mlflow.genai.datasets import get_dataset


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
