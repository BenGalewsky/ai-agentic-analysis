"""Load and select the questions of an MLflow evaluation dataset."""

from __future__ import annotations

from typing import Any

from mlflow import MlflowClient
from mlflow.genai.datasets import search_datasets


def load_questions(name: str, experiment: str) -> tuple[Any, list[dict[str, Any]]]:
    """Load the dataset and its records, in the order of their ``question_index`` tag."""
    # Search within the experiment: the basic-auth server rejects dataset searches
    # without experiment_ids (which get_dataset(name=...) sends) with a 403.
    exp = MlflowClient().get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment {experiment!r} not found")
    found = search_datasets(experiment_ids=exp.experiment_id, filter_string=f'name = "{name}"', max_results=1)
    if not found:
        raise SystemExit(f"Dataset {name!r} not found in experiment {experiment!r}")
    dataset = found[0]
    records = dataset.to_df().to_dict("records")
    if not records:
        raise SystemExit(f"Dataset {name!r} has no records")
    records.sort(key=lambda r: int(question_index(r) or 0))
    return dataset, records


def question_name(record: dict[str, Any]) -> str:
    return (record["tags"] or {}).get("name") or record["dataset_record_id"]


def question_index(record: dict[str, Any]) -> str:
    return (record["tags"] or {}).get("question_index", "")


def select_question(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Find a record by its ``name`` tag or its ``question_index`` tag."""
    for record in records:
        tags = record["tags"] or {}
        if key in (tags.get("name"), tags.get("question_index")):
            return record
    choices = ", ".join(question_name(r) for r in records)
    raise SystemExit(f"No question {key!r} in the dataset; choose a name or index from: {choices}")
