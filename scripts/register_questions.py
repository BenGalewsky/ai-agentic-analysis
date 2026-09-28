"""Register the hep-data-llm benchmark questions as records in an MLflow evaluation dataset.

Each question in ``questions.yaml`` becomes one record in the ``hep-data-llm-questions``
dataset, linked to the ``hep-plot-agent`` experiment:

    inputs        {"question": <question text>}
    expectations  {"plots": <references.plots>, "n_plots": <count>}
    tags          name, question_index, source, source_commit, datasets

Re-running is safe: records are upserted on their inputs, so an unchanged question
only has its expectations and tags refreshed. When a question's text changes, the
record carrying the old text under the same name is deleted.

    uv run scripts/register_questions.py --dry-run
    uv run scripts/register_questions.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import EvaluationDataset, create_dataset, get_dataset

COMMIT = "7ba7f67dcf9332f0235fd8204dc923b0842240c3"
SOURCE_URL = (
    f"https://github.com/gordonwatts/hep-data-llm/raw/{COMMIT}"
    "/src/hep_data_llm/config/questions.yaml"
)

DATASET_NAME = "hep-data-llm-questions"
EXPERIMENT = "hep-plot-agent"

# Question names, in the same order as the questions in questions.yaml.
NAMES = [
    "ETmissAllEvents",
    "JetPtAll",
    "JetPtCentral",
    "ETmissTwoJets40",
    "ETmissZmumu",
    "TrijetTopPtBtag",
    "JetHTLeptonCleaned",
    "WZTransverseMass",
    "JetMultiplicity",
    "ElectronPt25",
    "LeadingJetPt",
    "SameChargeDimuonMass",
    "ETmissZeroJets",
    "LeadingDijetDeltaR",
    "StackedLeadingJetPt",
    "TTbarMass3TeV",
]

DATASET_RE = re.compile(r"\b((?:user\.\w+|opendata|mc\d+_\w+):[\w.\-]+?)(?=\.?(?:\s|$))")


def load_questions(path: str | None) -> list[dict]:
    if path:
        with open(path) as f:
            return yaml.safe_load(f)["questions"]
    with urllib.request.urlopen(SOURCE_URL) as r:
        return yaml.safe_load(r.read())["questions"]


def record_for(name: str, q: dict, index: int) -> dict:
    plots = q.get("references", {}).get("plots", [])
    tags = {
        "name": name,
        "question_index": str(index),
        "source": "hep-data-llm/questions.yaml",
        "source_commit": COMMIT,
    }
    datasets = DATASET_RE.findall(q["text"])
    if datasets:
        tags["datasets"] = ",".join(datasets)
    return {
        "inputs": {"question": q["text"]},
        "expectations": {"plots": plots, "n_plots": len(plots)},
        "tags": tags,
    }


def get_or_create_dataset(experiment_id: str) -> EvaluationDataset:
    try:
        return get_dataset(name=DATASET_NAME)
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return create_dataset(
            name=DATASET_NAME,
            experiment_id=experiment_id,
            tags={"source": "hep-data-llm/questions.yaml", "source_commit": COMMIT},
        )


def stale_record_ids(dataset: EvaluationDataset, records: list[dict]) -> list[str]:
    """Existing records whose name matches a new record but whose question text differs."""
    current = {r["tags"]["name"]: r["inputs"] for r in records}
    df = dataset.to_df()
    if df.empty:
        return []
    return [
        row.dataset_record_id
        for row in df.itertuples()
        if (name := (row.tags or {}).get("name")) in current and row.inputs != current[name]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", help="Local questions.yaml (default: fetch pinned commit from GitHub)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be registered")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if len(questions) != len(NAMES):
        raise SystemExit(f"{len(questions)} questions but {len(NAMES)} names - update NAMES")

    records = [record_for(name, q, i) for i, (name, q) in enumerate(zip(NAMES, questions))]
    if args.dry_run:
        for r in records:
            print(f"{r['tags']['name']}\n  {r['inputs']['question'][:100]}...\n  {json.dumps(r['expectations'])}\n")
        return

    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment_id = mlflow.set_experiment(EXPERIMENT).experiment_id
    dataset = get_or_create_dataset(experiment_id)

    if stale := stale_record_ids(dataset, records):
        dataset.delete_records(stale)
        print(f"deleted   {len(stale)} record(s) with outdated question text")
    dataset.merge_records(records)
    print(f"merged    {len(records)} records into {DATASET_NAME} ({dataset.dataset_id})")


if __name__ == "__main__":
    main()
