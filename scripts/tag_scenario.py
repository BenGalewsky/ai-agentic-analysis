"""Tag a trial run and all of its child runs with ``scenario=<value>``.

Children are the runs whose ``mlflow.parentRunId`` tag names the parent, searched in the
parent's experiment; only trials logged with a run per question have any. An existing
``scenario`` tag is overwritten.

    uv run scripts/tag_scenario.py <trial_run_id> baseline --dry-run
    uv run scripts/tag_scenario.py <trial_run_id> baseline
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient
from mlflow.entities import Run

from agent_harness.config import connect_mlflow

TAG = "scenario"


def child_runs(client: MlflowClient, parent: Run) -> list[Run]:
    runs: list[Run] = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            page_token=page_token,
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="Parent (trial) run ID")
    ap.add_argument("scenario", help=f"Value for the '{TAG}' tag")
    ap.add_argument("--dry-run", action="store_true", help="List the runs that would be tagged")
    args = ap.parse_args()

    connect_mlflow()
    client = MlflowClient()
    parent = client.get_run(args.run_id)
    if grandparent := parent.data.tags.get("mlflow.parentRunId"):
        raise SystemExit(f"{args.run_id} is itself a child run; its parent is {grandparent}")
    runs = [parent, *child_runs(client, parent)]

    for run in runs:
        old = run.data.tags.get(TAG)
        change = f"{old!r} -> {args.scenario!r}" if old else repr(args.scenario)
        print(f"{run.info.run_id}  {run.info.run_name:<50}  {change}")
        if not args.dry_run:
            client.set_tag(run.info.run_id, TAG, args.scenario)

    verb = "would tag" if args.dry_run else "tagged"
    print(f"{verb} {len(runs)} run(s) (1 parent, {len(runs) - 1} children) with {TAG}={args.scenario}")


if __name__ == "__main__":
    main()
