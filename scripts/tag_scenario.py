"""Tag a trial run and all of its child runs, or every run of a sweep, with ``scenario=<value>``.

Children are the runs whose ``mlflow.parentRunId`` tag names the parent, searched in the
parent's experiment; only trials logged with a child run per question have any. With
``--sweep``, the runs are the question runs of one ``--scheme per-question`` invocation,
those tagged ``sweep=<id>`` in ``--experiment``. An existing ``scenario`` tag is overwritten.

    uv run scripts/tag_scenario.py <trial_run_id> baseline --dry-run
    uv run scripts/tag_scenario.py <trial_run_id> baseline
    uv run scripts/tag_scenario.py --sweep <sweep_id> baseline
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient
from mlflow.entities import Run

from agent_harness.config import DEFAULT_QUESTION_EXPERIMENT, connect_mlflow

TAG = "scenario"


def search_all(client: MlflowClient, experiment_id: str, filter_string: str) -> list[Run]:
    runs: list[Run] = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id], filter_string=filter_string, page_token=page_token
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            return runs


def child_runs(client: MlflowClient, parent: Run) -> list[Run]:
    return search_all(
        client, parent.info.experiment_id, f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    )


def sweep_runs(client: MlflowClient, sweep: str, experiment: str) -> list[Run]:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment {experiment!r} not found")
    runs = search_all(client, exp.experiment_id, f"tags.sweep = '{sweep}'")
    if not runs:
        raise SystemExit(f"No runs tagged sweep={sweep!r} in experiment {experiment!r}")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?", help="Parent (trial) run ID")
    ap.add_argument("scenario", help=f"Value for the '{TAG}' tag")
    ap.add_argument("--sweep", help="Tag the question runs of this sweep instead of a trial run")
    ap.add_argument(
        "--experiment",
        default=DEFAULT_QUESTION_EXPERIMENT,
        help=f"Experiment to find the --sweep's runs in (default: {DEFAULT_QUESTION_EXPERIMENT})",
    )
    ap.add_argument("--dry-run", action="store_true", help="List the runs that would be tagged")
    args = ap.parse_args()
    if (args.run_id is None) == (args.sweep is None):
        ap.error("give either a trial run ID or --sweep")

    connect_mlflow()
    client = MlflowClient()
    if args.sweep:
        runs = sweep_runs(client, args.sweep, args.experiment)
        breakdown = f"question runs of sweep {args.sweep}"
    else:
        parent = client.get_run(args.run_id)
        if grandparent := parent.data.tags.get("mlflow.parentRunId"):
            raise SystemExit(f"{args.run_id} is itself a child run; its parent is {grandparent}")
        runs = [parent, *child_runs(client, parent)]
        breakdown = f"1 parent, {len(runs) - 1} children"

    for run in runs:
        old = run.data.tags.get(TAG)
        change = f"{old!r} -> {args.scenario!r}" if old else repr(args.scenario)
        print(f"{run.info.run_id}  {run.info.run_name:<50}  {change}")
        if not args.dry_run:
            client.set_tag(run.info.run_id, TAG, args.scenario)

    verb = "would tag" if args.dry_run else "tagged"
    print(f"{verb} {len(runs)} run(s) ({breakdown}) with {TAG}={args.scenario}")


if __name__ == "__main__":
    main()
