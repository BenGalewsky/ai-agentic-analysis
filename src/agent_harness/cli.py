"""Run agent trials over an MLflow evaluation dataset of questions and report to MLflow.

A trial:
  1. Pulls a prompt (default: ``IRIS-HEP``) from the MLflow Prompt Registry.
  2. Loads the questions from an MLflow evaluation dataset (default:
     ``hep-data-llm-questions``, written by ``register_questions.py``).
  3. For each question, renders the prompt with the record's inputs, stages a clean
     workspace containing a snapshot of ``skills/`` and runs the agent harness in it
     as a subprocess - Claude Code (``claude -p``, the default) or opencode
     (``opencode run``) - streaming JSON events.
  4. Logs the trial as one MLflow evaluation run - params, rolled-up metrics and
     artifacts - with an MLflow trace per question, graded and annotated with the
     record's expectations.

Example:
    uv run run-trial
    uv run run-trial --question JetPtAll
    uv run run-trial --question JetPtAll --repeats 5
    uv run run-trial --prompt IRIS-HEP --prompt-version 1 --model opus
    uv run run-trial --harness opencode --model lumen/qwen3-coder-next
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from .config import (
    DEFAULT_DATASET,
    DEFAULT_EXPERIMENT,
    DEFAULT_MCP_CONFIG,
    DEFAULT_PROMPT,
    DEFAULT_TRIALS_DIR,
    connect_mlflow,
)
from .harnesses import HARNESSES
from .mcp_config import check_mcp_env
from .prompts import render_prompt, resolve_prompt
from .questions import load_questions, select_question
from .trial import run_trial


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Registered prompt name")
    parser.add_argument("--prompt-version", default=None, help="Prompt version (default: latest)")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="MLflow evaluation dataset whose records' inputs fill the prompt",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        metavar="NAME_OR_INDEX",
        help="Run only this question, by its name tag (e.g. JetPtAll) or question_index "
        "(repeatable; default: every question in the dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N questions, by question_index (applied after --question)",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=1,
        metavar="N",
        help="Run each question N times, to measure how consistent the agent is (default: 1)",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Value for a prompt template variable (repeatable; overrides the dataset's inputs)",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the parent trial run (default: <prompt>-v<version>-<harness>-<timestamp>)",
    )
    parser.add_argument(
        "--harness",
        default="claude",
        choices=sorted(HARNESSES),
        help="Agent harness to run each question with (default: claude)",
    )
    parser.add_argument(
        "--harness-bin",
        default=None,
        help="Path to the harness's CLI (default: claude or opencode on PATH)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model passed to the harness, e.g. opus for claude or lumen/qwen3-coder-next "
        "(provider/model) for opencode",
    )
    parser.add_argument(
        "--permission-mode",
        default=None,
        choices=["acceptEdits", "auto", "bypassPermissions", "dontAsk", "plan"],
        help="Claude Code permission mode for the trial (default: bypassPermissions, so the "
        "agent runs unattended; claude only)",
    )
    parser.add_argument(
        "--allowed-tools",
        default=None,
        help="Tools the agent may use without prompting (space-separated; claude only)",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=DEFAULT_TRIALS_DIR,
        help="Where trial workspaces are staged (default: $TRIAL_WORKSPACE_ROOT or ~/.cache/hep-agent-trials)",
    )
    parser.add_argument(
        "--mcp-config",
        action="append",
        default=None,
        metavar="PATH_OR_JSON",
        help=f"MCP server config file or inline JSON (repeatable, default: {DEFAULT_MCP_CONFIG})",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Run without any --mcp-config (the default config is skipped)",
    )
    parser.add_argument(
        "--strict-mcp-config",
        action="store_true",
        help="Ignore user/project MCP settings, so only --mcp-config servers are loaded "
        "(claude only)",
    )
    parser.add_argument(
        "--global-skills",
        action="store_true",
        help="Also let the agent see your own and your plugins' skills (default: only the "
        "staged skills/, plus Claude Code's built-in ones for claude)",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Subprocess timeout in seconds")
    parser.add_argument(
        "--max-budget-usd", type=float, default=None, help="Cap the spend on each question (claude only)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tracking_uri = connect_mlflow()
    mlflow.set_experiment(args.experiment)

    if args.no_mcp:
        args.mcp_config = []
    elif args.mcp_config is None:
        args.mcp_config = [str(DEFAULT_MCP_CONFIG)] if DEFAULT_MCP_CONFIG.exists() else []

    check_mcp_env(args.mcp_config)
    harness = HARNESSES[args.harness](args)
    harness.check_args()

    variables = dict(v.split("=", 1) for v in args.var)
    prompt = resolve_prompt(args.prompt, args.prompt_version)
    dataset, records = load_questions(args.dataset)
    if args.question:
        records = [select_question(records, key) for key in args.question]
    if args.limit is not None:
        records = records[: args.limit]
    # Fail on a template variable the dataset cannot fill before anything is launched.
    for record in records:
        render_prompt(prompt, {**record["inputs"], **variables})

    return 0 if run_trial(args, harness, prompt, dataset, records, variables, tracking_uri) else 1
