# ai-agentic-analysis

A test harness for measuring how well an agentic coding assistant can carry out a
HEP physics analysis task. A trial gives Claude Code a physics prompt plus a
curated set of domain skills, lets it work unattended in a clean workspace, and
records everything — cost, turns, tool calls, and the script and plot it produced —
to MLflow.

The question it exists to answer: *do the skills in `skills/` actually make the
agent better at writing a ServiceX/Awkward/hist analysis?* Because prompts are
versioned in the MLflow Prompt Registry and the skill tree is content-hashed into
every run, trials stay comparable as both evolve.

## How a trial works

`scripts/run_trial.py` performs one trial end to end:

1. Loads a prompt version from the MLflow Prompt Registry (default: `IRIS-HEP`,
   latest version).
2. Loads the questions from the `hep-data-llm-questions` MLflow evaluation
   dataset, which `scripts/register_questions.py` populates.
3. For each question, renders the prompt with the record's `inputs` (the
   `{{ question }}` variable), stages a clean workspace with `skills/` copied in
   as `.claude/skills`, and runs `claude --print` there as a subprocess,
   streaming JSON events to the console and to `claude_stream.jsonl`.
4. Logs each question as a child run — params, metrics, artifacts and an MLflow
   trace (one child span per tool call) — under a parent run for the whole
   trial, which carries the aggregate completion rate and cost.

Deliverables — the most recently modified `.py` and the most recently modified
image — are promoted to the `final/` artifact path, so a plot is always in the
same place across runs.

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and the
[Claude Code](https://claude.com/claude-code) CLI on `PATH`.

```bash
uv sync
```

Point the harness at your MLflow tracking server by copying `.env.example` to
`.env` and filling in the URI:

```bash
cp .env.example .env
```

`.env` is gitignored — keep credentials out of the repo.

## MCP servers

`scripts/mcp.json` declares the MCP servers the agent gets in every trial, and
the harness passes it to `claude` with `--mcp-config`. It currently holds the
UChicago Analysis Facility server, authenticated with a personal access token:

```json
{
  "mcpServers": {
    "af": {
      "type": "http",
      "url": "https://mcp.af.uchicago.edu/mcp/",
      "headers": { "Authorization": "Bearer ${MCP_BEARER_TOKEN}" }
    }
  }
}
```

A trial runs headless and cannot complete the browser OAuth flow, so it uses the
[static token](https://maniaclab.uchicago.edu/af-mcp-platform/connecting-a-client/)
route. Mint one at [mcp-portal.af.uchicago.edu/tokens/](https://mcp-portal.af.uchicago.edu/tokens/)
— it is shown exactly once — and put it in `.env`:

```
MCP_BEARER_TOKEN=<your-token>
```

The harness loads `.env` and the subprocess inherits it, so `claude` expands
`${MCP_BEARER_TOKEN}` at launch. An unset variable is not an error to the CLI —
it forwards the placeholder verbatim and the server answers 401 — so the harness
checks for it up front and refuses to start the trial.

The server's tools are named `mcp__af__<tool>`, and `mcp__af` in
`--allowed-tools` admits all of them. Use `--mcp-config` to point at a different
config (repeatable, also accepts inline JSON), `--no-mcp` to run without one, and
`--strict-mcp-config` to ignore whatever MCP servers your user and project
settings add, so a trial sees only what the config names.

## Running

Run every question in the dataset:

```bash
uv run scripts/run_trial.py
```

Run a single question, by name or `question_index`:

```bash
uv run scripts/run_trial.py --question JetPtAll
```

Run a few questions while developing, in a separate experiment so the benchmark
results stay clean:

```bash
uv run scripts/run_trial.py --question JetPtAll --question 2 --experiment hep-plot-agent-dev
uv run scripts/run_trial.py --limit 2 --experiment hep-plot-agent-dev
```

```bash
uv run scripts/run_trial.py --prompt IRIS-HEP --prompt-version 1 --model opus
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--prompt` / `--prompt-version` | Which registered prompt to run (default: latest `IRIS-HEP`) |
| `--dataset` | Evaluation dataset of questions (default: `hep-data-llm-questions`) |
| `--question` | Run only this question, by its `name` tag or `question_index` (repeatable) |
| `--limit` | Run only the first N questions, by `question_index` |
| `--var KEY=VALUE` | Fill a prompt template variable (repeatable, overrides the dataset's inputs) |
| `--experiment` | MLflow experiment name (default: `hep-plot-agent`) |
| `--model` | Model alias passed to `claude`, e.g. `opus` |
| `--allowed-tools` | Tools the agent may use without prompting |
| `--mcp-config` | MCP config file or inline JSON (repeatable, default: `scripts/mcp.json`) |
| `--no-mcp` | Run the trial with no MCP servers |
| `--strict-mcp-config` | Load only the servers in `--mcp-config`, ignoring user/project settings |
| `--permission-mode` | Defaults to `bypassPermissions` so the trial runs unattended |
| `--trials-dir` | Where workspaces are staged (default: `$TRIAL_WORKSPACE_ROOT` or `~/.cache/hep-agent-trials`) |
| `--timeout` | Subprocess timeout in seconds (default: 3600) |
| `--max-budget-usd` | Cap the spend on each question |

The script exits non-zero when any question fails, so it composes into a sweep.

## Grading

Each record's `expectations` hold reference values for every plot the question
asks for:

```json
{"plots": [{"avg_entries_per_event": 1.0, "mean": 16.451025}], "n_plots": 1}
```

The prompt has the agent's script print one line per plot describing the values it
filled the histogram with, in hep-data-llm's format:

```
METRIC: avg_entries_per_event=<N> mean=<M>
```

`scripts/grader.py` reads those lines from the last tool call that printed any (the
agent's final run of its script) and passes the question when there are exactly
`n_plots` of them and each reference plot is matched, one-to-one and in any order,
by a line whose `mean` is within 1%. As in hep-data-llm, `avg_entries_per_event` is
reported but not gated, since there are several valid ways to count entries.

The harness grades every question as it runs. To regrade trials already on disk,
for example with a tighter tolerance:

```bash
uv run scripts/grader.py ~/.cache/hep-agent-trials/<trial-dir> --tolerance 0.005
```

`grader.metrics_match` is also an MLflow scorer that reads the METRIC lines from a
logged trace's tool spans, so `mlflow.genai.evaluate` can rescore stored traces.

## What gets recorded

**Params** — prompt name/version/URI, dataset name and ID, model, permission
mode, allowed tools, the MCP config path and the server names it declares, the
skill list and its content hash. Question runs add the question name, index and
dataset record ID.

**Metrics** — per question: wall time, API duration, turns, cost in USD,
input/output/cache tokens, tool-call count, `num_tool_errors` (tool results
flagged as errors), `completed`, and whether a script and a plot were produced.
Skill usage is counted as `num_skill_calls` (invocations of the `Skill` tool),
`skill_calls_<skill>` for each skill, and `num_skill_file_reads` (`Read` calls on
a staged skill's files). The grade adds `metrics_match`, `num_metric_lines`,
`num_plots_expected`, `num_plots_matched` and each plot's `plot_<i>_mean_rel_err`
and `plot_<i>_avg_entries_rel_err`.

The parent run rolls these up so two trials — say, before and after a skills
change — compare at a glance:

| Group | Metrics |
| --- | --- |
| Outcomes | `accuracy`, `completion_rate`, `script_rate`, `plot_rate`, `plot_accuracy` (reference plots matched, giving partial credit on multi-plot questions), plus the `num_*` counts behind them |
| Cost and effort | `total_cost_usd`, `mean_cost_usd`, `cost_per_correct_usd`, `wall_seconds`, `mean_wall_seconds`, `mean_turns`, `mean_tool_calls`, `mean_tool_errors`, `tool_error_rate`, total input/output/cache tokens |
| Skill usage | `skill_usage_rate` (share of questions that invoked any skill), `num_skill_calls`, `mean_skill_calls`, `num_distinct_skills_used`, `num_skill_file_reads`, `skill_calls_<skill>` |

Rates and means compare across trials with different numbers of questions.
`plot_accuracy`, `cost_per_correct_usd` and `tool_error_rate` are left out when
their denominator is zero. The parent run also logs `questions.json`, a table
with one row per question, for side-by-side comparison in the MLflow UI.

**Artifacts** — the parent run holds the prompt template, the MCP config and a
snapshot of `skills/`. Each question run holds the rendered prompt, the record's
`inputs.json` and `expectations.json`, the raw event stream, `result.json`,
stderr, the agent's final message, everything it wrote under `outputs/`, the
promoted `final/` deliverables, and `grade.json` with the per-plot comparison.

**Trace** — the trial as a single agent span with a tool span per call, so a run
can be replayed in the MLflow UI. The grade is attached to it as a
`metrics_match` feedback assessment.

## Skills

The skills staged into each workspace cover the HEP Python stack:

| Skill | Covers |
| --- | --- |
| `analysis-spec-builder` | Turning a loose request into a written analysis specification |
| `servicex` | `func_adl` queries against ATLAS xAOD (PHYSLITE/PHYS), dataset selection, `deliver` |
| `awkward-array` | Jagged arrays and records: `ak.zip`, combinatorics, `argmin`/`argmax`, flattening, NumPy interop |
| `vector-awkward` | scikit-hep `vector` behaviors, deltaR, invariant masses, boosts |
| `hist` | Building, filling, slicing and plotting histograms with `hist` and `mplhep` |
| `standalone-script` | One-off scripts with PEP 723 inline metadata and a `uv run` shebang |
| `cli-creator` | Typer CLIs wired into `pyproject.toml` entry points |

Editing a skill changes `skills_hash`, which is the handle for comparing runs
before and after a skill change.

## Layout

```
scripts/
  run_trial.py           # the harness
  register_questions.py  # loads the benchmark questions into the MLflow dataset
  grader.py              # scores a trial's METRIC lines against the expectations
skills/         # domain skills staged into every trial workspace
trials/         # local trial output (gitignored)
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
