# ai-agentic-analysis

A test harness for measuring how well an agentic coding assistant can carry out a
HEP physics analysis task. A trial gives an agent harness — Claude Code or
[opencode](https://opencode.ai) — a physics prompt plus a curated set of domain
skills and MCP servers, lets it work unattended in a clean workspace, and records
everything — cost, turns, tool calls, and the script and plot it produced — to
MLflow.

The question it exists to answer: *do the skills in `skills/` actually make the
agent better at writing a ServiceX/Awkward/hist analysis?* Because prompts are
versioned in the MLflow Prompt Registry and the skill tree is content-hashed into
every run, trials stay comparable as both evolve. Because both harnesses get the
same prompt, skills and MCP servers and are graded and recorded the same way, a
trial per harness also answers *how much does the harness itself matter?*

## How a trial works

`uv run run-trial` performs one trial end to end:

1. Loads a prompt version from the MLflow Prompt Registry (default: `IRIS-HEP`,
   latest version).
2. Loads the questions from the `hep-data-llm-questions` MLflow evaluation
   dataset, which `scripts/register_questions.py` populates.
3. For each question, renders the prompt with the record's `inputs` (the
   `{{ question }}` variable), stages a clean workspace with `skills/` copied in
   where the harness looks for project skills, and runs the harness there as a
   subprocess, streaming JSON events to the console and to its stream file:

   | `--harness` | Runs | Skills staged in | Stream file |
   | --- | --- | --- | --- |
   | `claude` (default) | `claude --print --output-format stream-json` | `.claude/skills` | `claude_stream.jsonl` |
   | `opencode` | `opencode run --standalone --format json --auto` | `.opencode/skills` | `opencode_stream.jsonl` |

4. Logs the whole trial as one MLflow evaluation run, the shape
   `mlflow.genai.evaluate` gives its runs: the dataset as the run's input, an
   MLflow trace per question (one child span per tool call) carrying the grade
   and the record's expectations, and the aggregate completion rate, accuracy
   and cost as run metrics. With `--repeats N` the question is run N times, each
   repeat a trace of its own.

Deliverables — the most recently modified `.py` and the most recently modified
image — are promoted to the `final/` artifact path, so a plot is always in the
same place across runs.

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and the CLI of each
harness you run on `PATH`: [Claude Code](https://claude.com/claude-code) for
`--harness claude`, [opencode](https://opencode.ai) v2 for `--harness opencode`
(or point `--harness-bin` at either).

```bash
uv sync
```

This installs the harness package (`src/agent_harness/`) and its `run-trial` and
`grade-trial` commands. Run them from the repo root, since `skills/`, `mcp.json`
and `.env` are found relative to the working directory.

Point the harness at your MLflow tracking server by copying `.env.example` to
`.env` and filling in the URI:

```bash
cp .env.example .env
```

`.env` is gitignored — keep credentials out of the repo.

### opencode

opencode takes its providers and models from your own opencode config
(`~/.config/opencode/opencode.json`), so set up the provider you want there first
and name the model as `provider/model`:

```bash
uv run run-trial --harness opencode --model lumen/qwen3-coder-next
```

Each run uses `--standalone`, a private opencode server started in the workspace
with the harness's environment, rather than the shared background service, which
would use its own working directory, environment and config. `--auto` approves
every permission request, the equivalent of Claude Code's `bypassPermissions`.
The JSON stream leaves out the last step's token usage, so after each run the
harness exports the session (`opencode session export`) to
`opencode_session.json` and takes the cost, tokens, turns and outcome from it.
The cost is opencode's, computed from the prices in your provider config.

The sessions stay in opencode's own database, so they can be reopened, since
opencode has no equivalent of `--no-session-persistence`.

## MCP servers

`mcp.json` declares the MCP servers the agent gets in every trial, in Claude
Code's format. The harness passes it to `claude` with `--mcp-config`, and
translates it for opencode (see below). It currently holds the
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

The harness loads `.env` and the subprocess inherits it, so the agent expands
`${MCP_BEARER_TOKEN}` at launch. An unset variable is not an error to the CLI —
it forwards the placeholder verbatim and the server answers 401 — so the harness
checks for it up front and refuses to start the trial.

The server's tools are named `mcp__af__<tool>`, and `mcp__af` in
`--allowed-tools` admits all of them. Use `--mcp-config` to point at a different
config (repeatable, also accepts inline JSON), `--no-mcp` to run without one, and
`--strict-mcp-config` to ignore whatever MCP servers your user and project
settings add, so a trial sees only what the config names.

For opencode, each server becomes an entry under opencode's `mcp.servers`:
`http` and `sse` servers become `remote` ones, with OAuth turned off since a trial
cannot sign in, and `stdio` servers become `local` ones. `${VAR}` is rewritten as
opencode's `{env:VAR}`, so the token is still read from the environment and never
written down. A `${VAR:-default}` cannot be translated and stops the trial. The
result is handed to opencode as `OPENCODE_CONFIG_CONTENT`, merged over your own
config, and logged on the trial run as `opencode_config.json`. opencode names the
server's tools `af_<tool>`, and its Code Mode calls them from scripts run by its
`execute` tool. The harness counts MCP calls made either way.

## Running

Run every question in the dataset:

```bash
uv run run-trial
```

Run a single question, by name or `question_index`:

```bash
uv run run-trial --question JetPtAll
```

Run a few questions while developing, in a separate experiment so the benchmark
results stay clean:

```bash
uv run run-trial --question JetPtAll --question 2 --experiment hep-plot-agent-dev
uv run run-trial --limit 2 --experiment hep-plot-agent-dev
```

Run each question several times, to see how consistently the agent gets it right:

```bash
uv run run-trial --question JetPtAll --repeats 5
```

```bash
uv run run-trial --prompt IRIS-HEP --prompt-version 1 --model opus
```

Compare the harnesses by running a trial with each, on the same questions:

```bash
uv run run-trial --question JetPtAll --repeats 5 --model opus
uv run run-trial --question JetPtAll --repeats 5 --harness opencode --model lumen/qwen3-coder-next
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--prompt` / `--prompt-version` | Which registered prompt to run (default: latest `IRIS-HEP`) |
| `--dataset` | Evaluation dataset of questions (default: `hep-data-llm-questions`) |
| `--question` | Run only this question, by its `name` tag or `question_index` (repeatable) |
| `--limit` | Run only the first N questions, by `question_index` |
| `--repeats` | Run each question N times (default: 1) |
| `--var KEY=VALUE` | Fill a prompt template variable (repeatable, overrides the dataset's inputs) |
| `--experiment` | MLflow experiment name (default: `hep-plot-agent`) |
| `--run-name` | Name of the parent trial run (default: `<prompt>-v<version>-<harness>-<timestamp>`) |
| `--harness` | Agent harness: `claude` (default) or `opencode` |
| `--harness-bin` | Path to the harness's CLI (default: `claude` or `opencode` on `PATH`) |
| `--model` | Model passed to the harness, e.g. `opus` for claude, `lumen/qwen3-coder-next` for opencode |
| `--allowed-tools` | Tools the agent may use without prompting (claude only) |
| `--mcp-config` | MCP config file or inline JSON (repeatable, default: `mcp.json`) |
| `--no-mcp` | Run the trial with no MCP servers |
| `--strict-mcp-config` | Load only the servers in `--mcp-config`, ignoring user/project settings (claude only) |
| `--permission-mode` | Defaults to `bypassPermissions` so the trial runs unattended (claude only) |
| `--trials-dir` | Where workspaces are staged (default: `$TRIAL_WORKSPACE_ROOT` or `~/.cache/hep-agent-trials`) |
| `--global-skills` | Also let the agent see your own and your plugins' skills (default: only the staged `skills/`) |
| `--timeout` | Per-run timeout in seconds, after which the agent and everything it started are killed (default: 3600) |
| `--max-budget-usd` | Cap the spend on each question (claude only) |

Before anything runs, the harness checks that its CLI works (`--version`), and for
opencode that it is v2 and the model is named `provider/model`. Passing a
claude-only option with `--harness opencode` also stops the trial before it
starts, rather than being silently ignored. Ctrl-C stops the agent along with the
harness. The script exits non-zero when any
question fails, so it composes into a sweep.

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

The grader (`agent_harness/grader.py`) reads those lines from the last tool call
that printed any (the agent's final run of its script) and passes the question when there are exactly
`n_plots` of them and each reference plot is matched, one-to-one and in any order,
by a line whose `mean` is within 1%. As in hep-data-llm, `avg_entries_per_event` is
reported but not gated, since there are several valid ways to count entries.

The harness grades every question as it runs. To regrade trials already on disk,
for example with a tighter tolerance:

```bash
uv run grade-trial ~/.cache/hep-agent-trials/<trial-dir> --tolerance 0.005
```

With `--repeats`, each repeat is staged in `<trial-dir>/<question>/r<k>/`, and
the grader grades every repeat it finds against `<question>`'s record. It reads
whichever harness's stream file a run left, so Claude Code and opencode trials
regrade alike.

`agent_harness.grader.metrics_match` is also an MLflow scorer that reads the
METRIC lines from a logged trace's tool spans, so `mlflow.genai.evaluate` can rescore stored traces.

## What gets recorded

**Tags** — every trial run carries `run_type = trial`, `harness`, and
`mlflow.runType = genai_evaluate`, which lists it under the experiment's
**Evaluation runs**. Trials logged before question runs were folded into the
trial run also have a child run per question (`run_type = question`); to see
only the trials, filter the runs with:

```
tags.run_type = 'trial'
```

and add `and tags.harness = 'opencode'` to narrow them to one harness.

**Params** — the harness and its version, prompt name/version/URI, dataset name
and ID, model, permission mode, the MCP config path and the server names it
declares, the skill list and its content hash, and `num_repeats`; for claude also
the allowed tools and `strict_mcp_config`; and the question filter and count.

**Metrics** — per repeat, as attributes of its trace's agent span: `wall_seconds`, `duration_ms`, `api_duration_ms`,
`num_turns`, `cost_usd`, input/output/cache tokens, tool-call count, `num_tool_errors` (tool results
flagged as errors), `completed`, and whether a script and a plot were produced.
Skill usage is counted as `num_skill_calls` (invocations of the skill tool —
Claude Code's `Skill`, opencode's `skill`), `skill_calls_<skill>` for each skill,
and `num_skill_file_reads` (read calls on a staged skill's files); MCP usage as
`num_mcp_calls`. opencode does not report API time, so its runs have no
`api_duration_ms`, and a turn is one of its model steps; a shell command that
exits non-zero counts as a tool error under both harnesses. The grade adds `metrics_match`, `num_metric_lines`,
`num_plots_expected`, `num_plots_matched` and each plot's `plot_<i>_mean_rel_err`
and `plot_<i>_avg_entries_rel_err`.

The trial run rolls these up so two trials — say, before and after a skills
change — compare at a glance:

| Group | Metrics |
| --- | --- |
| Outcomes | `accuracy` (also logged as `metrics_match/mean`, the name `mlflow.genai.evaluate` gives a scorer's mean), `pass_at_k` (share of questions passed by at least one repeat), `pass_all_k` (share passed by every repeat), `completion_rate`, `script_rate`, `plot_rate`, `plot_accuracy` (reference plots matched, giving partial credit on multi-plot questions), plus the counts behind them: `total_completed`, `total_correct`, `total_produced_script`, `total_produced_plot`, `total_plots_expected`, `total_plots_matched` |
| Cost and effort | `cost_per_correct_usd`, `tool_error_rate`, and `total_` and `mean_` of `cost_usd`, `wall_seconds`, `duration_ms`, `api_duration_ms`, `turns`, `tool_calls` and `tool_errors`; `total_` input/output/cache tokens |
| Skill usage | `skill_usage_rate` (share of questions that invoked any skill), `total_skill_calls`, `mean_skill_calls`, `num_distinct_skills_used`, `total_skill_file_reads`, `total_skill_calls_<skill>` |
| MCP usage | `mcp_usage_rate` (share of questions that called any MCP tool), `total_mcp_calls`, `mean_mcp_calls` |

Rates and means are taken over every repeat of every question, and `pass_at_k`
and `pass_all_k` equal `accuracy` when each question runs once. Rates and means
compare across trials with different numbers of questions. `plot_accuracy`,
`cost_per_correct_usd` and `tool_error_rate` are left out when their denominator
is zero, and a rollup of a value the harness does not report, such as opencode's
`api_duration_ms`, is left out rather than logged as 0. The trial run also logs `questions.json`, a table with one row per
repeat of each question, for side-by-side comparison in the MLflow UI.

**Artifacts** — the trial run holds the prompt template, the MCP config (and,
for opencode, its translation `opencode_config.json`) and a snapshot of
`skills/`. Under `<question>/` it holds each question's rendered prompt, the record's
`inputs.json` and `expectations.json`, the raw event stream (and, for opencode,
`opencode_session.json`), `result.json` (Claude Code's result event, or the
opencode session's summary),
stderr, the agent's final message, everything it wrote under `outputs/`, the
promoted `final/` deliverables, and `grade.json` with the per-plot comparison.
With `--repeats` above 1, everything after `expectations.json` is per repeat and
sits under `<question>/r<k>/`, e.g. `JetPtAll/r2/final/`.

**Trace** — each repeat as a single agent span (`claude_trial` or
`opencode_trial`) with a tool span per call, marked as an error when the call
failed, so it
can be replayed in the MLflow UI. Its request is the dataset record's
`inputs` (the question), and the rendered prompt is the agent span's `prompt`
attribute. The grade is attached to it as a `metrics_match` feedback
assessment, and the record's expectations (`plots`, `n_plots`) as expectation
assessments; the repeat's metrics are attributes of the agent span, and its tags
carry `question`, `question_index`, `datasets`, `dataset_record_id`, `repeat`,
`status`, `failure_reason`, `session_id`, `grade` and the final script and plot
names. A repeat that produced no output, such as a crash, still gets a trace,
marked as an error.

Every trace belongs to the trial run, so the run's **Traces** tab is the trial's
evaluation table. Pick another trial under **compare to** there to see the two
side by side, matched question by question on their requests — which holds even
across prompt versions, since the request is the question rather than the
rendered prompt. With `--repeats`, the extra repeats of a question have no
partner and show as rows of their own.

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

By default a trial's agent sees only these skills, not the ones in your own
`~/.claude/skills`, `~/.agents/skills` or plugins, so results don't depend on whose
machine ran them. opencode is given a skill permission list that denies every
skill but the staged ones, which also hides its own built-in `opencode` and
`report` skills. Claude Code runs with `--setting-sources project,local`, which
skips your user settings and with them your skills and plugins. Its built-in
skills (`dataviz`, `loop`, `code-review` and so on) cannot be removed without
removing the staged ones too, so they remain. Skipping user settings also means
claude's `model` and effort settings no longer apply: pass `--model` to choose
one. `--global-skills` turns the isolation off, and is logged as the
`global_skills` param.

## Layout

```
src/agent_harness/       # the harness package
  cli.py                 # run-trial: options and setup
  trial.py               # runs the questions and their repeats under one trial run
  harnesses/             # the agents a trial can run, each read into one RunSummary
    base.py              # the Harness interface and the subprocess loop
    claude.py            # claude -p and its stream-json events
    opencode.py          # opencode run, its JSON events and the MCP config translation
  tracing.py             # MLflow traces
  rollup.py              # question- and trial-level metrics
  prompts.py, questions.py, workspace.py, mcp_config.py, config.py
  grader.py              # grade-trial: scores a trial's METRIC lines against the expectations
scripts/
  register_questions.py  # loads the benchmark questions into the MLflow dataset
mcp.json        # MCP servers the agent gets in every trial
skills/         # domain skills staged into every trial workspace
trials/         # local trial output (gitignored)
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
