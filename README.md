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

`run_trial.py` performs one trial end to end:

1. Loads a prompt version from the MLflow Prompt Registry (default: `TopQuark`,
   latest version).
2. Stages a clean workspace and copies `skills/` into it as `.claude/skills`.
3. Runs `claude --print` in that workspace as a subprocess, streaming JSON events
   to the console and to `claude_stream.jsonl`.
4. Logs params, metrics, artifacts and an MLflow trace (one child span per tool
   call) back to the tracking server.

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

```bash
uv run run_trial.py
```

```bash
uv run run_trial.py --prompt TopQuark --prompt-version 1 --model opus
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--prompt` / `--prompt-version` | Which registered prompt to run (default: latest `TopQuark`) |
| `--var KEY=VALUE` | Fill a prompt template variable (repeatable) |
| `--experiment` | MLflow experiment name (default: `hep-plot-agent`) |
| `--model` | Model alias passed to `claude`, e.g. `opus` |
| `--allowed-tools` | Tools the agent may use without prompting |
| `--mcp-config` | MCP config file or inline JSON (repeatable, default: `scripts/mcp.json`) |
| `--no-mcp` | Run the trial with no MCP servers |
| `--strict-mcp-config` | Load only the servers in `--mcp-config`, ignoring user/project settings |
| `--permission-mode` | Defaults to `bypassPermissions` so the trial runs unattended |
| `--trials-dir` | Where workspaces are staged (default: `$TRIAL_WORKSPACE_ROOT` or `~/.cache/hep-agent-trials`) |
| `--timeout` | Subprocess timeout in seconds (default: 3600) |
| `--max-budget-usd` | Cap the spend on a single trial |

The script exits non-zero when the trial fails, so it composes into a sweep.

## What gets recorded

**Params** — prompt name/version/URI, model, permission mode, allowed tools, the
MCP config path and the server names it declares, the skill list and its content
hash.

**Metrics** — wall time, API duration, turns, cost in USD, input/output/cache
tokens, tool-call count, `completed`, and whether a script and a plot were
produced.

**Artifacts** — the rendered prompt, a snapshot of `skills/`, the raw event
stream, `result.json`, stderr, the agent's final message, everything it wrote
under `outputs/`, and the promoted `final/` deliverables.

**Trace** — the trial as a single agent span with a tool span per call, so a run
can be replayed in the MLflow UI.

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
run_trial.py    # the harness
skills/         # domain skills staged into every trial workspace
trials/         # local trial output (gitignored)
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
