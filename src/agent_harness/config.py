"""Paths and defaults shared across the harness.

Paths are relative to the working directory, so the commands run from the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(".")
SKILLS_DIR = PROJECT_ROOT / "skills"
DEFAULT_TRIALS_DIR = Path(
    os.environ.get("TRIAL_WORKSPACE_ROOT")
    or Path.home() / ".cache" / "hep-agent-trials"
)

DEFAULT_EXPERIMENT = "hep-plot-agent"
DEFAULT_PROMPT = "IRIS-HEP"
DEFAULT_DATASET = "hep-data-llm-questions"
DEFAULT_ALLOWED_TOOLS = (
    "Bash Read Write Edit Glob Grep Skill WebFetch WebSearch TodoWrite mcp__af"
)
DEFAULT_MCP_CONFIG = PROJECT_ROOT / "mcp.json"

SCRIPT_SUFFIXES = (".py",)
PLOT_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".svg")
