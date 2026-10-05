"""Read and check the MCP server configs handed to Claude Code."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def is_inline_json(config: str) -> bool:
    """Whether an ``--mcp-config`` value is a JSON string rather than a file path."""
    return config.lstrip().startswith("{")


def read_mcp_config(config: str) -> str:
    return config if is_inline_json(config) else Path(config).read_text()


def mcp_server_names(configs: list[str]) -> list[str]:
    """Server names declared across the MCP configs, for logging/provenance."""
    names: list[str] = []
    for config in configs:
        names += list(json.loads(read_mcp_config(config)).get("mcpServers", {}))
    return sorted(set(names))


def check_mcp_env(configs: list[str]) -> None:
    """Fail before launch on an unset ``${VAR}`` in a config.

    Claude Code passes an unexpanded placeholder through verbatim, so a missing
    token surfaces only as an authentication failure once the trial is running.
    """
    missing = {
        name
        for config in configs
        for name, default in re.findall(r"\$\{(\w+)(:-[^}]*)?\}", read_mcp_config(config))
        if not default and name not in os.environ
    }
    if missing:
        raise SystemExit(
            f"MCP config needs unset environment variable(s): {', '.join(sorted(missing))} "
            "(expected in .env)"
        )
