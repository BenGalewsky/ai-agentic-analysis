"""Run Claude Code as a subprocess, streaming its JSON events."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the CLI invocation. The prompt goes over stdin, not argv: the
    ``--allowed-tools <tools...>`` option is variadic and would swallow a
    trailing positional prompt.
    """
    cmd = [
        args.claude_bin,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        args.permission_mode,
        "--no-session-persistence",
    ]
    for config in args.mcp_config:
        # The agent runs in a staged workspace, so relative paths from the repo
        # root would not resolve; inline JSON strings are passed through as-is.
        value = config if config.lstrip().startswith("{") else str(Path(config).resolve())
        cmd += ["--mcp-config", value]
    if args.strict_mcp_config:
        cmd.append("--strict-mcp-config")
    if args.allowed_tools:
        cmd += ["--allowed-tools", args.allowed_tools]
    if args.model:
        cmd += ["--model", args.model]
    if args.max_budget_usd:
        cmd += ["--max-budget-usd", str(args.max_budget_usd)]
    return cmd


def run_claude(
    cmd: list[str], prompt_text: str, workspace: Path, stream_path: Path, timeout: int
) -> tuple[list[dict[str, Any]], int, str]:
    """Run Claude Code, tee the event stream to disk, and return parsed events."""
    events: list[dict[str, Any]] = []
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}

    with stream_path.open("w") as stream_file:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(prompt_text)
        proc.stdin.close()
        try:
            for line in proc.stdout:
                stream_file.write(line)
                stream_file.flush()
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(event)
                log_event(event)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        stderr = proc.stderr.read() if proc.stderr else ""

    return events, proc.returncode, stderr


def log_event(event: dict[str, Any]) -> None:
    """Minimal console trace so a long trial is watchable."""
    kind = event.get("type")
    if kind == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                print(f"  [assistant] {block['text'].strip()[:160]}")
            elif block.get("type") == "tool_use":
                print(f"  [tool] {block.get('name')}")
    elif kind == "result":
        print(f"  [result] {event.get('subtype')} in {event.get('duration_ms', 0) / 1000:.1f}s")
