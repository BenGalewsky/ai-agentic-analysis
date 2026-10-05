"""The agent harnesses a trial can run: Claude Code and opencode."""

from __future__ import annotations

from .base import Harness, RunSummary, ToolCall, parse_event, read_events
from .claude import ClaudeCode
from .opencode import OpenCode

HARNESSES: dict[str, type[Harness]] = {h.name: h for h in (ClaudeCode, OpenCode)}

__all__ = ["HARNESSES", "Harness", "RunSummary", "ToolCall", "parse_event", "read_events"]
