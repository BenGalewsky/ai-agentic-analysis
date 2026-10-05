"""Claude Code, run as ``claude --print`` with its ``stream-json`` event stream."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import DEFAULT_ALLOWED_TOOLS
from ..mcp_config import is_inline_json
from .base import Harness, RunSummary, ToolCall

DEFAULT_PERMISSION_MODE = "bypassPermissions"


class ClaudeCode(Harness):
    name = "claude"
    default_bin = "claude"
    stream_file = "claude_stream.jsonl"
    skills_dir = ".claude/skills"

    @property
    def permission_mode(self) -> str:
        return self.args.permission_mode or DEFAULT_PERMISSION_MODE

    @property
    def allowed_tools(self) -> str:
        return DEFAULT_ALLOWED_TOOLS if self.args.allowed_tools is None else self.args.allowed_tools

    def command(self) -> list[str]:
        """The prompt goes over stdin, not argv: the ``--allowed-tools <tools...>``
        option is variadic and would swallow a trailing positional prompt.
        """
        args = self.args
        cmd = [
            self.bin,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            self.permission_mode,
            "--no-session-persistence",
        ]
        for config in args.mcp_config:
            # The agent runs in a staged workspace, so relative paths from the repo
            # root would not resolve; inline JSON strings are passed through as-is.
            value = config if is_inline_json(config) else str(Path(config).resolve())
            cmd += ["--mcp-config", value]
        if args.strict_mcp_config:
            cmd.append("--strict-mcp-config")
        if not args.global_skills:
            # Skips the user's settings, and with them their own and their plugins'
            # skills; the staged project skills and Claude Code's built-ins remain.
            cmd += ["--setting-sources", "project,local"]
        if self.allowed_tools:
            cmd += ["--allowed-tools", self.allowed_tools]
        if args.model:
            cmd += ["--model", args.model]
        if args.max_budget_usd:
            cmd += ["--max-budget-usd", str(args.max_budget_usd)]
        return cmd

    def params(self) -> dict[str, Any]:
        return {
            "permission_mode": self.permission_mode,
            "allowed_tools": self.allowed_tools,
            "strict_mcp_config": self.args.strict_mcp_config,
            "global_skills": self.args.global_skills,
        }

    def tool_calls(self, events: list[dict[str, Any]]) -> list[ToolCall]:
        """Flatten ``tool_use`` blocks and pair them with their results."""
        results = {block.get("tool_use_id"): block for block in tool_results(events)}

        calls = []
        for event in events:
            if event.get("type") != "assistant":
                continue
            for block in event.get("message", {}).get("content", []) or []:
                if block.get("type") == "tool_use":
                    result = results.get(block.get("id"), {})
                    calls.append(
                        ToolCall(
                            name=block.get("name"),
                            input=block.get("input"),
                            output=result.get("content"),
                            is_error=bool(result.get("is_error")),
                        )
                    )
        return calls

    def summarize(self, events: list[dict[str, Any]], run_dir: Path) -> RunSummary:
        result = result_event(events)
        usage = result.get("usage", {}) or {}
        calls = self.tool_calls(events)
        return RunSummary(
            calls=calls,
            result=result,
            final_text=result.get("result") or "",
            session_id=result.get("session_id", ""),
            is_error=bool(result.get("is_error")),
            failure_reason=result.get("subtype", ""),
            cost_usd=result.get("total_cost_usd", 0.0) or 0.0,
            input_tokens=usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("output_tokens", 0) or 0,
            cache_read_tokens=usage.get("cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0) or 0,
            num_turns=result.get("num_turns", 0) or 0,
            duration_ms=result.get("duration_ms", 0) or 0,
            api_duration_ms=result.get("duration_api_ms", 0) or 0,
            # Invocations of the ``Skill`` tool, and ``Read`` calls on a staged
            # skill's files, e.g. a reference doc beyond ``SKILL.md``.
            skill_calls=Counter(
                (call.input or {}).get("skill", "unknown") for call in calls if call.name == "Skill"
            ),
            num_skill_file_reads=sum(
                call.name == "Read"
                and f"{self.skills_dir}/" in str((call.input or {}).get("file_path", ""))
                for call in calls
            ),
            # MCP tools are named ``mcp__<server>__<tool>``.
            num_mcp_calls=sum((call.name or "").startswith("mcp__") for call in calls),
        )

    def log_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    print(f"  [assistant] {block['text'].strip()[:160]}")
                elif block.get("type") == "tool_use":
                    print(f"  [tool] {block.get('name')}")
        elif kind == "result":
            print(f"  [result] {event.get('subtype')} in {event.get('duration_ms', 0) / 1000:.1f}s")


def tool_results(events: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """The ``tool_result`` blocks of the user turns, in order."""
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield block


def result_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The final ``result`` event, or ``{}`` when the run ended without one."""
    return next((e for e in reversed(events) if e.get("type") == "result"), {})
