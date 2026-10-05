"""Read tool calls, skill use and the final result out of a Claude Code event stream."""

from __future__ import annotations

from collections import Counter
from typing import Any


def tool_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ``tool_use`` blocks and pair them with their results."""
    results: dict[str, Any] = {}
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block.get("content")

    calls = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if block.get("type") == "tool_use":
                calls.append(
                    {
                        "name": block.get("name"),
                        "input": block.get("input"),
                        "output": results.get(block.get("id")),
                    }
                )
    return calls


def skill_calls(calls: list[dict[str, Any]]) -> Counter[str]:
    """How many times the agent invoked each skill through the ``Skill`` tool."""
    return Counter(
        (call["input"] or {}).get("skill", "unknown") for call in calls if call["name"] == "Skill"
    )


def skill_file_reads(calls: list[dict[str, Any]]) -> int:
    """``Read`` calls on a staged skill's files, e.g. a reference doc beyond ``SKILL.md``."""
    return sum(
        call["name"] == "Read" and ".claude/skills/" in str((call["input"] or {}).get("file_path", ""))
        for call in calls
    )


def tool_errors(events: list[dict[str, Any]]) -> int:
    """Tool results flagged as errors: failed commands, bad edits, denied calls."""
    return sum(
        isinstance(block, dict) and block.get("type") == "tool_result" and bool(block.get("is_error"))
        for event in events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", []) or []
    )


def final_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("result") or ""
    return ""
