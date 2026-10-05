"""Read tool calls, skill use and the final result out of a Claude Code event stream."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from typing import Any


def parse_event(line: str) -> dict[str, Any] | None:
    """One ``stream-json`` line as an event; ``None`` for a blank or non-JSON line."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def tool_results(events: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """The ``tool_result`` blocks of the user turns, in order."""
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield block


def tool_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ``tool_use`` blocks and pair them with their results."""
    results = {block.get("tool_use_id"): block.get("content") for block in tool_results(events)}

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
    return sum(bool(block.get("is_error")) for block in tool_results(events))


def result_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The final ``result`` event, or ``{}`` when the run ended without one."""
    return next((e for e in reversed(events) if e.get("type") == "result"), {})


def final_text(events: list[dict[str, Any]]) -> str:
    return result_event(events).get("result") or ""
