"""opencode (v2), run as ``opencode run --standalone --format json``.

``--standalone`` gives each run a private server started from the workspace with
the harness's environment; without it the client hands the prompt to the shared
background service, which has its own working directory, environment and config.
The provider and model setup still comes from the user's global opencode config.

The JSON stream carries the tool calls, but not the final step's token usage, so
after each run the session is exported (``opencode session export``) and the
cost, tokens, turns and outcome are taken from that.
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import SKILLS_DIR
from ..mcp_config import mcp_server_names, read_mcp_config
from .base import Harness, RunSummary, ToolCall, agent_env, kill_tree

SESSION_FILE = "opencode_session.json"

# Options only Claude Code understands, by their argparse dest.
CLAUDE_ONLY = {
    "permission_mode": "--permission-mode",
    "allowed_tools": "--allowed-tools",
    "strict_mcp_config": "--strict-mcp-config",
    "max_budget_usd": "--max-budget-usd",
}


class OpenCode(Harness):
    name = "opencode"
    default_bin = "opencode"
    stream_file = "opencode_stream.jsonl"
    skills_dir = ".opencode/skills"

    def check_args(self) -> None:
        super().check_args()
        if not re.search(r"\bv?2\.", self.version):
            raise SystemExit(
                f"{self.bin_path()} is {self.version!r}; the opencode harness needs v2 "
                "(its run --standalone and --format json). Point --harness-bin at it, "
                "e.g. --harness-bin ~/.opencode/bin/opencode"
            )
        if self.args.model and "/" not in self.args.model:
            raise SystemExit(
                f"opencode names models as provider/model, e.g. lumen/{self.args.model}; "
                "`opencode models` lists them"
            )
        passed = [flag for dest, flag in CLAUDE_ONLY.items() if getattr(self.args, dest, None)]
        if passed:
            raise SystemExit(
                f"Claude Code option(s) not supported with --harness opencode: {', '.join(passed)} "
                "(opencode runs with --auto, approving every permission request)"
            )
        self.config()  # an untranslatable MCP config fails here, before launch

    def command(self) -> list[str]:
        cmd = [self.bin, "run", "--standalone", "--format", "json", "--auto"]
        if self.args.model:
            cmd += ["--model", self.args.model]
        return cmd

    def config(self) -> dict[str, Any]:
        if not self.args:
            return {}
        config = opencode_config(self.args.mcp_config)
        if not self.args.global_skills:
            # opencode also reads ~/.claude/skills, ~/.agents/skills and its own
            # global skills; hide every skill but the staged ones from the model.
            staged = sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir())
            config["permissions"] = [
                {"action": "skill", "resource": "*", "effect": "deny"},
                *({"action": "skill", "resource": name, "effect": "allow"} for name in staged),
            ]
        return config

    def env(self) -> dict[str, str]:
        # Inline config is merged over the user's and the project's, and keeps the
        # workspace free of a config file the agent could read or edit.
        config = self.config()
        return {"OPENCODE_CONFIG_CONTENT": json.dumps(config)} if config else {}

    def params(self) -> dict[str, Any]:
        return {"permission_mode": "auto", "global_skills": self.args.global_skills}

    def config_artifacts(self) -> dict[str, str]:
        config = self.config()
        return {"opencode_config.json": json.dumps(config, indent=2)} if config else {}

    def after_run(self, events: list[dict[str, Any]], run_dir: Path, workspace: Path) -> list[Path]:
        session_id = session_of(events)
        if not session_id:
            return []
        path = run_dir / SESSION_FILE
        # Written straight to the file, and killed with everything it started on a
        # timeout, so the export's own private server can never hold up the trial.
        with path.open("w") as out, tempfile.TemporaryFile("w+") as err:
            proc = subprocess.Popen(
                [self.bin, "session", "export", "--standalone", session_id],
                cwd=workspace,
                stdout=out,
                stderr=err,
                text=True,
                env=agent_env() | {"PWD": str(workspace)},
                start_new_session=True,
            )
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                kill_tree(proc.pid)
                proc.wait()
            except BaseException:
                kill_tree(proc.pid)
                raise
            err.seek(0)
            problem = err.read().strip()
        if proc.returncode != 0:
            path.unlink(missing_ok=True)
            reason = "timed out" if proc.returncode == -signal.SIGKILL else problem
            print(f"warning  : could not export opencode session {session_id}: {reason}")
            return []
        directory = (json.loads(path.read_text()).get("info") or {}).get("location", {}).get("directory")
        if directory and Path(directory).resolve() != workspace.resolve():
            print(f"warning  : opencode ran in {directory}, not the workspace {workspace}")
        return [path]

    def tool_calls(self, events: list[dict[str, Any]]) -> list[ToolCall]:
        calls = []
        for part in tool_parts(events):
            state = part.get("state") or {}
            failed = state.get("status") == "error"
            calls.append(
                ToolCall(
                    name=part.get("tool"),
                    input=state.get("input"),
                    output=state.get("error") if failed else state.get("output"),
                    # A shell command that exits non-zero still completes; count it
                    # as an error, as Claude Code does.
                    is_error=failed or exit_code(state) not in (None, 0),
                )
            )
        return calls

    def summarize(self, events: list[dict[str, Any]], run_dir: Path) -> RunSummary:
        session_path = run_dir / SESSION_FILE
        session = json.loads(session_path.read_text()) if session_path.exists() else {}
        info = session.get("info") or {}
        steps = [e["part"] for e in events if e.get("type") == "step_finish" and e.get("part")]

        # Prefer the session's totals; the stream's steps miss the final one.
        tokens = info.get("tokens") or step_tokens(steps)
        cost = info["cost"] if "cost" in info else sum(s.get("cost", 0) for s in steps)
        turns = sum(m.get("type") == "assistant" for m in session.get("messages", []))
        turns = turns or sum(e.get("type") == "step_start" for e in events)

        times = info.get("time") or {}
        if times.get("created") and times.get("idle"):
            duration_ms = times["idle"] - times["created"]
        elif events:
            duration_ms = events[-1].get("timestamp", 0) - events[0].get("timestamp", 0)
        else:
            duration_ms = None

        errors = [e.get("error") or {} for e in events if e.get("type") == "error"]
        outcome = info.get("outcome")
        is_error = bool(errors) or (outcome is not None and outcome != "succeeded")
        failure_reason = ""
        if errors:
            failure_reason = str(errors[0].get("type") or errors[0].get("name") or "error")
        elif is_error:
            failure_reason = str(outcome)

        texts = [
            e["part"].get("text", "")
            for e in events
            if e.get("type") == "text" and (e.get("part") or {}).get("text", "").strip()
        ]
        calls = self.tool_calls(events)
        return RunSummary(
            calls=calls,
            result=info,
            final_text=texts[-1].strip() if texts else "",
            session_id=session_of(events),
            is_error=is_error,
            failure_reason=failure_reason,
            cost_usd=cost or 0.0,
            input_tokens=tokens.get("input", 0) or 0,
            output_tokens=tokens.get("output", 0) or 0,
            cache_read_tokens=(tokens.get("cache") or {}).get("read", 0) or 0,
            cache_creation_tokens=(tokens.get("cache") or {}).get("write", 0) or 0,
            num_turns=turns,
            duration_ms=duration_ms,
            api_duration_ms=None,  # not reported
            skill_calls=Counter(
                (call.input or {}).get("id") or (call.input or {}).get("name") or "unknown"
                for call in calls
                if call.name == "skill"
            ),
            num_skill_file_reads=sum(
                call.name == "read" and f"{self.skills_dir}/" in read_path(call.input)
                for call in calls
            ),
            num_mcp_calls=self.mcp_calls(events),
        )

    def mcp_calls(self, events: list[dict[str, Any]]) -> int:
        """MCP tool calls, whether made directly or from a Code Mode ``execute`` script.

        opencode names a server's tools ``<server>_<tool>``; Code Mode calls them as
        ``tools["<server>"].<tool>()`` and records each as ``<server>.<tool>``.
        """
        servers = [sanitize(name) for name in mcp_server_names(self.args.mcp_config)] if self.args else []
        count = 0
        for part in tool_parts(events):
            tool = part.get("tool") or ""
            if any(tool.startswith(f"{server}_") for server in servers):
                count += 1
            nested = (((part.get("state") or {}).get("metadata") or {}).get("metadata") or {})
            for call in nested.get("toolCalls") or []:
                server = str(call.get("tool", "")).split(".", 1)[0]
                if server in servers or (not servers and "." in str(call.get("tool", ""))):
                    count += 1
        return count

    def log_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        part = event.get("part") or {}
        if kind == "text" and part.get("text", "").strip():
            print(f"  [assistant] {part['text'].strip()[:160]}")
        elif kind == "tool_use":
            status = (part.get("state") or {}).get("status")
            print(f"  [tool] {part.get('tool')}" + (" (error)" if status == "error" else ""))
        elif kind == "error":
            error = event.get("error") or {}
            print(f"  [error] {error.get('type') or error.get('name')}: {error.get('message', '')}")


def tool_parts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["part"] for e in events if e.get("type") == "tool_use" and e.get("part")]


def session_of(events: list[dict[str, Any]]) -> str:
    return next((e["sessionID"] for e in events if e.get("sessionID")), "")


def step_tokens(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Token usage summed over ``step_finish`` parts, in the session's shape."""

    def total(*path: str) -> int:
        def get(step: dict[str, Any]) -> int:
            value: Any = step.get("tokens") or {}
            for key in path:
                value = (value or {}).get(key, 0)
            return value or 0

        return sum(get(step) for step in steps)

    return {
        "input": total("input"),
        "output": total("output"),
        "cache": {"read": total("cache", "read"), "write": total("cache", "write")},
    }


def exit_code(state: dict[str, Any]) -> int | None:
    return ((state.get("metadata") or {}).get("metadata") or {}).get("exit")


def read_path(tool_input: Any) -> str:
    tool_input = tool_input or {}
    return str(tool_input.get("path") or tool_input.get("filePath") or tool_input.get("file_path") or "")


def sanitize(name: str) -> str:
    """A server name as opencode prefixes its tools with it."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


# --------------------------------------------------------------------------- #
# MCP config translation
# --------------------------------------------------------------------------- #
_ENV_REF = re.compile(r"\$\{(\w+)(:-[^}]*)?\}")


def opencode_config(configs: list[str]) -> dict[str, Any]:
    """Translate Claude Code MCP configs (``mcpServers``) into opencode's ``mcp.servers``."""
    servers = {}
    for config in configs:
        for name, server in json.loads(read_mcp_config(config)).get("mcpServers", {}).items():
            servers[name] = opencode_server(name, server)
    return {"mcp": {"servers": servers}} if servers else {}


def opencode_server(name: str, server: dict[str, Any]) -> dict[str, Any]:
    kind = server.get("type") or ("stdio" if "command" in server else "http")
    if kind in ("http", "sse"):
        # Remote servers default to OAuth, which a headless run cannot complete.
        translated: dict[str, Any] = {"type": "remote", "url": env_refs(name, server["url"]), "oauth": False}
        if headers := server.get("headers"):
            translated["headers"] = {k: env_refs(name, v) for k, v in headers.items()}
        return translated
    if kind == "stdio":
        command = [server["command"], *server.get("args", [])]
        translated = {"type": "local", "command": [env_refs(name, c) for c in command]}
        if env := server.get("env"):
            translated["environment"] = {k: env_refs(name, v) for k, v in env.items()}
        return translated
    raise SystemExit(f"MCP server {name!r} has type {kind!r}, which the opencode harness cannot translate")


def env_refs(server: str, value: str) -> str:
    """Rewrite ``${VAR}`` as opencode's ``{env:VAR}``, so the secret stays in the environment."""

    def replace(match: re.Match[str]) -> str:
        if match.group(2):
            raise SystemExit(
                f"MCP server {server!r} uses a default (${{{match.group(1)}{match.group(2)}}}), "
                "which opencode cannot express; set the variable instead"
            )
        return f"{{env:{match.group(1)}}}"

    return _ENV_REF.sub(replace, value)
