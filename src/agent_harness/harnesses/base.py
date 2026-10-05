"""What every agent harness provides, and the subprocess loop they share."""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def parse_event(line: str) -> dict[str, Any] | None:
    """One JSON-lines event; ``None`` for a blank or non-JSON line."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def descendants(pid: int) -> list[int]:
    """Every process below ``pid``, by parent PID - including ones that moved to a
    session of their own, as opencode's private server does.
    """
    listing = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=False
    ).stdout
    children: defaultdict[int, list[int]] = defaultdict(list)
    for line in listing.splitlines():
        child, parent = map(int, line.split())
        children[parent].append(child)
    found, stack = [], [pid]
    while stack:
        for child in children[stack.pop()]:
            found.append(child)
            stack.append(child)
    return found


def kill_tree(pid: int) -> None:
    """Kill a process and everything it started. The tree is read before anything is
    killed, since an orphaned child is reparented and can no longer be traced to it.
    """
    for target in [pid, *descendants(pid)]:
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass


def agent_env() -> dict[str, str]:
    """The harness's environment, minus MLflow's, which belongs to the harness alone."""
    return {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}


def read_events(path: Path) -> list[dict[str, Any]]:
    return [e for line in path.read_text().splitlines() if (e := parse_event(line)) is not None]


@dataclass
class ToolCall:
    name: str
    input: Any
    output: Any
    is_error: bool = False


@dataclass
class RunSummary:
    """One run of the agent, in the same terms whichever harness ran it.

    A value the harness does not report is ``None`` rather than 0, so it is left
    out of the metrics instead of reading as a measurement.
    """

    calls: list[ToolCall]
    result: dict[str, Any] = field(default_factory=dict)  # the harness's own final record
    final_text: str = ""
    session_id: str = ""
    is_error: bool = False
    failure_reason: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    num_turns: int = 0
    duration_ms: float | None = None
    api_duration_ms: float | None = None
    skill_calls: Counter[str] = field(default_factory=Counter)
    num_skill_file_reads: int = 0
    num_mcp_calls: int = 0

    @property
    def num_tool_errors(self) -> int:
        return sum(call.is_error for call in self.calls)


class Harness(ABC):
    """An agentic coding CLI that a trial runs headless in a staged workspace."""

    name: str
    default_bin: str
    # The raw event stream of each run, written by run-trial and read back by grade-trial.
    stream_file: str
    # Where the skills are staged inside the workspace, so the harness finds them natively.
    skills_dir: str

    def __init__(self, args: argparse.Namespace | None = None):
        self.args = args
        self.bin = getattr(args, "harness_bin", None) or self.default_bin

    def check_args(self) -> None:
        """Fail before launch on an option this harness cannot honor, or a CLI that
        does not run - otherwise every question would fail the same way, at no cost
        and with no events to say why.
        """
        if self.version is None:
            raise SystemExit(
                f"{self.bin_path()} --version failed; is the {self.name} CLI installed? "
                "Point --harness-bin at it otherwise."
            )

    def bin_path(self) -> str:
        """The CLI as it resolves on PATH, for messages."""
        return shutil.which(self.bin) or self.bin

    @functools.cached_property
    def version(self) -> str | None:
        """What the CLI reports for ``--version``; ``None`` when it does not run."""
        try:
            proc = subprocess.run(
                [self.bin, "--version"], capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return (proc.returncode == 0 and proc.stdout.strip()) or None

    @abstractmethod
    def command(self) -> list[str]:
        """The CLI invocation; the prompt is written to its stdin."""

    def env(self) -> dict[str, str]:
        """Environment to add for the agent's process."""
        return {}

    def params(self) -> dict[str, Any]:
        """Harness-specific settings to log as run params."""
        return {}

    def config_artifacts(self) -> dict[str, str]:
        """Configs the harness generated for the agent, logged on the trial run."""
        return {}

    def after_run(self, events: list[dict[str, Any]], run_dir: Path, workspace: Path) -> list[Path]:
        """Collect anything the stream leaves out; return the files written to ``run_dir``."""
        return []

    @abstractmethod
    def tool_calls(self, events: list[dict[str, Any]]) -> list[ToolCall]:
        """The run's tool calls in order, each with its result."""

    @abstractmethod
    def summarize(self, events: list[dict[str, Any]], run_dir: Path) -> RunSummary: ...

    @abstractmethod
    def log_event(self, event: dict[str, Any]) -> None:
        """Minimal console trace so a long trial is watchable."""

    def run(
        self, prompt_text: str, workspace: Path, stream_path: Path, timeout: int
    ) -> tuple[list[dict[str, Any]], int, str, bool]:
        """Run the agent, tee its event stream to disk, and return the parsed events,
        exit code, stderr and whether it timed out.

        On a timeout, or when the trial itself is interrupted, the agent is killed
        along with everything it started (shells, a private server), so nothing is
        left running. The agent has a session of its own, so Ctrl-C reaches only the
        harness, which then does that cleanup. The stream is read on its own thread,
        so a straggler still holding the pipe cannot hold up the trial.
        """
        events: list[dict[str, Any]] = []
        # opencode takes its working directory from PWD rather than the process's
        # own, so a PWD inherited from the harness would put the agent in the repo.
        env = agent_env() | self.env() | {"PWD": str(workspace)}
        timed_out = False

        with stream_path.open("w") as stream_file, tempfile.TemporaryFile("w+") as stderr_file:
            proc = subprocess.Popen(
                self.command(),
                cwd=workspace,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            assert proc.stdin is not None and proc.stdout is not None
            stdout = proc.stdout

            def pump() -> None:
                for line in stdout:
                    stream_file.write(line)
                    stream_file.flush()
                    if (event := parse_event(line)) is not None:
                        events.append(event)
                        self.log_event(event)

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            try:
                proc.stdin.write(prompt_text)
                proc.stdin.close()
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(proc.pid)
                proc.wait()
            except BaseException:
                kill_tree(proc.pid)
                raise
            reader.join(timeout=30)
            stderr_file.seek(0)
            stderr = stderr_file.read()

        return list(events), proc.returncode, stderr, timed_out
