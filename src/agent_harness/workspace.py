"""Stage a trial's workspace and pick out what the agent delivered."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .config import PLOT_SUFFIXES, SCRIPT_SUFFIXES, SKILLS_DIR


def hash_skills(skills_dir: Path) -> str:
    """Content hash over every skill file, so a run pins the exact skill revision."""
    digest = hashlib.sha256()
    for path in sorted(p for p in skills_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(skills_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def stage_workspace(root: Path, skills_dir: str) -> Path:
    """Create an empty workspace with ``skills/`` copied in as project skills, at
    ``skills_dir`` (relative to the workspace) where the harness looks for them.
    """
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    shutil.copytree(SKILLS_DIR, workspace / skills_dir)
    return workspace


def find_deliverables(outputs: Path) -> dict[str, Path]:
    """Pick the final script and plot out of everything the agent wrote.

    "Final" is the most recently modified file of each kind, which is what the
    agent lands on after any iteration.
    """

    def newest(suffixes: tuple[str, ...]) -> Path | None:
        candidates = [
            path
            for path in outputs.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        return max(candidates, key=lambda path: path.stat().st_mtime, default=None)

    found = {"script": newest(SCRIPT_SUFFIXES), "plot": newest(PLOT_SUFFIXES)}
    return {kind: path for kind, path in found.items() if path is not None}
