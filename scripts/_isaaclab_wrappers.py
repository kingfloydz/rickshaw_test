"""Shared helpers for project command-line scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from g1_rickshaw_lab.rl.runner import RunnerContext
    from g1_rickshaw_lab.workflows.rsl_rl import PlayOptions


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
DEFAULT_ISAACLAB_ROOT = REPOSITORY_ROOT.parent / "IsaacLab"
FOLLOW_CAMERA_EYE = (0.0, 4.2, 1.4)
FOLLOW_CAMERA_LOOKAT = (0.0, 0.0, 0.85)


def add_project_source_to_path() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


def add_isaaclab_sources_to_path() -> None:
    """Make a sibling Isaac Lab source checkout importable without stale editable installs."""

    root = isaaclab_root()
    source_dirs = (
        root / "source" / "isaaclab",
        root / "source" / "isaaclab_assets",
        root / "source" / "isaaclab_rl",
        root / "source" / "isaaclab_tasks",
    )
    for directory in reversed(source_dirs):
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def isaaclab_root() -> Path:
    return Path(os.environ.get("ISAACLAB_PATH", os.fspath(DEFAULT_ISAACLAB_ROOT)))


def require_existing_file(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def run_isaaclab_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    runner_context: RunnerContext,
    play_options: PlayOptions | None = None,
) -> None:
    """Run the project-owned RSL-RL launcher against the configured Isaac Lab."""

    add_isaaclab_sources_to_path()
    add_project_source_to_path()
    from g1_rickshaw_lab.workflows.rsl_rl import run_rsl_rl

    run_rsl_rl(
        mode,
        argv,
        runner_context=runner_context,
        play_options=play_options,
    )
