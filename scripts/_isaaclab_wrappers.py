"""Shared helpers for project command-line scripts."""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
DEFAULT_ISAACLAB_ROOT = REPOSITORY_ROOT.parent / "IsaacLab"


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


def run_isaaclab_rsl_rl(script_name: str, argv: list[str]) -> None:
    """Run Isaac Lab's RSL-RL train/play script with this task registered."""

    root = isaaclab_root()
    script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / script_name
    require_existing_file(script, f"Isaac Lab RSL-RL {script_name}")
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    add_isaaclab_sources_to_path()
    add_project_source_to_path()
    source = script.read_text(encoding="utf-8")
    placeholder = "# PLACEHOLDER: Extension template (do not remove this comment)"
    extension_import = (
        "import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401\n"
        "from g1_rickshaw_lab.training_contract import install_runner_hooks_from_environment\n"
        "install_runner_hooks_from_environment()\n"
        + placeholder
    )
    if placeholder not in source:
        raise RuntimeError(f"Isaac Lab script has no extension placeholder: {script}")
    source = source.replace(placeholder, extension_import, 1)
    if script_name == "play.py" and (video_dir := os.environ.get("G1_RICKSHAW_VIDEO_DIR")):
        video_marker = '"video_folder": os.path.join(log_dir, "videos", "play"),'
        if video_marker not in source:
            raise RuntimeError(f"Isaac Lab play script has no video-folder marker: {script}")
        source = source.replace(
            video_marker,
            f'"video_folder": {os.fspath(Path(video_dir).resolve())!r},',
            1,
        )
    if script_name == "play.py" and os.environ.get("G1_RICKSHAW_EXPORT_ONLY") == "1":
        loop_marker = "    dt = env.unwrapped.step_dt"
        if loop_marker not in source:
            raise RuntimeError(f"Isaac Lab play script has no export-loop marker: {script}")
        source = source.replace(
            loop_marker,
            "    env.close()\n    return\n\n" + loop_marker,
            1,
        )
    previous = sys.argv
    try:
        sys.argv = [os.fspath(script), *argv]
        globals_dict = {"__name__": "__main__", "__file__": os.fspath(script)}
        exec(compile(source, os.fspath(script), "exec"), globals_dict)
    finally:
        sys.argv = previous
