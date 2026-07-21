"""Shared Mjlab helpers for project command-line workflows."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from g1_rickshaw_lab.rl.runner import RunnerContext
    from g1_rickshaw_lab.workflows.rsl_rl import PlayOptions

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
FOLLOW_CAMERA_EYE = (0.0, 4.2, 1.4)
FOLLOW_CAMERA_LOOKAT = (0.0, 0.0, 0.85)


def add_project_source_to_path() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


def add_mjlab_sources_to_path() -> None:
    add_project_source_to_path()


def require_existing_file(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def configure_history_length(env_cfg: Any, history_length: int) -> None:
    """Update every Mjlab term that owns a temporal history contract."""

    runtime = env_cfg.events["initialize_task"].params["cfg"]
    runtime = replace(runtime, history_length=history_length)
    for event_name in ("initialize_task", "initialize_domain", "policy_state"):
        env_cfg.events[event_name].params["cfg"] = runtime
    env_cfg.policy_update = runtime
    for group_name in ("history", "teacher_dynamic_history"):
        env_cfg.observations[group_name].terms["history"].params[
            "history_length"
        ] = history_length
    env_cfg.history_length = history_length


def load_mjlab_configs(
    task: str,
    *,
    play: bool,
    num_envs: int,
    seed: int,
    history_length: int,
) -> tuple[Any, Any]:
    """Load a registered Mjlab environment/runner pair with shared dimensions."""

    import g1_rickshaw_lab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(task, play=play)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    configure_history_length(env_cfg, history_length)
    agent_cfg = load_rl_cfg(task)
    agent_cfg.seed = seed
    agent_cfg.actor.history_length = history_length
    return env_cfg, agent_cfg


def run_mjlab_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    runner_context: RunnerContext,
    play_options: PlayOptions | None = None,
) -> None:
    add_project_source_to_path()
    from g1_rickshaw_lab.workflows.rsl_rl import run_rsl_rl

    run_rsl_rl(mode, argv, runner_context=runner_context, play_options=play_options)


__all__ = [
    "FOLLOW_CAMERA_EYE",
    "FOLLOW_CAMERA_LOOKAT",
    "REPOSITORY_ROOT",
    "add_mjlab_sources_to_path",
    "add_project_source_to_path",
    "configure_history_length",
    "load_mjlab_configs",
    "require_existing_file",
    "run_mjlab_rsl_rl",
]
