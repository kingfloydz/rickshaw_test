"""mjlab task registration."""

from importlib.util import find_spec

TRAIN_TASK_ID = "Unitree-G1-Rickshaw-Flat"
PLAY_TASK_ID = TRAIN_TASK_ID

from .env_cfg import (
    G1RickshawDirectionalSlopeEnvCfg,
    G1RickshawDirectionalSlopePlayEnvCfg,
    g1_rickshaw_env_cfg,
)

if find_spec("mjlab") is not None:
    from mjlab.tasks.registry import register_mjlab_task

    from .agents.rsl_rl_cfg import g1_rickshaw_ppo_runner_cfg

    register_mjlab_task(
        task_id=TRAIN_TASK_ID,
        env_cfg=g1_rickshaw_env_cfg(play=False),
        play_env_cfg=g1_rickshaw_env_cfg(play=True),
        rl_cfg=g1_rickshaw_ppo_runner_cfg(),
    )

__all__ = [
    "G1RickshawDirectionalSlopeEnvCfg",
    "G1RickshawDirectionalSlopePlayEnvCfg",
    "PLAY_TASK_ID",
    "TRAIN_TASK_ID",
    "g1_rickshaw_env_cfg",
]
