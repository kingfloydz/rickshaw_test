"""Mjlab task registrations for the rickshaw training stages."""

from importlib.util import find_spec

from .env_cfg import (
    G1RickshawDirectionalSlopeEnvCfg,
    G1RickshawDirectionalSlopePlayEnvCfg,
    g1_rickshaw_env_cfg,
)

TRAIN_TASK_ID = "Mjlab-G1-Rickshaw-Directional-Slope-Teacher"
STUDENT_TASK_ID = "Mjlab-G1-Rickshaw-Directional-Slope-Student"
HISTORY_91_TEACHER_TASK_ID = TRAIN_TASK_ID + "-H91"
HISTORY_91_STUDENT_TASK_ID = STUDENT_TASK_ID + "-H91"
PLAY_TASK_ID = STUDENT_TASK_ID

if find_spec("mjlab") is not None:
    from mjlab.tasks.registry import register_mjlab_task

    from .agents.rsl_rl_cfg import (
        g1_rickshaw_student_ppo_runner_cfg,
        g1_rickshaw_teacher_ppo_runner_cfg,
    )

    for task_id, student, history_length in (
        (TRAIN_TASK_ID, False, 61),
        (STUDENT_TASK_ID, True, 61),
        (HISTORY_91_TEACHER_TASK_ID, False, 91),
        (HISTORY_91_STUDENT_TASK_ID, True, 91),
    ):
        runner_cfg = (
            g1_rickshaw_student_ppo_runner_cfg(history_length=history_length)
            if student
            else g1_rickshaw_teacher_ppo_runner_cfg(history_length=history_length)
        )
        register_mjlab_task(
            task_id=task_id,
            env_cfg=g1_rickshaw_env_cfg(play=False, history_length=history_length),
            play_env_cfg=g1_rickshaw_env_cfg(play=True, history_length=history_length),
            rl_cfg=runner_cfg,
        )

__all__ = [
    "G1RickshawDirectionalSlopeEnvCfg",
    "G1RickshawDirectionalSlopePlayEnvCfg",
    "HISTORY_91_STUDENT_TASK_ID",
    "HISTORY_91_TEACHER_TASK_ID",
    "PLAY_TASK_ID",
    "STUDENT_TASK_ID",
    "TRAIN_TASK_ID",
    "g1_rickshaw_env_cfg",
]
