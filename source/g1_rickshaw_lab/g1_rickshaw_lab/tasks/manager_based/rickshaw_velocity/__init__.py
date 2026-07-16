"""Gym registrations for G1 rickshaw velocity tracking.

The MDP numerical kernels are deliberately importable without Isaac Sim. Task
registration therefore happens only when Gymnasium and Isaac Lab are present.
"""

from importlib.util import find_spec


TRAIN_TASK_ID = "Isaac-G1-Rickshaw-Directional-Slope-v0"
PLAY_TASK_ID = "Isaac-G1-Rickshaw-Directional-Slope-Play-v0"


if find_spec("gymnasium") is not None and find_spec("isaaclab") is not None:
    import gymnasium as gym

    from .env_cfg import G1RickshawDirectionalSlopeEnvCfg, G1RickshawDirectionalSlopePlayEnvCfg

    def _register(
        task_id: str,
        env_cfg_entry_point: str,
        runner_cfg_entry_point: str,
        *,
        student_runner_cfg_entry_point: str | None = None,
    ) -> None:
        if task_id in gym.registry:
            return
        kwargs = {
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": runner_cfg_entry_point,
        }
        if student_runner_cfg_entry_point is not None:
            kwargs["rsl_rl_student_cfg_entry_point"] = student_runner_cfg_entry_point
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs=kwargs,
        )

    _register(
        TRAIN_TASK_ID,
        "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity:G1RickshawDirectionalSlopeEnvCfg",
        "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents:G1RickshawTeacherPPORunnerCfg",
        student_runner_cfg_entry_point=(
            "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents:"
            "G1RickshawStudentPPORunnerCfg"
        ),
    )
    _register(
        PLAY_TASK_ID,
        "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity:G1RickshawDirectionalSlopePlayEnvCfg",
        "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.agents:G1RickshawStudentPPORunnerCfg",
    )

    __all__ = [
        "G1RickshawDirectionalSlopeEnvCfg",
        "G1RickshawDirectionalSlopePlayEnvCfg",
        "PLAY_TASK_ID",
        "TRAIN_TASK_ID",
    ]
else:
    __all__ = ["PLAY_TASK_ID", "TRAIN_TASK_ID"]
