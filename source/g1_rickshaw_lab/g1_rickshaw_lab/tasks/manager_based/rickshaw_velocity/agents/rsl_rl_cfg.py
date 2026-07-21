"""Mjlab RSL-RL configurations for S0 teacher and S2 student PPO."""

from __future__ import annotations

from dataclasses import dataclass

from g1_rickshaw_lab.policy_schema import DEFAULT_CONTEXT_DIM, HISTORY_LENGTH
from g1_rickshaw_lab.training_contract import guide_max_iterations, training_artifact_interval


def _classes():
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    @dataclass
    class RickshawActorCfg(RslRlModelCfg):
        latent_dim: int = DEFAULT_CONTEXT_DIM
        history_length: int = HISTORY_LENGTH

    @dataclass
    class RickshawAlgorithmCfg(RslRlPpoAlgorithmCfg):
        context_learning_rate: float | None = None

    return RslRlOnPolicyRunnerCfg, RickshawActorCfg, RslRlModelCfg, RickshawAlgorithmCfg


def _runner_cfg(*, student: bool, latent_dim: int, history_length: int, rollout_steps: int):
    RunnerCfg, ActorCfg, ModelCfg, AlgorithmCfg = _classes()
    stage = "s2_student_ppo" if student else "s0_teacher"
    return RunnerCfg(
        seed=42,
        num_steps_per_env=rollout_steps,
        max_iterations=guide_max_iterations(stage, rollout_steps),
        save_interval=training_artifact_interval(rollout_steps),
        experiment_name="g1_rickshaw_student" if student else "g1_rickshaw_teacher",
        run_name="s2" if student else "s0",
        clip_actions=1.0,
        obs_groups={
            "actor": ("policy", "history")
            if student
            else ("policy", "history", "teacher_dynamic_history", "teacher_static"),
            "critic": ("policy", "critic"),
        },
        actor=ActorCfg(
            class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawActorModel",
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            latent_dim=latent_dim,
            history_length=history_length,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.4,
                "std_type": "log",
            },
        ),
        critic=ModelCfg(
            class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawCriticModel",
            hidden_dims=(256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=AlgorithmCfg(
            class_name="g1_rickshaw_lab.rl.rsl_rl_models:RickshawPPO",
            context_learning_rate=1.0e-4 if student else None,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.97,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def g1_rickshaw_teacher_ppo_runner_cfg(
    *, latent_dim: int = DEFAULT_CONTEXT_DIM, history_length: int = HISTORY_LENGTH, rollout_steps: int = 48
):
    return _runner_cfg(
        student=False,
        latent_dim=latent_dim,
        history_length=history_length,
        rollout_steps=rollout_steps,
    )


def g1_rickshaw_student_ppo_runner_cfg(
    *, latent_dim: int = DEFAULT_CONTEXT_DIM, history_length: int = HISTORY_LENGTH, rollout_steps: int = 48
):
    return _runner_cfg(
        student=True,
        latent_dim=latent_dim,
        history_length=history_length,
        rollout_steps=rollout_steps,
    )


g1_rickshaw_ppo_runner_cfg = g1_rickshaw_teacher_ppo_runner_cfg
G1RickshawTeacherPPORunnerCfg = g1_rickshaw_teacher_ppo_runner_cfg
G1RickshawStudentPPORunnerCfg = g1_rickshaw_student_ppo_runner_cfg

__all__ = [
    "G1RickshawStudentPPORunnerCfg",
    "G1RickshawTeacherPPORunnerCfg",
    "g1_rickshaw_ppo_runner_cfg",
    "g1_rickshaw_student_ppo_runner_cfg",
    "g1_rickshaw_teacher_ppo_runner_cfg",
]
