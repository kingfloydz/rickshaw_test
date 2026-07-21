"""RSL-RL configuration using mjlab's native runner schema."""

from __future__ import annotations


def g1_rickshaw_ppo_runner_cfg():
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=48,
        max_iterations=10_000,
        save_interval=200,
        experiment_name="g1_rickshaw_velocity",
        run_name="mjlab",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.4,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
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


G1RickshawTeacherPPORunnerCfg = g1_rickshaw_ppo_runner_cfg
G1RickshawStudentPPORunnerCfg = g1_rickshaw_ppo_runner_cfg

__all__ = [
    "G1RickshawStudentPPORunnerCfg",
    "G1RickshawTeacherPPORunnerCfg",
    "g1_rickshaw_ppo_runner_cfg",
]
