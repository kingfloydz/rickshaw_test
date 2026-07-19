from __future__ import annotations

import pytest
import torch

from g1_rickshaw_lab.training_contract import (
    DISTILLATION_ROLLOUT_STEPS,
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_PARAMETERS,
    ROLLOUT_DEFAULT_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_STAGE_SEQUENCE,
    SIGNED_SLOPE_LABELS,
    guide_max_iterations,
    rollout_scaled_iterations,
    s0_remaining_learning_iterations,
    s2_remaining_learning_iterations,
    training_artifact_interval,
    validate_student_checkpoint_architecture,
    validate_teacher_checkpoint_architecture,
    validate_rollout_stage_coverage,
)


def test_mainline_has_fixed_stage_budgets_and_19_slopes() -> None:
    assert GUIDE_MAX_ITERATIONS == {
        "s0_teacher": 4000,
        "s1_context_distillation": 3000,
        "s2_student_ppo": 2000,
    }
    assert ROLLOUT_STAGE_SEQUENCE == ("TRAINING",)
    assert SIGNED_SLOPE_LABELS == tuple(
        f"{value / 100:+.2f}" for value in range(-8, 11)
    )
    assert guide_max_iterations("s0_teacher") == 4000
    assert GUIDE_TRAINING_PARAMETERS["s0_teacher"] == {
        "domain_randomization": "startup_fixed",
        "terrain_slopes": "startup_balanced_fixed",
        "observation_noise": "unitree_g1_uniform",
    }
    with pytest.raises(ValueError, match="unknown training stage"):
        guide_max_iterations("legacy")


@pytest.mark.parametrize(
    ("rollout_steps", "s0_iterations", "s2_iterations", "artifact_interval"),
    (
        (24, 8000, 4000, 400),
        (48, 4000, 2000, 200),
        (64, 3000, 1500, 150),
    ),
)
def test_rollout_variants_preserve_transition_and_artifact_budgets(
    rollout_steps: int,
    s0_iterations: int,
    s2_iterations: int,
    artifact_interval: int,
) -> None:
    assert guide_max_iterations("s0_teacher", rollout_steps) == s0_iterations
    assert guide_max_iterations("s2_student_ppo", rollout_steps) == s2_iterations
    assert rollout_scaled_iterations(4000, rollout_steps) == s0_iterations
    assert training_artifact_interval(rollout_steps) == artifact_interval
    assert s0_iterations * rollout_steps == 4000 * 48
    assert s2_iterations * rollout_steps == 2000 * 48
    assert artifact_interval * rollout_steps == 200 * 48


@pytest.mark.parametrize(
    ("remaining", "function"),
    (
        (400, s0_remaining_learning_iterations),
        (400, s2_remaining_learning_iterations),
    ),
)
def test_ppo_resume_uses_only_the_remaining_iteration_budget(
    remaining: int,
    function,
) -> None:
    assert function(requested_iterations=2000, completed_iterations=1600) == remaining
    with pytest.raises(ValueError, match="exceeds"):
        function(requested_iterations=2000, completed_iterations=2001)


def test_single_training_rollout_manifest_is_accepted() -> None:
    num_envs = ROLLOUT_DEFAULT_NUM_ENVS
    num_steps = DISTILLATION_ROLLOUT_STEPS
    base, extra = divmod(num_envs, len(SIGNED_SLOPE_LABELS))
    environments = {
        label: base + (index < extra) for index, label in enumerate(SIGNED_SLOPE_LABELS)
    }
    samples = {label: count * num_steps for label, count in environments.items()}
    episodes = dict(environments)
    total = num_envs * num_steps
    segment = {
        "global_stage": "TRAINING",
        "valid_samples": total,
        "target_valid_samples": total,
        "full_environment_reset": True,
        "reset_policy_steps": 0,
        "slope_environment_distribution": environments,
        "slope_sample_distribution": samples,
        "slope_episode_distribution": episodes,
        "per_environment_stage_distribution": {"TRAINING": num_envs},
        "valid_sample_stage_distribution": {"TRAINING": total},
    }
    manifest = {
        "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "num_envs": num_envs,
        "num_steps_per_stage": num_steps,
        "signed_slopes": [value / 100 for value in range(-8, 11)],
        "stage_segments": [segment],
        "stage_sample_distribution": {"TRAINING": total},
        "num_samples": total,
        "slope_environment_distribution": environments,
        "slope_sample_distribution": samples,
        "slope_episode_distribution": episodes,
    }
    assert validate_rollout_stage_coverage(manifest) == {"TRAINING": total}


def test_rollout_manifest_rejects_non_training_segments() -> None:
    with pytest.raises(ValueError, match="exactly one TRAINING segment"):
        validate_rollout_stage_coverage(
            {
                "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
                "stage_segments": [],
            }
        )


@pytest.mark.parametrize("latent_dim", (8, 16, 24, 32))
def test_checkpoint_tensor_widths_match_the_recorded_latent(latent_dim: int) -> None:
    configuration = {"training_parameters": {"latent_dim": latent_dim}}
    student = {
        "model_state_dict": {
            "context_encoder.context.weight": torch.zeros(latent_dim, 64),
            "actor.network.0.weight": torch.zeros(512, 96 + latent_dim),
        }
    }
    teacher = {
        "actor_state_dict": {
            "encoder.context.weight": torch.zeros(latent_dim, 96),
            "policy.network.0.weight": torch.zeros(512, 96 + latent_dim),
        }
    }
    validate_student_checkpoint_architecture(student, configuration)
    validate_teacher_checkpoint_architecture(teacher, configuration)

    student["model_state_dict"]["actor.network.0.weight"] = torch.zeros(512, 112)
    if latent_dim == 16:
        student["model_state_dict"]["actor.network.0.weight"] = torch.zeros(512, 111)
    with pytest.raises(ValueError, match="recorded latent width"):
        validate_student_checkpoint_architecture(student, configuration)
