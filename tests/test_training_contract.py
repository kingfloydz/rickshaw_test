from __future__ import annotations

import pytest

from g1_rickshaw_lab.training_contract import (
    GUIDE_MAX_ITERATIONS,
    ROLLOUT_DEFAULT_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_STAGE_SEQUENCE,
    SIGNED_SLOPE_LABELS,
    guide_max_iterations,
    s0_remaining_learning_iterations,
    s2_remaining_learning_iterations,
    validate_rollout_stage_coverage,
)


def test_mainline_has_fixed_stage_budgets_and_19_slopes() -> None:
    assert GUIDE_MAX_ITERATIONS == {
        "s0_teacher": 6000,
        "s1_context_distillation": 4000,
        "s2_student_ppo": 2000,
    }
    assert ROLLOUT_STAGE_SEQUENCE == ("TRAINING",)
    assert SIGNED_SLOPE_LABELS == tuple(
        f"{value / 100:+.2f}" for value in range(-8, 11)
    )
    assert guide_max_iterations("s0_teacher") == 6000
    with pytest.raises(ValueError, match="unknown training stage"):
        guide_max_iterations("legacy")


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
    num_steps = 2
    base, extra = divmod(num_envs, len(SIGNED_SLOPE_LABELS))
    environments = {
        label: base + (index < extra)
        for index, label in enumerate(SIGNED_SLOPE_LABELS)
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
