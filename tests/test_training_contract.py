from __future__ import annotations

from g1_rickshaw_lab.training_contract import (
    ROLLOUT_FORMAL_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_PHYSICS_PARAMETER_NAMES,
    ROLLOUT_STAGE_SEQUENCE,
    SIGNED_SLOPE_LABELS,
    S0FixedSeedValidationState,
    validate_rollout_stage_coverage,
)


def _physics_distribution() -> dict[str, dict[str, float]]:
    return {
        name: {"minimum": 1.0, "mean": 1.0, "maximum": 1.0}
        for name in ROLLOUT_PHYSICS_PARAMETER_NAMES
    }


def test_training_contract_has_one_stage_and_19_slopes() -> None:
    assert ROLLOUT_STAGE_SEQUENCE == ("TRAINING",)
    assert SIGNED_SLOPE_LABELS == tuple(
        f"{value / 100:+.2f}" for value in range(-8, 11)
    )


def test_s0_validation_runs_every_200_iterations() -> None:
    state = S0FixedSeedValidationState()
    assert not state.should_evaluate(199)
    assert state.should_evaluate(200)
    assert not state.should_evaluate(201)


def test_single_training_rollout_manifest_is_accepted() -> None:
    num_steps = 2
    base, extra = divmod(ROLLOUT_FORMAL_NUM_ENVS, len(SIGNED_SLOPE_LABELS))
    environments = {
        label: base + (index < extra)
        for index, label in enumerate(SIGNED_SLOPE_LABELS)
    }
    samples = {label: count * num_steps for label, count in environments.items()}
    episodes = dict(environments)
    total = ROLLOUT_FORMAL_NUM_ENVS * num_steps
    segment = {
        "global_stage": "TRAINING",
        "valid_samples": total,
        "target_valid_samples": total,
        "full_environment_reset": True,
        "reset_policy_steps": 0,
        "slope_environment_distribution": environments,
        "slope_sample_distribution": samples,
        "slope_episode_distribution": episodes,
        "physics_distribution": _physics_distribution(),
        "per_environment_stage_distribution": {"TRAINING": ROLLOUT_FORMAL_NUM_ENVS},
        "valid_sample_stage_distribution": {"TRAINING": total},
    }
    manifest = {
        "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "num_envs": ROLLOUT_FORMAL_NUM_ENVS,
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
