from __future__ import annotations

from g1_rickshaw_lab.training_contract import (
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_PARAMETERS,
    ROLLOUT_FORMAL_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_PHYSICS_PARAMETER_NAMES,
    ROLLOUT_STAGE_SEQUENCE,
    SIGNED_SLOPE_LABELS,
    S0FixedSeedValidationState,
    s2_remaining_learning_iterations,
    validate_s1_training_completion,
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


def test_all_training_stages_use_patience_five() -> None:
    assert GUIDE_MAX_ITERATIONS == {
        "s0_teacher": 6000,
        "s1_context_distillation": 4000,
        "s2_student_ppo": 2000,
    }
    for stage in GUIDE_TRAINING_PARAMETERS.values():
        assert stage["validation_patience"] == 5


def test_fixed_seed_validation_stops_after_five_misses() -> None:
    state = S0FixedSeedValidationState()
    digest = "a" * 64
    assert not state.record(
        iteration=200,
        stage="training",
        score=1.0,
        report_sha256=digest,
    )
    for iteration in (400, 600, 800, 1000):
        assert not state.record(
            iteration=iteration,
            stage="training",
            score=1.0,
            report_sha256=digest,
        )
    assert state.record(
        iteration=1200,
        stage="training",
        score=1.0,
        report_sha256=digest,
    )


def test_s2_terminal_patience_prevents_resume() -> None:
    assert s2_remaining_learning_iterations(
        requested_iterations=2000,
        completed_iterations=1200,
        early_stopped=True,
    ) == 0


def test_s1_terminal_patience_is_a_complete_training_run() -> None:
    validate_s1_training_completion(
        {
            "training": {
                "completed_iterations": 1200,
                "early_stopped": True,
                "validation_history": [
                    {
                        "iteration": 1200,
                        "no_improvement_count": 5,
                    }
                ],
            }
        }
    )


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
