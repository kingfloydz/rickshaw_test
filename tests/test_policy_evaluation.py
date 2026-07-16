from __future__ import annotations

from g1_rickshaw_lab.policy_evaluation import (
    CROSS_CASE_LABELS,
    FINAL_ACCEPTANCE_STAGE_THRESHOLDS,
    FORMAL_EVALUATION_NUM_ENVS_MULTIPLE,
    SIGNED_SLOPES,
    Threshold,
    command_phase_labels,
    evaluate_s2_return_floor,
    slope_label,
    validate_final_acceptance_thresholds,
)


def test_evaluation_uses_19_slopes_and_one_training_distribution() -> None:
    assert SIGNED_SLOPES == tuple(value / 100 for value in range(-8, 11))
    assert CROSS_CASE_LABELS == ("RANDOM",)
    assert FORMAL_EVALUATION_NUM_ENVS_MULTIPLE == 19
    assert slope_label(-0.08) == "-0.08"
    assert slope_label(0.08) == "+0.08"
    assert slope_label(0.10) == "+0.10"


def test_training_threshold_authority_is_complete() -> None:
    thresholds = {
        f"stages.training.{suffix}": Threshold(operator, 0.1)
        for suffix, operator in FINAL_ACCEPTANCE_STAGE_THRESHOLDS.items()
    }
    validate_final_acceptance_thresholds(thresholds, curriculum_stages=("training",))


def test_s2_return_floor_compares_training_only() -> None:
    stages = {
        "training": {
            "context_interventions": {"baseline_return": {"mean": 2.0}}
        }
    }
    comparisons, failures = evaluate_s2_return_floor(stages, {"training": 1.5})
    assert comparisons["training"]["passed"] is True
    assert failures == []


def test_command_phase_labels_remain_deterministic() -> None:
    assert command_phase_labels([0.0, 0.5, 0.5], [0.0, 0.1, -0.1]) == [
        "standing",
        "accelerating",
        "decelerating",
    ]
