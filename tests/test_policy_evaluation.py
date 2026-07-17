"""CPU-only tests for policy diagnostic aggregation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from g1_rickshaw_lab.policy_evaluation import (
    CROSS_CASE_LABELS,
    SIGNED_SLOPES,
    MetricStore,
    PolicyEvaluationAccumulator,
    command_phase_labels,
    d6_wrench_channels,
    evaluate_s2_return_floor,
    slope_label,
)


def test_evaluation_uses_all_19_signed_slopes() -> None:
    assert SIGNED_SLOPES == tuple(value / 100 for value in range(-8, 11))
    assert CROSS_CASE_LABELS == ("RANDOM",)
    assert slope_label(-0.08) == "-0.08"
    assert slope_label(0.08) == "+0.08"
    assert slope_label(0.10) == "+0.10"


def test_s2_return_comparison_is_diagnostic_only() -> None:
    comparisons = evaluate_s2_return_floor(
        {"training": {"return": {"mean": 1.25}}},
        {"training": 1.5},
    )
    assert comparisons == {
        "training": {
            "s1_baseline_mean": 1.5,
            "s2_baseline_mean": 1.25,
            "delta": -0.25,
            "meets_or_exceeds_s1": False,
        }
    }


def test_command_phase_labels_are_deterministic() -> None:
    assert command_phase_labels([0.0, 0.5, 0.5], [0.0, 0.1, -0.1]) == [
        "standing",
        "accelerating",
        "decelerating",
    ]


def test_d6_wrench_channels_report_force_torque_and_asymmetry() -> None:
    wrench = torch.tensor(
        [
            [
                [3.0, 4.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 5.0, 0.0, 3.0, 4.0],
            ]
        ]
    )
    channels = d6_wrench_channels(wrench)
    torch.testing.assert_close(channels["force"], torch.tensor([5.0]))
    torch.testing.assert_close(channels["torque"], torch.tensor([5.0]))
    torch.testing.assert_close(channels["force_asymmetry"], torch.tensor([0.0]))
    torch.testing.assert_close(channels["torque_asymmetry"], torch.tensor([3.0 / 7.0]))


def test_metric_store_excludes_nonfinite_samples_but_records_them() -> None:
    store = MetricStore()
    store.add_samples(
        {
            "speed_error": np.asarray([1.0, np.nan, -1.0]),
            "overspeed": np.asarray([0.0, 1.0, 0.0]),
        }
    )
    store.add_episode(2.0, fell=False, causes=("timeout",))
    summary = store.summary()
    assert summary["samples"] == 2
    assert summary["non_finite_sample_counts"] == {
        "overspeed": 0,
        "speed_error": 1,
    }
    assert summary["tracking"]["speed_rmse_mps"] == pytest.approx(1.0)
    assert summary["episodes"]["return"]["mean"] == pytest.approx(2.0)


def test_accumulator_keeps_global_and_per_slope_diagnostics() -> None:
    accumulator = PolicyEvaluationAccumulator()
    accumulator.add_step(
        {"speed_error": [0.1, -0.2], "overspeed": [0.0, 1.0]},
        [0, len(SIGNED_SLOPES) - 1],
        stage_labels=["training", "training"],
        cross_case_labels=["RANDOM", "RANDOM"],
        phase_labels=["accelerating", "decelerating"],
    )
    accumulator.add_episode(
        0,
        3.0,
        fell=False,
        causes=("timeout",),
        phase_labels=("accelerating",),
        cross_case_label="RANDOM",
    )
    global_summary, per_slope = accumulator.summary()
    assert global_summary["samples"] == 2
    assert per_slope[slope_label(SIGNED_SLOPES[0])]["samples"] == 1
    assert per_slope[slope_label(SIGNED_SLOPES[-1])]["samples"] == 1
    assert accumulator.stratified_summary()["by_cross_case"]["RANDOM"]["samples"] == 2


def test_accumulator_rejects_invalid_slope_index() -> None:
    accumulator = PolicyEvaluationAccumulator()
    with pytest.raises(ValueError, match="slope_indices"):
        accumulator.add_step({"speed_error": [0.0]}, [len(SIGNED_SLOPES)])
