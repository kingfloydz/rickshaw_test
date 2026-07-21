"""CPU regressions for the Mjlab safety contract."""

from __future__ import annotations

import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.terminations import (
    IMMEDIATE_CAUSES,
    PERSISTENT_CAUSES,
    TERMINATION_CAUSES,
    PersistentSafetyCfg,
    PersistentTerminationState,
    TerminationCauseState,
    connection_safety_violation,
    finite_tensor_violation,
    persistent_condition_matrix,
)


def _persistent_cfg() -> PersistentSafetyCfg:
    return PersistentSafetyCfg(
        torso_tilt_max=0.5,
        hitch_height_bounds=(0.65, 0.85),
        rickshaw_pitch_bounds=(0.15, 0.45),
        lateral_corridor=0.3,
        heading_envelope=0.3,
        overspeed_margin=0.25,
        arm_torque_limit=0.9,
    )


def test_arm_hardware_limit_is_a_strict_ten_step_persistent_gate() -> None:
    zeros = torch.zeros(2)
    violations = persistent_condition_matrix(
        torch.full((2,), 0.7),
        zeros,
        torch.full((2,), 0.75),
        torch.full((2,), 0.3),
        zeros,
        zeros,
        zeros,
        zeros,
        torch.tensor([[0.9], [0.9001]]),
        torch.full((2,), 0.02),
        torch.ones(2, dtype=torch.bool),
        _persistent_cfg(),
    )
    arm_index = PERSISTENT_CAUSES.index("arm_torque")
    assert violations[:, arm_index].tolist() == [False, True]

    state = PersistentTerminationState.zeros(2)
    for _ in range(9):
        assert not torch.any(state.update(violations))
    assert state.update(violations).tolist() == [False, True]

    state.update(torch.zeros_like(violations))
    assert torch.count_nonzero(state.counters) == 0


def test_connection_limits_are_strict_and_check_absolute_impulse() -> None:
    residual = torch.tensor([[0.06, 0.0], [0.0601, 0.0], [0.0, 0.0]])
    impulse = torch.tensor([[1.7, 0.0], [0.0, 0.0], [-1.7001, 0.0]])
    assert connection_safety_violation(
        residual,
        impulse,
        residual_limit=0.06,
        impulse_limit=1.7,
    ).tolist() == [False, True, True]


def test_termination_histogram_and_non_finite_detection() -> None:
    state = TerminationCauseState.zeros(2)
    state.begin_policy_step()
    causes = torch.zeros(2, len(IMMEDIATE_CAUSES), dtype=torch.bool)
    causes[0, 0] = True
    causes[1, 2] = True
    state.record(IMMEDIATE_CAUSES, causes)

    histogram = state.histogram()
    assert set(histogram) == set(TERMINATION_CAUSES)
    assert histogram["non_finite"] == 1
    assert histogram[IMMEDIATE_CAUSES[2]] == 1

    value = torch.zeros(2, 3)
    value[1, 0] = torch.nan
    assert finite_tensor_violation(value).tolist() == [False, True]
