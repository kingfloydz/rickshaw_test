"""CPU regressions for startup randomization, commands, and safety."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actuation import (
    actuator_effort_limits,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    CommandState,
    DOMAIN_PARAMETER_NAMES,
    DomainRandomizationCfg,
    SpeedCommandSamplingCfg,
    advance_speed_command_resampling,
    initialize_domain_randomization,
    sample_domain_parameters,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.terminations import (
    IMMEDIATE_CAUSES,
    PERSISTENT_CAUSES,
    PersistentSafetyCfg,
    PersistentTerminationState,
    TERMINATION_CAUSES,
    TerminationCauseState,
    finite_tensor_violation,
    persistent_condition_matrix,
)


def _domain_cfg(*, enabled: bool = True) -> DomainRandomizationCfg:
    ranges = {
        "torso.mass_delta": (-1.0, 3.0),
        "payload.mass": (-3.0, 3.0),
        "payload.com.x": (0.3, 0.9),
        "payload.com.y": (-0.15, 0.15),
        "payload.com.z": (0.45, 0.95),
        "rolling_resistance.c_rr": (0.01, 0.03),
        "terrain.friction": (0.6, 1.1),
        "wheel.left_damping": (0.015, 0.025),
        "wheel.right_damping": (0.015, 0.025),
    }
    nominal = {
        "torso.mass_delta": 0.0,
        "payload.mass": 0.0,
        "payload.com.x": 0.6,
        "payload.com.y": 0.0,
        "payload.com.z": 0.7,
        "rolling_resistance.c_rr": 0.02,
        "terrain.friction": 1.0,
        "wheel.left_damping": 0.02,
        "wheel.right_damping": 0.02,
    }
    return DomainRandomizationCfg(
        enabled=enabled, ranges=ranges, nominal=nominal, calibration={}
    )


def test_startup_event_signature_accepts_event_manager_env_ids() -> None:
    parameters = tuple(inspect.signature(initialize_domain_randomization).parameters)
    assert parameters[:3] == ("env", "env_ids", "cfg")


def test_domain_schema_contains_only_startup_randomized_physics() -> None:
    cfg = _domain_cfg()
    values = sample_domain_parameters(
        cfg, 8192, generator=torch.Generator().manual_seed(7)
    )

    assert set(values) == set(DOMAIN_PARAMETER_NAMES)
    assert len(values) == 9
    assert torch.min(values["terrain.friction"]) >= 0.6
    assert torch.max(values["terrain.friction"]) <= 1.1
    assert torch.min(values["torso.mass_delta"]) >= -1.0
    assert torch.max(values["torso.mass_delta"]) <= 3.0
    assert torch.min(values["payload.mass"]) >= -3.0
    assert torch.max(values["payload.mass"]) <= 3.0


def test_disabled_domain_uses_nominal_values() -> None:
    cfg = _domain_cfg(enabled=False)
    first = sample_domain_parameters(cfg, 3)
    second = sample_domain_parameters(cfg, 3)

    for name in first:
        torch.testing.assert_close(first[name], second[name])
        assert torch.all(first[name] == cfg.nominal[name])


def test_actuator_effort_limits_never_uses_permissive_physx_limit() -> None:
    robot = SimpleNamespace(
        num_joints=2,
        data=SimpleNamespace(joint_effort_limits=torch.full((2, 2), 1.0e9)),
        actuators={
            "motor": SimpleNamespace(
                joint_indices=slice(0, 2),
                effort_limit=torch.tensor([[50.0, 25.0], [45.0, 22.5]]),
            )
        },
    )
    limits = actuator_effort_limits(robot, torch.tensor([1, 0]))
    assert torch.equal(limits, torch.tensor([[25.0, 50.0], [22.5, 45.0]]))
    assert torch.all(limits < robot.data.joint_effort_limits[:, [1, 0]])


def test_speed_command_resamples_on_the_ten_second_timer() -> None:
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        command_state=CommandState.zeros(2),
    )
    env.command_state.resampling_elapsed_s[:] = torch.tensor([9.98, 9.96])
    cfg = SpeedCommandSamplingCfg(standing_fraction=0.0)
    assert advance_speed_command_resampling(env, cfg).tolist() == [0]
    assert torch.allclose(
        env.command_state.resampling_elapsed_s, torch.tensor([0.0, 9.98])
    )
    assert advance_speed_command_resampling(env, cfg).tolist() == [1]


def test_arm_hardware_limit_is_a_strict_ten_step_persistent_gate() -> None:
    zeros = torch.zeros(2)
    cfg = PersistentSafetyCfg(
        torso_tilt_max=0.5,
        hitch_height_bounds=(0.65, 0.85),
        rickshaw_pitch_bounds=(0.15, 0.45),
        lateral_corridor=0.3,
        heading_envelope=0.3,
        overspeed_margin=0.25,
        arm_torque_limit=0.9,
    )
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
        cfg,
    )
    arm_index = PERSISTENT_CAUSES.index("arm_torque")
    assert violations[:, arm_index].tolist() == [False, True]
    state = PersistentTerminationState.zeros(2)
    for _ in range(9):
        assert not torch.any(state.update(violations))
    assert state.update(violations).tolist() == [False, True]


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
