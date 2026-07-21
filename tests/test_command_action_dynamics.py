"""Pure-Torch tests for the active Mjlab command and dynamics kernels."""

from __future__ import annotations

import math

import pytest
import torch

from g1_rickshaw_lab.g1_motor_defaults import (
    G1_ACTION_SCALE,
    G1_JOINT_EFFORT_LIMITS,
    G1_JOINT_STIFFNESS,
)
from g1_rickshaw_lab.policy_schema import ACTION_SCALE
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actions import (
    ACTION_DIM,
    ACTION_GROUP_DIMS,
    ButterworthActionState,
    action_scale_vector,
    butterworth_dc_gain,
    butterworth_gain,
    canonicalize_action_scale,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.dynamics import (
    FAT2ComRadiusState,
    SpeedReferenceCfg,
    SpeedReferenceState,
    WrenchConsistencyState,
    foot_support_polygon,
    rolling_resistance_wrench,
    sagittal_com_radius,
    torso_pitch_from_world_vertical,
    torso_tilt_from_slope_normal,
    update_speed_reference,
    update_wrench_consistency_state,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    CommandState,
)

DT = 0.02
SPEED_CFG = SpeedReferenceCfg(
    acceleration_limit=0.8,
    jerk_limit=2.5,
    response_time=0.5,
    velocity_tolerance=1.0e-3,
)


def _run_reference(start: float, target: float) -> tuple[torch.Tensor, torch.Tensor]:
    state = SpeedReferenceState(
        v_ref=torch.tensor([start], dtype=torch.float64),
        a_ref=torch.zeros(1, dtype=torch.float64),
    )
    sample = torch.tensor([target], dtype=torch.float64)
    velocities = [state.v_ref.clone()]
    accelerations = [state.a_ref.clone()]
    for _ in range(1_000):
        previous_acceleration = state.a_ref.clone()
        update_speed_reference(state, sample, DT, SPEED_CFG)
        assert torch.all(
            torch.abs(state.a_ref) <= SPEED_CFG.acceleration_limit + 1.0e-12
        )
        jerk = torch.abs((state.a_ref - previous_acceleration) / DT)
        assert torch.all(jerk <= SPEED_CFG.jerk_limit + 1.0e-12)
        velocities.append(state.v_ref.clone())
        accelerations.append(state.a_ref.clone())
        if state.v_ref.item() == target and state.a_ref.item() == 0.0:
            break
    else:
        raise AssertionError(f"reference did not settle from {start} to {target}")
    return torch.cat(velocities), torch.cat(accelerations)


@pytest.mark.parametrize(
    ("start", "target"),
    ((0.0, 1.0), (1.0, 0.2), (0.7, -0.4), (-0.3, 0.5)),
)
def test_speed_reference_respects_acceleration_and_jerk(
    start: float, target: float
) -> None:
    velocities, accelerations = _run_reference(start, target)
    assert velocities[-1].item() == target
    assert accelerations[-1].item() == 0.0


def test_command_reset_clears_selected_environments() -> None:
    state = CommandState.zeros(3, dtype=torch.float64)
    state.v_sample[:] = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    state.v_ref[:] = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
    state.a_ref[:] = torch.tensor([0.2, -0.2, 0.1], dtype=torch.float64)
    state.resampling_elapsed_s[:] = 1.0
    state.reset(torch.tensor([0, 2]))
    torch.testing.assert_close(
        state.v_sample, torch.tensor([0.0, 0.4, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        state.v_ref, torch.tensor([0.0, 0.3, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        state.a_ref, torch.tensor([0.0, -0.2, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        state.resampling_elapsed_s,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )


def test_action_scale_matches_unitree_motor_defaults() -> None:
    assert ACTION_GROUP_DIMS == {
        "lower": 12,
        "waist": 3,
        "shoulder": 6,
        "elbow": 2,
        "wrist": 6,
    }
    assert sum(ACTION_GROUP_DIMS.values()) == ACTION_DIM == 29
    expected = tuple(
        0.25 * effort / stiffness
        for effort, stiffness in zip(
            G1_JOINT_EFFORT_LIMITS, G1_JOINT_STIFFNESS, strict=True
        )
    )
    assert ACTION_SCALE == G1_ACTION_SCALE == expected
    torch.testing.assert_close(
        action_scale_vector(dtype=torch.float64),
        torch.tensor(expected, dtype=torch.float64),
    )


def test_butterworth_filter_response_and_reset() -> None:
    assert math.isclose(butterworth_dc_gain(), 1.0, abs_tol=3.0e-8)
    assert math.isclose(butterworth_gain(4.0), 1.0 / math.sqrt(2.0), abs_tol=3.0e-8)
    q_ref = torch.linspace(-0.6, 0.6, ACTION_DIM, dtype=torch.float64).unsqueeze(0)
    state = ButterworthActionState.create(q_ref)
    torch.testing.assert_close(
        state.process(torch.zeros_like(q_ref), action_scale_vector(dtype=torch.float64)),
        q_ref,
        rtol=0.0,
        atol=3.0e-8,
    )
    replacement = q_ref + 0.25
    state.reset(replacement)
    torch.testing.assert_close(state.target, replacement)


def test_canonicalize_action_scale_rejects_environment_variation() -> None:
    expected = torch.linspace(0.15, 0.30, 14)
    batched = expected.unsqueeze(0).expand(13, -1).clone()
    torch.testing.assert_close(canonicalize_action_scale(batched, 14, 13), expected)
    batched[7, 3] = 0.5
    with pytest.raises(ValueError, match="differs between environments"):
        canonicalize_action_scale(batched, 14, 13)


def test_rolling_resistance_opposes_each_wheel() -> None:
    dtype = torch.float64
    gamma = torch.tensor([0.12, -0.08], dtype=dtype)
    tangent = torch.stack(
        (torch.cos(gamma), torch.zeros_like(gamma), torch.sin(gamma)), dim=-1
    )
    normal = torch.stack(
        (-torch.sin(gamma), torch.zeros_like(gamma), torch.cos(gamma)), dim=-1
    )
    wheel_speed = torch.tensor([[1.0, 0.8], [-1.0, -0.7]], dtype=dtype)
    normal_force = torch.tensor([[310.0, 330.0], [280.0, 300.0]], dtype=dtype)
    c_rr = torch.tensor([0.02, 0.03], dtype=dtype)
    force, filtered_normal, measured_speed = rolling_resistance_wrench(
        wheel_speed[..., None] * tangent[:, None, :],
        normal_force[..., None] * normal[:, None, :],
        tangent,
        normal,
        c_rr,
        normal_force,
        velocity_epsilon=0.05,
        normal_force_filter_hz=20.0,
        dt=0.005,
    )
    force_s = torch.sum(force * tangent[:, None, :], dim=-1)
    expected = -c_rr[:, None] * normal_force * torch.tanh(wheel_speed / 0.05)
    torch.testing.assert_close(filtered_normal, normal_force)
    torch.testing.assert_close(measured_speed, wheel_speed)
    torch.testing.assert_close(force_s, expected)
    assert torch.all(force_s * wheel_speed < 0.0)


def test_wrench_consistency_requires_a_complete_window() -> None:
    state = WrenchConsistencyState.zeros(2, 3, dtype=torch.float64)
    analytic = torch.tensor([[100.0, 50.0], [100.0, 50.0]], dtype=torch.float64)
    measured = torch.tensor([[110.0, 55.0], [-100.0, 50.0]], dtype=torch.float64)
    for step in range(3):
        consistent, relative_error, filtered = update_wrench_consistency_state(
            state,
            analytic,
            measured,
            torch.ones(2, dtype=torch.bool),
            relative_tolerance=0.35,
            absolute_floor_n=5.0,
        )
        if step < 2:
            assert not torch.any(consistent)
    torch.testing.assert_close(consistent, torch.tensor([True, False]))
    torch.testing.assert_close(relative_error[0], torch.tensor([0.1, 0.1], dtype=torch.float64))
    torch.testing.assert_close(filtered, analytic)


def test_fat2_radius_excludes_lateral_offset_and_holds_invalid_samples() -> None:
    robot_com = torch.tensor([[0.6, 4.0, 0.3], [0.3, -7.0, 0.4]], dtype=torch.float64)
    radius = sagittal_com_radius(
        robot_com,
        torch.zeros_like(robot_com),
        torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=torch.float64),
    )
    torch.testing.assert_close(radius, torch.tensor([math.sqrt(0.45), 0.5], dtype=torch.float64))
    state = FAT2ComRadiusState.initialized(1, 2, 0.715, dtype=torch.float64)
    first = state.update(
        torch.tensor([0.6], dtype=torch.float64),
        torch.tensor([True]),
        minimum=0.5,
        maximum=0.85,
    )
    held = state.update(
        torch.tensor([float("nan")], dtype=torch.float64),
        torch.tensor([True]),
        minimum=0.5,
        maximum=0.85,
    )
    torch.testing.assert_close(first, held)


def test_foot_support_polygon_uses_collision_center_offset() -> None:
    points, mask, center = foot_support_polygon(
        torch.tensor([[[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]], dtype=torch.float64),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2], dtype=torch.float64),
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
        foot_half_length=0.085,
        foot_half_width=0.03,
        foot_center_offset_x=0.035,
    )
    torch.testing.assert_close(torch.amin(points[..., 0]), torch.tensor(-0.05, dtype=torch.float64))
    torch.testing.assert_close(torch.amax(points[..., 0]), torch.tensor(0.12, dtype=torch.float64))
    torch.testing.assert_close(center, torch.tensor([[0.035, 0.0, 0.0]], dtype=torch.float64))
    assert torch.all(mask)


@pytest.mark.parametrize("gradient", (-0.06, 0.0, 0.06))
def test_torso_pitch_is_measured_from_world_vertical(gradient: float) -> None:
    gamma = math.atan(gradient)
    tangent = torch.tensor([[math.cos(gamma), 0.0, math.sin(gamma)]], dtype=torch.float64)
    pitch = 0.19
    quaternion = torch.tensor(
        [[math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        torso_pitch_from_world_vertical(quaternion, tangent),
        torch.tensor([pitch], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_torso_tilt_ignores_yaw() -> None:
    angle = 0.31
    half = 0.5 * angle
    quaternions = torch.tensor(
        [
            [math.cos(half), math.sin(half), 0.0, 0.0],
            [math.cos(half), 0.0, math.sin(half), 0.0],
            [math.cos(half), 0.0, 0.0, math.sin(half)],
        ],
        dtype=torch.float64,
    )
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand(3, -1)
    torch.testing.assert_close(
        torso_tilt_from_slope_normal(quaternions, normal),
        torch.tensor([angle, angle, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
