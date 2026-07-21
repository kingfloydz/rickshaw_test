"""Pure-Torch tests for command limiting, action filtering, and cart forces."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.assets.rickshaw import HITCH_X, HITCH_Z, WHEEL_RADIUS
from g1_rickshaw_lab.static_equilibrium import fixed_contact_static_components

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import (
    dynamics as dynamics_module,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actions import (
    ACTION_DIM,
    ACTION_GROUP_DIMS,
    ACTION_GROUP_SCALES,
    ButterworthActionState,
    action_scale_vector,
    butterworth_dc_gain,
    butterworth_gain,
    canonicalize_action_scale,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.dynamics import (
    AnalyticForceCfg,
    CartInteractionWrenchState,
    FAT2ComRadiusState,
    GRAVITY,
    RickshawMassProperties,
    RollingResistanceCfg,
    SpeedReferenceCfg,
    SpeedReferenceState,
    WrenchConsistencyState,
    adapt_d6_reaction_wrench,
    analytic_handle_force,
    apply_rolling_resistance,
    articulation_center_of_mass,
    cart_ground_contact_force_w,
    configure_rolling_resistance,
    foot_support_polygon,
    project_hand_wrench_to_slope,
    rolling_resistance_wrench,
    sagittal_com_radius,
    torso_pitch_from_world_vertical,
    torso_tilt_from_slope_normal,
    update_analytic_rickshaw_force,
    update_speed_reference,
    update_wrench_consistency_state,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    CommandState,
    D6ConstraintManager,
    HandleConstraintCfg,
    RickshawPoseTargetCfg,
    SpeedCommandSamplingCfg,
    _write_effective_terrain_friction_to_physx,
    _write_payload_to_physx,
    d6_spatial_impulse_magnitudes,
    fit_cart_pose_to_hitch_targets,
    finish_closed_chain_reset,
    install_q_ref_from_reset_library,
    install_reset_pose_batch,
    recover_d6_wrench_on_robot,
    spatial_wrenches_sln_to_world,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import (
    events as events_module,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.terminations import (
    d6_safety_violation,
)


DT = 0.02
SPEED_CFG = SpeedReferenceCfg(
    acceleration_limit=0.8,
    jerk_limit=2.5,
    response_time=0.5,
    velocity_tolerance=1.0e-3,
)


class _FakePhysxView:
    def __init__(
        self,
        masses: torch.Tensor,
        coms: torch.Tensor,
        inertias: torch.Tensor,
        materials: torch.Tensor,
    ) -> None:
        self.masses = masses
        self.coms = coms
        self.inertias = inertias
        self.materials = materials

    def get_masses(self) -> torch.Tensor:
        return self.masses

    def get_coms(self) -> torch.Tensor:
        return self.coms

    def get_inertias(self) -> torch.Tensor:
        return self.inertias

    def get_material_properties(self) -> torch.Tensor:
        return self.materials

    def set_masses(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.masses[env_ids] = values[env_ids]

    def set_coms(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.coms[env_ids] = values[env_ids]

    def set_inertias(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.inertias[env_ids] = values[env_ids]

    def set_material_properties(
        self, values: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.materials[env_ids] = values[env_ids]


class _FakeScene(dict):
    @property
    def terrain(self):
        return self["terrain"]


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
        step_jerk = torch.abs((state.a_ref - previous_acceleration) / DT)
        assert torch.all(step_jerk <= SPEED_CFG.jerk_limit + 1.0e-12)
        velocities.append(state.v_ref.clone())
        accelerations.append(state.a_ref.clone())
        if state.v_ref.item() == target and state.a_ref.item() == 0.0:
            break
    else:
        raise AssertionError(f"reference did not settle from {start} to {target}")

    return torch.cat(velocities), torch.cat(accelerations)


@pytest.mark.parametrize(
    ("start", "target", "kind"),
    (
        (0.0, 1.0, "rise"),
        (1.0, 0.2, "fall"),
        (0.7, -0.4, "reverse"),
        (-0.3, 0.5, "reverse"),
    ),
)
def test_speed_reference_rise_fall_reverse_and_settle(
    start: float, target: float, kind: str
) -> None:
    velocities, accelerations = _run_reference(start, target)

    assert velocities[-1].item() == target
    assert accelerations[-1].item() == 0.0
    increments = torch.diff(velocities)
    expected_direction = math.copysign(1.0, target - start)
    assert torch.all(increments * expected_direction >= -1.0e-12)
    if kind == "reverse":
        assert velocities.min() <= 0.0 <= velocities.max()


def test_command_reset_clears_sample_reference_and_acceleration() -> None:
    state = CommandState.zeros(3, dtype=torch.float64)
    state.v_sample[:] = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    state.v_ref[:] = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
    state.a_ref[:] = torch.tensor([0.2, -0.2, 0.1], dtype=torch.float64)
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


def test_d6_wrench_recovery_uses_physx_parent_on_child_sign() -> None:
    incoming_force = torch.tensor([[[2.0, -3.0, -100.0]]], dtype=torch.float64)
    incoming_torque = torch.tensor([[[1.0, 14.0, -2.0]]], dtype=torch.float64)
    mass = torch.tensor([[[0.02]]], dtype=torch.float64)
    gravity = torch.tensor([[[0.0, 0.0, -9.81]]], dtype=torch.float64)
    acceleration = torch.tensor([[[0.5, -1.0, 2.0]]], dtype=torch.float64)
    inertial_torque = torch.tensor([[[0.1, 0.4, -0.2]]], dtype=torch.float64)

    wrench = recover_d6_wrench_on_robot(
        incoming_force,
        incoming_torque,
        mass,
        gravity,
        acceleration,
        inertial_torque,
    )
    expected = torch.tensor(
        [[[1.99, -2.98, -100.2362, 0.9, 13.6, -1.8]]], dtype=torch.float64
    )
    torch.testing.assert_close(wrench, expected)


def test_d6_spatial_impulse_keeps_linear_and_angular_safety_channels() -> None:
    impulse = torch.zeros((2, 2, 6), dtype=torch.float64)
    impulse[0, 0, :3] = torch.tensor([3.0, 4.0, 0.0])
    impulse[0, 1, 3:] = torch.tensor([0.0, 0.0, 12.0])
    impulse[1, 0, :3] = torch.tensor([0.0, 0.0, 2.0])
    impulse[1, 1, 3:] = torch.tensor([0.0, 8.0, 0.0])

    channels = d6_spatial_impulse_magnitudes(impulse)

    torch.testing.assert_close(
        channels, torch.tensor([[5.0, 12.0], [2.0, 8.0]], dtype=torch.float64)
    )
    violation = d6_safety_violation(
        torch.zeros(2, dtype=torch.float64),
        channels,
        residual_limit=0.1,
        impulse_limit=10.0,
    )
    torch.testing.assert_close(violation, torch.tensor([True, False]))


def test_cart_interaction_preserves_complete_per_side_d6_wrench(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d6_wrench = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]
    )
    force_on_cart = torch.tensor([[30.0, -6.0, 9.0]])
    runtime_state = SimpleNamespace(
        hand_force_w=torch.zeros((1, 3)),
        hand_torque_w=torch.ones((1, 3)),
        d6_truth_wrench_w=torch.zeros((1, 2, 6)),
        d6_wrench_w=torch.zeros((1, 2, 6)),
    )
    interaction_state = SimpleNamespace(
        finish=lambda *_args: (force_on_cart, torch.tensor([True]))
    )
    env = SimpleNamespace(
        read_d6_reaction_residual=lambda: (
            d6_wrench,
            torch.zeros(1),
            torch.zeros((1, 2)),
        ),
        cart_interaction_wrench_state=interaction_state,
        cfg=SimpleNamespace(sim=SimpleNamespace(gravity=(0.0, 0.0, -9.81))),
        device="cpu",
        step_dt=0.02,
        rickshaw_state=runtime_state,
        cart_interaction_wrench_valid=torch.zeros(1, dtype=torch.bool),
    )
    monkeypatch.setattr(
        dynamics_module,
        "cart_ground_contact_force_w",
        lambda _env: torch.zeros((1, 3)),
    )

    hand_force = dynamics_module.update_cart_interaction_wrench(
        env,
        cart_kinematics=(torch.zeros((1, 3)), torch.zeros((1, 3)), torch.ones(1)),
    )

    torch.testing.assert_close(hand_force, -force_on_cart)
    torch.testing.assert_close(runtime_state.hand_force_w, -force_on_cart)
    torch.testing.assert_close(runtime_state.hand_torque_w, torch.zeros((1, 3)))
    torch.testing.assert_close(runtime_state.d6_truth_wrench_w, d6_wrench)
    torch.testing.assert_close(runtime_state.d6_wrench_w, d6_wrench)
    assert not torch.equal(
        runtime_state.d6_wrench_w[:, 0], runtime_state.d6_wrench_w[:, 1]
    )
    assert torch.all(runtime_state.d6_wrench_w[..., 3:] != 0.0)


def test_replicated_d6_manager_retains_immutable_nominal_config() -> None:
    cfg = HandleConstraintCfg(
        robot_body_paths=("left", "right"),
        hitch_body_paths=("left_hitch", "right_hitch"),
        grasp_local_positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        grasp_local_quaternions_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        linear_stiffness=100.0,
        linear_damping=10.0,
        angular_stiffness=50.0,
        angular_damping=5.0,
        max_force=1000.0,
        max_torque=100.0,
        linear_limit=0.02,
        angular_limit=0.1,
        rotation_free_axes=(False, True, False),
        rotation_driven_axes=(True, False, True),
        reaction_is_joint_on_robot=True,
    )
    manager = D6ConstraintManager(SimpleNamespace(num_envs=3), cfg)
    manager.created = True
    assert manager.cfg.max_force == 1000.0
    assert not hasattr(manager, "parameter_values")
    assert not hasattr(manager, "set_physics_parameters")
    assert not hasattr(manager, "set_drive_fraction")


def test_wrench_consistency_uses_full_window_and_rejects_persistent_bias() -> None:
    state = WrenchConsistencyState.zeros(2, 3, dtype=torch.float64)
    analytic = torch.tensor([[100.0, 50.0], [100.0, 50.0]], dtype=torch.float64)
    measured = torch.tensor([[110.0, 55.0], [-100.0, 50.0]], dtype=torch.float64)
    valid = torch.ones(2, dtype=torch.bool)

    for step in range(3):
        consistent, relative_error, filtered_analytic = update_wrench_consistency_state(
            state,
            analytic,
            measured,
            valid,
            relative_tolerance=0.35,
            absolute_floor_n=5.0,
        )
        if step < 2:
            assert not torch.any(consistent)

    torch.testing.assert_close(consistent, torch.tensor([True, False]))
    torch.testing.assert_close(
        relative_error[0], torch.tensor([0.1, 0.1], dtype=torch.float64)
    )
    torch.testing.assert_close(filtered_analytic, analytic)

    state.reset(torch.tensor([0]))
    consistent, _, _ = update_wrench_consistency_state(
        state,
        analytic,
        measured,
        valid,
        relative_tolerance=0.35,
        absolute_floor_n=5.0,
    )
    assert not consistent[0]
    assert not consistent[1]


def test_wrench_consistency_handles_transient_signed_force_cancellation() -> None:
    state = WrenchConsistencyState.zeros(1, 4, dtype=torch.float64)
    analytic_tangent = (100.0, -100.0, 100.0, -80.0)
    measured_tangent = (95.0, -105.0, 95.0, -105.0)

    for analytic_s, measured_s in zip(analytic_tangent, measured_tangent, strict=True):
        consistent, relative_error, filtered_analytic = update_wrench_consistency_state(
            state,
            torch.tensor([[analytic_s, 100.0]], dtype=torch.float64),
            torch.tensor([[measured_s, 110.0]], dtype=torch.float64),
            torch.ones(1, dtype=torch.bool),
            relative_tolerance=0.35,
            absolute_floor_n=5.0,
        )

    assert consistent.item()
    torch.testing.assert_close(
        filtered_analytic, torch.tensor([[5.0, 100.0]], dtype=torch.float64)
    )
    # The signed means are +5 N and -5 N, but their 10 N impulse mismatch is
    # small relative to the 95 N mean absolute transient load.
    torch.testing.assert_close(
        relative_error,
        torch.tensor([[10.0 / 95.0, 0.1]], dtype=torch.float64),
    )


def test_fat2_sagittal_com_radius_excludes_lateral_offset() -> None:
    robot_com = torch.tensor([[0.6, 4.0, 0.3], [0.3, -7.0, 0.4]], dtype=torch.float64)
    support_center = torch.zeros_like(robot_com)
    tangent = torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    normal = torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=torch.float64)

    radius = sagittal_com_radius(robot_com, support_center, tangent, normal)

    torch.testing.assert_close(
        radius, torch.tensor([math.sqrt(0.45), 0.5], dtype=torch.float64)
    )


def test_fat2_com_radius_window_holds_invalid_samples_and_resets() -> None:
    reference = 0.715092420262594
    state = FAT2ComRadiusState.initialized(2, 3, reference, dtype=torch.float64)
    valid = torch.tensor([True, True])

    first = state.update(
        torch.tensor([0.60, 0.40], dtype=torch.float64),
        valid,
        minimum=0.50,
        maximum=0.85,
    ).clone()
    second = state.update(
        torch.tensor([0.70, 0.90], dtype=torch.float64),
        valid,
        minimum=0.50,
        maximum=0.85,
    ).clone()
    third = state.update(
        torch.tensor([0.80, 0.70], dtype=torch.float64),
        torch.tensor([True, False]),
        minimum=0.50,
        maximum=0.85,
    ).clone()
    fourth = state.update(
        torch.tensor([0.90, float("nan")], dtype=torch.float64),
        valid,
        minimum=0.50,
        maximum=0.85,
    ).clone()

    torch.testing.assert_close(first, torch.tensor([0.60, 0.50], dtype=torch.float64))
    torch.testing.assert_close(second, torch.tensor([0.65, 0.675], dtype=torch.float64))
    torch.testing.assert_close(third, torch.tensor([0.70, 0.675], dtype=torch.float64))
    torch.testing.assert_close(
        fourth, torch.tensor([(0.70 + 0.80 + 0.85) / 3.0, 0.675], dtype=torch.float64)
    )

    state.reset(torch.tensor([0]))
    torch.testing.assert_close(
        state.filtered_radius,
        torch.tensor([reference, 0.675], dtype=torch.float64),
    )
    assert state.count.tolist() == [0, 2]


def test_cart_interaction_wrench_uses_corrected_contact_window() -> None:
    state = CartInteractionWrenchState.initialized(
        torch.zeros((1, 3), dtype=torch.float64)
    )
    rolling = torch.tensor([[-2.0, 0.0, 0.0]], dtype=torch.float64)
    for contact_x in (100.0, 1.0, 2.0, 3.0):
        state.accumulate(
            torch.tensor([[contact_x, 0.0, 0.0]], dtype=torch.float64),
            rolling,
        )
    force, valid = state.finish(
        torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float64),
        1.0,
    )

    # Contact average is (1+2+3+4)/4=2.5 N after replacing the stale sample.
    torch.testing.assert_close(
        force, torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64)
    )
    assert valid.item()
    assert state.sample_count.item() == 0

    force, valid = state.finish(
        torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        1.0,
    )
    torch.testing.assert_close(force, torch.zeros_like(force))
    assert not valid.item()


def test_cart_ground_contact_excludes_non_wheel_cart_contacts() -> None:
    wheel_forces = torch.tensor(
        [[[1.0, 0.0, 10.0], [2.0, 0.0, 20.0], [99.0, 0.0, 0.0]]]
    )
    env = SimpleNamespace(
        scene={
            "wheel_contacts": SimpleNamespace(
                data=SimpleNamespace(net_forces_w=wheel_forces)
            )
        },
        wheel_sensor_ids=[0, 1],
    )

    torch.testing.assert_close(
        cart_ground_contact_force_w(env), torch.tensor([[3.0, 0.0, 30.0]])
    )


def test_closed_chain_reset_finishes_on_normal_controller(monkeypatch) -> None:
    env_ids = torch.tensor([0, 2])
    cleared: list[torch.Tensor] = []
    command_samples: list[torch.Tensor] = []
    action_resets: list[torch.Tensor] = []
    action_term = SimpleNamespace(reset=lambda ids: action_resets.append(ids.clone()))
    env = SimpleNamespace(
        device="cpu",
        action_manager=SimpleNamespace(_terms={"lower": action_term}),
        cfg=SimpleNamespace(
            policy_update=SimpleNamespace(command_sampling=SpeedCommandSamplingCfg()),
        ),
    )
    monkeypatch.setattr(
        events_module,
        "reset_task_state",
        lambda _env, ids: cleared.append(ids.clone()),
    )
    monkeypatch.setattr(
        events_module,
        "resample_speed_command",
        lambda _env, ids, _cfg: command_samples.append(ids.clone()),
    )

    finish_closed_chain_reset(env, env_ids)

    assert len(cleared) == 1
    torch.testing.assert_close(cleared[0], env_ids)
    assert len(command_samples) == 1
    torch.testing.assert_close(command_samples[0], env_ids)
    assert len(action_resets) == 1
    torch.testing.assert_close(action_resets[0], env_ids)


def test_fixed_contact_allocator_preserves_torch_batch_for_scalar_free_axis_torque() -> (
    None
):
    values = torch.tensor([100.0, 120.0], dtype=torch.float64)

    hand_rows, wheel_rows = fixed_contact_static_components(
        gravity_tangent=0.1 * values,
        gravity_normal=values,
        com_s=0.4 * torch.ones_like(values),
        com_l=torch.tensor([0.0, 0.01], dtype=values.dtype),
        com_n=0.3 * torch.ones_like(values),
        handle_s=1.2 * torch.ones_like(values),
        handle_n=0.5 * torch.ones_like(values),
        hitch_half_width=0.24,
        wheel_track=0.75,
        pitch_torque_on_robot=0.0,
    )

    assert all(torch.is_tensor(component) for row in hand_rows for component in row)
    assert all(torch.is_tensor(component) for row in wheel_rows for component in row)


def test_runtime_static_target_uses_compiled_tables() -> None:
    q_reset = torch.zeros((1, ACTION_DIM))
    q_ref = torch.full_like(q_reset, 7.0)
    env = SimpleNamespace(
        device="cpu",
        slope=torch.zeros(1),
        reset_pose_gradients=torch.zeros(1),
        reset_pose_index=torch.zeros(1, dtype=torch.long),
        reset_q_reset_table=q_reset,
        reset_q_ref_table=q_ref,
        action_state=SimpleNamespace(q_ref=torch.zeros_like(q_reset)),
        motor_strength=torch.ones(1),
        joint_model_error=torch.zeros_like(q_reset),
        policy_joint_stiffness=torch.full_like(q_reset, 2.0),
        reset_policy_joint_pos=torch.zeros_like(q_reset),
    )

    install_q_ref_from_reset_library(env, torch.tensor([0]))

    torch.testing.assert_close(env.action_state.q_ref, q_ref)


def test_runtime_static_target_uses_the_calibrated_nominal_reference() -> None:
    q_reset = torch.ones((1, ACTION_DIM))
    nominal_q_ref = torch.full_like(q_reset, 3.0)
    env = SimpleNamespace(
        device="cpu",
        slope=torch.zeros(1),
        reset_pose_gradients=torch.zeros(1),
        reset_pose_index=torch.zeros(1, dtype=torch.long),
        reset_q_reset_table=q_reset,
        reset_q_ref_table=nominal_q_ref,
        action_state=SimpleNamespace(q_ref=torch.zeros_like(q_reset)),
        reset_policy_joint_pos=torch.zeros_like(q_reset),
    )

    install_q_ref_from_reset_library(env, torch.tensor([0]))

    torch.testing.assert_close(env.action_state.q_ref, nominal_q_ref)


def test_stage_b_pose_batch_uses_one_fixed_pose_per_environment() -> None:
    poses = [
        SimpleNamespace(
            gradient=0.09,
            q_reset=[1.0] * ACTION_DIM,
            q_ref=[2.0] * ACTION_DIM,
            root_pitch=0.1,
            root_height=0.7,
            handle_wrenches_sln=[[0.0] * 6, [0.0] * 6],
        ),
        SimpleNamespace(
            gradient=0.09,
            q_reset=[3.0] * ACTION_DIM,
            q_ref=[4.0] * ACTION_DIM,
            root_pitch=0.2,
            root_height=0.8,
            handle_wrenches_sln=[[0.0] * 6, [0.0] * 6],
        ),
    ]
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        slope=torch.tensor([0.09, 0.09]),
        action_state=SimpleNamespace(q_ref=torch.zeros((2, ACTION_DIM))),
        motor_strength=torch.ones(2),
        joint_model_error=torch.zeros((2, ACTION_DIM)),
        reset_policy_joint_pos=torch.zeros((2, ACTION_DIM)),
    )

    install_reset_pose_batch(env, poses)
    install_q_ref_from_reset_library(env, torch.tensor([0, 1]))

    torch.testing.assert_close(
        env.action_state.q_ref,
        torch.tensor([[2.0] * ACTION_DIM, [4.0] * ACTION_DIM]),
    )
    torch.testing.assert_close(
        env.reset_policy_joint_pos,
        torch.tensor([[1.0] * ACTION_DIM, [3.0] * ACTION_DIM]),
    )
    assert env.reset_pose_index.tolist() == [0, 1]


def test_static_wrench_rotation_and_two_point_cart_fit_are_per_side() -> None:
    wrenches_sln = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]
    )
    tangent = torch.tensor([[0.0, 1.0, 0.0]])
    lateral = torch.tensor([[-1.0, 0.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    world = spatial_wrenches_sln_to_world(wrenches_sln, tangent, lateral, normal)
    torch.testing.assert_close(
        world[0, 0], torch.tensor([-2.0, 1.0, 3.0, -5.0, 4.0, 6.0])
    )
    torch.testing.assert_close(
        world[0, 1], torch.tensor([-8.0, 7.0, 9.0, -11.0, 10.0, 12.0])
    )

    cfg = RickshawPoseTargetCfg(
        hitch_height_target=0.85,
        hitch_height_tolerance=0.005,
        hitch_vertical_speed_tolerance=0.02,
    )
    angle = torch.tensor(0.02)
    quaternion = torch.tensor(
        [[torch.cos(0.5 * angle), 0.0, 0.0, torch.sin(0.5 * angle)]]
    )
    local = torch.tensor(
        [
            [
                [cfg.hitch_x, cfg.hitch_half_width, cfg.hitch_z],
                [cfg.hitch_x, -cfg.hitch_half_width, cfg.hitch_z],
            ]
        ]
    )
    translation = torch.tensor([[2.0, -0.4, 0.2]])
    targets = translation[:, None, :] + events_module.quat_apply_wxyz(
        quaternion[:, None, :].expand(-1, 2, -1), local
    )
    root, fitted_quaternion, error = fit_cart_pose_to_hitch_targets(
        targets,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        cfg,
    )
    torch.testing.assert_close(root, translation, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(fitted_quaternion, quaternion, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(error, torch.zeros_like(error), atol=1.0e-6, rtol=0.0)


def test_action_partition_scales_and_butterworth_acceptance() -> None:
    assert ACTION_GROUP_DIMS == {
        "lower": 12,
        "waist": 3,
        "shoulder": 6,
        "elbow": 2,
        "wrist": 6,
    }
    assert sum(ACTION_GROUP_DIMS.values()) == ACTION_DIM == 29
    scales = action_scale_vector(dtype=torch.float64)
    expected = [ACTION_GROUP_SCALES["lower"]] * 12
    expected.extend([ACTION_GROUP_SCALES["waist"]] * 3)
    for _ in range(2):
        expected.extend([ACTION_GROUP_SCALES["shoulder"]] * 3)
        expected.append(ACTION_GROUP_SCALES["elbow"])
        expected.extend([ACTION_GROUP_SCALES["wrist"]] * 3)
    torch.testing.assert_close(scales, torch.tensor(expected, dtype=torch.float64))

    assert math.isclose(butterworth_dc_gain(), 1.0, rel_tol=0.0, abs_tol=3.0e-8)
    assert math.isclose(
        butterworth_gain(4.0), 1.0 / math.sqrt(2.0), rel_tol=0.0, abs_tol=3.0e-8
    )


def test_canonicalize_action_scale_accepts_isaaclab_environment_batch() -> None:
    expected = torch.linspace(0.15, 0.30, 14)
    batched = expected.unsqueeze(0).expand(13, -1).clone()
    torch.testing.assert_close(canonicalize_action_scale(batched, 14, 13), expected)
    torch.testing.assert_close(
        canonicalize_action_scale(0.4, 12, 13), torch.full((12,), 0.4)
    )

    batched[7, 3] = 0.5
    with pytest.raises(ValueError, match="differs between environments"):
        canonicalize_action_scale(batched, 14, 13)


def test_butterworth_frequency_response_and_reset_to_q_ref() -> None:
    q_ref = torch.linspace(-0.6, 0.6, ACTION_DIM, dtype=torch.float64).unsqueeze(0)
    state = ButterworthActionState.create(q_ref)
    zero_target = state.process(
        torch.zeros_like(q_ref), action_scale_vector(dtype=torch.float64)
    )
    torch.testing.assert_close(zero_target, q_ref, rtol=0.0, atol=3.0e-8)

    state.process(torch.ones_like(q_ref), action_scale_vector(dtype=torch.float64))
    torch.testing.assert_close(state.prev_raw_action, torch.zeros_like(q_ref))
    torch.testing.assert_close(state.raw_action, torch.ones_like(q_ref))
    replacement = q_ref + 0.25
    state.reset(replacement)
    torch.testing.assert_close(state.prev_raw_action, torch.zeros_like(q_ref))
    torch.testing.assert_close(state.raw_action, torch.zeros_like(q_ref))
    torch.testing.assert_close(state.x_prev, replacement)
    torch.testing.assert_close(state.y_prev, replacement)
    torch.testing.assert_close(state.target, replacement)
    after_reset = state.process(
        torch.zeros_like(q_ref), action_scale_vector(dtype=torch.float64)
    )
    torch.testing.assert_close(after_reset, replacement, rtol=0.0, atol=3.0e-8)

    # Exercise the recurrence, not only the closed-form response helper.
    sine_state = ButterworthActionState.create(
        torch.zeros((1, ACTION_DIM), dtype=torch.float64)
    )
    inputs: list[float] = []
    outputs: list[float] = []
    for step in range(500):
        sample = math.sin(2.0 * math.pi * 4.0 * step / 50.0)
        output = sine_state.process(
            torch.full((1, ACTION_DIM), sample, dtype=torch.float64), 1.0
        )
        inputs.append(sample)
        outputs.append(output[0, 0].item())
    input_rms = torch.tensor(inputs[100:], dtype=torch.float64).square().mean().sqrt()
    output_rms = torch.tensor(outputs[100:], dtype=torch.float64).square().mean().sqrt()
    assert math.isclose(
        (output_rms / input_rms).item(),
        1.0 / math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=2.0e-6,
    )


def test_rolling_resistance_opposes_each_wheel_and_has_correct_magnitude() -> None:
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
    wheel_velocity = wheel_speed[..., None] * tangent[:, None, :]
    contact_force = normal_force[..., None] * normal[:, None, :]
    c_rr = torch.tensor([0.02, 0.03], dtype=dtype)

    force, filtered_normal, measured_speed = rolling_resistance_wrench(
        wheel_velocity,
        contact_force,
        tangent,
        normal,
        c_rr,
        normal_force,
        velocity_epsilon=0.05,
        normal_force_filter_hz=20.0,
        dt=0.005,
    )
    force_s = torch.sum(force * tangent[:, None, :], dim=-1)
    force_n = torch.sum(force * normal[:, None, :], dim=-1)
    expected = -c_rr[:, None] * normal_force * torch.tanh(wheel_speed / 0.05)

    torch.testing.assert_close(filtered_normal, normal_force)
    torch.testing.assert_close(measured_speed, wheel_speed)
    torch.testing.assert_close(force_s, expected)
    torch.testing.assert_close(
        force_n, torch.zeros_like(force_n), rtol=0.0, atol=1.0e-12
    )
    assert torch.all(force_s * wheel_speed < 0.0)
    torch.testing.assert_close(
        torch.sum(torch.abs(force_s), dim=-1),
        c_rr * torch.sum(normal_force, dim=-1),
        rtol=2.0e-12,
        atol=1.0e-12,
    )


def test_configure_rolling_resistance_replaces_both_frozen_bindings() -> None:
    original = RollingResistanceCfg(enabled=True)
    initialize_params = {"rolling_resistance_cfg": original, "sentinel": object()}
    env_cfg = SimpleNamespace(
        rolling_resistance=original,
        events=SimpleNamespace(
            initialize_mdp=SimpleNamespace(params=initialize_params)
        ),
    )

    configured = configure_rolling_resistance(env_cfg, enabled=False)

    assert original.enabled is True
    assert configured.enabled is False
    assert env_cfg.rolling_resistance is configured
    assert initialize_params["rolling_resistance_cfg"] is configured
    assert "sentinel" in initialize_params

    with pytest.raises(ValueError, match="must be boolean"):
        configure_rolling_resistance(env_cfg, enabled=0)  # type: ignore[arg-type]


def test_analytic_force_adapter_uses_sampled_rolling_coefficient(monkeypatch) -> None:
    sampled_c_rr = torch.tensor([0.021], dtype=torch.float64)
    captured: dict[str, torch.Tensor] = {}
    cart = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_w=torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        )
    )
    env = SimpleNamespace(
        scene={"rickshaw": cart},
        path_tangent_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        path_normal_w=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        gamma=torch.zeros(1, dtype=torch.float64),
        c_rr=sampled_c_rr,
        rickshaw_state=SimpleNamespace(
            wheel_normal_force=torch.tensor([[300.0, 310.0]], dtype=torch.float64)
        ),
        rickshaw_mass_properties=object(),
        analytic_force_state=object(),
        step_dt=0.02,
    )
    monkeypatch.setattr(
        dynamics_module,
        "rickshaw_pitch_from_quaternion",
        lambda *_args: torch.zeros(1, dtype=torch.float64),
    )
    monkeypatch.setattr(
        dynamics_module,
        "actual_rickshaw_geometry_in_slope_frame",
        lambda _env: (
            torch.zeros((1, 2), dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.float64),
        ),
    )

    def _capture_update(*args, **_kwargs) -> None:
        captured["c_rr"] = args[4]

    monkeypatch.setattr(
        dynamics_module, "update_analytic_handle_force_state", _capture_update
    )

    update_analytic_rickshaw_force(
        env,
        AnalyticForceCfg(minimum_wheel_normal_force=1.0),
    )

    assert captured["c_rr"] is sampled_c_rr


class _RollingResistanceCart:
    def __init__(self, wheel_velocity_w: torch.Tensor) -> None:
        self.data = SimpleNamespace(body_lin_vel_w=wheel_velocity_w)
        self.permanent_wrench_composer = self
        self.applied_force_w: torch.Tensor | None = None
        self.applied_torque_w: torch.Tensor | None = None
        self.applied_body_ids: list[int] | None = None
        self.applied_is_global: bool | None = None

    def set_forces_and_torques(
        self,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        *,
        body_ids: list[int],
        is_global: bool,
    ) -> None:
        self.applied_force_w = force_w.clone()
        self.applied_torque_w = torque_w.clone()
        self.applied_body_ids = body_ids
        self.applied_is_global = is_global


def _rolling_resistance_adapter_env() -> SimpleNamespace:
    dtype = torch.float64
    wheel_velocity = torch.tensor([[[1.0, 0.0, 0.0], [0.7, 0.0, 0.0]]], dtype=dtype)
    wheel_contact_force = torch.tensor(
        [[[0.0, 0.0, 300.0], [0.0, 0.0, 320.0]]], dtype=dtype
    )
    cart = _RollingResistanceCart(wheel_velocity)
    return SimpleNamespace(
        scene={
            "rickshaw": cart,
            "wheel_contacts": SimpleNamespace(
                data=SimpleNamespace(net_forces_w=wheel_contact_force)
            ),
        },
        wheel_body_ids=[0, 1],
        wheel_sensor_ids=[0, 1],
        path_tangent_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype),
        path_normal_w=torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
        c_rr=torch.tensor([0.025], dtype=dtype),
        rickshaw_state=SimpleNamespace(
            wheel_normal_force=torch.zeros((1, 2), dtype=dtype)
        ),
        physics_dt=0.005,
    )


def test_rolling_resistance_enabled_controls_physx_force_but_not_normal_filter() -> (
    None
):
    enabled_env = _rolling_resistance_adapter_env()
    disabled_env = _rolling_resistance_adapter_env()
    enabled_cfg = RollingResistanceCfg(enabled=True)
    disabled_cfg = RollingResistanceCfg(enabled=False)

    enabled_force = apply_rolling_resistance(enabled_env, enabled_cfg)
    disabled_force = apply_rolling_resistance(disabled_env, disabled_cfg)

    assert torch.any(enabled_force != 0.0)
    torch.testing.assert_close(disabled_force, torch.zeros_like(disabled_force))
    torch.testing.assert_close(
        disabled_env.rickshaw_state.wheel_normal_force,
        enabled_env.rickshaw_state.wheel_normal_force,
    )
    assert torch.all(disabled_env.rickshaw_state.wheel_normal_force > 0.0)

    disabled_cart = disabled_env.scene["rickshaw"]
    assert disabled_cart.applied_force_w is not None
    assert disabled_cart.applied_torque_w is not None
    torch.testing.assert_close(
        disabled_cart.applied_force_w,
        torch.zeros_like(disabled_cart.applied_force_w),
    )
    torch.testing.assert_close(
        disabled_cart.applied_torque_w,
        torch.zeros_like(disabled_cart.applied_torque_w),
    )
    assert disabled_cart.applied_body_ids == [0, 1]
    assert disabled_cart.applied_is_global is True


def test_whole_articulation_com_is_mass_weighted_not_root_body_only() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0, 0.5], [1.0, 0.0, 1.0], [-0.5, 0.0, 1.5]]],
        dtype=torch.float64,
    )
    velocities = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    masses = torch.tensor([[10.0, 20.0, 5.0]], dtype=torch.float64)

    com, velocity, total_mass = articulation_center_of_mass(
        positions, velocities, masses
    )

    torch.testing.assert_close(
        com, torch.tensor([[0.5, 0.0, 0.9285714285714286]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        velocity, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(total_mass, torch.tensor([35.0], dtype=torch.float64))
    assert not torch.allclose(com, positions[:, 0])


def _fake_asset(
    num_envs: int, body_names: tuple[str, ...], num_shapes: int
) -> SimpleNamespace:
    num_bodies = len(body_names)
    masses = torch.ones((num_envs, num_bodies), dtype=torch.float64)
    coms = torch.zeros((num_envs, num_bodies, 7), dtype=torch.float64)
    coms[..., 6] = 1.0
    inertias = (
        torch.eye(3, dtype=torch.float64)
        .reshape(1, 1, 9)
        .repeat(num_envs, num_bodies, 1)
    )
    materials = torch.zeros((num_envs, num_shapes, 3), dtype=torch.float64)
    return SimpleNamespace(
        body_names=body_names,
        root_physx_view=_FakePhysxView(masses, coms, inertias, materials),
    )


def test_payload_write_updates_physx_mass_com_and_inertia_from_default() -> None:
    cart = _fake_asset(
        2,
        (
            "base_link",
            "left_wheel_link",
            "right_wheel_link",
            "left_tow_hitch_link",
            "right_tow_hitch_link",
        ),
        3,
    )
    view = cart.root_physx_view
    view.masses[:] = torch.tensor([36.0, 2.0, 2.0, 0.02, 0.02], dtype=torch.float64)
    base_com = torch.tensor(
        [0.7227393855133334, 0.0, 0.7026899513333335], dtype=torch.float64
    )
    view.coms[:, 0, :3] = base_com
    view.inertias[:, 0] = torch.diag(
        torch.tensor([7.393572, 22.277208, 17.829456], dtype=torch.float64)
    ).reshape(9)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=_FakeScene(rickshaw=cart),
        rickshaw_body_masses=view.masses.clone(),
        rickshaw_total_mass=torch.sum(view.masses, dim=-1),
        rickshaw_body_mass_weights=(
            view.masses / torch.sum(view.masses, dim=-1)[:, None]
        ),
        rickshaw_body_com_pos_b=view.coms[..., :3].clone(),
    )
    payload_mass = torch.tensor([2.0], dtype=torch.float64)
    payload_com = torch.tensor([[1.0, 0.1, 0.9]], dtype=torch.float64)

    _write_payload_to_physx(env, torch.tensor([1]), payload_mass, payload_com)

    assert torch.sum(view.masses[0]).item() == pytest.approx(40.04)
    assert torch.sum(view.masses[1]).item() == pytest.approx(42.04)
    assert view.masses[1, 0].item() == pytest.approx(38.0)
    assert env.rickshaw_total_mass[1].item() == pytest.approx(42.04)
    torch.testing.assert_close(
        env.rickshaw_body_mass_weights[1],
        env.rickshaw_body_masses[1] / env.rickshaw_total_mass[1],
    )
    expected_com = (36.0 * base_com + 2.0 * payload_com[0]) / 38.0
    torch.testing.assert_close(view.coms[1, 0, :3], expected_com)
    assert not torch.equal(view.inertias[1, 0], view.inertias[0, 0])


def test_per_environment_terrain_friction_is_written_to_both_contact_assets() -> None:
    robot = _fake_asset(2, ("pelvis", "left_foot", "right_foot"), 4)
    cart = _fake_asset(2, ("base_link", "left_wheel_link", "right_wheel_link"), 3)
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            physics_material=SimpleNamespace(
                friction_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            )
        )
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=_FakeScene(robot=robot, rickshaw=cart, terrain=terrain),
    )

    _write_effective_terrain_friction_to_physx(
        env, torch.tensor([1]), torch.tensor([0.73])
    )

    for asset in (robot, cart):
        torch.testing.assert_close(
            asset.root_physx_view.materials[1, :, :2],
            torch.full_like(asset.root_physx_view.materials[1, :, :2], 0.73),
        )
        torch.testing.assert_close(
            asset.root_physx_view.materials[0],
            torch.zeros_like(asset.root_physx_view.materials[0]),
        )
    torch.testing.assert_close(env.terrain_friction, torch.tensor([1.0, 0.73]))


def _mass_properties(dtype: torch.dtype = torch.float64) -> RickshawMassProperties:
    def scalar(value: float) -> torch.Tensor:
        return torch.tensor([value], dtype=dtype)

    return RickshawMassProperties(
        m_cart=scalar(65.04),
        com_x_from_axle=scalar(0.686322),
        com_z_from_axle=scalar(0.677177 - WHEEL_RADIUS),
        pitch_inertia_about_axle=scalar(70.0),
        m_eff=scalar(67.5),
        b_eff=scalar(0.28),
        handle_x_from_axle=scalar(HITCH_X),
        handle_z_from_axle=scalar(HITCH_Z - WHEEL_RADIUS),
    )


def test_foot_support_polygon_uses_urdf_collision_center_offset() -> None:
    foot_position = torch.tensor(
        [[[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]], dtype=torch.float64
    )
    foot_quaternion = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64
    )
    points, mask, center = foot_support_polygon(
        foot_position,
        foot_quaternion,
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
        foot_half_length=0.085,
        foot_half_width=0.03,
        foot_center_offset_x=0.035,
    )

    torch.testing.assert_close(
        torch.amin(points[..., 0]), torch.tensor(-0.05, dtype=torch.float64)
    )
    torch.testing.assert_close(
        torch.amax(points[..., 0]), torch.tensor(0.12, dtype=torch.float64)
    )
    torch.testing.assert_close(
        center, torch.tensor([[0.035, 0.0, 0.0]], dtype=torch.float64)
    )
    assert torch.all(mask)


def test_analytic_handle_force_static_and_positive_pull_signs() -> None:
    properties = _mass_properties()
    zeros = torch.zeros(1, dtype=torch.float64)
    wheel_normal = torch.tensor([[310.0, 328.0]], dtype=torch.float64)

    static_t_s, static_t_n, valid = analytic_handle_force(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        wheel_normal,
        properties,
    )
    expected_static_t_n = (
        properties.m_cart
        * GRAVITY
        * properties.com_x_from_axle
        / properties.handle_x_from_axle
    )
    torch.testing.assert_close(static_t_s, zeros)
    torch.testing.assert_close(static_t_n, expected_static_t_n)
    assert valid.item()
    assert static_t_n.item() > 0.0

    pull_acceleration = torch.tensor([0.4], dtype=torch.float64)
    pull_t_s, pull_t_n, _ = analytic_handle_force(
        zeros,
        pull_acceleration,
        zeros,
        zeros,
        zeros,
        zeros,
        wheel_normal,
        properties,
    )
    assert pull_t_s.item() > 0.0
    assert pull_t_n.item() < static_t_n.item()
    torch.testing.assert_close(pull_t_s, properties.m_eff * pull_acceleration)
    torch.testing.assert_close(
        pull_t_n - static_t_n,
        properties.handle_z_from_axle * pull_t_s / properties.handle_x_from_axle,
    )

    # The runtime adapter supplies actual slope-frame axle/hitch/CoM poses.
    # These values deliberately differ from the nominal cart-frame geometry.
    actual_handle = torch.tensor([[1.5, 0.4]], dtype=torch.float64)
    actual_com = torch.tensor([[0.6, 0.2]], dtype=torch.float64)
    _, actual_static_n, _ = analytic_handle_force(
        zeros,
        zeros,
        zeros,
        torch.tensor([0.9], dtype=torch.float64),
        zeros,
        zeros,
        wheel_normal,
        properties,
        handle_from_axle_sn=actual_handle,
        com_from_axle_sn=actual_com,
    )
    _, actual_pull_n, _ = analytic_handle_force(
        zeros,
        pull_acceleration,
        zeros,
        torch.tensor([0.9], dtype=torch.float64),
        zeros,
        zeros,
        wheel_normal,
        properties,
        handle_from_axle_sn=actual_handle,
        com_from_axle_sn=actual_com,
    )
    torch.testing.assert_close(
        actual_static_n,
        properties.m_cart * GRAVITY * actual_com[:, 0] / actual_handle[:, 0],
    )
    torch.testing.assert_close(
        actual_pull_n - actual_static_n,
        actual_handle[:, 1] * pull_t_s / actual_handle[:, 0],
    )
    assert actual_pull_n.item() > actual_static_n.item()

    lifted_pitch = torch.tensor([0.3180382172908412], dtype=torch.float64)
    _, lifted_t_n, _ = analytic_handle_force(
        zeros,
        zeros,
        zeros,
        lifted_pitch,
        zeros,
        zeros,
        wheel_normal,
        properties,
    )
    torch.testing.assert_close(
        lifted_t_n,
        torch.tensor([195.61152904917537], dtype=torch.float64),
        rtol=1.0e-9,
        atol=1.0e-9,
    )

    tangent = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    lateral = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    force_w = pull_t_s[:, None] * tangent + pull_t_n[:, None] * normal
    torque_w = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64)
    projected_s, projected_n, projected_y = project_hand_wrench_to_slope(
        force_w, torque_w, tangent, normal, lateral
    )
    torch.testing.assert_close(projected_s, pull_t_s)
    torch.testing.assert_close(projected_n, pull_t_n)
    torch.testing.assert_close(projected_y, torch.tensor([3.0], dtype=torch.float64))

    raw_opposite_convention = -torch.cat((force_w, torque_w), dim=-1)
    adapted = adapt_d6_reaction_wrench(
        raw_opposite_convention, reaction_is_joint_on_body=False
    )
    torch.testing.assert_close(adapted, torch.cat((force_w, torque_w), dim=-1))


@pytest.mark.parametrize("gradient", (-0.06, 0.0, 0.06))
def test_torso_pitch_is_measured_from_world_vertical(gradient: float) -> None:
    dtype = torch.float64
    gamma = math.atan(gradient)
    tangent = torch.tensor([[math.cos(gamma), 0.0, math.sin(gamma)]], dtype=dtype)

    for pitch in (-0.24, 0.0, 0.19):
        quaternion = torch.tensor(
            [[math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0]],
            dtype=dtype,
        )
        measured = torso_pitch_from_world_vertical(quaternion, tangent)
        torch.testing.assert_close(
            measured,
            torch.tensor([pitch], dtype=dtype),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_torso_safety_tilt_detects_roll_and_pitch_but_not_yaw() -> None:
    dtype = torch.float64
    angle = 0.31
    half = 0.5 * angle
    quaternions = torch.tensor(
        [
            [math.cos(half), math.sin(half), 0.0, 0.0],
            [math.cos(half), 0.0, math.sin(half), 0.0],
            [math.cos(half), 0.0, 0.0, math.sin(half)],
        ],
        dtype=dtype,
    )
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype).expand(3, -1)
    measured = torso_tilt_from_slope_normal(quaternions, normal)
    torch.testing.assert_close(
        measured,
        torch.tensor([angle, angle, 0.0], dtype=dtype),
        rtol=0.0,
        atol=1.0e-12,
    )

    gamma = math.atan(0.06)
    slope_normal = torch.tensor([[-math.sin(gamma), 0.0, math.cos(gamma)]], dtype=dtype)
    slope_aligned = torch.tensor(
        [[math.cos(0.5 * gamma), 0.0, -math.sin(0.5 * gamma), 0.0]],
        dtype=dtype,
    )
    torch.testing.assert_close(
        torso_tilt_from_slope_normal(slope_aligned, slope_normal),
        torch.zeros(1, dtype=dtype),
        rtol=0.0,
        atol=1.0e-12,
    )
