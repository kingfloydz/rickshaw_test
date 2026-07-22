"""Mjlab lifecycle for the 19-slope rigid robot-rickshaw task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
)

from g1_rickshaw_lab.assets.g1_dex1 import GRASP_SITE_NAMES
from g1_rickshaw_lab.assets.rickshaw import (
    BASE_LINK_NAME,
    HITCH_SITE_NAMES,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    RICKSHAW_URDF_SPEC,
    WHEEL_JOINT_NAMES,
    WHEEL_LINK_NAMES,
)
from g1_rickshaw_lab.configuration import G1_JOINT_ORDER
from g1_rickshaw_lab.policy_schema import HISTORY_LENGTH, TEACHER_DYNAMIC_DIM
from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_GRADIENTS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)
from g1_rickshaw_lab.static_equilibrium import MujocoStaticEquilibrium, solve_mujoco_static_equilibrium

from .closed_chain import build_assembled_spec
from .mdp.curricula import apply_terrain_assignment, balanced_slope_assignment, weighted_slope_assignment
from .mdp.dynamics import (
    AnalyticForceCfg,
    AnalyticHandleForceState,
    FAT2Cfg,
    FAT2ComRadiusState,
    RickshawMassProperties,
    SupportPolygonCfg,
    WrenchConsistencyState,
    ZMPCfg,
    ZMPKinematicState,
    combine_mass_properties,
    convex_support_margin,
    effective_cart_mass,
    effective_wheel_damping,
    fat2_reference_angle,
    filtered_first_derivative,
    foot_support_polygon,
    rickshaw_pitch_from_quaternion,
    rolling_resistance_wrench,
    sagittal_com_radius,
    slope_zmp,
    torso_pitch_from_world_vertical,
    update_analytic_handle_force_state,
    update_wrench_consistency_state,
)
from .mdp.events import (
    CommandState,
    DomainRandomizationCfg,
    PathTrackingState,
    RickshawRuntimeState,
    SpeedCommandSamplingCfg,
    StabilityState,
    _update_teacher_static_domain,
    compute_path_tracking_errors,
    resample_speed_command,
    sample_domain_parameters,
)
from .mdp.observations import ObservationHistoryState
from .task_spec import RickshawPoseTargetCfg


@dataclass(frozen=True)
class MjlabTaskRuntimeCfg:
    domain: DomainRandomizationCfg
    command: SpeedCommandSamplingCfg
    speed_acceleration_limit: float
    speed_jerk_limit: float
    rickshaw_pose: RickshawPoseTargetCfg
    analytic_force: AnalyticForceCfg
    fat2: FAT2Cfg
    support: SupportPolygonCfg
    zmp: ZMPCfg
    history_length: int = HISTORY_LENGTH
    shuffle_slopes: bool = True
    play: bool = False


def _ids(entity: Any, kind: str, names: tuple[str, ...]) -> torch.Tensor:
    finder = getattr(entity, f"find_{kind}")
    indices, resolved = finder(names, preserve_order=True)
    if tuple(resolved) != names:
        raise RuntimeError(f"{kind} order mismatch: expected {names}, got {tuple(resolved)}")
    return torch.as_tensor(indices, device=entity.data.device, dtype=torch.long)


def _slope_frame(gradient: torch.Tensor) -> tuple[torch.Tensor, ...]:
    gamma = torch.atan(gradient)
    zeros = torch.zeros_like(gamma)
    tangent = torch.stack((torch.cos(gamma), zeros, torch.sin(gamma)), dim=-1)
    lateral = torch.stack((zeros, torch.ones_like(gamma), zeros), dim=-1)
    normal = torch.stack((-torch.sin(gamma), zeros, torch.cos(gamma)), dim=-1)
    quat = quat_from_euler_xyz(zeros, -gamma, zeros)
    return gamma, tangent, lateral, normal, quat


def assign_mjlab_slope_slots(env: Any, slots: torch.Tensor) -> None:
    """Apply one canonical slope slot and matching reset library entry per environment."""

    slots = torch.as_tensor(slots, device=env.device, dtype=torch.long).reshape(-1)
    if slots.shape != (env.num_envs,) or torch.any((slots < 0) | (slots >= SLOPE_COUNT)):
        raise ValueError("slope slots must have shape [num_envs] and lie in the 19-slope grid")
    levels = torch.tensor(SLOPE_TERRAIN_LEVELS, device=env.device)[slots]
    terrain_types = torch.tensor(SLOPE_TERRAIN_TYPES, device=env.device)[slots]
    apply_terrain_assignment(env, levels, terrain_types)
    gradients = torch.tensor(SLOPE_GRADIENTS, device=env.device, dtype=torch.float32)
    env.slope_slot = slots
    env.slope = gradients[slots]
    (
        env.gamma,
        env.path_tangent_w,
        env.path_lateral_w,
        env.path_normal_w,
        env.slope_quat_w,
    ) = _slope_frame(env.slope)


def _body_mass_kinematics(env: Any, entity_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entity = env.scene[entity_name]
    body_ids = entity.indexing.body_ids
    masses = env.sim.model.body_mass[:, body_ids]
    total = masses.sum(dim=-1)
    weights = masses / total[:, None]
    position = torch.sum(entity.data.body_com_pos_w * weights[..., None], dim=1)
    velocity = torch.sum(entity.data.body_com_lin_vel_w * weights[..., None], dim=1)
    return position, velocity, total


def _precompute_statics() -> tuple[Any, tuple[MujocoStaticEquilibrium, ...]]:
    model = build_assembled_spec(with_ground=True).compile()
    model.opt.timestep = 0.002
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    solutions: list[MujocoStaticEquilibrium] = []
    qpos_seed = None
    for gradient in SLOPE_GRADIENTS:
        solution = solve_mujoco_static_equilibrium(model, gradient, qpos_seed=qpos_seed)
        solutions.append(solution)
        qpos_seed = solution.qpos
    result = tuple(solutions)
    if len(result) != SLOPE_COUNT or tuple(item.gradient for item in result) != SLOPE_GRADIENTS:
        raise RuntimeError("MuJoCo static library must contain exactly the configured 19 slopes")
    return model, result


def initialize_mjlab_task(env: Any, env_ids: torch.Tensor | None, cfg: MjlabTaskRuntimeCfg) -> None:
    """Assign slopes, solve all equilibria, and allocate policy-rate state."""

    del env_ids
    if cfg.play:
        slots, _, _ = balanced_slope_assignment(
            env.num_envs, device=env.device, shuffle=False
        )
    else:
        slots, _, _ = weighted_slope_assignment(
            env.num_envs, device=env.device, shuffle=cfg.shuffle_slopes
        )
    assign_mjlab_slope_slots(env, slots)

    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    env.policy_joint_ids = _ids(robot, "joints", G1_JOINT_ORDER)
    env.policy_actuator_ids = _ids(robot, "actuators", G1_JOINT_ORDER)
    env.wheel_joint_ids = _ids(cart, "joints", WHEEL_JOINT_NAMES)
    env.wheel_body_ids = _ids(cart, "bodies", WHEEL_LINK_NAMES)
    env.hitch_site_ids = _ids(cart, "sites", HITCH_SITE_NAMES)
    env.grasp_site_ids = _ids(robot, "sites", GRASP_SITE_NAMES)
    env.foot_body_ids = _ids(robot, "bodies", ("left_ankle_roll_link", "right_ankle_roll_link"))
    env.torso_body_id = int(_ids(robot, "bodies", ("torso_link",))[0])
    env.pelvis_body_id = int(_ids(robot, "bodies", ("pelvis",))[0])
    env.cart_base_body_id = int(_ids(cart, "bodies", (BASE_LINK_NAME,))[0])

    if tuple(robot.joint_names) != G1_JOINT_ORDER:
        raise RuntimeError("fixed-gripper MuJoCo robot must expose exactly the 29 policy joints")
    env._mujoco_static_model, env._mujoco_static_equilibria = _precompute_statics()
    model = env._mujoco_static_model
    env.static_joint_position_table = torch.as_tensor(
        np.stack(
            [
                [
                    item.qpos[int(model.joint(f"robot/{name}").qposadr[0])]
                    for name in G1_JOINT_ORDER
                ]
                for item in env._mujoco_static_equilibria
            ]
        ),
        device=env.device,
        dtype=torch.float32,
    )
    env.static_q_ref_table = torch.as_tensor(
        np.stack([item.joint_position_target for item in env._mujoco_static_equilibria]),
        device=env.device,
        dtype=torch.float32,
    )
    env.static_actuator_torque_table = torch.as_tensor(
        np.stack([item.joint_actuator_torque for item in env._mujoco_static_equilibria]),
        device=env.device,
        dtype=torch.float32,
    )
    env.static_fat2_table = torch.tensor(
        [item.fat2_reference_angle for item in env._mujoco_static_equilibria],
        device=env.device,
        dtype=torch.float32,
    )

    env.runtime_cfg = cfg
    env.command_state = CommandState.zeros(env.num_envs, device=env.device)
    env.path_state = PathTrackingState.zeros(env.num_envs, device=env.device)
    env.rickshaw_state = RickshawRuntimeState.zeros(env.num_envs, device=env.device)
    env.stability_state = StabilityState.zeros(env.num_envs, device=env.device)
    env.observation_history_state = ObservationHistoryState.zeros(
        env.num_envs, history_length=cfg.history_length, device=env.device
    )
    env.teacher_dynamic_history_state = ObservationHistoryState.zeros(
        env.num_envs,
        history_length=cfg.history_length,
        observation_dim=TEACHER_DYNAMIC_DIM,
        device=env.device,
    )
    env.rickshaw_pose_cfg = cfg.rickshaw_pose
    env.fat2_wrench_consistency_state = WrenchConsistencyState.zeros(
        env.num_envs, cfg.fat2.wrench_consistency_window_steps, device=env.device
    )
    env.fat2_com_radius_state = FAT2ComRadiusState.initialized(
        env.num_envs,
        cfg.fat2.wrench_consistency_window_steps,
        cfg.fat2.com_radius,
        device=env.device,
    )
    env.fat_com_radius = env.fat2_com_radius_state.filtered_radius
    env.fat_com_radius_raw = env.fat_com_radius.clone()
    zeros = torch.zeros(env.num_envs, device=env.device)
    env.analytic_force_state = AnalyticHandleForceState.initialized(zeros, zeros)
    env.zmp_kinematic_state = ZMPKinematicState.initialized(zeros, zeros)
    env.cart_previous_com_velocity_w = torch.zeros((env.num_envs, 3), device=env.device)
    env.cart_interaction_wrench_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.last_rolling_force_w = torch.zeros((env.num_envs, 2, 3), device=env.device)
    env.policy_robot_speed_s = zeros.clone()
    env.policy_robot_speed_l = zeros.clone()
    env.policy_robot_velocity_n = zeros.clone()
    env._mjlab_physical_state_step = -1
    env._mjlab_observation_state_step = -1


@requires_model_fields(
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "geom_friction",
    "dof_damping",
    recompute=RecomputeLevel.set_const,
)
def initialize_mjlab_domain(env: Any, env_ids: torch.Tensor | None, cfg: MjlabTaskRuntimeCfg) -> None:
    """Sample the nine startup-fixed physics parameters and write MuJoCo fields."""

    del env_ids
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    sampled = sample_domain_parameters(cfg.domain, env.num_envs, device=env.device)
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    torso_global = robot.indexing.body_ids[env.torso_body_id]
    base_global = cart.indexing.body_ids[env.cart_base_body_id]
    env_grid = ids

    default_robot_mass = env.sim.get_default_field("body_mass")[robot.indexing.body_ids]
    env._default_robot_masses_cpu = default_robot_mass[None, :].repeat(env.num_envs, 1).cpu()
    torso_mass = default_robot_mass[env.torso_body_id] + sampled["torso.mass_delta"]
    env.sim.model.body_mass[env_grid, torso_global] = torso_mass
    env.torso_mass_delta = sampled["torso.mass_delta"]
    env.effective_torso_mass = torso_mass

    default_mass = env.sim.get_default_field("body_mass")[base_global]
    default_com = env.sim.get_default_field("body_ipos")[base_global]
    default_principal = env.sim.get_default_field("body_inertia")[base_global]
    default_quat = env.sim.get_default_field("body_iquat")[base_global]
    rotation = matrix_from_quat(default_quat)
    default_inertia = rotation @ torch.diag(default_principal) @ rotation.mT
    payload_mass = sampled["payload.mass"]
    payload_com = torch.stack(
        (sampled["payload.com.x"], sampled["payload.com.y"], sampled["payload.com.z"]), dim=-1
    )
    total_mass, total_com, total_inertia = combine_mass_properties(
        default_mass.expand(env.num_envs),
        default_com.expand(env.num_envs, -1),
        default_inertia.expand(env.num_envs, -1, -1),
        payload_mass,
        payload_com,
        torch.zeros((env.num_envs, 3, 3), device=env.device),
    )
    principal, axes = torch.linalg.eigh(total_inertia)
    reflected = torch.linalg.det(axes) < 0
    axes[reflected, :, 2] *= -1
    env.sim.model.body_mass[env_grid, base_global] = total_mass
    env.sim.model.body_ipos[env_grid, base_global] = total_com
    env.sim.model.body_inertia[env_grid, base_global] = principal
    env.sim.model.body_iquat[env_grid, base_global] = quat_from_matrix(axes)
    env._payload_mass = payload_mass
    env._payload_com = payload_com

    friction = sampled["terrain.friction"]
    geom_ids = torch.cat(
        (
            robot.indexing.geom_ids,
            cart.indexing.geom_ids,
            env.scene.terrain.indexing.geom_ids,
        )
    ).to(dtype=torch.long)
    friction_grid, geom_grid = torch.meshgrid(ids, geom_ids, indexing="ij")
    env.sim.model.geom_friction[friction_grid, geom_grid, 0] = friction[:, None]
    env.terrain_friction = friction
    wheel_dof_ids = cart.indexing.joint_v_adr[env.wheel_joint_ids].to(
        dtype=torch.long
    )
    dof_grid, wheel_grid = torch.meshgrid(ids, wheel_dof_ids, indexing="ij")
    wheel_damping = torch.stack((sampled["wheel.left_damping"], sampled["wheel.right_damping"]), dim=-1)
    env.sim.model.dof_damping[dof_grid, wheel_grid] = wheel_damping
    env._wheel_damping = wheel_damping

    env.c_rr = sampled["rolling_resistance.c_rr"]
    cart_mass = RICKSHAW_TOTAL_MASS + payload_mass
    nominal_com = torch.tensor(RICKSHAW_CENTER_OF_MASS, device=env.device)
    cart_com = (
        RICKSHAW_TOTAL_MASS * nominal_com[None, :] + payload_mass[:, None] * payload_com
    ) / cart_mass[:, None]
    env.effective_cart_mass_com = torch.cat((cart_mass[:, None], cart_com), dim=-1)
    wheel_radius = torch.full((env.num_envs, 2), RICKSHAW_URDF_SPEC.wheel_radius, device=env.device)
    wheel_spin = torch.full(
        (env.num_envs, 2), RICKSHAW_URDF_SPEC.wheel_inertia_diagonal[1], device=env.device
    )
    axle_z = RICKSHAW_URDF_SPEC.wheel_radius
    env.rickshaw_mass_properties = RickshawMassProperties(
        m_cart=cart_mass,
        com_x_from_axle=cart_com[:, 0],
        com_z_from_axle=cart_com[:, 2] - axle_z,
        pitch_inertia_about_axle=torch.full(
            (env.num_envs,), float(cfg.domain.calibration["rickshaw.pitch_inertia_about_axle"]), device=env.device
        ),
        m_eff=effective_cart_mass(cart_mass, wheel_spin, wheel_radius),
        b_eff=effective_wheel_damping(wheel_damping, wheel_radius),
        handle_x_from_axle=torch.full((env.num_envs,), RICKSHAW_URDF_SPEC.hitch_x, device=env.device),
        handle_z_from_axle=torch.full(
            (env.num_envs,), RICKSHAW_URDF_SPEC.hitch_z - axle_z, device=env.device
        ),
    )
    _update_teacher_static_domain(env, cfg.domain, sampled)


def _transform_pose(env: Any, local_pose: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    tangent = env.path_tangent_w[env_ids]
    lateral = env.path_lateral_w[env_ids]
    normal = env.path_normal_w[env_ids]
    position = (
        env.scene.env_origins[env_ids]
        + tangent * local_pose[:, 0:1]
        + lateral * local_pose[:, 1:2]
        + normal * local_pose[:, 2:3]
    )
    quaternion = quat_mul(env.slope_quat_w[env_ids], local_pose[:, 3:7])
    return torch.cat((position, quaternion), dim=-1)


def reset_from_mujoco_statics(env: Any, env_ids: torch.Tensor | None) -> None:
    """Reset each environment from its own precomputed slope equilibrium."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    slots = env.slope_slot[env_ids]
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    model = env._mujoco_static_model

    def qadr(name: str) -> int:
        return int(model.joint(name).qposadr[0])

    qpos = torch.stack(
        [torch.as_tensor(env._mujoco_static_equilibria[int(slot)].qpos) for slot in slots.cpu()], dim=0
    ).to(device=env.device, dtype=torch.float32)
    robot_root = qadr("robot/floating_base_joint")
    cart_root = qadr("rickshaw/floating_base_joint")
    robot_pose = _transform_pose(env, qpos[:, robot_root : robot_root + 7], env_ids)
    cart_pose = _transform_pose(env, qpos[:, cart_root : cart_root + 7], env_ids)
    zeros6 = torch.zeros((env_ids.numel(), 6), device=env.device)
    robot.write_root_link_pose_to_sim(robot_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(zeros6, env_ids=env_ids)
    cart.write_root_link_pose_to_sim(cart_pose, env_ids=env_ids)
    cart.write_root_link_velocity_to_sim(zeros6, env_ids=env_ids)

    robot_joint_pos = env.static_joint_position_table[slots]
    robot.write_joint_state_to_sim(robot_joint_pos, torch.zeros_like(robot_joint_pos), env_ids=env_ids)
    wheel_pos = torch.stack(
        (qpos[:, qadr("rickshaw/left_wheel_joint")], qpos[:, qadr("rickshaw/right_wheel_joint")]), dim=-1
    )
    cart.write_joint_state_to_sim(
        wheel_pos, torch.zeros_like(wheel_pos), joint_ids=env.wheel_joint_ids, env_ids=env_ids
    )
    q_ref = env.static_q_ref_table[slots]
    env.action_manager.get_term("joint_pos").set_reference(q_ref, env_ids)
    # Entity.set_joint_position_target uses direct tensor indexing, unlike the
    # write_* state helpers.  Explicitly form the env-by-joint outer product.
    robot.set_joint_position_target(
        q_ref,
        joint_ids=env.policy_joint_ids.unsqueeze(0),
        env_ids=env_ids.unsqueeze(1),
    )
    env.command_state.reset(env_ids)
    resample_speed_command(env, env_ids, env.runtime_cfg.command)
    env.path_state.lateral_error[env_ids] = 0.0
    env.path_state.heading_error[env_ids] = 0.0
    env.rickshaw_state.wheel_normal_force[env_ids] = 0.0
    env.rickshaw_state.two_wheel_contact[env_ids] = False
    env.rickshaw_state.connection_wrench_w[env_ids] = 0.0
    env.rickshaw_state.connection_truth_wrench_w[env_ids] = 0.0
    env.rickshaw_state.hand_force_w[env_ids] = 0.0
    env.rickshaw_state.hand_torque_w[env_ids] = 0.0
    env.stability_state.theta_fat[env_ids] = env.static_fat2_table[slots]
    env.stability_state.fat_valid[env_ids] = False
    env.stability_state.zmp_valid[env_ids] = False
    env.observation_history_state.reset(env_ids)
    env.teacher_dynamic_history_state.reset(env_ids)
    env.fat2_wrench_consistency_state.reset(env_ids)
    env.fat2_com_radius_state.reset(env_ids)
    env.cart_previous_com_velocity_w[env_ids] = 0.0
    env.cart_interaction_wrench_valid[env_ids] = False
    env.last_rolling_force_w[env_ids] = 0.0
    env._mjlab_physical_state_step = -1
    env._mjlab_observation_state_step = -1


def ensure_mjlab_physical_state(env: Any) -> None:
    """Refresh all task state exactly once per policy step."""

    step = int(env.common_step_counter)
    if env._mjlab_physical_state_step == step:
        return
    env._mjlab_physical_state_step = step
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    origin = env.scene.env_origins

    robot_velocity = robot.data.root_link_lin_vel_w
    env.policy_robot_speed_s[:] = torch.sum(robot_velocity * env.path_tangent_w, dim=-1)
    env.policy_robot_speed_l[:] = torch.sum(robot_velocity * env.path_lateral_w, dim=-1)
    env.policy_robot_velocity_n[:] = torch.sum(robot_velocity * env.path_normal_w, dim=-1)
    lateral_error, heading_error = compute_path_tracking_errors(
        robot.data.root_link_pos_w,
        cart.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        origin,
        env.path_tangent_w,
        env.path_lateral_w,
    )
    env.path_state.lateral_error[:] = lateral_error
    env.path_state.heading_error[:] = heading_error

    hitch_position = torch.mean(cart.data.site_pos_w[:, env.hitch_site_ids], dim=1)
    hitch_velocity = torch.mean(cart.data.site_lin_vel_w[:, env.hitch_site_ids], dim=1)
    env.rickshaw_state.hitch_height[:] = torch.sum((hitch_position - origin) * env.path_normal_w, dim=-1)
    env.rickshaw_state.hitch_vertical_speed[:] = torch.sum(hitch_velocity * env.path_normal_w, dim=-1)
    pitch = rickshaw_pitch_from_quaternion(
        cart.data.root_link_quat_w, env.path_tangent_w, env.path_normal_w
    )
    env.rickshaw_state.pitch[:] = pitch
    grasp_positions = robot.data.site_pos_w[:, env.grasp_site_ids]
    hitch_positions = cart.data.site_pos_w[:, env.hitch_site_ids]
    connection_position_error = torch.linalg.vector_norm(
        grasp_positions - hitch_positions, dim=-1
    )
    env.rickshaw_state.connection_residual[:] = torch.amax(
        connection_position_error, dim=-1
    )

    wheel_sensor = env.scene["wheel_contacts"]
    wheel_force = wheel_sensor.data.force
    mean_wheel_force = torch.mean(wheel_sensor.data.force_history, dim=2)
    wheel_normal = torch.clamp(torch.sum(wheel_force * env.path_normal_w[:, None, :], dim=-1), min=0.0)
    env.rickshaw_state.wheel_normal_force[:] = wheel_normal
    env.rickshaw_state.two_wheel_contact[:] = torch.all(wheel_normal > 1.0, dim=-1)
    cart_com, cart_velocity, cart_mass = _body_mass_kinematics(env, "rickshaw")
    acceleration = (cart_velocity - env.cart_previous_com_velocity_w) / env.step_dt
    gravity = torch.tensor((0.0, 0.0, -9.81), device=env.device)
    force_on_cart = (
        cart_mass[:, None] * acceleration
        - cart_mass[:, None] * gravity
        - torch.sum(mean_wheel_force, dim=1)
        - torch.sum(env.last_rolling_force_w, dim=1)
    )
    valid_force = step > 0
    env.cart_interaction_wrench_valid[:] = valid_force & torch.all(torch.isfinite(force_on_cart), dim=-1)
    force_on_cart = torch.where(
        env.cart_interaction_wrench_valid[:, None], force_on_cart, torch.zeros_like(force_on_cart)
    )
    env.rickshaw_state.hand_force_w[:] = -force_on_cart
    env.rickshaw_state.hand_torque_w.zero_()
    env.rickshaw_state.connection_truth_wrench_w[..., :3] = 0.5 * env.rickshaw_state.hand_force_w[:, None, :]
    env.rickshaw_state.connection_truth_wrench_w[..., 3:] = 0.0
    env.rickshaw_state.connection_wrench_w[:] = env.rickshaw_state.connection_truth_wrench_w
    env.cart_previous_com_velocity_w[:] = cart_velocity

    foot_force = env.scene["robot_contacts"].data.force
    foot_contact = torch.linalg.vector_norm(foot_force, dim=-1) > 1.0
    points, mask, support_center = foot_support_polygon(
        robot.data.body_link_pos_w[:, env.foot_body_ids],
        robot.data.body_link_quat_w[:, env.foot_body_ids],
        foot_contact,
        origin,
        env.path_tangent_w,
        env.path_lateral_w,
        foot_half_length=env.runtime_cfg.support.foot_half_length,
        foot_half_width=env.runtime_cfg.support.foot_half_width,
        foot_center_offset_x=env.runtime_cfg.support.foot_center_offset_x,
    )
    env.stability_state.support_points_sy[:] = points
    env.stability_state.support_point_mask[:] = mask
    env.stability_state.support_center_w[:] = support_center

    cart_speed = torch.sum(cart_velocity * env.path_tangent_w, dim=-1)
    update_analytic_handle_force_state(
        env.analytic_force_state,
        cart_speed,
        pitch,
        env.gamma,
        env.c_rr,
        wheel_normal,
        env.rickshaw_mass_properties,
        env.step_dt,
        env.runtime_cfg.analytic_force,
    )
    analytic_sn = torch.stack((env.analytic_force_state.t_s, env.analytic_force_state.t_n), dim=-1)
    measured_sn = torch.stack(
        (
            torch.sum(force_on_cart * env.path_tangent_w, dim=-1),
            torch.sum(force_on_cart * env.path_normal_w, dim=-1),
        ),
        dim=-1,
    )
    consistent, relative_error, filtered = update_wrench_consistency_state(
        env.fat2_wrench_consistency_state,
        analytic_sn,
        measured_sn,
        env.analytic_force_state.valid & env.cart_interaction_wrench_valid,
        relative_tolerance=env.runtime_cfg.fat2.wrench_consistency_relative_tolerance,
        absolute_floor_n=env.runtime_cfg.fat2.wrench_consistency_absolute_floor_n,
    )
    env.stability_state.fat_wrench_consistent[:] = consistent
    env.stability_state.fat_wrench_relative_error[:] = relative_error
    robot_com, robot_com_velocity, robot_mass = _body_mass_kinematics(env, "robot")
    radius = sagittal_com_radius(robot_com, support_center, env.path_tangent_w, env.path_normal_w)
    env.fat_com_radius_raw[:] = radius
    env.fat2_com_radius_state.update(
        radius,
        torch.any(mask, dim=-1),
        minimum=env.runtime_cfg.fat2.com_radius_bounds[0],
        maximum=env.runtime_cfg.fat2.com_radius_bounds[1],
    )
    handle_delta = hitch_position - support_center
    handle_s = torch.sum(handle_delta * env.path_tangent_w, dim=-1)
    handle_n = torch.sum(handle_delta * env.path_normal_w, dim=-1)
    env.stability_state.theta_fat[:] = fat2_reference_angle(
        handle_s,
        handle_n,
        -filtered[:, 0],
        -filtered[:, 1],
        robot_mass,
        env.fat_com_radius,
        env.runtime_cfg.fat2.theta_max,
    )
    env.stability_state.fat_valid[:] = consistent & torch.any(mask, dim=-1)
    env.stability_state.torso_pitch[:] = torso_pitch_from_world_vertical(
        robot.data.body_link_quat_w[:, env.torso_body_id], env.path_tangent_w
    )

    relative_com = robot_com - origin
    com_s = torch.sum(relative_com * env.path_tangent_w, dim=-1)
    com_n = torch.sum(relative_com * env.path_normal_w, dim=-1)
    velocity_s = torch.sum(robot_com_velocity * env.path_tangent_w, dim=-1)
    velocity_n = torch.sum(robot_com_velocity * env.path_normal_w, dim=-1)
    acceleration_s = filtered_first_derivative(
        velocity_s, env.zmp_kinematic_state.tangential_velocity_filter, env.step_dt
    )
    acceleration_n = filtered_first_derivative(
        velocity_n, env.zmp_kinematic_state.normal_velocity_filter, env.step_dt
    )
    handle_origin = hitch_position - origin
    hs = torch.sum(handle_origin * env.path_tangent_w, dim=-1)
    hn = torch.sum(handle_origin * env.path_normal_w, dim=-1)
    hand_force = env.rickshaw_state.hand_force_w
    fs = torch.sum(hand_force * env.path_tangent_w, dim=-1)
    fn = torch.sum(hand_force * env.path_normal_w, dim=-1)
    zmp_s, _, reaction_n, dynamics_valid = slope_zmp(
        com_s,
        com_n,
        acceleration_s,
        acceleration_n,
        hs,
        hn,
        fs,
        fn,
        torch.zeros_like(fs),
        robot_mass,
        env.gamma,
        min_ground_reaction=env.runtime_cfg.zmp.min_ground_reaction,
    )
    support_relative = support_center - origin
    support_y = torch.sum(support_relative * env.path_lateral_w, dim=-1)
    margin, polygon_valid = convex_support_margin(points, torch.stack((zmp_s, support_y), dim=-1), mask)
    zmp_valid = dynamics_valid & polygon_valid & env.cart_interaction_wrench_valid
    env.stability_state.zmp_s[:] = zmp_s
    env.stability_state.ground_reaction_normal[:] = reaction_n
    env.stability_state.zmp_margin[:] = torch.where(zmp_valid, margin, torch.zeros_like(margin))
    env.stability_state.zmp_valid[:] = zmp_valid

    rolling_force, _, _ = rolling_resistance_wrench(
        cart.data.body_link_lin_vel_w[:, env.wheel_body_ids],
        wheel_force,
        env.path_tangent_w,
        env.path_normal_w,
        env.c_rr,
        wheel_normal,
        dt=env.step_dt,
    )
    env.last_rolling_force_w[:] = rolling_force
    cart.data.write_external_wrench(
        rolling_force,
        torch.zeros_like(rolling_force),
        body_ids=env.wheel_body_ids.tolist(),
    )


def advance_mjlab_policy_state(env: Any, env_ids: torch.Tensor | None, cfg: MjlabTaskRuntimeCfg) -> None:
    """Advance commands and the online FAT2/ZMP state once per policy step."""

    del env_ids
    env.command_state.resampling_elapsed_s += env.step_dt
    due = torch.nonzero(
        env.command_state.resampling_elapsed_s >= cfg.command.resampling_time_s - 1.0e-9,
        as_tuple=False,
    ).flatten()
    if due.numel():
        resample_speed_command(env, due, cfg.command)
    from .mdp.dynamics import SpeedReferenceCfg, update_speed_reference

    update_speed_reference(
        env.command_state,
        env.command_state.v_sample,
        env.step_dt,
        SpeedReferenceCfg(
            acceleration_limit=cfg.speed_acceleration_limit,
            jerk_limit=cfg.speed_jerk_limit,
        ),
    )
    ensure_mjlab_physical_state(env)


__all__ = [
    "MjlabTaskRuntimeCfg",
    "advance_mjlab_policy_state",
    "assign_mjlab_slope_slots",
    "ensure_mjlab_physical_state",
    "initialize_mjlab_domain",
    "initialize_mjlab_task",
    "reset_from_mujoco_statics",
]
