"""Fixed-contact statics shared by reset generation and runtime loading.

The hand wrench convention is robot-on-cart.  Wheel contact forces are
ground-on-cart.  Components use the slope frame ``(s, l, n)`` where ``s`` is
the path tangent, ``l`` is lateral (left), and ``n`` is the terrain normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .configuration import (
    G1_JOINT_ORDER,
)
from .assets.g1_dex1 import G1_JOINT_EFFORT_LIMITS, G1_JOINT_STIFFNESS


@dataclass(frozen=True)
class FixedContactStaticSolution:
    """Scalar fixed-contact solution for two hitches and two passive wheels."""

    handle_wrenches_sln: tuple[tuple[float, ...], tuple[float, ...]]
    wheel_contact_forces_sln: tuple[tuple[float, ...], tuple[float, ...]]
    cart_force_residual_sln: tuple[float, float, float]
    cart_moment_residual_sln: tuple[float, float, float]


@dataclass(frozen=True)
class MujocoStaticEquilibrium:
    """One MuJoCo equilibrium used directly by the reset event."""

    gradient: float
    qpos: np.ndarray
    joint_position_target: np.ndarray
    joint_actuator_torque: np.ndarray
    fat2_reference_angle: float
    equality_position_error: float
    support_height_error: float
    acceleration_error: float
    actuator_torque_error: float
    actuator_torque_ratio: float


@dataclass(frozen=True)
class MujocoStaticSolverCfg:
    max_nfev: int = 300
    forward_max_nfev: int = 300
    position_scale: float = 0.003
    support_scale: float = 0.0005
    contact_penetration: float = -0.001
    unactuated_force_scale: float = 1.0
    torque_limit_fraction: float = 0.8
    posture_scale: float = 1.0
    fat2_scale: float = 0.12
    position_tolerance: float = 0.003
    support_tolerance: float = 0.003
    acceleration_tolerance: float = 6.0
    actuator_torque_tolerance: float = 1.0e-6
    actuator_torque_ratio_tolerance: float = 1.0
    robot_mass: float = 34.1299349
    robot_com_radius: float = 0.715092420262594
    theta_max: float = 0.8


def fixed_contact_static_components(
    *,
    gravity_tangent: Any,
    gravity_normal: Any,
    com_s: Any,
    com_l: Any,
    com_n: Any,
    handle_s: Any,
    handle_n: Any,
    hitch_half_width: float,
    wheel_track: float,
    pitch_torque_on_robot: Any = 0.0,
) -> tuple[tuple[tuple[Any, ...], tuple[Any, ...]], tuple[tuple[Any, ...], tuple[Any, ...]]]:
    """Allocate the closed-chain static wrench using passive-wheel mechanics.

    The wheel bearings are passive, so a zero-speed equilibrium cannot rely on
    wheel tangent force: the two hitches carry the cart's full downslope load.
    Equal normal hand loading is selected from the redundant lateral load
    family; the wheels carry the lateral-CoM roll moment.  A lateral CoM offset
    gives unequal hand tangent forces to cancel yaw.  The corresponding joint
    torque is affine in ``F_s,left - F_s,right`` and is stored as a separate
    reset-library basis.

    Inputs may be floats, NumPy arrays, or Torch tensors.  The function uses
    only elementwise arithmetic so the exact same equations can be evaluated
    offline and for a batch of randomized runtime environments.
    """

    zero = gravity_tangent * 0.0
    hand_tangent_total = gravity_tangent
    gravity_pitch_moment = com_s * gravity_normal - com_n * gravity_tangent
    hand_normal_total = (handle_n * hand_tangent_total + gravity_pitch_moment - pitch_torque_on_robot) / handle_s

    hand_tangent_difference = com_l * gravity_tangent / hitch_half_width
    hand_normal_difference = zero
    wheel_normal_total = gravity_normal - hand_normal_total
    wheel_normal_difference = 2.0 * com_l * gravity_normal / wheel_track

    # Preserve the batch type when callers pass the physically required scalar
    # zero for the crossbar's free pitch axis.
    hand_torque_on_cart = zero - pitch_torque_on_robot
    left_hand = (
        0.5 * (hand_tangent_total + hand_tangent_difference),
        zero,
        0.5 * (hand_normal_total + hand_normal_difference),
        zero,
        0.5 * hand_torque_on_cart,
        zero,
    )
    right_hand = (
        0.5 * (hand_tangent_total - hand_tangent_difference),
        zero,
        0.5 * (hand_normal_total - hand_normal_difference),
        zero,
        0.5 * hand_torque_on_cart,
        zero,
    )
    left_wheel = (
        zero,
        zero,
        0.5 * (wheel_normal_total + wheel_normal_difference),
    )
    right_wheel = (
        zero,
        zero,
        0.5 * (wheel_normal_total - wheel_normal_difference),
    )
    return (left_hand, right_hand), (left_wheel, right_wheel)


def solve_fixed_contact_statics(
    *,
    mass: float,
    gradient: float,
    com_from_axle_sln: tuple[float, float, float],
    handle_from_axle_sn: tuple[float, float],
    hitch_half_width: float,
    wheel_track: float,
    pitch_torque_on_robot: float = 0.0,
    gravity: float = 9.81,
) -> FixedContactStaticSolution:
    """Return and independently verify one scalar fixed-contact equilibrium."""

    scalars = (
        mass,
        gradient,
        *com_from_axle_sln,
        *handle_from_axle_sn,
        hitch_half_width,
        wheel_track,
        pitch_torque_on_robot,
        gravity,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("fixed-contact statics inputs must be finite")
    if mass <= 0.0 or gravity <= 0.0:
        raise ValueError("mass and gravity must be positive")
    if handle_from_axle_sn[0] <= 0.0:
        raise ValueError("handle tangent offset from the axle must be positive")
    if hitch_half_width <= 0.0 or wheel_track <= 0.0:
        raise ValueError("hitch half-width and wheel track must be positive")

    gamma = math.atan(gradient)
    gravity_tangent = mass * gravity * math.sin(gamma)
    gravity_normal = mass * gravity * math.cos(gamma)
    hand_wrenches, wheel_forces = fixed_contact_static_components(
        gravity_tangent=gravity_tangent,
        gravity_normal=gravity_normal,
        com_s=com_from_axle_sln[0],
        com_l=com_from_axle_sln[1],
        com_n=com_from_axle_sln[2],
        handle_s=handle_from_axle_sn[0],
        handle_n=handle_from_axle_sn[1],
        hitch_half_width=hitch_half_width,
        wheel_track=wheel_track,
        pitch_torque_on_robot=pitch_torque_on_robot,
    )
    hand_wrenches = tuple(tuple(float(value) for value in row) for row in hand_wrenches)
    wheel_forces = tuple(tuple(float(value) for value in row) for row in wheel_forces)

    gravity_force = (-gravity_tangent, 0.0, -gravity_normal)
    force_residual = tuple(
        gravity_force[axis] + sum(wrench[axis] for wrench in hand_wrenches) + sum(force[axis] for force in wheel_forces)
        for axis in range(3)
    )
    com_s, com_l, com_n = com_from_axle_sln
    handle_s, handle_n = handle_from_axle_sn
    gravity_moment = (
        -com_l * gravity_normal,
        com_s * gravity_normal - com_n * gravity_tangent,
        com_l * gravity_tangent,
    )
    moment_residual = [float(value) for value in gravity_moment]
    for lateral, wrench in zip((hitch_half_width, -hitch_half_width), hand_wrenches, strict=True):
        force_s, force_l, force_n, torque_s, torque_l, torque_n = wrench
        moment_residual[0] += lateral * force_n - handle_n * force_l + torque_s
        moment_residual[1] += handle_n * force_s - handle_s * force_n + torque_l
        moment_residual[2] += handle_s * force_l - lateral * force_s + torque_n
    for lateral, force in zip((0.5 * wheel_track, -0.5 * wheel_track), wheel_forces, strict=True):
        force_s, _force_l, force_n = force
        moment_residual[0] += lateral * force_n
        moment_residual[2] += -lateral * force_s

    return FixedContactStaticSolution(
        handle_wrenches_sln=hand_wrenches,  # type: ignore[arg-type]
        wheel_contact_forces_sln=wheel_forces,  # type: ignore[arg-type]
        cart_force_residual_sln=tuple(float(value) for value in force_residual),
        cart_moment_residual_sln=tuple(moment_residual),
    )


def fat2_reference_angle_scalar(
    *,
    handle_s: float,
    handle_n: float,
    hand_force_s: float,
    hand_force_n: float,
    robot_mass: float,
    com_radius: float,
    theta_max: float,
) -> float:
    """Full-wrench FAT2 prior used by both static initialization and reward."""

    hand_moment = handle_s * hand_force_n - handle_n * hand_force_s
    ratio = hand_moment / (robot_mass * 9.81 * com_radius)
    limit = math.sin(theta_max)
    return math.asin(max(-limit, min(limit, ratio)))


def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _rpy_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        (
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        )
    )


def _joint_qpos_address(model: Any, name: str) -> int:
    return int(model.joint(name).qposadr[0])


def _joint_dof_address(model: Any, name: str) -> int:
    return int(model.joint(name).dofadr[0])


def _nominal_qpos(model: Any) -> np.ndarray:
    """Produce one deterministic IK seed; it is never used as a reset state."""

    import mujoco

    qpos = model.qpos0.copy()
    robot_root = _joint_qpos_address(model, "robot/floating_base_joint")
    rickshaw_root = _joint_qpos_address(model, "rickshaw/floating_base_joint")
    qpos[robot_root : robot_root + 7] = (0.0, 0.0, 0.72, 1.0, 0.0, 0.0, 0.0)
    values = {
        "hip_pitch": -0.32,
        "knee": 0.92,
        "ankle_pitch": -0.34,
        "left_shoulder_pitch": 0.33,
        "right_shoulder_pitch": 0.33,
        "left_shoulder_roll": -0.08,
        "right_shoulder_roll": 0.08,
        "left_shoulder_yaw": 0.70,
        "right_shoulder_yaw": -0.70,
        "left_elbow": 0.22,
        "right_elbow": 0.22,
        "left_wrist_roll": -1.18,
        "right_wrist_roll": 1.18,
        "left_wrist_pitch": 0.76,
        "right_wrist_pitch": 0.76,
        "left_wrist_yaw": -1.50,
        "right_wrist_yaw": 1.50,
    }
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith("robot/") or name.endswith("floating_base_joint"):
            continue
        short_name = name.removeprefix("robot/").removesuffix("_joint")
        value = next((target for pattern, target in values.items() if pattern in short_name), 0.0)
        qpos[int(model.jnt_qposadr[joint_id])] = value

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    grasp_midpoint = 0.5 * (data.site("robot/left_grasp_site").xpos + data.site("robot/right_grasp_site").xpos)
    hitch_midpoint = np.array((1.664929, 0.0, 0.105747))
    wheel_radius = 0.3
    low, high = 0.0, 0.7
    for _ in range(48):
        angle = 0.5 * (low + high)
        height = (
            wheel_radius * (1.0 - math.cos(angle))
            + math.sin(angle) * hitch_midpoint[0]
            + math.cos(angle) * hitch_midpoint[2]
        )
        if height < grasp_midpoint[2]:
            low = angle
        else:
            high = angle
    angle = 0.5 * (low + high)
    quat = _quat_from_rpy(np.array((0.0, -angle, 0.0)))
    rotation = np.array(
        (
            (math.cos(angle), 0.0, -math.sin(angle)),
            (0.0, 1.0, 0.0),
            (math.sin(angle), 0.0, math.cos(angle)),
        )
    )
    cart_position = grasp_midpoint - rotation @ hitch_midpoint
    qpos[rickshaw_root : rickshaw_root + 7] = (*tuple(cart_position), *tuple(quat))
    return qpos


def solve_mujoco_static_equilibrium(
    model: Any,
    gradient: float,
    *,
    cfg: MujocoStaticSolverCfg | None = None,
    qpos_seed: np.ndarray | None = None,
) -> MujocoStaticEquilibrium:
    """Solve a fixed-contact equilibrium with MuJoCo dynamics.

    The optimization has no dynamic settling phase. Inverse dynamics first solves the
    fixed-contact pose under the grasp, support, hardware-limit, and FAT2 constraints.
    A second forward-dynamics solve then finds and validates the bounded actuator torque.
    """

    import mujoco
    from scipy.optimize import least_squares

    if cfg is None:
        cfg = MujocoStaticSolverCfg()
    if not math.isfinite(gradient):
        raise ValueError("gradient must be finite")
    q_seed = _nominal_qpos(model) if qpos_seed is None else np.asarray(qpos_seed, dtype=float).copy()
    if q_seed.shape != (model.nq,):
        raise ValueError(f"qpos_seed must have shape ({model.nq},)")

    robot_root_q = _joint_qpos_address(model, "robot/floating_base_joint")
    cart_root_q = _joint_qpos_address(model, "rickshaw/floating_base_joint")
    robot_root_v = _joint_dof_address(model, "robot/floating_base_joint")
    cart_root_v = _joint_dof_address(model, "rickshaw/floating_base_joint")
    robot_joint_ids = [
        index
        for index in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "").startswith("robot/")
        and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    ]
    robot_joint_q = np.array([model.jnt_qposadr[index] for index in robot_joint_ids], dtype=int)
    robot_joint_v = np.array([model.jnt_dofadr[index] for index in robot_joint_ids], dtype=int)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "" for index in robot_joint_ids]
    if tuple(name.removeprefix("robot/") for name in joint_names) != G1_JOINT_ORDER:
        raise ValueError("MuJoCo robot joint order does not match the 29-joint policy contract")
    stiffness = np.asarray(G1_JOINT_STIFFNESS)
    effort_limits = np.asarray(G1_JOINT_EFFORT_LIMITS)
    robot_actuator_ids = np.array(
        [
            actuator_id
            for joint_id in robot_joint_ids
            for actuator_id in range(model.nu)
            if model.actuator_trnid[actuator_id, 0] == joint_id
        ],
        dtype=int,
    )
    if robot_actuator_ids.size not in (0, len(robot_joint_ids)):
        raise ValueError("static model must have either zero or one actuator per robot joint")
    wheel_v = np.array(
        [
            _joint_dof_address(model, "rickshaw/left_wheel_joint"),
            _joint_dof_address(model, "rickshaw/right_wheel_joint"),
        ]
    )
    unactuated_v = np.concatenate(
        (
            np.arange(robot_root_v, robot_root_v + 6),
            np.arange(cart_root_v, cart_root_v + 6),
            wheel_v,
        )
    )
    inverse_balance_v = unactuated_v

    foot_geoms: list[int] = []
    for body_name in ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link"):
        body_id = model.body(body_name).id
        first = model.body_geomadr[body_id]
        count = model.body_geomnum[body_id]
        foot_geoms.extend(
            index for index in range(first, first + count) if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_SPHERE
        )
    wheel_geoms = [
        int(model.body_geomadr[model.body(name).id])
        for name in ("rickshaw/left_wheel_link", "rickshaw/right_wheel_link")
    ]
    if len(foot_geoms) != 8:
        raise ValueError(f"expected eight foot contact spheres, got {len(foot_geoms)}")

    static_cart = solve_fixed_contact_statics(
        mass=40.04,
        gradient=gradient,
        com_from_axle_sln=(0.6514788970649351, 0.0, 0.2944321827032967),
        handle_from_axle_sn=(1.664929, -0.194253),
        hitch_half_width=0.276,
        wheel_track=0.756462,
    )
    hand_force_s = -sum(wrench[0] for wrench in static_cart.handle_wrenches_sln)
    hand_force_n = -sum(wrench[2] for wrench in static_cart.handle_wrenches_sln)

    pose_seed = np.concatenate(
        (
            q_seed[robot_root_q : robot_root_q + 3],
            _rpy_from_quat(q_seed[robot_root_q + 3 : robot_root_q + 7]),
            q_seed[robot_joint_q],
            q_seed[cart_root_q : cart_root_q + 3],
            _rpy_from_quat(q_seed[cart_root_q + 3 : cart_root_q + 7]),
        )
    )
    x0 = pose_seed
    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    lower[:3], upper[:3] = (-0.2, -0.1, 0.55), (0.2, 0.1, 0.9)
    lower[3:6], upper[3:6] = (-0.35, -0.6, -0.2), (0.35, 0.6, 0.2)
    joint_start = 6
    joint_end = joint_start + len(robot_joint_ids)
    for offset, joint_id in enumerate(robot_joint_ids):
        if model.jnt_limited[joint_id]:
            lower[joint_start + offset], upper[joint_start + offset] = model.jnt_range[joint_id]
    lower[joint_end : joint_end + 3] = (-2.0, -0.4, -0.05)
    upper[joint_end : joint_end + 3] = (0.5, 0.4, 0.2)
    lower[-3:], upper[-3:] = (-0.35, -0.6, -0.2), (0.35, 0.6, 0.2)

    data = mujoco.MjData(model)
    gravity_original = model.opt.gravity.copy()
    gamma = math.atan(gradient)
    model.opt.gravity[:] = (-9.81 * math.sin(gamma), 0.0, -9.81 * math.cos(gamma))
    diagnostics: dict[str, float] = {}

    def unpack(x: np.ndarray) -> None:
        data.qpos[:] = q_seed
        data.qpos[robot_root_q : robot_root_q + 3] = x[:3]
        data.qpos[robot_root_q + 3 : robot_root_q + 7] = _quat_from_rpy(x[3:6])
        data.qpos[robot_joint_q] = x[joint_start:joint_end]
        data.qpos[cart_root_q : cart_root_q + 3] = x[joint_end : joint_end + 3]
        data.qpos[cart_root_q + 3 : cart_root_q + 7] = _quat_from_rpy(x[-3:])
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        data.qfrc_applied[:] = 0.0
        if robot_actuator_ids.size:
            data.ctrl[robot_actuator_ids] = data.qpos[robot_joint_q]

    def kinematic_residuals() -> tuple[np.ndarray, np.ndarray, float, float]:
        position_errors: list[np.ndarray] = []
        for side in ("left", "right"):
            grasp = data.site(f"robot/{side}_grasp_site")
            hitch = data.site(f"rickshaw/{side}_hitch_site")
            position_errors.append(np.asarray(grasp.xpos) - np.asarray(hitch.xpos))
        foot_height = np.array(
            [data.geom_xpos[index, 2] - model.geom_size[index, 0] for index in foot_geoms]
        )
        wheel_height = np.array(
            [data.geom_xpos[index, 2] - model.geom_size[index, 0] for index in wheel_geoms]
        )
        foot_center = np.mean(data.geom_xpos[foot_geoms], axis=0)
        handle_center = 0.5 * (
            data.site("robot/left_grasp_site").xpos + data.site("robot/right_grasp_site").xpos
        )
        fat2 = fat2_reference_angle_scalar(
            handle_s=float(handle_center[0] - foot_center[0]),
            handle_n=float(handle_center[2] - foot_center[2]),
            hand_force_s=hand_force_s,
            hand_force_n=hand_force_n,
            robot_mass=cfg.robot_mass,
            com_radius=cfg.robot_com_radius,
            theta_max=cfg.theta_max,
        )
        torso = data.body("robot/torso_link").xmat.reshape(3, 3)
        torso_pitch = math.atan2(-torso[2, 0], math.hypot(torso[0, 0], torso[1, 0]))
        return (
            np.concatenate(position_errors),
            np.concatenate((foot_height, wheel_height)) - cfg.contact_penetration,
            fat2,
            torso_pitch,
        )

    def residual(x: np.ndarray) -> np.ndarray:
        unpack(x)
        mujoco.mj_forward(model, data)
        position_error, support_error, fat2, torso_pitch = kinematic_residuals()

        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        unactuated_force = data.qfrc_inverse[inverse_balance_v]
        joint_torque = data.qfrc_inverse[robot_joint_v]
        torque_ratio = np.abs(joint_torque) / effort_limits
        torque_excess = np.sign(joint_torque) * np.maximum(
            torque_ratio - cfg.torque_limit_fraction,
            0.0,
        )

        diagnostics["fat2"] = fat2

        return np.concatenate(
            (
                position_error / cfg.position_scale,
                support_error / cfg.support_scale,
                unactuated_force / cfg.unactuated_force_scale,
                torque_excess / (1.0 - cfg.torque_limit_fraction),
                (x[joint_start:joint_end] - q_seed[robot_joint_q]) / cfg.posture_scale,
                np.array(((torso_pitch - fat2) / cfg.fat2_scale, x[0] / 0.05, x[1] / 0.03)),
            )
        )

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=cfg.max_nfev,
            xtol=1.0e-9,
            ftol=1.0e-9,
            gtol=1.0e-9,
        )
        unpack(result.x)
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        torque_seed = np.clip(
            data.qfrc_inverse[robot_joint_v],
            -effort_limits,
            effort_limits,
        )
        acceleration_scale = np.ones(model.nv)

        def forward_residual(joint_torque: np.ndarray) -> np.ndarray:
            unpack(result.x)
            data.qfrc_applied[robot_joint_v] = joint_torque
            mujoco.mj_forward(model, data)
            return np.concatenate(
                (
                    data.qacc / acceleration_scale,
                    joint_torque / (100.0 * effort_limits),
                )
            )

        forward_result = least_squares(
            forward_residual,
            torque_seed,
            bounds=(-effort_limits, effort_limits),
            max_nfev=cfg.forward_max_nfev,
            diff_step=1.0e-4,
            x_scale="jac",
            xtol=1.0e-9,
            ftol=1.0e-9,
            gtol=1.0e-9,
        )
        final_residual = forward_residual(forward_result.x)
        final_qpos = data.qpos.copy()
        joint_torque = forward_result.x.copy()
        actuator_target = final_qpos[robot_joint_q] + joint_torque / stiffness
        actuator_torque_ratio = float(np.max(np.abs(joint_torque) / effort_limits))

        data.qfrc_applied[:] = 0.0
        if robot_actuator_ids.size:
            data.ctrl[robot_actuator_ids] = actuator_target
        else:
            data.qfrc_applied[robot_joint_v] = joint_torque
        mujoco.mj_forward(model, data)
        _, _, final_fat2, _ = kinematic_residuals()
        final_acceleration = data.qacc.copy()
        applied_joint_torque = data.qfrc_actuator[robot_joint_v] + data.qfrc_applied[robot_joint_v]
        position_error = max(
            np.linalg.norm(data.site(f"robot/{side}_grasp_site").xpos - data.site(f"rickshaw/{side}_hitch_site").xpos)
            for side in ("left", "right")
        )
        support_error = max(
            *(
                abs(data.geom_xpos[index, 2] - model.geom_size[index, 0] - cfg.contact_penetration)
                for index in foot_geoms
            ),
            *(
                abs(data.geom_xpos[index, 2] - model.geom_size[index, 0] - cfg.contact_penetration)
                for index in wheel_geoms
            ),
        )
        actuator_torque_error = float(
            np.linalg.norm(applied_joint_torque - joint_torque, ord=np.inf)
        )
        acceleration_error = float(np.linalg.norm(final_acceleration, ord=np.inf))
        if not np.all(np.isfinite(final_residual)):
            raise RuntimeError("MuJoCo static solve produced non-finite residuals")
        converged = (
            position_error <= cfg.position_tolerance
            and support_error <= cfg.support_tolerance
            and acceleration_error <= cfg.acceleration_tolerance
            and actuator_torque_error <= cfg.actuator_torque_tolerance
            and actuator_torque_ratio <= cfg.actuator_torque_ratio_tolerance + 1.0e-6
        )
        if not converged:
            worst_torque_joint = joint_names[int(np.argmax(np.abs(joint_torque) / effort_limits))]
            raise RuntimeError(
                "MuJoCo static solve failed: "
                f"inverse={result.message}; forward={forward_result.message}; "
                f"position={position_error:.6g}, "
                f"support={support_error:.6g}, qacc={acceleration_error:.6g}, "
                f"actuator_error={actuator_torque_error:.6g}, "
                f"torque_ratio={actuator_torque_ratio:.6g}, "
                f"torque_joint={worst_torque_joint}"
            )
        return MujocoStaticEquilibrium(
            gradient=gradient,
            qpos=final_qpos,
            joint_position_target=actuator_target,
            joint_actuator_torque=joint_torque,
            fat2_reference_angle=final_fat2,
            equality_position_error=float(position_error),
            support_height_error=float(support_error),
            acceleration_error=acceleration_error,
            actuator_torque_error=actuator_torque_error,
            actuator_torque_ratio=actuator_torque_ratio,
        )
    finally:
        model.opt.gravity[:] = gravity_original


__all__ = [
    "FixedContactStaticSolution",
    "MujocoStaticEquilibrium",
    "MujocoStaticSolverCfg",
    "fat2_reference_angle_scalar",
    "fixed_contact_static_components",
    "solve_mujoco_static_equilibrium",
    "solve_fixed_contact_statics",
]
