"""The deliberately small reward set specified for G1 rickshaw tracking."""

from __future__ import annotations

from typing import Any

import torch

from g1_rickshaw_lab.policy_schema import ACTION_SCALE

REWARD_WEIGHTS = {
    "track_speed_exp": 1.0,
    "lateral_error_l2": -0.5,
    "heading_error_l2": -0.5,
    "zmp_margin_barrier": -2.0,
    "hitch_height_exp": 0.5,
    "hitch_height_recovery_l2": -0.25,
    "fat2_prior_exp": 0.1,
    "feet_landing": 0.25,
    "feet_air_time_excess_l2": -0.25,
    "feet_slide": -0.20,
    "terrain_normal_velocity_l2": -0.25,
    "joint_power_l1": -2.0e-4,
    "processed_action_rate_l2": -0.01,
    "hip_yaw_roll_reference_l2": -0.05,
    "pelvis_height_limits_l2": -1.0,
    "joint_position_limits": -1.0,
    "termination": -200.0,
}

# Every reward callable returns a dimensionless value; SI scales are explicit.
SPEED_ERROR_SCALE_MPS = 0.5
LATERAL_ERROR_SCALE_M = 0.30
HEADING_ERROR_SCALE_RAD = 0.30
ZMP_MARGIN_SCALE_M = 0.02
HITCH_HEIGHT_ERROR_SCALE_M = 0.02
HITCH_HEIGHT_RECOVERY_DEADBAND_M = 0.05
HITCH_HEIGHT_RECOVERY_SCALE_M = 0.05
FAT2_ERROR_SCALE_RAD = 0.12
FEET_LANDING_TARGET_AIR_TIME_S = 0.30
FEET_LANDING_SIGMA_S = 0.12
FEET_MAX_AIR_TIME_S = 0.50
FEET_AIR_TIME_EXCESS_SCALE_S = 0.20
MOVING_COMMAND_THRESHOLD_MPS = 0.05
FEET_SLIDE_NORMALIZER_MPS = 1.0
TERRAIN_NORMAL_VELOCITY_SCALE_MPS = 0.25
JOINT_POWER_NORMALIZER_W = 1.0
HIP_YAW_ROLL_REFERENCE_SCALE_RAD = 0.20
HIP_YAW_ROLL_POLICY_INDICES = (1, 2, 7, 8)
PELVIS_HEIGHT_BOUNDS_M = (0.58, 0.87)
PELVIS_HEIGHT_ERROR_SCALE_M = 0.05
JOINT_LIMIT_NORMALIZER_RAD = 1.0

REWARD_NORMALIZATION_SCALES = {
    "track_speed_exp": {"scale": SPEED_ERROR_SCALE_MPS, "unit": "m/s"},
    "lateral_error_l2": {"scale": LATERAL_ERROR_SCALE_M, "unit": "m"},
    "heading_error_l2": {"scale": HEADING_ERROR_SCALE_RAD, "unit": "rad"},
    "zmp_margin_barrier": {"scale": ZMP_MARGIN_SCALE_M, "unit": "m"},
    "hitch_height_exp": {"scale": HITCH_HEIGHT_ERROR_SCALE_M, "unit": "m"},
    "hitch_height_recovery_l2": {
        "scale": HITCH_HEIGHT_RECOVERY_SCALE_M,
        "unit": "m",
    },
    "fat2_prior_exp": {"scale": FAT2_ERROR_SCALE_RAD, "unit": "rad"},
    "feet_landing": {
        "scale": FEET_LANDING_SIGMA_S,
        "unit": "s",
    },
    "feet_air_time_excess_l2": {
        "scale": FEET_AIR_TIME_EXCESS_SCALE_S,
        "unit": "s",
    },
    "feet_slide": {"scale": FEET_SLIDE_NORMALIZER_MPS, "unit": "m/s"},
    "terrain_normal_velocity_l2": {
        "scale": TERRAIN_NORMAL_VELOCITY_SCALE_MPS,
        "unit": "m/s",
    },
    "joint_power_l1": {"scale": JOINT_POWER_NORMALIZER_W, "unit": "W"},
    "processed_action_rate_l2": {
        "scale": 1.0,
        "unit": "normalized_action",
    },
    "hip_yaw_roll_reference_l2": {
        "scale": HIP_YAW_ROLL_REFERENCE_SCALE_RAD,
        "unit": "rad",
    },
    "pelvis_height_limits_l2": {
        "scale": PELVIS_HEIGHT_ERROR_SCALE_M,
        "unit": "m",
    },
    "joint_position_limits": {"scale": JOINT_LIMIT_NORMALIZER_RAD, "unit": "rad"},
    "termination": {"scale": 1.0, "unit": "binary"},
}


def track_speed_exp_value(
    v_ref: torch.Tensor,
    v_robot_s: torch.Tensor,
    v_robot_l: torch.Tensor,
) -> torch.Tensor:
    velocity_error = torch.square(v_ref - v_robot_s) + torch.square(v_robot_l)
    return torch.exp(-velocity_error / SPEED_ERROR_SCALE_MPS**2)


def lateral_error_l2_value(lateral_error: torch.Tensor) -> torch.Tensor:
    return torch.square(lateral_error / LATERAL_ERROR_SCALE_M)


def heading_error_l2_value(heading_error: torch.Tensor) -> torch.Tensor:
    wrapped = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    return torch.square(wrapped / HEADING_ERROR_SCALE_RAD)


def zmp_margin_barrier_value(zmp_margin: torch.Tensor) -> torch.Tensor:
    return torch.square(
        torch.relu(ZMP_MARGIN_SCALE_M - zmp_margin) / ZMP_MARGIN_SCALE_M
    )


def hitch_height_exp_value(
    hitch_height: torch.Tensor,
    target_height: float,
    two_wheel_contact: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(
        -torch.square((hitch_height - target_height) / HITCH_HEIGHT_ERROR_SCALE_M)
    ) * two_wheel_contact.to(hitch_height.dtype)


def hitch_height_recovery_l2_value(
    hitch_height: torch.Tensor,
    target_height: float,
    *,
    deadband: float = HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    scale: float = HITCH_HEIGHT_RECOVERY_SCALE_M,
) -> torch.Tensor:
    """Provide a restoring gradient after hitch error leaves the local tracking region."""

    if deadband < 0.0:
        raise ValueError("hitch height recovery deadband must be non-negative")
    if scale <= 0.0:
        raise ValueError("hitch height recovery scale must be positive")
    normalized = torch.relu(torch.abs(hitch_height - target_height) - deadband) / scale
    return torch.where(
        normalized <= 1.0,
        torch.square(normalized),
        2.0 * normalized - 1.0,
    )


def fat2_prior_exp_value(
    torso_pitch: torch.Tensor,
    theta_fat: torch.Tensor,
    valid: torch.Tensor,
    *,
    sigma: float = FAT2_ERROR_SCALE_RAD,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("FAT2 sigma must be positive")
    return torch.exp(-torch.square((torso_pitch - theta_fat) / sigma)) * valid.to(
        torso_pitch.dtype
    )


def terrain_normal_velocity_l2_value(normal_velocity: torch.Tensor) -> torch.Tensor:
    return torch.square(normal_velocity / TERRAIN_NORMAL_VELOCITY_SCALE_MPS)


def joint_power_l1_value(torque: torch.Tensor, joint_velocity: torch.Tensor) -> torch.Tensor:
    if torque.shape != joint_velocity.shape:
        raise ValueError("joint torque and velocity shapes differ")
    return torch.sum(torch.abs(torque * joint_velocity), dim=-1) / JOINT_POWER_NORMALIZER_W


def processed_action_rate_l2_value(
    target: torch.Tensor,
    previous_target: torch.Tensor,
) -> torch.Tensor:
    if target.shape != previous_target.shape:
        raise ValueError("processed action histories must have identical shapes")
    action_scale = target.new_tensor(ACTION_SCALE)
    return torch.mean(torch.square((target - previous_target) / action_scale), dim=-1)


def hip_yaw_roll_reference_l2_value(
    joint_position: torch.Tensor,
    reference_position: torch.Tensor,
    scale: float = HIP_YAW_ROLL_REFERENCE_SCALE_RAD,
) -> torch.Tensor:
    if joint_position.shape != reference_position.shape:
        raise ValueError("hip joint positions and references must have identical shapes")
    if scale <= 0.0:
        raise ValueError("hip reference error scale must be positive")
    return torch.mean(torch.square((joint_position - reference_position) / scale), dim=-1)


def feet_landing_value(
    first_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    v_ref: torch.Tensor,
) -> torch.Tensor:
    """Score the previous swing only when exactly one foot lands this step."""

    contact = first_contact.to(last_air_time.dtype)
    single_landing = torch.sum(first_contact, dim=-1) == 1
    moving = torch.abs(v_ref) > MOVING_COMMAND_THRESHOLD_MPS
    landing_kernel = torch.exp(
        -torch.square(
            (last_air_time - FEET_LANDING_TARGET_AIR_TIME_S) / FEET_LANDING_SIGMA_S
        )
    )
    overlong = torch.square(
        torch.relu(last_air_time - FEET_MAX_AIR_TIME_S)
        / FEET_AIR_TIME_EXCESS_SCALE_S
    )
    gated_kernel = landing_kernel * moving[:, None].to(last_air_time.dtype)
    return torch.sum(contact * (gated_kernel - overlong), dim=-1) * single_landing.to(
        last_air_time.dtype
    )


def feet_air_time_excess_l2_value(current_air_time: torch.Tensor) -> torch.Tensor:
    excess = (
        torch.relu(current_air_time - FEET_MAX_AIR_TIME_S)
        / FEET_AIR_TIME_EXCESS_SCALE_S
    )
    return torch.sum(torch.square(excess), dim=-1)


def pelvis_height_limits_l2_value(
    pelvis_height: torch.Tensor,
    bounds: tuple[float, float] = PELVIS_HEIGHT_BOUNDS_M,
    scale: float = PELVIS_HEIGHT_ERROR_SCALE_M,
) -> torch.Tensor:
    """Penalize pelvis height only after it leaves the allowed interval."""

    lower, upper = bounds
    if upper <= lower:
        raise ValueError("pelvis height upper bound must exceed its lower bound")
    if scale <= 0.0:
        raise ValueError("pelvis height error scale must be positive")
    violation = torch.relu(lower - pelvis_height) + torch.relu(pelvis_height - upper)
    return torch.square(violation / scale)


def track_speed_exp(env: Any) -> torch.Tensor:
    return track_speed_exp_value(
        env.command_state.v_ref,
        env.policy_robot_speed_s,
        env.policy_robot_speed_l,
    )


def lateral_error_l2(env: Any) -> torch.Tensor:
    return lateral_error_l2_value(env.path_state.lateral_error)


def heading_error_l2(env: Any) -> torch.Tensor:
    return heading_error_l2_value(env.path_state.heading_error)


def zmp_margin_barrier(env: Any) -> torch.Tensor:
    return zmp_margin_barrier_value(env.stability_state.zmp_margin)


def hitch_height_exp(env: Any) -> torch.Tensor:
    return hitch_height_exp_value(
        env.rickshaw_state.hitch_height,
        env.rickshaw_pose_cfg.hitch_height_target,
        env.rickshaw_state.two_wheel_contact,
    )


def hitch_height_recovery_l2(
    env: Any,
    deadband: float = HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    scale: float = HITCH_HEIGHT_RECOVERY_SCALE_M,
) -> torch.Tensor:
    return hitch_height_recovery_l2_value(
        env.rickshaw_state.hitch_height,
        env.rickshaw_pose_cfg.hitch_height_target,
        deadband=deadband,
        scale=scale,
    )


def fat2_prior_exp(env: Any, sigma: float = FAT2_ERROR_SCALE_RAD) -> torch.Tensor:
    return fat2_prior_exp_value(
        env.stability_state.torso_pitch,
        env.stability_state.theta_fat,
        env.stability_state.fat_valid,
        sigma=sigma,
    )


def _resolve_body_ids(entity_cfg: Any | None, fallback: Any) -> Any:
    if entity_cfg is not None:
        body_ids = getattr(entity_cfg, "body_ids", None)
        if body_ids is not None:
            return body_ids
    return fallback


def feet_landing(
    env: Any,
    sensor_cfg: Any | None = None,
) -> torch.Tensor:
    """Reward a clean landing while a moving command is active."""

    sensor_name = "robot_contacts" if sensor_cfg is None else getattr(sensor_cfg, "name", "robot_contacts")
    sensor = env.scene[sensor_name]
    body_ids = _resolve_body_ids(sensor_cfg, env.foot_sensor_ids)
    return feet_landing_value(
        sensor.compute_first_contact(env.step_dt)[:, body_ids],
        sensor.data.last_air_time[:, body_ids],
        env.command_state.v_ref,
    )


def feet_air_time_excess_l2(
    env: Any, sensor_cfg: Any | None = None
) -> torch.Tensor:
    sensor_name = "robot_contacts" if sensor_cfg is None else getattr(sensor_cfg, "name", "robot_contacts")
    sensor = env.scene[sensor_name]
    body_ids = _resolve_body_ids(sensor_cfg, env.foot_sensor_ids)
    return feet_air_time_excess_l2_value(sensor.data.current_air_time[:, body_ids])


def feet_slide(
    env: Any,
    sensor_cfg: Any | None = None,
    asset_cfg: Any | None = None,
) -> torch.Tensor:
    """Sum slope-plane foot speed for feet that are currently in contact."""

    sensor_name = "robot_contacts" if sensor_cfg is None else getattr(sensor_cfg, "name", "robot_contacts")
    sensor = env.scene[sensor_name]
    sensor_ids = _resolve_body_ids(sensor_cfg, env.foot_sensor_ids)
    contact = sensor.data.current_contact_time[:, sensor_ids] > 0.0

    asset_name = "robot" if asset_cfg is None else getattr(asset_cfg, "name", "robot")
    robot = env.scene[asset_name]
    body_ids = _resolve_body_ids(asset_cfg, env.foot_body_ids)
    velocity_w = robot.data.body_lin_vel_w[:, body_ids]
    velocity_s = torch.sum(velocity_w * env.path_tangent_w[:, None, :], dim=-1)
    velocity_y = torch.sum(velocity_w * env.path_lateral_w[:, None, :], dim=-1)
    slide_speed = (
        torch.sqrt(torch.square(velocity_s) + torch.square(velocity_y))
        / FEET_SLIDE_NORMALIZER_MPS
    )
    return torch.sum(slide_speed * contact.to(slide_speed.dtype), dim=-1)


def terrain_normal_velocity_l2(env: Any) -> torch.Tensor:
    return terrain_normal_velocity_l2_value(env.policy_robot_velocity_n)


def _policy_joint_ids(env: Any, asset_cfg: Any | None) -> Any:
    if asset_cfg is not None and getattr(asset_cfg, "joint_ids", None) is not None:
        return asset_cfg.joint_ids
    return env.policy_joint_ids


def joint_power_l1(env: Any, asset_cfg: Any | None = None) -> torch.Tensor:
    asset_name = "robot" if asset_cfg is None else getattr(asset_cfg, "name", "robot")
    robot = env.scene[asset_name]
    joint_ids = _policy_joint_ids(env, asset_cfg)
    return joint_power_l1_value(
        robot.data.applied_torque[:, joint_ids], robot.data.joint_vel[:, joint_ids]
    )


def processed_action_rate_l2(env: Any) -> torch.Tensor:
    return processed_action_rate_l2_value(
        env.action_state.target, env.action_state.prev_target
    )


def hip_yaw_roll_reference_l2(
    env: Any,
    policy_indices: tuple[int, ...] = HIP_YAW_ROLL_POLICY_INDICES,
    scale: float = HIP_YAW_ROLL_REFERENCE_SCALE_RAD,
) -> torch.Tensor:
    robot = env.scene["robot"]
    policy_position = robot.data.joint_pos[:, env.policy_joint_ids]
    index = torch.as_tensor(policy_indices, device=policy_position.device, dtype=torch.long)
    return hip_yaw_roll_reference_l2_value(
        torch.index_select(policy_position, dim=1, index=index),
        torch.index_select(env.action_state.q_ref, dim=1, index=index),
        scale=scale,
    )


def pelvis_height_limits_l2(
    env: Any,
    asset_cfg: Any | None = None,
    bounds: tuple[float, float] = PELVIS_HEIGHT_BOUNDS_M,
    scale: float = PELVIS_HEIGHT_ERROR_SCALE_M,
) -> torch.Tensor:
    """Measure pelvis clearance along the local terrain normal."""

    asset_name = "robot" if asset_cfg is None else getattr(asset_cfg, "name", "robot")
    robot = env.scene[asset_name]
    if asset_cfg is None:
        pelvis_position_w = robot.data.root_pos_w
    else:
        body_ids = _resolve_body_ids(asset_cfg, None)
        if body_ids is None:
            raise ValueError("pelvis height reward asset_cfg has no resolved body IDs")
        configured_position_w = robot.data.body_pos_w[:, body_ids]
        if configured_position_w.shape[1] != 1:
            raise ValueError("pelvis height reward requires exactly one configured body")
        pelvis_position_w = configured_position_w[:, 0]
    terrain_origin_w = env.scene.terrain.env_origins
    pelvis_height = torch.sum(
        (pelvis_position_w - terrain_origin_w) * env.path_normal_w,
        dim=-1,
    )
    return pelvis_height_limits_l2_value(pelvis_height, bounds=bounds, scale=scale)


def joint_position_limits(env: Any, asset_cfg: Any | None = None) -> torch.Tensor:
    """Isaac Lab soft-limit penalty restricted to the persisted 29 joints."""

    asset_name = "robot" if asset_cfg is None else getattr(asset_cfg, "name", "robot")
    robot = env.scene[asset_name]
    joint_ids = _policy_joint_ids(env, asset_cfg)
    position = robot.data.joint_pos[:, joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    below = torch.clamp(limits[..., 0] - position, min=0.0)
    above = torch.clamp(position - limits[..., 1], min=0.0)
    return torch.sum(below + above, dim=-1) / JOINT_LIMIT_NORMALIZER_RAD


def termination(env: Any) -> torch.Tensor:
    """Return one only for non-timeout termination causes."""

    manager = env.termination_manager
    terminated = manager.terminated
    return (terminated & ~manager.time_outs).to(dtype=torch.float32)


__all__ = [
    "FAT2_ERROR_SCALE_RAD",
    "FEET_AIR_TIME_EXCESS_SCALE_S",
    "FEET_LANDING_SIGMA_S",
    "FEET_LANDING_TARGET_AIR_TIME_S",
    "FEET_MAX_AIR_TIME_S",
    "FEET_SLIDE_NORMALIZER_MPS",
    "HEADING_ERROR_SCALE_RAD",
    "HIP_YAW_ROLL_POLICY_INDICES",
    "HIP_YAW_ROLL_REFERENCE_SCALE_RAD",
    "HITCH_HEIGHT_ERROR_SCALE_M",
    "HITCH_HEIGHT_RECOVERY_DEADBAND_M",
    "HITCH_HEIGHT_RECOVERY_SCALE_M",
    "JOINT_LIMIT_NORMALIZER_RAD",
    "JOINT_POWER_NORMALIZER_W",
    "LATERAL_ERROR_SCALE_M",
    "PELVIS_HEIGHT_BOUNDS_M",
    "PELVIS_HEIGHT_ERROR_SCALE_M",
    "REWARD_NORMALIZATION_SCALES",
    "REWARD_WEIGHTS",
    "SPEED_ERROR_SCALE_MPS",
    "MOVING_COMMAND_THRESHOLD_MPS",
    "TERRAIN_NORMAL_VELOCITY_SCALE_MPS",
    "ZMP_MARGIN_SCALE_M",
    "fat2_prior_exp",
    "fat2_prior_exp_value",
    "feet_air_time_excess_l2",
    "feet_air_time_excess_l2_value",
    "feet_landing",
    "feet_landing_value",
    "feet_slide",
    "heading_error_l2",
    "heading_error_l2_value",
    "hip_yaw_roll_reference_l2",
    "hip_yaw_roll_reference_l2_value",
    "hitch_height_exp",
    "hitch_height_exp_value",
    "hitch_height_recovery_l2",
    "hitch_height_recovery_l2_value",
    "joint_position_limits",
    "joint_power_l1",
    "joint_power_l1_value",
    "lateral_error_l2",
    "lateral_error_l2_value",
    "pelvis_height_limits_l2",
    "pelvis_height_limits_l2_value",
    "processed_action_rate_l2",
    "processed_action_rate_l2_value",
    "termination",
    "terrain_normal_velocity_l2",
    "terrain_normal_velocity_l2_value",
    "track_speed_exp",
    "track_speed_exp_value",
    "zmp_margin_barrier",
    "zmp_margin_barrier_value",
]
