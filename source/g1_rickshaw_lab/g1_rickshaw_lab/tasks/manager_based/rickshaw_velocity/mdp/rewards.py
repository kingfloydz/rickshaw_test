"""The deliberately small reward set specified for G1 rickshaw tracking."""

from __future__ import annotations

from typing import Any

import torch


REWARD_WEIGHTS = {
    "track_speed_exp": 2.0,
    "lateral_error_l2": -0.5,
    "heading_error_l2": -0.5,
    "zmp_margin_barrier": -2.0,
    "hitch_height_exp": 0.5,
    "fat2_prior_exp": 0.1,
    "feet_air_time": 0.10,
    "feet_slide": -0.10,
    "terrain_normal_velocity_l2": -0.5,
    "joint_power_l1": -1.0e-4,
    "processed_action_rate_l2": -0.01,
    "processed_action_jerk_l2": -0.005,
    "joint_position_limits": -1.0,
    "termination": -200.0,
}

# Every reward callable returns a dimensionless value.  Unit-valued SI
# normalizers make the formerly implicit units explicit without changing the
# numerical reward signal or any persisted policy training scale.
SPEED_ERROR_SCALE_MPS = 0.25
LATERAL_ERROR_SCALE_M = 0.30
HEADING_ERROR_SCALE_RAD = 0.30
ZMP_MARGIN_SCALE_M = 0.02
HITCH_HEIGHT_ERROR_SCALE_M = 0.02
FAT2_ERROR_SCALE_RAD = 0.12
FEET_AIR_TIME_NORMALIZER_S = 1.0
FEET_AIR_TIME_THRESHOLD_S = 0.4
FEET_SLIDE_NORMALIZER_MPS = 1.0
TERRAIN_NORMAL_VELOCITY_SCALE_MPS = 0.25
JOINT_POWER_NORMALIZER_W = 1.0
PROCESSED_ACTION_RATE_SCALE_RAD = 0.05
PROCESSED_ACTION_JERK_SCALE_RAD = 0.03
JOINT_LIMIT_NORMALIZER_RAD = 1.0

REWARD_NORMALIZATION_SCALES = {
    "track_speed_exp": {"scale": SPEED_ERROR_SCALE_MPS, "unit": "m/s"},
    "lateral_error_l2": {"scale": LATERAL_ERROR_SCALE_M, "unit": "m"},
    "heading_error_l2": {"scale": HEADING_ERROR_SCALE_RAD, "unit": "rad"},
    "zmp_margin_barrier": {"scale": ZMP_MARGIN_SCALE_M, "unit": "m"},
    "hitch_height_exp": {"scale": HITCH_HEIGHT_ERROR_SCALE_M, "unit": "m"},
    "fat2_prior_exp": {"scale": FAT2_ERROR_SCALE_RAD, "unit": "rad"},
    "feet_air_time": {"scale": FEET_AIR_TIME_NORMALIZER_S, "unit": "s"},
    "feet_slide": {"scale": FEET_SLIDE_NORMALIZER_MPS, "unit": "m/s"},
    "terrain_normal_velocity_l2": {
        "scale": TERRAIN_NORMAL_VELOCITY_SCALE_MPS,
        "unit": "m/s",
    },
    "joint_power_l1": {"scale": JOINT_POWER_NORMALIZER_W, "unit": "W"},
    "processed_action_rate_l2": {
        "scale": PROCESSED_ACTION_RATE_SCALE_RAD,
        "unit": "rad",
    },
    "processed_action_jerk_l2": {
        "scale": PROCESSED_ACTION_JERK_SCALE_RAD,
        "unit": "rad",
    },
    "joint_position_limits": {"scale": JOINT_LIMIT_NORMALIZER_RAD, "unit": "rad"},
    "termination": {"scale": 1.0, "unit": "binary"},
}


def track_speed_exp_value(v_ref: torch.Tensor, v_robot_s: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.square((v_ref - v_robot_s) / SPEED_ERROR_SCALE_MPS))


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
    target: torch.Tensor, previous_target: torch.Tensor
) -> torch.Tensor:
    if target.shape != previous_target.shape:
        raise ValueError("processed action histories must have identical shapes")
    return torch.mean(
        torch.square((target - previous_target) / PROCESSED_ACTION_RATE_SCALE_RAD),
        dim=-1,
    )


def processed_action_jerk_l2_value(
    target: torch.Tensor,
    previous_target: torch.Tensor,
    previous_previous_target: torch.Tensor,
) -> torch.Tensor:
    if target.shape != previous_target.shape or target.shape != previous_previous_target.shape:
        raise ValueError("processed action histories must have identical shapes")
    jerk = target - 2.0 * previous_target + previous_previous_target
    return torch.mean(torch.square(jerk / PROCESSED_ACTION_JERK_SCALE_RAD), dim=-1)


def track_speed_exp(env: Any) -> torch.Tensor:
    return track_speed_exp_value(env.command_state.v_ref, env.policy_robot_speed_s)


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


def feet_air_time(
    env: Any,
    sensor_cfg: Any | None = None,
    threshold: float = FEET_AIR_TIME_THRESHOLD_S,
) -> torch.Tensor:
    """Reward first contact after 0.4 s of air time, gated for moving commands."""

    if threshold != FEET_AIR_TIME_THRESHOLD_S:
        raise ValueError("the specified feet-air-time threshold is 0.4 s")
    sensor_name = "robot_contacts" if sensor_cfg is None else getattr(sensor_cfg, "name", "robot_contacts")
    sensor = env.scene[sensor_name]
    body_ids = _resolve_body_ids(sensor_cfg, env.foot_sensor_ids)
    first_contact = sensor.compute_first_contact(env.step_dt)[:, body_ids]
    last_air_time = sensor.data.last_air_time[:, body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=-1)
    reward = reward / FEET_AIR_TIME_NORMALIZER_S
    reward = reward * (env.command_state.v_ref > 0.1).to(reward.dtype)
    return reward


def feet_slide(
    env: Any,
    sensor_cfg: Any | None = None,
    asset_cfg: Any | None = None,
) -> torch.Tensor:
    """Sum slope-plane foot speed for feet that are currently in contact."""

    sensor_name = "robot_contacts" if sensor_cfg is None else getattr(sensor_cfg, "name", "robot_contacts")
    sensor = env.scene[sensor_name]
    sensor_ids = _resolve_body_ids(sensor_cfg, env.foot_sensor_ids)
    if hasattr(sensor.data, "current_contact_time"):
        contact = sensor.data.current_contact_time[:, sensor_ids] > 0.0
    else:
        raise AttributeError("contact sensor must expose current_contact_time for feet_slide")

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


def processed_action_jerk_l2(env: Any) -> torch.Tensor:
    return processed_action_jerk_l2_value(
        env.action_state.target,
        env.action_state.prev_target,
        env.action_state.prev_prev_target,
    )


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
    if hasattr(manager, "time_outs"):
        timeout = manager.time_outs
    elif hasattr(env, "time_out_buf"):
        timeout = env.time_out_buf
    else:
        raise AttributeError("TerminationManager must expose time_outs for timeout exclusion")
    return (terminated & ~timeout).to(dtype=torch.float32)


__all__ = [
    "FAT2_ERROR_SCALE_RAD",
    "FEET_AIR_TIME_NORMALIZER_S",
    "FEET_AIR_TIME_THRESHOLD_S",
    "FEET_SLIDE_NORMALIZER_MPS",
    "HEADING_ERROR_SCALE_RAD",
    "HITCH_HEIGHT_ERROR_SCALE_M",
    "JOINT_LIMIT_NORMALIZER_RAD",
    "JOINT_POWER_NORMALIZER_W",
    "LATERAL_ERROR_SCALE_M",
    "PROCESSED_ACTION_JERK_SCALE_RAD",
    "PROCESSED_ACTION_RATE_SCALE_RAD",
    "REWARD_NORMALIZATION_SCALES",
    "REWARD_WEIGHTS",
    "SPEED_ERROR_SCALE_MPS",
    "TERRAIN_NORMAL_VELOCITY_SCALE_MPS",
    "ZMP_MARGIN_SCALE_M",
    "fat2_prior_exp",
    "fat2_prior_exp_value",
    "feet_air_time",
    "feet_slide",
    "heading_error_l2",
    "heading_error_l2_value",
    "hitch_height_exp",
    "hitch_height_exp_value",
    "joint_position_limits",
    "joint_power_l1",
    "joint_power_l1_value",
    "lateral_error_l2",
    "lateral_error_l2_value",
    "processed_action_jerk_l2",
    "processed_action_jerk_l2_value",
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
