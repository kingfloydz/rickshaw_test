"""The deliberately small reward set specified for G1 rickshaw tracking."""

from __future__ import annotations

import torch

REWARD_WEIGHTS = {
    "track_speed_exp": 1.5,
    "lateral_error_l2": -0.5,
    "heading_error_l2": -0.5,
    "zmp_margin_barrier": 0.0,
    "hitch_height_exp": 0.5,
    "hitch_height_recovery_l2": -0.25,
    "fat2_prior_exp": 0.0,
    "feet_gait": 0.25,
    "feet_swing_height": -20.0,
    "feet_slide": -0.20,
    "terrain_normal_velocity_l2": -0.25,
    "joint_power_l1": -2.0e-4,
    "processed_action_rate_l2": -0.03,
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
MOVING_COMMAND_THRESHOLD_MPS = 0.05
GAIT_PERIOD_S = 1.20
GAIT_PHASE_OFFSETS = (0.0, 0.5)
GAIT_STANCE_THRESHOLD = 0.55
FOOT_SWING_HEIGHT_TARGET_M = 0.07
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
    "feet_gait": {"scale": GAIT_PERIOD_S, "unit": "s"},
    "feet_swing_height": {"scale": FOOT_SWING_HEIGHT_TARGET_M, "unit": "m"},
    "feet_slide": {"scale": 1.0, "unit": "m/s"},
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
    lateral_penalty_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    velocity_error = torch.square(v_ref - v_robot_s) + lateral_penalty_scale * torch.square(v_robot_l)
    return torch.exp(-velocity_error / SPEED_ERROR_SCALE_MPS**2)


def lateral_error_l2_value(lateral_error: torch.Tensor) -> torch.Tensor:
    return torch.square(lateral_error / LATERAL_ERROR_SCALE_M)


def heading_error_l2_value(heading_error: torch.Tensor) -> torch.Tensor:
    wrapped = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    return torch.square(wrapped / HEADING_ERROR_SCALE_RAD)


def zmp_margin_barrier_value(zmp_margin: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.relu(ZMP_MARGIN_SCALE_M - zmp_margin) / ZMP_MARGIN_SCALE_M)


def hitch_height_exp_value(
    hitch_height: torch.Tensor,
    target_height: float,
    two_wheel_contact: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(-torch.square((hitch_height - target_height) / HITCH_HEIGHT_ERROR_SCALE_M)) * two_wheel_contact.to(
        hitch_height.dtype
    )


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
    return torch.exp(-torch.square((torso_pitch - theta_fat) / sigma)) * valid.to(torso_pitch.dtype)


def terrain_normal_velocity_l2_value(normal_velocity: torch.Tensor) -> torch.Tensor:
    return torch.square(normal_velocity / TERRAIN_NORMAL_VELOCITY_SCALE_MPS)


def joint_power_l1_value(torque: torch.Tensor, joint_velocity: torch.Tensor) -> torch.Tensor:
    if torque.shape != joint_velocity.shape:
        raise ValueError("joint torque and velocity shapes differ")
    return torch.sum(torch.abs(torque * joint_velocity), dim=-1) / JOINT_POWER_NORMALIZER_W


def processed_action_rate_l2_value(
    action: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(torch.square(action - previous_action), dim=-1)


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


def feet_gait_value(
    episode_time_s: torch.Tensor,
    is_contact: torch.Tensor,
    v_ref: torch.Tensor,
    *,
    period: float = GAIT_PERIOD_S,
    offsets: tuple[float, float] = GAIT_PHASE_OFFSETS,
    threshold: float = GAIT_STANCE_THRESHOLD,
) -> torch.Tensor:
    global_phase = torch.remainder(episode_time_s, period) / period
    offsets_tensor = episode_time_s.new_tensor(offsets)
    phase = torch.remainder(global_phase[:, None] + offsets_tensor[None, :], 1.0)
    desired_contact = phase < threshold
    contact_match = ~(desired_contact ^ is_contact)
    moving = torch.abs(v_ref) > MOVING_COMMAND_THRESHOLD_MPS
    return torch.sum(contact_match.to(episode_time_s.dtype), dim=-1) * moving.to(episode_time_s.dtype)


def feet_swing_height_value(
    foot_height: torch.Tensor,
    is_contact: torch.Tensor,
    *,
    target_height: float = FOOT_SWING_HEIGHT_TARGET_M,
) -> torch.Tensor:
    height_error = torch.square(foot_height - target_height) * ~is_contact
    return torch.sum(height_error, dim=-1)


def feet_slide_value(
    foot_velocity: torch.Tensor,
    is_contact: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(torch.square(foot_velocity) * is_contact[..., None], dim=(1, 2))


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


__all__ = [
    "FAT2_ERROR_SCALE_RAD",
    "FOOT_SWING_HEIGHT_TARGET_M",
    "GAIT_PERIOD_S",
    "GAIT_PHASE_OFFSETS",
    "GAIT_STANCE_THRESHOLD",
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
    "fat2_prior_exp_value",
    "feet_swing_height_value",
    "feet_gait_value",
    "feet_slide_value",
    "heading_error_l2_value",
    "hip_yaw_roll_reference_l2_value",
    "hitch_height_exp_value",
    "hitch_height_recovery_l2_value",
    "joint_power_l1_value",
    "lateral_error_l2_value",
    "pelvis_height_limits_l2_value",
    "processed_action_rate_l2_value",
    "terrain_normal_velocity_l2_value",
    "track_speed_exp_value",
    "zmp_margin_barrier_value",
]
