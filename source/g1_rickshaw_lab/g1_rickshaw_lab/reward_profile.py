"""Reward-weight override contract shared by training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

FAT2_REWARD_TERM = "fat2_prior_exp"
REWARD_WEIGHT_OVERRIDES_KEY = "reward_weight_overrides"
GUIDE_REWARD_TERMS = (
    "track_speed_exp",
    "lateral_error_l2",
    "heading_error_l2",
    "zmp_margin_barrier",
    "hitch_height_exp",
    "hitch_height_recovery_l2",
    "fat2_prior_exp",
    "feet_gait",
    "feet_swing_height",
    "feet_slide",
    "terrain_normal_velocity_l2",
    "joint_power_l1",
    "processed_action_rate_l2",
    "hip_yaw_roll_reference_l2",
    "pelvis_height_limits_l2",
    "joint_position_limits",
    "termination",
)


def validate_reward_weight_overrides(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("reward weight overrides must be a mapping")
    if FAT2_REWARD_TERM in value:
        raise ValueError("fat2_prior_exp must use the dedicated FAT2 parameter")
    unknown = set(value) - set(GUIDE_REWARD_TERMS)
    if unknown:
        raise ValueError(f"unknown reward terms: {sorted(unknown)}")
    result: dict[str, float] = {}
    for name in GUIDE_REWARD_TERMS:
        if name not in value:
            continue
        weight = value[name]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"reward weight {name!r} must be numeric")
        parsed = float(weight)
        if not math.isfinite(parsed):
            raise ValueError(f"reward weight {name!r} must be finite")
        result[name] = parsed
    return result


def parse_reward_weight_arguments(values: Sequence[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        name, separator, raw_weight = value.partition("=")
        if not separator or not name or not raw_weight:
            raise ValueError("--reward-weight must use TERM=WEIGHT")
        if name == FAT2_REWARD_TERM:
            raise ValueError("use --fat2-weight for fat2_prior_exp")
        if name in parsed:
            raise ValueError(f"duplicate reward weight override: {name}")
        try:
            parsed[name] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"reward weight {name!r} must be numeric") from exc
    return validate_reward_weight_overrides(parsed)


def reward_weight_overrides_from_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, float]:
    resolved = configuration.get("resolved_parameters")
    if not isinstance(resolved, Mapping):
        raise ValueError("training configuration has no resolved parameters")
    return validate_reward_weight_overrides(resolved.get(REWARD_WEIGHT_OVERRIDES_KEY, {}))


def reward_weight_hydra_overrides(
    overrides: Mapping[str, float],
) -> list[str]:
    validated = validate_reward_weight_overrides(overrides)
    return [f"env.rewards.{name}.weight={validated[name]!r}" for name in validated]


def apply_reward_weight_overrides(env_cfg: Any, overrides: Mapping[str, float]) -> None:
    for name, weight in validate_reward_weight_overrides(overrides).items():
        rewards = env_cfg.rewards
        term = rewards[name] if isinstance(rewards, Mapping) else getattr(rewards, name)
        term.weight = weight


__all__ = [
    "FAT2_REWARD_TERM",
    "GUIDE_REWARD_TERMS",
    "REWARD_WEIGHT_OVERRIDES_KEY",
    "apply_reward_weight_overrides",
    "parse_reward_weight_arguments",
    "reward_weight_hydra_overrides",
    "reward_weight_overrides_from_configuration",
    "validate_reward_weight_overrides",
]
