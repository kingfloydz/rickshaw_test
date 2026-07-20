"""Startup-fixed slope assignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    terrain_index_for_gradient,
)


def speed_command_levels(
    env: Any,
    env_ids: Sequence[int],
    reward_term_name: str = "track_speed_exp",
) -> torch.Tensor:
    """Expand the forward command range when tracking reaches 80 percent."""

    command_cfg = env.cfg.policy_update.command_sampling
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_cfg.weight * 0.8:
        command_cfg.maximum = min(
            command_cfg.maximum + command_cfg.curriculum_step,
            command_cfg.limit_maximum,
        )

    return torch.tensor(command_cfg.maximum, device=env.device)


def balanced_slope_assignment(
    num_envs: int,
    *,
    device: torch.device | str,
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign all 19 slopes with counts differing by at most one."""

    slots = torch.arange(num_envs, device=device, dtype=torch.long) % SLOPE_COUNT
    if shuffle:
        slots = slots[torch.randperm(num_envs, device=device)]
    levels = torch.tensor(SLOPE_TERRAIN_LEVELS, device=device)[slots]
    terrain_types = torch.tensor(SLOPE_TERRAIN_TYPES, device=device)[slots]
    return slots, levels, terrain_types


def apply_terrain_assignment(
    env: Any,
    levels: Any,
    terrain_types: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one explicit terrain row/column pair per environment."""

    level_tensor = torch.as_tensor(levels, device=env.device, dtype=torch.long).reshape(-1)
    type_tensor = torch.as_tensor(terrain_types, device=env.device, dtype=torch.long).reshape(-1)
    if level_tensor.numel() != env.num_envs or type_tensor.numel() != env.num_envs:
        raise ValueError("terrain assignment size must match the environment count")
    terrain = env.scene.terrain
    terrain.terrain_levels.copy_(level_tensor)
    terrain.terrain_types.copy_(type_tensor)
    terrain.env_origins.copy_(terrain.terrain_origins[level_tensor, type_tensor])
    return level_tensor, type_tensor


def assign_terrain_slopes(
    env: Any,
    slopes: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign exact configured slopes in environment order."""

    indices = tuple(terrain_index_for_gradient(float(slope)) for slope in slopes)
    return apply_terrain_assignment(
        env,
        [level for level, _ in indices],
        [terrain_type for _, terrain_type in indices],
    )


def randomize_startup_slopes(
    env: Any,
    env_ids: Any,
    *,
    shuffle: bool = True,
) -> None:
    """Map an exactly balanced slope set to all environments."""

    del env_ids
    _, levels, terrain_types = balanced_slope_assignment(
        env.num_envs,
        device=env.device,
        shuffle=shuffle,
    )
    apply_terrain_assignment(env, levels, terrain_types)


__all__ = [
    "apply_terrain_assignment",
    "assign_terrain_slopes",
    "balanced_slope_assignment",
    "randomize_startup_slopes",
    "speed_command_levels",
]
