"""Startup-fixed slope assignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_GRADIENTS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    terrain_index_for_gradient,
)


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


def weighted_slope_assignment(
    num_envs: int,
    *,
    device: torch.device | str,
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concentrate training environments near level ground while retaining every slope."""

    slopes = torch.tensor(SLOPE_GRADIENTS, device=device)
    side_limit = torch.where(slopes < 0.0, -slopes[0], slopes[-1])
    weights = 2.0 - torch.abs(slopes) / side_limit

    minimum_count = int(num_envs >= SLOPE_COUNT)
    counts = torch.full((SLOPE_COUNT,), minimum_count, device=device, dtype=torch.long)
    remaining = num_envs - minimum_count * SLOPE_COUNT
    quota = weights * remaining / torch.sum(weights)
    counts += torch.floor(quota).to(torch.long)
    remainder = num_envs - int(torch.sum(counts).item())
    if remainder:
        counts[torch.topk(quota - torch.floor(quota), remainder).indices] += 1

    slots = torch.repeat_interleave(torch.arange(SLOPE_COUNT, device=device), counts)
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
    """Assign the center-weighted training slope distribution once at startup."""

    del env_ids
    _, levels, terrain_types = weighted_slope_assignment(
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
    "weighted_slope_assignment",
]
