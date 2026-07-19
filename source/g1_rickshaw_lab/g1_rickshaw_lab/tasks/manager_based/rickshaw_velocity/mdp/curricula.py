"""Startup-fixed slope assignment."""

from __future__ import annotations

from typing import Any

import torch

from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
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


def randomize_startup_slopes(env: Any, env_ids: Any) -> None:
    """Randomly map an exactly balanced slope set to all environments."""

    del env_ids
    _, levels, terrain_types = balanced_slope_assignment(env.num_envs, device=env.device, shuffle=True)
    terrain = env.scene.terrain
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(terrain_types)
    terrain.env_origins.copy_(terrain.terrain_origins[levels, terrain_types])


__all__ = ["balanced_slope_assignment", "randomize_startup_slopes"]
