"""Regression tests for startup-fixed balanced slopes."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    TERRAIN_NUM_COLS,
    TERRAIN_NUM_ROWS,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.curricula import (
    balanced_slope_assignment,
    randomize_startup_slopes,
)


def test_balanced_assignment_covers_every_configured_slope() -> None:
    slots, levels, terrain_types = balanced_slope_assignment(
        2 * SLOPE_COUNT + 7, device="cpu"
    )

    assert slots[:SLOPE_COUNT].tolist() == list(range(SLOPE_COUNT))
    assert levels[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_LEVELS)
    assert terrain_types[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_TYPES)
    counts = torch.bincount(slots, minlength=SLOPE_COUNT)
    assert int(torch.max(counts) - torch.min(counts)) <= 1


def test_startup_randomizes_environment_mapping_without_changing_balance() -> None:
    num_envs = 8192
    terrain_origins = torch.arange(
        TERRAIN_NUM_ROWS * TERRAIN_NUM_COLS * 3, dtype=torch.float32
    ).reshape(TERRAIN_NUM_ROWS, TERRAIN_NUM_COLS, 3)
    terrain = SimpleNamespace(
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        terrain_types=torch.zeros(num_envs, dtype=torch.long),
        terrain_origins=terrain_origins,
        env_origins=torch.zeros((num_envs, 3)),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(terrain=terrain),
    )

    torch.manual_seed(42)
    randomize_startup_slopes(env, None)

    pairs = torch.stack((terrain.terrain_levels, terrain.terrain_types), dim=-1)
    expected_pairs = torch.tensor(
        tuple(zip(SLOPE_TERRAIN_LEVELS, SLOPE_TERRAIN_TYPES, strict=True))
    )
    counts = torch.tensor(
        [torch.all(pairs == pair, dim=-1).sum() for pair in expected_pairs]
    )
    assert int(torch.max(counts) - torch.min(counts)) <= 1
    assert not torch.equal(
        terrain.terrain_levels[:SLOPE_COUNT],
        torch.tensor(SLOPE_TERRAIN_LEVELS),
    )
    torch.testing.assert_close(
        terrain.env_origins,
        terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
    )
