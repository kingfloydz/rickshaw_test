"""Regression tests for startup-fixed balanced slopes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    TERRAIN_NUM_COLS,
    TERRAIN_NUM_ROWS,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.curricula import (
    assign_terrain_slopes,
    balanced_slope_assignment,
    randomize_startup_slopes,
    speed_command_levels,
    weighted_slope_assignment,
)


class _RewardManager:
    def __init__(self, reward: float, episode_length_s: float) -> None:
        self._episode_sums = {
            "track_speed_exp": torch.tensor([reward * episode_length_s])
        }

    @staticmethod
    def get_term_cfg(_name: str) -> SimpleNamespace:
        return SimpleNamespace(weight=1.0)


def test_speed_command_curriculum_matches_unitree_threshold_and_step() -> None:
    command_cfg = SimpleNamespace(maximum=0.1, limit_maximum=0.3, curriculum_step=0.1)
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            policy_update=SimpleNamespace(command_sampling=command_cfg),
        ),
        reward_manager=_RewardManager(reward=0.81, episode_length_s=20.0),
        max_episode_length_s=20.0,
        max_episode_length=1000,
        common_step_counter=1000,
        device="cpu",
    )

    level = speed_command_levels(env, torch.tensor([0]))

    assert command_cfg.maximum == 0.2
    torch.testing.assert_close(level, torch.tensor(0.2))


def test_balanced_assignment_covers_every_configured_slope() -> None:
    slots, levels, terrain_types = balanced_slope_assignment(
        2 * SLOPE_COUNT + 7, device="cpu"
    )

    assert slots[:SLOPE_COUNT].tolist() == list(range(SLOPE_COUNT))
    assert levels[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_LEVELS)
    assert terrain_types[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_TYPES)
    counts = torch.bincount(slots, minlength=SLOPE_COUNT)
    assert int(torch.max(counts) - torch.min(counts)) <= 1


def test_training_slope_assignment_tapers_from_level_ground() -> None:
    slots, _, _ = weighted_slope_assignment(8192, device="cpu")
    counts = torch.bincount(slots, minlength=SLOPE_COUNT)
    flat_index = 8

    assert torch.all(torch.diff(counts[: flat_index + 1]) >= 0)
    assert torch.all(torch.diff(counts[flat_index:]) <= 0)
    assert counts[flat_index] / counts[0] == pytest.approx(2.0, rel=0.02)
    assert counts[flat_index] / counts[-1] == pytest.approx(2.0, rel=0.02)


def test_startup_randomizes_environment_mapping_without_changing_distribution() -> None:
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
    expected_slots, _, _ = weighted_slope_assignment(num_envs, device="cpu")
    torch.testing.assert_close(
        counts, torch.bincount(expected_slots, minlength=SLOPE_COUNT)
    )
    assert not torch.equal(
        terrain.terrain_levels[:SLOPE_COUNT],
        torch.tensor(SLOPE_TERRAIN_LEVELS),
    )
    torch.testing.assert_close(
        terrain.env_origins,
        terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
    )


def test_startup_can_keep_canonical_environment_order_for_rendering() -> None:
    num_envs = SLOPE_COUNT
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

    randomize_startup_slopes(env, None, shuffle=False)

    assert terrain.terrain_levels.tolist() == list(SLOPE_TERRAIN_LEVELS)
    assert terrain.terrain_types.tolist() == list(SLOPE_TERRAIN_TYPES)


def test_explicit_slope_assignment_uses_the_shared_runtime_writer() -> None:
    num_envs = 3
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

    assign_terrain_slopes(env, (-0.08, 0.0, 0.10))

    torch.testing.assert_close(
        terrain.env_origins,
        terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
    )
