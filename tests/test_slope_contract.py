"""Canonical slope-grid and terrain-index contracts."""

from __future__ import annotations

import pytest

from g1_rickshaw_lab.slope_contract import (
    FORMAL_EVALUATION_ENVS_PER_SLOPE,
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_COUNT,
    SLOPE_GRADIENTS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    balanced_slope_counts,
    terrain_index_for_gradient,
)


def test_slope_grid_drives_terrain_and_evaluation_dimensions() -> None:
    assert SLOPE_GRADIENTS == tuple(value / 100.0 for value in range(-8, 11))
    assert SLOPE_COUNT == 19
    assert len(SLOPE_TERRAIN_LEVELS) == SLOPE_COUNT
    assert len(SLOPE_TERRAIN_TYPES) == SLOPE_COUNT
    assert FORMAL_EVALUATION_NUM_ENVS == (
        SLOPE_COUNT * FORMAL_EVALUATION_ENVS_PER_SLOPE
    )


@pytest.mark.parametrize(
    ("gradient", "expected"),
    ((-0.08, (7, 18)), (0.0, (0, 0)), (0.08, (7, 9)), (0.10, (9, 9))),
)
def test_terrain_index_for_gradient(gradient: float, expected: tuple[int, int]) -> None:
    assert terrain_index_for_gradient(gradient) == expected


@pytest.mark.parametrize("gradient", (-0.09, 0.105, 0.11, float("nan")))
def test_terrain_index_rejects_unconfigured_gradient(gradient: float) -> None:
    with pytest.raises(ValueError):
        terrain_index_for_gradient(gradient)


def test_balanced_slope_counts_are_deterministic() -> None:
    counts = balanced_slope_counts(4096)
    assert counts == (216,) * 11 + (215,) * 8
    assert sum(counts) == 4096

    with pytest.raises(ValueError, match="positive integer"):
        balanced_slope_counts(0)
