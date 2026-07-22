"""Canonical slope-grid and terrain-index contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import g1_rickshaw_lab.static_equilibrium as static_equilibrium
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
    counts = balanced_slope_counts(8192)
    assert counts == (432,) * 3 + (431,) * 16
    assert sum(counts) == 8192

    with pytest.raises(ValueError, match="positive integer"):
        balanced_slope_counts(0)


def test_static_library_continues_from_zero_independently_by_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, float | None]] = []

    def solve(
        _model: object,
        gradient: float,
        *,
        cfg: object = None,
        qpos_seed: np.ndarray | None = None,
    ):
        del cfg
        calls.append((gradient, None if qpos_seed is None else float(qpos_seed[0])))
        return SimpleNamespace(gradient=gradient, qpos=np.array([gradient]))

    monkeypatch.setattr(static_equilibrium, "solve_mujoco_static_equilibrium", solve)
    result = static_equilibrium.solve_mujoco_static_equilibria(
        object(), SLOPE_GRADIENTS
    )

    positive = tuple(gradient for gradient in SLOPE_GRADIENTS if gradient > 0.0)
    negative = tuple(
        sorted(
            (gradient for gradient in SLOPE_GRADIENTS if gradient < 0.0), reverse=True
        )
    )
    assert tuple(gradient for gradient, _seed in calls) == (0.0, *positive, *negative)
    assert calls[0][1] is None
    assert tuple(seed for _gradient, seed in calls[1 : 1 + len(positive)]) == (
        0.0,
        *positive[:-1],
    )
    assert tuple(seed for _gradient, seed in calls[1 + len(positive) :]) == (
        0.0,
        *negative[:-1],
    )
    assert tuple(solution.gradient for solution in result) == SLOPE_GRADIENTS
