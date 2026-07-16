"""Canonical slope grid, terrain indexing, and evaluation sizing."""

from __future__ import annotations

import math
from typing import Final


SLOPE_PERCENT_MIN: Final[int] = -8
SLOPE_PERCENT_MAX: Final[int] = 10
SLOPE_PERCENT_STEP: Final[int] = 1
SLOPE_PERCENTAGES: Final[tuple[int, ...]] = tuple(
    range(SLOPE_PERCENT_MIN, SLOPE_PERCENT_MAX + 1, SLOPE_PERCENT_STEP)
)
SLOPE_GRADIENTS: Final[tuple[float, ...]] = tuple(
    value / 100.0 for value in SLOPE_PERCENTAGES
)
SLOPE_LABELS: Final[tuple[str, ...]] = tuple(
    f"{gradient:+.2f}" for gradient in SLOPE_GRADIENTS
)
SLOPE_COUNT: Final[int] = len(SLOPE_GRADIENTS)
TERRAIN_COLUMNS_PER_TYPE: Final[int] = 9
FLAT_TERRAIN_TYPE: Final[int] = 0
UPHILL_TERRAIN_TYPE: Final[int] = TERRAIN_COLUMNS_PER_TYPE
DOWNHILL_TERRAIN_TYPE: Final[int] = 2 * TERRAIN_COLUMNS_PER_TYPE
TERRAIN_NUM_COLS: Final[int] = 3 * TERRAIN_COLUMNS_PER_TYPE
TERRAIN_NUM_ROWS: Final[int] = max(abs(SLOPE_PERCENT_MIN), abs(SLOPE_PERCENT_MAX))
MAX_TRAINING_DOWNHILL_LEVEL: Final[int] = abs(SLOPE_PERCENT_MIN) - 1

FORMAL_EVALUATION_ENVS_PER_SLOPE: Final[int] = 20
FORMAL_EVALUATION_NUM_ENVS: Final[int] = (
    SLOPE_COUNT * FORMAL_EVALUATION_ENVS_PER_SLOPE
)


def slope_percent(gradient: float) -> int:
    """Convert a configured gradient to its exact integer percentage."""

    value = float(gradient)
    if not math.isfinite(value):
        raise ValueError("slope gradient must be finite")
    percent = round(value * 100.0)
    if not math.isclose(value, percent / 100.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"slope gradient is not on the 0.01 grid: {gradient}")
    return int(percent)


def terrain_index_for_gradient(gradient: float) -> tuple[int, int]:
    """Return the terrain level and type column for one configured slope."""

    percent = slope_percent(gradient)
    if percent not in SLOPE_PERCENTAGES:
        raise ValueError(f"slope gradient is outside the configured grid: {gradient}")
    if percent == 0:
        return 0, FLAT_TERRAIN_TYPE
    level = abs(percent) - 1
    terrain_type = UPHILL_TERRAIN_TYPE if percent > 0 else DOWNHILL_TERRAIN_TYPE
    return level, terrain_type


def balanced_slope_counts(num_envs: int) -> tuple[int, ...]:
    """Distribute environments deterministically over the configured slopes."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    base_count, extra_count = divmod(num_envs, SLOPE_COUNT)
    return tuple(
        base_count + (1 if index < extra_count else 0)
        for index in range(SLOPE_COUNT)
    )


SLOPE_TERRAIN_INDICES: Final[tuple[tuple[int, int], ...]] = tuple(
    terrain_index_for_gradient(gradient) for gradient in SLOPE_GRADIENTS
)
SLOPE_TERRAIN_LEVELS: Final[tuple[int, ...]] = tuple(
    level for level, _terrain_type in SLOPE_TERRAIN_INDICES
)
SLOPE_TERRAIN_TYPES: Final[tuple[int, ...]] = tuple(
    terrain_type for _level, terrain_type in SLOPE_TERRAIN_INDICES
)


__all__ = [
    "DOWNHILL_TERRAIN_TYPE",
    "FLAT_TERRAIN_TYPE",
    "FORMAL_EVALUATION_ENVS_PER_SLOPE",
    "FORMAL_EVALUATION_NUM_ENVS",
    "MAX_TRAINING_DOWNHILL_LEVEL",
    "SLOPE_COUNT",
    "SLOPE_GRADIENTS",
    "SLOPE_LABELS",
    "SLOPE_PERCENTAGES",
    "SLOPE_PERCENT_MAX",
    "SLOPE_PERCENT_MIN",
    "SLOPE_PERCENT_STEP",
    "SLOPE_TERRAIN_INDICES",
    "SLOPE_TERRAIN_LEVELS",
    "SLOPE_TERRAIN_TYPES",
    "TERRAIN_COLUMNS_PER_TYPE",
    "TERRAIN_NUM_COLS",
    "TERRAIN_NUM_ROWS",
    "UPHILL_TERRAIN_TYPE",
    "balanced_slope_counts",
    "slope_percent",
    "terrain_index_for_gradient",
]
