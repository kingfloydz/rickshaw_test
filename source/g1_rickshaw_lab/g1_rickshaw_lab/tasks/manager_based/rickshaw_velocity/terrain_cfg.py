"""Directional-slope terrain and rickshaw reset geometry for Mjlab."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass
from typing import Protocol

import numpy as np

from g1_rickshaw_lab.assets.rickshaw import HITCH_HALF_WIDTH, HITCH_X, HITCH_Z, WHEEL_RADIUS
from g1_rickshaw_lab.slope_contract import (
    TERRAIN_NUM_COLS,
    TERRAIN_NUM_ROWS,
)

TERRAIN_SEED = 42
TERRAIN_SIZE = (26.0, 6.0)
TERRAIN_SPAWN_X = 4.0


def _difficulty_level(difficulty: float) -> int:
    difficulty = float(difficulty)
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be finite and in [0, 1], got {difficulty}")
    return min(int(difficulty * TERRAIN_NUM_ROWS), TERRAIN_NUM_ROWS - 1)


def directional_slope_gradient(difficulty: float, direction: int) -> tuple[float, int]:
    """Map normalized difficulty and direction to one guide-defined gradient."""

    if direction not in (-1, 0, 1):
        raise ValueError(f"direction must be -1, 0, or 1, got {direction}")
    level = _difficulty_level(difficulty)
    return direction * (0.01 + 0.01 * level), level


def make_mjlab_directional_slopes_cfg():
    """Build the exact 10-by-27 directional slope grid for mjlab."""

    import mujoco
    from mjlab.terrains.terrain_generator import (
        SubTerrainCfg,
        TerrainGeneratorCfg,
        TerrainGeometry,
        TerrainOutput,
    )

    @dataclass(kw_only=True)
    class MjlabDirectionalPlaneSlopeCfg(SubTerrainCfg):
        direction: int = 0
        spawn_x: float = TERRAIN_SPAWN_X
        thickness: float = 1.0

        def function(self, difficulty, spec, rng):
            del rng
            gradient, _ = directional_slope_gradient(difficulty, self.direction)
            gamma = math.atan(gradient)
            length, width = self.size
            half_length = length / (2.0 * math.cos(gamma))
            surface_center = np.array(
                (length / 2.0, width / 2.0, gradient * (length / 2.0 - self.spawn_x)),
                dtype=np.float64,
            )
            normal = np.array((-math.sin(gamma), 0.0, math.cos(gamma)), dtype=np.float64)
            geom = spec.body("terrain").add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(half_length, width / 2.0, self.thickness / 2.0),
                pos=surface_center - 0.5 * self.thickness * normal,
                quat=(math.cos(0.5 * gamma), 0.0, -math.sin(0.5 * gamma), 0.0),
            )
            return TerrainOutput(
                origin=np.array((self.spawn_x, width / 2.0, 0.0), dtype=np.float64),
                geometries=[TerrainGeometry(geom=geom, color=(0.45, 0.48, 0.46, 1.0))],
            )

    return TerrainGeneratorCfg(
        seed=TERRAIN_SEED,
        curriculum=True,
        size=TERRAIN_SIZE,
        num_rows=TERRAIN_NUM_ROWS,
        num_cols=TERRAIN_NUM_COLS,
        border_width=0.0,
        color_scheme="none",
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            "flat": MjlabDirectionalPlaneSlopeCfg(proportion=1.0 / 3.0, direction=0),
            "uphill": MjlabDirectionalPlaneSlopeCfg(proportion=1.0 / 3.0, direction=1),
            "downhill": MjlabDirectionalPlaneSlopeCfg(proportion=1.0 / 3.0, direction=-1),
        },
    )


class _RickshawPoseLike(Protocol):
    wheel_radius: float
    hitch_x: float
    hitch_z: float
    hitch_half_width: float
    hitch_height_target: float


@dataclass(kw_only=True)
class RickshawPoseTargetCfg:
    """Rickshaw front-lift target and reset acceptance tolerances."""

    wheel_radius: float = WHEEL_RADIUS
    hitch_x: float = HITCH_X
    hitch_z: float = HITCH_Z
    hitch_half_width: float = HITCH_HALF_WIDTH
    hitch_height_target: float = MISSING
    hitch_height_tolerance: float = MISSING
    hitch_vertical_speed_tolerance: float = MISSING


def _validate_pose_dimensions(cfg: _RickshawPoseLike) -> tuple[float, float, float, float]:
    wheel_radius = float(cfg.wheel_radius)
    hitch_x = float(cfg.hitch_x)
    hitch_z = float(cfg.hitch_z)
    target = float(cfg.hitch_height_target)
    values = (wheel_radius, hitch_x, hitch_z, target)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rickshaw pose dimensions and target must be finite")
    if wheel_radius <= 0.0 or hitch_x <= 0.0:
        raise ValueError("wheel_radius and hitch_x must be positive")
    return values


def target_pitch_from_hitch_height(cfg: _RickshawPoseLike) -> float:
    """Solve the positive front-lift angle alpha from target hitch height."""

    wheel_radius, hitch_x, hitch_z, target = _validate_pose_dimensions(cfg)
    radius = math.hypot(hitch_x, hitch_z - wheel_radius)
    phase = math.atan2(hitch_z - wheel_radius, hitch_x)
    ratio = (target - wheel_radius) / radius
    if not -1.0 <= ratio <= 1.0:
        minimum = wheel_radius - radius
        maximum = wheel_radius + radius
        raise ValueError(
            f"infeasible hitch_height_target={target}; reachable range is "
            f"[{minimum}, {maximum}]"
        )
    return math.asin(ratio) - phase


def hitch_height_from_pitch(alpha: float, cfg: _RickshawPoseLike) -> float:
    """Forward geometry H(alpha) measured along the terrain normal."""

    wheel_radius, hitch_x, hitch_z, _ = _validate_pose_dimensions(cfg)
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return (
        wheel_radius
        + hitch_x * math.sin(alpha)
        + (hitch_z - wheel_radius) * math.cos(alpha)
    )


def cart_root_height_from_pitch(alpha: float, cfg: _RickshawPoseLike) -> float:
    """Base/root normal offset that keeps both wheel centers at radius height."""

    wheel_radius, _, _, _ = _validate_pose_dimensions(cfg)
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return wheel_radius * (1.0 - math.cos(alpha))


def hitch_height_round_trip_error(cfg: _RickshawPoseLike) -> float:
    """Compute the acceptance invariant ``H -> alpha -> H`` error."""

    alpha = target_pitch_from_hitch_height(cfg)
    return abs(hitch_height_from_pitch(alpha, cfg) - float(cfg.hitch_height_target))


__all__ = [
    "RickshawPoseTargetCfg",
    "TERRAIN_NUM_COLS",
    "TERRAIN_NUM_ROWS",
    "TERRAIN_SEED",
    "TERRAIN_SIZE",
    "TERRAIN_SPAWN_X",
    "cart_root_height_from_pitch",
    "directional_slope_gradient",
    "hitch_height_from_pitch",
    "hitch_height_round_trip_error",
    "make_mjlab_directional_slopes_cfg",
    "target_pitch_from_hitch_height",
]
