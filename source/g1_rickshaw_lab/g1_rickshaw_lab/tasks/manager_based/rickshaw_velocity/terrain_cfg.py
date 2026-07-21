"""Directional-slope terrain and rickshaw reset geometry for Mjlab."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
import math
from typing import Protocol

import numpy as np

from g1_rickshaw_lab.assets.rickshaw import HITCH_HALF_WIDTH, HITCH_X, HITCH_Z, WHEEL_RADIUS
from g1_rickshaw_lab.slope_contract import (
    TERRAIN_COLUMNS_PER_TYPE,
    TERRAIN_NUM_COLS,
    TERRAIN_NUM_ROWS,
)


TERRAIN_SEED = 42
TERRAIN_SIZE = (26.0, 6.0)
TERRAIN_SPAWN_X = 4.0
TERRAIN_GRADIENT_MAGNITUDES = tuple(
    0.01 * level for level in range(1, TERRAIN_NUM_ROWS + 1)
)
ALL_SIGNED_TERRAIN_GRADIENTS = (
    0.0,
    *TERRAIN_GRADIENT_MAGNITUDES,
    *(-value for value in TERRAIN_GRADIENT_MAGNITUDES),
)


@dataclass(frozen=True)
class DirectionalSlopeGeometry:
    """Watertight box-under-plane mesh data in local patch coordinates."""

    vertices: np.ndarray
    faces: np.ndarray
    origin: np.ndarray
    gradient: float
    level: int


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


def directional_plane_slope_geometry(difficulty: float, cfg) -> DirectionalSlopeGeometry:
    """Generate the exact guide mesh and spawn origin without trimesh."""

    try:
        length, width = (float(value) for value in cfg.size)
        direction = int(cfg.direction)
        spawn_x = float(cfg.spawn_x)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("cfg must expose size=(length, width), direction, and spawn_x") from exc

    if not (math.isfinite(length) and math.isfinite(width) and length > 0.0 and width > 0.0):
        raise ValueError(f"terrain size must be finite and positive, got {(length, width)}")
    if not math.isfinite(spawn_x) or not 0.0 <= spawn_x <= length:
        raise ValueError(f"spawn_x must lie in [0, {length}], got {spawn_x}")

    slope, level = directional_slope_gradient(difficulty, direction)
    z0 = -slope * spawn_x
    z1 = slope * (length - spawn_x)
    bottom_height = min(z0, z1) - 1.0

    vertices = np.array(
        [
            [0.0, 0.0, z0],
            [length, 0.0, z1],
            [length, width, z1],
            [0.0, width, z0],
            [0.0, 0.0, bottom_height],
            [length, 0.0, bottom_height],
            [length, width, bottom_height],
            [0.0, width, bottom_height],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    origin = np.array([spawn_x, width / 2.0, 0.0], dtype=np.float64)
    return DirectionalSlopeGeometry(vertices, faces, origin, slope, level)


def directional_plane_slope(difficulty: float, cfg):
    """Return the legacy mesh representation used by offline geometry checks."""

    geometry = directional_plane_slope_geometry(difficulty, cfg)
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise RuntimeError("directional_plane_slope requires trimesh") from exc
    mesh = trimesh.Trimesh(
        vertices=geometry.vertices,
        faces=geometry.faces,
        process=False,
    )
    return [mesh], geometry.origin


@dataclass(kw_only=True)
class DirectionalPlaneSlopeCfg:
    """Dependency-free parameters for directional-plane geometry checks."""

    proportion: float = 1.0
    size: tuple[float, float] = TERRAIN_SIZE
    direction: int = 0
    spawn_x: float = TERRAIN_SPAWN_X

    function = staticmethod(directional_plane_slope)


@dataclass(frozen=True)
class DirectionalSlopesMetadata:
    seed: int
    curriculum: bool
    size: tuple[float, float]
    num_rows: int
    num_cols: int
    border_width: float
    use_cache: bool
    sub_terrains: dict[str, DirectionalPlaneSlopeCfg]


DIRECTIONAL_SLOPES_CFG = DirectionalSlopesMetadata(
    seed=TERRAIN_SEED,
    curriculum=True,
    size=TERRAIN_SIZE,
    num_rows=TERRAIN_NUM_ROWS,
    num_cols=TERRAIN_NUM_COLS,
    border_width=0.0,
    use_cache=False,
    sub_terrains={
        "flat": DirectionalPlaneSlopeCfg(proportion=1.0 / 3.0, direction=0),
        "uphill": DirectionalPlaneSlopeCfg(proportion=1.0 / 3.0, direction=1),
        "downhill": DirectionalPlaneSlopeCfg(proportion=1.0 / 3.0, direction=-1),
    },
)


def signed_gradient_from_terrain(levels, columns):
    """Vectorized 10-row/27-column terrain-level mapping.

    Columns 0..8 are flat, 9..17 uphill, and 18..26 downhill. The
    returned scalar/array is suitable for ``gamma = atan(gradient)``.
    """

    level_array = np.asarray(levels)
    column_array = np.asarray(columns)
    level_array, column_array = np.broadcast_arrays(level_array, column_array)
    if not np.all(np.isfinite(level_array)) or not np.all(level_array == np.floor(level_array)):
        raise ValueError("terrain levels must be finite integers")
    if not np.all(np.isfinite(column_array)) or not np.all(column_array == np.floor(column_array)):
        raise ValueError("terrain columns must be finite integers")
    if np.any((level_array < 0) | (level_array >= TERRAIN_NUM_ROWS)):
        raise ValueError(f"terrain levels must be in [0, {TERRAIN_NUM_ROWS - 1}]")
    if np.any((column_array < 0) | (column_array >= TERRAIN_NUM_COLS)):
        raise ValueError(f"terrain columns must be in [0, {TERRAIN_NUM_COLS - 1}]")

    magnitude = 0.01 + 0.01 * level_array.astype(np.float64)
    sign = np.where(
        column_array < TERRAIN_COLUMNS_PER_TYPE,
        0.0,
        np.where(column_array < 2 * TERRAIN_COLUMNS_PER_TYPE, 1.0, -1.0),
    )
    gradient = sign * magnitude
    return float(gradient) if gradient.ndim == 0 else gradient


@dataclass(frozen=True)
class SlopeFrame:
    """Orthonormal path frame with rotation-matrix columns (e_s, e_y, e_n)."""

    gradient: np.ndarray
    gamma: np.ndarray
    tangent: np.ndarray
    lateral: np.ndarray
    normal: np.ndarray
    rotation_matrix: np.ndarray


def slope_frame_from_gradient(gradient) -> SlopeFrame:
    """Construct the exact +X path frame for scalar or array gradients."""

    gradient_array = np.asarray(gradient, dtype=np.float64)
    if not np.all(np.isfinite(gradient_array)):
        raise ValueError("gradient must be finite")
    gamma = np.arctan(gradient_array)
    zeros = np.zeros_like(gamma)
    tangent = np.stack((np.cos(gamma), zeros, np.sin(gamma)), axis=-1)
    lateral = np.stack((zeros, np.ones_like(gamma), zeros), axis=-1)
    normal = np.stack((-np.sin(gamma), zeros, np.cos(gamma)), axis=-1)
    rotation_matrix = np.stack((tangent, lateral, normal), axis=-1)
    return SlopeFrame(gradient_array, gamma, tangent, lateral, normal, rotation_matrix)


def slope_frame_from_terrain(levels, columns) -> SlopeFrame:
    """Map terrain indices directly to the reset/path frame."""

    return slope_frame_from_gradient(signed_gradient_from_terrain(levels, columns))


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
    "ALL_SIGNED_TERRAIN_GRADIENTS",
    "DIRECTIONAL_SLOPES_CFG",
    "DirectionalPlaneSlopeCfg",
    "DirectionalSlopeGeometry",
    "RickshawPoseTargetCfg",
    "SlopeFrame",
    "TERRAIN_COLUMNS_PER_TYPE",
    "TERRAIN_GRADIENT_MAGNITUDES",
    "TERRAIN_NUM_COLS",
    "TERRAIN_NUM_ROWS",
    "TERRAIN_SEED",
    "TERRAIN_SIZE",
    "TERRAIN_SPAWN_X",
    "cart_root_height_from_pitch",
    "directional_plane_slope",
    "directional_plane_slope_geometry",
    "directional_slope_gradient",
    "hitch_height_from_pitch",
    "hitch_height_round_trip_error",
    "make_mjlab_directional_slopes_cfg",
    "signed_gradient_from_terrain",
    "slope_frame_from_gradient",
    "slope_frame_from_terrain",
    "target_pitch_from_hitch_height",
]
