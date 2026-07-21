"""Simulator-independent physical specifications for the rickshaw task."""

from __future__ import annotations

from dataclasses import MISSING, dataclass

from g1_rickshaw_lab.rickshaw_spec import HITCH_HALF_WIDTH, HITCH_X, HITCH_Z, WHEEL_RADIUS


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


__all__ = ["RickshawPoseTargetCfg"]
