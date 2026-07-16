"""Pure mechanical specification shared by geometry and simulation code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RickshawUrdfSpec:
    """Guide-defined mechanical source of truth in SI units."""

    base_mass: float = 36.0
    base_inertia_diagonal: tuple[float, float, float] = (7.393572, 22.277208, 17.829456)
    wheel_mass: float = 2.0
    wheel_inertia_diagonal: tuple[float, float, float] = (0.071184, 0.140624, 0.071184)
    hitch_link_mass: float = 0.02
    total_mass: float = 40.04
    center_of_mass: tuple[float, float, float] = (
        0.651664276415584,
        0.0,
        0.669432082993007,
    )
    wheel_radius: float = 0.374999
    wheel_width: float = 0.072548
    wheel_track: float = 0.756462
    wheel_joint_damping: float = 0.02
    wheel_joint_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    # These connection frames are a normative geometry ABI in the guide.
    hitch_x: float = 1.85049373
    hitch_z: float = 0.18164719
    hitch_half_width: float = 0.235


RICKSHAW_URDF_SPEC = RickshawUrdfSpec()
RICKSHAW_TOTAL_MASS = RICKSHAW_URDF_SPEC.total_mass
RICKSHAW_CENTER_OF_MASS = RICKSHAW_URDF_SPEC.center_of_mass
WHEEL_RADIUS = RICKSHAW_URDF_SPEC.wheel_radius
WHEEL_WIDTH = RICKSHAW_URDF_SPEC.wheel_width
WHEEL_TRACK = RICKSHAW_URDF_SPEC.wheel_track
WHEEL_JOINT_DAMPING = RICKSHAW_URDF_SPEC.wheel_joint_damping
HITCH_X = RICKSHAW_URDF_SPEC.hitch_x
HITCH_Z = RICKSHAW_URDF_SPEC.hitch_z
HITCH_HALF_WIDTH = RICKSHAW_URDF_SPEC.hitch_half_width


__all__ = [
    "HITCH_HALF_WIDTH",
    "HITCH_X",
    "HITCH_Z",
    "RICKSHAW_CENTER_OF_MASS",
    "RICKSHAW_TOTAL_MASS",
    "RICKSHAW_URDF_SPEC",
    "RickshawUrdfSpec",
    "WHEEL_JOINT_DAMPING",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
    "WHEEL_WIDTH",
]
