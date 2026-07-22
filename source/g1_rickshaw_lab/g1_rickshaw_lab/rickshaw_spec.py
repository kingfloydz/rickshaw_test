"""Pure mechanical specification shared by geometry and simulation code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RickshawUrdfSpec:
    """Mechanical source of truth in the MuJoCo assembly frame."""

    base_mass: float = 36.0
    base_com_x_before_shift: float = 0.7427393855133334
    center_of_mass_rearward_shift: float = 0.6427393855133334
    base_com_x: float = 0.1
    base_inertia_diagonal: tuple[float, float, float] = (7.393572, 22.277208, 17.829456)
    wheel_mass: float = 2.0
    wheel_inertia_diagonal: tuple[float, float, float] = (0.04587720205066667, 0.09, 0.04587720205066667)
    hitch_link_mass: float = 0.02
    total_mass: float = 40.04
    center_of_mass: tuple[float, float, float] = (
        0.09157335564435565,
        0.0,
        0.5944321827032967,
    )
    wheel_radius: float = 0.3
    wheel_width: float = 0.072548
    wheel_track: float = 0.756462
    wheel_joint_damping: float = 0.02
    wheel_joint_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Given body.stl points are (lateral, longitudinal, vertical).  The mesh is
    # rotated into the +X-forward/+Y-left assembly frame, and the complete body
    # is lowered with the smaller wheels so that its axle stays concentric.
    body_stl_hitch_points: tuple[tuple[float, float, float], ...] = (
        (0.276, -1.664929, 0.180746),
        (-0.276, -1.664929, 0.180746),
    )
    body_vertical_offset: float = -0.074999
    hitch_x: float = 1.664929
    hitch_z: float = 0.105747
    hitch_half_width: float = 0.276


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
