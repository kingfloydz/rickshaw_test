"""MuJoCo/mjlab rickshaw asset and source-URDF validation."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from ..project_paths import ASSET_ROOT
from ..rickshaw_spec import (
    HITCH_HALF_WIDTH,
    HITCH_X,
    HITCH_Z,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    RICKSHAW_URDF_SPEC,
    WHEEL_JOINT_DAMPING,
    WHEEL_RADIUS,
    WHEEL_TRACK,
    WHEEL_WIDTH,
    RickshawUrdfSpec,
)
from .mujoco_spec import (
    GROUND_COLLISION_BIT,
    RICKSHAW_COLLISION_BIT,
    ROBOT_COLLISION_BIT,
    add_free_joint,
    load_urdf_spec,
)

RICKSHAW_ASSET_DIR = ASSET_ROOT / "rickshaw"
RICKSHAW_URDF_PATH = RICKSHAW_ASSET_DIR / "rickshaw.urdf"
RICKSHAW_URDF = str(RICKSHAW_URDF_PATH)
RICKSHAW_MESH_PATHS = tuple(RICKSHAW_ASSET_DIR / name for name in ("body.stl", "left_wheel.stl", "right_wheel.stl"))

BASE_LINK_NAME = "base_link"
WHEEL_LINK_NAMES = ("left_wheel_link", "right_wheel_link")
WHEEL_JOINT_NAMES = ("left_wheel_joint", "right_wheel_joint")
HITCH_LINK_NAMES = ("left_tow_hitch_link", "right_tow_hitch_link")
HITCH_JOINT_NAMES = ("left_tow_hitch_joint", "right_tow_hitch_joint")
HITCH_SITE_NAMES = ("left_hitch_site", "right_hitch_site")
TOW_ROD_COLLISION_GEOM_NAMES = ("left_tow_rod_collision", "right_tow_rod_collision")


class RickshawAssetValidationError(ValueError):
    pass


def _vector(element: ET.Element | None, attribute: str) -> tuple[float, ...]:
    if element is None:
        raise ValueError("missing XML element")
    return tuple(float(value) for value in element.attrib[attribute].split())


def validate_rickshaw_urdf(path: str | Path = RICKSHAW_URDF_PATH, *, tolerance: float = 1.0e-6) -> tuple[str, ...]:
    """Validate the geometry and inertia values that MuJoCo consumes."""

    root = ET.parse(Path(path)).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    spec = RICKSHAW_URDF_SPEC
    issues: list[str] = []

    def scalar(label: str, actual: float, expected: float) -> None:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            issues.append(f"{label}: expected {expected}, got {actual}")

    required_links = (BASE_LINK_NAME, *WHEEL_LINK_NAMES, *HITCH_LINK_NAMES)
    required_joints = (*WHEEL_JOINT_NAMES, *HITCH_JOINT_NAMES)
    for name in required_links:
        if name not in links:
            issues.append(f"missing link: {name}")
    for name in required_joints:
        if name not in joints:
            issues.append(f"missing joint: {name}")
    if issues:
        return tuple(issues)

    base_origin = _vector(links[BASE_LINK_NAME].find("inertial/origin"), "xyz")
    scalar("base CoM x", base_origin[0], spec.base_com_x)
    scalar(
        "base CoM rearward shift",
        spec.base_com_x_before_shift - base_origin[0],
        spec.center_of_mass_rearward_shift,
    )
    scalar("base CoM z", base_origin[2], 0.6276898532066667)

    for name in WHEEL_LINK_NAMES:
        inertia = links[name].find("inertial/inertia")
        for axis, expected in zip(("ixx", "iyy", "izz"), spec.wheel_inertia_diagonal, strict=True):
            scalar(f"{name} {axis}", float(inertia.attrib[axis]), expected)  # type: ignore[union-attr]
        cylinder = links[name].find("collision/geometry/cylinder")
        scalar(f"{name} radius", float(cylinder.attrib["radius"]), spec.wheel_radius)  # type: ignore[union-attr]
        scalar(f"{name} width", float(cylinder.attrib["length"]), spec.wheel_width)  # type: ignore[union-attr]

    expected_wheels = {
        "left_wheel_joint": (0.0, spec.wheel_track / 2.0, spec.wheel_radius),
        "right_wheel_joint": (0.0, -spec.wheel_track / 2.0, spec.wheel_radius),
    }
    for name, expected in expected_wheels.items():
        actual = _vector(joints[name].find("origin"), "xyz")
        for axis, (value, target) in enumerate(zip(actual, expected, strict=True)):
            scalar(f"{name} origin[{axis}]", value, target)
        scalar(f"{name} damping", float(joints[name].find("dynamics").attrib["damping"]), spec.wheel_joint_damping)  # type: ignore[union-attr]

    expected_hitches = {
        "left_tow_hitch_joint": (spec.hitch_x, spec.hitch_half_width, spec.hitch_z),
        "right_tow_hitch_joint": (spec.hitch_x, -spec.hitch_half_width, spec.hitch_z),
    }
    for name, expected in expected_hitches.items():
        if joints[name].attrib.get("type") != "fixed":
            issues.append(f"{name} must be fixed")
        actual = _vector(joints[name].find("origin"), "xyz")
        for axis, (value, target) in enumerate(zip(actual, expected, strict=True)):
            scalar(f"{name} origin[{axis}]", value, target)

    masses_and_positions = [(36.0, base_origin)]
    masses_and_positions.extend((2.0, expected_wheels[name]) for name in WHEEL_JOINT_NAMES)
    masses_and_positions.extend((0.02, expected_hitches[name]) for name in HITCH_JOINT_NAMES)
    total_mass = sum(mass for mass, _ in masses_and_positions)
    center = tuple(
        sum(mass * position[axis] for mass, position in masses_and_positions) / total_mass for axis in range(3)
    )
    scalar("total mass", total_mass, spec.total_mass)
    for axis, (value, target) in enumerate(zip(center, spec.center_of_mass, strict=True)):
        scalar(f"center of mass[{axis}]", value, target)
    return tuple(issues)


def assert_valid_rickshaw_urdf(path: str | Path = RICKSHAW_URDF_PATH) -> None:
    issues = validate_rickshaw_urdf(path)
    if issues:
        raise RickshawAssetValidationError("invalid rickshaw URDF: " + "; ".join(issues))


def get_rickshaw_spec() -> mujoco.MjSpec:
    """Build the passive two-wheel rickshaw and its two hitch sites."""

    assert_valid_rickshaw_urdf()
    spec = load_urdf_spec(RICKSHAW_URDF_PATH)
    spec.compiler.discardvisual = 0
    add_free_joint(spec, BASE_LINK_NAME)
    for geom in spec.geoms:
        geom.contype = RICKSHAW_COLLISION_BIT
        geom.conaffinity = GROUND_COLLISION_BIT
        geom.group = 3
    for name, lateral in zip(TOW_ROD_COLLISION_GEOM_NAMES, (0.276, -0.276), strict=True):
        spec.body(BASE_LINK_NAME).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=(0.016, 0.0, 0.0),
            fromto=(0.676, lateral, 0.214, 1.94034, lateral, 0.105747),
            contype=RICKSHAW_COLLISION_BIT,
            conaffinity=ROBOT_COLLISION_BIT,
            group=3,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
    visual_meshes = (
        (
            BASE_LINK_NAME,
            "body_visual",
            "body.stl",
            (0.0001, 0.0001, 0.0001),
            (1.94034, -0.414504, -0.074999),
            (0.7071067811882787, 0.0, 0.0, 0.7071067811848163),
            (0.18, 0.004, 0.008, 1.0),
        ),
        (
            WHEEL_LINK_NAMES[0],
            "left_wheel_visual",
            "right_wheel.stl",
            (0.0001, 0.00008, 0.00008),
            (1.552272, -0.792735, -0.3),
            (0.7071067811882787, 0.0, 0.0, 0.7071067811848163),
            (0.05, 0.05, 0.05, 1.0),
        ),
        (
            WHEEL_LINK_NAMES[1],
            "right_wheel_visual",
            "left_wheel.stl",
            (0.0001, 0.00008, 0.00008),
            (1.552272, -0.036274, -0.3),
            (0.7071067811882787, 0.0, 0.0, 0.7071067811848163),
            (0.05, 0.05, 0.05, 1.0),
        ),
    )
    for body_name, name, filename, scale, pos, quat, rgba in visual_meshes:
        mesh = spec.add_mesh(name=f"{name}_mesh", file=filename, scale=scale)
        spec.body(body_name).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=pos,
            quat=quat,
            contype=0,
            conaffinity=0,
            group=0,
            rgba=rgba,
        )
    for body_name, site_name in zip(HITCH_LINK_NAMES, HITCH_SITE_NAMES, strict=True):
        spec.body(body_name).add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.006, 0.0, 0.0),
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
    return spec


def get_rickshaw_cfg():
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=get_rickshaw_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(-1.664929, 0.0, 0.0),
            joint_pos={r".*_wheel_joint": 0.0},
            joint_vel={r".*_wheel_joint": 0.0},
        ),
    )


build_rickshaw_cfg = get_rickshaw_cfg
RICKSHAW_CFG = None


def missing_rickshaw_assets() -> tuple[Path, ...]:
    return tuple(path for path in (RICKSHAW_URDF_PATH, *RICKSHAW_MESH_PATHS) if not path.is_file())


__all__ = [
    "BASE_LINK_NAME",
    "HITCH_HALF_WIDTH",
    "HITCH_LINK_NAMES",
    "HITCH_SITE_NAMES",
    "TOW_ROD_COLLISION_GEOM_NAMES",
    "HITCH_X",
    "HITCH_Z",
    "RICKSHAW_ASSET_DIR",
    "RICKSHAW_CENTER_OF_MASS",
    "RICKSHAW_CFG",
    "RICKSHAW_TOTAL_MASS",
    "RICKSHAW_URDF",
    "RICKSHAW_URDF_PATH",
    "RICKSHAW_URDF_SPEC",
    "RickshawAssetValidationError",
    "RickshawUrdfSpec",
    "WHEEL_JOINT_DAMPING",
    "WHEEL_JOINT_NAMES",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
    "WHEEL_WIDTH",
    "assert_valid_rickshaw_urdf",
    "build_rickshaw_cfg",
    "get_rickshaw_cfg",
    "get_rickshaw_spec",
    "missing_rickshaw_assets",
    "validate_rickshaw_urdf",
]
