"""Rickshaw asset configuration and source-URDF validation."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from ..rickshaw_spec import (
    HITCH_HALF_WIDTH,
    HITCH_X,
    HITCH_Z,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    RICKSHAW_URDF_SPEC,
    RickshawUrdfSpec,
    WHEEL_JOINT_DAMPING,
    WHEEL_RADIUS,
    WHEEL_TRACK,
    WHEEL_WIDTH,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RICKSHAW_ASSET_DIR = REPOSITORY_ROOT / "assets" / "rickshaw"
RICKSHAW_URDF_PATH = RICKSHAW_ASSET_DIR / "rickshaw.urdf"
RICKSHAW_USD_PATH = RICKSHAW_ASSET_DIR / "rickshaw.usd"
RICKSHAW_MESH_PATHS = (
    RICKSHAW_ASSET_DIR / "body.stl",
    RICKSHAW_ASSET_DIR / "left_wheel.stl",
    RICKSHAW_ASSET_DIR / "right_wheel.stl",
)

RICKSHAW_URDF = str(RICKSHAW_URDF_PATH)
RICKSHAW_USD = str(RICKSHAW_USD_PATH)

BASE_LINK_NAME = "base_link"
WHEEL_LINK_NAMES = ("left_wheel_link", "right_wheel_link")
WHEEL_JOINT_NAMES = ("left_wheel_joint", "right_wheel_joint")
HITCH_LINK_NAMES = ("left_tow_hitch_link", "right_tow_hitch_link")
HITCH_JOINT_NAMES = ("left_tow_hitch_joint", "right_tow_hitch_joint")


class IsaacLabUnavailableError(RuntimeError):
    """Raised when a simulation config is requested outside Isaac Lab."""


class RickshawAssetValidationError(ValueError):
    """Raised when the source URDF violates the implementation guide."""


try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.utils import configclass
except ModuleNotFoundError as exc:  # Expected for pure unit-test installs.
    sim_utils = None
    ArticulationCfg = None
    _ISAACLAB_IMPORT_ERROR: ModuleNotFoundError | None = exc

    def configclass(cls):
        return dataclass(cls, kw_only=True)

else:
    _ISAACLAB_IMPORT_ERROR = None


@configclass
class HandleConstraintCfg:
    """D6 drive/limit values that must come from handle calibration.

    The hitch-side local frame is identity. Dex-side local frames are calibrated
    grasp centers. A physically free rotation axis must not be assigned a drive.
    """

    linear_stiffness: float = MISSING
    linear_damping: float = MISSING
    angular_stiffness: float = MISSING
    angular_damping: float = MISSING
    max_force: float = MISSING
    max_torque: float = MISSING
    linear_limit: float = MISSING
    angular_limit: float = MISSING


HANDLE_CONSTRAINT_CALIBRATION_REQUIRED = (
    "linear_stiffness",
    "linear_damping",
    "angular_stiffness for each constrained rotation axis",
    "angular_damping for each constrained rotation axis",
    "max_force",
    "max_torque",
    "linear_limit",
    "angular_limit for each limited rotation axis",
    "left Dex D6 rigid-body prim",
    "left Dex grasp-center local pose",
    "right Dex D6 rigid-body prim",
    "right Dex grasp-center local pose",
)


def require_isaaclab() -> None:
    if ArticulationCfg is None:
        raise IsaacLabUnavailableError(
            "RICKSHAW_CFG requires Isaac Lab. Launch through Isaac Sim/Isaac Lab "
            "after installing the extension."
        ) from _ISAACLAB_IMPORT_ERROR


def build_rickshaw_cfg(*, require_usd: bool = False):
    """Build the passive-wheel rickshaw articulation configuration."""

    require_isaaclab()
    if require_usd and not RICKSHAW_USD_PATH.is_file():
        raise FileNotFoundError(
            f"Missing rickshaw USD: {RICKSHAW_USD_PATH}. Convert "
            f"{RICKSHAW_URDF_PATH} with --joint-target-type none."
        )

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Rickshaw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RICKSHAW_USD,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*_wheel_joint": 0.0},
            joint_vel={".*_wheel_joint": 0.0},
        ),
        actuators={},
    )


RICKSHAW_CFG = build_rickshaw_cfg() if ArticulationCfg is not None else None


def missing_rickshaw_assets() -> tuple[Path, ...]:
    """Return source/conversion artifacts missing from the final asset layout."""

    required = (RICKSHAW_URDF_PATH, *RICKSHAW_MESH_PATHS, RICKSHAW_USD_PATH)
    return tuple(path for path in required if not path.is_file())


def _parse_vector(value: str | None, *, length: int = 3) -> tuple[float, ...]:
    if value is None:
        raise ValueError("missing vector attribute")
    result = tuple(float(item) for item in value.split())
    if len(result) != length:
        raise ValueError(f"expected {length} values, got {len(result)}")
    return result


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _check_scalar(
    issues: list[str], label: str, actual: float, expected: float, tolerance: float
) -> None:
    if not _close(actual, expected, tolerance):
        issues.append(f"{label}: expected {expected}, got {actual}")


def _link_map(root: ET.Element) -> dict[str, ET.Element]:
    return {element.attrib["name"]: element for element in root.findall("link")}


def _joint_map(root: ET.Element) -> dict[str, ET.Element]:
    return {element.attrib["name"]: element for element in root.findall("joint")}


def _mass(link: ET.Element) -> float:
    element = link.find("inertial/mass")
    if element is None:
        raise ValueError("missing inertial/mass")
    return float(element.attrib["value"])


def _inertia_diagonal(link: ET.Element) -> tuple[float, float, float]:
    element = link.find("inertial/inertia")
    if element is None:
        raise ValueError("missing inertial/inertia")
    return tuple(float(element.attrib[key]) for key in ("ixx", "iyy", "izz"))


def _inertia_products(link: ET.Element) -> tuple[float, float, float]:
    element = link.find("inertial/inertia")
    if element is None:
        raise ValueError("missing inertial/inertia")
    return tuple(float(element.attrib[key]) for key in ("ixy", "ixz", "iyz"))


def _origin_xyz(element: ET.Element, path: str) -> tuple[float, float, float]:
    origin = element.find(path)
    if origin is None:
        raise ValueError(f"missing {path}")
    return _parse_vector(origin.attrib.get("xyz"))  # type: ignore[return-value]


def validate_rickshaw_urdf(
    path: str | Path = RICKSHAW_URDF_PATH,
    *,
    tolerance: float = 1.0e-5,
    validate_collision_geometry: bool = True,
) -> tuple[str, ...]:
    """Return every guide violation found in a source rickshaw URDF.

    This checks the values that must survive conversion. Runtime USD inspection
    must additionally verify PhysX mass/inertia, joint axes, frames, and that the
    two fixed hitch joints were not merged.
    """

    urdf_path = Path(path)
    if not urdf_path.is_file():
        return (f"missing URDF: {urdf_path}",)

    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return (f"cannot parse URDF {urdf_path}: {exc}",)

    issues: list[str] = []
    links = _link_map(root)
    joints = _joint_map(root)
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

    spec = RICKSHAW_URDF_SPEC
    try:
        _check_scalar(issues, "base mass", _mass(links[BASE_LINK_NAME]), spec.base_mass, tolerance)
        for index, (actual, expected) in enumerate(
            zip(_inertia_diagonal(links[BASE_LINK_NAME]), spec.base_inertia_diagonal)
        ):
            _check_scalar(issues, f"base inertia[{index}]", actual, expected, tolerance)
        for product, actual in zip(("ixy", "ixz", "iyz"), _inertia_products(links[BASE_LINK_NAME])):
            _check_scalar(issues, f"base inertia {product}", actual, 0.0, tolerance)

        for name in WHEEL_LINK_NAMES:
            _check_scalar(issues, f"{name} mass", _mass(links[name]), spec.wheel_mass, tolerance)
            for index, (actual, expected) in enumerate(
                zip(_inertia_diagonal(links[name]), spec.wheel_inertia_diagonal)
            ):
                _check_scalar(issues, f"{name} inertia[{index}]", actual, expected, tolerance)
            for product, actual in zip(("ixy", "ixz", "iyz"), _inertia_products(links[name])):
                _check_scalar(issues, f"{name} inertia {product}", actual, 0.0, tolerance)

        for name in HITCH_LINK_NAMES:
            _check_scalar(issues, f"{name} mass", _mass(links[name]), spec.hitch_link_mass, tolerance)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"invalid inertial data: {exc}")

    for name in WHEEL_JOINT_NAMES:
        joint = joints[name]
        if joint.attrib.get("type") != "continuous":
            issues.append(f"{name}: expected continuous passive joint")
        try:
            axis_element = joint.find("axis")
            if axis_element is None:
                raise ValueError("missing axis")
            axis = _parse_vector(axis_element.attrib.get("xyz"))
            for index, (actual, expected) in enumerate(zip(axis, spec.wheel_joint_axis)):
                _check_scalar(issues, f"{name} axis[{index}]", actual, expected, tolerance)
            dynamics = joint.find("dynamics")
            if dynamics is None:
                raise ValueError("missing dynamics")
            damping = float(dynamics.attrib["damping"])
            _check_scalar(issues, f"{name} damping", damping, spec.wheel_joint_damping, tolerance)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{name}: {exc}")

    expected_wheel_origins = {
        WHEEL_JOINT_NAMES[0]: (0.0, spec.wheel_track / 2.0, spec.wheel_radius),
        WHEEL_JOINT_NAMES[1]: (0.0, -spec.wheel_track / 2.0, spec.wheel_radius),
    }
    for name, expected in expected_wheel_origins.items():
        try:
            actual = _origin_xyz(joints[name], "origin")
            for index, (component, target) in enumerate(zip(actual, expected)):
                _check_scalar(issues, f"{name} origin[{index}]", component, target, tolerance)
        except (TypeError, ValueError) as exc:
            issues.append(f"{name}: {exc}")

    expected_hitches = {
        HITCH_JOINT_NAMES[0]: (spec.hitch_x, spec.hitch_half_width, spec.hitch_z),
        HITCH_JOINT_NAMES[1]: (spec.hitch_x, -spec.hitch_half_width, spec.hitch_z),
    }
    for name, expected in expected_hitches.items():
        joint = joints[name]
        if joint.attrib.get("type") != "fixed":
            issues.append(f"{name}: expected fixed joint")
        try:
            actual = _origin_xyz(joint, "origin")
            for index, (component, target) in enumerate(zip(actual, expected)):
                _check_scalar(issues, f"{name} origin[{index}]", component, target, tolerance)
        except (TypeError, ValueError) as exc:
            issues.append(f"{name}: {exc}")

    # At zero wheel phase, all required joint rotations are identity in the
    # specified URDF. This direct mass-weighted check catches frame regressions.
    try:
        base_com = _origin_xyz(links[BASE_LINK_NAME], "inertial/origin")
        masses_and_positions = [(spec.base_mass, base_com)]
        for joint_name, link_name in zip(WHEEL_JOINT_NAMES, WHEEL_LINK_NAMES):
            masses_and_positions.append((_mass(links[link_name]), _origin_xyz(joints[joint_name], "origin")))
        for joint_name, link_name in zip(HITCH_JOINT_NAMES, HITCH_LINK_NAMES):
            masses_and_positions.append((_mass(links[link_name]), _origin_xyz(joints[joint_name], "origin")))
        total_mass = sum(mass for mass, _ in masses_and_positions)
        total_com = tuple(
            sum(mass * position[axis] for mass, position in masses_and_positions) / total_mass
            for axis in range(3)
        )
        _check_scalar(issues, "total mass", total_mass, spec.total_mass, tolerance)
        for index, (actual, expected) in enumerate(zip(total_com, spec.center_of_mass)):
            _check_scalar(issues, f"center of mass[{index}]", actual, expected, tolerance)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        issues.append(f"cannot compute total mass/center of mass: {exc}")

    if validate_collision_geometry:
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename:
                issues.append("mesh element has no filename")
            elif "://" not in filename and not (urdf_path.parent / filename).is_file():
                issues.append(f"missing referenced mesh: {filename}")

        base_visual_mesh = links[BASE_LINK_NAME].find("visual/geometry/mesh")
        base_collision_mesh = links[BASE_LINK_NAME].find("collision/geometry/mesh")
        if base_collision_mesh is None:
            issues.append("base_link collision must use a simplified convex mesh/decomposition")
        elif (
            base_visual_mesh is not None
            and base_collision_mesh.attrib.get("filename") == base_visual_mesh.attrib.get("filename")
        ):
            issues.append("base_link collision reuses the high-detail visual mesh")

        for name in WHEEL_LINK_NAMES:
            cylinder = links[name].find("collision/geometry/cylinder")
            if cylinder is None:
                issues.append(f"{name} collision must be a Y-axis cylinder")
                continue
            try:
                radius = float(cylinder.attrib["radius"])
                width = float(cylinder.attrib["length"])
                _check_scalar(issues, f"{name} collision radius", radius, spec.wheel_radius, tolerance)
                _check_scalar(issues, f"{name} collision width", width, spec.wheel_width, tolerance)
                collision_origin = links[name].find("collision/origin")
                if collision_origin is None:
                    raise ValueError("missing collision origin for Y-axis orientation")
                roll, pitch, _ = _parse_vector(collision_origin.attrib.get("rpy"))
                # A URDF cylinder is Z-aligned; +/- pi/2 about X maps it to Y.
                if not _close(abs(roll), math.pi / 2.0, tolerance) or not _close(pitch, 0.0, tolerance):
                    issues.append(
                        f"{name} collision cylinder is not Y-aligned: "
                        f"rpy={collision_origin.attrib.get('rpy')}"
                    )
            except (KeyError, ValueError) as exc:
                issues.append(f"{name} invalid collision cylinder: {exc}")

    return tuple(issues)


def assert_valid_rickshaw_urdf(
    path: str | Path = RICKSHAW_URDF_PATH,
    *,
    tolerance: float = 1.0e-5,
    validate_collision_geometry: bool = True,
) -> None:
    """Raise one aggregate exception for all source-URDF violations."""

    issues = validate_rickshaw_urdf(
        path,
        tolerance=tolerance,
        validate_collision_geometry=validate_collision_geometry,
    )
    if issues:
        formatted = "\n  - ".join(issues)
        raise RickshawAssetValidationError(f"Invalid rickshaw URDF:\n  - {formatted}")


__all__ = [
    "HANDLE_CONSTRAINT_CALIBRATION_REQUIRED",
    "HITCH_HALF_WIDTH",
    "HITCH_LINK_NAMES",
    "HITCH_X",
    "HITCH_Z",
    "HandleConstraintCfg",
    "RICKSHAW_ASSET_DIR",
    "RICKSHAW_CENTER_OF_MASS",
    "RICKSHAW_CFG",
    "RICKSHAW_MESH_PATHS",
    "RICKSHAW_TOTAL_MASS",
    "RICKSHAW_URDF",
    "RICKSHAW_URDF_PATH",
    "RICKSHAW_URDF_SPEC",
    "RICKSHAW_USD",
    "RICKSHAW_USD_PATH",
    "RickshawAssetValidationError",
    "RickshawUrdfSpec",
    "WHEEL_JOINT_DAMPING",
    "WHEEL_JOINT_NAMES",
    "WHEEL_RADIUS",
    "WHEEL_TRACK",
    "WHEEL_WIDTH",
    "assert_valid_rickshaw_urdf",
    "build_rickshaw_cfg",
    "missing_rickshaw_assets",
    "require_isaaclab",
    "validate_rickshaw_urdf",
]
