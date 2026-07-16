"""G1 29-DoF with two Dex1-1 grippers.

The USD referenced here is an offline build artifact.  It must be converted from
Unitree's combined G1 + Dex1-1 URDF without merging fixed joints.  This module is
deliberately importable outside Isaac Lab so asset inspection and joint-order
tests can run in a normal Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

from .._hashing import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = REPOSITORY_ROOT / "assets"
G1_DEX1_ASSET_DIR = ASSET_ROOT / "g1_dex1"
G1_DEX1_URDF_PATH = G1_DEX1_ASSET_DIR / "g1_29dof_mode_15_with_dex1_1.urdf"
G1_DEX1_USD_PATH = G1_DEX1_ASSET_DIR / "g1_29dof_mode_15_with_dex1_1.usd"

# String aliases are convenient for Isaac Lab configuration fields and retain
# the names used by the implementation guide.
G1_DEX1_URDF = str(G1_DEX1_URDF_PATH)
G1_DEX1_USD = str(G1_DEX1_USD_PATH)

G1_DOF_COUNT = 29
DEX_DOF_COUNT = 4
COMBINED_DOF_COUNT = G1_DOF_COUNT + DEX_DOF_COUNT
G1_TOTAL_MASS = 34.1299349
RETAINED_SENSOR_LINK_MASS = 1.0e-6
RETAINED_SENSOR_LINK_NAMES = (
    "imu_in_torso",
    "imu_in_pelvis",
    "d435_link",
    "mid360_link",
)

LOWER_JOINT_PATTERN = r".*_(hip|knee|ankle)_.*"
WAIST_JOINT_PATTERN = r"waist_.*_joint"
ARM_JOINT_PATTERN = r".*_(shoulder|elbow|wrist)_.*"
DEX_JOINT_PATTERN = r"(left|right)_dex1_finger_joint_[12]"

# Continuous G1 arm limits used by the reset solver and hardware safety gates.
# The upstream locomanipulation asset uses a permissive 300 N*m solver value
# for every implicit arm joint, which is not a physical actuator rating.
G1_ARM_HARDWARE_EFFORT_LIMITS = {
    ".*_shoulder_.*_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_(pitch|yaw)_joint": 13.4,
}

EXPECTED_GROUP_COUNTS = {
    "lower": 12,
    "waist": 3,
    "arm": 14,
    "dex": 4,
}

# These values cannot be inferred from the provided CAD/URDF.  They must remain
# calibration inputs rather than guessed actuator settings.
DEX_CALIBRATION_REQUIRED = (
    "actuator stiffness",
    "actuator damping",
    "effort limit",
    "velocity limit",
    "q_open",
    "q_grasp",
    "grasp velocity",
    "grasp timeout",
    "left grasp-center frame",
    "right grasp-center frame",
)


class IsaacLabUnavailableError(RuntimeError):
    """Raised when an Isaac Lab configuration is requested without Isaac Lab."""


class AssetValidationError(ValueError):
    """Raised when an asset violates a required structural invariant."""


try:
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG as _G1_29DOF_CFG
except ModuleNotFoundError as exc:  # Normal for pure geometry/unit-test installs.
    _G1_29DOF_CFG = None
    _ISAACLAB_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _ISAACLAB_IMPORT_ERROR = None


def require_isaaclab() -> None:
    """Fail with actionable context when the simulation stack is unavailable."""

    if _G1_29DOF_CFG is None:
        raise IsaacLabUnavailableError(
            "G1_RICKSHAW_CFG requires Isaac Lab and isaaclab_assets. Launch "
            "through Isaac Sim/Isaac Lab after installing the extension."
        ) from _ISAACLAB_IMPORT_ERROR


def build_g1_rickshaw_cfg(*, require_usd: bool = False):
    """Create an independent Isaac Lab articulation config for the combined USD.

    The inherited actuator expressions intentionally cover only the 29 G1
    joints. Dex drives must be added only after the required hardware values have
    been calibrated; the four Dex joints are never part of the RL action.
    """

    require_isaaclab()
    if require_usd and not G1_DEX1_USD_PATH.is_file():
        raise FileNotFoundError(
            f"Missing combined G1+Dex1-1 USD: {G1_DEX1_USD_PATH}. Convert "
            f"{G1_DEX1_URDF_PATH} without merging fixed joints."
        )

    cfg = _G1_29DOF_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.spawn.usd_path = G1_DEX1_USD
    cfg.spawn.activate_contact_sensors = True
    cfg.spawn.articulation_props.solver_position_iteration_count = 4
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    # The upstream 29-DoF config also carries an Inspire-style three-finger
    # actuator.  This asset has four calibrated Dex1 joints instead.
    cfg.actuators.pop("hands", None)
    arms = cfg.actuators["arms"]
    arms.effort_limit = None
    arms.effort_limit_sim = dict(G1_ARM_HARDWARE_EFFORT_LIMITS)
    # The combined source URDF is already +X-forward.  The upstream Nucleus
    # asset uses a 90-degree yaw in its initial state, which must not leak into
    # this converted asset's path frame.
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    return cfg


# Keep this module importable in tooling environments that do not ship Isaac
# Lab. Consumers that need a live config should call build_g1_rickshaw_cfg().
G1_RICKSHAW_CFG = build_g1_rickshaw_cfg() if _G1_29DOF_CFG is not None else None


@dataclass(frozen=True)
class JointPartition:
    """Validated joint groups in the articulation's authoritative USD order."""

    lower_ids: tuple[int, ...]
    waist_ids: tuple[int, ...]
    arm_ids: tuple[int, ...]
    dex_ids: tuple[int, ...]
    lower_names: tuple[str, ...]
    waist_names: tuple[str, ...]
    arm_names: tuple[str, ...]
    dex_names: tuple[str, ...]

    @property
    def action_ids(self) -> tuple[int, ...]:
        """The 29-D action order; serialize this exact tuple in checkpoints."""

        return self.lower_ids + self.waist_ids + self.arm_ids

    @property
    def action_names(self) -> tuple[str, ...]:
        """The 29-D action names; serialize this exact tuple in checkpoints."""

        return self.lower_names + self.waist_names + self.arm_names

    @property
    def all_ids(self) -> tuple[int, ...]:
        return self.action_ids + self.dex_ids

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.action_names + self.dex_names


def _matching_indices(names: Sequence[str], pattern: str) -> tuple[int, ...]:
    expression = re.compile(pattern)
    return tuple(index for index, name in enumerate(names) if expression.fullmatch(name))


def partition_joint_names(joint_names: Iterable[str]) -> JointPartition:
    """Classify and validate all 33 joints while preserving source ordering.

    Regex matching is used once during asset inspection. The returned explicit
    order, rather than regex ordering, is the deployment/checkpoint contract.
    """

    names = tuple(joint_names)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise AssetValidationError(f"Duplicate joint names: {duplicates}")

    lower_ids = _matching_indices(names, LOWER_JOINT_PATTERN)
    waist_ids = _matching_indices(names, WAIST_JOINT_PATTERN)
    arm_ids = _matching_indices(names, ARM_JOINT_PATTERN)
    dex_ids = _matching_indices(names, DEX_JOINT_PATTERN)

    groups = {
        "lower": lower_ids,
        "waist": waist_ids,
        "arm": arm_ids,
        "dex": dex_ids,
    }
    actual_counts = {key: len(value) for key, value in groups.items()}
    if actual_counts != EXPECTED_GROUP_COUNTS:
        raise AssetValidationError(
            f"Unexpected G1+Dex joint partition: {actual_counts}; "
            f"expected {EXPECTED_GROUP_COUNTS}"
        )

    flattened = lower_ids + waist_ids + arm_ids + dex_ids
    if len(flattened) != len(set(flattened)):
        raise AssetValidationError("G1/Dex joint regex groups overlap")
    if len(flattened) != COMBINED_DOF_COUNT:
        raise AssetValidationError(
            f"Expected {COMBINED_DOF_COUNT} classified joints, got {len(flattened)}"
        )

    def selected(indices: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(names[index] for index in indices)

    return JointPartition(
        lower_ids=lower_ids,
        waist_ids=waist_ids,
        arm_ids=arm_ids,
        dex_ids=dex_ids,
        lower_names=selected(lower_ids),
        waist_names=selected(waist_ids),
        arm_names=selected(arm_ids),
        dex_names=selected(dex_ids),
    )


def partition_articulation_joints(robot) -> JointPartition:
    """Validate a spawned articulation using its USD joint order."""

    try:
        names = robot.joint_names
    except AttributeError as exc:
        raise TypeError("robot must expose an ordered joint_names sequence") from exc
    return partition_joint_names(names)


def validate_g1_urdf_inertials(
    path: str | Path = G1_DEX1_URDF_PATH,
    *,
    mass_tolerance: float = 1.0e-7,
) -> tuple[str, ...]:
    """Validate the retained-link inertials and calibrated whole-robot mass.

    PhysX assigns a default 1 kg mass to a retained URDF link without an
    inertial block.  Since fixed joints deliberately remain unmerged, every
    link must carry explicit positive mass and inertia data.
    """

    try:
        root = ET.parse(Path(path)).getroot()
    except (ET.ParseError, OSError) as exc:
        return (f"cannot parse G1 URDF {path}: {exc}",)

    issues: list[str] = []
    links = {link.attrib["name"]: link for link in root.findall("link")}
    total_mass = 0.0
    for name, link in links.items():
        mass_element = link.find("inertial/mass")
        inertia_element = link.find("inertial/inertia")
        if mass_element is None or inertia_element is None:
            issues.append(f"{name}: retained link lacks explicit inertial data")
            continue
        try:
            mass = float(mass_element.attrib["value"])
            diagonal = tuple(float(inertia_element.attrib[key]) for key in ("ixx", "iyy", "izz"))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{name}: invalid inertial data ({exc})")
            continue
        if mass <= 0.0:
            issues.append(f"{name}: mass must be positive, got {mass}")
        if any(value <= 0.0 for value in diagonal):
            issues.append(f"{name}: inertia diagonal must be positive, got {diagonal}")
        total_mass += mass

    for name in RETAINED_SENSOR_LINK_NAMES:
        link = links.get(name)
        if link is None:
            issues.append(f"missing retained sensor link: {name}")
            continue
        mass_element = link.find("inertial/mass")
        if mass_element is not None:
            actual = float(mass_element.attrib["value"])
            if abs(actual - RETAINED_SENSOR_LINK_MASS) > mass_tolerance:
                issues.append(
                    f"{name}: expected frame mass {RETAINED_SENSOR_LINK_MASS}, got {actual}"
                )

    if abs(total_mass - G1_TOTAL_MASS) > mass_tolerance:
        issues.append(f"whole-robot mass: expected {G1_TOTAL_MASS}, got {total_mass}")
    return tuple(issues)


def missing_g1_dex1_assets() -> tuple[Path, ...]:
    """Return required combined-asset artifacts that have not been generated."""

    return tuple(path for path in (G1_DEX1_URDF_PATH, G1_DEX1_USD_PATH) if not path.is_file())


__all__ = [
    "ASSET_ROOT",
    "AssetValidationError",
    "COMBINED_DOF_COUNT",
    "DEX_CALIBRATION_REQUIRED",
    "DEX_DOF_COUNT",
    "DEX_JOINT_PATTERN",
    "G1_DEX1_ASSET_DIR",
    "G1_DEX1_URDF",
    "G1_DEX1_URDF_PATH",
    "G1_DEX1_USD",
    "G1_DEX1_USD_PATH",
    "G1_DOF_COUNT",
    "G1_RICKSHAW_CFG",
    "G1_ARM_HARDWARE_EFFORT_LIMITS",
    "G1_TOTAL_MASS",
    "JointPartition",
    "RETAINED_SENSOR_LINK_MASS",
    "RETAINED_SENSOR_LINK_NAMES",
    "build_g1_rickshaw_cfg",
    "missing_g1_dex1_assets",
    "partition_articulation_joints",
    "partition_joint_names",
    "require_isaaclab",
    "sha256_file",
    "validate_g1_urdf_inertials",
]
