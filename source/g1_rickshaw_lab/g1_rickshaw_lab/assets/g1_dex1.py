"""MuJoCo/mjlab asset for Unitree G1 with fixed Dex1 grippers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco

from g1_rickshaw_lab.project_paths import ASSET_ROOT

from .mujoco_spec import (
    ALL_COLLISION_BITS,
    GRIPPER_COLLISION_BIT,
    ROBOT_COLLISION_BIT,
    add_free_joint,
    load_urdf_spec,
    set_body_collision,
)

G1_DEX1_ASSET_DIR = ASSET_ROOT / "g1_dex1"
G1_DEX1_URDF_PATH = G1_DEX1_ASSET_DIR / "g1_29dof_mode_15_with_dex1_1.urdf"
G1_DEX1_URDF = str(G1_DEX1_URDF_PATH)

G1_DOF_COUNT = 29
COMBINED_DOF_COUNT = G1_DOF_COUNT
G1_TOTAL_MASS = 34.1299349
FIXED_GRIP_POSITION = -0.01609
GRASP_SITE_X = 0.11066269
GRASP_SITE_NAMES = ("left_grasp_site", "right_grasp_site")
GRIPPER_BODY_NAMES = (
    "left_dex1_base_link",
    "left_dex1_finger_link_1",
    "left_dex1_finger_link_2",
    "right_dex1_base_link",
    "right_dex1_finger_link_1",
    "right_dex1_finger_link_2",
)

LOWER_JOINT_PATTERN = r".*_(hip|knee|ankle)_.*"
WAIST_JOINT_PATTERN = r"waist_.*_joint"
ARM_JOINT_PATTERN = r".*_(shoulder|elbow|wrist)_.*"
EXPECTED_GROUP_COUNTS = {"lower": 12, "waist": 3, "arm": 14}


class AssetValidationError(ValueError):
    """Raised when the fixed-gripper G1 asset violates its contract."""


@dataclass(frozen=True)
class JointPartition:
    lower_ids: tuple[int, ...]
    waist_ids: tuple[int, ...]
    arm_ids: tuple[int, ...]
    lower_names: tuple[str, ...]
    waist_names: tuple[str, ...]
    arm_names: tuple[str, ...]

    @property
    def action_ids(self) -> tuple[int, ...]:
        return self.lower_ids + self.waist_ids + self.arm_ids

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.lower_names + self.waist_names + self.arm_names


def _matching_indices(names: Sequence[str], pattern: str) -> tuple[int, ...]:
    expression = re.compile(pattern)
    return tuple(index for index, name in enumerate(names) if expression.fullmatch(name))


def partition_joint_names(joint_names: Iterable[str]) -> JointPartition:
    """Return the checkpoint/action order for the 29 movable G1 joints."""

    names = tuple(joint_names)
    if len(names) != len(set(names)):
        raise AssetValidationError("joint names must be unique")
    lower_ids = _matching_indices(names, LOWER_JOINT_PATTERN)
    waist_ids = _matching_indices(names, WAIST_JOINT_PATTERN)
    arm_ids = _matching_indices(names, ARM_JOINT_PATTERN)
    counts = {"lower": len(lower_ids), "waist": len(waist_ids), "arm": len(arm_ids)}
    if counts != EXPECTED_GROUP_COUNTS:
        raise AssetValidationError(f"unexpected G1 joint partition: {counts}")
    all_ids = lower_ids + waist_ids + arm_ids
    if len(all_ids) != G1_DOF_COUNT or len(set(all_ids)) != G1_DOF_COUNT:
        raise AssetValidationError("the G1 action partition must contain 29 distinct joints")

    def selected(indices: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(names[index] for index in indices)

    return JointPartition(
        lower_ids,
        waist_ids,
        arm_ids,
        selected(lower_ids),
        selected(waist_ids),
        selected(arm_ids),
    )


def partition_articulation_joints(robot) -> JointPartition:
    return partition_joint_names(robot.joint_names)


def validate_g1_urdf(path: str | Path = G1_DEX1_URDF_PATH) -> tuple[str, ...]:
    """Validate 29 DoF and the calibrated, zero-DoF gripper posture."""

    root = ET.parse(Path(path)).getroot()
    movable = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]
    issues: list[str] = []
    try:
        partition_joint_names(movable)
    except AssetValidationError as exc:
        issues.append(str(exc))

    expected_origins = {
        "left_dex1_finger_joint_1": (0.0, -FIXED_GRIP_POSITION, 0.0),
        "left_dex1_finger_joint_2": (0.0, FIXED_GRIP_POSITION, 0.0),
        "right_dex1_finger_joint_1": (0.0, -FIXED_GRIP_POSITION, 0.0),
        "right_dex1_finger_joint_2": (0.0, FIXED_GRIP_POSITION, 0.0),
    }
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    for name, expected in expected_origins.items():
        joint = joints.get(name)
        if joint is None or joint.attrib.get("type") != "fixed":
            issues.append(f"{name} must be fixed")
            continue
        origin = joint.find("origin")
        actual = tuple(float(value) for value in origin.attrib["xyz"].split()) if origin is not None else ()
        if actual != expected:
            issues.append(f"{name} origin: expected {expected}, got {actual}")
    return tuple(issues)


def get_g1_spec() -> mujoco.MjSpec:
    """Build the floating G1 spec and its two calibrated grasp sites."""

    issues = validate_g1_urdf()
    if issues:
        raise AssetValidationError("; ".join(issues))
    spec = load_urdf_spec(G1_DEX1_URDF_PATH)
    add_free_joint(spec, "pelvis")

    for geom in spec.geoms:
        geom.contype = ROBOT_COLLISION_BIT
        geom.conaffinity = ALL_COLLISION_BITS
    set_body_collision(
        spec,
        GRIPPER_BODY_NAMES,
        contype=GRIPPER_COLLISION_BIT,
        conaffinity=1 | ROBOT_COLLISION_BIT,
    )

    site_frames = (
        ("left_dex1_base_link", GRASP_SITE_NAMES[0], (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)),
        ("right_dex1_base_link", GRASP_SITE_NAMES[1], (0.7071067811865476, -0.7071067811865475, 0.0, 0.0)),
    )
    for body_name, site_name, quat in site_frames:
        spec.body(body_name).add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.006, 0.0, 0.0),
            pos=(GRASP_SITE_X, 0.0, 0.0),
            quat=quat,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
    return spec


def get_g1_robot_cfg():
    """Return a fresh mjlab EntityCfg; imports mjlab only when requested."""

    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    actuator_groups = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(r".*_(hip|knee)_joint",), stiffness=300.0, damping=10.0, effort_limit=139.0
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(r".*_ankle_(pitch|roll)_joint",), stiffness=200.0, damping=5.0, effort_limit=50.0
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(r"waist_.*_joint",), stiffness=5000.0, damping=5.0, effort_limit=50.0
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(r".*_(shoulder_.*|elbow|wrist_roll)_joint",),
            stiffness=3000.0,
            damping=10.0,
            effort_limit=25.0,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(r".*_wrist_(pitch|yaw)_joint",),
            stiffness=3000.0,
            damping=10.0,
            effort_limit=13.4,
        ),
    )
    return EntityCfg(
        spec_fn=get_g1_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.72),
            joint_pos={
                r".*_hip_pitch_joint": -0.32,
                r".*_knee_joint": 0.92,
                r".*_ankle_pitch_joint": -0.34,
                r".*_shoulder_pitch_joint": 0.35,
                r"left_shoulder_roll_joint": -0.08,
                r"right_shoulder_roll_joint": 0.08,
                r"left_elbow_joint": 0.22,
                r"right_elbow_joint": 0.22,
                r".*": 0.0,
            },
            joint_vel={r".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(actuators=actuator_groups, soft_joint_pos_limit_factor=0.9),
    )


build_g1_rickshaw_cfg = get_g1_robot_cfg
G1_RICKSHAW_CFG = None


def missing_g1_dex1_assets() -> tuple[Path, ...]:
    return () if G1_DEX1_URDF_PATH.is_file() else (G1_DEX1_URDF_PATH,)


__all__ = [
    "AssetValidationError",
    "COMBINED_DOF_COUNT",
    "FIXED_GRIP_POSITION",
    "G1_DEX1_URDF",
    "G1_DEX1_URDF_PATH",
    "G1_DOF_COUNT",
    "G1_RICKSHAW_CFG",
    "G1_TOTAL_MASS",
    "GRASP_SITE_NAMES",
    "JointPartition",
    "build_g1_rickshaw_cfg",
    "get_g1_robot_cfg",
    "get_g1_spec",
    "missing_g1_dex1_assets",
    "partition_articulation_joints",
    "partition_joint_names",
    "validate_g1_urdf",
]
