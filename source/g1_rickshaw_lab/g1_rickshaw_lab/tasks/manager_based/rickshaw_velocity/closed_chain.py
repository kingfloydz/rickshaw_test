"""MuJoCo site-weld closed chain for the fixed grippers and rickshaw."""

from __future__ import annotations

import mujoco

from g1_rickshaw_lab.assets.g1_dex1 import GRASP_SITE_NAMES, get_g1_spec
from g1_rickshaw_lab.assets.mujoco_spec import (
    ALL_COLLISION_BITS,
    GRIPPER_COLLISION_BIT,
    GROUND_COLLISION_BIT,
    RICKSHAW_COLLISION_BIT,
)
from g1_rickshaw_lab.assets.rickshaw import HITCH_SITE_NAMES, get_rickshaw_spec

ROBOT_ENTITY_NAME = "robot"
RICKSHAW_ENTITY_NAME = "rickshaw"
WELD_NAMES = ("left_grasp_weld", "right_grasp_weld")


def add_closed_chain_constraints(spec: mujoco.MjSpec) -> None:
    """Add two rigid site welds after mjlab has attached both entities."""

    for side, grasp_site, hitch_site in zip(("left", "right"), GRASP_SITE_NAMES, HITCH_SITE_NAMES, strict=True):
        name1 = f"{ROBOT_ENTITY_NAME}/{grasp_site}"
        name2 = f"{RICKSHAW_ENTITY_NAME}/{hitch_site}"
        if spec.site(name1) is None or spec.site(name2) is None:
            raise ValueError(f"missing {side} closed-chain sites: {name1}, {name2}")
        spec.add_equality(
            name=f"{side}_grasp_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=name1,
            name2=name2,
            active=1,
            solref=(0.004, 1.0),
            solimp=(0.95, 0.99, 0.002, 0.5, 2.0),
        )


def build_assembled_spec(*, with_ground: bool = True) -> mujoco.MjSpec:
    """Build a standalone one-environment model for validation/statics."""

    spec = mujoco.MjSpec()
    if with_ground:
        ground = spec.worldbody.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.05),
        )
        ground.contype = GROUND_COLLISION_BIT
        ground.conaffinity = ALL_COLLISION_BITS
        ground.friction[:3] = (1.0, 0.005, 0.0001)
    spec.attach(get_g1_spec(), prefix=f"{ROBOT_ENTITY_NAME}/", frame=spec.worldbody.add_frame())
    spec.attach(get_rickshaw_spec(), prefix=f"{RICKSHAW_ENTITY_NAME}/", frame=spec.worldbody.add_frame())
    add_closed_chain_constraints(spec)
    return spec


def validate_assembled_model(model: mujoco.MjModel) -> tuple[str, ...]:
    issues: list[str] = []
    if model.neq != 2:
        issues.append(f"expected two equality constraints, got {model.neq}")
    for name in WELD_NAMES:
        equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if equality_id < 0:
            issues.append(f"missing equality: {name}")
        elif model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_WELD:
            issues.append(f"{name} is not a weld")
    movable_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)}
    for forbidden in (
        "robot/left_dex1_finger_joint_1",
        "robot/left_dex1_finger_joint_2",
        "robot/right_dex1_finger_joint_1",
        "robot/right_dex1_finger_joint_2",
    ):
        if forbidden in movable_names:
            issues.append(f"gripper joint still movable: {forbidden}")
    gripper_geoms = [index for index in range(model.ngeom) if model.geom_contype[index] == GRIPPER_COLLISION_BIT]
    rickshaw_geoms = [index for index in range(model.ngeom) if model.geom_contype[index] == RICKSHAW_COLLISION_BIT]
    if not gripper_geoms or not rickshaw_geoms:
        issues.append("missing gripper or rickshaw collision class")
    for gripper_geom in gripper_geoms:
        for rickshaw_geom in rickshaw_geoms:
            enabled = bool(
                model.geom_contype[gripper_geom] & model.geom_conaffinity[rickshaw_geom]
                or model.geom_contype[rickshaw_geom] & model.geom_conaffinity[gripper_geom]
            )
            if enabled:
                issues.append("gripper-rickshaw collision filtering is not symmetric")
                return tuple(issues)
    return tuple(issues)


__all__ = [
    "ROBOT_ENTITY_NAME",
    "RICKSHAW_ENTITY_NAME",
    "WELD_NAMES",
    "add_closed_chain_constraints",
    "build_assembled_spec",
    "validate_assembled_model",
]
