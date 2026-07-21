"""MuJoCo site-connect closed chain for the fixed grippers and rickshaw."""

from __future__ import annotations

import mujoco

from g1_rickshaw_lab.assets.g1_dex1 import GRASP_SITE_NAMES, add_g1_position_actuators, get_g1_spec
from g1_rickshaw_lab.assets.mujoco_spec import (
    ALL_COLLISION_BITS,
    GROUND_COLLISION_BIT,
    ROBOT_COLLISION_BIT,
)
from g1_rickshaw_lab.assets.rickshaw import (
    HITCH_SITE_NAMES,
    TOW_ROD_COLLISION_GEOM_NAMES,
    get_rickshaw_spec,
)

ROBOT_ENTITY_NAME = "robot"
RICKSHAW_ENTITY_NAME = "rickshaw"
CONNECTION_NAMES = ("left_grasp_connection", "right_grasp_connection")


def add_closed_chain_constraints(spec: mujoco.MjSpec) -> None:
    """Connect both fixed gripper centers to the rickshaw crossbar."""

    for side, grasp_site, hitch_site in zip(("left", "right"), GRASP_SITE_NAMES, HITCH_SITE_NAMES, strict=True):
        name1 = f"{ROBOT_ENTITY_NAME}/{grasp_site}"
        name2 = f"{RICKSHAW_ENTITY_NAME}/{hitch_site}"
        if spec.site(name1) is None or spec.site(name2) is None:
            raise ValueError(f"missing {side} closed-chain sites: {name1}, {name2}")
        spec.add_equality(
            name=f"{side}_grasp_connection",
            type=mujoco.mjtEq.mjEQ_CONNECT,
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
    add_g1_position_actuators(spec, prefix=f"{ROBOT_ENTITY_NAME}/")
    add_closed_chain_constraints(spec)
    return spec


def validate_assembled_model(model: mujoco.MjModel) -> tuple[str, ...]:
    issues: list[str] = []
    if model.neq != 2:
        issues.append(f"expected two equality constraints, got {model.neq}")
    for name in CONNECTION_NAMES:
        equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if equality_id < 0:
            issues.append(f"missing equality: {name}")
        elif model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_CONNECT:
            issues.append(f"{name} is not a site connection")
    movable_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)}
    for forbidden in (
        "robot/left_dex1_finger_joint_1",
        "robot/left_dex1_finger_joint_2",
        "robot/right_dex1_finger_joint_1",
        "robot/right_dex1_finger_joint_2",
    ):
        if forbidden in movable_names:
            issues.append(f"gripper joint still movable: {forbidden}")
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])) or ""
        for index in range(model.ngeom)
    ]
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        for index in range(model.ngeom)
    ]
    robot_geoms = [index for index, name in enumerate(body_names) if name.startswith("robot/")]
    rickshaw_geoms = [index for index, name in enumerate(body_names) if name.startswith("rickshaw/")]
    if not robot_geoms or not rickshaw_geoms:
        issues.append("missing robot or rickshaw collision class")
    if any(
        model.geom_contype[index] not in (0, ROBOT_COLLISION_BIT)
        or model.geom_conaffinity[index] != GROUND_COLLISION_BIT
        for index in robot_geoms
    ):
        issues.append("G1 geoms must use ground/tow-rod collision filtering without self collision")
    tow_rods = {
        f"rickshaw/{name}" for name in TOW_ROD_COLLISION_GEOM_NAMES
    }
    if {geom_names[index] for index in rickshaw_geoms if model.geom_conaffinity[index] == ROBOT_COLLISION_BIT} != tow_rods:
        issues.append("only the two tow-rod geoms may collide with G1")
    for robot_geom in robot_geoms:
        for rickshaw_geom in rickshaw_geoms:
            enabled = bool(
                model.geom_contype[robot_geom] & model.geom_conaffinity[rickshaw_geom]
                or model.geom_contype[rickshaw_geom] & model.geom_conaffinity[robot_geom]
            )
            should_collide = (
                model.geom_contype[robot_geom] == ROBOT_COLLISION_BIT
                and geom_names[rickshaw_geom] in tow_rods
            )
            if enabled != should_collide:
                issues.append("G1-rickshaw collision filtering must retain only non-gripper/tow-rod pairs")
                return tuple(issues)
    return tuple(issues)


__all__ = [
    "ROBOT_ENTITY_NAME",
    "RICKSHAW_ENTITY_NAME",
    "CONNECTION_NAMES",
    "add_closed_chain_constraints",
    "build_assembled_spec",
    "validate_assembled_model",
]
