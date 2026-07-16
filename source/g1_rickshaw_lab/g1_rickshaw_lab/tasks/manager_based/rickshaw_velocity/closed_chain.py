"""Scene spawner for the replicated robot-rickshaw closed chain."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdPhysics


def _expand_env_path(path: str, env_namespace: str) -> str:
    if path.startswith("/"):
        return path.replace("{ENV_NS}", env_namespace).replace(
            "{ENV_REGEX_NS}", env_namespace
        )
    return f"{env_namespace}/{path.lstrip('/')}"


def _set_local_pose(
    joint: UsdPhysics.Joint,
    position: tuple[float, ...],
    quaternion: tuple[float, ...],
    side: int,
) -> None:
    pos = Gf.Vec3f(*position)
    quat = Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
    if side == 0:
        joint.CreateLocalPos0Attr().Set(pos)
        joint.CreateLocalRot0Attr().Set(quat)
    else:
        joint.CreateLocalPos1Attr().Set(pos)
        joint.CreateLocalRot1Attr().Set(quat)


def _apply_limit(
    prim: Usd.Prim, axis: str, lower: float, upper: float
) -> None:
    limit = UsdPhysics.LimitAPI.Apply(prim, axis)
    limit.CreateLowAttr().Set(lower)
    limit.CreateHighAttr().Set(upper)


def _apply_drive(
    prim: Usd.Prim,
    axis: str,
    stiffness: float,
    damping: float,
    maximum: float,
) -> None:
    drive = UsdPhysics.DriveAPI.Apply(prim, axis)
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(stiffness)
    drive.CreateDampingAttr().Set(damping)
    drive.CreateMaxForceAttr().Set(maximum)


@clone
def spawn_replicated_dual_d6(
    prim_path: str,
    cfg: "ReplicatedDualD6SpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Author two D6 joints in the source environment before scene replication."""

    del kwargs
    handle = cfg.handle_constraint
    handle.validate()
    stage = get_current_stage()
    env_namespace = prim_path.rsplit("/", 1)[0]
    constraints_prim = create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )

    robot_root_path = f"{env_namespace}/Robot/pelvis"
    cart_root_path = f"{env_namespace}/Rickshaw/base_link"
    for label, body_path in (
        ("robot articulation root", robot_root_path),
        ("rickshaw articulation root", cart_root_path),
    ):
        if not stage.GetPrimAtPath(body_path).IsValid():
            raise RuntimeError(f"{label} prim does not exist: {body_path}")
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(
        stage.GetPrimAtPath(robot_root_path)
    )
    filtered_pairs.CreateFilteredPairsRel().AddTarget(Sdf.Path(cart_root_path))

    for side_index, side_name in enumerate(("left", "right")):
        robot_path = _expand_env_path(
            handle.robot_body_paths[side_index], env_namespace
        )
        hitch_path = _expand_env_path(
            handle.hitch_body_paths[side_index], env_namespace
        )
        for body_path in (robot_path, hitch_path):
            if not stage.GetPrimAtPath(body_path).IsValid():
                raise RuntimeError(f"D6 body prim does not exist: {body_path}")
        joint_path = handle.joint_prim_path_template.format(
            ENV_NS=env_namespace, side=side_name, env_id=0
        )
        joint = UsdPhysics.Joint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(robot_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(hitch_path)])
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        _set_local_pose(
            joint,
            handle.grasp_local_positions[side_index],
            handle.grasp_local_quaternions_wxyz[side_index],
            0,
        )
        _set_local_pose(joint, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 1)

        prim = joint.GetPrim()
        for axis in ("transX", "transY", "transZ"):
            _apply_limit(prim, axis, -handle.linear_limit, handle.linear_limit)
            _apply_drive(
                prim,
                axis,
                handle.linear_stiffness,
                handle.linear_damping,
                handle.max_force,
            )
        for axis_index, axis in enumerate(("rotX", "rotY", "rotZ")):
            if handle.rotation_free_axes[axis_index]:
                continue
            angular_limit_deg = math.degrees(handle.angular_limit)
            _apply_limit(prim, axis, -angular_limit_deg, angular_limit_deg)
            if handle.rotation_driven_axes[axis_index]:
                _apply_drive(
                    prim,
                    axis,
                    handle.angular_stiffness,
                    handle.angular_damping,
                    handle.max_torque,
                )
    return constraints_prim


@configclass
class ReplicatedDualD6SpawnerCfg(SpawnerCfg):
    """Configuration for the source-environment dual-D6 spawner."""

    func: Callable = spawn_replicated_dual_d6
    handle_constraint: Any = MISSING


__all__ = ["ReplicatedDualD6SpawnerCfg", "spawn_replicated_dual_d6"]
