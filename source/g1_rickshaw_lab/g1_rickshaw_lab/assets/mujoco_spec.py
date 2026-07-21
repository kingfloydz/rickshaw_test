"""Small MuJoCo spec helpers shared by project assets."""

from __future__ import annotations

from pathlib import Path

import mujoco

GROUND_COLLISION_BIT = 1
ROBOT_COLLISION_BIT = 2
GRIPPER_COLLISION_BIT = 4
RICKSHAW_COLLISION_BIT = 8
ALL_COLLISION_BITS = 15


def load_urdf_spec(path: Path) -> mujoco.MjSpec:
    """Load a URDF and its meshes without passing a Unicode path to MuJoCo."""

    xml = path.read_text(encoding="utf-8")
    roots = (path.parent / "meshes",) if (path.parent / "meshes").is_dir() else (path.parent,)
    assets = {
        mesh.relative_to(path.parent).as_posix(): mesh.read_bytes()
        for root in roots
        for mesh in root.iterdir()
        if mesh.is_file() and mesh != path
    }
    return mujoco.MjSpec.from_string(xml, assets=assets)


def add_free_joint(spec: mujoco.MjSpec, root_body_name: str) -> None:
    root = spec.body(root_body_name)
    if any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints):
        raise ValueError(f"{spec.modelname} already has a free joint")
    root.add_freejoint(name="floating_base_joint")


def set_body_collision(
    spec: mujoco.MjSpec,
    body_names: tuple[str, ...],
    *,
    contype: int,
    conaffinity: int,
) -> None:
    for body_name in body_names:
        for geom in spec.body(body_name).geoms:
            geom.contype = contype
            geom.conaffinity = conaffinity


__all__ = [
    "ALL_COLLISION_BITS",
    "GRIPPER_COLLISION_BIT",
    "GROUND_COLLISION_BIT",
    "RICKSHAW_COLLISION_BIT",
    "ROBOT_COLLISION_BIT",
    "add_free_joint",
    "load_urdf_spec",
    "set_body_collision",
]
