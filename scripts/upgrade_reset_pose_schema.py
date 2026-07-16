#!/usr/bin/env python3
"""Upgrade a schema-v3 reset library with joint fixed-contact statics fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from solve_reset_poses import (
    DEFAULT_URDF,
    GRAVITY,
    _body_id,
    _cart_fixed_contact_statics,
    _joint_qpos_address,
    _load_floating_urdf_model,
    _solver_dependencies,
    _tangent_difference_torque_basis,
    _write_json_yaml,
)

from g1_rickshaw_lab.configuration import (  # noqa: E402
    G1_JOINT_ORDER,
    RESET_POSE_SCHEMA_VERSION,
    ResetPoseLibrary,
)
from g1_rickshaw_lab.rickshaw_spec import HITCH_X, HITCH_Z, WHEEL_RADIUS  # noqa: E402


def _upgrade_mapping(
    mapping: dict[str, Any],
    *,
    urdf: Path,
    dex_q_grasp: float = -0.01609,
    wrist_to_dex_base_x: float = 0.0415,
    grasp_center_x: float = 0.11066269,
    hitch_height: float = 0.85,
) -> dict[str, Any]:
    mujoco, np, _least_squares, _rotation = _solver_dependencies()
    if mapping.get("schema_version") not in (3, RESET_POSE_SCHEMA_VERSION):
        raise ValueError("input must be a schema-v3 or schema-v4 reset library")
    if tuple(mapping.get("joint_order", ())) != G1_JOINT_ORDER:
        raise ValueError("reset library joint order is not canonical")

    model = _load_floating_urdf_model(urdf.resolve(), mujoco)
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in G1_JOINT_ORDER
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise RuntimeError("URDF is missing a policy joint")
    qpos_addresses = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    )
    dof_addresses = np.asarray(
        [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
    )
    wrist_ids = tuple(
        _body_id(model, f"{side}_wrist_yaw_link", mujoco)
        for side in ("left", "right")
    )
    foot_ids = tuple(
        _body_id(model, f"{side}_ankle_roll_link", mujoco)
        for side in ("left", "right")
    )
    wrist_to_grasp = np.asarray(
        (wrist_to_dex_base_x + grasp_center_x, 0.0, 0.0), dtype=np.float64
    )
    alpha_radius = math.hypot(HITCH_X, HITCH_Z - WHEEL_RADIUS)
    alpha_phase = math.atan2(HITCH_Z - WHEEL_RADIUS, HITCH_X)
    alpha = math.asin((hitch_height - WHEEL_RADIUS) / alpha_radius) - alpha_phase

    base_qpos = np.zeros(model.nq, dtype=np.float64)
    for name in (
        "left_dex1_finger_joint_1",
        "left_dex1_finger_joint_2",
        "right_dex1_finger_joint_1",
        "right_dex1_finger_joint_2",
    ):
        base_qpos[_joint_qpos_address(model, name, mujoco)] = dex_q_grasp

    upgraded_poses: list[dict[str, Any]] = []
    for raw_pose in mapping.get("poses", ()):
        pose = dict(raw_pose)
        gradient = float(pose["gradient"])
        root_pitch = float(pose.get("root_pitch", 0.0))
        qpos = base_qpos.copy()
        qpos[2] = float(pose.get("root_height", 0.75))
        qpos[3:7] = (
            math.cos(0.5 * root_pitch),
            0.0,
            math.sin(0.5 * root_pitch),
            0.0,
        )
        qpos[qpos_addresses] = np.asarray(pose["q_reset"], dtype=np.float64)
        model.opt.gravity[:] = (
            -GRAVITY * math.sin(math.atan(gradient)),
            0.0,
            -GRAVITY * math.cos(math.atan(gradient)),
        )
        tangent_difference_basis, root_residual, _support_torque = (
            _tangent_difference_torque_basis(
                model=model,
                data=data,
                qpos=qpos,
                wrist_body_ids=wrist_ids,
                foot_body_ids=foot_ids,
                wrist_to_grasp=wrist_to_grasp,
                dof_addresses=dof_addresses,
                mujoco=mujoco,
                np=np,
            )
        )
        if float(np.max(np.abs(root_residual))) > 1.0e-9:
            raise RuntimeError(
                f"gradient {gradient:+.2f} differential basis does not close root equilibrium"
            )
        cart_solution = _cart_fixed_contact_statics(gradient, alpha, 0.0, np)
        if max(
            max(abs(value) for value in cart_solution.cart_force_residual_sln),
            max(abs(value) for value in cart_solution.cart_moment_residual_sln),
        ) > 1.0e-9:
            raise RuntimeError(
                f"gradient {gradient:+.2f} does not close cart equilibrium"
            )
        pose["tau_per_tangent_difference"] = [
            float(value) for value in tangent_difference_basis
        ]
        pose["handle_wrenches_sln"] = [
            [float(value) for value in row]
            for row in cart_solution.handle_wrenches_sln
        ]
        pose["wheel_contact_forces_sln"] = [
            [float(value) for value in row]
            for row in cart_solution.wheel_contact_forces_sln
        ]
        upgraded_poses.append(pose)

    upgraded = dict(mapping)
    upgraded["schema_version"] = RESET_POSE_SCHEMA_VERSION
    upgraded["poses"] = upgraded_poses
    ResetPoseLibrary.from_mapping(upgraded)
    return upgraded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?", default=Path("config/reset_poses.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    source = args.path.resolve()
    destination = source if args.output is None else args.output.resolve()
    mapping = json.loads(source.read_text(encoding="utf-8"))
    upgraded = _upgrade_mapping(mapping, urdf=args.urdf)
    _write_json_yaml(destination, upgraded)
    print(f"upgraded reset library: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
