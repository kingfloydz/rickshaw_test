#!/usr/bin/env python3
"""Generate local arm IK seeds for a 0.45 m two-hand grasp width."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
import solve_reset_poses as reset_solver  # noqa: E402


GRASP_HALF_WIDTH_M = 0.225
WRIST_TO_GRASP_M = np.asarray((0.0415 + 0.11066269, 0.0, 0.0))
JOINT_MARGIN_RAD = 0.06
SELF_COLLISION_MARGIN_M = 0.001
ROBUST_SURFACE_RETREAT_M = 0.020
TORQUE_FEASIBLE_SURFACE_RETREAT_M = 0.0175
ELBOW_CONTACT_OFFSET_M = 0.001313551445491612
CART_BASE_CONTACT_OFFSET_M = 0.01486112829297781
CONTACT_DISTANCE_M = ELBOW_CONTACT_OFFSET_M + CART_BASE_CONTACT_OFFSET_M
ARM_EFFORT_LIMITS_NM = np.asarray((25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4))

# Material points recovered from the two active left-elbow contact patches.
LEFT_ELBOW_CONTACT_POINTS_M = (
    np.asarray((0.0978, 0.02065, 0.0119)),
    np.asarray((0.0980, 0.0284, -0.0009)),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-poses",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "reset_poses.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "outputs"
            / "diagnostics"
            / "grasp_width_045_local_ik_seeds.json"
        ),
    )
    return parser


def _orientation_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target.T @ actual).as_rotvec()


def _serial(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> int:
    args = _parser().parse_args()
    model = reset_solver._load_floating_urdf_model(reset_solver.DEFAULT_URDF, mujoco)
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )
    torso_geoms = np.flatnonzero(model.geom_bodyid == torso_id)

    joint_qpos_addresses: list[int] = []
    joint_dof_addresses: list[int] = []
    joint_lower: list[float] = []
    joint_upper: list[float] = []
    for name in reset_solver.G1_JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        joint_lower.append(float(model.jnt_range[joint_id, 0]))
        joint_upper.append(float(model.jnt_range[joint_id, 1]))
    qpos_addresses = np.asarray(joint_qpos_addresses)
    dof_addresses = np.asarray(joint_dof_addresses)
    lower = np.asarray(joint_lower)
    upper = np.asarray(joint_upper)

    source = json.loads(args.reset_poses.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for pose in source["poses"]:
        qpos = np.zeros(model.nq)
        root_pitch = float(pose["root_pitch"])
        qpos[:7] = (
            0.0,
            0.0,
            float(pose["root_height"]),
            math.cos(0.5 * root_pitch),
            0.0,
            math.sin(0.5 * root_pitch),
            0.0,
        )
        q_reset = np.asarray(pose["q_reset"], dtype=np.float64)
        qpos[qpos_addresses] = q_reset
        gamma = math.atan(float(pose["gradient"]))
        model.opt.gravity[:] = (
            -9.81 * math.sin(gamma),
            0.0,
            -9.81 * math.cos(gamma),
        )
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)

        modes: dict[str, dict[str, Any]] = {
            "grasp_only": {},
            "self_collision_free_torque_optimized": {},
            "robust_20mm_surface_retreat": {},
            "torque_optimized_17p5mm_surface_retreat": {},
        }
        for mode_name, surface_retreat, optimize_torque, avoid_self_collision in (
            ("grasp_only", None, False, False),
            ("self_collision_free_torque_optimized", None, True, True),
            (
                "robust_20mm_surface_retreat",
                ROBUST_SURFACE_RETREAT_M,
                False,
                False,
            ),
            (
                "torque_optimized_17p5mm_surface_retreat",
                TORQUE_FEASIBLE_SURFACE_RETREAT_M,
                True,
                False,
            ),
        ):
            q_seed = q_reset.copy()
            q_delta = np.zeros_like(q_reset)
            side_rows: dict[str, dict[str, Any]] = {}
            for side, sign, first_joint in (
                ("left", 1.0, 15),
                ("right", -1.0, 22),
            ):
                joint_indices = np.arange(first_joint, first_joint + 7)
                arm_qpos_addresses = qpos_addresses[joint_indices]
                arm_dof_addresses = dof_addresses[joint_indices]
                wrist_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"
                )
                elbow_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link"
                )
                arm_body_ids = tuple(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"{side}_{link}_link",
                    )
                    for link in (
                        "shoulder_pitch",
                        "shoulder_roll",
                        "shoulder_yaw",
                        "elbow",
                        "wrist_roll",
                        "wrist_pitch",
                        "wrist_yaw",
                    )
                )
                arm_geoms = np.concatenate(
                    [
                        np.flatnonzero(model.geom_bodyid == body_id)
                        for body_id in arm_body_ids
                    ]
                )
                torso_arm_geom_pairs = tuple(
                    (int(torso_geom), int(arm_geom))
                    for torso_geom in torso_geoms
                    for arm_geom in arm_geoms
                    if reset_solver._geom_pair_can_collide(
                        model, int(torso_geom), int(arm_geom)
                    )
                )
                collision_from_to = np.empty(6, dtype=np.float64)
                data.qpos[:] = qpos
                mujoco.mj_kinematics(model, data)
                wrist_rotation_0 = data.xmat[wrist_id].reshape(3, 3).copy()
                grasp_position_0 = (
                    data.xpos[wrist_id] + wrist_rotation_0 @ WRIST_TO_GRASP_M
                )
                grasp_local_rotation = Rotation.from_rotvec(
                    (sign * math.pi / 2.0, 0.0, 0.0)
                ).as_matrix()
                grasp_rotation_0 = wrist_rotation_0 @ grasp_local_rotation
                elbow_position_0 = data.xpos[elbow_id].copy()
                elbow_rotation_0 = data.xmat[elbow_id].reshape(3, 3).copy()
                local_points = tuple(
                    point * np.asarray((1.0, sign, 1.0))
                    for point in LEFT_ELBOW_CONTACT_POINTS_M
                )
                surface_y_0 = tuple(
                    float((elbow_position_0 + elbow_rotation_0 @ point)[1])
                    for point in local_points
                )
                x0 = q_reset[joint_indices]
                bounds = list(
                    zip(
                        lower[joint_indices] + JOINT_MARGIN_RAD,
                        upper[joint_indices] - JOINT_MARGIN_RAD,
                        strict=True,
                    )
                )

                def set_arm(x: np.ndarray) -> None:
                    data.qpos[:] = qpos
                    data.qpos[arm_qpos_addresses] = x
                    mujoco.mj_kinematics(model, data)

                def constraints(x: np.ndarray) -> np.ndarray:
                    set_arm(x)
                    wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
                    grasp_position = (
                        data.xpos[wrist_id] + wrist_rotation @ WRIST_TO_GRASP_M
                    )
                    grasp_rotation_error = _orientation_error(
                        wrist_rotation @ grasp_local_rotation, grasp_rotation_0
                    )
                    values = [
                        grasp_position[0] - grasp_position_0[0],
                        grasp_position[1] - sign * GRASP_HALF_WIDTH_M,
                        grasp_position[2] - grasp_position_0[2],
                        grasp_rotation_error[0],
                        grasp_rotation_error[2],
                    ]
                    if surface_retreat is not None:
                        elbow_rotation = data.xmat[elbow_id].reshape(3, 3)
                        surface_y = float(
                            (data.xpos[elbow_id] + elbow_rotation @ local_points[0])[1]
                        )
                        values.append(
                            surface_y - (surface_y_0[0] - sign * surface_retreat)
                        )
                    return np.asarray(values)

                def self_collision_margins(x: np.ndarray) -> np.ndarray:
                    set_arm(x)
                    return np.asarray(
                        [
                            mujoco.mj_geomDistance(
                                model,
                                data,
                                torso_geom,
                                arm_geom,
                                0.05,
                                collision_from_to,
                            )
                            - SELF_COLLISION_MARGIN_M
                            for torso_geom, arm_geom in torso_arm_geom_pairs
                        ],
                        dtype=np.float64,
                    )

                initial_constraints: list[dict[str, Any]] = [
                    {"type": "eq", "fun": constraints}
                ]
                if avoid_self_collision:
                    initial_constraints.append(
                        {"type": "ineq", "fun": self_collision_margins}
                    )
                result = minimize(
                    lambda x: float(np.sum(np.square(x - x0))),
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=tuple(initial_constraints),
                    options={"ftol": 1.0e-13, "maxiter": 1500},
                )
                hand_wrench_on_robot = -np.asarray(
                    pose["handle_wrenches_sln"][0 if side == "left" else 1],
                    dtype=np.float64,
                )

                def torque_ratio(x: np.ndarray) -> np.ndarray:
                    set_arm(x)
                    mujoco.mj_forward(model, data)
                    external_generalized_force = np.zeros(model.nv)
                    wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
                    grasp_position = (
                        data.xpos[wrist_id] + wrist_rotation @ WRIST_TO_GRASP_M
                    )
                    mujoco.mj_applyFT(
                        model,
                        data,
                        hand_wrench_on_robot[:3],
                        hand_wrench_on_robot[3:],
                        grasp_position,
                        wrist_id,
                        external_generalized_force,
                    )
                    required = data.qfrc_bias - external_generalized_force
                    return required[arm_dof_addresses] / ARM_EFFORT_LIMITS_NM

                if optimize_torque:
                    initial_ratio = torque_ratio(result.x)
                    augmented_initial = np.concatenate(
                        (result.x, (float(np.max(np.abs(initial_ratio))),))
                    )

                    def augmented_constraint(y: np.ndarray) -> np.ndarray:
                        return constraints(y[:7])

                    def torque_epigraph(y: np.ndarray) -> np.ndarray:
                        ratio = torque_ratio(y[:7])
                        return np.concatenate((y[7] - ratio, y[7] + ratio))

                    augmented_constraints: list[dict[str, Any]] = [
                        {"type": "eq", "fun": augmented_constraint},
                        {"type": "ineq", "fun": torque_epigraph},
                    ]
                    if avoid_self_collision:
                        augmented_constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda y: self_collision_margins(y[:7]),
                            }
                        )
                    result = minimize(
                        lambda y: float(
                            y[7] + 1.0e-6 * np.sum(np.square(y[:7] - x0))
                        ),
                        augmented_initial,
                        method="SLSQP",
                        bounds=(*bounds, (0.0, 2.0)),
                        constraints=tuple(augmented_constraints),
                        options={"ftol": 1.0e-13, "maxiter": 3000},
                    )
                    result.x = result.x[:7]
                residual = constraints(result.x)
                elbow_rotation = data.xmat[elbow_id].reshape(3, 3)
                surface_y = tuple(
                    float((data.xpos[elbow_id] + elbow_rotation @ point)[1])
                    for point in local_points
                )
                inward_retreat = np.asarray(
                    [sign * (before - after) for before, after in zip(surface_y_0, surface_y)]
                )
                delta = result.x - x0
                static_torque_ratio = torque_ratio(result.x)
                minimum_self_collision_margin = float(
                    np.min(self_collision_margins(result.x))
                )
                q_seed[joint_indices] = result.x
                q_delta[joint_indices] = delta
                side_rows[side] = {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "maximum_constraint_residual": float(np.max(np.abs(residual))),
                    "joint_delta_rad": delta,
                    "maximum_absolute_joint_delta_rad": float(np.max(np.abs(delta))),
                    "static_arm_torque_ratio": static_torque_ratio,
                    "maximum_static_arm_torque_ratio": float(
                        np.max(np.abs(static_torque_ratio))
                    ),
                    "minimum_self_collision_margin_m": (
                        minimum_self_collision_margin
                    ),
                    "surface_inward_retreat_m": inward_retreat,
                    "minimum_contact_distance_margin_m": float(
                        np.min(inward_retreat) - CONTACT_DISTANCE_M
                    ),
                }
            modes[mode_name] = {
                "q_seed": q_seed,
                "q_delta": q_delta,
                "sides": side_rows,
            }
            data.qpos[:] = qpos
            data.qpos[qpos_addresses] = q_seed
            mujoco.mj_forward(model, data)
            modes[mode_name]["self_collision_count"] = int(data.ncon)
        rows.append(
            {
                "gradient": float(pose["gradient"]),
                "root_pitch": root_pitch,
                "root_height": float(pose["root_height"]),
                "modes": modes,
            }
        )

    report = {
        "tool": "analyze_grasp_width_clearance",
        "joint_order": list(reset_solver.G1_JOINT_ORDER),
        "grasp_half_width_m": GRASP_HALF_WIDTH_M,
        "grasp_center_distance_m": 2.0 * GRASP_HALF_WIDTH_M,
        "self_collision_margin_m": SELF_COLLISION_MARGIN_M,
        "runtime_physx_offsets": {
            "elbow_contact_offset_m": ELBOW_CONTACT_OFFSET_M,
            "cart_base_contact_offset_m": CART_BASE_CONTACT_OFFSET_M,
            "combined_contact_distance_m": CONTACT_DISTANCE_M,
            "rest_offset_m": 0.0,
        },
        "robust_surface_retreat_target_m": ROBUST_SURFACE_RETREAT_M,
        "torque_feasible_surface_retreat_target_m": (
            TORQUE_FEASIBLE_SURFACE_RETREAT_M
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, default=_serial, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
