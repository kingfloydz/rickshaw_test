#!/usr/bin/env python3
"""Run the canonical reset-pose pipeline for all configured slopes.

Stage A runs 50 least-squares starts per slope and retains every candidate that
passes the original static hard gates. Stage B evaluates every retained pose in
an isolated, single-use Isaac Sim process for at most 2000 policy steps per
batch. The selected pose library is then validated once more as an assembled
unit and is published only if every hard gate passes. Selection is
lexicographic: survival steps, dynamic gate count, worst normalized risk, then
mean normalized risk.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import faulthandler
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

faulthandler.enable(all_threads=True)

from _isaaclab_wrappers import (
    REPOSITORY_ROOT,
    add_isaaclab_sources_to_path,
    add_project_source_to_path,
    isaaclab_root,
)

add_project_source_to_path()

from g1_rickshaw_lab.configuration import (  # noqa: E402
    G1_JOINT_ORDER,
    RESET_TORQUE_LIMIT_FRACTION,
    ResetPose,
    ResetPoseLibrary,
    SLOPE_GRADIENTS,
    load_reset_pose_library,
)
from g1_rickshaw_lab.rickshaw_spec import (  # noqa: E402
    HITCH_HALF_WIDTH,
    HITCH_X,
    HITCH_Z,
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
    WHEEL_RADIUS,
    WHEEL_TRACK,
)
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    TERRAIN_COLUMNS_PER_TYPE,
    terrain_index_for_gradient,
)
from g1_rickshaw_lab.static_equilibrium import (  # noqa: E402
    solve_fixed_contact_statics,
)


DEFAULT_URDF = (
    REPOSITORY_ROOT / "assets" / "g1_dex1" / "g1_29dof_mode_15_with_dex1_1.urdf"
)
GRAVITY = 9.81
POLICY_DOF_COUNT = len(G1_JOINT_ORDER)
ROOT_PITCH_INDEX = POLICY_DOF_COUNT
ROOT_HEIGHT_INDEX = POLICY_DOF_COUNT + 1
HAND_X_INDEX = POLICY_DOF_COUNT + 2
FOOT_X_INDEX = POLICY_DOF_COUNT + 3
DEX_FORWARD_MAX_PITCH_DEGREES = 70.0
DEX_FORWARD_MAX_PITCH_RAD = math.radians(DEX_FORWARD_MAX_PITCH_DEGREES)
DEX_FORWARD_MIN_DOT = math.cos(DEX_FORWARD_MAX_PITCH_RAD)
DEX_FORWARD_MAX_LATERAL = 0.01
RESET_STATIC_MULTISTARTS = 50
DEFAULT_TORQUE_TARGET = 0.80
BALANCED_ARM_TORQUE_WEIGHT = 20.0
BALANCED_ARM_TORQUE_HINGE_WEIGHT = 500.0
BALANCED_ARM_TORQUE_TARGET = 0.78
BALANCED_ARM_SEED_OFFSET = 3


def _stage_a_solve_plan() -> tuple[tuple[float, float | None], ...]:
    """Return execution order and continuation parent for every slope."""

    zero_index = SLOPE_GRADIENTS.index(0.0)
    positive = tuple(SLOPE_GRADIENTS[zero_index + 1 :])
    negative = tuple(reversed(SLOPE_GRADIENTS[:zero_index]))
    plan: list[tuple[float, float | None]] = [(0.0, None)]
    for branch in (positive, negative):
        parent = 0.0
        for gradient in branch:
            plan.append((gradient, parent))
            parent = gradient
    return tuple(plan)
MILD_ARM_TORQUE_WEIGHT = 10.0
MILD_ARM_TORQUE_HINGE_WEIGHT = 250.0
MILD_ARM_TORQUE_TARGET = 0.79
MILD_ARM_SEED_OFFSET = 1
DEFAULT_HAND_X_MIN = 0.12
DEFAULT_HAND_X_MAX = 0.26
DEFAULT_FAT2_ERROR_TOLERANCE = 0.12
DEFAULT_MAXIMUM_TORSO_PITCH = 0.45
DEFAULT_MAXIMUM_CONTINUATION_JOINT_DELTA = 0.35
DEFAULT_MAXIMUM_CONTINUATION_ARM_DELTA = 0.30
DEFAULT_Q_REF_JOINT_MARGIN = 0.04
DEFAULT_SOLVER_TOLERANCE = 1.0e-9
DEFAULT_SOLVER_WORKERS = 20
DEFAULT_ZMP_OPTIMIZATION_RESERVE_FRACTION = 0.025
PIPELINE_WORKER_PROGRESS_ENV = "G1_RICKSHAW_RESET_WORKER_PROGRESS"
CANDIDATE_BATCH_ENV = "G1_RICKSHAW_RESET_CANDIDATE_BATCH"
CANDIDATE_BATCH_SCHEMA_VERSION = 1
CANDIDATE_PROGRESS_SCHEMA_VERSION = 3
DEFAULT_STAGE_B_CANDIDATES_PER_SLOPE = TERRAIN_COLUMNS_PER_TYPE

# The solver parallelizes independent starts with processes. Keep each child on
# one BLAS/OpenMP thread so --workers maps predictably to occupied CPU cores.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")


def _static_load_scales(uncertainty: float) -> tuple[float, ...]:
    """Return unique load cases in deterministic worst-case evaluation order."""

    if uncertainty == 0.0:
        return (1.0,)
    return (1.0 - uncertainty, 1.0, 1.0 + uncertainty)


def _zmp_optimization_target(minimum_margin: float, reserve_fraction: float) -> float:
    """Keep the least-squares target safely inside the hard ZMP boundary."""

    return minimum_margin * (1.0 + reserve_fraction)


def _solver_worker_count(requested: int, multistarts: int) -> int:
    """Resolve ``--workers=0`` without oversubscribing the multistart bank."""

    available = os.cpu_count() or 1
    workers = min(DEFAULT_SOLVER_WORKERS, available) if requested == 0 else requested
    return max(1, min(workers, multistarts))


def _run_multistarts(worker: Any, count: int, workers: int) -> list[Any]:
    """Evaluate independent starts in forked, memory-isolated MuJoCo workers."""

    if workers == 1:
        return [worker(index) for index in range(count)]
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        return [worker(index) for index in range(count)]

    def child_entry(index: int, sender: Any) -> None:
        try:
            sender.send((True, worker(index)))
        except BaseException:  # Propagate a useful child traceback to the parent.
            sender.send((False, traceback.format_exc()))
        finally:
            sender.close()

    results: list[Any] = [None] * count
    for batch_start in range(0, count, workers):
        children: list[tuple[int, Any, Any, Any]] = []
        try:
            for index in range(batch_start, min(batch_start + workers, count)):
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=child_entry, args=(index, sender))
                try:
                    process.start()
                except BaseException:
                    receiver.close()
                    sender.close()
                    raise
                children.append((index, process, receiver, sender))
                sender.close()
            for index, process, receiver, _sender in children:
                try:
                    succeeded, payload = receiver.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        f"multistart worker {index + 1} exited without a result "
                        f"(exit code {process.exitcode})"
                    ) from exc
                finally:
                    receiver.close()
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"multistart worker {index + 1} failed with exit code "
                        f"{process.exitcode}"
                    )
                if not succeeded:
                    raise RuntimeError(
                        f"multistart worker {index + 1} raised an exception:\n{payload}"
                    )
                results[index] = payload
        finally:
            for _index, _process, receiver, sender in children:
                receiver.close()
                sender.close()
            for _index, process, _receiver, _sender in children:
                if process.is_alive():
                    process.terminate()
            for _index, process, _receiver, _sender in children:
                process.join()
    return results


def _minimum_joint_limit_margin(
    q: Any, joint_lower: Any, joint_upper: Any, np: Any
) -> float:
    """Return the smallest signed distance to a hard joint limit."""

    return float(np.min(np.minimum(q - joint_lower, joint_upper - q)))


def _worst_case_torque_ratios(
    required_by_scale: Any, effort_limits: Any, np: Any
) -> tuple[Any, float, float, float]:
    """Aggregate per-joint and body-group torque ratios across load cases."""

    ratios = np.abs(np.asarray(required_by_scale)) / effort_limits
    per_joint = np.max(ratios, axis=0)
    return (
        per_joint,
        float(np.max(per_joint[:12])),
        float(np.max(per_joint[12:15])),
        float(np.max(per_joint[15:])),
    )


def _allocate_support_torques(
    total_torque: Any, normal_fractions: Any, np: Any
) -> Any:
    """Distribute a realizable ground moment only to load-bearing feet."""

    fractions = np.asarray(normal_fractions, dtype=np.float64)
    return fractions[:, None] * np.asarray(total_torque, dtype=np.float64)[None, :]


def _per_foot_support_wrench_ratios(
    *,
    forces: Any,
    free_torques: Any,
    contact_points: Any,
    foot_origins: Any,
    contact_bounds: tuple[float, float, float, float],
    friction: float,
    np: Any,
    eps: float = 1.0e-9,
) -> tuple[Any, Any]:
    """Check unilateral, CoP, friction, and yaw capacity for each sole."""

    forces = np.asarray(forces, dtype=np.float64)
    free_torques = np.asarray(free_torques, dtype=np.float64)
    contact_points = np.asarray(contact_points, dtype=np.float64)
    foot_origins = np.asarray(foot_origins, dtype=np.float64)
    minimum_x, maximum_x, minimum_y, maximum_y = contact_bounds
    center_x = 0.5 * (minimum_x + maximum_x)
    center_y = 0.5 * (minimum_y + maximum_y)
    half_x = 0.5 * (maximum_x - minimum_x)
    half_y = 0.5 * (maximum_y - minimum_y)
    components = np.zeros((2, 4), dtype=np.float64)

    for foot_index in range(2):
        force = forces[foot_index]
        torque = free_torques[foot_index]
        normal_force = float(force[2])
        wrench_norm = float(np.linalg.norm(force)) + float(np.linalg.norm(torque))
        if normal_force <= eps:
            if normal_force < -eps or wrench_norm > eps:
                components[foot_index, :] = math.inf
            continue

        point = contact_points[foot_index] - foot_origins[foot_index]
        cop_x = float(point[0] - torque[1] / normal_force)
        cop_y = float(point[1] + torque[0] / normal_force)
        components[foot_index, 0] = abs(cop_x - center_x) / half_x
        components[foot_index, 1] = abs(cop_y - center_y) / half_y
        components[foot_index, 2] = (
            float(np.linalg.norm(force[:2])) / (friction * normal_force)
        )
        yaw_capacity = friction * normal_force * min(half_x, half_y)
        components[foot_index, 3] = abs(float(torque[2])) / yaw_capacity

    return np.max(components, axis=1), components


def _candidate_constraint_violation(
    *,
    hard_residual: float,
    hard_tolerance: float,
    lower_torque_ratio: float,
    waist_torque_ratio: float,
    arm_torque_ratio: float,
    q_ref_joint_margin: float,
    joint_margin: float,
    minimum_dex_forward_dot: float,
    maximum_dex_forward_lateral: float,
    zmp_margin: float,
    minimum_zmp_margin: float,
    friction_ratio: float,
    nominal_friction: float,
    root_equilibrium_residual: float,
    root_equilibrium_tolerance: float,
    fat2_error: float,
    fat2_error_tolerance: float,
    torso_pitch: float,
    maximum_torso_pitch: float,
    continuation_joint_delta: float,
    maximum_continuation_joint_delta: float,
    continuation_arm_delta: float,
    maximum_continuation_arm_delta: float,
    support_torque_ratio: float,
    self_collision_count: int,
    minimum_q_ref_joint_margin: float = 0.0,
    fat2_moment_error: float = 0.0,
    fat2_moment_tolerance: float = math.inf,
) -> float:
    """Return the largest normalized hard-constraint excess for one IK start."""

    return max(
        hard_residual / hard_tolerance - 1.0,
        lower_torque_ratio / RESET_TORQUE_LIMIT_FRACTION - 1.0,
        waist_torque_ratio / RESET_TORQUE_LIMIT_FRACTION - 1.0,
        arm_torque_ratio / RESET_TORQUE_LIMIT_FRACTION - 1.0,
        (minimum_q_ref_joint_margin - q_ref_joint_margin)
        / max(minimum_q_ref_joint_margin, joint_margin, 1.0e-6),
        (DEX_FORWARD_MIN_DOT - minimum_dex_forward_dot)
        / (1.0 - DEX_FORWARD_MIN_DOT),
        maximum_dex_forward_lateral / DEX_FORWARD_MAX_LATERAL - 1.0,
        (minimum_zmp_margin - zmp_margin) / max(minimum_zmp_margin, 1.0e-6),
        friction_ratio / nominal_friction - 1.0,
        root_equilibrium_residual / root_equilibrium_tolerance - 1.0,
        fat2_error / fat2_error_tolerance - 1.0,
        abs(torso_pitch) / maximum_torso_pitch - 1.0,
        continuation_joint_delta / maximum_continuation_joint_delta - 1.0,
        continuation_arm_delta / maximum_continuation_arm_delta - 1.0,
        support_torque_ratio - 1.0,
        fat2_moment_error / fat2_moment_tolerance - 1.0,
        float(self_collision_count > 0),
        0.0,
    )


def _candidate_rank_key(metrics: dict[str, Any], root_height_target: float) -> tuple[Any, ...]:
    """Prefer robust, continuous pulling poses once all hard gates pass."""

    return (
        metrics["violation"] > 0.0,
        metrics["violation"],
        metrics["fat2_error"],
        metrics.get("fat2_moment_error", 0.0),
        -metrics["zmp_margin"],
        max(
            metrics["arm_torque_ratio"],
            metrics["waist_torque_ratio"],
            metrics["lower_torque_ratio"],
        ),
        metrics["arm_torque_ratio"],
        metrics["waist_torque_ratio"],
        metrics["lower_torque_ratio"],
        metrics["continuation_arm_delta"],
        metrics["continuation_joint_delta"],
        metrics["arm_posture_error"],
        metrics["hand_x_error"],
        metrics["hard_residual"],
        metrics["cost"],
        abs(metrics["root_height"] - root_height_target),
    )


def _geom_pair_can_collide(model: Any, geom1: int, geom2: int) -> bool:
    """Mirror the MuJoCo filters relevant to source-URDF self collision.

    ``mj_geomDistance`` deliberately ignores contact filtering.  Calling it for
    welded or parent-child geometry therefore penalizes normal mechanical
    overlap, most notably the torso/shoulder interfaces in this model.
    """

    body1 = int(model.body_weldid[int(model.geom_bodyid[geom1])])
    body2 = int(model.body_weldid[int(model.geom_bodyid[geom2])])
    if body1 == body2:
        return False
    contact_mask = (
        int(model.geom_contype[geom1]) & int(model.geom_conaffinity[geom2])
    ) | (int(model.geom_contype[geom2]) & int(model.geom_conaffinity[geom1]))
    if contact_mask == 0:
        return False

    parent1 = int(model.body_weldid[int(model.body_parentid[body1])])
    parent2 = int(model.body_weldid[int(model.body_parentid[body2])])
    if parent1 == body2 or parent2 == body1:
        return False

    signature = (min(body1, body2) << 16) + max(body1, body2)
    return signature not in {int(value) for value in model.exclude_signature}


def _solver_dependencies():
    try:
        import mujoco
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "reset-pose generation requires numpy, scipy, and mujoco; run this script "
            "with the configured IsaacLab Python after installing the IK dependencies"
        ) from exc
    return mujoco, np, least_squares, Rotation


def _load_floating_urdf_model(urdf_path: Path, mujoco: Any):
    if not urdf_path.is_file():
        raise FileNotFoundError(f"G1+Dex URDF does not exist: {urdf_path}")
    asset_root = urdf_path.parent
    xml = urdf_path.read_text(encoding="utf-8").replace(
        'meshdir="meshes"', 'meshdir="."'
    )
    before = (
        '  <!-- [CAUTION] uncomment when convert to mujoco -->\n'
        '  <!-- <link name="world"></link>'
    )
    after = '  <link name="world"></link>'
    closing_before = '    <child link="pelvis"/>\n  </joint> -->'
    closing_after = '    <child link="pelvis"/>\n  </joint>'
    if before not in xml or closing_before not in xml:
        raise RuntimeError("source URDF no longer contains the expected floating-base block")
    xml = xml.replace(before, after, 1).replace(closing_before, closing_after, 1)
    assets = {
        str(path.relative_to(asset_root)): path.read_bytes()
        for path in (asset_root / "meshes").iterdir()
        if path.is_file()
    }
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def _joint_qpos_address(model: Any, name: str, mujoco: Any) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"URDF model lacks joint {name!r}")
    return int(model.jnt_qposadr[joint_id])


def _body_id(model: Any, name: str, mujoco: Any) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise RuntimeError(f"URDF model lacks body {name!r}")
    return int(result)


def _foot_contact_geometry(
    model: Any,
    body_ids: tuple[int, int],
    mujoco: Any,
    np: Any,
) -> tuple[float, float, float, float, float]:
    """Derive the flat-foot contact plane and support hull from URDF spheres."""

    per_foot: list[tuple[float, float, float, float, float]] = []
    for body_id in body_ids:
        geom_ids = [
            geom_id
            for geom_id in np.flatnonzero(model.geom_bodyid == body_id)
            if int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        ]
        if not geom_ids:
            raise RuntimeError("each ankle-roll body must contain collision geometry")
        if any(
            int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE)
            for geom_id in geom_ids
        ):
            raise RuntimeError("foot support derivation currently requires collision spheres")
        centers = np.asarray([model.geom_pos[geom_id] for geom_id in geom_ids])
        radii = np.asarray([model.geom_size[geom_id, 0] for geom_id in geom_ids])
        bottoms = centers[:, 2] - radii
        if float(np.ptp(bottoms)) > 1.0e-7:
            raise RuntimeError("foot collision spheres are not coplanar")
        per_foot.append(
            (
                -float(np.mean(bottoms)),
                float(np.min(centers[:, 0])),
                float(np.max(centers[:, 0])),
                float(np.min(centers[:, 1])),
                float(np.max(centers[:, 1])),
            )
        )
    if not np.allclose(per_foot[0], per_foot[1], atol=1.0e-7, rtol=0.0):
        raise RuntimeError("left and right foot contact geometry must match")
    return per_foot[0]


def _nominal_posture(np: Any) -> Any:
    q = np.zeros(29, dtype=np.float64)
    for offset in (0, 6):
        q[offset : offset + 6] = (-0.26, 0.0, 0.0, 0.675, -0.416, 0.0)
    # The D6 joint remains free about the crossbar. These seeds keep both
    # grippers near the forward-facing branch selected by the soft Dex +X prior.
    q[15:22] = (0.128, -0.089, 0.883, 0.333, -1.046, 0.818, -1.554)
    q[22:29] = (0.156, 0.089, -0.880, 0.306, 1.079, 0.800, 1.554)
    return q


def _high_handle_posture(np: Any) -> Any:
    """Deterministic higher-root seed that stays on the pulling-arm branch."""

    q = np.zeros(29, dtype=np.float64)
    for offset in (0, 6):
        q[offset : offset + 6] = (-0.45, 0.0, 0.0, 1.00, -0.55, 0.0)
    q[14] = 0.10
    q[15:22] = (0.35, -0.09, 0.90, 0.30, -1.10, 0.82, -1.50)
    q[22:29] = (0.35, 0.09, -0.90, 0.30, 1.10, 0.82, 1.50)
    return q


def _cart_fixed_contact_statics(
    gradient: float,
    alpha: float,
    pitch_torque_on_robot: float,
    np: Any,
    hitch_half_width: float = HITCH_HALF_WIDTH,
) -> Any:
    """Solve the nominal two-hitch/two-wheel cart equilibrium."""

    cosine = math.cos(alpha)
    sine = math.sin(alpha)
    hitch_from_axle = np.asarray((HITCH_X, 0.0, HITCH_Z - WHEEL_RADIUS))
    com_from_axle = np.asarray(
        (
            RICKSHAW_CENTER_OF_MASS[0],
            RICKSHAW_CENTER_OF_MASS[1],
            RICKSHAW_CENTER_OF_MASS[2] - WHEEL_RADIUS,
        )
    )
    rotation = np.asarray(
        ((cosine, 0.0, -sine), (0.0, 1.0, 0.0), (sine, 0.0, cosine))
    )
    handle = rotation @ hitch_from_axle
    com = rotation @ com_from_axle
    return solve_fixed_contact_statics(
        mass=RICKSHAW_TOTAL_MASS,
        gradient=gradient,
        com_from_axle_sln=(float(com[0]), float(com[1]), float(com[2])),
        handle_from_axle_sn=(float(handle[0]), float(handle[2])),
        hitch_half_width=hitch_half_width,
        wheel_track=WHEEL_TRACK,
        pitch_torque_on_robot=pitch_torque_on_robot,
        gravity=GRAVITY,
    )


def _solve_library(
    args: argparse.Namespace,
    *,
    resume_candidate_output: Path | None = None,
) -> tuple[ResetPoseLibrary, list[dict[str, Any]], dict[float, list[dict[str, Any]]]]:
    mujoco, np, least_squares, Rotation = _solver_dependencies()
    if not math.isfinite(args.hand_x_target):
        raise ValueError("hand-x target must be finite")
    hand_x_min = args.hand_x_min
    hand_x_max = args.hand_x_max
    if not math.isfinite(hand_x_min) or not math.isfinite(hand_x_max):
        raise ValueError("hand-x bounds must be finite")
    if not hand_x_min < hand_x_max:
        raise ValueError("hand-x minimum must be smaller than hand-x maximum")
    if not hand_x_min <= args.hand_x_target <= hand_x_max:
        raise ValueError("hand-x soft target must lie inside the hand-x bounds")
    if not math.isfinite(args.hand_x_slope_span) or args.hand_x_slope_span < 0.0:
        raise ValueError("hand-x slope span must be finite and non-negative")
    maximum_gradient = max(abs(float(value)) for value in SLOPE_GRADIENTS)

    def hand_x_target_for_gradient(gradient: float) -> float:
        return args.hand_x_target - (
            float(gradient) / maximum_gradient
        ) * args.hand_x_slope_span

    endpoint_targets = tuple(
        hand_x_target_for_gradient(gradient)
        for gradient in (SLOPE_GRADIENTS[0], SLOPE_GRADIENTS[-1])
    )
    if any(not hand_x_min <= target <= hand_x_max for target in endpoint_targets):
        raise ValueError(
            "slope-adaptive hand-x targets must remain inside the hand-x bounds"
        )
    model = _load_floating_urdf_model(Path(args.urdf).resolve(), mujoco)
    data = mujoco.MjData(model)

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in G1_JOINT_ORDER
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        missing = [name for name, joint_id in zip(G1_JOINT_ORDER, joint_ids) if joint_id < 0]
        raise RuntimeError(f"IK model is missing policy joints: {missing}")
    qpos_addresses = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    dof_addresses = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    joint_lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    joint_upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    lower = joint_lower + args.joint_margin
    upper = joint_upper - args.joint_margin
    for index in (18, 25):
        lower[index] = max(lower[index], args.minimum_elbow_flexion)
    for index in (15, 22):
        upper[index] = min(upper[index], args.maximum_shoulder_pitch)
    for index in (19, 26):
        lower[index] = max(lower[index], -args.maximum_wrist_roll)
        upper[index] = min(upper[index], args.maximum_wrist_roll)
    if np.any(lower >= upper):
        raise ValueError("joint margin or pulling-branch bounds remove a usable joint range")

    base_qpos = np.zeros(model.nq, dtype=np.float64)
    base_qpos[:7] = (0.0, 0.0, args.root_height, 1.0, 0.0, 0.0, 0.0)
    for name in (
        "left_dex1_finger_joint_1",
        "left_dex1_finger_joint_2",
        "right_dex1_finger_joint_1",
        "right_dex1_finger_joint_2",
    ):
        base_qpos[_joint_qpos_address(model, name, mujoco)] = args.dex_q_grasp

    bodies = {
        name: _body_id(model, name, mujoco)
        for name in (
            "pelvis",
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        )
    }
    (
        sole_body_height,
        contact_min_x,
        contact_max_x,
        contact_min_y,
        contact_max_y,
    ) = _foot_contact_geometry(
        model,
        (
            bodies["left_ankle_roll_link"],
            bodies["right_ankle_roll_link"],
        ),
        mujoco,
        np,
    )
    if abs(args.sole_body_height - sole_body_height) > 1.0e-7:
        raise ValueError(
            f"--sole-body-height={args.sole_body_height:.6f} does not match the "
            f"source-URDF contact height {sole_body_height:.6f}"
        )
    if args.foot_min_x < contact_min_x - 1.0e-7 or args.foot_max_x > contact_max_x + 1.0e-7:
        raise ValueError("configured foot X support bounds exceed the URDF contact hull")
    foot_local_half_width = max(abs(contact_min_y), abs(contact_max_y))
    torso_geoms = np.flatnonzero(model.geom_bodyid == bodies["torso_link"])
    arm_body_names = tuple(
        f"{side}_{link}_link"
        for side in ("left", "right")
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
    arm_body_ids = tuple(_body_id(model, name, mujoco) for name in arm_body_names)
    arm_geoms = np.concatenate(
        [np.flatnonzero(model.geom_bodyid == body_id) for body_id in arm_body_ids]
    )
    torso_arm_geom_pairs = tuple(
        (int(torso_geom), int(arm_geom))
        for torso_geom in torso_geoms
        for arm_geom in arm_geoms
        if _geom_pair_can_collide(model, int(torso_geom), int(arm_geom))
    )
    collision_from_to = np.empty(6, dtype=np.float64)
    robot_mass = float(model.body_subtreemass[bodies["pelvis"]])
    wrist_to_grasp = np.asarray(
        (args.wrist_to_dex_base_x + args.grasp_center_x, 0.0, 0.0),
        dtype=np.float64,
    )

    nominal = _nominal_posture(np)
    initial_qpos = base_qpos.copy()
    initial_qpos[qpos_addresses] = nominal
    data.qpos[:] = initial_qpos
    mujoco.mj_kinematics(model, data)
    foot_lateral = 0.5 * (
        abs(data.xpos[bodies["left_ankle_roll_link"], 1])
        + abs(data.xpos[bodies["right_ankle_roll_link"], 1])
    )
    support_min_y = -foot_lateral - foot_local_half_width
    support_max_y = foot_lateral + foot_local_half_width

    alpha_radius = math.hypot(HITCH_X, HITCH_Z - WHEEL_RADIUS)
    alpha_phase = math.atan2(HITCH_Z - WHEEL_RADIUS, HITCH_X)
    alpha_ratio = (args.hitch_height - WHEEL_RADIUS) / alpha_radius
    if not -1.0 <= alpha_ratio <= 1.0:
        raise ValueError("requested hitch height is geometrically infeasible")
    alpha = math.asin(alpha_ratio) - alpha_phase
    cart_rotation = Rotation.from_rotvec((0.0, -alpha, 0.0))
    grasp_roll = args.grasp_frame_roll
    grasp_local_rotations = {
        "left": Rotation.from_rotvec((grasp_roll, 0.0, 0.0)).as_matrix(),
        "right": Rotation.from_rotvec((-grasp_roll, 0.0, 0.0)).as_matrix(),
    }
    foot_x_by_slope = None
    if args.foot_x_by_slope is not None:
        foot_x_by_slope = dict(
            zip(SLOPE_GRADIENTS, args.foot_x_by_slope, strict=True)
        )
    root_pitch_by_slope = None
    if args.root_pitch_by_slope is not None:
        root_pitch_by_slope = dict(
            zip(SLOPE_GRADIENTS, args.root_pitch_by_slope, strict=True)
        )
    root_height_by_slope = None
    if args.root_height_by_slope is not None:
        root_height_by_slope = dict(
            zip(SLOPE_GRADIENTS, args.root_height_by_slope, strict=True)
        )
    def orientation_error(actual: Any, target: Any) -> Any:
        return Rotation.from_matrix(target.T @ actual).as_rotvec()

    diagnostics: list[dict[str, Any]] = []
    solved: dict[float, ResetPose] = {}
    candidate_bank: dict[float, list[dict[str, Any]]] = {}
    if resume_candidate_output is not None:
        solved, diagnostics, candidate_bank = _load_candidate_progress(
            resume_candidate_output,
            args.full_pose_multistarts,
            args.root_height,
            args,
        )
    flat_seed = np.concatenate(
        (
            nominal,
            np.asarray(
                (
                    0.0,
                    args.root_height,
                    hand_x_target_for_gradient(0.0),
                    -0.035,
                )
            ),
        )
    )
    seed_library = None
    if args.seed_library:
        seed_path = Path(args.seed_library).resolve()
        try:
            seed_library = load_reset_pose_library(seed_path)
            flat_pose = seed_library.pose_for_gradient(0.0)
            flat_seed[:POLICY_DOF_COUNT] = flat_pose.q_reset
            flat_seed[ROOT_PITCH_INDEX] = flat_pose.root_pitch
            flat_seed[ROOT_HEIGHT_INDEX] = flat_pose.root_height
            flat_seed[HAND_X_INDEX] = hand_x_target_for_gradient(0.0)
            flat_seed[FOOT_X_INDEX] = -0.035
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid IK seed library {seed_path}: {exc}") from exc
    lower_effort_limits = np.asarray((88.0, 88.0, 88.0, 139.0, 50.0, 50.0) * 2)
    waist_effort_limits = np.asarray((88.0, 50.0, 50.0))
    arm_effort_limits = np.asarray((25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4) * 2)
    policy_effort_limits = np.concatenate(
        (lower_effort_limits, waist_effort_limits, arm_effort_limits)
    )
    lower_stiffness = np.asarray(
        (
            args.leg_stiffness,
            args.leg_stiffness,
            args.leg_stiffness,
            args.leg_stiffness,
            args.foot_stiffness,
            args.foot_stiffness,
        )
        * 2,
        dtype=np.float64,
    )
    policy_stiffness = np.concatenate(
        (
            lower_stiffness,
            np.full(3, args.waist_stiffness, dtype=np.float64),
            np.full(14, args.arm_stiffness, dtype=np.float64),
        )
    )

    def solve_gradient(
        gradient: float,
        seed: Any,
        *,
        enforce_continuity: bool,
    ) -> Any:
        gamma = math.atan(gradient)
        root_pitch_seed = (
            float(seed[ROOT_PITCH_INDEX])
            if root_pitch_by_slope is None
            else root_pitch_by_slope[gradient]
        )
        root_height_seed = (
            float(seed[ROOT_HEIGHT_INDEX])
            if root_height_by_slope is None
            else root_height_by_slope[gradient]
        )
        gradient_base_qpos = base_qpos.copy()
        foot_x_target = (
            -0.035 if foot_x_by_slope is None else foot_x_by_slope[gradient]
        )
        hand_x_target = hand_x_target_for_gradient(gradient)
        cart_static_solution = _cart_fixed_contact_statics(
            gradient,
            alpha,
            0.0,
            np,
            hitch_half_width=args.hitch_half_width,
        )
        handle_wrenches_on_cart = np.asarray(
            cart_static_solution.handle_wrenches_sln, dtype=np.float64
        )
        wheel_contact_forces = np.asarray(
            cart_static_solution.wheel_contact_forces_sln, dtype=np.float64
        )
        cart_force_residual = np.asarray(
            cart_static_solution.cart_force_residual_sln, dtype=np.float64
        )
        cart_moment_residual = np.asarray(
            cart_static_solution.cart_moment_residual_sln, dtype=np.float64
        )
        if max(
            float(np.max(np.abs(cart_force_residual))),
            float(np.max(np.abs(cart_moment_residual))),
        ) > args.cart_equilibrium_tolerance:
            raise RuntimeError(
                f"gradient {gradient:+.2f} cart equilibrium residual exceeds "
                f"{args.cart_equilibrium_tolerance:.3e}: force="
                f"{np.array2string(cart_force_residual, precision=6)}, moment="
                f"{np.array2string(cart_moment_residual, precision=6)}"
            )
        if float(np.min(wheel_contact_forces[:, 2])) < args.minimum_wheel_normal_force:
            raise RuntimeError(
                f"gradient {gradient:+.2f} lifts a wheel below the static "
                f"{args.minimum_wheel_normal_force:.3f} N gate"
            )
        handle_force_norms = np.linalg.norm(handle_wrenches_on_cart[:, :3], axis=1)
        handle_torque_norms = np.linalg.norm(handle_wrenches_on_cart[:, 3:], axis=1)
        if float(np.max(handle_force_norms)) > args.d6_force_limit:
            raise RuntimeError(
                f"gradient {gradient:+.2f} requires a D6 force above "
                f"{args.d6_force_limit:.3f} N"
            )
        if float(np.max(handle_torque_norms)) > args.d6_torque_limit:
            raise RuntimeError(
                f"gradient {gradient:+.2f} requires a D6 torque above "
                f"{args.d6_torque_limit:.3f} Nm"
            )
        # Robot-side inverse statics uses the equal-and-opposite cart wrench.
        handle_wrenches_on_robot = -handle_wrenches_on_cart
        hand_force = np.sum(handle_wrenches_on_robot[:, :3], axis=0)
        nominal_cart_force_sn = np.asarray(
            (-hand_force[0], -hand_force[2]), dtype=np.float64
        )
        nominal_cart_tangent_difference = float(
            handle_wrenches_on_cart[0, 0] - handle_wrenches_on_cart[1, 0]
        )
        gravity_force = np.asarray(
            (
                -robot_mass * GRAVITY * math.sin(gamma),
                0.0,
                -robot_mass * GRAVITY * math.cos(gamma),
            )
        )
        world_upright_target = Rotation.from_rotvec((0.0, gamma, 0.0)).as_matrix()
        model.opt.gravity[:] = (
            -GRAVITY * math.sin(gamma),
            0.0,
            -GRAVITY * math.cos(gamma),
        )

        def qpos_from_vector(vector: Any) -> Any:
            qpos = gradient_base_qpos.copy()
            root_pitch = float(vector[ROOT_PITCH_INDEX])
            qpos[2] = float(vector[ROOT_HEIGHT_INDEX])
            qpos[3:7] = (
                math.cos(0.5 * root_pitch),
                0.0,
                math.sin(0.5 * root_pitch),
                0.0,
            )
            qpos[qpos_addresses] = vector[:POLICY_DOF_COUNT]
            return qpos

        load_scales = _static_load_scales(args.static_load_uncertainty)

        def resolved_hand_load(
            hand_force_scale: float,
            cart_force_sn: tuple[float, float] | None,
            cart_tangent_difference: float = 0.0,
        ) -> Any:
            if cart_force_sn is None:
                return hand_force_scale * handle_wrenches_on_robot
            cart_tangent, cart_normal = cart_force_sn
            cart_wrenches = np.zeros((2, 6), dtype=np.float64)
            cart_wrenches[:, 0] = np.asarray(
                (
                    0.5 * (cart_tangent + cart_tangent_difference),
                    0.5 * (cart_tangent - cart_tangent_difference),
                )
            )
            cart_wrenches[:, 2] = 0.5 * cart_normal
            return -cart_wrenches

        def grasp_positions_from_state() -> dict[str, Any]:
            positions: dict[str, Any] = {}
            for side in ("left", "right"):
                wrist_id = bodies[f"{side}_wrist_yaw_link"]
                wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
                positions[side] = (
                    data.xpos[wrist_id] + wrist_rotation @ wrist_to_grasp
                )
            return positions

        def static_zmp_from_state(
            hand_force_scale: float = 1.0,
            cart_force_sn: tuple[float, float] | None = None,
        ) -> tuple[Any, Any, Any]:
            com = data.subtree_com[bodies["pelvis"]]
            hand_wrenches = resolved_hand_load(
                hand_force_scale, cart_force_sn
            )
            scaled_hand_force = np.sum(hand_wrenches[:, :3], axis=0)
            reaction = -(gravity_force + scaled_hand_force)
            external_moment = np.cross(com, gravity_force)
            for hand_index, grasp_position in enumerate(
                grasp_positions_from_state().values()
            ):
                external_moment += np.cross(
                    grasp_position, hand_wrenches[hand_index, :3]
                )
                external_moment += hand_wrenches[hand_index, 3:]
            if reaction[2] <= 0.0:
                return np.full(2, np.nan), com.copy(), reaction
            zmp = np.asarray(
                (
                    external_moment[1] / reaction[2],
                    -external_moment[0] / reaction[2],
                ),
                dtype=np.float64,
            )
            return zmp, com.copy(), reaction

        def foot_contact_distribution() -> dict[str, tuple[float, float]]:
            # Select one fixed member of the double-support wrench family.  A
            # load-dependent split makes the inverse-statics map non-affine and
            # prevents exact online payload compensation.
            return {
                "left": (0.5, foot_lateral),
                "right": (0.5, -foot_lateral),
            }

        def required_joint_torque(
            qpos: Any,
            hand_force_scale: float = 1.0,
            *,
            state_is_current: bool = False,
            cart_force_sn: tuple[float, float] | None = None,
            cart_tangent_difference: float = 0.0,
        ) -> tuple[Any, Any, Any, Any]:
            if not state_is_current:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
            external_generalized_force = np.zeros(model.nv, dtype=np.float64)
            hand_wrenches = resolved_hand_load(
                hand_force_scale,
                cart_force_sn,
                cart_tangent_difference,
            )
            grasp_positions = grasp_positions_from_state()
            for hand_index, side in enumerate(("left", "right")):
                wrist_id = bodies[f"{side}_wrist_yaw_link"]
                mujoco.mj_applyFT(
                    model,
                    data,
                    hand_wrenches[hand_index, :3],
                    hand_wrenches[hand_index, 3:],
                    grasp_positions[side],
                    wrist_id,
                    external_generalized_force,
                )

            zmp, _, reaction = static_zmp_from_state(
                hand_force_scale, cart_force_sn
            )
            foot_contacts = foot_contact_distribution()
            for side in ("left", "right"):
                foot_id = bodies[f"{side}_ankle_roll_link"]
                fraction, contact_y = foot_contacts[side]
                contact_point = np.asarray((zmp[0], contact_y, 0.0))
                mujoco.mj_applyFT(
                    model,
                    data,
                    fraction * reaction,
                    np.zeros(3),
                    contact_point,
                    foot_id,
                    external_generalized_force,
                )

            raw_root_residual = data.qfrc_bias[:6] - external_generalized_force[:6]
            torque_map = np.zeros((6, 3), dtype=np.float64)
            for axis in range(3):
                generalized = np.zeros(model.nv, dtype=np.float64)
                unit_torque = np.zeros(3, dtype=np.float64)
                unit_torque[axis] = 1.0
                for side in ("left", "right"):
                    foot_id = bodies[f"{side}_ankle_roll_link"]
                    fraction, _ = foot_contacts[side]
                    mujoco.mj_applyFT(
                        model,
                        data,
                        np.zeros(3),
                        fraction * unit_torque,
                        data.xpos[foot_id],
                        foot_id,
                        generalized,
                    )
                torque_map[:, axis] = generalized[:6]
            support_torque, *_ = np.linalg.lstsq(
                torque_map, raw_root_residual, rcond=None
            )
            for side in ("left", "right"):
                foot_id = bodies[f"{side}_ankle_roll_link"]
                fraction, _ = foot_contacts[side]
                mujoco.mj_applyFT(
                    model,
                    data,
                    np.zeros(3),
                    fraction * support_torque,
                    data.xpos[foot_id],
                    foot_id,
                    external_generalized_force,
                )

            required = data.qfrc_bias - external_generalized_force
            return (
                required[dof_addresses],
                required[:6].copy(),
                support_torque,
                raw_root_residual,
            )

        def static_torque_basis(
            qpos: Any,
            *,
            state_is_current: bool = False,
        ) -> tuple[Any, Any, Any, Any]:
            tau_unloaded = required_joint_torque(
                qpos,
                state_is_current=state_is_current,
                cart_force_sn=(0.0, 0.0),
            )[0]
            tau_tangent = required_joint_torque(
                qpos,
                state_is_current=True,
                cart_force_sn=(1.0, 0.0),
            )[0]
            tau_normal = required_joint_torque(
                qpos,
                state_is_current=True,
                cart_force_sn=(0.0, 1.0),
            )[0]
            tau_tangent_difference = required_joint_torque(
                qpos,
                state_is_current=True,
                cart_force_sn=(0.0, 0.0),
                cart_tangent_difference=1.0,
            )[0]
            return (
                tau_unloaded,
                tau_tangent - tau_unloaded,
                tau_normal - tau_unloaded,
                tau_tangent_difference - tau_unloaded,
            )

        def support_wrench_ratios(
            support_torque: Any, reaction: Any, zmp: Any
        ) -> tuple[float, Any]:
            contacts = foot_contact_distribution()
            sides = ("left", "right")
            fractions = np.asarray([contacts[side][0] for side in sides])
            forces = fractions[:, None] * reaction[None, :]
            free_torques = _allocate_support_torques(support_torque, fractions, np)
            contact_points = np.asarray(
                [(float(zmp[0]), contacts[side][1], 0.0) for side in sides]
            )
            foot_origins = np.asarray(
                [data.xpos[bodies[f"{side}_ankle_roll_link"]] for side in sides]
            )
            ratios, components = _per_foot_support_wrench_ratios(
                forces=forces,
                free_torques=free_torques,
                contact_points=contact_points,
                foot_origins=foot_origins,
                contact_bounds=(
                    contact_min_x,
                    contact_max_x,
                    contact_min_y,
                    contact_max_y,
                ),
                friction=args.nominal_friction,
                np=np,
            )
            return float(np.max(ratios)), components

        def evaluate_static_cases(
            qpos: Any,
            foot_x: float,
            *,
            state_is_current: bool = False,
        ) -> dict[str, Any]:
            if not state_is_current:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
            support_min_x = foot_x + args.foot_min_x
            support_max_x = foot_x + args.foot_max_x
            cases: list[dict[str, Any]] = []
            required_by_scale: list[Any] = []
            # Both endpoints of the reset homotopy must be statically feasible.
            # The zero-load endpoint supplies gravity compensation while the
            # existing load cases cover the fully coupled grasp state.
            endpoint_scales = tuple(dict.fromkeys((0.0, *load_scales)))
            for scale in endpoint_scales:
                required, root_components, support_torque, raw_root_components = (
                    required_joint_torque(
                        qpos,
                        scale,
                        state_is_current=True,
                    )
                )
                zmp, com, reaction = static_zmp_from_state(scale)
                support_ratio, support_components = support_wrench_ratios(
                    support_torque, reaction, zmp
                )
                zmp_margin = min(
                    float(zmp[0]) - support_min_x,
                    support_max_x - float(zmp[0]),
                    float(zmp[1]) - support_min_y,
                    support_max_y - float(zmp[1]),
                )
                friction_ratio = float(np.linalg.norm(reaction[:2])) / float(reaction[2])
                required_by_scale.append(required)
                cases.append(
                    {
                        "scale": scale,
                        "required": required,
                        "root_components": root_components,
                        "raw_root_components": raw_root_components,
                        "support_torque": support_torque,
                        "support_wrench_ratio": support_ratio,
                        "support_wrench_components": support_components,
                        "zmp": zmp,
                        "com": com,
                        "reaction": reaction,
                        "zmp_margin": zmp_margin,
                        "friction_ratio": friction_ratio,
                    }
                )
            torque_ratio, lower_ratio, waist_ratio, arm_ratio = (
                _worst_case_torque_ratios(required_by_scale, policy_effort_limits, np)
            )
            nominal = min(cases, key=lambda case: abs(float(case["scale"]) - 1.0))
            unloaded = min(cases, key=lambda case: abs(float(case["scale"])))
            return {
                "cases": cases,
                "nominal": nominal,
                "unloaded": unloaded,
                "torque_ratio": torque_ratio,
                "lower_torque_ratio": lower_ratio,
                "waist_torque_ratio": waist_ratio,
                "arm_torque_ratio": arm_ratio,
                "zmp_margin": min(float(case["zmp_margin"]) for case in cases),
                "friction_ratio": max(float(case["friction_ratio"]) for case in cases),
                "root_equilibrium_residual": max(
                    float(np.max(np.abs(case["root_components"]))) for case in cases
                ),
                "raw_root_equilibrium_residual": max(
                    float(np.max(np.abs(case["raw_root_components"]))) for case in cases
                ),
                "support_wrench_ratio": max(
                    float(case["support_wrench_ratio"]) for case in cases
                ),
            }

        def fat2_moment_balance(
            com: Any, hand_x: float, foot_x: float
        ) -> tuple[float, float, float]:
            support_center = foot_x + args.foot_center_offset_x
            handle_s = hand_x - support_center
            hand_moment = handle_s * hand_force[2] - args.hitch_height * hand_force[0]
            required_gravity_moment = hand_moment
            gravity_counter_moment = robot_mass * GRAVITY * (
                (float(com[0]) - support_center) * math.cos(gamma)
                - float(com[2]) * math.sin(gamma)
            )
            return (
                required_gravity_moment,
                gravity_counter_moment,
                gravity_counter_moment - required_gravity_moment,
            )

        def fat2_target_angle(com: Any, hand_x: float, foot_x: float) -> float:
            required_gravity_moment, _, _ = fat2_moment_balance(
                com, hand_x, foot_x
            )
            support_center = foot_x + args.foot_center_offset_x
            com_radius = math.hypot(com[0] - support_center, com[2])
            ratio = required_gravity_moment / (robot_mass * GRAVITY * com_radius)
            limit = math.sin(args.fat2_theta_max)
            return math.asin(float(np.clip(ratio, -limit, limit)))

        def reset_torso_target_angle(com: Any, hand_x: float, foot_x: float) -> float:
            fat2_angle = fat2_target_angle(com, hand_x, foot_x)
            if gradient == 0.0:
                return fat2_angle
            uphill_sign = math.copysign(1.0, gradient)
            return float(
                np.clip(
                    fat2_angle + uphill_sign * args.uphill_lean_margin,
                    -args.fat2_theta_max,
                    args.fat2_theta_max,
                )
            )

        def torso_pitch_from_world(qpos: Any) -> float:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            torso_rotation = data.xmat[bodies["torso_link"]].reshape(3, 3)
            return (
                math.atan2(
                    float(torso_rotation[0, 2]),
                    float(torso_rotation[2, 2]),
                )
                - gamma
            )

        def project_seed_to_fat2(seed_vector: Any) -> tuple[Any, float, float, float]:
            projected = seed_vector.copy()
            initial_pitch = torso_pitch_from_world(qpos_from_vector(projected))
            target_pitch = initial_pitch
            for _ in range(args.fat2_seed_iterations):
                qpos = qpos_from_vector(projected)
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                mujoco.mj_comPos(model, data)
                com = data.subtree_com[bodies["pelvis"]]
                target_pitch = reset_torso_target_angle(
                    com,
                    float(projected[HAND_X_INDEX]),
                    float(projected[FOOT_X_INDEX]),
                )
                current_pitch = torso_pitch_from_world(qpos)
                error = target_pitch - current_pitch
                if abs(error) <= args.fat2_seed_tolerance:
                    break

                root_fraction = (
                    args.fat2_seed_root_fraction
                    if root_pitch_by_slope is None
                    else 0.0
                )
                root_delta = root_fraction * error
                projected[ROOT_PITCH_INDEX] = np.clip(
                    projected[ROOT_PITCH_INDEX] + root_delta,
                    args.root_pitch_min,
                    args.root_pitch_max,
                )
                projected[14] = np.clip(
                    projected[14] + error - root_delta,
                    lower[14],
                    upper[14],
                )

            final_qpos = qpos_from_vector(projected)
            data.qpos[:] = final_qpos
            mujoco.mj_forward(model, data)
            final_com = data.subtree_com[bodies["pelvis"]]
            target_pitch = reset_torso_target_angle(
                final_com,
                float(projected[HAND_X_INDEX]),
                float(projected[FOOT_X_INDEX]),
            )
            projected_pitch = torso_pitch_from_world(final_qpos)
            return projected, initial_pitch, target_pitch, projected_pitch

        def residual(vector: Any, torque_profile: dict[str, float]) -> Any:
            qpos = qpos_from_vector(vector)
            hand_x = float(vector[HAND_X_INDEX])
            foot_x = float(vector[FOOT_X_INDEX])
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            values: list[float] = []
            dex_forward_errors: list[float] = []
            dex_forward_target = np.asarray(
                (
                    math.cos(args.dex_forward_target_pitch),
                    0.0,
                    math.sin(args.dex_forward_target_pitch),
                )
            )
            for side, sign in (("left", 1.0), ("right", -1.0)):
                foot_id = bodies[f"{side}_ankle_roll_link"]
                foot_target = np.asarray((foot_x, sign * foot_lateral, sole_body_height))
                values.extend(args.hard_weight * (data.xpos[foot_id] - foot_target))
                values.extend(
                    args.hard_weight
                    * orientation_error(data.xmat[foot_id].reshape(3, 3), np.eye(3))
                )

                wrist_id = bodies[f"{side}_wrist_yaw_link"]
                wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
                grasp_position = data.xpos[wrist_id] + wrist_rotation @ wrist_to_grasp
                grasp_target = np.asarray(
                    (hand_x, sign * args.hitch_half_width, args.hitch_height)
                )
                values.extend(args.hard_weight * (grasp_position - grasp_target))
                grasp_rotation = wrist_rotation @ grasp_local_rotations[side]
                grasp_orientation_error = orientation_error(
                    grasp_rotation, cart_rotation.as_matrix()
                )
                # The rod axis is hitch-frame +Y. Rotation about the rod is
                # physically free, so only rotX/rotZ belong to the hard IK.
                grasp_orientation_error[1] = 0.0
                values.extend(args.hard_weight * grasp_orientation_error)
                dex_forward_errors.extend(
                    wrist_rotation[:, 0] - dex_forward_target
                )

            values.extend(args.dex_forward_weight * np.asarray(dex_forward_errors))

            posture_target = nominal.copy()
            posture_target[14] = gamma
            posture_weight = np.full(29, 0.25)
            posture_weight[:12] = 0.15
            posture_weight[12:15] = 1.0
            posture_weight[[16, 17, 23, 24]] = 1.0
            posture_weight[[18, 20, 21, 25, 27, 28]] = 0.5
            values.extend(
                posture_weight * (vector[:POLICY_DOF_COUNT] - posture_target)
            )
            values.extend(
                (
                    args.root_pitch_weight * float(vector[ROOT_PITCH_INDEX]),
                    args.root_height_weight
                    * (float(vector[ROOT_HEIGHT_INDEX]) - args.root_height)
                    / 0.05,
                )
            )
            values.extend(
                (
                    args.hand_x_weight * (hand_x - hand_x_target) / 0.05,
                    args.foot_x_weight * (foot_x - foot_x_target) / 0.05,
                )
            )

            static_cases = evaluate_static_cases(qpos, foot_x, state_is_current=True)
            nominal_static = static_cases["nominal"]
            zmp = nominal_static["zmp"]
            com = nominal_static["com"]
            support_center = foot_x + args.foot_center_offset_x
            values.append(args.zmp_center_weight * (zmp[0] - support_center) / 0.02)
            values.append(args.zmp_center_weight * zmp[1] / 0.02)
            zmp_optimization_target = _zmp_optimization_target(
                args.minimum_zmp_margin,
                args.zmp_optimization_reserve_fraction,
            )
            for static_case in static_cases["cases"]:
                margin_deficit = max(
                    zmp_optimization_target - float(static_case["zmp_margin"]),
                    0.0,
                )
                values.append(
                    args.zmp_margin_weight
                    * margin_deficit
                    / max(args.minimum_zmp_margin, 1.0e-6)
                )
            torso_rotation = data.xmat[bodies["torso_link"]].reshape(3, 3)
            reset_torso_angle = reset_torso_target_angle(
                com, float(hand_x), float(foot_x)
            )
            _, _, fat2_moment_error = fat2_moment_balance(com, hand_x, foot_x)
            # The guide defines FAT2 relative to world vertical. In this solver
            # the terrain normal is +Z, while world vertical is pitched by
            # gamma, so compose both angles in the slope frame.
            fat2_target = Rotation.from_rotvec(
                (0.0, gamma + reset_torso_angle, 0.0)
            ).as_matrix()
            values.extend(
                args.world_upright_weight
                * orientation_error(torso_rotation, world_upright_target)
            )
            values.extend(
                args.fat2_torso_weight * orientation_error(torso_rotation, fat2_target)
            )
            values.append(
                args.fat2_moment_weight
                * fat2_moment_error
                / args.fat2_moment_tolerance
            )

            torque_ratio = static_cases["torque_ratio"]
            lower_torque_ratio = torque_ratio[:12]
            waist_torque_ratio = torque_ratio[12:15]
            arm_torque_ratio = torque_ratio[15:]
            values.extend(args.lower_torque_weight * lower_torque_ratio)
            values.extend(
                args.lower_torque_hinge_weight
                * np.maximum(
                    lower_torque_ratio - args.ik_lower_torque_target_fraction, 0.0
                )
            )
            values.extend(args.lower_torque_weight * waist_torque_ratio)
            values.extend(
                args.lower_torque_hinge_weight
                * np.maximum(
                    waist_torque_ratio - args.ik_lower_torque_target_fraction, 0.0
                )
            )
            values.extend(torque_profile["weight"] * arm_torque_ratio)
            values.extend(
                torque_profile["hinge_weight"]
                * np.maximum(
                    arm_torque_ratio - torque_profile["target_fraction"], 0.0
                )
            )
            q_ref = (
                vector[:POLICY_DOF_COUNT]
                + nominal_static["required"] / policy_stiffness
            )
            q_ref_lower_margin = q_ref - joint_lower
            q_ref_upper_margin = joint_upper - q_ref
            values.extend(
                args.q_ref_margin_weight
                * np.maximum(args.q_ref_joint_margin - q_ref_lower_margin, 0.0)
                / 0.01
            )
            values.extend(
                args.q_ref_margin_weight
                * np.maximum(args.q_ref_joint_margin - q_ref_upper_margin, 0.0)
                / 0.01
            )
            for torso_geom, arm_geom in torso_arm_geom_pairs:
                distance = mujoco.mj_geomDistance(
                    model,
                    data,
                    torso_geom,
                    arm_geom,
                    args.collision_detection_distance,
                    collision_from_to,
                )
                values.append(
                    args.collision_weight
                    * max(args.minimum_collision_distance - distance, 0.0)
                )
            return np.asarray(values, dtype=np.float64)

        foot_lower = -0.15
        foot_upper = 0.25
        if foot_x_by_slope is not None:
            foot_lower = foot_x_target - args.foot_x_tolerance
            foot_upper = foot_x_target + args.foot_x_tolerance
        root_pitch_lower = args.root_pitch_min
        root_pitch_upper = args.root_pitch_max
        if root_pitch_by_slope is not None:
            root_pitch_lower = root_pitch_seed - 1.0e-7
            root_pitch_upper = root_pitch_seed + 1.0e-7
        root_height_lower = args.root_height_min
        root_height_upper = args.root_height_max
        if root_height_by_slope is not None:
            root_height_lower = root_height_seed - 1.0e-7
            root_height_upper = root_height_seed + 1.0e-7
        lower_bound = np.concatenate(
            (
                lower,
                (
                    root_pitch_lower,
                    root_height_lower,
                    hand_x_min,
                    foot_lower,
                ),
            )
        )
        upper_bound = np.concatenate(
            (
                upper,
                (
                    root_pitch_upper,
                    root_height_upper,
                    hand_x_max,
                    foot_upper,
                ),
            )
        )
        base_seed = seed.copy()
        base_seed[ROOT_PITCH_INDEX] = root_pitch_seed
        base_seed[ROOT_HEIGHT_INDEX] = root_height_seed
        elevated_seed = base_seed.copy()
        elevated_seed[:POLICY_DOF_COUNT] = _high_handle_posture(np)
        elevated_seed[14] += gamma
        if root_pitch_by_slope is None:
            elevated_seed[ROOT_PITCH_INDEX] = -0.10
        if root_height_by_slope is None:
            elevated_seed[ROOT_HEIGHT_INDEX] = 0.66
        elevated_seed[HAND_X_INDEX] = hand_x_target
        elevated_seed[FOOT_X_INDEX] = 0.01
        root_offsets = (0.0, -0.04, 0.04, -0.08, 0.08)
        height_offsets = (0.0, -0.025, -0.025, 0.025, 0.025)
        hand_offsets = (0.0, -0.03, 0.03, -0.06, 0.06)
        arm_seed_scale = np.asarray((0.25, 0.20, 0.30, 0.25, 0.30, 0.30, 0.30))
        right_symmetry = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
        candidates: list[
            tuple[Any, tuple[float, float, float], dict[str, Any], dict[str, Any]]
        ] = []
        family_size = (args.full_pose_multistarts + 1) // 2

        def solve_start(start_index: int) -> Any:
            profile_slot = start_index % 4
            torque_profile_name = (
                "balanced_arm"
                if profile_slot == 1
                else "mild_arm"
                if profile_slot == 3
                else "nominal"
            )
            balanced_arm_profile = torque_profile_name == "balanced_arm"
            mild_arm_profile = torque_profile_name == "mild_arm"
            unperturbed_profile_start = start_index in (0, 1, 3)
            torque_profile = {
                "weight": (
                    max(args.arm_torque_weight, BALANCED_ARM_TORQUE_WEIGHT)
                    if balanced_arm_profile
                    else max(args.arm_torque_weight, MILD_ARM_TORQUE_WEIGHT)
                    if mild_arm_profile
                    else args.arm_torque_weight
                ),
                "hinge_weight": (
                    max(
                        args.arm_torque_hinge_weight,
                        BALANCED_ARM_TORQUE_HINGE_WEIGHT,
                    )
                    if balanced_arm_profile
                    else max(
                        args.arm_torque_hinge_weight,
                        MILD_ARM_TORQUE_HINGE_WEIGHT,
                    )
                    if mild_arm_profile
                    else args.arm_torque_hinge_weight
                ),
                "target_fraction": (
                    min(
                        args.ik_arm_torque_target_fraction,
                        BALANCED_ARM_TORQUE_TARGET,
                    )
                    if balanced_arm_profile
                    else min(
                        args.ik_arm_torque_target_fraction,
                        MILD_ARM_TORQUE_TARGET,
                    )
                    if mild_arm_profile
                    else args.ik_arm_torque_target_fraction
                ),
            }
            elevated_family = start_index >= family_size
            family_index = start_index - family_size if elevated_family else start_index
            start = elevated_seed.copy() if elevated_family else base_seed.copy()
            offset_index = family_index % len(root_offsets)
            scale = 1.0 + family_index // len(root_offsets)
            if not unperturbed_profile_start:
                if root_pitch_by_slope is None:
                    start[ROOT_PITCH_INDEX] += scale * root_offsets[offset_index]
                if root_height_by_slope is None:
                    start[ROOT_HEIGHT_INDEX] += scale * height_offsets[offset_index]
                start[HAND_X_INDEX] += scale * hand_offsets[offset_index]
            if family_index > 0 and not unperturbed_profile_start:
                pair_index = (family_index - 1) // 2
                pair_sign = -1.0 if family_index % 2 == 0 else 1.0
                rng = np.random.default_rng(
                    args.seed
                    + (BALANCED_ARM_SEED_OFFSET if balanced_arm_profile else 0)
                    + (MILD_ARM_SEED_OFFSET if mild_arm_profile else 0)
                    + int(round((gradient - SLOPE_GRADIENTS[0]) * 100.0)) * 10_000
                    + (1_000_000 if elevated_family else 0)
                    + pair_index
                )
                left_delta = (
                    pair_sign
                    * args.arm_seed_noise_scale
                    * arm_seed_scale
                    * rng.standard_normal(7)
                )
                start[15:22] += left_delta
                start[22:29] += right_symmetry * left_delta
            preserve_base_seed = not elevated_family and unperturbed_profile_start
            if preserve_base_seed:
                physical_seed = start.copy()
                physical_qpos = qpos_from_vector(physical_seed)
                data.qpos[:] = physical_qpos
                mujoco.mj_forward(model, data)
                physical_seed[HAND_X_INDEX] = float(
                    np.mean(
                        [
                            (
                                data.xpos[bodies[f"{side}_wrist_yaw_link"]]
                                + data.xmat[bodies[f"{side}_wrist_yaw_link"]]
                                .reshape(3, 3)
                                @ wrist_to_grasp
                            )[0]
                            for side in ("left", "right")
                        ]
                    )
                )
                physical_seed[FOOT_X_INDEX] = float(
                    np.mean(
                        [
                            data.xpos[bodies[f"{side}_ankle_roll_link"], 0]
                            for side in ("left", "right")
                        ]
                    )
                )
                start = physical_seed
                seed_qpos = qpos_from_vector(start)
                data.qpos[:] = seed_qpos
                mujoco.mj_forward(model, data)
                mujoco.mj_comPos(model, data)
                seed_com = data.subtree_com[bodies["pelvis"]]
                seed_initial_pitch = torso_pitch_from_world(seed_qpos)
                seed_target_pitch = reset_torso_target_angle(
                    seed_com,
                    float(start[HAND_X_INDEX]),
                    float(start[FOOT_X_INDEX]),
                )
                seed_projected_pitch = seed_initial_pitch
            else:
                (
                    start,
                    seed_initial_pitch,
                    seed_target_pitch,
                    seed_projected_pitch,
                ) = project_seed_to_fat2(start)
            bounded_seed = np.minimum(
                np.maximum(start, np.nextafter(lower_bound, upper_bound)),
                np.nextafter(upper_bound, lower_bound),
            )
            candidate = least_squares(
                lambda vector: residual(vector, torque_profile),
                bounded_seed,
                bounds=(lower_bound, upper_bound),
                max_nfev=args.max_evaluations,
                xtol=args.solver_tolerance,
                ftol=args.solver_tolerance,
                gtol=args.solver_tolerance,
                x_scale="jac",
            )
            if not np.all(np.isfinite(candidate.x)):
                return None
            candidate_qpos = qpos_from_vector(candidate.x)
            candidate_hard_error = float(
                np.max(
                    np.abs(residual(candidate.x, torque_profile)[:24])
                )
                / args.hard_weight
            )
            candidate_foot_x = float(candidate.x[FOOT_X_INDEX])
            candidate_static = evaluate_static_cases(
                candidate_qpos,
                candidate_foot_x,
            )
            candidate_nominal_static = candidate_static["nominal"]
            candidate_required = candidate_nominal_static["required"]
            (
                candidate_tau_unloaded,
                candidate_tau_per_tangent_force,
                candidate_tau_per_normal_force,
                candidate_tau_per_tangent_difference,
            ) = static_torque_basis(
                candidate_qpos,
                state_is_current=True,
            )
            candidate_basis_required = (
                candidate_tau_unloaded
                + nominal_cart_force_sn[0] * candidate_tau_per_tangent_force
                + nominal_cart_force_sn[1] * candidate_tau_per_normal_force
                + nominal_cart_tangent_difference
                * candidate_tau_per_tangent_difference
            )
            if not np.allclose(
                candidate_basis_required,
                candidate_required,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise RuntimeError(
                    "static torque basis does not reconstruct the nominal load"
                )
            candidate_torque_ratio = candidate_static["torque_ratio"]
            candidate_lower_ratio = candidate_static["lower_torque_ratio"]
            candidate_waist_ratio = candidate_static["waist_torque_ratio"]
            candidate_arm_ratio = candidate_static["arm_torque_ratio"]
            candidate_arm_joint = int(np.argmax(candidate_torque_ratio[15:])) + 15
            candidate_q_ref = candidate.x[:POLICY_DOF_COUNT] + (
                candidate_basis_required / policy_stiffness
            )
            candidate_q_ref_unloaded = (
                candidate.x[:POLICY_DOF_COUNT]
                + candidate_tau_unloaded / policy_stiffness
            )
            candidate_q_ref_margin = min(
                _minimum_joint_limit_margin(
                    candidate_q_ref, joint_lower, joint_upper, np
                ),
                _minimum_joint_limit_margin(
                    candidate_q_ref_unloaded, joint_lower, joint_upper, np
                ),
            )
            candidate_com = candidate_nominal_static["com"]
            candidate_zmp_margin = candidate_static["zmp_margin"]
            candidate_friction_ratio = candidate_static["friction_ratio"]
            candidate_support_torque_ratio = candidate_static["support_wrench_ratio"]
            candidate_root_residual = candidate_static["root_equilibrium_residual"]
            candidate_raw_root_residual = candidate_static[
                "raw_root_equilibrium_residual"
            ]
            candidate_torso_pitch = torso_pitch_from_world(candidate_qpos)
            candidate_fat2_target = reset_torso_target_angle(
                candidate_com,
                float(candidate.x[HAND_X_INDEX]),
                candidate_foot_x,
            )
            candidate_fat2_error = abs(candidate_torso_pitch - candidate_fat2_target)
            _, _, candidate_fat2_moment_signed_error = fat2_moment_balance(
                candidate_com,
                float(candidate.x[HAND_X_INDEX]),
                candidate_foot_x,
            )
            candidate_fat2_moment_error = abs(candidate_fat2_moment_signed_error)
            candidate_continuation_joint_delta = float(
                np.max(
                    np.abs(
                        candidate.x[:POLICY_DOF_COUNT]
                        - base_seed[:POLICY_DOF_COUNT]
                    )
                )
            )
            candidate_continuation_arm_delta = float(
                np.max(np.abs(candidate.x[15:POLICY_DOF_COUNT] - base_seed[15:POLICY_DOF_COUNT]))
            )
            candidate_arm_posture_error = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            candidate.x[15:POLICY_DOF_COUNT]
                            - nominal[15:POLICY_DOF_COUNT]
                        )
                    )
                )
            )
            candidate_hand_x_error = abs(
                float(candidate.x[HAND_X_INDEX]) - hand_x_target
            )
            data.qpos[:] = candidate_qpos
            mujoco.mj_forward(model, data)
            candidate_self_collision_count = int(data.ncon)
            candidate_dex_forward = [
                data.xmat[bodies[f"{side}_wrist_yaw_link"]].reshape(3, 3)[:, 0]
                for side in ("left", "right")
            ]
            candidate_minimum_dex_forward_dot = min(
                float(vector[0]) for vector in candidate_dex_forward
            )
            candidate_maximum_dex_forward_lateral = max(
                abs(float(vector[1])) for vector in candidate_dex_forward
            )
            metrics = (
                candidate_hard_error,
                float(candidate.x[ROOT_HEIGHT_INDEX]),
                candidate_lower_ratio,
                candidate_waist_ratio,
                candidate_arm_ratio,
                candidate_q_ref_margin,
                candidate_minimum_dex_forward_dot,
                candidate_maximum_dex_forward_lateral,
                candidate_zmp_margin,
                candidate_friction_ratio,
                candidate_root_residual,
                candidate_raw_root_residual,
                candidate_support_torque_ratio,
                candidate_torso_pitch,
                candidate_fat2_error,
                candidate_fat2_moment_error,
                candidate_continuation_joint_delta,
                candidate_continuation_arm_delta,
                candidate_arm_posture_error,
                candidate_hand_x_error,
            )
            if not all(math.isfinite(value) for value in metrics):
                return None
            violation = _candidate_constraint_violation(
                hard_residual=candidate_hard_error,
                hard_tolerance=args.hard_tolerance,
                lower_torque_ratio=candidate_lower_ratio,
                waist_torque_ratio=candidate_waist_ratio,
                arm_torque_ratio=candidate_arm_ratio,
                q_ref_joint_margin=candidate_q_ref_margin,
                joint_margin=args.joint_margin,
                minimum_q_ref_joint_margin=args.q_ref_joint_margin,
                minimum_dex_forward_dot=candidate_minimum_dex_forward_dot,
                maximum_dex_forward_lateral=candidate_maximum_dex_forward_lateral,
                zmp_margin=candidate_zmp_margin,
                minimum_zmp_margin=args.minimum_zmp_margin,
                friction_ratio=candidate_friction_ratio,
                nominal_friction=args.nominal_friction,
                root_equilibrium_residual=candidate_root_residual,
                root_equilibrium_tolerance=args.root_equilibrium_tolerance,
                fat2_error=candidate_fat2_error,
                fat2_error_tolerance=args.fat2_error_tolerance,
                fat2_moment_error=candidate_fat2_moment_error,
                fat2_moment_tolerance=args.fat2_moment_tolerance,
                torso_pitch=candidate_torso_pitch,
                maximum_torso_pitch=args.maximum_torso_pitch,
                continuation_joint_delta=candidate_continuation_joint_delta,
                maximum_continuation_joint_delta=(
                    args.maximum_continuation_joint_delta
                    if enforce_continuity
                    else math.inf
                ),
                continuation_arm_delta=candidate_continuation_arm_delta,
                maximum_continuation_arm_delta=(
                    args.maximum_continuation_arm_delta
                    if enforce_continuity
                    else math.inf
                ),
                support_torque_ratio=candidate_support_torque_ratio,
                self_collision_count=candidate_self_collision_count,
            )
            pose_mapping = {
                "gradient": float(gradient),
                "q_reset": [
                    float(value) for value in candidate.x[:POLICY_DOF_COUNT]
                ],
                "q_ref_unloaded": [
                    float(value) for value in candidate_q_ref_unloaded
                ],
                "tau_unloaded": [
                    float(value) for value in candidate_tau_unloaded
                ],
                "tau_per_tangent_force": [
                    float(value) for value in candidate_tau_per_tangent_force
                ],
                "tau_per_normal_force": [
                    float(value) for value in candidate_tau_per_normal_force
                ],
                "tau_per_tangent_difference": [
                    float(value) for value in candidate_tau_per_tangent_difference
                ],
                "handle_wrenches_sln": [
                    [float(value) for value in row]
                    for row in handle_wrenches_on_cart
                ],
                "wheel_contact_forces_sln": [
                    [float(value) for value in row]
                    for row in wheel_contact_forces
                ],
                "q_ref": [float(value) for value in candidate_q_ref],
                "root_pitch": float(candidate.x[ROOT_PITCH_INDEX]),
                "root_height": float(candidate.x[ROOT_HEIGHT_INDEX]),
            }
            return (
                candidate,
                (
                    seed_initial_pitch,
                    seed_target_pitch,
                    seed_projected_pitch,
                ),
                {
                        "start": float(start_index + 1),
                        "seed_family": (
                            "elevated_handle"
                            if elevated_family
                            else "unprojected_continuation"
                            if preserve_base_seed
                            else "continuation"
                        ),
                        "torque_objective_profile": torque_profile_name,
                        "root_height": float(candidate.x[ROOT_HEIGHT_INDEX]),
                        "hard_residual": candidate_hard_error,
                        "lower_torque_ratio": candidate_lower_ratio,
                        "waist_torque_ratio": candidate_waist_ratio,
                        "arm_torque_ratio": candidate_arm_ratio,
                        "arm_torque_joint": G1_JOINT_ORDER[candidate_arm_joint],
                        "q_ref_joint_margin": candidate_q_ref_margin,
                        "zmp_margin": candidate_zmp_margin,
                        "friction_ratio": candidate_friction_ratio,
                        "root_equilibrium_residual": candidate_root_residual,
                        "raw_root_equilibrium_residual": candidate_raw_root_residual,
                        "support_torque_ratio": candidate_support_torque_ratio,
                        "fat2_error": candidate_fat2_error,
                        "fat2_moment_error": candidate_fat2_moment_error,
                        "torso_pitch": candidate_torso_pitch,
                        "continuation_joint_delta": candidate_continuation_joint_delta,
                        "continuation_arm_delta": candidate_continuation_arm_delta,
                        "arm_posture_error": candidate_arm_posture_error,
                        "hand_x_error": candidate_hand_x_error,
                        "self_collision_count": candidate_self_collision_count,
                        "minimum_dex_forward_dot_slope_plus_x": (
                            candidate_minimum_dex_forward_dot
                        ),
                        "maximum_abs_dex_forward_dot_slope_y": (
                            candidate_maximum_dex_forward_lateral
                        ),
                        "violation": violation,
                        "cost": float(candidate.cost),
                        "solver_success": bool(candidate.success),
                        "solver_status": int(candidate.status),
                },
                pose_mapping,
            )
 
        worker_count = _solver_worker_count(args.workers, args.full_pose_multistarts)
        candidates.extend(
            candidate
            for candidate in _run_multistarts(
                solve_start, args.full_pose_multistarts, worker_count
            )
            if candidate is not None
        )
        if not candidates:
            raise RuntimeError(f"IK failed for gradient {gradient:+.2f} from all starts")
        feasible_candidates = [
            item for item in candidates if item[2]["violation"] <= 0.0
        ]
        if not feasible_candidates:
            ranked_metrics = sorted(
                (item[2] for item in candidates),
                key=lambda metrics: _candidate_rank_key(metrics, args.root_height),
            )[:3]
            raise RuntimeError(
                f"no statically feasible IK candidate for gradient {gradient:+.2f}; "
                f"best starts={json.dumps(ranked_metrics, sort_keys=True)}"
            )
        candidate_bank[gradient] = [
            {
                "candidate_id": int(item[2]["start"]),
                "pose": item[3],
                "static_metrics": item[2],
            }
            for item in sorted(
                feasible_candidates, key=lambda item: int(item[2]["start"])
            )
        ]
        selected_start = min(
            range(len(candidates)),
            key=lambda index: _candidate_rank_key(candidates[index][2], args.root_height),
        )
        result, fat2_seed_values, selected_metrics, _ = candidates[selected_start]
        (
            fat2_seed_initial_pitch,
            fat2_seed_target_pitch,
            fat2_seed_projected_pitch,
        ) = fat2_seed_values

        qpos = qpos_from_vector(result.x)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        selected_profile_name = selected_metrics["torque_objective_profile"]
        selected_balanced_profile = selected_profile_name == "balanced_arm"
        selected_mild_profile = selected_profile_name == "mild_arm"
        selected_torque_profile = {
            "weight": (
                max(args.arm_torque_weight, BALANCED_ARM_TORQUE_WEIGHT)
                if selected_balanced_profile
                else max(args.arm_torque_weight, MILD_ARM_TORQUE_WEIGHT)
                if selected_mild_profile
                else args.arm_torque_weight
            ),
            "hinge_weight": (
                max(
                    args.arm_torque_hinge_weight,
                    BALANCED_ARM_TORQUE_HINGE_WEIGHT,
                )
                if selected_balanced_profile
                else max(
                    args.arm_torque_hinge_weight,
                    MILD_ARM_TORQUE_HINGE_WEIGHT,
                )
                if selected_mild_profile
                else args.arm_torque_hinge_weight
            ),
            "target_fraction": (
                min(
                    args.ik_arm_torque_target_fraction,
                    BALANCED_ARM_TORQUE_TARGET,
                )
                if selected_balanced_profile
                else min(
                    args.ik_arm_torque_target_fraction,
                    MILD_ARM_TORQUE_TARGET,
                )
                if selected_mild_profile
                else args.ik_arm_torque_target_fraction
            ),
        }
        hard_error = float(
            np.max(np.abs(residual(result.x, selected_torque_profile)[:24]))
            / args.hard_weight
        )
        foot_x = float(result.x[-1])
        static_cases = evaluate_static_cases(qpos, foot_x)
        nominal_static = static_cases["nominal"]
        zmp = nominal_static["zmp"]
        com = nominal_static["com"]
        zmp_margin = static_cases["zmp_margin"]
        friction_ratio = static_cases["friction_ratio"]
        joint_margin = _minimum_joint_limit_margin(
            result.x[:POLICY_DOF_COUNT], joint_lower, joint_upper, np
        )
        required_torque = nominal_static["required"]
        (
            tau_unloaded,
            tau_per_tangent_force,
            tau_per_normal_force,
            tau_per_tangent_difference,
        ) = static_torque_basis(
            qpos,
            state_is_current=True,
        )
        basis_required_torque = (
            tau_unloaded
            + nominal_cart_force_sn[0] * tau_per_tangent_force
            + nominal_cart_force_sn[1] * tau_per_normal_force
            + nominal_cart_tangent_difference * tau_per_tangent_difference
        )
        if not np.allclose(
            basis_required_torque,
            required_torque,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise RuntimeError("selected static torque basis is not affine")
        root_case = max(
            static_cases["cases"],
            key=lambda case: float(np.max(np.abs(case["root_components"]))),
        )
        support_case = max(
            static_cases["cases"], key=lambda case: float(case["support_wrench_ratio"])
        )
        root_equilibrium_components = root_case["root_components"]
        raw_root_equilibrium_components = root_case["raw_root_components"]
        support_torque = nominal_static["support_torque"]
        root_equilibrium_residual = static_cases["root_equilibrium_residual"]
        raw_root_equilibrium_residual = static_cases["raw_root_equilibrium_residual"]
        support_torque_ratio = static_cases["support_wrench_ratio"]
        torque_ratio = static_cases["torque_ratio"]
        lower_torque_ratio = torque_ratio[:12]
        waist_torque_ratio = torque_ratio[12:15]
        arm_torque_ratio = torque_ratio[15:]
        maximum_lower_torque_ratio = float(np.max(lower_torque_ratio))
        maximum_waist_torque_ratio = float(np.max(waist_torque_ratio))
        maximum_arm_torque_ratio = float(np.max(arm_torque_ratio))
        maximum_arm_torque_index = int(np.argmax(arm_torque_ratio)) + 15
        q_reset = result.x[:POLICY_DOF_COUNT].copy()
        q_ref_unloaded = q_reset + tau_unloaded / policy_stiffness
        q_ref = q_reset + basis_required_torque / policy_stiffness
        q_ref_joint_margin = min(
            _minimum_joint_limit_margin(q_ref, joint_lower, joint_upper, np),
            _minimum_joint_limit_margin(
                q_ref_unloaded, joint_lower, joint_upper, np
            ),
        )
        torso_rotation = data.xmat[bodies["torso_link"]].reshape(3, 3)
        dex_forward = {
            side: data.xmat[bodies[f"{side}_wrist_yaw_link"]].reshape(3, 3)[:, 0]
            for side in ("left", "right")
        }
        minimum_dex_forward_dot_slope_x = min(
            float(vector[0]) for vector in dex_forward.values()
        )
        maximum_abs_dex_forward_dot_slope_y = max(
            abs(float(vector[1])) for vector in dex_forward.values()
        )
        dex_forward_pitch = {
            side: math.atan2(float(vector[2]), float(vector[0]))
            for side, vector in dex_forward.items()
        }
        torso_pitch = (
            math.atan2(float(torso_rotation[0, 2]), float(torso_rotation[2, 2])) - gamma
        )
        fat2_angle = fat2_target_angle(com, float(result.x[-2]), foot_x)
        reset_torso_angle = reset_torso_target_angle(
            com, float(result.x[-2]), foot_x
        )
        fat2_error = abs(torso_pitch - reset_torso_angle)
        (
            fat2_required_gravity_moment,
            fat2_gravity_counter_moment,
            fat2_gravity_moment_signed_error,
        ) = fat2_moment_balance(
            com, float(result.x[-2]), foot_x
        )
        fat2_gravity_moment_error = abs(fat2_gravity_moment_signed_error)
        continuation_joint_delta = float(
            np.max(
                np.abs(
                    result.x[:POLICY_DOF_COUNT]
                    - base_seed[:POLICY_DOF_COUNT]
                )
            )
        )
        continuation_arm_delta = float(
            np.max(
                np.abs(
                    result.x[15:POLICY_DOF_COUNT]
                    - base_seed[15:POLICY_DOF_COUNT]
                )
            )
        )
        if not hand_x_min - 1.0e-7 <= float(result.x[-2]) <= hand_x_max + 1.0e-7:
            raise RuntimeError(f"gradient {gradient:+.2f} violates the hand-x safety interval")
        if hard_error > args.hard_tolerance:
            raise RuntimeError(
                f"gradient {gradient:+.2f} IK residual {hard_error:.3e} exceeds "
                f"{args.hard_tolerance:.3e}"
            )
        if minimum_dex_forward_dot_slope_x < DEX_FORWARD_MIN_DOT:
            raise RuntimeError(
                f"gradient {gradient:+.2f} Dex +X exceeds the "
                f"+/-{DEX_FORWARD_MAX_PITCH_DEGREES:g} deg forward "
                f"cone: minimum dot={minimum_dex_forward_dot_slope_x:.8f}"
            )
        if maximum_abs_dex_forward_dot_slope_y > DEX_FORWARD_MAX_LATERAL:
            raise RuntimeError(
                f"gradient {gradient:+.2f} Dex +X has excessive lateral component: "
                f"maximum abs dot={maximum_abs_dex_forward_dot_slope_y:.8f}"
            )
        if joint_margin < args.joint_margin - 1.0e-6:
            raise RuntimeError(f"gradient {gradient:+.2f} violates the joint margin")
        if q_ref_joint_margin < args.q_ref_joint_margin - 1.0e-6:
            raise RuntimeError(
                f"gradient {gradient:+.2f} static-preload target joint margin "
                f"{q_ref_joint_margin:.4f} rad is below {args.q_ref_joint_margin:.4f} rad"
            )
        if root_equilibrium_residual > args.root_equilibrium_tolerance:
            raise RuntimeError(
                f"gradient {gradient:+.2f} root equilibrium residual "
                f"{root_equilibrium_residual:.3e} exceeds "
                f"{args.root_equilibrium_tolerance:.3e}; components="
                f"{np.array2string(root_equilibrium_components, precision=6)}"
            )
        if support_torque_ratio > 1.0:
            raise RuntimeError(
                f"gradient {gradient:+.2f} requires unsupported foot contact torque; "
                f"ratio={support_torque_ratio:.3f}"
            )
        if zmp_margin < args.minimum_zmp_margin:
            raise RuntimeError(
                f"gradient {gradient:+.2f} static ZMP margin {zmp_margin:.4f} m is too small"
            )
        if friction_ratio > args.nominal_friction:
            raise RuntimeError(
                f"gradient {gradient:+.2f} requires friction {friction_ratio:.3f}"
            )
        if fat2_error > args.fat2_error_tolerance:
            raise RuntimeError(
                f"gradient {gradient:+.2f} final FAT2 error {fat2_error:.3f} rad exceeds "
                f"{args.fat2_error_tolerance:.3f} rad"
            )
        if fat2_gravity_moment_error > args.fat2_moment_tolerance:
            raise RuntimeError(
                f"gradient {gradient:+.2f} FAT2 gravity-moment error "
                f"{fat2_gravity_moment_error:.3f} Nm exceeds "
                f"{args.fat2_moment_tolerance:.3f} Nm"
            )
        if abs(torso_pitch) > args.maximum_torso_pitch:
            raise RuntimeError(
                f"gradient {gradient:+.2f} torso pitch {torso_pitch:+.3f} rad exceeds "
                f"+/-{args.maximum_torso_pitch:.3f} rad"
            )
        if (
            enforce_continuity
            and continuation_joint_delta > args.maximum_continuation_joint_delta
        ):
            raise RuntimeError(
                f"gradient {gradient:+.2f} changes a joint by "
                f"{continuation_joint_delta:.3f} rad from its continuation seed"
            )
        if (
            enforce_continuity
            and continuation_arm_delta > args.maximum_continuation_arm_delta
        ):
            raise RuntimeError(
                f"gradient {gradient:+.2f} changes an arm joint by "
                f"{continuation_arm_delta:.3f} rad from its continuation seed"
            )
        if maximum_arm_torque_ratio > RESET_TORQUE_LIMIT_FRACTION:
            raise RuntimeError(
                f"gradient {gradient:+.2f} arm torque reaches "
                f"{maximum_arm_torque_ratio:.3f} of its hardware limit"
            )
        if maximum_lower_torque_ratio > RESET_TORQUE_LIMIT_FRACTION:
            raise RuntimeError(
                f"gradient {gradient:+.2f} lower-body torque reaches "
                f"{maximum_lower_torque_ratio:.3f} of its hardware limit"
            )
        if maximum_waist_torque_ratio > RESET_TORQUE_LIMIT_FRACTION:
            raise RuntimeError(
                f"gradient {gradient:+.2f} waist torque reaches "
                f"{maximum_waist_torque_ratio:.3f} of its hardware limit"
            )
        if data.ncon:
            contact_pairs = []
            for contact in data.contact[: data.ncon]:
                body_names = []
                for geom_id in (int(contact.geom1), int(contact.geom2)):
                    body_id = int(model.geom_bodyid[geom_id])
                    body_names.append(
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        or f"body_{body_id}"
                    )
                contact_pairs.append("/".join(body_names))
            raise RuntimeError(
                f"gradient {gradient:+.2f} has {data.ncon} self-collision contact(s): "
                f"{sorted(set(contact_pairs))}"
            )

        diagnostics.append(
            {
                "gradient": gradient,
                "root_pitch": float(result.x[ROOT_PITCH_INDEX]),
                "root_height": float(result.x[ROOT_HEIGHT_INDEX]),
                "hard_residual": hard_error,
                "minimum_dex_forward_dot_slope_plus_x": (
                    minimum_dex_forward_dot_slope_x
                ),
                "maximum_abs_dex_forward_dot_slope_y": (
                    maximum_abs_dex_forward_dot_slope_y
                ),
                "dex_forward_pitch_rad": dex_forward_pitch,
                "joint_margin": joint_margin,
                "q_ref_joint_margin": q_ref_joint_margin,
                "q_ref_joint_margin_required": args.q_ref_joint_margin,
                "root_equilibrium_residual": root_equilibrium_residual,
                "cart_equilibrium_force_residual_sln": [
                    float(value) for value in cart_force_residual
                ],
                "cart_equilibrium_moment_residual_sln": [
                    float(value) for value in cart_moment_residual
                ],
                "handle_wrenches_sln": [
                    [float(value) for value in row]
                    for row in handle_wrenches_on_cart
                ],
                "wheel_contact_forces_sln": [
                    [float(value) for value in row]
                    for row in wheel_contact_forces
                ],
                "minimum_wheel_normal_force": float(
                    np.min(wheel_contact_forces[:, 2])
                ),
                "maximum_handle_force": float(np.max(handle_force_norms)),
                "maximum_handle_torque": float(np.max(handle_torque_norms)),
                "root_equilibrium_worst_load_scale": float(root_case["scale"]),
                "root_equilibrium_components": [
                    float(value) for value in root_equilibrium_components
                ],
                "raw_root_equilibrium_residual": raw_root_equilibrium_residual,
                "raw_root_equilibrium_components": [
                    float(value) for value in raw_root_equilibrium_components
                ],
                "support_contact_torque": [float(value) for value in support_torque],
                "support_contact_torque_ratio": support_torque_ratio,
                "support_wrench_worst_load_scale": float(support_case["scale"]),
                "support_wrench_components_per_foot": [
                    [float(value) for value in row]
                    for row in support_case["support_wrench_components"]
                ],
                "maximum_preload_offset": float(np.max(np.abs(q_ref - q_reset))),
                "zmp_margin": float(zmp_margin),
                "zmp_s": float(zmp[0]),
                "zmp_y": float(zmp[1]),
                "static_load_uncertainty": args.static_load_uncertainty,
                "static_load_scales": [float(value) for value in load_scales],
                "static_load_cases": [
                    {
                        "scale": float(case["scale"]),
                        "zmp_s": float(case["zmp"][0]),
                        "zmp_y": float(case["zmp"][1]),
                        "zmp_margin": float(case["zmp_margin"]),
                        "friction_ratio": float(case["friction_ratio"]),
                        "support_wrench_ratio": float(case["support_wrench_ratio"]),
                        "lower_torque_ratio": float(
                            np.max(np.abs(case["required"][:12]) / policy_effort_limits[:12])
                        ),
                        "waist_torque_ratio": float(
                            np.max(
                                np.abs(case["required"][12:15])
                                / policy_effort_limits[12:15]
                            )
                        ),
                        "arm_torque_ratio": float(
                            np.max(
                                np.abs(case["required"][15:])
                                / policy_effort_limits[15:]
                            )
                        ),
                    }
                    for case in static_cases["cases"]
                ],
                "friction_ratio": friction_ratio,
                "lower_torque_ratio": maximum_lower_torque_ratio,
                "waist_torque_ratio": maximum_waist_torque_ratio,
                "arm_torque_ratio": maximum_arm_torque_ratio,
                "maximum_arm_torque_joint": G1_JOINT_ORDER[maximum_arm_torque_index],
                "required_joint_torque_nm": {
                    name: float(value)
                    for name, value in zip(G1_JOINT_ORDER, required_torque, strict=True)
                },
                "joint_torque_ratio": {
                    name: float(value)
                    for name, value in zip(G1_JOINT_ORDER, torque_ratio, strict=True)
                },
                "torso_pitch": torso_pitch,
                "fat2_target_pitch": fat2_angle,
                "uphill_lean_margin_rad": args.uphill_lean_margin,
                "reset_torso_target_pitch": reset_torso_angle,
                "fat2_error": fat2_error,
                "fat2_required_gravity_moment_nm": fat2_required_gravity_moment,
                "fat2_gravity_counter_moment_nm": fat2_gravity_counter_moment,
                "fat2_gravity_moment_error_nm": (
                    fat2_gravity_moment_signed_error
                ),
                "fat2_gravity_moment_error_abs_nm": fat2_gravity_moment_error,
                "fat2_gravity_moment_tolerance_nm": args.fat2_moment_tolerance,
                "maximum_torso_pitch": args.maximum_torso_pitch,
                "continuation_joint_delta": continuation_joint_delta,
                "continuation_arm_delta": continuation_arm_delta,
                "continuation_gate_enabled": enforce_continuity,
                "fat2_seed_initial_torso_pitch": fat2_seed_initial_pitch,
                "fat2_seed_target_pitch": fat2_seed_target_pitch,
                "fat2_seed_projected_torso_pitch": fat2_seed_projected_pitch,
                "fat2_seed_projection_error": (
                    fat2_seed_projected_pitch - fat2_seed_target_pitch
                ),
                "full_pose_multistarts": len(candidates),
                "solver_workers": worker_count,
                "selected_multistart": int(selected_metrics["start"]),
                "multistart_diagnostics": [item[2] for item in candidates],
                "hand_x": float(result.x[-2]),
                "hand_x_bounds": [float(hand_x_min), float(hand_x_max)],
                "foot_x": foot_x,
                "foot_x_target": float(foot_x_target),
                "com_radius": float(
                    math.hypot(com[0] - (foot_x + args.foot_center_offset_x), com[2])
                ),
                "evaluations": float(result.nfev),
            }
        )
        solved[gradient] = ResetPose(
            gradient=gradient,
            q_reset=tuple(float(value) for value in q_reset),
            q_ref_unloaded=tuple(float(value) for value in q_ref_unloaded),
            tau_unloaded=tuple(float(value) for value in tau_unloaded),
            tau_per_tangent_force=tuple(
                float(value) for value in tau_per_tangent_force
            ),
            tau_per_normal_force=tuple(
                float(value) for value in tau_per_normal_force
            ),
            tau_per_tangent_difference=tuple(
                float(value) for value in tau_per_tangent_difference
            ),
            handle_wrenches_sln=tuple(
                tuple(float(value) for value in row)
                for row in handle_wrenches_on_cart
            ),
            wheel_contact_forces_sln=tuple(
                tuple(float(value) for value in row)
                for row in wheel_contact_forces
            ),
            q_ref=tuple(float(value) for value in q_ref),
            root_pitch=float(result.x[ROOT_PITCH_INDEX]),
            root_height=float(result.x[ROOT_HEIGHT_INDEX]),
        )
        _write_candidate_progress(args, diagnostics, candidate_bank)
        print(
            f"solved slope {gradient:+.2f}: root_height={result.x[ROOT_HEIGHT_INDEX]:.3f} "
            f"root_pitch={result.x[ROOT_PITCH_INDEX]:+.3f} "
            f"arm_ratio={maximum_arm_torque_ratio:.3f}",
            flush=True,
        )
        return result.x

    diagnostics_by_slope = {float(row["gradient"]): row for row in diagnostics}
    solved_vectors: dict[float, Any] = {}

    def cached_vector(gradient: float) -> Any:
        pose = solved[gradient]
        diagnostic = diagnostics_by_slope[gradient]
        return np.concatenate(
            (
                np.asarray(pose.q_reset, dtype=np.float64),
                np.asarray(
                    (
                        pose.root_pitch,
                        pose.root_height,
                        diagnostic["hand_x"],
                        diagnostic["foot_x"],
                    ),
                    dtype=np.float64,
                ),
            )
        )

    for gradient, parent in _stage_a_solve_plan():
        if parent is not None and parent not in solved_vectors:
            raise RuntimeError(
                f"candidate progress for {gradient:+.2f} is missing its "
                f"continuation parent {parent:+.2f}"
            )
        if gradient in solved:
            solved_vectors[gradient] = cached_vector(gradient)
            print(f"reused Stage A slope {gradient:+.2f}", flush=True)
            continue
        seed = flat_seed if parent is None else solved_vectors[parent]
        solved_vectors[gradient] = solve_gradient(
            gradient,
            seed,
            enforce_continuity=parent is not None,
        )

    library = ResetPoseLibrary(poses=tuple(solved[gradient] for gradient in SLOPE_GRADIENTS))
    return (
        library,
        sorted(diagnostics, key=lambda item: item["gradient"]),
        candidate_bank,
    )


def _prepare_atomic_text(path: Path, content: str) -> Path:
    """Write a durable same-directory temporary file for a later atomic replace."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _prepare_atomic_copy(source: Path, destination: Path) -> Path:
    """Copy a staged artifact beside its destination without exposing it yet."""

    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with Path(source).open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_yaml(path: Path, mapping: dict[str, Any]) -> None:
    temporary = _prepare_atomic_text(
        Path(path), json.dumps(mapping, indent=2) + "\n"
    )
    try:
        os.replace(temporary, Path(path).resolve())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = _prepare_atomic_text(Path(path), content)
    try:
        os.replace(temporary, Path(path).resolve())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_atomic_copy(source: Path, destination: Path) -> None:
    temporary = _prepare_atomic_copy(source, destination)
    try:
        os.replace(temporary, destination.resolve())
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _commit_pipeline_publications(
    publications: list[tuple[Path, Path]],
    prepared_library: Path,
    output_path: Path,
) -> None:
    """Commit evidence first and roll it back if the final library commit fails."""

    backups: dict[Path, Path | None] = {}
    published: list[Path] = []

    def cleanup(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        for _temporary, destination in publications:
            destination = destination.resolve()
            if destination.exists() and not destination.is_file():
                raise RuntimeError(
                    f"reset pipeline publication target is not a file: {destination}"
                )
            backups[destination] = (
                _prepare_atomic_copy(destination, destination)
                if destination.is_file()
                else None
            )
    except BaseException:
        cleanup([temporary for temporary, _destination in publications])
        cleanup([backup for backup in backups.values() if backup is not None])
        cleanup([prepared_library])
        raise

    try:
        for temporary, destination in publications:
            destination = destination.resolve()
            os.replace(temporary, destination)
            published.append(destination)
        os.replace(prepared_library, output_path.resolve())
    except BaseException as exc:
        rollback_failures: list[str] = []
        retained_backups: set[Path] = set()
        for destination in reversed(published):
            backup = backups[destination]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except OSError as rollback_exc:
                detail = f"{destination}: {rollback_exc}"
                if backup is not None:
                    retained_backups.add(backup)
                    detail += f"; backup preserved at {backup}"
                rollback_failures.append(detail)
        cleanup([temporary for temporary, _destination in publications])
        cleanup(
            [
                backup
                for backup in backups.values()
                if backup is not None and backup not in retained_backups
            ]
        )
        cleanup([prepared_library])
        if rollback_failures:
            raise RuntimeError(
                "reset publication failed and evidence rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from exc
        raise

    cleanup([backup for backup in backups.values() if backup is not None])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("config/reset_poses.yaml")
    )
    parser.add_argument(
        "--seed-library",
        help="Optional JSON-compatible reset-pose library used only as an IK warm start.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument(
        "--root-height",
        type=float,
        default=0.75,
        help="Nominal root height used as the full-pose optimization seed and soft target.",
    )
    parser.add_argument("--root-height-min", type=float, default=0.65)
    parser.add_argument("--root-height-max", type=float, default=0.82)
    parser.add_argument("--root-pitch-min", type=float, default=-0.35)
    parser.add_argument("--root-pitch-max", type=float, default=0.35)
    parser.add_argument(
        "--root-pitch-weight",
        type=float,
        default=0.05,
        help="Soft natural-pelvis prior; zero leaves root pitch entirely to other terms.",
    )
    parser.add_argument(
        "--root-height-weight",
        type=float,
        default=0.5,
        help="Soft regularization toward --root-height, normalized by 5 cm.",
    )
    parser.add_argument(
        "--root-height-by-slope",
        type=float,
        nargs=len(SLOPE_GRADIENTS),
        help="Optional root heights in exact -0.08..+0.10 slope order.",
    )
    parser.add_argument("--leg-stiffness", type=float, default=300.0)
    parser.add_argument("--foot-stiffness", type=float, default=200.0)
    parser.add_argument("--waist-stiffness", type=float, default=5000.0)
    parser.add_argument("--arm-stiffness", type=float, default=3000.0)
    parser.add_argument("--root-equilibrium-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--cart-equilibrium-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--minimum-wheel-normal-force", type=float, default=25.0)
    parser.add_argument("--d6-force-limit", type=float, default=300.0)
    parser.add_argument("--d6-torque-limit", type=float, default=35.0)
    parser.add_argument("--hitch-height", type=float, default=0.85)
    parser.add_argument("--hitch-half-width", type=float, default=HITCH_HALF_WIDTH)
    parser.add_argument(
        "--hand-x-target",
        type=float,
        default=0.19,
        help="Soft robot-relative grasp-position target at zero slope.",
    )
    parser.add_argument(
        "--hand-x-slope-span",
        type=float,
        default=0.07,
        help=(
            "Signed endpoint shift about --hand-x-target, normalized by the largest "
            "configured absolute gradient; +0.10 uses center-span."
        ),
    )
    parser.add_argument("--hand-x-min", type=float, default=DEFAULT_HAND_X_MIN)
    parser.add_argument("--hand-x-max", type=float, default=DEFAULT_HAND_X_MAX)
    parser.add_argument(
        "--hand-x-weight",
        type=float,
        default=1.0,
        help="Scaled pulling-branch prior on (hand_x - target) / 0.05 m.",
    )
    parser.add_argument("--foot-x-weight", type=float, default=0.1)
    parser.add_argument("--sole-body-height", type=float, default=0.035)
    parser.add_argument("--foot-min-x", type=float, default=-0.05)
    parser.add_argument("--foot-max-x", type=float, default=0.12)
    parser.add_argument("--foot-center-offset-x", type=float, default=0.035)
    parser.add_argument(
        "--foot-x-by-slope",
        type=float,
        nargs=len(SLOPE_GRADIENTS),
        help="Optional ankle-roll X targets in exact -0.08..+0.10 slope order.",
    )
    parser.add_argument("--foot-x-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--root-pitch-by-slope",
        type=float,
        nargs=len(SLOPE_GRADIENTS),
        help="Optional root pitches relative to the slope frame in exact -0.08..+0.10 order.",
    )
    parser.add_argument("--wrist-to-dex-base-x", type=float, default=0.0415)
    parser.add_argument("--grasp-center-x", type=float, default=0.11066269)
    parser.add_argument(
        "--grasp-frame-roll",
        type=float,
        default=math.pi / 2.0,
        help="Calibrated magnitude of the left/right Dex grasp-frame local X rotation.",
    )
    parser.add_argument(
        "--dex-forward-weight",
        type=float,
        default=10.0,
        help="Soft branch-selection weight that points Dex local +X along path +X.",
    )
    parser.add_argument(
        "--dex-forward-target-pitch",
        type=float,
        default=math.radians(50.0),
        help="Preferred Dex local +X pitch about crossbar +Y inside the accepted cone.",
    )
    parser.add_argument("--dex-q-grasp", type=float, default=-0.01609)
    parser.add_argument("--joint-margin", type=float, default=0.06)
    parser.add_argument(
        "--q-ref-joint-margin",
        type=float,
        default=DEFAULT_Q_REF_JOINT_MARGIN,
        help="Minimum hard-limit margin retained by the nominal static PD target.",
    )
    parser.add_argument(
        "--q-ref-margin-weight",
        type=float,
        default=50.0,
        help="Hinge penalty that steers q_ref away from the minimum joint margin.",
    )
    parser.add_argument("--minimum-zmp-margin", type=float, default=0.02)
    parser.add_argument("--nominal-friction", type=float, default=0.60)
    parser.add_argument(
        "--static-load-uncertainty",
        type=float,
        default=0.0,
        help=(
            "Symmetric hand-load perturbation used by robust ZMP, contact-wrench, "
            "and hardware-torque gates. The calibrated reset load is exact by default."
        ),
    )
    parser.add_argument("--hard-weight", type=float, default=1.0e4)
    parser.add_argument("--hard-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--zmp-center-weight", type=float, default=10.0)
    parser.add_argument(
        "--zmp-margin-weight",
        type=float,
        default=100.0,
        help=(
            "Hinge weight that enforces --minimum-zmp-margin for every static "
            "load case, including the unloaded reset endpoint."
        ),
    )
    parser.add_argument(
        "--zmp-optimization-reserve-fraction",
        type=float,
        default=DEFAULT_ZMP_OPTIMIZATION_RESERVE_FRACTION,
        help=(
            "Relative reserve above --minimum-zmp-margin used only by the "
            "least-squares objective; the hard acceptance threshold is unchanged."
        ),
    )
    parser.add_argument("--world-upright-weight", type=float, default=0.0)
    parser.add_argument("--fat2-torso-weight", type=float, default=20.0)
    parser.add_argument(
        "--fat2-moment-weight",
        type=float,
        default=5.0,
        help="Weight on gravity-moment cancellation of the hitch wrench.",
    )
    parser.add_argument(
        "--fat2-moment-tolerance",
        type=float,
        default=1.0,
        help="Maximum gravity-versus-hitch pitch-moment mismatch in Nm.",
    )
    parser.add_argument("--fat2-theta-max", type=float, default=0.5)
    parser.add_argument(
        "--fat2-error-tolerance",
        type=float,
        default=DEFAULT_FAT2_ERROR_TOLERANCE,
        help="Maximum final torso error from the reset-specific FAT2 target.",
    )
    parser.add_argument(
        "--maximum-torso-pitch",
        type=float,
        default=DEFAULT_MAXIMUM_TORSO_PITCH,
        help="Maximum absolute torso pitch relative to world vertical.",
    )
    parser.add_argument(
        "--uphill-lean-margin",
        type=float,
        default=0.12,
        help=(
            "Offline reset-only torso margin toward the uphill direction. The "
            "analytic FAT2 angle remains separately reported and unchanged."
        ),
    )
    parser.add_argument("--fat2-seed-iterations", type=int, default=8)
    parser.add_argument("--fat2-seed-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--fat2-seed-root-fraction",
        type=float,
        default=0.25,
        help="Fraction of FAT2 seed correction assigned to root pitch; waist pitch gets the rest.",
    )
    parser.add_argument("--lower-torque-weight", type=float, default=5.0)
    parser.add_argument("--lower-torque-hinge-weight", type=float, default=100.0)
    parser.add_argument(
        "--ik-lower-torque-target-fraction", type=float, default=DEFAULT_TORQUE_TARGET
    )
    parser.add_argument("--arm-torque-weight", type=float, default=5.0)
    parser.add_argument("--arm-torque-hinge-weight", type=float, default=100.0)
    parser.add_argument(
        "--ik-arm-torque-target-fraction", type=float, default=DEFAULT_TORQUE_TARGET
    )
    parser.add_argument("--minimum-elbow-flexion", type=float, default=0.0)
    parser.add_argument("--maximum-shoulder-pitch", type=float, default=1.0)
    parser.add_argument("--maximum-wrist-roll", type=float, default=1.5)
    parser.add_argument(
        "--maximum-continuation-joint-delta",
        type=float,
        default=DEFAULT_MAXIMUM_CONTINUATION_JOINT_DELTA,
    )
    parser.add_argument(
        "--maximum-continuation-arm-delta",
        type=float,
        default=DEFAULT_MAXIMUM_CONTINUATION_ARM_DELTA,
    )
    parser.add_argument("--minimum-collision-distance", type=float, default=1.0e-5)
    parser.add_argument("--collision-detection-distance", type=float, default=0.05)
    parser.add_argument("--collision-weight", type=float, default=1.0e5)
    parser.add_argument(
        "--full-pose-multistarts", type=int, default=RESET_STATIC_MULTISTARTS
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel fork workers for independent starts; 0 uses up to "
            f"{DEFAULT_SOLVER_WORKERS} cores."
        ),
    )
    parser.add_argument(
        "--arm-seed-noise-scale",
        type=float,
        default=1.0,
        help="Deterministic mirrored arm perturbation scale for full-pose multistarts.",
    )
    parser.add_argument("--max-evaluations", type=int, default=50000)
    parser.add_argument(
        "--solver-tolerance", type=float, default=DEFAULT_SOLVER_TOLERANCE
    )

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("outputs/reset_pose_candidates.json"),
    )
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help=(
            "Reuse completed slopes from --candidate-output and solve only missing "
            "slopes; a complete cache skips Stage A."
        ),
    )
    parser.add_argument(
        "--stage-b-candidates-per-slope",
        type=int,
        default=DEFAULT_STAGE_B_CANDIDATES_PER_SLOPE,
        help=(
            "Candidates for each slope packed into one replicated-physics Stage B "
            f"process; maximum {TERRAIN_COLUMNS_PER_TYPE} matches the available "
            "non-overlapping terrain columns."
        ),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help=(
            "Diagnostic mode: write the statically selected configured-pose library and "
            "candidate bank without launching PhysX. The output is not certified."
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/reset_pose_search_report.json"),
    )
    parser.add_argument(
        "--alignment-output",
        type=Path,
        default=Path("outputs/validation/reset_alignment_1000.json"),
        help="Canonical assembled-library validation report written only on success.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reset_pose_search_report.md"),
    )
    parser.add_argument(
        "--validate-existing",
        type=Path,
        metavar="RESET_POSES",
        help=(
            f"Skip search and validate an existing {len(SLOPE_GRADIENTS)}-slope "
            "library with the same hard gates."
        ),
    )
    parser.add_argument(
        "--_pipeline-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--timeseries-stride", type=int, default=0)
    parser.add_argument("--foot-damping", type=float)
    parser.add_argument("--leg-damping", type=float)
    parser.add_argument("--stable-displacement-limit", type=float, default=0.05)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.validate_existing is None:
        path_roles = {
            "output": Path(args.output).resolve(),
            "candidate-output": args.candidate_output.resolve(),
            "report-output": args.report_output.resolve(),
            "alignment-output": args.alignment_output.resolve(),
            "summary-output": args.summary_output.resolve(),
        }
    else:
        path_roles = {
            "validate-existing": args.validate_existing.resolve(),
            "report-output": args.report_output.resolve(),
            "alignment-output": args.alignment_output.resolve(),
        }
    by_path: dict[Path, list[str]] = {}
    for role, path in path_roles.items():
        by_path.setdefault(path, []).append(role)
    collisions = [roles for roles in by_path.values() if len(roles) > 1]
    if collisions:
        raise ValueError(
            "reset pipeline input/output paths must be distinct: "
            + "; ".join(", ".join(roles) for roles in collisions)
        )

    if not args.root_pitch_min < args.root_pitch_max:
        raise ValueError("--root-pitch-min must be smaller than --root-pitch-max")
    if not args.root_height_min < args.root_height_max:
        raise ValueError("--root-height-min must be smaller than --root-height-max")
    if not args.root_height_min <= args.root_height <= args.root_height_max:
        raise ValueError("--root-height must lie inside the root-height bounds")
    if args.root_pitch_weight < 0.0 or args.root_height_weight < 0.0:
        raise ValueError("root-pose prior weights must be non-negative")
    if args.hand_x_weight <= 0.0 or args.foot_x_weight < 0.0:
        raise ValueError("hand-x weight must be positive and foot-x weight non-negative")
    if not math.isfinite(args.zmp_margin_weight) or args.zmp_margin_weight <= 0.0:
        raise ValueError("--zmp-margin-weight must be finite and positive")
    if (
        not math.isfinite(args.zmp_optimization_reserve_fraction)
        or not 0.0 <= args.zmp_optimization_reserve_fraction < 1.0
    ):
        raise ValueError("--zmp-optimization-reserve-fraction must lie in [0, 1)")
    if not math.isfinite(args.hand_x_slope_span) or args.hand_x_slope_span < 0.0:
        raise ValueError("--hand-x-slope-span must be finite and non-negative")
    if args.dex_forward_weight <= 0.0:
        raise ValueError("--dex-forward-weight must be positive")
    if abs(args.dex_forward_target_pitch) > DEX_FORWARD_MAX_PITCH_RAD:
        raise ValueError("--dex-forward-target-pitch must lie inside the accepted cone")
    if args.fat2_torso_weight <= 0.0 or args.fat2_moment_weight <= 0.0:
        raise ValueError("FAT2 torso and moment weights must be positive")
    if not 0.0 <= args.uphill_lean_margin < args.fat2_theta_max:
        raise ValueError("--uphill-lean-margin must lie in [0, --fat2-theta-max)")
    if args.fat2_seed_iterations <= 0 or args.fat2_seed_tolerance <= 0.0:
        raise ValueError("FAT2 seed projection iterations and tolerance must be positive")
    if not 0.0 <= args.fat2_seed_root_fraction <= 1.0:
        raise ValueError("--fat2-seed-root-fraction must lie in [0, 1]")
    if args.full_pose_multistarts <= 0 or args.max_evaluations <= 0:
        raise ValueError("multistarts and maximum evaluations must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if not math.isfinite(args.solver_tolerance) or not 0.0 < args.solver_tolerance < 1.0:
        raise ValueError("--solver-tolerance must lie in (0, 1)")
    if args.arm_seed_noise_scale < 0.0 or not math.isfinite(args.arm_seed_noise_scale):
        raise ValueError("--arm-seed-noise-scale must be finite and non-negative")
    for name in (
        "lower_torque_weight",
        "lower_torque_hinge_weight",
        "arm_torque_weight",
        "arm_torque_hinge_weight",
        "q_ref_margin_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in (
        "leg_stiffness",
        "foot_stiffness",
        "waist_stiffness",
        "arm_stiffness",
        "root_equilibrium_tolerance",
        "cart_equilibrium_tolerance",
        "minimum_wheel_normal_force",
        "d6_force_limit",
        "d6_torque_limit",
        "hard_weight",
        "hard_tolerance",
        "joint_margin",
        "q_ref_joint_margin",
        "minimum_zmp_margin",
        "nominal_friction",
        "foot_x_tolerance",
        "sole_body_height",
        "fat2_error_tolerance",
        "fat2_moment_tolerance",
        "maximum_torso_pitch",
        "maximum_shoulder_pitch",
        "maximum_wrist_roll",
        "maximum_continuation_joint_delta",
        "maximum_continuation_arm_delta",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.q_ref_joint_margin > args.joint_margin:
        raise ValueError("--q-ref-joint-margin must not exceed --joint-margin")
    for name in (
        "zmp_center_weight",
        "world_upright_weight",
        "collision_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if (
        not math.isfinite(args.minimum_collision_distance)
        or args.minimum_collision_distance < 0.0
    ):
        raise ValueError("--minimum-collision-distance must be finite and non-negative")
    if (
        not math.isfinite(args.collision_detection_distance)
        or args.collision_detection_distance < args.minimum_collision_distance
    ):
        raise ValueError(
            "--collision-detection-distance must be finite and at least the minimum distance"
        )
    if not 0.0 < args.fat2_theta_max < math.pi / 2.0:
        raise ValueError("--fat2-theta-max must lie in (0, pi/2)")
    if args.maximum_torso_pitch > args.fat2_theta_max:
        raise ValueError("--maximum-torso-pitch must not exceed --fat2-theta-max")
    if (
        not math.isfinite(args.static_load_uncertainty)
        or not 0.0 <= args.static_load_uncertainty < 0.5
    ):
        raise ValueError("--static-load-uncertainty must lie in [0, 0.5)")
    if not math.isfinite(args.minimum_elbow_flexion):
        raise ValueError("--minimum-elbow-flexion must be finite")
    if not args.foot_min_x < args.foot_max_x:
        raise ValueError("--foot-min-x must be smaller than --foot-max-x")
    if not args.foot_min_x <= args.foot_center_offset_x <= args.foot_max_x:
        raise ValueError("--foot-center-offset-x must lie inside the support interval")
    if not (
        0.0
        <= args.ik_lower_torque_target_fraction
        <= RESET_TORQUE_LIMIT_FRACTION
    ):
        raise ValueError(
            "--ik-lower-torque-target-fraction must not exceed the lower or waist hard limit"
        )
    if not (
        0.0 <= args.ik_arm_torque_target_fraction <= RESET_TORQUE_LIMIT_FRACTION
    ):
        raise ValueError(
            "--ik-arm-torque-target-fraction must not exceed the arm hard limit"
        )
    fixed_slope_values = {
        "foot-x-by-slope": args.foot_x_by_slope,
        "root-pitch-by-slope": args.root_pitch_by_slope,
        "root-height-by-slope": args.root_height_by_slope,
    }
    for label, values in fixed_slope_values.items():
        if values is not None and any(not math.isfinite(value) for value in values):
            raise ValueError(f"--{label} values must all be finite")
    if args.root_pitch_by_slope is not None and any(
        not args.root_pitch_min <= value <= args.root_pitch_max
        for value in args.root_pitch_by_slope
    ):
        raise ValueError("--root-pitch-by-slope values must lie inside the root-pitch bounds")
    if args.root_height_by_slope is not None and any(
        not args.root_height_min <= value <= args.root_height_max
        for value in args.root_height_by_slope
    ):
        raise ValueError("--root-height-by-slope values must lie inside the root-height bounds")

    if args.validate_existing is not None:
        if not args.validate_existing.is_file():
            raise FileNotFoundError(
                f"--validate-existing does not exist: {args.validate_existing}"
            )
        if args.steps < 1000:
            raise ValueError("formal existing-library validation requires --steps >= 1000")
        return

    if not 1 <= args.stage_b_candidates_per_slope <= TERRAIN_COLUMNS_PER_TYPE:
        raise ValueError(
            "--stage-b-candidates-per-slope must lie in "
            f"[1, {TERRAIN_COLUMNS_PER_TYPE}]"
        )
    if args.full_pose_multistarts != RESET_STATIC_MULTISTARTS:
        raise ValueError(
            "the complete search requires exactly "
            f"{RESET_STATIC_MULTISTARTS} least-squares starts per slope"
        )
    if not 1000 <= args.steps <= 2000:
        raise ValueError("the complete reset pipeline requires --steps in [1000, 2000]")
    if args.timeseries_stride < 0:
        raise ValueError("--timeseries-stride must be non-negative")
    if args.stable_displacement_limit <= 0.0:
        raise ValueError("--stable-displacement-limit must be positive")
DEX_FORWARD_INITIAL_MAX_LATERAL = 0.01
DEX_FORWARD_ROLLOUT_MAX_LATERAL = 0.03

def _terrain_indices(slopes: tuple[float, ...]) -> tuple[list[int], list[int]]:
    indices = tuple(terrain_index_for_gradient(slope) for slope in slopes)
    return [level for level, _ in indices], [terrain_type for _, terrain_type in indices]


def _set_signed_slope_origins(env, slopes: tuple[float, ...]) -> None:
    levels, columns = _terrain_indices(slopes)
    _set_terrain_origins(env, levels, columns)


def _set_terrain_origins(
    env: Any, levels: list[int], columns: list[int]
) -> None:
    if len(levels) != env.num_envs or len(columns) != env.num_envs:
        raise ValueError("terrain index rows must match the environment count")
    terrain = env.scene.terrain
    level_tensor = torch.tensor(levels, device=env.device, dtype=torch.long)
    column_tensor = torch.tensor(columns, device=env.device, dtype=torch.long)
    terrain.terrain_levels.copy_(level_tensor)
    terrain.terrain_types.copy_(column_tensor)
    terrain.env_origins.copy_(terrain.terrain_origins[level_tensor, column_tensor])


def _grasp_positions(env) -> torch.Tensor:
    robot = env.scene["robot"]
    positions = robot.data.body_pos_w[:, env.grasp_body_ids]
    quaternions = robot.data.body_quat_w[:, env.grasp_body_ids]
    local = torch.tensor(
        env.d6_constraint_manager.cfg.grasp_local_positions,
        device=env.device,
        dtype=positions.dtype,
    ).view(1, 2, 3)
    return positions + quat_apply_wxyz(quaternions, local)


def _orientation_metrics(env) -> dict[str, torch.Tensor]:
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    unit_x = torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(env.num_envs, -1)
    robot_forward = quat_apply_wxyz(robot.data.root_quat_w, unit_x)
    cart_forward = quat_apply_wxyz(cart.data.root_quat_w, unit_x)
    tangent = env.path_tangent_w
    lateral = env.path_lateral_w
    normal = env.path_normal_w
    robot_projected = robot_forward - torch.sum(
        robot_forward * normal, dim=-1, keepdim=True
    ) * normal
    robot_projected = torch.nn.functional.normalize(robot_projected, dim=-1)
    cart_projected = cart_forward - torch.sum(cart_forward * normal, dim=-1, keepdim=True) * normal
    cart_projected = torch.nn.functional.normalize(cart_projected, dim=-1)
    robot_from_cart = robot.data.root_pos_w - cart.data.root_pos_w
    dex_forward = quat_apply_wxyz(
        robot.data.body_quat_w[:, env.grasp_body_ids],
        unit_x[:, None, :].expand(-1, 2, -1),
    )
    return {
        "robot_forward_dot_path_tangent": torch.sum(robot_projected * tangent, dim=-1),
        "robot_forward_dot_path_lateral": torch.sum(robot_projected * lateral, dim=-1),
        "cart_forward_dot_path_tangent": torch.sum(cart_forward * tangent, dim=-1),
        "cart_projected_heading_dot_path_tangent": torch.sum(cart_projected * tangent, dim=-1),
        "cart_forward_dot_path_lateral": torch.sum(cart_forward * lateral, dim=-1),
        "robot_ahead_of_cart_m": torch.sum(robot_from_cart * tangent, dim=-1),
        "dex_forward_dot_path_tangent": torch.sum(
            dex_forward * tangent[:, None, :], dim=-1
        ),
        "dex_forward_dot_path_lateral": torch.sum(
            dex_forward * lateral[:, None, :], dim=-1
        ),
        "dex_forward_dot_path_normal": torch.sum(
            dex_forward * normal[:, None, :], dim=-1
        ),
    }


def _initial_alignment(env) -> dict[str, torch.Tensor]:
    cart = env.scene["rickshaw"]
    origins = env.scene.terrain.env_origins
    grasp = _grasp_positions(env)
    hitch = cart.data.body_pos_w[:, env.hitch_body_ids]
    cart_conjugate = torch.cat((cart.data.root_quat_w[:, :1], -cart.data.root_quat_w[:, 1:]), dim=-1)
    hitch_local = quat_apply_wxyz(
        cart_conjugate[:, None, :].expand(-1, 2, -1),
        hitch - cart.data.root_pos_w[:, None, :],
    )
    expected_hitch_local = torch.tensor(
        ((HITCH_X, HITCH_HALF_WIDTH, HITCH_Z), (HITCH_X, -HITCH_HALF_WIDTH, HITCH_Z)),
        device=env.device,
        dtype=hitch.dtype,
    ).view(1, 2, 3)
    relative_to_origin = grasp - origins[:, None, :]
    expected_preload = env.d6_preload_offset_w[:, None, :]
    actual_preload = hitch - grasp
    return {
        "grasp_hitch_position_error_m": torch.linalg.vector_norm(grasp - hitch, dim=-1),
        "grasp_hitch_offset_w_m": actual_preload,
        "expected_d6_preload_offset_w_m": expected_preload.expand_as(actual_preload),
        "d6_preload_error_w_m": actual_preload - expected_preload,
        "d6_preload_position_error_m": torch.linalg.vector_norm(
            actual_preload - expected_preload, dim=-1
        ),
        "hitch_local_frame_error_m": torch.linalg.vector_norm(hitch_local - expected_hitch_local, dim=-1),
        "hand_path_position_m": torch.sum(relative_to_origin * env.path_tangent_w[:, None, :], dim=-1),
        "hand_lateral_position_m": torch.sum(relative_to_origin * env.path_lateral_w[:, None, :], dim=-1),
        "hand_normal_height_m": torch.sum(relative_to_origin * env.path_normal_w[:, None, :], dim=-1),
        **_orientation_metrics(env),
    }


def _rows(slopes: tuple[float, ...], metrics: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    cpu = {name: value.detach().cpu() for name, value in metrics.items()}
    result: list[dict[str, object]] = []
    for index, slope in enumerate(slopes):
        row: dict[str, object] = {"environment_index": index, "slope": slope}
        for name, values in cpu.items():
            value = values[index]
            row[name] = value.tolist() if value.ndim else float(value)
        result.append(row)
    return result


def _per_environment_max(value: torch.Tensor) -> torch.Tensor:
    if value.ndim <= 1:
        return value
    return value.reshape(value.shape[0], -1).amax(dim=-1)


def _static_preload_joint_evidence_for_poses(
    rows: list[tuple[float, int | None, ResetPose]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for slope, candidate_id, pose in rows:
        offsets = [
            abs(reference - reset)
            for reference, reset in zip(pose.q_ref, pose.q_reset, strict=True)
        ]
        groups = (
            (
                "lower",
                0,
                RESET_LOWER_STIFFNESS,
                LOWER_HARDWARE_EFFORT_LIMITS,
            ),
            (
                "arm",
                15,
                RESET_ARM_STIFFNESS,
                ARM_HARDWARE_EFFORT_LIMITS,
            ),
            (
                "waist",
                12,
                RESET_WAIST_STIFFNESS,
                WAIST_HARDWARE_EFFORT_LIMITS,
            ),
        )
        row: dict[str, object] = {"slope": float(slope)}
        if candidate_id is not None:
            row["candidate_id"] = int(candidate_id)
        for label, begin, stiffness, limits in groups:
            entries = []
            for local_index, (gain, limit) in enumerate(
                zip(stiffness, limits, strict=True)
            ):
                joint_index = begin + local_index
                required_torque = offsets[joint_index] * gain
                entries.append(
                    {
                        "joint": G1_JOINT_ORDER[joint_index],
                        "ratio": required_torque / limit,
                        "required_preload_torque_nm": required_torque,
                        "hardware_effort_limit_nm": limit,
                    }
                )
            peak = max(entries, key=lambda entry: float(entry["ratio"]))
            row[f"{label}_peak"] = peak
            row[f"{label}_joints_above_0p7"] = [
                entry for entry in entries if float(entry["ratio"]) > 0.7
            ]
        result.append(row)
    return result


def _static_preload_joint_evidence(reset_library) -> list[dict[str, object]]:
    return _static_preload_joint_evidence_for_poses(
        [
            (float(slope), None, reset_library.pose_for_gradient(slope))
            for slope in SLOPE_GRADIENTS
        ]
    )


def _input_binding(
    cfg: Any, candidate_batch_path: Path | None = None
) -> dict[str, object]:
    feasibility_path = Path(cfg.feasibility_path).resolve()
    reset_pose_path = Path(cfg.reset_pose_path).resolve()
    result: dict[str, object] = {
        "feasibility_path": str(feasibility_path),
        "reset_pose_path": str(reset_pose_path),
    }
    if candidate_batch_path is not None:
        result["candidate_batch_path"] = str(candidate_batch_path.resolve())
    return result


def _bind_reset_pose_library(cfg: Any, reset_pose_path: Path) -> ResetPoseLibrary:
    """Keep the validation path and the runtime reset library on the same file."""

    resolved = Path(reset_pose_path).resolve()
    library = load_reset_pose_library(resolved)
    cfg.reset_pose_path = os.fspath(resolved)
    cfg.reset_pose_library = library
    return library


def _configure_validation_horizon(cfg: Any, steps: int) -> None:
    """Keep the audit horizon inside one episode so its final state is observable."""

    if steps <= 0:
        raise ValueError("validation steps must be positive")
    step_dt = float(cfg.sim.dt) * int(cfg.decimation)
    if step_dt <= 0.0:
        raise ValueError("validation environment step_dt must be positive")
    cfg.episode_length_s = (steps + 1) * step_dt


def _load_simulation_dependencies() -> None:
    global gym, torch
    global ARM_HARDWARE_EFFORT_LIMITS, LOWER_HARDWARE_EFFORT_LIMITS
    global RESET_ARM_STIFFNESS, RESET_LOWER_STIFFNESS, RESET_WAIST_STIFFNESS
    global WAIST_HARDWARE_EFFORT_LIMITS, static_preload_hardware_ratios
    global static_waist_preload_hardware_ratios
    global G1RickshawDirectionalSlopePlayEnvCfg, PLAY_TASK_ID
    global quat_apply_wxyz, actuator_effort_limits, install_reset_pose_batch
    global RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT

    try:
        from pxr import Usd as _Usd  # noqa: F401
    except ModuleNotFoundError as exc:
        launcher = isaaclab_root() / "isaaclab.sh"
        raise RuntimeError(
            "Isaac Sim started but its USD Python module 'pxr' is unavailable. "
            "Do not run this program with plain `python`; launch it through "
            f"`{launcher} -p scripts/solve_reset_poses.py --headless "
            "--reuse-candidates` so Kit's Python and native-library paths are set."
        ) from exc

    import gymnasium as gym_module
    import torch as torch_module
    from g1_rickshaw_lab.configuration import (
        ARM_HARDWARE_EFFORT_LIMITS as arm_limits,
        LOWER_HARDWARE_EFFORT_LIMITS as lower_limits,
        RESET_ARM_STIFFNESS as arm_stiffness,
        RESET_LOWER_STIFFNESS as lower_stiffness,
        RESET_WAIST_STIFFNESS as waist_stiffness,
        WAIST_HARDWARE_EFFORT_LIMITS as waist_limits,
        static_preload_hardware_ratios as preload_ratios,
        static_waist_preload_hardware_ratios as waist_preload_ratios,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import (
        G1RickshawDirectionalSlopePlayEnvCfg as env_cfg,
        PLAY_TASK_ID as task_id,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
        install_reset_pose_batch as install_pose_batch,
        quat_apply_wxyz as apply_quaternion,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actuation import (
        actuator_effort_limits as effort_limits,
    )
    from g1_rickshaw_lab.validation import (
        RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT as torque_contract,
    )

    gym = gym_module
    torch = torch_module
    ARM_HARDWARE_EFFORT_LIMITS = arm_limits
    LOWER_HARDWARE_EFFORT_LIMITS = lower_limits
    RESET_ARM_STIFFNESS = arm_stiffness
    RESET_LOWER_STIFFNESS = lower_stiffness
    RESET_WAIST_STIFFNESS = waist_stiffness
    WAIST_HARDWARE_EFFORT_LIMITS = waist_limits
    static_preload_hardware_ratios = preload_ratios
    static_waist_preload_hardware_ratios = waist_preload_ratios
    G1RickshawDirectionalSlopePlayEnvCfg = env_cfg
    PLAY_TASK_ID = task_id
    quat_apply_wxyz = apply_quaternion
    install_reset_pose_batch = install_pose_batch
    actuator_effort_limits = effort_limits
    RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT = torque_contract

def _validate_round(
    args_cli: argparse.Namespace, reset_pose_path: Path
) -> dict[str, Any]:
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive")
    if args_cli.timeseries_stride < 0:
        raise ValueError("--timeseries-stride must be non-negative")
    if args_cli.stable_displacement_limit <= 0.0:
        raise ValueError("--stable-displacement-limit must be positive")
    candidate_batch_value = os.environ.get(CANDIDATE_BATCH_ENV)
    candidate_batch_path = (
        Path(candidate_batch_value).resolve()
        if candidate_batch_value is not None
        else None
    )
    candidate_batch = (
        _load_candidate_batch(candidate_batch_path)
        if candidate_batch_path is not None
        else None
    )
    slopes = (
        tuple(float(row["slope"]) for row in candidate_batch)
        if candidate_batch is not None
        else tuple(float(value) for value in SLOPE_GRADIENTS)
    )
    os.environ["G1_RICKSHAW_RESET_POSES"] = os.fspath(reset_pose_path.resolve())
    cfg = G1RickshawDirectionalSlopePlayEnvCfg()
    # Reset validation always exercises the real closed chain, never the
    # synthetic hand-load pretraining stage.
    cfg.domain_randomization = replace(
        cfg.domain_randomization,
        enabled=False,
    )
    cfg.events.initialize_domain.params = {"cfg": cfg.domain_randomization}
    reset_library = _bind_reset_pose_library(cfg, reset_pose_path)
    cfg.scene.num_envs = len(slopes)
    cfg.sim.device = args_cli.device
    _configure_validation_horizon(cfg, args_cli.steps)
    if candidate_batch is None:
        static_lower_preload_ratio, static_arm_preload_ratio = (
            static_preload_hardware_ratios(reset_library)
        )
        static_waist_preload_ratio = static_waist_preload_hardware_ratios(reset_library)
        static_preload_joint_evidence = _static_preload_joint_evidence(reset_library)
    else:
        static_preload_joint_evidence = _static_preload_joint_evidence_for_poses(
            [
                (row["slope"], row["candidate_id"], row["pose"])
                for row in candidate_batch
            ]
        )
        static_lower_preload_ratio = tuple(
            float(row["lower_peak"]["ratio"])
            for row in static_preload_joint_evidence
        )
        static_waist_preload_ratio = tuple(
            float(row["waist_peak"]["ratio"])
            for row in static_preload_joint_evidence
        )
        static_arm_preload_ratio = tuple(
            float(row["arm_peak"]["ratio"])
            for row in static_preload_joint_evidence
        )
    if args_cli.foot_stiffness is not None:
        if args_cli.foot_stiffness <= 0.0:
            raise ValueError("--foot-stiffness must be positive")
        cfg.scene.robot.actuators["feet"].stiffness = args_cli.foot_stiffness
    if args_cli.foot_damping is not None:
        if args_cli.foot_damping < 0.0:
            raise ValueError("--foot-damping must be non-negative")
        cfg.scene.robot.actuators["feet"].damping = args_cli.foot_damping
    if args_cli.leg_stiffness is not None:
        if args_cli.leg_stiffness <= 0.0:
            raise ValueError("--leg-stiffness must be positive")
        cfg.scene.robot.actuators["legs"].stiffness = args_cli.leg_stiffness
    if args_cli.leg_damping is not None:
        if args_cli.leg_damping < 0.0:
            raise ValueError("--leg-damping must be non-negative")
        cfg.scene.robot.actuators["legs"].damping = args_cli.leg_damping
    cfg.policy_update = replace(
        cfg.policy_update,
        command_sampling=replace(
            cfg.policy_update.command_sampling,
            standing_fraction=1.0,
        ),
    )
    cfg.events.policy_interval.params["cfg"] = cfg.policy_update
    report_inputs = _input_binding(cfg, candidate_batch_path)
    env = gym.make(PLAY_TASK_ID, cfg=cfg)
    base = env.unwrapped
    if candidate_batch is not None:
        install_reset_pose_batch(
            base, [row["pose"] for row in candidate_batch]
        )
    if base.max_episode_length <= args_cli.steps:
        raise RuntimeError(
            "reset validation episode must be longer than the requested horizon: "
            f"max_episode_length={base.max_episode_length}, steps={args_cli.steps}"
        )
    immediate_safety_cfg = base.termination_manager.get_term_cfg(
        "immediate_safety"
    ).params["cfg"]
    persistent_safety_cfg = base.termination_manager.get_term_cfg(
        "persistent_safety"
    ).params["cfg"]
    arm_safety_limit = float(persistent_safety_cfg.arm_torque_limit)
    d6_residual_limit = float(immediate_safety_cfg.d6_residual_limit)
    d6_impulse_limit = float(immediate_safety_cfg.d6_impulse_limit)
    try:
        if candidate_batch is None:
            _set_signed_slope_origins(base, slopes)
        else:
            _set_terrain_origins(
                base,
                [row["terrain_level"] for row in candidate_batch],
                [row["terrain_column"] for row in candidate_batch],
            )
        env.reset(seed=args_cli.seed)
        base.command_state.v_sample.zero_()
        base.command_state.v_ref.zero_()
        base.command_state.a_ref.zero_()
        expected_slopes = torch.tensor(slopes, device=base.device, dtype=base.slope.dtype)
        if not torch.allclose(base.slope, expected_slopes, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(
                f"fixed terrain assignment did not resolve to the required {len(slopes)} slopes: "
                f"expected={expected_slopes.detach().cpu().tolist()}, "
                f"actual={base.slope.detach().cpu().tolist()}"
            )
        initial = _initial_alignment(base)
        initial_grasp_positions = _grasp_positions(base).clone()
        initial_root_position = base.scene["robot"].data.root_pos_w.clone()

        max_d6_residual = torch.zeros(base.num_envs, device=base.device)
        max_d6_position_residual = torch.zeros_like(max_d6_residual)
        max_d6_rotation_residual = torch.zeros_like(max_d6_residual)
        max_d6_position_step = torch.zeros(base.num_envs, device=base.device, dtype=torch.long)
        max_d6_rotation_step = torch.zeros_like(max_d6_position_step)
        max_d6_impulse = torch.zeros_like(max_d6_residual)
        max_arm_torque_ratio = torch.zeros_like(max_d6_residual)
        max_arm_torque_joint = torch.full(
            (base.num_envs,), -1, device=base.device, dtype=torch.long
        )
        max_arm_torque_step = torch.zeros_like(max_arm_torque_joint)
        max_waist_torque_ratio = torch.zeros_like(max_d6_residual)
        max_waist_torque_joint = torch.full_like(max_arm_torque_joint, -1)
        max_waist_torque_step = torch.zeros_like(max_arm_torque_joint)
        active_d6_residual = torch.zeros_like(max_d6_residual)
        active_d6_impulse = torch.zeros_like(max_d6_residual)
        active_arm_torque_ratio = torch.zeros_like(max_d6_residual)
        active_waist_torque_ratio = torch.zeros_like(max_d6_residual)
        active_lower_torque_ratio = torch.zeros_like(max_d6_residual)
        active_policy_steps = torch.zeros(
            base.num_envs, device=base.device, dtype=torch.long
        )
        minimum_root_normal_height = torch.full_like(max_d6_residual, torch.inf)
        maximum_abs_torso_pitch = torch.zeros_like(max_d6_residual)
        maximum_policy_joint_error = torch.zeros_like(max_d6_residual)
        maximum_lower_torque_ratio = torch.zeros_like(max_d6_residual)
        maximum_lower_torque_joint = torch.full_like(max_arm_torque_joint, -1)
        maximum_lower_torque_step = torch.zeros_like(max_arm_torque_joint)
        minimum_dex_forward_dot_path_tangent = torch.ones_like(max_d6_residual)
        maximum_abs_dex_forward_dot_path_lateral = torch.zeros_like(max_d6_residual)
        first_fall_step = torch.zeros(
            base.num_envs, device=base.device, dtype=torch.long
        )
        first_fall_path_displacement = torch.zeros_like(max_d6_residual)
        first_fall_torso_pitch = torch.zeros_like(max_d6_residual)
        timeseries: list[dict[str, object]] = []
        termination_steps: list[list[int]] = [[] for _ in slopes]
        actions = torch.zeros(env.action_space.shape, device=base.device)
        robot = base.scene["robot"]
        effort_limits = actuator_effort_limits(robot, base.arm_joint_ids)
        lower_joint_ids = base.policy_joint_ids[:12]
        lower_effort_limits = actuator_effort_limits(robot, lower_joint_ids)
        waist_joint_ids = base.policy_joint_ids[12:15]
        waist_effort_limits = actuator_effort_limits(robot, waist_joint_ids)

        with torch.inference_mode():
            for step_index in range(args_cli.steps):
                safety_active = torch.ones(
                    base.num_envs, device=base.device, dtype=torch.bool
                )
                _, _, terminated, _truncated, _ = env.step(actions)
                d6_residual = _per_environment_max(base.rickshaw_state.d6_residual)
                d6_impulse = _per_environment_max(base.rickshaw_state.d6_impulse)
                max_d6_residual = torch.maximum(max_d6_residual, d6_residual)
                _, position_residual, rotation_residual, _ = base.d6_reaction_residual_provider(
                    base.d6_constraint_manager.joint_paths
                )
                position_norm = torch.linalg.vector_norm(position_residual, dim=-1).amax(dim=-1)
                rotation_norm = torch.linalg.vector_norm(rotation_residual, dim=-1).amax(dim=-1)
                position_peak = position_norm > max_d6_position_residual
                rotation_peak = rotation_norm > max_d6_rotation_residual
                max_d6_position_residual = torch.maximum(max_d6_position_residual, position_norm)
                max_d6_rotation_residual = torch.maximum(max_d6_rotation_residual, rotation_norm)
                max_d6_position_step[position_peak] = step_index + 1
                max_d6_rotation_step[rotation_peak] = step_index + 1
                max_d6_impulse = torch.maximum(max_d6_impulse, d6_impulse)
                arm_joint_ratio = (
                    torch.abs(robot.data.applied_torque[:, base.arm_joint_ids])
                    / effort_limits
                )
                arm_ratio, arm_joint = torch.max(arm_joint_ratio, dim=-1)
                new_arm_peak = arm_ratio > max_arm_torque_ratio
                max_arm_torque_joint[new_arm_peak] = arm_joint[new_arm_peak]
                max_arm_torque_step[new_arm_peak] = step_index + 1
                max_arm_torque_ratio = torch.maximum(max_arm_torque_ratio, arm_ratio)
                waist_joint_ratio = (
                    torch.abs(robot.data.applied_torque[:, waist_joint_ids])
                    / waist_effort_limits
                )
                waist_ratio, waist_joint = torch.max(waist_joint_ratio, dim=-1)
                new_waist_peak = waist_ratio > max_waist_torque_ratio
                max_waist_torque_joint[new_waist_peak] = waist_joint[new_waist_peak]
                max_waist_torque_step[new_waist_peak] = step_index + 1
                max_waist_torque_ratio = torch.maximum(
                    max_waist_torque_ratio, waist_ratio
                )
                root_normal_height = torch.sum(
                    (robot.data.root_pos_w - base.scene.terrain.env_origins)
                    * base.path_normal_w,
                    dim=-1,
                )
                abs_torso_pitch = torch.abs(base.stability_state.torso_pitch)
                policy_joint_error = torch.max(
                    torch.abs(
                        robot.data.joint_pos[:, base.policy_joint_ids]
                        - base.action_state.q_ref
                    ),
                    dim=-1,
                ).values
                lower_joint_ratio = (
                    torch.abs(robot.data.applied_torque[:, lower_joint_ids])
                    / lower_effort_limits
                )
                lower_torque_ratio, lower_torque_joint = torch.max(
                    lower_joint_ratio, dim=-1
                )
                new_lower_peak = lower_torque_ratio > maximum_lower_torque_ratio
                maximum_lower_torque_joint[new_lower_peak] = lower_torque_joint[
                    new_lower_peak
                ]
                maximum_lower_torque_step[new_lower_peak] = step_index + 1
                orientation = _orientation_metrics(base)
                dex_forward_tangent = torch.min(
                    orientation["dex_forward_dot_path_tangent"], dim=-1
                ).values
                dex_forward_lateral = torch.max(
                    torch.abs(orientation["dex_forward_dot_path_lateral"]), dim=-1
                ).values
                minimum_dex_forward_dot_path_tangent = torch.minimum(
                    minimum_dex_forward_dot_path_tangent, dex_forward_tangent
                )
                maximum_abs_dex_forward_dot_path_lateral = torch.maximum(
                    maximum_abs_dex_forward_dot_path_lateral, dex_forward_lateral
                )
                minimum_root_normal_height = torch.minimum(
                    minimum_root_normal_height, root_normal_height
                )
                maximum_abs_torso_pitch = torch.maximum(
                    maximum_abs_torso_pitch, abs_torso_pitch
                )
                maximum_policy_joint_error = torch.maximum(
                    maximum_policy_joint_error, policy_joint_error
                )
                maximum_lower_torque_ratio = torch.maximum(
                    maximum_lower_torque_ratio, lower_torque_ratio
                )
                root_path_displacement = torch.sum(
                    (robot.data.root_pos_w - initial_root_position)
                    * base.path_tangent_w,
                    dim=-1,
                )
                standing = (root_normal_height >= 0.60) & (abs_torso_pitch <= 0.45)
                newly_fallen = (~standing) & (first_fall_step == 0)
                first_fall_step[newly_fallen] = step_index + 1
                first_fall_path_displacement[newly_fallen] = root_path_displacement[
                    newly_fallen
                ]
                first_fall_torso_pitch[newly_fallen] = base.stability_state.torso_pitch[
                    newly_fallen
                ]
                if args_cli.timeseries_stride and (
                    step_index == 0
                    or (step_index + 1) % args_cli.timeseries_stride == 0
                    or step_index + 1 == args_cli.steps
                ):
                    hand_force = base.rickshaw_state.hand_force_w
                    hand_torque = base.rickshaw_state.hand_torque_w
                    hand_force_s = torch.sum(hand_force * base.path_tangent_w, dim=-1)
                    hand_force_n = torch.sum(hand_force * base.path_normal_w, dim=-1)
                    hand_torque_y = torch.sum(hand_torque * base.path_lateral_w, dim=-1)
                    policy_error = torch.abs(
                        robot.data.joint_pos[:, base.policy_joint_ids]
                        - base.action_state.q_ref
                    )
                    lower_error = torch.max(policy_error[:, :12], dim=-1).values
                    arm_error = torch.max(policy_error[:, 15:], dim=-1).values
                    lower_computed_torque = torch.max(
                        torch.abs(robot.data.computed_torque[:, lower_joint_ids]), dim=-1
                    ).values
                    lower_applied_torque = torch.max(
                        torch.abs(robot.data.applied_torque[:, lower_joint_ids]), dim=-1
                    ).values
                    timeseries.append(
                        {
                            "step": step_index + 1,
                            "root_normal_height_m": root_normal_height.detach().cpu().tolist(),
                            "root_path_displacement_m": (
                                root_path_displacement.detach().cpu().tolist()
                            ),
                            "torso_pitch_rad": base.stability_state.torso_pitch.detach().cpu().tolist(),
                            "dex_forward_dot_path_tangent": orientation[
                                "dex_forward_dot_path_tangent"
                            ].detach().cpu().tolist(),
                            "fat2_target_pitch_rad": base.stability_state.theta_fat.detach().cpu().tolist(),
                            "zmp_margin_m": base.stability_state.zmp_margin.detach().cpu().tolist(),
                            "zmp_valid": base.stability_state.zmp_valid.detach().cpu().tolist(),
                            "hand_force_s_n": hand_force_s.detach().cpu().tolist(),
                            "hand_force_n_n": hand_force_n.detach().cpu().tolist(),
                            "hand_torque_y_nm": hand_torque_y.detach().cpu().tolist(),
                            "analytic_t_s_n": base.analytic_force_state.t_s.detach().cpu().tolist(),
                            "analytic_t_n_n": base.analytic_force_state.t_n.detach().cpu().tolist(),
                            "policy_joint_error_rad": policy_joint_error.detach().cpu().tolist(),
                            "lower_joint_error_rad": lower_error.detach().cpu().tolist(),
                            "arm_joint_error_rad": arm_error.detach().cpu().tolist(),
                            "lower_computed_torque_nm": lower_computed_torque.detach().cpu().tolist(),
                            "lower_applied_torque_nm": lower_applied_torque.detach().cpu().tolist(),
                        }
                    )
                active_policy_steps += safety_active.to(torch.long)
                active_d6_residual = torch.maximum(
                    active_d6_residual,
                    torch.where(safety_active, d6_residual, 0.0),
                )
                active_d6_impulse = torch.maximum(
                    active_d6_impulse,
                    torch.where(safety_active, d6_impulse, 0.0),
                )
                active_arm_torque_ratio = torch.maximum(
                    active_arm_torque_ratio,
                    torch.where(safety_active, arm_ratio, 0.0),
                )
                active_waist_torque_ratio = torch.maximum(
                    active_waist_torque_ratio,
                    torch.where(safety_active, waist_ratio, 0.0),
                )
                active_lower_torque_ratio = torch.maximum(
                    active_lower_torque_ratio,
                    torch.where(safety_active, lower_torque_ratio, 0.0),
                )
                failed = terminated.detach().cpu()
                for env_index in torch.nonzero(failed, as_tuple=False).flatten().tolist():
                    termination_steps[env_index].append(step_index + 1)

        final_gap = torch.linalg.vector_norm(
            _grasp_positions(base) - base.scene["rickshaw"].data.body_pos_w[:, base.hitch_body_ids],
            dim=-1,
        )
        final_orientation = _orientation_metrics(base)
        final_root_normal_height = torch.sum(
            (robot.data.root_pos_w - base.scene.terrain.env_origins)
            * base.path_normal_w,
            dim=-1,
        )
        final_path_displacement = torch.sum(
            (robot.data.root_pos_w - initial_root_position) * base.path_tangent_w,
            dim=-1,
        )
        final_torso_pitch = base.stability_state.torso_pitch
        final_path_velocity = torch.sum(
            robot.data.root_lin_vel_w * base.path_tangent_w, dim=-1
        )
        final_fat_wrench_consistent = (
            base.stability_state.fat_wrench_consistent.clone()
        )
        final_fat_wrench_relative_error = (
            base.stability_state.fat_wrench_relative_error.clone()
        )
        report = {
            "schema_version": 2,
            "tool": "validate_reset_alignment",
            "task": PLAY_TASK_ID,
            "steps": args_cli.steps,
            "slopes": list(slopes),
            "seed": args_cli.seed,
            "continuous_standing": True,
            "physics_mode": "fixed",
            "inputs": report_inputs,
            "torque_measurement_contract": dict(
                RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT
            ),
            "safety_thresholds": {
                "arm_torque_ratio": arm_safety_limit,
                "d6_residual_m_or_rad": d6_residual_limit,
                "d6_impulse_n_s": d6_impulse_limit,
                "static_lower_preload_ratio": RESET_TORQUE_LIMIT_FRACTION,
                "static_waist_preload_ratio": RESET_TORQUE_LIMIT_FRACTION,
                "static_arm_preload_ratio": RESET_TORQUE_LIMIT_FRACTION,
            },
            "rickshaw_pose_contract": {
                "hitch_height_target_m": float(cfg.rickshaw_pose.hitch_height_target),
                "hitch_height_tolerance_m": float(
                    cfg.rickshaw_pose.hitch_height_tolerance
                ),
                "hand_path_position_contract": (
                    "finite robot-relative position; closed-chain alignment, "
                    "robot-ahead and collision gates define admissibility"
                ),
            },
            "static_preload_hardware_evidence": static_preload_joint_evidence,
            "expected_hitch_local_frames_m": [
                [HITCH_X, HITCH_HALF_WIDTH, HITCH_Z],
                [HITCH_X, -HITCH_HALF_WIDTH, HITCH_Z],
            ],
            "initial": _rows(slopes, initial),
            "rollout": [
                {
                    "environment_index": index,
                    "slope": slope,
                    "max_d6_residual_m_or_rad": float(max_d6_residual[index]),
                    "max_d6_position_residual_m": float(max_d6_position_residual[index]),
                    "max_d6_position_residual_step": int(max_d6_position_step[index]),
                    "max_d6_rotation_residual_rad": float(max_d6_rotation_residual[index]),
                    "max_d6_rotation_residual_step": int(max_d6_rotation_step[index]),
                    "max_d6_impulse_n_s": float(max_d6_impulse[index]),
                    "max_arm_torque_ratio": float(max_arm_torque_ratio[index]),
                    "max_arm_torque_joint": (
                        G1_JOINT_ORDER[15 + int(max_arm_torque_joint[index].item())]
                        if int(max_arm_torque_joint[index].item()) >= 0
                        else None
                    ),
                    "max_arm_torque_step": int(max_arm_torque_step[index]),
                    "max_waist_torque_ratio": float(max_waist_torque_ratio[index]),
                    "max_waist_torque_joint": (
                        G1_JOINT_ORDER[12 + int(max_waist_torque_joint[index].item())]
                        if int(max_waist_torque_joint[index].item()) >= 0
                        else None
                    ),
                    "max_waist_torque_step": int(max_waist_torque_step[index]),
                    "static_preload_lower_hardware_ratio": (
                        static_lower_preload_ratio[index]
                    ),
                    "static_preload_waist_hardware_ratio": (
                        static_waist_preload_ratio[index]
                    ),
                    "static_preload_arm_hardware_ratio": (
                        static_arm_preload_ratio[index]
                    ),
                    "final_fat_wrench_consistent": bool(
                        final_fat_wrench_consistent[index]
                    ),
                    "final_fat_wrench_relative_error": (
                        final_fat_wrench_relative_error[index]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "safety_active_policy_steps": int(active_policy_steps[index]),
                    "safety_active_d6_residual_max_m_or_rad": float(
                        active_d6_residual[index]
                    ),
                    "safety_active_d6_impulse_max_n_s": float(active_d6_impulse[index]),
                    "safety_active_arm_torque_ratio_max": float(
                        active_arm_torque_ratio[index]
                    ),
                    "safety_active_waist_torque_ratio_max": float(
                        active_waist_torque_ratio[index]
                    ),
                    "minimum_root_normal_height_m": float(
                        minimum_root_normal_height[index]
                    ),
                    "maximum_abs_torso_pitch_rad": float(
                        maximum_abs_torso_pitch[index]
                    ),
                    "maximum_policy_joint_error_rad": float(
                        maximum_policy_joint_error[index]
                    ),
                    "maximum_lower_torque_ratio": float(
                        maximum_lower_torque_ratio[index]
                    ),
                    "maximum_lower_torque_joint": (
                        G1_JOINT_ORDER[
                            int(maximum_lower_torque_joint[index].item())
                        ]
                        if int(maximum_lower_torque_joint[index].item()) >= 0
                        else None
                    ),
                    "maximum_lower_torque_step": int(
                        maximum_lower_torque_step[index]
                    ),
                    "minimum_dex_forward_dot_path_tangent": float(
                        minimum_dex_forward_dot_path_tangent[index]
                    ),
                    "maximum_abs_dex_forward_dot_path_lateral": float(
                        maximum_abs_dex_forward_dot_path_lateral[index]
                    ),
                    "first_fall_step": int(first_fall_step[index]),
                    "first_fall_root_path_displacement_m": float(
                        first_fall_path_displacement[index]
                    ),
                    "first_fall_torso_pitch_rad": float(
                        first_fall_torso_pitch[index]
                    ),
                    "final_root_normal_height_m": float(
                        final_root_normal_height[index]
                    ),
                    "final_root_path_displacement_m": float(
                        final_path_displacement[index]
                    ),
                    "final_torso_pitch_rad": float(final_torso_pitch[index]),
                    "final_root_path_velocity_mps": float(final_path_velocity[index]),
                    "final_grasp_hitch_position_error_m": final_gap[index].detach().cpu().tolist(),
                    "final_robot_forward_dot_path_tangent": float(
                        final_orientation["robot_forward_dot_path_tangent"][index]
                    ),
                    "final_cart_projected_heading_dot_path_tangent": float(
                        final_orientation["cart_projected_heading_dot_path_tangent"][index]
                    ),
                    "final_dex_forward_dot_path_tangent": (
                        final_orientation["dex_forward_dot_path_tangent"][index]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "termination_steps": termination_steps[index],
                }
                for index, slope in enumerate(slopes)
            ],
            "timeseries": timeseries,
        }
        if candidate_batch is not None:
            report["candidate_bindings"] = [
                {
                    "environment_index": index,
                    "slope": row["slope"],
                    "candidate_id": row["candidate_id"],
                    "terrain_level": row["terrain_level"],
                    "terrain_column": row["terrain_column"],
                }
                for index, row in enumerate(candidate_batch)
            ]

        initial_gap_max = float(torch.max(initial["grasp_hitch_position_error_m"]))
        initial_preload_error_max = float(torch.max(initial["d6_preload_position_error_m"]))
        initial_grasp_spacing = torch.linalg.vector_norm(
            initial_grasp_positions[:, 0] - initial_grasp_positions[:, 1], dim=-1
        )
        hitch_frame_error_max = float(torch.max(initial["hitch_local_frame_error_m"]))
        hitch_height_target = float(cfg.rickshaw_pose.hitch_height_target)
        hand_height_error_max = float(
            torch.max(torch.abs(initial["hand_normal_height_m"] - hitch_height_target))
        )
        hand_path_min = float(torch.min(initial["hand_path_position_m"]))
        hand_path_max = float(torch.max(initial["hand_path_position_m"]))
        hand_path_finite = bool(torch.all(torch.isfinite(initial["hand_path_position_m"])))
        minimum_robot_heading = float(torch.min(initial["robot_forward_dot_path_tangent"]))
        minimum_cart_heading = float(torch.min(initial["cart_projected_heading_dot_path_tangent"]))
        minimum_robot_ahead = float(torch.min(initial["robot_ahead_of_cart_m"]))
        minimum_initial_dex_forward = float(
            torch.min(initial["dex_forward_dot_path_tangent"])
        )
        maximum_initial_abs_dex_lateral = float(
            torch.max(torch.abs(initial["dex_forward_dot_path_lateral"]))
        )
        minimum_rollout_dex_forward = float(
            torch.min(minimum_dex_forward_dot_path_tangent)
        )
        maximum_rollout_abs_dex_lateral = float(
            torch.max(maximum_abs_dex_forward_dot_path_lateral)
        )
        rollout_residual_max = float(torch.max(max_d6_residual))
        rollout_position_residual_max = float(torch.max(max_d6_position_residual))
        rollout_rotation_residual_max = float(torch.max(max_d6_rotation_residual))
        rollout_impulse_max = float(torch.max(max_d6_impulse))
        rollout_torque_max = float(torch.max(max_arm_torque_ratio))
        rollout_waist_torque_max = float(torch.max(max_waist_torque_ratio))
        active_steps_min = int(torch.min(active_policy_steps))
        active_residual_max = float(torch.max(active_d6_residual))
        active_impulse_max = float(torch.max(active_d6_impulse))
        active_torque_max = float(torch.max(active_arm_torque_ratio))
        active_waist_torque_max = float(torch.max(active_waist_torque_ratio))
        active_lower_torque_max = float(torch.max(active_lower_torque_ratio))
        standing_root_min = float(torch.min(minimum_root_normal_height))
        standing_torso_max = float(torch.max(maximum_abs_torso_pitch))
        standing_joint_error_max = float(torch.max(maximum_policy_joint_error))
        standing_lower_torque_max = float(torch.max(maximum_lower_torque_ratio))
        final_path_displacement_max = float(torch.max(torch.abs(final_path_displacement)))
        non_timeout_termination_count = sum(len(values) for values in termination_steps)
        rollout_arm_peak_env = int(torch.argmax(max_arm_torque_ratio))
        rollout_waist_peak_env = int(torch.argmax(max_waist_torque_ratio))
        rollout_lower_peak_env = int(torch.argmax(maximum_lower_torque_ratio))
        rollout_arm_peak_joint = int(
            max_arm_torque_joint[rollout_arm_peak_env].item()
        )
        rollout_waist_peak_joint = int(
            max_waist_torque_joint[rollout_waist_peak_env].item()
        )
        rollout_lower_peak_joint = int(
            maximum_lower_torque_joint[rollout_lower_peak_env].item()
        )
        if (
            rollout_arm_peak_joint < 0
            or rollout_waist_peak_joint < 0
            or rollout_lower_peak_joint < 0
        ):
            raise RuntimeError("torque audit collected no arm/waist/lower actuator samples")
        static_arm_violations = [
            {"slope": row["slope"], **entry}
            for row in static_preload_joint_evidence
            for entry in row["arm_joints_above_0p7"]
        ]
        static_arm_worst = max(
            (
                {"slope": row["slope"], **row["arm_peak"]}
                for row in static_preload_joint_evidence
            ),
            key=lambda entry: float(entry["ratio"]),
        )
        checks = {
            "initial_d6_preload_error_le_1mm": initial_preload_error_max <= 1.0e-3,
            "usd_hitch_local_frame_error_le_0p1mm": hitch_frame_error_max <= 1.0e-4,
            "hand_height_error_le_1mm": hand_height_error_max <= 1.0e-3,
            "hand_path_position_is_finite": hand_path_finite,
            "initial_grasp_hitch_alignment_within_runtime_tolerance": (
                initial_gap_max <= cfg.reset_validation.hand_position_tolerance
            ),
            "robot_forward_is_path_plus_x": minimum_robot_heading >= 0.9999,
            "cart_projected_heading_is_path_plus_x": minimum_cart_heading >= 0.9999,
            "robot_is_ahead_of_cart": minimum_robot_ahead > 0.0,
            "dex_plus_x_faces_path_plus_x": (
                minimum_initial_dex_forward >= DEX_FORWARD_MIN_DOT
                and maximum_initial_abs_dex_lateral
                <= DEX_FORWARD_INITIAL_MAX_LATERAL
            ),
            "rollout_d6_residual_within_limit": (
                rollout_residual_max <= d6_residual_limit
            ),
            "rollout_d6_impulse_within_limit": (
                rollout_impulse_max <= d6_impulse_limit
            ),
            "static_preload_lower_within_hardware_margin": (
                max(static_lower_preload_ratio)
                <= RESET_TORQUE_LIMIT_FRACTION
            ),
            "static_preload_waist_within_hardware_margin": (
                max(static_waist_preload_ratio)
                <= RESET_TORQUE_LIMIT_FRACTION
            ),
            "static_preload_arm_within_hardware_margin": (
                max(static_arm_preload_ratio) <= RESET_TORQUE_LIMIT_FRACTION
            ),
        }
        checks.update(
            {
                "continuous_standing_root_height_ge_0p60m": standing_root_min >= 0.60,
                "continuous_standing_torso_tilt_le_0p45rad": standing_torso_max <= 0.45,
                "continuous_standing_joint_error_le_0p35rad": (
                    standing_joint_error_max <= 0.35
                ),
                "continuous_standing_no_fall_step": not bool(torch.any(first_fall_step)),
                "no_non_timeout_termination": non_timeout_termination_count == 0,
                "fat2_wrench_consistent_at_horizon": bool(
                    torch.all(final_fat_wrench_consistent)
                ),
            }
        )
        report["summary"] = {
            "initial_grasp_hitch_error_max_m": initial_gap_max,
            "initial_grasp_hitch_tolerance_m": float(
                cfg.reset_validation.hand_position_tolerance
            ),
            "initial_d6_preload_error_max_m": initial_preload_error_max,
            "usd_hitch_local_frame_error_max_m": hitch_frame_error_max,
            "hand_height_error_max_m": hand_height_error_max,
            "hand_path_position_min_m": hand_path_min,
            "hand_path_position_max_m": hand_path_max,
            "grasp_center_spacing_min_m": float(torch.min(initial_grasp_spacing)),
            "grasp_center_spacing_max_m": float(torch.max(initial_grasp_spacing)),
            "minimum_robot_forward_dot_path_tangent": minimum_robot_heading,
            "minimum_cart_projected_heading_dot_path_tangent": minimum_cart_heading,
            "minimum_robot_ahead_of_cart_m": minimum_robot_ahead,
            "dex_forward_initial_max_pitch_rad": DEX_FORWARD_MAX_PITCH_RAD,
            "dex_forward_rollout_max_pitch_rad": DEX_FORWARD_MAX_PITCH_RAD,
            "dex_forward_initial_max_lateral": DEX_FORWARD_INITIAL_MAX_LATERAL,
            "dex_forward_rollout_max_lateral": DEX_FORWARD_ROLLOUT_MAX_LATERAL,
            "minimum_initial_dex_forward_dot_path_tangent": (
                minimum_initial_dex_forward
            ),
            "maximum_initial_abs_dex_forward_dot_path_lateral": (
                maximum_initial_abs_dex_lateral
            ),
            "minimum_rollout_dex_forward_dot_path_tangent": (
                minimum_rollout_dex_forward
            ),
            "maximum_rollout_abs_dex_forward_dot_path_lateral": (
                maximum_rollout_abs_dex_lateral
            ),
            "rollout_d6_residual_max_m_or_rad": rollout_residual_max,
            "rollout_d6_position_residual_max_m": rollout_position_residual_max,
            "rollout_d6_rotation_residual_max_rad": rollout_rotation_residual_max,
            "rollout_d6_impulse_max_n_s": rollout_impulse_max,
            "rollout_arm_torque_ratio_max": rollout_torque_max,
            "rollout_arm_torque_ratio_worst_slope": slopes[rollout_arm_peak_env],
            "rollout_arm_torque_ratio_worst_joint": (
                G1_JOINT_ORDER[15 + rollout_arm_peak_joint]
            ),
            "rollout_arm_torque_ratio_worst_step": int(
                max_arm_torque_step[rollout_arm_peak_env]
            ),
            "rollout_waist_torque_ratio_max": rollout_waist_torque_max,
            "rollout_waist_torque_ratio_worst_slope": slopes[
                rollout_waist_peak_env
            ],
            "rollout_waist_torque_ratio_worst_joint": G1_JOINT_ORDER[
                12 + rollout_waist_peak_joint
            ],
            "rollout_waist_torque_ratio_worst_step": int(
                max_waist_torque_step[rollout_waist_peak_env]
            ),
            "static_preload_lower_hardware_ratio_max": max(
                static_lower_preload_ratio
            ),
            "static_preload_waist_hardware_ratio_max": max(
                static_waist_preload_ratio
            ),
            "static_preload_arm_hardware_ratio_max": max(static_arm_preload_ratio),
            "static_preload_arm_0p7_violation_count": len(static_arm_violations),
            "static_preload_arm_0p7_violation_slopes": sorted(
                {float(entry["slope"]) for entry in static_arm_violations}
            ),
            "static_preload_arm_worst": static_arm_worst,
            "final_fat_wrench_consistent_count": int(
                torch.sum(final_fat_wrench_consistent)
            ),
            "final_fat_wrench_relative_error_max": float(
                torch.max(final_fat_wrench_relative_error)
            ),
            "non_timeout_termination_count": non_timeout_termination_count,
            "non_timeout_termination_steps": termination_steps,
            "validation_max_episode_length": int(base.max_episode_length),
            "safety_active_policy_steps_min": active_steps_min,
            "safety_active_d6_residual_max_m_or_rad": active_residual_max,
            "safety_active_d6_impulse_max_n_s": active_impulse_max,
            "safety_active_arm_torque_ratio_max": active_torque_max,
            "safety_active_waist_torque_ratio_max": active_waist_torque_max,
            "safety_active_lower_torque_ratio_max": active_lower_torque_max,
            "continuous_standing_root_height_min_m": standing_root_min,
            "continuous_standing_abs_torso_pitch_max_rad": standing_torso_max,
            "continuous_standing_policy_joint_error_max_rad": standing_joint_error_max,
            "continuous_standing_lower_torque_ratio_max": standing_lower_torque_max,
            "continuous_standing_lower_torque_ratio_worst_slope": slopes[
                rollout_lower_peak_env
            ],
            "continuous_standing_lower_torque_ratio_worst_joint": G1_JOINT_ORDER[
                rollout_lower_peak_joint
            ],
            "continuous_standing_lower_torque_ratio_worst_step": int(
                maximum_lower_torque_step[rollout_lower_peak_env]
            ),
            "continuous_standing_final_path_displacement_max_m": (
                final_path_displacement_max
            ),
            "checks": checks,
        }
        report["status"] = "passed" if all(checks.values()) else "failed"
        report["failures"] = [name for name, passed in checks.items() if not passed]
        if _input_binding(cfg, candidate_batch_path) != report_inputs:
            raise RuntimeError("reset-alignment inputs changed while the rollout was running")
        return report
    finally:
        env.close()

def _assert_warp_not_imported_before_launch() -> None:
    """Ensure AppLauncher can select Isaac Sim's bundled Warp build."""

    wp = sys.modules.get("warp")
    if wp is None:
        return
    raise RuntimeError(
        "Warp was imported before Isaac Lab AppLauncher. Keep Isaac Lab task and "
        "asset imports after AppLauncher so Isaac Sim can select its bundled Warp "
        f"build (loaded file: {getattr(wp, '__file__', None)!r})."
    )


def _pose_from_mapping(mapping: dict[str, Any]) -> ResetPose:
    return ResetPose(
        gradient=float(mapping["gradient"]),
        q_reset=tuple(float(value) for value in mapping["q_reset"]),
        q_ref_unloaded=tuple(
            float(value) for value in mapping["q_ref_unloaded"]
        ),
        tau_unloaded=tuple(
            float(value) for value in mapping["tau_unloaded"]
        ),
        tau_per_tangent_force=tuple(
            float(value) for value in mapping["tau_per_tangent_force"]
        ),
        tau_per_normal_force=tuple(
            float(value) for value in mapping["tau_per_normal_force"]
        ),
        tau_per_tangent_difference=tuple(
            float(value) for value in mapping["tau_per_tangent_difference"]
        ),
        handle_wrenches_sln=tuple(
            tuple(float(value) for value in row)
            for row in mapping["handle_wrenches_sln"]
        ),
        wheel_contact_forces_sln=tuple(
            tuple(float(value) for value in row)
            for row in mapping["wheel_contact_forces_sln"]
        ),
        q_ref=tuple(float(value) for value in mapping["q_ref"]),
        root_pitch=float(mapping["root_pitch"]),
        root_height=float(mapping["root_height"]),
    )


def _candidate_batches(
    candidate_bank: dict[float, list[dict[str, Any]]],
    candidates_per_slope: int,
) -> list[list[tuple[dict[str, Any], int, int]]]:
    """Pack candidates onto unique equivalent terrain columns."""

    if not 1 <= candidates_per_slope <= TERRAIN_COLUMNS_PER_TYPE:
        raise ValueError(
            f"candidates_per_slope must lie in [1, {TERRAIN_COLUMNS_PER_TYPE}]"
        )
    rounds = max(
        math.ceil(len(records) / candidates_per_slope)
        for records in candidate_bank.values()
    )
    batches: list[list[tuple[dict[str, Any], int, int]]] = []
    for round_index in range(rounds):
        batch: list[tuple[dict[str, Any], int, int]] = []
        begin = round_index * candidates_per_slope
        end = begin + candidates_per_slope
        for slope in SLOPE_GRADIENTS:
            level, base_column = terrain_index_for_gradient(slope)
            for lane, record in enumerate(candidate_bank[float(slope)][begin:end]):
                batch.append((record, level, base_column + lane))
        batches.append(batch)
    return batches


def _candidate_batch_mapping(
    entries: list[tuple[dict[str, Any], int, int]],
) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_BATCH_SCHEMA_VERSION,
        "candidates": [
            {
                "slope": float(source["pose"]["gradient"]),
                "candidate_id": int(source["candidate_id"]),
                "terrain_level": int(level),
                "terrain_column": int(column),
                "pose": source["pose"],
            }
            for source, level, column in entries
        ],
    }


def _load_candidate_batch(path: Path) -> list[dict[str, Any]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if set(mapping) != {"schema_version", "candidates"}:
        raise RuntimeError("candidate batch uses an invalid top-level schema")
    if mapping["schema_version"] != CANDIDATE_BATCH_SCHEMA_VERSION:
        raise RuntimeError("candidate batch schema_version is unsupported")
    rows = mapping["candidates"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("candidate batch must contain at least one candidate")
    result: list[dict[str, Any]] = []
    identities: set[tuple[float, int]] = set()
    terrain_tiles: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "slope",
            "candidate_id",
            "terrain_level",
            "terrain_column",
            "pose",
        }:
            raise RuntimeError("candidate batch row uses an invalid schema")
        slope = float(row["slope"])
        candidate_id = row["candidate_id"]
        level = row["terrain_level"]
        column = row["terrain_column"]
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise RuntimeError("candidate batch candidate_id must be an integer")
        if isinstance(level, bool) or not isinstance(level, int):
            raise RuntimeError("candidate batch terrain_level must be an integer")
        if isinstance(column, bool) or not isinstance(column, int):
            raise RuntimeError("candidate batch terrain_column must be an integer")
        expected_level, base_column = terrain_index_for_gradient(slope)
        if level != expected_level or not (
            base_column <= column < base_column + TERRAIN_COLUMNS_PER_TYPE
        ):
            raise RuntimeError("candidate batch terrain tile does not match its slope")
        pose = _pose_from_mapping(row["pose"])
        if not math.isclose(pose.gradient, slope, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("candidate batch pose gradient does not match its slope")
        identity = (slope, candidate_id)
        terrain_tile = (level, column)
        if identity in identities:
            raise RuntimeError("candidate batch contains a duplicate candidate")
        if terrain_tile in terrain_tiles:
            raise RuntimeError("candidate batch assigns multiple candidates to one terrain tile")
        identities.add(identity)
        terrain_tiles.add(terrain_tile)
        result.append(
            {
                "slope": slope,
                "candidate_id": candidate_id,
                "terrain_level": level,
                "terrain_column": column,
                "pose": pose,
            }
        )
    return result


def _survival_steps(rollout: dict[str, Any], maximum_steps: int) -> tuple[int, int | None]:
    failure_steps = [int(value) for value in rollout["termination_steps"]]
    first_fall = int(rollout["first_fall_step"])
    if first_fall > 0:
        failure_steps.append(first_fall)
    if not failure_steps:
        return maximum_steps, None
    failure_step = min(failure_steps)
    return max(0, failure_step - 1), failure_step


def _candidate_score(
    initial: dict[str, Any],
    rollout: dict[str, Any],
    thresholds: dict[str, float],
    hitch_height_target: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    survival, failure_step = _survival_steps(rollout, args.steps)
    initial_dex = min(float(value) for value in initial["dex_forward_dot_path_tangent"])
    initial_lateral = max(
        abs(float(value)) for value in initial["dex_forward_dot_path_lateral"]
    )
    final_dex = min(float(value) for value in rollout["final_dex_forward_dot_path_tangent"])
    final_grasp = max(
        float(value) for value in rollout["final_grasp_hitch_position_error_m"]
    )
    checks = {
        "no_fall_or_termination": failure_step is None,
        "initial_d6_preload_le_1mm": max(
            float(value) for value in initial["d6_preload_position_error_m"]
        ) <= 1.0e-3,
        "initial_hitch_frame_le_0p1mm": max(
            float(value) for value in initial["hitch_local_frame_error_m"]
        ) <= 1.0e-4,
        "initial_hand_height_le_1mm": max(
            abs(float(value) - hitch_height_target)
            for value in initial["hand_normal_height_m"]
        ) <= 1.0e-3,
        "initial_robot_forward_plus_x": float(
            initial["robot_forward_dot_path_tangent"]
        ) >= 0.9999,
        "initial_cart_forward_plus_x": float(
            initial["cart_projected_heading_dot_path_tangent"]
        ) >= 0.9999,
        "initial_robot_ahead": float(initial["robot_ahead_of_cart_m"]) > 0.0,
        "initial_dex_plus_x": (
            initial_dex >= DEX_FORWARD_MIN_DOT
            and initial_lateral <= DEX_FORWARD_INITIAL_MAX_LATERAL
        ),
        "d6_residual": float(rollout["max_d6_residual_m_or_rad"])
        <= float(thresholds["d6_residual_m_or_rad"]),
        "d6_impulse": float(rollout["max_d6_impulse_n_s"])
        <= float(thresholds["d6_impulse_n_s"]),
        "arm_torque": float(rollout["max_arm_torque_ratio"])
        <= RESET_TORQUE_LIMIT_FRACTION,
        "waist_torque": float(rollout["max_waist_torque_ratio"])
        <= RESET_TORQUE_LIMIT_FRACTION,
        "lower_torque": float(rollout["maximum_lower_torque_ratio"])
        <= RESET_TORQUE_LIMIT_FRACTION,
        "root_height": float(rollout["minimum_root_normal_height_m"]) >= 0.60,
        "torso_pitch": float(rollout["maximum_abs_torso_pitch_rad"]) <= 0.45,
        "joint_error": float(rollout["maximum_policy_joint_error_rad"]) <= 0.35,
        "path_displacement": abs(float(rollout["final_root_path_displacement_m"]))
        <= args.stable_displacement_limit,
        "rollout_dex_plus_x": (
            float(rollout["minimum_dex_forward_dot_path_tangent"])
            >= DEX_FORWARD_MIN_DOT
            and float(rollout["maximum_abs_dex_forward_dot_path_lateral"])
            <= DEX_FORWARD_ROLLOUT_MAX_LATERAL
            and final_dex >= DEX_FORWARD_MIN_DOT
        ),
        "final_fat_wrench_consistent": bool(rollout["final_fat_wrench_consistent"]),
    }
    risks = {
        "initial_hand_height": max(
            abs(float(value) - hitch_height_target)
            for value in initial["hand_normal_height_m"]
        ) / 1.0e-3,
        "d6_residual": float(rollout["max_d6_residual_m_or_rad"])
        / float(thresholds["d6_residual_m_or_rad"]),
        "d6_impulse": float(rollout["max_d6_impulse_n_s"])
        / float(thresholds["d6_impulse_n_s"]),
        "arm_torque": float(rollout["max_arm_torque_ratio"])
        / RESET_TORQUE_LIMIT_FRACTION,
        "waist_torque": float(rollout["max_waist_torque_ratio"])
        / RESET_TORQUE_LIMIT_FRACTION,
        "lower_torque": float(rollout["maximum_lower_torque_ratio"])
        / RESET_TORQUE_LIMIT_FRACTION,
        "torso_pitch": float(rollout["maximum_abs_torso_pitch_rad"]) / 0.45,
        "joint_error": float(rollout["maximum_policy_joint_error_rad"]) / 0.35,
        "path_displacement": abs(float(rollout["final_root_path_displacement_m"]))
        / args.stable_displacement_limit,
        "dex_pitch": max(0.0, 1.0 - float(
            rollout["minimum_dex_forward_dot_path_tangent"]
        )) / (1.0 - DEX_FORWARD_MIN_DOT),
        "dex_lateral": float(rollout["maximum_abs_dex_forward_dot_path_lateral"])
        / DEX_FORWARD_ROLLOUT_MAX_LATERAL,
        "final_grasp": final_grasp / 1.0e-3,
    }
    finite_risks = [value for value in risks.values() if math.isfinite(value)]
    worst_risk = max(finite_risks, default=math.inf)
    mean_risk = sum(finite_risks) / len(finite_risks) if finite_risks else math.inf
    return {
        "survival_steps": survival,
        "failure_step": failure_step,
        "dynamic_checks": checks,
        "dynamic_checks_passed": sum(bool(value) for value in checks.values()),
        "dynamic_checks_total": len(checks),
        "all_dynamic_checks_passed": all(checks.values()),
        "normalized_risks": risks,
        "worst_normalized_risk": worst_risk,
        "mean_normalized_risk": mean_risk,
    }


def _selection_key(record: dict[str, Any]) -> tuple[Any, ...]:
    score = record["score"]
    static = record["static_metrics"]
    return (
        -int(score["survival_steps"]),
        -int(score["dynamic_checks_passed"]),
        float(score["worst_normalized_risk"]),
        float(score["mean_normalized_risk"]),
        float(static["fat2_error"]),
        -float(static["zmp_margin"]),
        int(record["candidate_id"]),
    )


def _candidate_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Record the solver arguments needed to safely reuse Stage A candidates."""

    excluded_arguments = {
        "alignment_output",
        "_pipeline_worker",
        "anim_recording_enabled",
        "anim_recording_start_time",
        "anim_recording_stop_time",
        "candidate_output",
        "device",
        "device_explicit",
        "enable_cameras",
        "experience",
        "foot_damping",
        "headless",
        "info",
        "kit_args",
        "leg_damping",
        "livestream",
        "output",
        "rendering_mode",
        "report_output",
        "reuse_candidates",
        "stable_displacement_limit",
        "static_only",
        "stage_b_candidates_per_slope",
        "steps",
        "summary_output",
        "timeseries_stride",
        "validate_existing",
        "verbose",
        "xr",
        # This changes only the soft feasibility reserve. Cached candidates have
        # already passed the unchanged hard ZMP threshold and remain reusable.
        "zmp_optimization_reserve_fraction",
    }

    def json_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return value

    arguments = {
        name: json_value(value)
        for name, value in sorted(vars(args).items())
        if name not in excluded_arguments
    }
    return {
        "arguments": arguments,
        "configured_slopes": [float(value) for value in SLOPE_GRADIENTS],
        "stage_a_solve_plan": [
            {"gradient": float(gradient), "parent": parent}
            for gradient, parent in _stage_a_solve_plan()
        ],
    }


def _candidate_output_mapping(
    diagnostics: list[dict[str, Any]],
    candidate_bank: dict[float, list[dict[str, Any]]],
    attempted: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    diagnostics_by_slope = {float(row["gradient"]): row for row in diagnostics}
    completed_slopes = tuple(
        float(slope)
        for slope in SLOPE_GRADIENTS
        if candidate_bank.get(float(slope)) and float(slope) in diagnostics_by_slope
    )
    return {
        "schema_version": CANDIDATE_PROGRESS_SCHEMA_VERSION,
        "candidate_contract": _candidate_contract(args),
        "attempted_least_squares_starts_per_slope": attempted,
        "complete": len(completed_slopes) == len(SLOPE_GRADIENTS),
        "slopes": [
            {
                "slope": float(slope),
                "eligible_count": len(candidate_bank[slope]),
                "rejected_count": attempted - len(candidate_bank[slope]),
                "candidates": candidate_bank[slope],
                "stage_a_diagnostics": diagnostics_by_slope[slope],
            }
            for slope in completed_slopes
        ],
    }


def _write_candidate_progress(
    args: argparse.Namespace,
    diagnostics: list[dict[str, Any]],
    candidate_bank: dict[float, list[dict[str, Any]]],
) -> None:
    """Persist Stage A progress to staging and the parent-authorized public cache."""

    mapping = _candidate_output_mapping(
        diagnostics,
        candidate_bank,
        args.full_pose_multistarts,
        args,
    )
    _write_json_yaml(args.candidate_output, mapping)
    if not args._pipeline_worker:
        return
    progress_value = os.environ.get(PIPELINE_WORKER_PROGRESS_ENV)
    if not progress_value:
        raise RuntimeError("internal reset worker has no public progress path")
    progress_path = Path(progress_value).resolve()
    if progress_path != args.candidate_output.resolve():
        _write_json_yaml(progress_path, mapping)


def _load_candidate_progress(
    path: Path,
    attempted: int,
    root_height_target: float,
    args: argparse.Namespace,
) -> tuple[
    dict[float, ResetPose],
    list[dict[str, Any]],
    dict[float, list[dict[str, Any]]],
]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate output does not exist: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema_version") != CANDIDATE_PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("candidate output uses an unsupported schema; rerun Stage A")
    stored_contract = mapping.get("candidate_contract")
    current_contract = _candidate_contract(args)
    if stored_contract != current_contract:
        raise RuntimeError(
            "candidate output does not match the current solver arguments"
        )
    if int(mapping.get("attempted_least_squares_starts_per_slope", -1)) != attempted:
        raise RuntimeError(
            f"candidate output was not generated with {attempted} starts per slope"
        )
    rows = mapping.get("slopes")
    if not isinstance(rows, list):
        raise RuntimeError("candidate output has no slope rows")
    by_slope = {float(row["slope"]): row for row in rows}
    configured_slopes = set(float(value) for value in SLOPE_GRADIENTS)
    if len(by_slope) != len(rows) or not set(by_slope).issubset(configured_slopes):
        raise RuntimeError(
            "candidate output contains duplicate or unconfigured slope rows"
        )
    candidate_bank: dict[float, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []
    baseline_poses: dict[float, ResetPose] = {}
    for slope in SLOPE_GRADIENTS:
        if float(slope) not in by_slope:
            continue
        row = by_slope[float(slope)]
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"candidate output has no eligible pose for slope {slope:+.2f}")
        for candidate in candidates:
            pose_gradient = float(candidate["pose"]["gradient"])
            if not math.isclose(pose_gradient, float(slope), abs_tol=1.0e-12):
                raise RuntimeError(
                    f"candidate pose gradient {pose_gradient} is stored under slope {slope}"
                )
            if float(candidate["static_metrics"]["violation"]) > 0.0:
                raise RuntimeError(
                    f"candidate {candidate['candidate_id']} for slope {slope} is not feasible"
                )
        candidate_bank[float(slope)] = candidates
        diagnostics.append(row["stage_a_diagnostics"])
        baseline = min(
            candidates,
            key=lambda candidate: _candidate_rank_key(
                candidate["static_metrics"], root_height_target
            ),
        )
        baseline_poses[float(slope)] = _pose_from_mapping(baseline["pose"])
    return baseline_poses, diagnostics, candidate_bank


def _load_candidate_output(
    path: Path,
    attempted: int,
    root_height_target: float,
    args: argparse.Namespace,
) -> tuple[ResetPoseLibrary, list[dict[str, Any]], dict[float, list[dict[str, Any]]]]:
    baseline_poses, diagnostics, candidate_bank = _load_candidate_progress(
        path, attempted, root_height_target, args
    )
    missing = [
        float(slope) for slope in SLOPE_GRADIENTS if float(slope) not in baseline_poses
    ]
    if missing:
        raise RuntimeError(
            "candidate output is partial; resume Stage A for slopes: "
            + ", ".join(f"{slope:+.2f}" for slope in missing)
        )
    return (
        ResetPoseLibrary(
            poses=tuple(baseline_poses[float(slope)] for slope in SLOPE_GRADIENTS)
        ),
        diagnostics,
        candidate_bank,
    )


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Reset pose search report",
        "",
        f"- Status: **{report['status']}**",
        f"- Least-squares starts: {report['attempted_per_slope']} per slope",
        f"- Isaac Lab horizon: {report['steps']} policy steps",
        "- Ranking: survival, passed dynamic checks, worst normalized risk, mean normalized risk",
        f"- Final library: `{report['final_library']}`",
        "",
        "| Slope | Eligible | Winner | Survival | Checks | Worst risk | Arm | Lower | D6 residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["winners"]:
        rollout = row["rollout"]
        score = row["score"]
        lines.append(
            "| {slope:+.2f} | {eligible}/50 | {candidate} | {survival}/{steps} | "
            "{passed}/{total} | {risk:.3f} | {arm:.3f} | {lower:.3f} | {d6:.4g} |".format(
                slope=row["slope"],
                eligible=row["eligible_count"],
                candidate=row["candidate_id"],
                survival=score["survival_steps"],
                steps=report["steps"],
                passed=score["dynamic_checks_passed"],
                total=score["dynamic_checks_total"],
                risk=score["worst_normalized_risk"],
                arm=rollout["max_arm_torque_ratio"],
                lower=rollout["maximum_lower_torque_ratio"],
                d6=rollout["max_d6_residual_m_or_rad"],
            )
        )
    lines.extend(
        [
            "",
            "A survival value equal to the horizon means no fall, termination, or truncation was observed.",
            "A normalized risk of 1.0 is the corresponding configured limit.",
            "",
        ]
    )
    return "\n".join(lines)


def _retarget_validation_report(
    report: dict[str, Any], reset_pose_path: Path
) -> dict[str, Any]:
    """Bind a successfully validated temporary library to its published path."""

    result = dict(report)
    inputs = dict(result["inputs"])
    inputs["reset_pose_path"] = str(reset_pose_path.resolve())
    result["inputs"] = inputs
    return result


def _assembled_validation_report_errors(
    report: dict[str, Any],
    reset_pose_path: Path,
    steps: int,
    *,
    require_passed: bool = True,
) -> list[str]:
    """Independently verify the final worker's certification binding."""

    errors: list[str] = []
    if report.get("schema_version") != 2:
        errors.append("assembled validation report schema_version is not 2")
    if report.get("tool") != "validate_reset_alignment":
        errors.append("assembled validation report tool is invalid")
    if require_passed and report.get("status") != "passed":
        errors.append("assembled validation report status is not passed")
    elif report.get("status") not in {"passed", "failed"}:
        errors.append("assembled validation report status is invalid")
    if report.get("steps") != steps:
        errors.append("assembled validation report horizon does not match --steps")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("assembled validation report inputs are missing")
        return errors

    expected_path = reset_pose_path.resolve()
    reported_path = inputs.get("reset_pose_path")
    try:
        path_matches = Path(reported_path).resolve() == expected_path
    except TypeError:
        path_matches = False
    if not path_matches:
        errors.append("assembled validation report is bound to a different reset library")

    return errors


def _assembled_validation_command(
    args: argparse.Namespace,
    reset_pose_path: Path,
    alignment_output: Path,
    failure_output: Path,
) -> list[str]:
    """Build the isolated final-validation invocation of this same pipeline."""

    staging_dir = alignment_output.parent
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--validate-existing",
        str(reset_pose_path.resolve()),
        "--steps",
        str(args.steps),
        "--alignment-output",
        str(alignment_output.resolve()),
        "--report-output",
        str(failure_output.resolve()),
        "--output",
        str((staging_dir / "unused_output.yaml").resolve()),
        "--candidate-output",
        str((staging_dir / "unused_candidates.json").resolve()),
        "--summary-output",
        str((staging_dir / "unused_summary.md").resolve()),
        "--seed",
        str(args.seed),
        "--timeseries-stride",
        str(args.timeseries_stride),
        "--stable-displacement-limit",
        str(args.stable_displacement_limit),
        "--device",
        str(args.device),
    ]
    for name in ("foot_stiffness", "foot_damping", "leg_stiffness", "leg_damping"):
        value = getattr(args, name)
        if value is not None:
            command.extend((f"--{name.replace('_', '-')}", str(value)))
    if args.headless:
        command.append("--headless")
    return command


def _run_candidate_validation_process(
    args: argparse.Namespace,
    reset_pose_path: Path,
    candidate_batch_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one candidate batch in a disposable Isaac Sim process."""

    passed_report_path = output_dir / "alignment.json"
    failed_report_path = output_dir / "failed.json"
    expected_candidates = _load_candidate_batch(candidate_batch_path)
    environment = os.environ.copy()
    environment[CANDIDATE_BATCH_ENV] = str(candidate_batch_path.resolve())
    result = subprocess.run(
        _assembled_validation_command(
            args,
            reset_pose_path,
            passed_report_path,
            failed_report_path,
        ),
        check=False,
        env=environment,
    )
    passed = passed_report_path.is_file()
    failed = failed_report_path.is_file()
    if passed == failed:
        raise RuntimeError(
            f"candidate rollout worker exited with code {result.returncode} "
            "without exactly one validation report"
        )
    report_path = passed_report_path if passed else failed_report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_returncode = 0 if passed else 1
    expected_status = "passed" if passed else "failed"
    if (
        result.returncode != expected_returncode
        or report.get("status") != expected_status
    ):
        raise RuntimeError(
            "candidate rollout worker exit code, artifact, and status disagree"
        )
    binding_errors = _assembled_validation_report_errors(
        report,
        reset_pose_path,
        args.steps,
        require_passed=False,
    )
    if binding_errors:
        raise RuntimeError("; ".join(binding_errors))
    inputs = report.get("inputs", {})
    if inputs.get("candidate_batch_path") != str(candidate_batch_path.resolve()):
        raise RuntimeError("candidate rollout report is bound to a different candidate batch")
    expected_bindings = [
        {
            "environment_index": index,
            "slope": row["slope"],
            "candidate_id": row["candidate_id"],
            "terrain_level": row["terrain_level"],
            "terrain_column": row["terrain_column"],
        }
        for index, row in enumerate(expected_candidates)
    ]
    if (
        tuple(float(value) for value in report.get("slopes", ()))
        != tuple(row["slope"] for row in expected_candidates)
        or len(report.get("initial", ())) != len(expected_candidates)
        or len(report.get("rollout", ())) != len(expected_candidates)
        or report.get("candidate_bindings") != expected_bindings
    ):
        raise RuntimeError("candidate rollout report slope rows are invalid")
    return report


def _run_isolated_candidate_rollouts(
    args: argparse.Namespace,
    baseline_library: ResetPoseLibrary,
    candidate_bank: dict[float, list[dict[str, Any]]],
) -> int:
    """Evaluate each candidate batch in a fresh Isaac Sim process."""

    evaluated: dict[float, list[dict[str, Any]]] = {
        float(slope): [] for slope in SLOPE_GRADIENTS
    }
    candidates_per_slope = int(
        getattr(
            args,
            "stage_b_candidates_per_slope",
            DEFAULT_STAGE_B_CANDIDATES_PER_SLOPE,
        )
    )
    batches = _candidate_batches(candidate_bank, candidates_per_slope)
    with tempfile.TemporaryDirectory(
        prefix="reset-pose-search-", dir=Path(args.output).resolve().parent
    ) as temporary:
        temporary_path = Path(temporary)
        baseline_path = temporary_path / "baseline_reset_poses.json"
        _write_json_yaml(baseline_path, baseline_library.to_mapping())
        for batch_index, entries in enumerate(batches):
            round_dir = temporary_path / f"batch-{batch_index + 1:02d}"
            round_dir.mkdir()
            candidate_batch_path = round_dir / "candidate_batch.json"
            _write_json_yaml(
                candidate_batch_path, _candidate_batch_mapping(entries)
            )
            print(
                f"Stage B: rollout batch {batch_index + 1}/{len(batches)} "
                f"({len(entries)} candidates, up to {candidates_per_slope} per slope, "
                "fresh Isaac Sim process)",
                flush=True,
            )
            started = time.monotonic()
            round_report = _run_candidate_validation_process(
                args, baseline_path, candidate_batch_path, round_dir
            )
            print(
                f"Stage B: rollout batch {batch_index + 1}/{len(batches)} completed "
                f"in {time.monotonic() - started:.1f}s "
                f"(status={round_report['status']})",
                flush=True,
            )
            thresholds = round_report["safety_thresholds"]
            hitch_height_target = float(
                round_report["rickshaw_pose_contract"]["hitch_height_target_m"]
            )
            for env_index, (source, _level, _column) in enumerate(entries):
                slope = float(source["pose"]["gradient"])
                initial = round_report["initial"][env_index]
                rollout = round_report["rollout"][env_index]
                evaluated[slope].append(
                    {
                        "slope": float(slope),
                        "candidate_id": int(source["candidate_id"]),
                        "static_metrics": source["static_metrics"],
                        "pose": source["pose"],
                        "initial": initial,
                        "rollout": rollout,
                        "score": _candidate_score(
                            initial,
                            rollout,
                            thresholds,
                            hitch_height_target,
                            args,
                        ),
                    }
                )

    winners: list[dict[str, Any]] = []
    final_poses: list[ResetPose] = []
    for slope in SLOPE_GRADIENTS:
        records = evaluated[float(slope)]
        winner = min(records, key=_selection_key)
        winner = {**winner, "eligible_count": len(records)}
        winners.append(winner)
        final_poses.append(_pose_from_mapping(winner["pose"]))
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection_passed = all(
        row["score"]["survival_steps"] == args.steps
        and row["score"]["all_dynamic_checks_passed"]
        for row in winners
    )
    if selection_passed:
        _write_json_yaml(
            output_path,
            ResetPoseLibrary(poses=tuple(final_poses)).to_mapping(),
        )

    report = {
        "schema_version": 1,
        "tool": "generate_and_select_reset_poses",
        "status": "candidate_passed" if selection_passed else "failed",
        "attempted_per_slope": args.full_pose_multistarts,
        "steps": args.steps,
        "selection_order": [
            "maximum survival_steps",
            "maximum dynamic_checks_passed",
            "minimum worst_normalized_risk",
            "minimum mean_normalized_risk",
            "minimum static FAT2 error",
            "maximum static ZMP margin",
            "minimum candidate_id",
        ],
        "final_library": str(output_path) if selection_passed else None,
        "intended_final_library": str(output_path),
        "alignment_evidence": None,
        "final_validation": None,
        "candidate_evidence": str(args.candidate_output.resolve()),
        "stage_b_candidates_per_slope": candidates_per_slope,
        "stage_b_batch_count": len(batches),
        "winners": winners,
        "all_candidates": [
            record
            for slope in SLOPE_GRADIENTS
            for record in evaluated[float(slope)]
        ],
    }
    _write_json_yaml(args.report_output, report)
    _write_text_atomic(args.summary_output, _markdown_summary(report))
    print(f"JSON evidence: {args.report_output.resolve()}")
    print(f"summary: {args.summary_output.resolve()}")
    if not selection_passed:
        raise RuntimeError("no winner passed every candidate-level dynamic gate")
    print(f"staged reset library: {output_path}")
    return 0


def _run_pipeline_parent(args: argparse.Namespace) -> int:
    """Orchestrate isolated candidate and final-validation workers."""

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_publications: list[tuple[Path, Path]] = []
    prepared_library: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="reset-pipeline-", dir=output_path.parent
    ) as staging:
        staging_dir = Path(staging)
        staged_library = staging_dir / "reset_poses.yaml"
        staged_candidates = staging_dir / "candidates.json"
        staged_search_report = staging_dir / "search_report.json"
        staged_search_summary = staging_dir / "search_summary.md"
        staged_alignment = staging_dir / "alignment.json"
        staged_failure = staging_dir / "alignment_failed.json"
        staged_unused_alignment = staging_dir / "worker_alignment.json"

        if args.reuse_candidates:
            shutil.copy2(args.candidate_output, staged_candidates)
        worker_environment = os.environ.copy()
        worker_environment[PIPELINE_WORKER_PROGRESS_ENV] = str(
            args.candidate_output.resolve()
        )

        worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--_pipeline-worker",
            "--output",
            str(staged_library),
            "--candidate-output",
            str(staged_candidates),
            "--report-output",
            str(staged_search_report),
            "--summary-output",
            str(staged_search_summary),
            "--alignment-output",
            str(staged_unused_alignment),
        ]
        worker = subprocess.run(
            worker_command, check=False, env=worker_environment
        )
        if staged_search_report.is_file():
            search_report = json.loads(
                staged_search_report.read_text(encoding="utf-8")
            )
        else:
            search_report = {
                "schema_version": 1,
                "tool": "generate_and_select_reset_poses",
                "status": "failed",
                "attempted_per_slope": args.full_pose_multistarts,
                "steps": args.steps,
                "final_library": None,
                "winners": [],
                "failures": [
                    "candidate worker exited without a search report: "
                    f"{worker.returncode}"
                ],
            }
        search_report["candidate_evidence"] = str(args.candidate_output.resolve())
        worker_passed = (
            worker.returncode == 0
            and staged_library.is_file()
            and search_report.get("status") == "candidate_passed"
        )
        if not worker_passed:
            failures = list(search_report.get("failures", []))
            failures.insert(
                0, f"reset candidate worker failed with exit code {worker.returncode}"
            )
            search_report.update(
                {
                    "status": "failed",
                    "failures": failures,
                    "final_library": None,
                    "intended_final_library": str(output_path),
                    "alignment_evidence": None,
                }
            )
            completed_slopes: list[float] = []
            if staged_candidates.is_file():
                _publish_atomic_copy(staged_candidates, args.candidate_output)
                try:
                    candidate_mapping = json.loads(
                        staged_candidates.read_text(encoding="utf-8")
                    )
                    completed_slopes = [
                        float(row["slope"])
                        for row in candidate_mapping.get("slopes", ())
                    ]
                except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                    completed_slopes = []
            search_report["stage_a_completed_slopes"] = completed_slopes
            search_report["stage_a_completed_count"] = len(completed_slopes)
            search_report["stage_a_total_count"] = len(SLOPE_GRADIENTS)
            _write_json_yaml(args.report_output, search_report)
            _write_text_atomic(args.summary_output, _markdown_summary(search_report))
            if completed_slopes:
                raise RuntimeError(
                    f"reset candidate worker failed after saving {len(completed_slopes)}/"
                    f"{len(SLOPE_GRADIENTS)} Stage A slopes to "
                    f"{args.candidate_output}; rerun with --reuse-candidates"
                )
            raise RuntimeError(
                "reset candidate worker did not produce a publishable candidate bank"
            )

        validation = subprocess.run(
            _assembled_validation_command(
                args, staged_library, staged_alignment, staged_failure
            ),
            check=False,
        )
        validation_path = staged_alignment if staged_alignment.is_file() else staged_failure
        if validation_path.is_file():
            final_validation = json.loads(validation_path.read_text(encoding="utf-8"))
        else:
            final_validation = {
                "status": "failed",
                "failures": [
                    "assembled validation worker exited without a report: "
                    f"{validation.returncode}"
                ],
            }
        validation_errors = _assembled_validation_report_errors(
            final_validation, staged_library, args.steps
        )
        certified = validation.returncode == 0 and not validation_errors
        if validation_errors:
            final_validation = dict(final_validation)
            final_validation["status"] = "failed"
            final_validation["failures"] = [
                *list(final_validation.get("failures", [])),
                *validation_errors,
            ]
        report = dict(search_report)
        report["final_validation"] = final_validation
        report["intended_final_library"] = str(output_path)
        report["final_library"] = str(output_path) if certified else None
        report["alignment_evidence"] = (
            str(args.alignment_output.resolve()) if certified else None
        )
        report["status"] = "passed" if certified else "failed"

        if certified:
            final_validation = _retarget_validation_report(final_validation, output_path)
            report["final_validation"] = final_validation
        else:
            failures = list(final_validation.get("failures", []))
            failures.insert(0, "assembled reset validation did not pass")
            report["failures"] = failures
            if staged_candidates.is_file():
                _publish_atomic_copy(staged_candidates, args.candidate_output)
            _write_json_yaml(args.report_output, report)
            _write_text_atomic(args.summary_output, _markdown_summary(report))
            raise RuntimeError(
                "reset library was not published: "
                + ", ".join(report["failures"])
            )

        try:
            prepared_publications.append(
                (
                    _prepare_atomic_copy(staged_candidates, args.candidate_output),
                    args.candidate_output.resolve(),
                )
            )
            prepared_publications.extend(
                [
                    (
                        _prepare_atomic_text(
                            args.report_output,
                            json.dumps(report, indent=2) + "\n",
                        ),
                        args.report_output.resolve(),
                    ),
                    (
                        _prepare_atomic_text(
                            args.summary_output, _markdown_summary(report)
                        ),
                        args.summary_output.resolve(),
                    ),
                    (
                        _prepare_atomic_text(
                            args.alignment_output,
                            json.dumps(final_validation, indent=2) + "\n",
                        ),
                        args.alignment_output.resolve(),
                    ),
                ]
            )
            prepared_library = _prepare_atomic_copy(staged_library, output_path)
        except BaseException:
            for temporary, _destination in prepared_publications:
                temporary.unlink(missing_ok=True)
            if prepared_library is not None:
                prepared_library.unlink(missing_ok=True)
            raise

    if prepared_library is None:
        raise RuntimeError("reset pipeline did not prepare a certified library")
    success_messages = (
        f"selected reset library: {output_path}",
        f"alignment evidence: {args.alignment_output.resolve()}",
        f"JSON evidence: {args.report_output.resolve()}",
        f"summary: {args.summary_output.resolve()}",
    )
    _commit_pipeline_publications(
        prepared_publications, prepared_library, output_path
    )

    for message in success_messages:
        try:
            print(message)
        except OSError:
            pass
    return 0


def main() -> int:
    parser = _build_parser()
    # Keep pure CLI validation and --help usable without importing the heavy
    # Isaac runtime. AppLauncher-specific options are handled after its parser
    # extension below.
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        parser.parse_args()
    preliminary_args, unknown_args = parser.parse_known_args()
    if not unknown_args:
        _validate_arguments(preliminary_args)
    add_isaaclab_sources_to_path()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _validate_arguments(args)

    if args.static_only:
        if args.validate_existing is not None:
            raise ValueError("--static-only cannot be combined with --validate-existing")
        if args.reuse_candidates:
            completed_poses, _diagnostics, _candidate_bank = _load_candidate_progress(
                args.candidate_output,
                args.full_pose_multistarts,
                args.root_height,
                args,
            )
            print(
                "Stage A diagnostic: resuming "
                f"{len(completed_poses)}/{len(SLOPE_GRADIENTS)} cached slopes",
                flush=True,
            )
            baseline_library, _diagnostics, _candidate_bank = _solve_library(
                args, resume_candidate_output=args.candidate_output
            )
        else:
            print("Stage A diagnostic: solving static candidates only", flush=True)
            baseline_library, _diagnostics, _candidate_bank = _solve_library(args)
        _write_json_yaml(args.output, baseline_library.to_mapping())
        print(f"uncertified static library: {args.output.resolve()}", flush=True)
        return 0

    if not args._pipeline_worker and args.validate_existing is None:
        return _run_pipeline_parent(args)

    if args.validate_existing is None:
        if args.reuse_candidates:
            completed_poses, _diagnostics, _candidate_bank = _load_candidate_progress(
                args.candidate_output,
                args.full_pose_multistarts,
                args.root_height,
                args,
            )
            completed_count = len(completed_poses)
            if completed_count == len(SLOPE_GRADIENTS):
                print(
                    f"Stage A: reusing complete candidates from {args.candidate_output}",
                    flush=True,
                )
                baseline_library, _diagnostics, candidate_bank = _load_candidate_output(
                    args.candidate_output,
                    args.full_pose_multistarts,
                    args.root_height,
                    args,
                )
            else:
                print(
                    "Stage A: resuming "
                    f"{completed_count}/{len(SLOPE_GRADIENTS)} slopes from "
                    f"{args.candidate_output}",
                    flush=True,
                )
                baseline_library, _diagnostics, candidate_bank = _solve_library(
                    args, resume_candidate_output=args.candidate_output
                )
        else:
            print(
                "Stage A: solving "
                f"{RESET_STATIC_MULTISTARTS} starts on each of "
                f"{len(SLOPE_GRADIENTS)} slopes",
                flush=True,
            )
            baseline_library, _diagnostics, candidate_bank = _solve_library(args)
            missing = [
                slope for slope in SLOPE_GRADIENTS if not candidate_bank.get(slope)
            ]
            if missing:
                raise RuntimeError(
                    f"no statically feasible candidates for slopes: {missing}"
                )
        return _run_isolated_candidate_rollouts(
            args, baseline_library, candidate_bank
        )

    print(
        f"Stage A: validating existing library {args.validate_existing}",
        flush=True,
    )

    print("Stage B: launching Isaac Sim", flush=True)
    _assert_warp_not_imported_before_launch()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        print("Stage B: Isaac Sim startup completed", flush=True)
        print("Stage B: loading simulation dependencies", flush=True)
        _load_simulation_dependencies()
        print("Stage B: simulation dependencies loaded", flush=True)

        validation = _validate_round(args, args.validate_existing)
        if validation["status"] != "passed":
            _write_json_yaml(args.report_output, validation)
            print(
                "assembled reset validation failed: "
                + ", ".join(validation["failures"]),
                flush=True,
            )
            print(f"failure report: {args.report_output.resolve()}", flush=True)
            os._exit(1)
        _write_json_yaml(args.alignment_output, validation)
        print(
            f"assembled reset validation: {args.alignment_output.resolve()}",
            flush=True,
        )
        simulation_app.close()
        return 0
    except BaseException:
        # Isaac Sim 5.1 may terminate the process from simulation_app.close().
        # Emit the original exception before any Kit shutdown can mask it.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    raise SystemExit(exit_code)
