#!/usr/bin/env python3
"""Run mandatory coast-down and measured/analytic cart-wrench checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys
import traceback
from types import MethodType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_PATH = Path(
    os.environ.get("ISAACLAB_PATH", REPOSITORY_ROOT.parent / "IsaacLab")
).resolve()
for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    package_path = ISAACLAB_PATH / "source" / package_name
    if package_path.is_dir() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
PROJECT_SOURCE = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", default="Isaac-G1-Rickshaw-Directional-Slope-Play-v0"
)
parser.add_argument(
    "--feasibility", type=Path, default=REPOSITORY_ROOT / "config/feasibility_envelope.yaml"
)
parser.add_argument(
    "--reset-poses", type=Path, default=REPOSITORY_ROOT / "config/reset_poses.yaml"
)
parser.add_argument(
    "--output",
    type=Path,
    default=REPOSITORY_ROOT / "outputs/validation/dynamics_report.json",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--settling-steps", type=int, default=40)
parser.add_argument("--measurement-steps", type=int, default=30)
parser.add_argument("--window-start", type=int, default=5)
parser.add_argument("--coast-speed", type=float, default=0.8)
parser.add_argument("--coast-relative-tolerance", type=float, default=0.20)
parser.add_argument(
    "--wrench-relative-tolerance",
    type=float,
    default=None,
    help="Must match the independent normative measured/analytic dynamics gate.",
)
parser.add_argument(
    "--wrench-absolute-floor-n",
    type=float,
    default=None,
    help="Must match the independent normative measured/analytic dynamics gate.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from g1_rickshaw_lab.assets.rickshaw import (  # noqa: E402
    RICKSHAW_TOTAL_MASS,
    WHEEL_RADIUS,
    build_rickshaw_cfg,
)
from g1_rickshaw_lab.configuration import (  # noqa: E402
    load_feasibility_envelope,
    load_reset_pose_library,
)
from g1_rickshaw_lab.slope_contract import terrain_index_for_gradient  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import env_cfg as task_cfg  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp  # noqa: E402
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.dynamics import GRAVITY  # noqa: E402
from g1_rickshaw_lab.validation import (  # noqa: E402
    MAX_WRENCH_RELATIVE_TOLERANCE,
    WRENCH_ABSOLUTE_FLOOR_N,
    build_report,
    compare_wrench_component,
    evaluate_coast_down,
    validation_input_assets,
    write_json_atomic,
)


NUM_ENVS = 6
CONDITION_IDS = (2, 3, 4, 5)
CONDITION_NAMES = (
    "flat_static",
    "flat_constant_speed",
    "uphill_acceleration",
    "downhill_braking",
)
CONDITION_SLOPES = (0.0, 0.0, 0.06, -0.06)
COAST_LOCAL_POSITION = (40.0, 40.0, 20.0)
COAST_COMPARISON_IDS = (0, 1)


def _create_coast_rails(env, env_ids) -> None:
    """Constrain isolated coast carts to world-X translation."""

    del env_ids
    stage = env.scene.stage
    xform_cache = UsdGeom.XformCache()
    for index in range(env.num_envs):
        base_path = f"/World/envs/env_{index}/CoastCart/base_link"
        base = stage.GetPrimAtPath(base_path)
        if not base.IsValid():
            raise RuntimeError("coast cart base prim is missing during prestartup")
        anchor = xform_cache.GetLocalToWorldTransform(base).ExtractTranslation()
        joint = UsdPhysics.PrismaticJoint.Define(
            stage, f"/World/envs/env_{index}/Constraints/coast_x_rail"
        )
        joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.x)
        joint.CreateLowerLimitAttr().Set(-100.0)
        joint.CreateUpperLimitAttr().Set(100.0)


_COAST_CART_CFG = build_rickshaw_cfg(require_usd=True).replace(
    prim_path="{ENV_REGEX_NS}/CoastCart"
)
_COAST_CART_CFG.init_state.pos = COAST_LOCAL_POSITION


@configclass
class DynamicsValidationSceneCfg(task_cfg.G1RickshawSceneCfg):
    coast_cart = _COAST_CART_CFG


@configclass
class DynamicsValidationEventCfg(task_cfg.EventCfg):
    create_coast_rails = EventTerm(func=_create_coast_rails, mode="prestartup")


@configclass
class DynamicsValidationEnvCfg(task_cfg.G1RickshawDirectionalSlopePlayEnvCfg):
    scene: DynamicsValidationSceneCfg = DynamicsValidationSceneCfg(
        num_envs=NUM_ENVS,
        env_spacing=6.0,
        replicate_physics=False,
    )
    events: DynamicsValidationEventCfg = DynamicsValidationEventCfg()


def _set_slopes(env) -> None:
    slopes = (0.0, 0.0, *CONDITION_SLOPES)
    indices = [terrain_index_for_gradient(slope) for slope in slopes]
    levels = torch.tensor([item[0] for item in indices], device=env.device)
    columns = torch.tensor([item[1] for item in indices], device=env.device)
    terrain = env.scene.terrain
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(columns)
    terrain.env_origins.copy_(terrain.terrain_origins[levels, columns])


def _linear_slope(time: torch.Tensor, value: torch.Tensor) -> float:
    centered = time - torch.mean(time)
    denominator = torch.sum(centered * centered)
    if denominator <= 0.0:
        raise RuntimeError("coast-down window has zero time span")
    return float(torch.sum(centered * (value - torch.mean(value))) / denominator)


def _reset_controlled_velocities(base, speed: torch.Tensor) -> None:
    ids = torch.tensor(CONDITION_IDS, device=base.device, dtype=torch.long)
    tangent = base.path_tangent_w[ids]
    velocity = torch.zeros((len(CONDITION_IDS), 6), device=base.device)
    velocity[:, :3] = tangent * speed[:, None]
    base.scene["robot"].write_root_velocity_to_sim(velocity, env_ids=ids)
    base.scene["rickshaw"].write_root_velocity_to_sim(velocity, env_ids=ids)
    wheel_velocity = -speed[:, None].expand(-1, 2) / WHEEL_RADIUS
    base.scene["rickshaw"].write_joint_state_to_sim(
        base.scene["rickshaw"].data.joint_pos[ids][:, base.wheel_joint_ids],
        wheel_velocity,
        joint_ids=base.wheel_joint_ids,
        env_ids=ids,
    )


def _set_control_forces(base) -> dict[str, float]:
    robot = base.scene["robot"]
    pelvis_ids, _ = robot.find_bodies("pelvis")
    if len(pelvis_ids) != 1:
        raise RuntimeError("dynamics validation requires exactly one pelvis body")
    force_magnitudes = torch.tensor(
        (0.0, 0.0, 80.0, -60.0), device=base.device
    )
    forces = torch.zeros((len(CONDITION_IDS), 1, 3), device=base.device)
    ids = torch.tensor(CONDITION_IDS, device=base.device, dtype=torch.long)
    forces[:, 0] = force_magnitudes[:, None] * base.path_tangent_w[ids]
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces,
        torch.zeros_like(forces),
        body_ids=pelvis_ids,
        env_ids=ids,
        is_global=True,
    )
    return {
        name: float(value)
        for name, value in zip(CONDITION_NAMES, force_magnitudes.tolist(), strict=True)
    }


def _run() -> tuple[dict[str, object], list[str], dict[str, object]]:
    envelope = load_feasibility_envelope(args.feasibility)
    load_reset_pose_library(args.reset_poses)
    calibrated_wrench_tolerance = MAX_WRENCH_RELATIVE_TOLERANCE
    calibrated_wrench_floor = WRENCH_ABSOLUTE_FLOOR_N
    if (
        args.wrench_relative_tolerance is not None
        and args.wrench_relative_tolerance != calibrated_wrench_tolerance
    ):
        raise ValueError(
            "--wrench-relative-tolerance must match the normative dynamics gate"
        )
    if (
        args.wrench_absolute_floor_n is not None
        and args.wrench_absolute_floor_n != calibrated_wrench_floor
    ):
        raise ValueError(
            "--wrench-absolute-floor-n must match the normative dynamics gate"
        )
    wrench_relative_tolerance = calibrated_wrench_tolerance
    wrench_absolute_floor_n = calibrated_wrench_floor
    if args.settling_steps <= 0 or args.measurement_steps <= 0:
        raise ValueError("settling and measurement steps must be positive")
    if not 0 <= args.window_start < args.measurement_steps:
        raise ValueError("window-start must lie inside the measurement window")
    if args.coast_speed <= 3.0 * 0.05:
        raise ValueError("coast speed must be outside the rolling-resistance smoothing region")

    feasibility_path = args.feasibility.resolve()
    reset_pose_path = args.reset_poses.resolve()
    os.environ["G1_RICKSHAW_FEASIBILITY_ENVELOPE"] = os.fspath(feasibility_path)
    os.environ["G1_RICKSHAW_RESET_POSES"] = os.fspath(reset_pose_path)
    cfg = DynamicsValidationEnvCfg()
    if Path(cfg.feasibility_path).resolve() != feasibility_path:
        raise RuntimeError("environment did not load the requested feasibility envelope")
    if Path(cfg.reset_pose_path).resolve() != reset_pose_path:
        raise RuntimeError("environment did not load the requested reset-pose library")
    cfg.scene.num_envs = NUM_ENVS
    cfg.sim.device = args.device
    cfg.curriculum = None
    # Prescribed velocities and pelvis forces intentionally violate policy
    # overspeed/pose envelopes.  This isolated check gates force balance via
    # full-window analytic validity, not policy episode termination.
    cfg.terminations.immediate_safety = None
    cfg.terminations.persistent_safety = None
    # Keep the directional generator structured. Only the level manager is disabled.
    cfg.scene.terrain.terrain_generator.curriculum = True
    runtime_cfg = replace(cfg.runtime_randomization, sample_ranges=False)
    cfg.runtime_randomization = runtime_cfg
    cfg.events.sample_physics.params = {"cfg": runtime_cfg}

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    failures: list[str] = []
    try:
        _set_slopes(base)
        env.reset(seed=args.seed)
        actions = torch.zeros(env.action_space.shape, device=base.device)
        original_pre_physics_step = base._g1_rickshaw_pre_physics_step
        base._diagnostic_d6_substep_wrench_sum = torch.zeros_like(
            base.rickshaw_state.d6_wrench_w
        )
        base._diagnostic_d6_substep_count = 0

        def diagnostic_pre_physics_step(self) -> None:
            original_pre_physics_step()
            wrench, _, _ = self.read_d6_reaction_residual()
            self._diagnostic_d6_substep_wrench_sum += wrench
            self._diagnostic_d6_substep_count += 1

        base._g1_rickshaw_pre_physics_step = MethodType(
            diagnostic_pre_physics_step, base
        )
        with torch.inference_mode():
            for _ in range(args.settling_steps):
                base._diagnostic_d6_substep_wrench_sum.zero_()
                base._diagnostic_d6_substep_count = 0
                _, _, terminated, _, _ = env.step(actions)
                if torch.any(terminated[torch.tensor(CONDITION_IDS, device=base.device)]):
                    failures.append("a D6 condition terminated during settling")
                    break

            coast = base.scene["coast_cart"]
            comparison_ids = torch.tensor(
                COAST_COMPARISON_IDS, device=base.device, dtype=torch.long
            )
            root_velocity = torch.zeros((NUM_ENVS, 6), device=base.device)
            root_velocity[comparison_ids, 0] = args.coast_speed
            coast.write_root_velocity_to_sim(root_velocity)
            wheel_ids, wheel_names = coast.find_joints(
                ".*_wheel_joint", preserve_order=True
            )
            if len(wheel_ids) != 2:
                raise RuntimeError(f"coast cart wheel joints are invalid: {wheel_names}")
            wheel_velocity = torch.zeros((NUM_ENVS, 2), device=base.device)
            wheel_velocity[comparison_ids] = -args.coast_speed / WHEEL_RADIUS
            coast.write_joint_state_to_sim(
                coast.data.joint_pos[:, wheel_ids],
                wheel_velocity,
                joint_ids=wheel_ids,
            )
            coast_wheel_ids, wheel_body_names = coast.find_bodies(
                ".*_wheel_link", preserve_order=True
            )
            if len(coast_wheel_ids) != 2:
                raise RuntimeError(
                    f"coast cart wheel bodies are invalid: {wheel_body_names}"
                )

            condition_speeds = torch.tensor(
                (0.0, 0.2, 0.1, 0.2), device=base.device
            )
            _reset_controlled_velocities(base, condition_speeds)
            applied_forces = _set_control_forces(base)

            maximum_c_rr = envelope.ranges["rolling_resistance.c_rr"].maximum
            c_rr = torch.zeros(NUM_ENVS, device=base.device)
            c_rr[COAST_COMPARISON_IDS[1]] = maximum_c_rr
            wheel_weight = 0.5 * RICKSHAW_TOTAL_MASS * GRAVITY
            filtered_normal = torch.full(
                (NUM_ENVS, 2), wheel_weight, device=base.device
            )
            coast_contact_force = torch.zeros(
                (NUM_ENVS, 2, 3), device=base.device
            )
            coast_contact_force[..., 2] = wheel_weight
            coast_time: list[float] = []
            coast_speed: list[torch.Tensor] = []
            coast_normal: list[torch.Tensor] = []
            analytic_s: list[torch.Tensor] = []
            analytic_n: list[torch.Tensor] = []
            measured_s: list[torch.Tensor] = []
            measured_n: list[torch.Tensor] = []
            policy_sample_measured_s: list[torch.Tensor] = []
            substep_d6_measured_s: list[torch.Tensor] = []
            all_cart_contact_s: list[torch.Tensor] = []
            ground_contact_s: list[torch.Tensor] = []
            wheel_contact_s: list[torch.Tensor] = []
            cart_com_acceleration_s: list[torch.Tensor] = []
            wheel_angular_acceleration: list[torch.Tensor] = []
            wheel_incoming_joint_torque_y: list[torch.Tensor] = []
            rolling_resistance_s: list[torch.Tensor] = []
            hitch_incoming_force_raw: list[torch.Tensor] = []
            hitch_incoming_force_raw_direct_sn: list[torch.Tensor] = []
            hitch_incoming_force_parent_rotated_sn: list[torch.Tensor] = []
            hitch_incoming_force_body_rotated_sn: list[torch.Tensor] = []
            hitch_relative_quaternion_wxyz: list[torch.Tensor] = []
            cart_pitch: list[torch.Tensor] = []
            valid_windows = torch.ones(len(CONDITION_IDS), device=base.device, dtype=torch.bool)
            terminated_conditions = torch.zeros_like(valid_windows)

            controlled_cart = base.scene["rickshaw"]
            controlled_wheel_ids, _ = controlled_cart.find_joints(
                ".*_wheel_joint", preserve_order=True
            )
            controlled_wheel_body_ids, _ = controlled_cart.find_bodies(
                ".*_wheel_link", preserve_order=True
            )
            if len(controlled_wheel_ids) != 2 or len(controlled_wheel_body_ids) != 2:
                raise RuntimeError("controlled rickshaw wheel joints/bodies are invalid")
            body_mass = controlled_cart.data.default_mass.to(
                device=base.device,
                dtype=controlled_cart.data.body_com_vel_w.dtype,
            )
            cart_mass = body_mass.sum(dim=-1)
            cart_com_velocity_w = torch.sum(
                body_mass[..., None] * controlled_cart.data.body_com_vel_w[..., :3],
                dim=1,
            ) / cart_mass[:, None]
            previous_cart_com_velocity_w = cart_com_velocity_w.clone()
            previous_cart_com_velocity_s = torch.sum(
                previous_cart_com_velocity_w * base.path_tangent_w, dim=-1
            )
            previous_wheel_velocity = controlled_cart.data.joint_vel[
                :, controlled_wheel_ids
            ].clone()

            for step in range(args.measurement_steps):
                base._diagnostic_d6_substep_wrench_sum.zero_()
                base._diagnostic_d6_substep_count = 0
                wheel_velocity = coast.data.body_lin_vel_w[:, coast_wheel_ids]
                tangent = torch.tensor(
                    (1.0, 0.0, 0.0), device=base.device
                ).expand(NUM_ENVS, -1)
                normal = torch.tensor(
                    (0.0, 0.0, 1.0), device=base.device
                ).expand(NUM_ENVS, -1)
                force_w, filtered_normal, _ = mdp.rolling_resistance_wrench(
                    wheel_velocity,
                    coast_contact_force,
                    tangent,
                    normal,
                    c_rr,
                    filtered_normal,
                    velocity_epsilon=0.05,
                    normal_force_filter_hz=20.0,
                    dt=base.physics_dt,
                )
                coast.permanent_wrench_composer.set_forces_and_torques(
                    force_w,
                    torch.zeros_like(force_w),
                    body_ids=coast_wheel_ids,
                    is_global=True,
                )
                _, _, terminated, _, _ = env.step(actions)
                cart_com_velocity_w = torch.sum(
                    body_mass[..., None]
                    * controlled_cart.data.body_com_vel_w[..., :3],
                    dim=1,
                ) / cart_mass[:, None]
                current_cart_com_velocity_s = torch.sum(
                    cart_com_velocity_w * base.path_tangent_w, dim=-1
                )
                current_wheel_velocity = controlled_cart.data.joint_vel[
                    :, controlled_wheel_ids
                ]
                condition_terminated = terminated[
                    torch.tensor(CONDITION_IDS, device=base.device)
                ]
                terminated_conditions |= condition_terminated
                if step >= args.window_start:
                    if base._diagnostic_d6_substep_count != cfg.decimation:
                        raise RuntimeError(
                            "D6 diagnostic did not sample every physics substep"
                        )
                    substep_wrench = (
                        base._diagnostic_d6_substep_wrench_sum
                        / base._diagnostic_d6_substep_count
                    )
                    substep_force_on_cart = -torch.sum(
                        substep_wrench[list(CONDITION_IDS), :, :3], dim=1
                    )
                    substep_d6_measured_s.append(
                        torch.sum(
                            substep_force_on_cart
                            * base.path_tangent_w[list(CONDITION_IDS)],
                            dim=-1,
                        )
                    )
                    coast_time.append((step - args.window_start) * base.step_dt)
                    coast_speed.append(
                        coast.data.root_lin_vel_w[comparison_ids, 0].clone()
                    )
                    coast_normal.append(
                        filtered_normal[comparison_ids].sum(dim=-1).clone()
                    )
                    analytic_s.append(base.analytic_force_state.t_s[list(CONDITION_IDS)].clone())
                    analytic_n.append(base.analytic_force_state.t_n[list(CONDITION_IDS)].clone())
                    wheel_contacts_w = base.scene["wheel_contacts"].data.net_forces_w
                    ground_contact_force_w = torch.sum(wheel_contacts_w, dim=1)
                    cart_com_acceleration_w = (
                        cart_com_velocity_w - previous_cart_com_velocity_w
                    ) / base.step_dt
                    controlled_wheel_velocity_w = controlled_cart.data.body_lin_vel_w[
                        :, controlled_wheel_body_ids
                    ]
                    controlled_wheel_velocity_s = torch.sum(
                        controlled_wheel_velocity_w
                        * base.path_tangent_w[:, None, :],
                        dim=-1,
                    )
                    rolling_resistance_s_all = torch.sum(
                        -base.c_rr[:, None]
                        * base.rickshaw_state.wheel_normal_force
                        * torch.tanh(controlled_wheel_velocity_s / 0.05),
                        dim=-1,
                    )
                    gravity_w = torch.tensor(
                        cfg.sim.gravity,
                        device=base.device,
                        dtype=cart_com_acceleration_w.dtype,
                    )
                    momentum_balance_force_on_cart_w = (
                        cart_mass[:, None] * cart_com_acceleration_w
                        - cart_mass[:, None] * gravity_w
                        - ground_contact_force_w
                        - rolling_resistance_s_all[:, None] * base.path_tangent_w
                    )
                    force_on_cart = momentum_balance_force_on_cart_w[
                        list(CONDITION_IDS)
                    ]
                    measured_s.append(
                        torch.sum(
                            force_on_cart
                            * base.path_tangent_w[list(CONDITION_IDS)],
                            dim=-1,
                        )
                    )
                    measured_n.append(
                        torch.sum(
                            force_on_cart
                            * base.path_normal_w[list(CONDITION_IDS)],
                            dim=-1,
                        )
                    )
                    policy_sample_force_on_cart = -base.rickshaw_state.hand_force_w[
                        list(CONDITION_IDS)
                    ]
                    policy_sample_measured_s.append(
                        torch.sum(
                            policy_sample_force_on_cart
                            * base.path_tangent_w[list(CONDITION_IDS)],
                            dim=-1,
                        )
                    )
                    all_cart_contact_s.append(
                        torch.sum(
                            ground_contact_force_w * base.path_tangent_w,
                            dim=-1,
                        )[list(CONDITION_IDS)]
                    )
                    ground_contact_s.append(
                        torch.sum(
                            ground_contact_force_w * base.path_tangent_w,
                            dim=-1,
                        )[list(CONDITION_IDS)]
                    )
                    wheel_contact_s.append(
                        torch.sum(
                            torch.sum(wheel_contacts_w, dim=1)
                            * base.path_tangent_w,
                            dim=-1,
                        )[list(CONDITION_IDS)]
                    )
                    cart_com_acceleration_s.append(
                        (
                            (current_cart_com_velocity_s - previous_cart_com_velocity_s)
                            / base.step_dt
                        )[list(CONDITION_IDS)]
                    )
                    wheel_angular_acceleration.append(
                        (
                            (current_wheel_velocity - previous_wheel_velocity)
                            / base.step_dt
                        )[list(CONDITION_IDS)]
                    )
                    wheel_incoming_joint_torque_y.append(
                        controlled_cart.data.body_incoming_joint_wrench_b[
                            list(CONDITION_IDS)
                        ][:, controlled_wheel_body_ids, 4]
                    )
                    rolling_resistance_s.append(
                        rolling_resistance_s_all[list(CONDITION_IDS)]
                    )
                    incoming_force_raw = controlled_cart.data.body_incoming_joint_wrench_b[
                        :, base.hitch_body_ids, :3
                    ]
                    incoming_force_parent_rotated = mdp.quat_apply_wxyz(
                        controlled_cart.data.body_quat_w[:, :1].expand_as(
                            controlled_cart.data.body_quat_w[:, base.hitch_body_ids]
                        ),
                        incoming_force_raw,
                    )
                    incoming_force_body_rotated = mdp.quat_apply_wxyz(
                        controlled_cart.data.body_quat_w[:, base.hitch_body_ids],
                        incoming_force_raw,
                    )
                    raw_sum = torch.sum(incoming_force_raw, dim=1)
                    parent_rotated_sum = torch.sum(
                        incoming_force_parent_rotated, dim=1
                    )
                    hitch_incoming_force_raw.append(
                        incoming_force_raw[list(CONDITION_IDS)]
                    )
                    hitch_incoming_force_raw_direct_sn.append(
                        torch.stack(
                            (
                                torch.sum(raw_sum * base.path_tangent_w, dim=-1),
                                torch.sum(raw_sum * base.path_normal_w, dim=-1),
                            ),
                            dim=-1,
                        )[list(CONDITION_IDS)]
                    )
                    hitch_incoming_force_parent_rotated_sn.append(
                        torch.stack(
                            (
                                torch.sum(
                                    parent_rotated_sum * base.path_tangent_w,
                                    dim=-1,
                                ),
                                torch.sum(
                                    parent_rotated_sum * base.path_normal_w,
                                    dim=-1,
                                ),
                            ),
                            dim=-1,
                        )[list(CONDITION_IDS)]
                    )
                    body_rotated_sum = torch.sum(incoming_force_body_rotated, dim=1)
                    hitch_incoming_force_body_rotated_sn.append(
                        torch.stack(
                            (
                                torch.sum(
                                    body_rotated_sum * base.path_tangent_w,
                                    dim=-1,
                                ),
                                torch.sum(
                                    body_rotated_sum * base.path_normal_w,
                                    dim=-1,
                                ),
                            ),
                            dim=-1,
                        )[list(CONDITION_IDS)]
                    )
                    base_conjugate = torch.cat(
                        (
                            controlled_cart.data.body_quat_w[:, :1, :1],
                            -controlled_cart.data.body_quat_w[:, :1, 1:],
                        ),
                        dim=-1,
                    ).expand_as(
                        controlled_cart.data.body_quat_w[:, base.hitch_body_ids]
                    )
                    hitch_relative_quaternion_wxyz.append(
                        mdp.quat_multiply_wxyz(
                            base_conjugate,
                            controlled_cart.data.body_quat_w[:, base.hitch_body_ids],
                        )[list(CONDITION_IDS)]
                    )
                    cart_pitch.append(
                        base.rickshaw_state.pitch[list(CONDITION_IDS)]
                    )
                    valid_windows &= base.analytic_force_state.valid[list(CONDITION_IDS)]
                previous_cart_com_velocity_w = cart_com_velocity_w.clone()
                previous_cart_com_velocity_s = current_cart_com_velocity_s.clone()
                previous_wheel_velocity = current_wheel_velocity.clone()

        if not coast_time:
            raise RuntimeError("dynamics validation did not collect a measurement window")
        time_tensor = torch.tensor(coast_time)
        coast_speed_tensor = torch.stack(coast_speed).detach().cpu()
        acceleration_without = _linear_slope(time_tensor, coast_speed_tensor[:, 0])
        acceleration_with = _linear_slope(time_tensor, coast_speed_tensor[:, 1])
        normal_force = float(torch.stack(coast_normal)[:, 1].mean().detach().cpu())
        coast_masses = (
            coast.root_physx_view.get_masses()
            .sum(dim=-1)[list(COAST_COMPARISON_IDS)]
            .detach()
            .cpu()
            .tolist()
        )
        coast_mass = coast_masses[1]
        coast_result = evaluate_coast_down(
            mass_kg=coast_mass,
            mean_normal_force_n=normal_force,
            c_rr=maximum_c_rr,
            acceleration_without_rr_mps2=acceleration_without,
            acceleration_with_rr_mps2=acceleration_with,
            relative_tolerance=args.coast_relative_tolerance,
        )
        if not coast_result.passed:
            failures.append(
                "coast-down rolling-resistance force differs from c_rr*N or does not change deceleration"
            )
        if any(abs(mass - RICKSHAW_TOTAL_MASS) > 0.05 for mass in coast_masses):
            failures.append(
                f"coast cart PhysX masses {coast_masses} differ from {RICKSHAW_TOTAL_MASS:.6f} kg"
            )

        analytic_s_tensor = torch.stack(analytic_s).detach().cpu()
        analytic_n_tensor = torch.stack(analytic_n).detach().cpu()
        measured_s_tensor = torch.stack(measured_s).detach().cpu()
        measured_n_tensor = torch.stack(measured_n).detach().cpu()
        policy_sample_measured_s_tensor = torch.stack(
            policy_sample_measured_s
        ).detach().cpu()
        substep_d6_measured_s_tensor = torch.stack(
            substep_d6_measured_s
        ).detach().cpu()
        all_cart_contact_s_tensor = torch.stack(all_cart_contact_s).detach().cpu()
        ground_contact_s_tensor = torch.stack(ground_contact_s).detach().cpu()
        wheel_contact_s_tensor = torch.stack(wheel_contact_s).detach().cpu()
        cart_com_acceleration_s_tensor = torch.stack(
            cart_com_acceleration_s
        ).detach().cpu()
        wheel_angular_acceleration_tensor = torch.stack(
            wheel_angular_acceleration
        ).detach().cpu()
        wheel_incoming_joint_torque_y_tensor = torch.stack(
            wheel_incoming_joint_torque_y
        ).detach().cpu()
        rolling_resistance_s_tensor = torch.stack(
            rolling_resistance_s
        ).detach().cpu()
        hitch_incoming_force_raw_tensor = torch.stack(
            hitch_incoming_force_raw
        ).detach().cpu()
        hitch_incoming_force_raw_direct_sn_tensor = torch.stack(
            hitch_incoming_force_raw_direct_sn
        ).detach().cpu()
        hitch_incoming_force_parent_rotated_sn_tensor = torch.stack(
            hitch_incoming_force_parent_rotated_sn
        ).detach().cpu()
        hitch_incoming_force_body_rotated_sn_tensor = torch.stack(
            hitch_incoming_force_body_rotated_sn
        ).detach().cpu()
        hitch_relative_quaternion_wxyz_tensor = torch.stack(
            hitch_relative_quaternion_wxyz
        ).detach().cpu()
        cart_pitch_tensor = torch.stack(cart_pitch).detach().cpu()
        condition_ids = torch.tensor(CONDITION_IDS, device=base.device)
        gravity_w = torch.tensor(
            cfg.sim.gravity, device=base.device, dtype=base.path_tangent_w.dtype
        )
        gravity_s_tensor = (
            cart_mass * torch.sum(base.path_tangent_w * gravity_w, dim=-1)
        )[condition_ids].detach().cpu()
        condition_cart_mass = cart_mass[condition_ids].detach().cpu()
        momentum_balance_identity_residual_tensor = (
            measured_s_tensor
            + ground_contact_s_tensor
            + rolling_resistance_s_tensor
            + gravity_s_tensor[None, :]
            - condition_cart_mass[None, :] * cart_com_acceleration_s_tensor
        )
        incoming_proxy_force_balance_residual_tensor = (
            substep_d6_measured_s_tensor
            + ground_contact_s_tensor
            + rolling_resistance_s_tensor
            + gravity_s_tensor[None, :]
            - condition_cart_mass[None, :] * cart_com_acceleration_s_tensor
        )
        condition_metrics: dict[str, object] = {}
        for index, name in enumerate(CONDITION_NAMES):
            tangent_result = compare_wrench_component(
                analytic_s_tensor[:, index].tolist(),
                measured_s_tensor[:, index].tolist(),
                relative_tolerance=wrench_relative_tolerance,
                absolute_floor=wrench_absolute_floor_n,
            )
            normal_result = compare_wrench_component(
                analytic_n_tensor[:, index].tolist(),
                measured_n_tensor[:, index].tolist(),
                relative_tolerance=wrench_relative_tolerance,
                absolute_floor=wrench_absolute_floor_n,
            )
            condition_passed = (
                tangent_result.passed
                and normal_result.passed
                and bool(valid_windows[index])
                and not bool(terminated_conditions[index])
            )
            if not condition_passed:
                failures.append(f"{name} measured/analytic wrench comparison failed")
            condition_metrics[name] = {
                "slope": CONDITION_SLOPES[index],
                "tangential": asdict(tangent_result),
                "normal": asdict(normal_result),
                "analytic_valid_entire_window": bool(valid_windows[index]),
                "terminated": bool(terminated_conditions[index]),
                "force_source_diagnostics": {
                    "all_cart_contact_tangential_mean_n": float(
                        all_cart_contact_s_tensor[:, index].mean()
                    ),
                    "wheel_ground_contact_tangential_mean_n": float(
                        ground_contact_s_tensor[:, index].mean()
                    ),
                    "non_wheel_cart_contact_tangential_mean_n": float(
                        (
                            all_cart_contact_s_tensor[:, index]
                            - ground_contact_s_tensor[:, index]
                        ).mean()
                    ),
                    "runtime_momentum_balance_tangential_mean_n": float(
                        policy_sample_measured_s_tensor[:, index].mean()
                    ),
                    "incoming_proxy_substep_average_tangential_mean_n": float(
                        substep_d6_measured_s_tensor[:, index].mean()
                    ),
                    "momentum_balance_inferred_tangential_mean_n": float(
                        measured_s_tensor[:, index].mean()
                    ),
                    "wheel_contact_tangential_mean_n": float(
                        wheel_contact_s_tensor[:, index].mean()
                    ),
                    "cart_com_tangential_acceleration_mean_mps2": float(
                        cart_com_acceleration_s_tensor[:, index].mean()
                    ),
                    "wheel_angular_acceleration_mean_radps2": (
                        wheel_angular_acceleration_tensor[:, index].mean(dim=0).tolist()
                    ),
                    "wheel_incoming_joint_torque_y_mean_nm": (
                        wheel_incoming_joint_torque_y_tensor[:, index]
                        .mean(dim=0)
                        .tolist()
                    ),
                    "rolling_resistance_tangential_mean_n": float(
                        rolling_resistance_s_tensor[:, index].mean()
                    ),
                    "gravity_tangential_n": float(gravity_s_tensor[index]),
                    "momentum_balance_identity_residual_mean_n": float(
                        momentum_balance_identity_residual_tensor[:, index].mean()
                    ),
                    "momentum_balance_identity_residual_abs_max_n": float(
                        momentum_balance_identity_residual_tensor[:, index]
                        .abs()
                        .max()
                    ),
                    "incoming_proxy_force_balance_residual_mean_n": float(
                        incoming_proxy_force_balance_residual_tensor[:, index].mean()
                    ),
                    "incoming_proxy_force_balance_residual_abs_max_n": float(
                        incoming_proxy_force_balance_residual_tensor[:, index]
                        .abs()
                        .max()
                    ),
                    "cart_pitch_mean_rad": float(cart_pitch_tensor[:, index].mean()),
                    "hitch_incoming_force_raw_per_side_mean": (
                        hitch_incoming_force_raw_tensor[:, index]
                        .mean(dim=0)
                        .tolist()
                    ),
                    "hitch_incoming_force_raw_direct_sn_mean_n": (
                        hitch_incoming_force_raw_direct_sn_tensor[:, index]
                        .mean(dim=0)
                        .tolist()
                    ),
                    "hitch_incoming_force_parent_rotated_sn_mean_n": (
                        hitch_incoming_force_parent_rotated_sn_tensor[:, index]
                        .mean(dim=0)
                        .tolist()
                    ),
                    "hitch_incoming_force_body_rotated_sn_mean_n": (
                        hitch_incoming_force_body_rotated_sn_tensor[:, index]
                        .mean(dim=0)
                        .tolist()
                    ),
                    "hitch_relative_quaternion_wxyz_mean": (
                        hitch_relative_quaternion_wxyz_tensor[:, index]
                        .mean(dim=0)
                        .tolist()
                    ),
                },
            }

        metrics = {
            "coast_down": {
                **asdict(coast_result),
                "mass_kg": coast_mass,
                "masses_kg": coast_masses,
                "mean_normal_force_n": normal_force,
                "c_rr": maximum_c_rr,
                "acceleration_without_rr_mps2": acceleration_without,
                "acceleration_with_rr_mps2": acceleration_with,
                "sample_count": len(coast_time),
            },
            "d6_analytic_conditions": condition_metrics,
        }
        metadata = {
            "seed": args.seed,
            "physics_dt": base.physics_dt,
            "policy_dt": base.step_dt,
            "settling_steps": args.settling_steps,
            "measurement_steps": args.measurement_steps,
            "window_start": args.window_start,
            "coast_speed_mps": args.coast_speed,
            "coast_relative_tolerance": args.coast_relative_tolerance,
            "wrench_relative_tolerance": wrench_relative_tolerance,
            "wrench_absolute_floor_n": wrench_absolute_floor_n,
            "fat_wrench_absolute_floor_n": float(
                envelope.calibration["fat.wrench_consistency_absolute_floor_n"]
            ),
            "fat_wrench_consistency_window_steps": int(
                envelope.calibration["fat.wrench_consistency_window_steps"]
            ),
            "controlled_pelvis_force_n": applied_forces,
            "coast_wheel_force_location": "wheel centers",
            "coast_normal_force_source": "level_vehicle_weight",
            "coast_rail_free_axes": ["transX"],
            "measured_wrench_source": "whole_cart_momentum_balance",
            "ground_contact_force_source": "two_wheel_contact_sensor_net_forces",
            "incoming_joint_wrench_role": "constraint_residual_impulse_proxy_only",
            "d6_force_sign": "momentum balance gives robot-on-cart; hand force is opposite",
            "policy_safety_terminations_disabled": True,
        }
        return metrics, failures, metadata
    finally:
        env.close()


def main() -> int:
    metrics: dict[str, object] = {}
    metadata: dict[str, object] = {}
    failures: list[str] = []
    try:
        metrics, failures, metadata = _run()
    except Exception as exc:
        failures.append(f"runtime error: {type(exc).__name__}: {exc}")
        metadata["traceback"] = traceback.format_exc()
    report = build_report(
        tool="validate_dynamics",
        task=args.task,
        passed=not failures,
        feasibility_path=args.feasibility,
        reset_pose_path=args.reset_poses,
        assets=validation_input_assets(REPOSITORY_ROOT),
        metrics=metrics,
        failures=failures,
        metadata=metadata,
    )
    output = write_json_atomic(args.output, report)
    print(f"dynamics validation {report['status']}: {output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
