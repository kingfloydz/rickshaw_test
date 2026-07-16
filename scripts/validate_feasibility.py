#!/usr/bin/env python3
"""Derive the RL feasibility envelope from the configured-slope PhysX scan."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from g1_rickshaw_lab.configuration import (  # noqa: E402
    FeasibilityEnvelope,
    SLOPE_GRADIENTS,
    load_feasibility_envelope,
    load_reset_pose_library,
)
from g1_rickshaw_lab.slope_contract import terrain_index_for_gradient  # noqa: E402
from g1_rickshaw_lab.validation import (  # noqa: E402
    FEASIBILITY_MEASUREMENT_SOURCES,
    FEASIBILITY_MINIMUM_PASS_FRACTION,
    VALIDATION_REPORT_SCHEMA_VERSION,
    asset_hashes,
    assert_safety_thresholds_match,
    build_feasibility_scan_plan,
    build_report,
    derive_feasibility_envelope_mapping,
    evaluate_feasibility_sample,
    load_safety_threshold_authority,
    select_conservative_limit,
    sha256_file,
    synchronize_runtime_randomization_events,
    utc_timestamp,
    validate_safety_authority_source_evidence,
    validation_input_assets,
    write_json_atomic,
    write_yaml_atomic,
)

SAFETY_FACTOR = 0.8
FORCE_DIRECTIONS = (-1.0, 1.0)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="Isaac-G1-Rickshaw-Directional-Slope-Play-v0"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT / "config/feasibility_envelope.yaml",
        help="Physical ranges and calibration template; command limits are ignored as candidates.",
    )
    parser.add_argument(
        "--envelope-output",
        type=Path,
        default=REPOSITORY_ROOT / "config/feasibility_envelope.yaml",
        help="Derived envelope, written atomically only after a full pass.",
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=REPOSITORY_ROOT / "config/feasibility_scan_candidates.yaml",
        help="Independent, repeatable acceleration/jerk search candidates.",
    )
    parser.add_argument(
        "--safety-authority",
        type=Path,
        default=REPOSITORY_ROOT / "config/safety_authority.yaml",
        help="Independent, content-addressed hardware safety-threshold authority.",
    )
    parser.add_argument(
        "--reset-poses",
        type=Path,
        default=REPOSITORY_ROOT / "config/reset_poses.yaml",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate input schemas without launching PhysX or producing a gate report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/validation/feasibility_report.json",
        help="Content-addressed validation report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps-per-point",
        type=int,
        default=60,
        help="Policy steps in each force-ramp trial.",
    )
    parser.add_argument(
        "--acceleration-candidates",
        default=None,
        help="Optional comma-separated positive m/s^2 override candidates.",
    )
    parser.add_argument(
        "--jerk-candidates",
        default=None,
        help="Optional comma-separated positive m/s^3 override candidates.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run nominal/cross diagnostics only; never writes an envelope or passes the gate.",
    )
    return parser


def _validate_candidate_values(values: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} values must be explicit numbers, not booleans")
    parsed = tuple(float(value) for value in values)
    if len(parsed) < 2:
        raise ValueError(f"{name} must contain at least two search candidates")
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError(f"{name} values must be finite and positive")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError(f"{name} values must be strictly increasing and unique")
    return parsed


def _load_candidate_config(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to load command scan candidates") from exc
    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate candidate config key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    with path.open("r", encoding="utf-8") as stream:
        mapping = yaml.load(stream, Loader=UniqueKeySafeLoader)
    if not isinstance(mapping, dict):
        raise ValueError("candidate config must be a YAML mapping")
    expected = {
        "schema_version",
        "acceleration_candidates_mps2",
        "jerk_candidates_mps3",
    }
    if set(mapping) != expected:
        raise ValueError(
            f"candidate config fields must be exactly {sorted(expected)}, got {sorted(mapping)}"
        )
    if not isinstance(mapping["schema_version"], int) or isinstance(
        mapping["schema_version"], bool
    ) or mapping["schema_version"] != 1:
        raise ValueError("candidate config schema_version must be 1")
    return (
        _validate_candidate_values(
            mapping["acceleration_candidates_mps2"],
            name="acceleration_candidates_mps2",
        ),
        _validate_candidate_values(
            mapping["jerk_candidates_mps3"], name="jerk_candidates_mps3"
        ),
    )


def _candidate_override(
    explicit: str | None, configured: tuple[float, ...], *, name: str
) -> tuple[float, ...]:
    if explicit is None:
        return configured
    pieces = [piece.strip() for piece in explicit.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError(f"--{name}-candidates must be a comma-separated number list")
    return _validate_candidate_values(
        [float(piece) for piece in pieces], name=f"--{name}-candidates"
    )


def _additional_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "command_candidate_config": args.candidate_config,
        "safety_threshold_authority": args.safety_authority,
    }


def _preflight_inputs(args: argparse.Namespace) -> tuple[FeasibilityEnvelope, Any]:
    envelope = load_feasibility_envelope(args.input)
    _load_candidate_config(args.candidate_config)
    load_reset_pose_library(args.reset_poses)
    authority = load_safety_threshold_authority(
        args.safety_authority,
        forbidden_source_paths=(args.input, args.envelope_output),
    )
    assert_safety_thresholds_match(
        envelope.calibration,
        authority.thresholds,
        label="candidate feasibility calibration",
    )
    validate_safety_authority_source_evidence(
        authority,
        task=args.task,
        assets_sha256=asset_hashes(validation_input_assets(REPOSITORY_ROOT)),
        feasibility_path=args.input,
        reset_pose_path=args.reset_poses,
        reset_pose_sha256=sha256_file(args.reset_poses),
    )
    return envelope, authority


def _failure_report_template(args: argparse.Namespace) -> dict[str, Any]:
    return build_report(
        tool="validate_feasibility",
        task=args.task,
        passed=False,
        feasibility_path=args.input,
        reset_pose_path=args.reset_poses,
        assets=validation_input_assets(REPOSITORY_ROOT),
        additional_inputs=_additional_inputs(args),
        metrics={},
        failures=["feasibility scan has not completed"],
        metadata={
            "seed": args.seed,
            "steps_per_point": args.steps_per_point,
            "failure_phase": "pending",
        },
    )


def _write_failed_report(
    args: argparse.Namespace,
    template: dict[str, Any],
    exc: BaseException,
    *,
    phase: str,
    traceback_text: str,
) -> Path:
    report = copy.deepcopy(template)
    failure = f"runtime error: {type(exc).__name__}: {exc}"
    report["status"] = "failed"
    report["metrics"] = {}
    report["failures"] = [failure]
    report["metadata"] = {
        **dict(report.get("metadata", {})),
        "failure_phase": phase,
        "traceback": traceback_text,
    }
    output = write_json_atomic(args.output, report)
    print(f"feasibility scan failed: {output}")
    print(f"FAIL: {failure}")
    return output


def _verify_recorded_outcome(args: argparse.Namespace, exit_code: Any) -> int:
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code not in {0, 1}:
        raise RuntimeError(f"scan returned invalid exit code {exit_code!r}")
    try:
        report = json.loads(args.output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"scan did not write a readable report: {exc}") from exc
    expected_status = "passed" if exit_code == 0 else "failed"
    if (
        not isinstance(report, dict)
        or report.get("tool") != "validate_feasibility"
        or report.get("status") != expected_status
    ):
        raise RuntimeError(
            f"scan exit code {exit_code} disagrees with its recorded report status"
        )
    return exit_code


def _run_scan_with_app(
    args: argparse.Namespace,
    app_args: argparse.Namespace,
    simulation_app: Any,
    failure_template: dict[str, Any],
    *,
    scan_fn: Any = None,
) -> int:
    """Run the Kit portion and persist all failures before closing the app."""

    runner = _run_scan_in_kit if scan_fn is None else scan_fn
    exit_code = 1
    try:
        try:
            exit_code = _verify_recorded_outcome(args, runner(args, app_args))
        except BaseException as exc:
            _write_failed_report(
                args,
                failure_template,
                exc,
                phase="kit_runtime",
                traceback_text=traceback.format_exc(),
            )
            exit_code = 1
    finally:
        try:
            simulation_app.close()
        except BaseException as exc:
            _write_failed_report(
                args,
                failure_template,
                exc,
                phase="simulation_app_close",
                traceback_text=traceback.format_exc(),
            )
            exit_code = 1
    return exit_code


def _run_scan(
    args: argparse.Namespace,
    app_argv: list[str],
    failure_template: dict[str, Any],
) -> int:
    # Invalidate any stale passed report before importing or launching Kit.
    write_json_atomic(args.output, failure_template)
    isaaclab_path = Path(
        os.environ.get("ISAACLAB_PATH", REPOSITORY_ROOT.parent / "IsaacLab")
    ).resolve()
    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
        package_path = isaaclab_path / "source" / package_name
        if package_path.is_dir() and str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(app_argv)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    return _run_scan_with_app(
        args,
        app_args,
        simulation_app,
        failure_template,
    )


def _serial_float(value: Any) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _coverage_summary(
    rows: list[dict[str, Any]],
    candidates: tuple[float, ...],
    *,
    point_count: int,
) -> tuple[dict[float, bool], dict[str, Any]]:
    expected_per_candidate = point_count * len(SLOPE_GRADIENTS) * len(FORCE_DIRECTIONS)
    result: dict[float, bool] = {}
    details: dict[str, Any] = {}
    for candidate in candidates:
        selected = [row for row in rows if row["candidate"] == candidate]
        by_slope = {
            f"{slope:+.2f}": {
                "observed": sum(row["slope"] == slope for row in selected),
                "passed": sum(
                    row["slope"] == slope and bool(row["passed"]) for row in selected
                ),
            }
            for slope in SLOPE_GRADIENTS
        }
        complete = len(selected) == expected_per_candidate
        passed_count = sum(bool(row["passed"]) for row in selected)
        passed = (
            complete
            and passed_count / expected_per_candidate
            >= FEASIBILITY_MINIMUM_PASS_FRACTION
        )
        result[candidate] = passed
        details[format(candidate, ".12g")] = {
            "complete": complete,
            "all_physical_endpoints_passed": passed,
            "expected_trials": expected_per_candidate,
            "observed_trials": len(selected),
            "passed_trials": passed_count,
            "slope_coverage": by_slope,
        }
    return result, details


def _run_scan_in_kit(args: argparse.Namespace, app_args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    from g1_rickshaw_lab.assets.rickshaw import BASE_LINK_NAME
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import (
        G1RickshawDirectionalSlopePlayEnvCfg, mdp)

    candidate_path = args.input.resolve()
    reset_pose_path = args.reset_poses.resolve()
    envelope_output = args.envelope_output.resolve()
    command_candidate_path = args.candidate_config.resolve()
    safety_authority_path = args.safety_authority.resolve()
    configured_acceleration, configured_jerk = _load_candidate_config(
        command_candidate_path
    )
    command_candidate_sha256 = sha256_file(command_candidate_path)
    candidate_sha256 = sha256_file(candidate_path)
    envelope = load_feasibility_envelope(candidate_path)
    load_reset_pose_library(reset_pose_path)
    safety_authority = load_safety_threshold_authority(
        safety_authority_path,
        forbidden_source_paths=(candidate_path, envelope_output),
    )
    safety_authority_sha256 = sha256_file(safety_authority_path)
    assert_safety_thresholds_match(
        envelope.calibration,
        safety_authority.thresholds,
        label="candidate feasibility calibration",
    )
    if args.steps_per_point <= 1:
        raise ValueError("--steps-per-point must be greater than one")
    acceleration_candidates = _candidate_override(
        args.acceleration_candidates,
        configured_acceleration,
        name="acceleration",
    )
    jerk_candidates = _candidate_override(
        args.jerk_candidates,
        configured_jerk,
        name="jerk",
    )
    range_pairs = {
        name: (interval.minimum, interval.maximum)
        for name, interval in envelope.ranges.items()
    }
    full_plan = build_feasibility_scan_plan(range_pairs)
    if args.quick:
        plan = tuple(
            point
            for point in full_plan
            if point.name == "nominal" or point.name.startswith("cross:")
        )
        acceleration_candidates = (acceleration_candidates[-1],)
        jerk_candidates = (jerk_candidates[-1],)
    else:
        plan = full_plan

    os.environ["G1_RICKSHAW_FEASIBILITY_ENVELOPE"] = os.fspath(candidate_path)
    os.environ["G1_RICKSHAW_RESET_POSES"] = os.fspath(reset_pose_path)
    cfg = G1RickshawDirectionalSlopePlayEnvCfg()
    if Path(cfg.feasibility_path).resolve() != candidate_path:
        raise RuntimeError("environment did not load the requested candidate envelope")
    if Path(cfg.reset_pose_path).resolve() != reset_pose_path:
        raise RuntimeError("environment did not load the requested reset-pose library")
    cfg.scene.num_envs = len(SLOPE_GRADIENTS)
    cfg.sim.device = app_args.device
    cfg.curriculum = None
    cfg.scene.terrain.terrain_generator.curriculum = True
    scan_randomization = replace(
        cfg.runtime_randomization,
        sample_ranges=False,
        curriculum=replace(
            cfg.runtime_randomization.curriculum,
            cross_case_fraction=0.0,
        ),
    )
    synchronize_runtime_randomization_events(cfg, scan_randomization)

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    all_ids = torch.arange(base.num_envs, device=base.device, dtype=torch.long)
    actions = torch.zeros(env.action_space.shape, device=base.device)
    acceleration_rows: list[dict[str, Any]] = []
    jerk_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    safety_cfg = base.termination_manager.get_term_cfg("immediate_safety").params[
        "cfg"
    ]
    minimum_wheel_force = float(
        safety_authority.thresholds["safety.minimum_wheel_normal_force"]
    )
    d6_impulse_limit = float(
        safety_authority.thresholds["safety.d6_impulse_limit"]
    )
    if not math.isclose(
        float(safety_cfg.wheel_lift_normal_force_threshold),
        minimum_wheel_force,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        float(safety_cfg.d6_impulse_limit),
        d6_impulse_limit,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "environment safety configuration does not match the independent authority"
        )

    def clear_cart_force(cart: Any, base_body_ids: list[int]) -> None:
        zeros = torch.zeros((base.num_envs, 1, 3), device=base.device)
        cart.permanent_wrench_composer.set_forces_and_torques(
            zeros,
            zeros,
            body_ids=base_body_ids,
            env_ids=all_ids,
            is_global=True,
        )

    def configure_physical_point(point: Any) -> None:
        singleton_ranges = {
            name: (
                float(cfg.runtime_randomization.nominal_values.get(name, 0.5 * (low + high))),
                float(cfg.runtime_randomization.nominal_values.get(name, 0.5 * (low + high))),
            )
            for name, (low, high) in cfg.runtime_randomization.ranges.items()
        }
        for name, value in point.values.items():
            singleton_ranges[name] = (float(value), float(value))
        singleton_nominal = {
            name: singleton_ranges[name][0]
            for name in cfg.runtime_randomization.nominal_values
        }
        point_randomization = replace(
            cfg.runtime_randomization,
            ranges=singleton_ranges,
            nominal_values=singleton_nominal,
            sample_ranges=True,
            curriculum=replace(
                cfg.runtime_randomization.curriculum,
                cross_case_fraction=0.0,
            ),
        )
        mdp.sample_episode_physics(base, all_ids, point_randomization)
        mdp.reset_closed_chain(base, all_ids)
        base.episode_length_buf[all_ids] = 0
        base.command_state.v_sample[all_ids] = 0.0
        base.command_state.v_ref[all_ids] = 0.0
        base.command_state.a_ref[all_ids] = 0.0

    def prepare_trial(cart: Any, base_body_ids: list[int]) -> torch.Tensor:
        clear_cart_force(cart, base_body_ids)
        clean = torch.ones(base.num_envs, device=base.device, dtype=torch.bool)
        base.command_state.v_sample[all_ids] = 0.0
        base.command_state.v_ref[all_ids] = 0.0
        base.command_state.a_ref[all_ids] = 0.0
        return clean

    def run_force_trial(
        *,
        stage: str,
        point_index: int,
        point: Any,
        candidate: float,
        target_acceleration: float,
        ramp_jerk: float,
        direction: float,
    ) -> list[dict[str, Any]]:
        configure_physical_point(point)
        cart = base.scene["rickshaw"]
        base_body_ids, base_body_names = cart.find_bodies(
            BASE_LINK_NAME, preserve_order=True
        )
        if len(base_body_ids) != 1:
            raise RuntimeError(
                f"force scan requires one cart base body, got {base_body_names}"
            )
        base_body_ids = [int(base_body_ids[0])]
        initially_valid = prepare_trial(cart, base_body_ids)

        cart_masses = (
            cart.root_physx_view.get_masses()
            .sum(dim=-1)
            .to(device=base.device, dtype=torch.float32)
        )
        if cart_masses.shape != (base.num_envs,) or torch.any(cart_masses <= 0.0):
            raise RuntimeError("invalid per-environment PhysX cart masses")
        ramp_steps = max(
            1, int(math.ceil(target_acceleration / (ramp_jerk * base.step_dt)))
        )
        if ramp_steps >= args.steps_per_point:
            raise RuntimeError(
                f"{stage} ramp needs {ramp_steps} steps at jerk={ramp_jerk:g}, "
                f"but --steps-per-point={args.steps_per_point} leaves no hold sample"
            )

        inf = torch.full((base.num_envs,), torch.inf, device=base.device)
        minimum_left = inf.clone()
        minimum_right = inf.clone()
        minimum_friction_margin = inf.clone()
        minimum_zmp = inf.clone()
        minimum_joint_margin = inf.clone()
        maximum_arm_ratio = torch.zeros_like(inf)
        maximum_leg_ratio = torch.zeros_like(inf)
        maximum_waist_ratio = torch.zeros_like(inf)
        maximum_d6_force_ratio = torch.zeros_like(inf)
        maximum_d6_torque_ratio = torch.zeros_like(inf)
        maximum_d6_impulse_ratio = torch.zeros_like(inf)
        finite = initially_valid.clone()
        measured = torch.zeros_like(initially_valid)
        terminated_any = ~initially_valid
        alive = initially_valid.clone()
        acceleration_min = inf.clone()
        acceleration_max = -inf.clone()
        acceleration_sum = torch.zeros_like(inf)
        acceleration_count = torch.zeros_like(inf)
        acceleration_abs_peak = torch.zeros_like(inf)

        robot = base.scene["robot"]
        arm_limits = mdp.actuator_effort_limits(robot, base.arm_joint_ids)
        leg_ids = base.policy_joint_ids[:12]
        leg_limits = mdp.actuator_effort_limits(robot, leg_ids)
        waist_ids = base.policy_joint_ids[12:15]
        waist_limits = mdp.actuator_effort_limits(robot, waist_ids)
        hard_limits = robot.data.joint_pos_limits[:, base.policy_joint_ids]
        if hard_limits.ndim == 2:
            hard_limits = hard_limits.unsqueeze(0).expand(base.num_envs, -1, -1)
        previous_speed = torch.sum(
            cart.data.root_lin_vel_w * base.path_tangent_w, dim=-1
        ).clone()
        peak_force_w = (
            direction
            * cart_masses[:, None]
            * target_acceleration
            * base.path_tangent_w
        )

        for step in range(args.steps_per_point):
            applied_acceleration = min(
                target_acceleration, ramp_jerk * base.step_dt * float(step + 1)
            )
            force_w = (
                direction
                * cart_masses[:, None]
                * applied_acceleration
                * base.path_tangent_w
            )
            force_w[~alive] = 0.0
            cart.permanent_wrench_composer.set_forces_and_torques(
                force_w[:, None, :],
                torch.zeros_like(force_w[:, None, :]),
                body_ids=base_body_ids,
                env_ids=all_ids,
                is_global=True,
            )
            _, _, terminated, truncated, _ = env.step(actions)
            ended = terminated | truncated
            terminated_any |= ended
            active = alive & ~ended
            alive &= ~ended

            speed = torch.sum(cart.data.root_lin_vel_w * base.path_tangent_w, dim=-1)
            measured_acceleration = (speed - previous_speed) / base.step_dt
            previous_speed = speed.clone()
            if not torch.any(active):
                continue
            measured |= active
            acceleration_min[active] = torch.minimum(
                acceleration_min[active], measured_acceleration[active]
            )
            acceleration_max[active] = torch.maximum(
                acceleration_max[active], measured_acceleration[active]
            )
            acceleration_sum[active] += measured_acceleration[active]
            acceleration_count[active] += 1.0
            acceleration_abs_peak[active] = torch.maximum(
                acceleration_abs_peak[active], torch.abs(measured_acceleration[active])
            )

            wheel = base.rickshaw_state.wheel_normal_force
            minimum_left[active] = torch.minimum(minimum_left[active], wheel[active, 0])
            minimum_right[active] = torch.minimum(minimum_right[active], wheel[active, 1])
            foot_force = base.scene["robot_contacts"].data.net_forces_w[
                :, base.foot_sensor_ids
            ]
            normal_force = torch.clamp(
                torch.sum(foot_force * base.path_normal_w[:, None, :], dim=-1),
                min=0.0,
            )
            tangent_force = foot_force - normal_force[..., None] * base.path_normal_w[:, None, :]
            friction = base.teacher_extrinsic_values["terrain.friction"]
            friction_margin = torch.amin(
                friction[:, None] * normal_force
                - torch.linalg.vector_norm(tangent_force, dim=-1),
                dim=-1,
            )
            minimum_friction_margin[active] = torch.minimum(
                minimum_friction_margin[active], friction_margin[active]
            )
            zmp = torch.where(
                base.stability_state.zmp_valid,
                base.stability_state.zmp_margin,
                torch.full_like(base.stability_state.zmp_margin, -torch.inf),
            )
            minimum_zmp[active] = torch.minimum(minimum_zmp[active], zmp[active])
            arm_ratio = torch.amax(
                torch.abs(robot.data.applied_torque[:, base.arm_joint_ids]) / arm_limits,
                dim=-1,
            )
            leg_ratio = torch.amax(
                torch.abs(robot.data.applied_torque[:, leg_ids]) / leg_limits, dim=-1
            )
            waist_ratio = torch.amax(
                torch.abs(robot.data.applied_torque[:, waist_ids]) / waist_limits,
                dim=-1,
            )
            maximum_arm_ratio[active] = torch.maximum(
                maximum_arm_ratio[active], arm_ratio[active]
            )
            maximum_leg_ratio[active] = torch.maximum(
                maximum_leg_ratio[active], leg_ratio[active]
            )
            maximum_waist_ratio[active] = torch.maximum(
                maximum_waist_ratio[active], waist_ratio[active]
            )
            d6_proxy_wrench = getattr(base, "d6_incoming_joint_proxy_w", None)
            if not torch.is_tensor(d6_proxy_wrench) or d6_proxy_wrench.shape != (
                base.num_envs,
                2,
                6,
            ):
                raise RuntimeError(
                    "feasibility scan requires the retained-hitch incoming D6 proxy"
                )
            d6_force = torch.linalg.vector_norm(
                d6_proxy_wrench[..., :3], dim=-1
            ).amax(dim=-1)
            d6_torque = torch.linalg.vector_norm(
                d6_proxy_wrench[..., 3:], dim=-1
            ).amax(dim=-1)
            max_force = base.d6_constraint_manager.parameter_values(
                "max_force", all_ids, device=base.device, dtype=d6_force.dtype
            )
            max_torque = base.d6_constraint_manager.parameter_values(
                "max_torque", all_ids, device=base.device, dtype=d6_force.dtype
            )
            maximum_d6_force_ratio[active] = torch.maximum(
                maximum_d6_force_ratio[active], (d6_force / max_force)[active]
            )
            maximum_d6_torque_ratio[active] = torch.maximum(
                maximum_d6_torque_ratio[active], (d6_torque / max_torque)[active]
            )
            impulse_ratio = torch.amax(
                base.rickshaw_state.d6_impulse / d6_impulse_limit, dim=-1
            )
            maximum_d6_impulse_ratio[active] = torch.maximum(
                maximum_d6_impulse_ratio[active], impulse_ratio[active]
            )
            q_ref = base.action_state.q_ref
            joint_margin = torch.amin(
                torch.minimum(q_ref - hard_limits[..., 0], hard_limits[..., 1] - q_ref),
                dim=-1,
            )
            minimum_joint_margin[active] = torch.minimum(
                minimum_joint_margin[active], joint_margin[active]
            )
            checked = (
                robot.data.root_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                cart.data.root_state_w,
                d6_proxy_wrench,
                base.rickshaw_state.d6_residual,
                base.rickshaw_state.d6_impulse,
                base.stability_state.zmp_margin,
                base.action_state.target,
            )
            finite &= ~mdp.finite_tensor_violation(*checked)

        clear_cart_force(cart, base_body_ids)
        rows: list[dict[str, Any]] = []
        for env_index, slope in enumerate(SLOPE_GRADIENTS):
            raw_metrics = {
                "left_wheel_normal_force": float(minimum_left[env_index]),
                "right_wheel_normal_force": float(minimum_right[env_index]),
                "friction_cone_margin": float(minimum_friction_margin[env_index]),
                "zmp_margin": float(minimum_zmp[env_index]),
                "arm_torque_ratio": float(maximum_arm_ratio[env_index]),
                "leg_torque_ratio": float(maximum_leg_ratio[env_index]),
                "waist_torque_ratio": float(maximum_waist_ratio[env_index]),
                "d6_force_ratio": float(maximum_d6_force_ratio[env_index]),
                "d6_torque_ratio": float(maximum_d6_torque_ratio[env_index]),
                "d6_impulse_ratio": float(maximum_d6_impulse_ratio[env_index]),
                "joint_limit_margin": float(minimum_joint_margin[env_index]),
                "finite": bool(finite[env_index]),
                "terminated": bool(terminated_any[env_index]),
            }
            numeric_finite = all(
                math.isfinite(raw_metrics[name])
                for name in (
                    "left_wheel_normal_force",
                    "right_wheel_normal_force",
                    "friction_cone_margin",
                    "zmp_margin",
                    "arm_torque_ratio",
                    "leg_torque_ratio",
                    "waist_torque_ratio",
                    "d6_force_ratio",
                    "d6_torque_ratio",
                    "d6_impulse_ratio",
                    "joint_limit_margin",
                )
            )
            if not bool(measured[env_index]):
                passed = False
                reasons = ["no force-ramp samples were collected"]
            elif not numeric_finite:
                passed = False
                reasons = ["force-ramp metrics contain NaN/Inf"]
            else:
                passed, reasons = evaluate_feasibility_sample(
                    raw_metrics,
                    minimum_wheel_normal_force=minimum_wheel_force,
                )
            if raw_metrics["terminated"]:
                passed = False
                reasons.append("environment terminated during force-ramp trial")
            if stage == "jerk" and raw_metrics["d6_impulse_ratio"] > 1.0:
                passed = False
                reasons.append("D6 impulse threshold was exceeded")
            count = float(acceleration_count[env_index])
            dynamic_evidence = {
                "cart_mass_kg": float(cart_masses[env_index]),
                "target_equivalent_acceleration_mps2": target_acceleration,
                "applied_force_peak_n": float(cart_masses[env_index] * target_acceleration),
                "applied_force_peak_w_n": [
                    float(value) for value in peak_force_w[env_index].tolist()
                ],
                "force_body": BASE_LINK_NAME,
                "force_api": (
                    "Articulation.permanent_wrench_composer."
                    "set_forces_and_torques"
                ),
                "arm_actuator_effort_limit_nm": {
                    "minimum": float(torch.amin(arm_limits[env_index])),
                    "maximum": float(torch.amax(arm_limits[env_index])),
                },
                "leg_actuator_effort_limit_nm": {
                    "minimum": float(torch.amin(leg_limits[env_index])),
                    "maximum": float(torch.amax(leg_limits[env_index])),
                },
                "waist_actuator_effort_limit_nm": {
                    "minimum": float(torch.amin(waist_limits[env_index])),
                    "maximum": float(torch.amax(waist_limits[env_index])),
                },
                "ramp_jerk_mps3": ramp_jerk,
                "ramp_steps": ramp_steps,
                "hold_steps": args.steps_per_point - ramp_steps,
                "measured_cart_acceleration_mps2": {
                    "minimum": _serial_float(acceleration_min[env_index]),
                    "maximum": _serial_float(acceleration_max[env_index]),
                    "mean": (
                        _serial_float(acceleration_sum[env_index] / count)
                        if count > 0.0
                        else None
                    ),
                    "peak_absolute": _serial_float(acceleration_abs_peak[env_index]),
                    "sample_count": int(count),
                },
            }
            rows.append(
                {
                    "stage": stage,
                    "point_index": point_index,
                    "point": point.name,
                    "slope": slope,
                    "force_direction": int(direction),
                    "candidate": candidate,
                    "parameters": dict(point.values),
                    "metrics": {
                        name: (
                            value
                            if isinstance(value, bool)
                            else _serial_float(value)
                        )
                        for name, value in raw_metrics.items()
                    },
                    "dynamic_evidence": dynamic_evidence,
                    "passed": passed,
                    "failures": reasons,
                }
            )
        return rows

    try:
        indices = [terrain_index_for_gradient(slope) for slope in SLOPE_GRADIENTS]
        levels = torch.tensor([item[0] for item in indices], device=base.device)
        columns = torch.tensor([item[1] for item in indices], device=base.device)
        terrain = base.scene.terrain
        terrain.terrain_levels.copy_(levels)
        terrain.terrain_types.copy_(columns)
        terrain.env_origins.copy_(terrain.terrain_origins[levels, columns])
        env.reset(seed=args.seed)

        with torch.inference_mode():
            for candidate in acceleration_candidates:
                # Use half the trial for a deliberately slow acceleration-load ramp.
                acceleration_scan_jerk = candidate / (
                    max(1, args.steps_per_point // 2) * base.step_dt
                )
                for point_index, point in enumerate(plan):
                    for direction in FORCE_DIRECTIONS:
                        acceleration_rows.extend(
                            run_force_trial(
                                stage="acceleration",
                                point_index=point_index,
                                point=point,
                                candidate=candidate,
                                target_acceleration=candidate,
                                ramp_jerk=acceleration_scan_jerk,
                                direction=direction,
                            )
                        )

            acceleration_passed, acceleration_coverage = _coverage_summary(
                acceleration_rows,
                acceleration_candidates,
                point_count=len(plan),
            )
            try:
                acceleration_selection = select_conservative_limit(
                    acceleration_passed, safety_factor=SAFETY_FACTOR
                )
            except ValueError as exc:
                acceleration_selection = None
                failures.append(f"acceleration search failed: {exc}")

            jerk_coverage: dict[str, Any] = {}
            jerk_passed: dict[float, bool] = {}
            jerk_selection = None
            if acceleration_selection is not None:
                target_acceleration = acceleration_selection.derived_limit
                for candidate in jerk_candidates:
                    for point_index, point in enumerate(plan):
                        for direction in FORCE_DIRECTIONS:
                            jerk_rows.extend(
                                run_force_trial(
                                    stage="jerk",
                                    point_index=point_index,
                                    point=point,
                                    candidate=candidate,
                                    target_acceleration=target_acceleration,
                                    ramp_jerk=candidate,
                                    direction=direction,
                                )
                            )
                jerk_passed, jerk_coverage = _coverage_summary(
                    jerk_rows,
                    jerk_candidates,
                    point_count=len(plan),
                )
                try:
                    jerk_selection = select_conservative_limit(
                        jerk_passed, safety_factor=SAFETY_FACTOR
                    )
                except ValueError as exc:
                    failures.append(f"jerk search failed: {exc}")

        if args.quick:
            failures.append(
                "quick scan has incomplete physical endpoint/candidate coverage "
                "and cannot authorize training"
            )

        generated_path: Path | None = None
        derived_mapping: dict[str, Any] | None = None
        if not failures and acceleration_selection is not None and jerk_selection is not None:
            # A long scan cannot authorize output if its independent thresholds or
            # provenance changed while PhysX was running.
            current_authority = load_safety_threshold_authority(
                safety_authority_path,
                forbidden_source_paths=(candidate_path, envelope_output),
            )
            if sha256_file(safety_authority_path) != safety_authority_sha256:
                raise RuntimeError("safety threshold authority changed during the scan")
            assert_safety_thresholds_match(
                envelope.calibration,
                current_authority.thresholds,
                label="candidate feasibility calibration",
            )
            validate_safety_authority_source_evidence(
                current_authority,
                task=args.task,
                assets_sha256=asset_hashes(
                    validation_input_assets(REPOSITORY_ROOT)
                ),
                feasibility_path=candidate_path,
                reset_pose_path=reset_pose_path,
                reset_pose_sha256=sha256_file(reset_pose_path),
            )
            safety_authority = current_authority
            derived_mapping = derive_feasibility_envelope_mapping(
                envelope.to_mapping(),
                acceleration_limit=acceleration_selection.derived_limit,
                jerk_limit=jerk_selection.derived_limit,
                safety_thresholds=safety_authority.thresholds,
            )
            # Validate the exact object before the only write that can authorize training.
            FeasibilityEnvelope.from_mapping(derived_mapping)
            generated_path = write_yaml_atomic(envelope_output, derived_mapping)
            load_feasibility_envelope(generated_path)

        metrics = {
            "coverage": "quick" if args.quick else "full",
            "physical_scan_points": [point.name for point in plan],
            "physical_scan_point_count": len(plan),
            "slopes": list(SLOPE_GRADIENTS),
            "force_directions": list(FORCE_DIRECTIONS),
            "force_definition": {
                "body": BASE_LINK_NAME,
                "frame": "world",
                "direction": "signed path tangent",
                "magnitude": "actual PhysX cart mass * equivalent acceleration ramp",
            },
            "command_candidate_config": {
                "path": str(command_candidate_path),
                "sha256": command_candidate_sha256,
                "cli_acceleration_override": args.acceleration_candidates,
                "cli_jerk_override": args.jerk_candidates,
            },
            "safety_threshold_authority": safety_authority.evidence_record(),
            "acceleration_search": {
                "candidates_mps2": list(acceleration_candidates),
                "candidate_full_coverage_passed": {
                    format(value, ".12g"): passed
                    for value, passed in acceleration_passed.items()
                },
                "coverage": acceleration_coverage,
                "maximum_fully_feasible_mps2": (
                    acceleration_selection.maximum_feasible
                    if acceleration_selection is not None
                    else None
                ),
                "derived_limit_mps2": (
                    acceleration_selection.derived_limit
                    if acceleration_selection is not None
                    else None
                ),
                "safety_factor": SAFETY_FACTOR,
                "trials": acceleration_rows,
            },
            "jerk_search": {
                "candidates_mps3": list(jerk_candidates),
                "acceleration_held_at_derived_limit_mps2": (
                    acceleration_selection.derived_limit
                    if acceleration_selection is not None
                    else None
                ),
                "d6_impulse_threshold_ratio": 1.0,
                "candidate_full_coverage_passed": {
                    format(value, ".12g"): passed
                    for value, passed in jerk_passed.items()
                },
                "coverage": jerk_coverage,
                "maximum_fully_feasible_mps3": (
                    jerk_selection.maximum_feasible if jerk_selection is not None else None
                ),
                "derived_limit_mps3": (
                    jerk_selection.derived_limit if jerk_selection is not None else None
                ),
                "safety_factor": SAFETY_FACTOR,
                "trials": jerk_rows,
            },
            "generated_envelope": (
                {
                    "path": str(generated_path),
                    "sha256": sha256_file(generated_path),
                    "physical_ranges_preserved": all(
                        derived_mapping["ranges"][name]
                        == envelope.to_mapping()["ranges"][name]
                        for name in envelope.ranges
                        if not name.startswith("command.")
                    ),
                }
                if generated_path is not None and derived_mapping is not None
                else None
            ),
            "requirements": {
                "minimum_wheel_normal_force_n": minimum_wheel_force,
                "minimum_zmp_margin_m": 0.02,
                "maximum_arm_leg_torque_ratio": 0.7,
                "maximum_waist_torque_ratio": 0.7,
                "maximum_d6_force_torque_ratio": 0.7,
                "d6_impulse_limit": d6_impulse_limit,
                "jerk_maximum_d6_impulse_ratio": 1.0,
                "minimum_joint_limit_margin_rad": 0.02,
            },
            "measurement_sources": dict(FEASIBILITY_MEASUREMENT_SOURCES),
        }
        report_feasibility = generated_path if generated_path is not None else candidate_path
        report = build_report(
            tool="validate_feasibility",
            task=args.task,
            passed=generated_path is not None and not failures,
            feasibility_path=report_feasibility,
            reset_pose_path=reset_pose_path,
            assets=validation_input_assets(REPOSITORY_ROOT),
            additional_inputs={
                "command_candidate_config": command_candidate_path,
                "safety_threshold_authority": safety_authority_path,
            },
            metrics=metrics,
            failures=failures,
            metadata={
                "seed": args.seed,
                "steps_per_point": args.steps_per_point,
                "physics_dt": base.physics_dt,
                "policy_dt": base.step_dt,
                "physical_template_path": str(candidate_path),
                "physical_template_sha256": candidate_sha256,
                "command_candidate_config_path": str(command_candidate_path),
                "command_candidate_config_sha256": command_candidate_sha256,
                "safety_threshold_authority_path": str(safety_authority_path),
                "safety_threshold_authority_sha256": safety_authority_sha256,
                "safety_threshold_authority_id": safety_authority.authority_id,
                "derived_envelope_path": str(envelope_output),
                "range_order": list(envelope.ranges),
            },
        )
        output = write_json_atomic(args.output, report)
        print(f"feasibility scan {report['status']}: {output}")
        if generated_path is not None:
            print(f"derived feasibility envelope: {generated_path}")
        elif failures:
            print(f"failed conditions: {len(failures)} (see report for evidence)")
        return 0 if report["status"] == "passed" else 1
    finally:
        env.close()


def _write_preflight_failure(
    args: argparse.Namespace, exc: BaseException, *, traceback_text: str
) -> Path:
    """Invalidate a stale pass even when malformed inputs cannot be hashed."""

    failure = f"preflight error: {type(exc).__name__}: {exc}"
    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "tool": "validate_feasibility",
        "status": "failed",
        "task": str(args.task),
        "created_utc": utc_timestamp(),
        "inputs": {},
        "metrics": {},
        "failures": [failure],
        "metadata": {"failure_phase": "preflight", "traceback": traceback_text},
    }
    output = write_json_atomic(args.output, report)
    print(f"feasibility scan failed: {output}")
    print(f"FAIL: {failure}")
    return output


def main() -> int:
    parser = _base_parser()
    args, app_argv = parser.parse_known_args()
    try:
        envelope, safety_authority = _preflight_inputs(args)
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        if args.schema_only:
            print(f"FAIL: schema preflight error: {type(exc).__name__}: {exc}")
        else:
            _write_preflight_failure(args, exc, traceback_text=traceback_text)
        return 1
    if args.schema_only:
        print(
            "validated feasibility, command-candidate, reset-pose, and safety-authority "
            f"schemas: {envelope.source_path}, {args.candidate_config.resolve()}, "
            f"{args.reset_poses.resolve()}, {safety_authority.source_path}"
        )
        return 0

    try:
        failure_template = _failure_report_template(args)
    except BaseException as exc:
        _write_preflight_failure(args, exc, traceback_text=traceback.format_exc())
        return 1
    try:
        return _run_scan(args, app_argv, failure_template)
    except BaseException as exc:
        try:
            _write_failed_report(
                args,
                failure_template,
                exc,
                phase="kit_launch_or_outer_runtime",
                traceback_text=traceback.format_exc(),
            )
        except BaseException as report_exc:
            print(
                "FAIL: could not update the prewritten failed report: "
                f"{type(report_exc).__name__}: {report_exc}"
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
