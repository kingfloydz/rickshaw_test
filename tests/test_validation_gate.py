"""Tests for content-addressed pre-training validation gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from g1_rickshaw_lab.validation import (
    FEASIBILITY_MEASUREMENT_SOURCES,
    GUIDE_SCAN_RANGE_ORDER,
    RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT,
    SAFETY_THRESHOLD_FIELDS,
    VALIDATION_SIGNED_SLOPES,
    ValidationGateError,
    build_feasibility_scan_plan,
    build_positive_candidate_grid,
    build_report,
    compare_wrench_component,
    derive_feasibility_envelope_mapping,
    evaluate_coast_down,
    evaluate_feasibility_sample,
    load_report,
    load_safety_threshold_authority,
    reset_dynamics_feasibility_sha256,
    select_conservative_limit,
    validate_training_gate,
    validate_safety_authority_source_evidence,
    validation_input_assets,
    validation_runtime_sources,
    asset_hashes,
    write_json_atomic,
    write_yaml_atomic,
)

SAFETY_THRESHOLDS = {
    "safety.theta_max": 0.5,
    "safety.illegal_contact_force_threshold": 5.0,
    "safety.robot_cart_contact_force_threshold": 5.0,
    "safety.cart_ground_contact_force_threshold": 5.0,
    "safety.minimum_wheel_normal_force": 25.0,
    "safety.min_ground_reaction": 100.0,
    "safety.d6_residual_limit": 0.06,
    "safety.d6_impulse_limit": 1.7,
    "safety.hitch_height_bounds": [0.65, 0.85],
    "safety.rickshaw_pitch_bounds": [0.15, 0.45],
    "safety.corridor_half_width": 0.3,
    "safety.heading_error_limit": 0.3,
    "safety.overspeed_margin": 0.25,
    "safety.arm_torque_limit": 0.9,
}


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    feasibility = tmp_path / "feasibility.yaml"
    reset = tmp_path / "reset.yaml"
    asset = tmp_path / "robot.urdf"
    feasibility.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slopes": list(VALIDATION_SIGNED_SLOPES),
                "joint_order": [],
                "ranges": {
                    "payload.mass": {"min": 0.0, "max": 10.0},
                    "command.acceleration_limit": {"min": 0.1, "max": 0.2},
                    "command.jerk_limit": {"min": 0.5, "max": 1.0},
                },
                "calibration": SAFETY_THRESHOLDS,
            }
        ),
        encoding="utf-8",
    )
    reset.write_text("reset\n", encoding="utf-8")
    asset.write_text("asset\n", encoding="utf-8")
    return feasibility, reset, {"robot_urdf": asset}


def _safety_authority(
    tmp_path: Path,
    *,
    assets: dict[str, Path] | None = None,
    feasibility: Path | None = None,
    reset_pose: Path | None = None,
) -> Path:
    guide = tmp_path / "implementation_guide.txt"
    guide.write_text("implementation guide evidence\n", encoding="utf-8")
    reset_alignment = tmp_path / "reset_alignment.json"
    reset_alignment.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tool": "validate_reset_alignment",
                "status": "passed",
                "task": "task-v0",
                "steps": 1000,
                "slopes": list(VALIDATION_SIGNED_SLOPES),
                "seed": 42,
                "continuous_standing": True,
                "sample_physics_ranges": False,
                "torque_measurement_contract": dict(
                    RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT
                ),
                "safety_thresholds": {
                    "arm_torque_ratio": 0.9,
                    "d6_residual_m_or_rad": 0.06,
                    "d6_impulse_n_s": 1.7,
                    "static_lower_preload_ratio": 0.7,
                    "static_arm_preload_ratio": 0.98,
                },
                "summary": {
                    "rollout_d6_residual_max_m_or_rad": 0.03,
                    "rollout_d6_impulse_max_n_s": 1.5,
                    "rollout_arm_torque_ratio_max": 0.3,
                    "static_preload_arm_hardware_ratio_max": 0.3,
                    "continuous_standing_abs_torso_pitch_max_rad": 0.25,
                    "checks": {"all_test_checks": True},
                },
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    if reset_pose is None:
        reset_pose = tmp_path / "authority_reset_poses.yaml"
        reset_pose.write_text("reset poses\n", encoding="utf-8")
    if feasibility is None:
        feasibility = tmp_path / "authority_feasibility.yaml"
        feasibility.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "slopes": list(VALIDATION_SIGNED_SLOPES),
                    "joint_order": [],
                    "ranges": {"payload.mass": {"min": 0.0, "max": 1.0}},
                    "calibration": {"control.stiffness": 1.0},
                }
            ),
            encoding="utf-8",
        )
    reset_alignment_mapping = json.loads(reset_alignment.read_text(encoding="utf-8"))
    reset_alignment_mapping["inputs"] = {
        "feasibility_path": str(feasibility.resolve()),
        "feasibility_reset_dynamics_sha256": reset_dynamics_feasibility_sha256(
            feasibility
        ),
        "reset_pose_path": str(reset_pose.resolve()),
        "reset_pose_sha256": hashlib.sha256(reset_pose.read_bytes()).hexdigest(),
        "assets": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (assets or {}).items()
        },
        "runtime_sources": asset_hashes(validation_runtime_sources()),
    }
    reset_alignment.write_text(json.dumps(reset_alignment_mapping), encoding="utf-8")
    sources = {
        "implementation_guide": guide,
        "reset_pose_library": reset_pose,
        "reset_alignment": reset_alignment,
    }
    source_files = {
        name: {
            "path": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        for name, source in sources.items()
    }
    authority = {
        "schema_version": 1,
        "authority_id": "unit-test-safety-authority",
        "provenance": {"method": "independent unit-test acceptance record"},
        "source_files": source_files,
        "thresholds": {
            name: {
                "value": value,
                "sources": list(source_files),
                "rationale": f"independent test rationale for {name}",
            }
            for name, value in SAFETY_THRESHOLDS.items()
        },
    }
    assert set(authority["thresholds"]) == set(SAFETY_THRESHOLD_FIELDS)
    return write_yaml_atomic(tmp_path / "safety_authority.yaml", authority)


def _point_names() -> tuple[str, ...]:
    names = ["nominal"]
    for name in GUIDE_SCAN_RANGE_ORDER:
        names.extend((f"{name}:minimum", f"{name}:maximum"))
    names.extend(
        (
            "cross:heavy_high_rr",
            "cross:low_friction_downhill",
            "cross:soft_d6_heavy",
        )
    )
    return tuple(names)


def _point_parameters(point: str) -> dict[str, float]:
    lows = {name: float(index) for index, name in enumerate(GUIDE_SCAN_RANGE_ORDER)}
    highs = {name: value + 1.0 for name, value in lows.items()}
    nominal = {name: 0.5 * (lows[name] + highs[name]) for name in GUIDE_SCAN_RANGE_ORDER}
    result = dict(nominal)
    if point.endswith(":minimum"):
        name = point.removesuffix(":minimum")
        result[name] = lows[name]
    elif point.endswith(":maximum"):
        name = point.removesuffix(":maximum")
        result[name] = highs[name]
    elif point == "cross:heavy_high_rr":
        for name in ("payload.mass", "payload.com.x", "rolling_resistance.c_rr"):
            result[name] = highs[name]
    elif point == "cross:low_friction_downhill":
        result["terrain.friction"] = lows["terrain.friction"]
    elif point == "cross:soft_d6_heavy":
        result["payload.mass"] = highs["payload.mass"]
        for name in (
            "d6.linear_stiffness",
            "d6.linear_damping",
            "d6.angular_stiffness",
            "d6.angular_damping",
        ):
            result[name] = lows[name]
    return result


def _search_evidence(
    *, stage: str, candidates: tuple[float, float], target_acceleration: float
) -> dict:
    points = _point_names()
    expected_per_candidate = len(points) * len(VALIDATION_SIGNED_SLOPES) * 2
    trials = []
    coverage = {}
    passed_map = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate_passed = candidate_index == 0
        key = format(candidate, ".12g")
        passed_map[key] = candidate_passed
        coverage[key] = {
            "complete": True,
            "all_physical_endpoints_passed": candidate_passed,
            "expected_trials": expected_per_candidate,
            "observed_trials": expected_per_candidate,
            "passed_trials": expected_per_candidate if candidate_passed else 0,
            "slope_coverage": {
                f"{slope:+.2f}": {
                    "observed": len(points) * 2,
                    "passed": len(points) * 2 if candidate_passed else 0,
                }
                for slope in VALIDATION_SIGNED_SLOPES
            },
        }
        for point_index, point in enumerate(points):
            for slope in VALIDATION_SIGNED_SLOPES:
                for direction in (-1.0, 1.0):
                    target = candidate if stage == "acceleration" else target_acceleration
                    trials.append(
                        {
                            "stage": stage,
                            "point_index": point_index,
                            "point": point,
                            "slope": slope,
                            "force_direction": int(direction),
                            "candidate": candidate,
                            "parameters": _point_parameters(point),
                            "metrics": {
                                "left_wheel_normal_force": 100.0,
                                "right_wheel_normal_force": 100.0,
                                "friction_cone_margin": 0.2,
                                "zmp_margin": 0.03,
                                "arm_torque_ratio": 0.5,
                                "leg_torque_ratio": 0.5,
                                "waist_torque_ratio": 0.5,
                                "d6_force_ratio": 0.5,
                                "d6_torque_ratio": 0.5,
                                "d6_impulse_ratio": 0.5,
                                "joint_limit_margin": 0.05,
                                "finite": True,
                                "terminated": False,
                            },
                            "dynamic_evidence": {
                                "cart_mass_kg": 40.04,
                                "target_equivalent_acceleration_mps2": target,
                                "applied_force_peak_n": 40.04 * target,
                                "applied_force_peak_w_n": [40.04 * target, 0.0, 0.0],
                                "force_body": "base_link",
                                "force_api": (
                                    "Articulation.permanent_wrench_composer."
                                    "set_forces_and_torques"
                                ),
                                "arm_actuator_effort_limit_nm": {
                                    "minimum": 13.4,
                                    "maximum": 25.0,
                                },
                                "leg_actuator_effort_limit_nm": {
                                    "minimum": 50.0,
                                    "maximum": 139.0,
                                },
                                "waist_actuator_effort_limit_nm": {
                                    "minimum": 50.0,
                                    "maximum": 88.0,
                                },
                                "ramp_jerk_mps3": 0.5,
                                "ramp_steps": 10,
                                "hold_steps": 50,
                                "measured_cart_acceleration_mps2": {
                                    "minimum": -0.1,
                                    "maximum": 0.1,
                                    "mean": 0.0,
                                    "peak_absolute": 0.1,
                                    "sample_count": 50,
                                },
                            },
                            "passed": candidate_passed,
                            "failures": [] if candidate_passed else ["candidate boundary"],
                        }
                    )
    suffix = "mps2" if stage == "acceleration" else "mps3"
    result = {
        "candidate_full_coverage_passed": passed_map,
        "coverage": coverage,
        f"maximum_fully_feasible_{suffix}": candidates[0],
        f"derived_limit_{suffix}": 0.8 * candidates[0],
        "safety_factor": 0.8,
        "trials": trials,
    }
    result["candidates_mps2" if stage == "acceleration" else "candidates_mps3"] = list(candidates)
    if stage == "jerk":
        result["acceleration_held_at_derived_limit_mps2"] = target_acceleration
        result["d6_impulse_threshold_ratio"] = 1.0
    return result


def _feasibility_evidence(
    feasibility: Path, candidates: Path, safety_authority: Path
) -> dict:
    acceleration = _search_evidence(
        stage="acceleration", candidates=(0.1, 0.2), target_acceleration=0.1
    )
    acceleration_limit = acceleration["derived_limit_mps2"]
    authority = load_safety_threshold_authority(
        safety_authority, forbidden_source_paths=(feasibility,)
    )
    return {
        "coverage": "full",
        "physical_scan_points": list(_point_names()),
        "physical_scan_point_count": len(_point_names()),
        "slopes": list(VALIDATION_SIGNED_SLOPES),
        "force_directions": [-1.0, 1.0],
        "force_definition": {
            "body": "base_link",
            "frame": "world",
            "direction": "signed path tangent",
            "magnitude": "actual PhysX cart mass * equivalent acceleration ramp",
        },
        "command_candidate_config": {
            "path": str(candidates.resolve()),
            "sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
            "cli_acceleration_override": None,
            "cli_jerk_override": None,
        },
        "safety_threshold_authority": authority.evidence_record(),
        "acceleration_search": acceleration,
        "jerk_search": _search_evidence(
            stage="jerk", candidates=(0.5, 1.0), target_acceleration=acceleration_limit
        ),
        "generated_envelope": {
            "path": str(feasibility.resolve()),
            "sha256": hashlib.sha256(feasibility.read_bytes()).hexdigest(),
            "physical_ranges_preserved": True,
        },
        "requirements": {
            "minimum_wheel_normal_force_n": 25.0,
            "minimum_zmp_margin_m": 0.02,
            "maximum_arm_leg_torque_ratio": 0.7,
            "maximum_waist_torque_ratio": 0.7,
            "maximum_d6_force_torque_ratio": 0.7,
            "d6_impulse_limit": 1.7,
            "jerk_maximum_d6_impulse_ratio": 1.0,
            "minimum_joint_limit_margin_rad": 0.02,
        },
        "measurement_sources": dict(FEASIBILITY_MEASUREMENT_SOURCES),
    }


def _dynamics_evidence() -> tuple[dict, dict]:
    comparison = {
        "analytic_mean": 100.0,
        "measured_mean": 100.0,
        "relative_error": 0.0,
        "same_sign": True,
        "passed": True,
    }
    slopes = {
        "flat_static": 0.0,
        "flat_constant_speed": 0.0,
        "uphill_acceleration": 0.06,
        "downhill_braking": -0.06,
    }
    metrics = {
        "coast_down": {
            "measured_force_n": 8.008,
            "expected_force_n": 8.008,
            "relative_error": 0.0,
            "deceleration_delta_mps2": 0.2,
            "passed": True,
            "mass_kg": 40.04,
            "masses_kg": [40.04, 40.04],
            "mean_normal_force_n": 400.4,
            "c_rr": 0.02,
            "acceleration_without_rr_mps2": -0.01,
            "acceleration_with_rr_mps2": -0.21,
            "sample_count": 90,
        },
        "d6_analytic_conditions": {
            name: {
                "slope": slope,
                "tangential": dict(comparison),
                "normal": dict(comparison),
                "analytic_valid_entire_window": True,
                "terminated": False,
            }
            for name, slope in slopes.items()
        },
    }
    metadata = {
        "seed": 42,
        "physics_dt": 0.005,
        "policy_dt": 0.02,
        "settling_steps": 40,
        "measurement_steps": 120,
        "window_start": 30,
        "coast_speed_mps": 0.8,
        "coast_relative_tolerance": 0.2,
        "wrench_relative_tolerance": 0.35,
        "wrench_absolute_floor_n": 12.0,
        "policy_safety_terminations_disabled": True,
        "coast_normal_force_source": "level_vehicle_weight",
        "coast_rail_free_axes": ["transX"],
        "controlled_pelvis_force_n": {
            "flat_static": 0.0,
            "flat_constant_speed": 0.0,
            "uphill_acceleration": 300.0,
            "downhill_braking": -180.0,
        },
        "coast_wheel_force_location": "wheel centers",
        "measured_wrench_source": "whole_cart_momentum_balance",
        "ground_contact_force_source": "two_wheel_contact_sensor_net_forces",
        "incoming_joint_wrench_role": "constraint_residual_impulse_proxy_only",
        "d6_force_sign": "momentum balance gives robot-on-cart; hand force is opposite",
    }
    return metrics, metadata


def _write_valid_reports(
    validation_dir: Path,
    *,
    feasibility: Path,
    reset: Path,
    assets: dict[str, Path],
    candidates: Path,
    runtime_sources: dict[str, Path] | None = None,
) -> Path:
    dynamics_metrics, dynamics_metadata = _dynamics_evidence()
    safety_authority = _safety_authority(
        validation_dir.parent,
        assets=assets,
        feasibility=feasibility,
        reset_pose=reset,
    )
    for tool in ("validate_feasibility", "validate_dynamics"):
        report = build_report(
            tool=tool,
            task="task-v0",
            passed=True,
            feasibility_path=feasibility,
            reset_pose_path=reset,
            assets=assets,
            additional_inputs=(
                {
                    "command_candidate_config": candidates,
                    "safety_threshold_authority": safety_authority,
                }
                if tool == "validate_feasibility"
                else None
            ),
            runtime_sources=runtime_sources,
            metrics=(
                _feasibility_evidence(feasibility, candidates, safety_authority)
                if tool == "validate_feasibility"
                else dynamics_metrics
            ),
            metadata={} if tool == "validate_feasibility" else dynamics_metadata,
        )
        write_json_atomic(
            validation_dir / f"{tool.removeprefix('validate_')}_report.json", report
        )
    return safety_authority


def test_training_gate_rejects_stale_or_failed_reports(tmp_path: Path) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )

    reports = validate_training_gate(
        validation_dir,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        task="task-v0",
    )
    assert set(reports) == {"validate_feasibility", "validate_dynamics"}

    assets["robot_urdf"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValidationGateError, match="stale"):
        validate_training_gate(
            validation_dir,
            feasibility_path=feasibility,
            reset_pose_path=reset,
            assets=assets,
            task="task-v0",
        )

    assets["robot_urdf"].write_text("asset\n", encoding="utf-8")
    path = validation_dir / "dynamics_report.json"
    failed = json.loads(path.read_text(encoding="utf-8"))
    failed["status"] = "failed"
    failed["failures"] = ["coast-down mismatch"]
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValidationGateError, match="did not pass"):
        validate_training_gate(
            validation_dir,
            feasibility_path=feasibility,
            reset_pose_path=reset,
            assets=assets,
            task="task-v0",
        )


def test_training_gate_rejects_stale_tool_specific_additional_input(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )
    validate_training_gate(
        validation_dir,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        task="task-v0",
    )
    candidates.write_text("candidates: [0.1, 0.3]\n", encoding="utf-8")
    with pytest.raises(ValidationGateError, match="additional input"):
        validate_training_gate(
            validation_dir,
            feasibility_path=feasibility,
            reset_pose_path=reset,
            assets=assets,
            task="task-v0",
        )


def test_feasibility_gate_rejects_stale_safety_authority_provenance(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    authority_path = _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )
    authority = load_safety_threshold_authority(
        authority_path, forbidden_source_paths=(feasibility,)
    )
    authority.sources["reset_alignment"].path.write_text(
        "changed reset evidence\n", encoding="utf-8"
    )

    with pytest.raises(ValidationGateError, match="authority.*stale"):
        load_report(
            validation_dir / "feasibility_report.json",
            expected_tool="validate_feasibility",
        )


def test_reset_alignment_authority_rejects_stale_runtime_binding(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    authority_path = _safety_authority(
        tmp_path, assets=assets, feasibility=feasibility, reset_pose=reset
    )
    authority = load_safety_threshold_authority(
        authority_path, forbidden_source_paths=(feasibility,)
    )
    reset_report_path = authority.sources["reset_alignment"].path
    reset_report = json.loads(reset_report_path.read_text(encoding="utf-8"))
    runtime_sources = reset_report["inputs"]["runtime_sources"]
    first_source = next(iter(runtime_sources))
    runtime_sources[first_source] = "0" * 64
    reset_report_path.write_text(json.dumps(reset_report), encoding="utf-8")

    with pytest.raises(ValidationGateError, match="stale for the current runtime sources"):
        validate_safety_authority_source_evidence(
            authority,
            task="task-v0",
            assets_sha256={
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in assets.items()
            },
            feasibility_path=feasibility,
            reset_pose_path=reset,
            reset_pose_sha256=hashlib.sha256(reset.read_bytes()).hexdigest(),
        )


def test_safety_authority_rejects_feasibility_envelope_as_provenance(
    tmp_path: Path,
) -> None:
    feasibility, reset, _ = _inputs(tmp_path)
    authority_path = _safety_authority(tmp_path, reset_pose=reset)
    import yaml

    authority_mapping = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    authority_mapping["source_files"]["implementation_guide"] = {
        "path": feasibility.name,
        "sha256": hashlib.sha256(feasibility.read_bytes()).hexdigest(),
    }
    write_yaml_atomic(authority_path, authority_mapping)

    with pytest.raises(ValueError, match="independent source"):
        load_safety_threshold_authority(
            authority_path, forbidden_source_paths=(feasibility,)
        )


def test_feasibility_gate_rejects_envelope_threshold_not_from_authority(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )
    envelope = json.loads(feasibility.read_text(encoding="utf-8"))
    envelope["calibration"]["safety.d6_impulse_limit"] = 1.8
    feasibility.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValidationGateError, match="not bound to the safety authority"):
        load_report(
            validation_dir / "feasibility_report.json",
            expected_tool="validate_feasibility",
        )


def test_training_gate_rejects_empty_passed_evidence(tmp_path: Path) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    report = build_report(
        tool="validate_dynamics",
        task="task-v0",
        passed=True,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        metrics={},
    )
    path = tmp_path / "dynamics_report.json"
    write_json_atomic(path, report)
    with pytest.raises(ValidationGateError, match="lacks required evidence"):
        load_report(path, expected_tool="validate_dynamics")


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        (
            "measured_wrench_source",
            "incoming_joint_wrench",
            "whole-cart momentum balance",
        ),
        (
            "incoming_joint_wrench_role",
            "physical_hand_force",
            "isolated from the physical force gate",
        ),
        (
            "ground_contact_force_source",
            "all_cart_contacts",
            "isolate wheel-ground contact force",
        ),
    ),
)
def test_dynamics_gate_rejects_incoming_joint_wrench_as_physical_force(
    tmp_path: Path,
    field: str,
    invalid_value: str,
    message: str,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    metrics, metadata = _dynamics_evidence()
    metadata[field] = invalid_value
    report = build_report(
        tool="validate_dynamics",
        task="task-v0",
        passed=True,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        metrics=metrics,
        metadata=metadata,
    )
    path = tmp_path / "dynamics_report.json"
    write_json_atomic(path, report)

    with pytest.raises(ValidationGateError, match=message):
        load_report(path, expected_tool="validate_dynamics")


def test_training_gate_rejects_mislabeled_physical_scan_parameters(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )

    path = validation_dir / "feasibility_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for search_name in ("acceleration_search", "jerk_search"):
        for row in report["metrics"][search_name]["trials"]:
            if row["point"] == "payload.mass:minimum":
                row["parameters"]["payload.com.x"] += 0.25
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValidationGateError, match="changes parameters other than"):
        load_report(path, expected_tool="validate_feasibility")


def test_feasibility_gate_rejects_nonphysical_measurement_source(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )
    path = validation_dir / "feasibility_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["metrics"]["measurement_sources"]["arm_leg_torque_ratio"] = (
        "PhysX solver effort limit"
    )
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValidationGateError, match="physical measurement sources"):
        load_report(path, expected_tool="validate_feasibility")


def test_feasibility_gate_requires_waist_actuator_limit_evidence(
    tmp_path: Path,
) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )
    path = validation_dir / "feasibility_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    del report["metrics"]["acceleration_search"]["trials"][0]["dynamic_evidence"][
        "waist_actuator_effort_limit_nm"
    ]
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValidationGateError, match="waist_actuator_effort_limit_nm"):
        load_report(path, expected_tool="validate_feasibility")


def test_feasibility_gate_rejects_waist_torque_above_limit(tmp_path: Path) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
    )
    path = validation_dir / "feasibility_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["metrics"]["acceleration_search"]["trials"][0]["metrics"][
        "waist_torque_ratio"
    ] = 1.000001
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValidationGateError, match="claims pass with infeasible metrics"):
        load_report(path, expected_tool="validate_feasibility")


def test_training_gate_rejects_stale_runtime_source(tmp_path: Path) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates: [0.1, 0.2]\n", encoding="utf-8")
    runtime_source = tmp_path / "dynamics_source.py"
    runtime_source.write_text("VERSION = 1\n", encoding="utf-8")
    runtime_sources = {"dynamics_source.py": runtime_source}
    validation_dir = tmp_path / "validation"
    _write_valid_reports(
        validation_dir,
        feasibility=feasibility,
        reset=reset,
        assets=assets,
        candidates=candidates,
        runtime_sources=runtime_sources,
    )
    validate_training_gate(
        validation_dir,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        task="task-v0",
        runtime_sources=runtime_sources,
    )
    runtime_source.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(ValidationGateError, match="task/validator sources"):
        validate_training_gate(
            validation_dir,
            feasibility_path=feasibility,
            reset_pose_path=reset,
            assets=assets,
            task="task-v0",
            runtime_sources=runtime_sources,
        )


def test_dynamics_acceptance_math() -> None:
    coast = evaluate_coast_down(
        mass_kg=40.04,
        mean_normal_force_n=400.4,
        c_rr=0.02,
        acceleration_without_rr_mps2=-0.01,
        acceleration_with_rr_mps2=-0.21,
        relative_tolerance=0.05,
    )
    assert coast.passed
    assert coast.measured_force_n == pytest.approx(8.008)

    match = compare_wrench_component(
        [100.0, 102.0, 98.0], [101.0, 100.0, 99.0], relative_tolerance=0.05
    )
    assert match.passed
    wrong_sign = compare_wrench_component(
        [100.0, 100.0], [-100.0, -100.0], relative_tolerance=3.0
    )
    assert not wrong_sign.passed


def test_feasibility_sample_requires_every_constraint() -> None:
    metrics = {
        "left_wheel_normal_force": 100.0,
        "right_wheel_normal_force": 100.0,
        "friction_cone_margin": 0.2,
        "zmp_margin": 0.03,
        "arm_torque_ratio": 0.5,
        "leg_torque_ratio": 0.6,
        "waist_torque_ratio": 0.4,
        "d6_force_ratio": 0.5,
        "d6_torque_ratio": 0.5,
        "joint_limit_margin": 0.05,
        "finite": True,
    }
    passed, failures = evaluate_feasibility_sample(
        metrics, minimum_wheel_normal_force=25.0
    )
    assert passed and failures == []

    metrics["arm_torque_ratio"] = 1.0
    metrics["leg_torque_ratio"] = 1.0
    metrics["waist_torque_ratio"] = 1.0
    passed, failures = evaluate_feasibility_sample(
        metrics, minimum_wheel_normal_force=25.0
    )
    assert passed and failures == []
    metrics["leg_torque_ratio"] = 1.000001
    passed, failures = evaluate_feasibility_sample(
        metrics, minimum_wheel_normal_force=25.0
    )
    assert not passed
    assert any("arm/leg/waist torque" in failure for failure in failures)

    metrics["leg_torque_ratio"] = 0.6
    metrics["waist_torque_ratio"] = 1.000001
    passed, failures = evaluate_feasibility_sample(
        metrics, minimum_wheel_normal_force=25.0
    )
    assert not passed
    assert any("arm/leg/waist torque" in failure for failure in failures)

    metrics["arm_torque_ratio"] = 0.5
    metrics["leg_torque_ratio"] = 0.6
    metrics["waist_torque_ratio"] = 0.5
    metrics["zmp_margin"] = 0.019
    metrics["d6_force_ratio"] = 0.71
    passed, failures = evaluate_feasibility_sample(
        metrics, minimum_wheel_normal_force=25.0
    )
    assert not passed
    assert any("ZMP" in failure for failure in failures)
    assert any("D6" in failure for failure in failures)


def test_feasibility_scan_plan_has_stable_endpoints_and_crosses() -> None:
    ranges = {
        name: (float(index), float(index + 1))
        for index, name in enumerate(GUIDE_SCAN_RANGE_ORDER)
    }
    plan = build_feasibility_scan_plan(ranges)
    names = [point.name for point in plan]

    assert names[:3] == [
        "nominal",
        "payload.mass:minimum",
        "payload.mass:maximum",
    ]
    assert len(plan) == 1 + 2 * len(GUIDE_SCAN_RANGE_ORDER) + 3
    assert names[-3:] == [
        "cross:heavy_high_rr",
        "cross:low_friction_downhill",
        "cross:soft_d6_heavy",
    ]
    assert plan[-2].required_slope == -0.06
    assert "command.acceleration_limit" not in GUIDE_SCAN_RANGE_ORDER
    assert "command.jerk_limit" not in GUIDE_SCAN_RANGE_ORDER
    assert plan[-1].values["payload.mass"] == ranges["payload.mass"][1]
    assert plan[-1].values["d6.linear_stiffness"] == ranges["d6.linear_stiffness"][0]


def test_command_candidate_selection_requires_contiguous_full_coverage() -> None:
    assert build_positive_candidate_grid((0.1, 0.3), count=3) == pytest.approx(
        (0.1, 0.2, 0.3)
    )
    assert build_positive_candidate_grid((0.6, 0.6), count=5) == (0.6,)

    selected = select_conservative_limit(
        {0.1: True, 0.2: True, 0.3: False}, safety_factor=0.8
    )
    assert selected.maximum_feasible == pytest.approx(0.2)
    assert selected.derived_limit == pytest.approx(0.16)
    # A later isolated pass cannot bridge an unfeasible candidate.
    selected = select_conservative_limit(
        {0.1: True, 0.2: False, 0.3: True}, safety_factor=0.8
    )
    assert selected.maximum_feasible == pytest.approx(0.1)
    with pytest.raises(ValueError, match="lowest candidate"):
        select_conservative_limit({0.1: False, 0.2: True})


def test_reset_dynamics_projection_excludes_scan_outputs_and_authority_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feasibility.yaml"
    mapping = {
        "schema_version": 1,
        "slopes": list(VALIDATION_SIGNED_SLOPES),
        "joint_order": ["joint"],
        "ranges": {
            "payload.mass": {"min": 0.0, "max": 10.0},
            "command.acceleration_limit": {"min": 0.1, "max": 0.1},
            "command.jerk_limit": {"min": 0.5, "max": 0.5},
        },
        "calibration": {
            "control.stiffness": 300.0,
            "safety.arm_torque_limit": 0.9,
        },
    }
    path.write_text(json.dumps(mapping), encoding="utf-8")
    baseline = reset_dynamics_feasibility_sha256(path)
    mapping["ranges"]["command.acceleration_limit"] = {"min": 0.2, "max": 0.2}
    mapping["calibration"]["safety.arm_torque_limit"] = 0.8
    path.write_text(json.dumps(mapping), encoding="utf-8")
    assert reset_dynamics_feasibility_sha256(path) == baseline
    mapping["ranges"]["payload.mass"]["max"] = 11.0
    path.write_text(json.dumps(mapping), encoding="utf-8")
    assert reset_dynamics_feasibility_sha256(path) != baseline

def test_derived_envelope_preserves_physics_and_writes_yaml_atomically(
    tmp_path: Path,
) -> None:
    candidate = {
        "schema_version": 1,
        "ranges": {
            "payload.mass": {"min": 0.0, "max": 10.0},
            "command.acceleration_limit": {"min": 0.1, "max": 0.4},
            "command.jerk_limit": {"min": 0.5, "max": 2.0},
        },
        "calibration": {"sentinel": [1.0, 2.0]},
    }
    derived = derive_feasibility_envelope_mapping(
        candidate, acceleration_limit=0.24, jerk_limit=1.2
    )
    assert derived["ranges"]["payload.mass"] == candidate["ranges"]["payload.mass"]
    assert derived["calibration"] == candidate["calibration"]
    assert derived["ranges"]["command.acceleration_limit"] == {
        "min": 0.24,
        "max": 0.24,
    }
    assert derived["ranges"]["command.jerk_limit"] == {
        "min": 1.2,
        "max": 1.2,
    }
    # The candidate mapping is not mutated while deriving the output authority.
    assert candidate["ranges"]["command.acceleration_limit"]["max"] == 0.4

    output = write_yaml_atomic(tmp_path / "feasibility_envelope.yaml", derived)
    assert output.is_file()
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    import yaml

    assert yaml.safe_load(output.read_text(encoding="utf-8")) == derived


def test_asset_provenance_includes_nested_usd_and_mesh_dependencies(tmp_path: Path) -> None:
    files = (
        "assets/g1_dex1/g1_29dof_mode_15_with_dex1_1.urdf",
        "assets/g1_dex1/g1_29dof_mode_15_with_dex1_1.usd",
        "assets/g1_dex1/configuration/physics.usd",
        "assets/g1_dex1/meshes/link.STL",
        "assets/rickshaw/rickshaw.urdf",
        "assets/rickshaw/rickshaw.usd",
        "assets/rickshaw/configuration/physics.usd",
        "assets/rickshaw/body.stl",
    )
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    ignored = tmp_path / "assets/rickshaw/.asset_hash"
    ignored.write_text("generated", encoding="utf-8")

    assets = validation_input_assets(tmp_path)
    assert "g1_dex1/configuration/physics.usd" in assets
    assert "g1_dex1/meshes/link.STL" in assets
    assert "rickshaw/body.stl" in assets
    assert "rickshaw/.asset_hash" not in assets
