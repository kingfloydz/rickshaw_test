"""Tests for structured offline validation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_rickshaw_lab.validation import (
    GUIDE_SCAN_RANGE_ORDER,
    SAFETY_THRESHOLD_FIELDS,
    VALIDATION_SIGNED_SLOPES,
    ValidationReportError,
    build_feasibility_scan_plan,
    build_positive_candidate_grid,
    build_report,
    compare_wrench_component,
    derive_feasibility_envelope_mapping,
    evaluate_coast_down,
    evaluate_feasibility_sample,
    load_report,
    load_safety_threshold_authority,
    select_conservative_limit,
    validation_input_assets,
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
    feasibility.write_text("schema_version: 1\n", encoding="utf-8")
    reset.write_text("poses: []\n", encoding="utf-8")
    asset.write_text("robot\n", encoding="utf-8")
    return feasibility, reset, {"robot_urdf": asset}


def _authority(tmp_path: Path, *, forbidden_source: Path | None = None) -> Path:
    sources = {}
    for name in ("implementation_guide", "reset_pose_library", "reset_alignment"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        sources[name] = {"path": path.name}
    if forbidden_source is not None:
        sources["implementation_guide"] = {"path": forbidden_source.name}
    mapping = {
        "schema_version": 1,
        "authority_id": "unit-test-authority",
        "provenance": {"method": "independent engineering review"},
        "source_files": sources,
        "thresholds": {
            name: {
                "value": value,
                "sources": list(sources),
                "rationale": f"test rationale for {name}",
            }
            for name, value in SAFETY_THRESHOLDS.items()
        },
    }
    assert set(mapping["thresholds"]) == set(SAFETY_THRESHOLD_FIELDS)
    return write_yaml_atomic(tmp_path / "safety_authority.yaml", mapping)


def test_report_round_trip_keeps_paths_status_and_diagnostics(tmp_path: Path) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    report = build_report(
        tool="validate_dynamics",
        task="task-v0",
        passed=False,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        metrics={"coast_down": {"relative_error": 0.02}},
        failures=["diagnostic mismatch"],
        metadata={"samples": 100},
    )
    path = write_json_atomic(tmp_path / "report.json", report)
    loaded = load_report(path, expected_tool="validate_dynamics")
    assert loaded["status"] == "failed"
    assert loaded["failures"] == ["diagnostic mismatch"]
    assert loaded["inputs"]["assets"]["robot_urdf"] == str(assets["robot_urdf"].resolve())


def test_report_rejects_nonfinite_metrics_and_invalid_failures(tmp_path: Path) -> None:
    feasibility, reset, assets = _inputs(tmp_path)
    report = build_report(
        tool="validate_dynamics",
        task="task-v0",
        passed=False,
        feasibility_path=feasibility,
        reset_pose_path=reset,
        assets=assets,
        metrics={"value": float("nan")},
        failures=["non-finite"],
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValidationReportError, match="finite"):
        load_report(path)

    report["metrics"] = {"value": 1.0}
    report["failures"] = [3]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValidationReportError, match="string list"):
        load_report(path)


def test_safety_authority_keeps_source_path_structure(tmp_path: Path) -> None:
    authority = load_safety_threshold_authority(_authority(tmp_path))
    assert set(authority.thresholds) == set(SAFETY_THRESHOLD_FIELDS)
    evidence = authority.evidence_record()
    assert set(evidence["source_files"]["implementation_guide"]) == {"path"}


def test_safety_authority_rejects_candidate_as_its_own_source(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("ranges: {}\n", encoding="utf-8")
    authority_path = _authority(tmp_path, forbidden_source=candidate)
    with pytest.raises(ValueError, match="independent source"):
        load_safety_threshold_authority(
            authority_path, forbidden_source_paths=(candidate,)
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
    assert compare_wrench_component(
        [100.0, 102.0, 98.0], [101.0, 100.0, 99.0], relative_tolerance=0.05
    ).passed
    assert not compare_wrench_component(
        [100.0, 100.0], [-100.0, -100.0], relative_tolerance=3.0
    ).passed


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
    assert names[:3] == ["nominal", "payload.mass:minimum", "payload.mass:maximum"]
    assert len(plan) == 1 + 2 * len(GUIDE_SCAN_RANGE_ORDER) + 2
    assert names[-2:] == [
        "cross:heavy_high_rr",
        "cross:low_friction_downhill",
    ]
    assert plan[-1].required_slope == -0.06


def test_command_candidate_selection_requires_contiguous_full_coverage() -> None:
    assert build_positive_candidate_grid((0.1, 0.3), count=3) == pytest.approx(
        (0.1, 0.2, 0.3)
    )
    selected = select_conservative_limit(
        {0.1: True, 0.2: False, 0.3: True}, safety_factor=0.8
    )
    assert selected.maximum_feasible == pytest.approx(0.1)
    with pytest.raises(ValueError, match="lowest candidate"):
        select_conservative_limit({0.1: False, 0.2: True})


def test_derived_envelope_preserves_calibration_and_writes_yaml(tmp_path: Path) -> None:
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
    output = write_yaml_atomic(tmp_path / "feasibility_envelope.yaml", derived)
    assert output.is_file()
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_asset_discovery_includes_nested_dependencies(tmp_path: Path) -> None:
    files = (
        "assets/g1_dex1/g1_29dof_mode_15_with_dex1_1.urdf",
        "assets/g1_dex1/g1_29dof_mode_15_with_dex1_1.usd",
        "assets/g1_dex1/configuration/physics.usd",
        "assets/g1_dex1/meshes/link.STL",
        "assets/rickshaw/rickshaw.urdf",
        "assets/rickshaw/rickshaw.usd",
        "assets/rickshaw/body.stl",
    )
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    assets = validation_input_assets(tmp_path)
    assert "g1_dex1/configuration/physics.usd" in assets
    assert "g1_dex1/meshes/link.STL" in assets
    assert "rickshaw/body.stl" in assets
