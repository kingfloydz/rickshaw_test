"""Pure constraint-ranking tests for the 19-slope reset-pose solver."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import stat
import subprocess
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import solve_reset_poses as solver_module  # noqa: E402
from solve_reset_poses import (  # noqa: E402
    DEFAULT_URDF,
    DEFAULT_HARDWARE_TORQUE_LIMIT,
    _allocate_support_torques,
    _assembled_validation_report_errors,
    _assembled_validation_command,
    _bind_reset_pose_library,
    _build_parser,
    _candidate_contract,
    _candidate_constraint_violation,
    _candidate_output_mapping,
    _candidate_rank_key,
    _commit_pipeline_publications,
    _foot_contact_geometry,
    _per_foot_support_wrench_ratios,
    _prepare_atomic_text,
    _solver_worker_count,
    _stage_a_solve_plan,
    _load_candidate_progress,
    _static_load_scales,
    _retarget_validation_report,
    _run_multistarts,
    _run_candidate_validation_process,
    _run_isolated_candidate_rollouts,
    _run_pipeline_parent,
    _validate_arguments,
    _write_candidate_progress,
    _worst_case_torque_ratios,
    _write_text_atomic,
    main,
)
from g1_rickshaw_lab.static_equilibrium import solve_fixed_contact_statics  # noqa: E402


def _synthetic_reset_pose_mapping() -> dict[str, object]:
    pose_fields = {
        "root_pitch": 0.0,
        "root_height": 0.75,
        "q_reset": [0.0] * 29,
        "q_ref_unloaded": [0.0] * 29,
        "tau_unloaded": [0.0] * 29,
        "tau_per_tangent_force": [0.0] * 29,
        "tau_per_normal_force": [0.0] * 29,
        "tau_per_tangent_difference": [0.0] * 29,
        "handle_wrenches_sln": [[0.0] * 6, [0.0] * 6],
        "wheel_contact_forces_sln": [[0.0, 0.0, 100.0]] * 2,
        "q_ref": [0.0] * 29,
    }
    return {
        "schema_version": 4,
        "joint_order": list(solver_module.G1_JOINT_ORDER),
        "poses": [
            {"gradient": float(gradient), **pose_fields}
            for gradient in solver_module.SLOPE_GRADIENTS
        ],
    }


def test_solver_import_does_not_load_isaac_runtime_before_app_launcher() -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SCRIPTS_ROOT)!r})\n"
        "import solve_reset_poses\n"
        "polluted = sorted(name for name in sys.modules "
        "if name == 'warp' or name.startswith('warp.') "
        "or name == 'isaaclab' or name.startswith('isaaclab.'))\n"
        "assert not polluted, polluted\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=SCRIPTS_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_solver_help_exits_successfully() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "solve_reset_poses.py"),
            "--help",
        ],
        cwd=SCRIPTS_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: solve_reset_poses.py" in result.stdout
    assert "--validate-existing" in result.stdout
    assert "--alignment-output" in result.stdout
    assert "Traceback" not in result.stderr


def test_canonical_output_is_the_default() -> None:
    assert _build_parser().parse_args([]).output == Path("config/reset_poses.yaml")


def test_candidate_cache_contract_binds_stage_a_inputs() -> None:
    arguments = SimpleNamespace(
        urdf=str(DEFAULT_URDF),
        seed_library=None,
        seed=42,
        output=Path("one.yaml"),
        report_output=Path("one.json"),
        reuse_candidates=False,
    )
    original = _candidate_contract(arguments)
    same_inputs = _candidate_contract(
        SimpleNamespace(**{**vars(arguments), "output": Path("two.yaml")})
    )
    changed_seed = _candidate_contract(
        SimpleNamespace(**{**vars(arguments), "seed": 43})
    )

    assert original == same_inputs
    assert original != changed_seed
    assert set(original) == {
        "arguments",
        "configured_slopes",
        "stage_a_solve_plan",
    }
    assert original["configured_slopes"] == list(solver_module.SLOPE_GRADIENTS)


def test_stage_a_extended_uphill_continuation_uses_positive_parent() -> None:
    ordered_plan = _stage_a_solve_plan()
    plan = dict(ordered_plan)

    assert plan[0.09] == 0.08
    assert plan[0.10] == 0.09
    assert plan[-0.01] == 0.0
    assert plan[-0.08] == -0.07
    gradients = [gradient for gradient, _parent in ordered_plan]
    assert gradients.index(0.10) < gradients.index(-0.01)


def test_partial_candidate_cache_round_trips_completed_slopes(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args([])
    slope = 0.0
    pose = next(
        row
        for row in _synthetic_reset_pose_mapping()["poses"]
        if row["gradient"] == slope
    )
    diagnostics = [{"gradient": slope, "hand_x": 0.19, "foot_x": -0.035}]
    candidate_bank = {
        slope: [
            {
                "candidate_id": 1,
                "pose": pose,
                "static_metrics": _rank_metrics(),
            }
        ]
    }
    mapping = _candidate_output_mapping(
        diagnostics, candidate_bank, args.full_pose_multistarts, args
    )
    path = tmp_path / "partial_candidates.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    solved, loaded_diagnostics, loaded_candidates = _load_candidate_progress(
        path, args.full_pose_multistarts, args.root_height, args
    )

    assert mapping["schema_version"] == 3
    assert mapping["complete"] is False
    assert set(solved) == {slope}
    assert loaded_diagnostics == diagnostics
    assert loaded_candidates == candidate_bank


def test_internal_worker_writes_public_candidate_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging.json"
    public = tmp_path / "public.json"
    args = _build_parser().parse_args(
        ["--candidate-output", os.fspath(staging), "--_pipeline-worker", "token"]
    )
    slope = 0.0
    pose = next(
        row
        for row in _synthetic_reset_pose_mapping()["poses"]
        if row["gradient"] == slope
    )
    diagnostics = [{"gradient": slope, "hand_x": 0.19, "foot_x": -0.035}]
    candidate_bank = {
        slope: [
            {
                "candidate_id": 1,
                "pose": pose,
                "static_metrics": _rank_metrics(),
            }
        ]
    }
    monkeypatch.setenv(solver_module.PIPELINE_WORKER_PROGRESS_ENV, os.fspath(public))

    _write_candidate_progress(args, diagnostics, candidate_bank)

    assert json.loads(staging.read_text(encoding="utf-8")) == json.loads(
        public.read_text(encoding="utf-8")
    )


def test_existing_library_validation_requires_formal_horizon(tmp_path: Path) -> None:
    reset_poses = tmp_path / "reset_poses.yaml"
    reset_poses.write_text("{}\n", encoding="utf-8")
    args = _build_parser().parse_args(
        ["--validate-existing", str(reset_poses), "--steps", "999"]
    )

    with pytest.raises(ValueError, match="requires --steps >= 1000"):
        _validate_arguments(args)


def test_complete_pipeline_rejects_a_nonformal_horizon_before_search() -> None:
    args = _build_parser().parse_args(["--steps", "999"])

    with pytest.raises(ValueError, match="requires --steps in \\[1000, 2000\\]"):
        _validate_arguments(args)


def test_internal_pipeline_worker_rejects_direct_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _build_parser().parse_args(["--_pipeline-worker", "forged-token"])
    monkeypatch.delenv("G1_RICKSHAW_RESET_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("G1_RICKSHAW_RESET_WORKER_STAGING", raising=False)

    with pytest.raises(ValueError, match="requires parent authorization"):
        _validate_arguments(args)


def test_reset_pipeline_rejects_input_output_path_collisions(tmp_path: Path) -> None:
    shared = tmp_path / "shared.json"
    args = _build_parser().parse_args(
        ["--output", str(shared), "--candidate-output", str(shared)]
    )

    with pytest.raises(ValueError, match="paths must be distinct"):
        _validate_arguments(args)


def test_validation_report_is_rebound_only_to_published_path(tmp_path: Path) -> None:
    published = tmp_path / "config" / "reset_poses.yaml"
    report = {
        "status": "passed",
        "inputs": {
            "reset_pose_path": "/temporary/reset_poses.yaml",
        },
    }

    rebound = _retarget_validation_report(report, published)

    assert rebound["inputs"]["reset_pose_path"] == str(published.resolve())
    assert report["inputs"]["reset_pose_path"] == "/temporary/reset_poses.yaml"


def test_validation_path_override_rebinds_the_runtime_pose_library(
    tmp_path: Path,
) -> None:
    mapping = _synthetic_reset_pose_mapping()
    mapping["poses"][0]["root_height"] += 0.001
    default_path = tmp_path / "default_reset_poses.json"
    default_path.write_text(json.dumps(_synthetic_reset_pose_mapping()), encoding="utf-8")
    override_path = tmp_path / "override_reset_poses.json"
    override_path.write_text(json.dumps(mapping), encoding="utf-8")
    cfg = SimpleNamespace(
        reset_pose_path=str(default_path),
        reset_pose_library=solver_module.load_reset_pose_library(default_path),
    )

    rebound = _bind_reset_pose_library(cfg, override_path)

    assert Path(cfg.reset_pose_path) == override_path.resolve()
    assert cfg.reset_pose_library is rebound
    rebound_height = rebound.pose_for_gradient(
        mapping["poses"][0]["gradient"]
    ).root_height
    assert rebound_height == pytest.approx(mapping["poses"][0]["root_height"])


def test_assembled_validation_report_must_bind_the_staged_library(
    tmp_path: Path,
) -> None:
    reset_poses = tmp_path / "reset_poses.yaml"
    reset_poses.write_text("library\n", encoding="utf-8")
    report = {
        "schema_version": 2,
        "tool": "validate_reset_alignment",
        "status": "passed",
        "steps": 1000,
        "inputs": {
            "reset_pose_path": str(reset_poses.resolve()),
        },
    }

    assert _assembled_validation_report_errors(report, reset_poses, 1000) == []
    report["inputs"]["reset_pose_path"] = str(tmp_path / "other.yaml")
    assert "different reset library" in " ".join(
        _assembled_validation_report_errors(report, reset_poses, 1000)
    )


def test_assembled_validation_uses_the_same_pipeline_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        steps=1000,
        seed=42,
        timeseries_stride=100,
        stable_displacement_limit=0.05,
        static_lower_preload_limit=0.85,
        static_waist_preload_limit=0.85,
        static_arm_preload_limit=0.85,
        device="cuda:0",
        foot_stiffness=None,
        foot_damping=None,
        leg_stiffness=None,
        leg_damping=None,
        headless=True,
    )
    command = _assembled_validation_command(
        args,
        tmp_path / "reset_poses.yaml",
        tmp_path / "alignment.json",
        tmp_path / "failed.json",
    )

    assert command[1] == str(SCRIPTS_ROOT / "solve_reset_poses.py")
    assert "--validate-existing" in command
    assert "--alignment-output" in command
    assert "--headless" in command


def test_candidate_validation_process_accepts_a_failed_physics_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_poses = tmp_path / "reset_poses.json"
    reset_poses.write_text("library\n", encoding="utf-8")
    output_dir = tmp_path / "batch"
    output_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        failure_path = Path(command[command.index("--report-output") + 1])
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "tool": "validate_reset_alignment",
                    "status": "failed",
                    "steps": 1000,
                    "inputs": {
                        "reset_pose_path": str(reset_poses.resolve()),
                    },
                    "slopes": [float(value) for value in solver_module.SLOPE_GRADIENTS],
                    "initial": [{} for _ in solver_module.SLOPE_GRADIENTS],
                    "rollout": [{} for _ in solver_module.SLOPE_GRADIENTS],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1)

    args = SimpleNamespace(
        steps=1000,
        seed=42,
        timeseries_stride=0,
        stable_displacement_limit=0.05,
        static_lower_preload_limit=0.85,
        static_waist_preload_limit=0.85,
        static_arm_preload_limit=0.85,
        device="cuda:0",
        foot_stiffness=None,
        foot_damping=None,
        leg_stiffness=None,
        leg_damping=None,
        headless=True,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    report = _run_candidate_validation_process(args, reset_poses, output_dir)

    assert report["status"] == "failed"
    assert len(calls) == 1
    assert calls[0][1] == str(SCRIPTS_ROOT / "solve_reset_poses.py")
    assert "--validate-existing" in calls[0]


def test_candidate_rollout_batches_use_distinct_process_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = solver_module.ResetPoseLibrary.from_mapping(
        _synthetic_reset_pose_mapping()
    )
    candidate_bank: dict[float, list[dict[str, object]]] = {}
    for slope in solver_module.SLOPE_GRADIENTS:
        pose = baseline.pose_for_gradient(slope).to_mapping()
        candidate_bank[float(slope)] = [
            {
                "candidate_id": 0,
                "pose": pose,
                "static_metrics": {"fat2_error": 0.0, "zmp_margin": 1.0},
            }
        ]
    candidate_bank[float(solver_module.SLOPE_GRADIENTS[0])].append(
        {
            "candidate_id": 1,
            "pose": baseline.pose_for_gradient(
                solver_module.SLOPE_GRADIENTS[0]
            ).to_mapping(),
            "static_metrics": {"fat2_error": 0.0, "zmp_margin": 1.0},
        }
    )
    workspaces: list[tuple[Path, Path]] = []

    def fake_validation(
        args: SimpleNamespace, reset_pose_path: Path, output_dir: Path
    ) -> dict[str, object]:
        workspaces.append((reset_pose_path, output_dir))
        rollout = [
            {
                "max_arm_torque_ratio": 0.1,
                "maximum_lower_torque_ratio": 0.1,
                "max_d6_residual_m_or_rad": 0.0,
            }
            for _ in solver_module.SLOPE_GRADIENTS
        ]
        return {
            "status": "failed",
            "safety_thresholds": {},
            "rickshaw_pose_contract": {"hitch_height_target_m": 0.85},
            "initial": [{} for _ in solver_module.SLOPE_GRADIENTS],
            "rollout": rollout,
        }

    def passing_score(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "survival_steps": 1000,
            "dynamic_checks_passed": 1,
            "dynamic_checks_total": 1,
            "all_dynamic_checks_passed": True,
            "worst_normalized_risk": 0.1,
            "mean_normalized_risk": 0.1,
        }

    args = SimpleNamespace(
        output=tmp_path / "reset_poses.yaml",
        candidate_output=tmp_path / "candidates.json",
        report_output=tmp_path / "report.json",
        summary_output=tmp_path / "report.md",
        full_pose_multistarts=50,
        steps=1000,
    )
    monkeypatch.setattr(
        solver_module, "_run_candidate_validation_process", fake_validation
    )
    monkeypatch.setattr(solver_module, "_candidate_score", passing_score)

    assert _run_isolated_candidate_rollouts(args, baseline, candidate_bank) == 0
    assert len(workspaces) == 2
    assert workspaces[0][0] != workspaces[1][0]
    assert workspaces[0][1] != workspaces[1][1]


def test_pipeline_parent_publishes_only_after_isolated_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reset_poses.yaml"
    alignment = tmp_path / "alignment.json"
    report = tmp_path / "search_report.json"
    summary = tmp_path / "search_report.md"
    calls: list[list[str]] = []

    def option_path(command: list[str], option: str) -> Path:
        return Path(command[command.index(option) + 1])

    def fake_run(
        command: list[str], *, check: bool, env: dict[str, str] | None = None
    ) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        if "--_pipeline-worker" in command:
            option_path(command, "--output").write_text("library\n", encoding="utf-8")
            option_path(command, "--candidate-output").write_text(
                "candidates\n", encoding="utf-8"
            )
            option_path(command, "--report-output").write_text(
                json.dumps(
                    {
                        "status": "candidate_passed",
                        "attempted_per_slope": 50,
                        "steps": 1000,
                        "winners": [],
                    }
                ),
                encoding="utf-8",
            )
        else:
            reset_path = option_path(command, "--validate-existing")
            option_path(command, "--alignment-output").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "tool": "validate_reset_alignment",
                        "status": "passed",
                        "failures": [],
                        "steps": 1000,
                        "inputs": {
                            "reset_pose_path": str(reset_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    args = SimpleNamespace(
        output=output,
        alignment_output=alignment,
        report_output=report,
        summary_output=summary,
        candidate_output=tmp_path / "candidates.json",
        steps=1000,
        seed=42,
        timeseries_stride=0,
        stable_displacement_limit=0.05,
        static_lower_preload_limit=0.85,
        static_waist_preload_limit=0.85,
        static_arm_preload_limit=0.85,
        device="cuda:0",
        foot_stiffness=None,
        foot_damping=None,
        leg_stiffness=None,
        leg_damping=None,
        headless=True,
        reuse_candidates=False,
        full_pose_multistarts=50,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["solve_reset_poses.py"])
    replaced_destinations: list[Path] = []
    real_replace = solver_module.os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replaced_destinations.append(Path(destination).resolve())
        real_replace(source, destination)

    monkeypatch.setattr(solver_module.os, "replace", recording_replace)

    assert _run_pipeline_parent(args) == 0
    assert len(calls) == 2
    assert output.read_text(encoding="utf-8") == "library\n"
    alignment_report = json.loads(alignment.read_text(encoding="utf-8"))
    assert alignment_report["inputs"]["reset_pose_path"] == str(output.resolve())
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "passed"
    assert replaced_destinations[-1] == output.resolve()


def test_pipeline_parent_rejects_a_report_bound_to_a_different_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reset_poses.yaml"
    output.write_text("existing-library\n", encoding="utf-8")
    alignment = tmp_path / "alignment.json"
    report = tmp_path / "search_report.json"
    summary = tmp_path / "search_report.md"

    def option_path(command: list[str], option: str) -> Path:
        return Path(command[command.index(option) + 1])

    def fake_run(
        command: list[str], *, check: bool, env: dict[str, str] | None = None
    ) -> SimpleNamespace:
        assert check is False
        if "--_pipeline-worker" in command:
            option_path(command, "--output").write_text(
                "new-library\n", encoding="utf-8"
            )
            option_path(command, "--candidate-output").write_text(
                "candidates\n", encoding="utf-8"
            )
            option_path(command, "--report-output").write_text(
                json.dumps(
                    {
                        "status": "candidate_passed",
                        "attempted_per_slope": 50,
                        "steps": 1000,
                        "winners": [],
                    }
                ),
                encoding="utf-8",
            )
        else:
            option_path(command, "--alignment-output").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "tool": "validate_reset_alignment",
                        "status": "passed",
                        "failures": [],
                        "steps": 1000,
                        "inputs": {
                            "reset_pose_path": str(tmp_path / "different.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    args = SimpleNamespace(
        output=output,
        alignment_output=alignment,
        report_output=report,
        summary_output=summary,
        candidate_output=tmp_path / "candidates.json",
        steps=1000,
        seed=42,
        timeseries_stride=0,
        stable_displacement_limit=0.05,
        static_lower_preload_limit=0.85,
        static_waist_preload_limit=0.85,
        static_arm_preload_limit=0.85,
        device="cuda:0",
        foot_stiffness=None,
        foot_damping=None,
        leg_stiffness=None,
        leg_damping=None,
        headless=True,
        reuse_candidates=False,
        full_pose_multistarts=50,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["solve_reset_poses.py"])

    with pytest.raises(RuntimeError, match="different reset library"):
        _run_pipeline_parent(args)

    assert output.read_text(encoding="utf-8") == "existing-library\n"
    assert not alignment.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "failed"


def test_pipeline_parent_publishes_partial_stage_a_cache_on_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_output = tmp_path / "candidates.json"
    candidate_output.write_text("old cache\n", encoding="utf-8")
    args = SimpleNamespace(
        output=tmp_path / "reset_poses.yaml",
        alignment_output=tmp_path / "alignment.json",
        report_output=tmp_path / "report.json",
        summary_output=tmp_path / "report.md",
        candidate_output=candidate_output,
        steps=1000,
        seed=42,
        timeseries_stride=0,
        stable_displacement_limit=0.05,
        static_lower_preload_limit=0.85,
        static_waist_preload_limit=0.85,
        static_arm_preload_limit=0.85,
        device="cuda:0",
        foot_stiffness=None,
        foot_damping=None,
        leg_stiffness=None,
        leg_damping=None,
        headless=True,
        reuse_candidates=True,
        full_pose_multistarts=50,
    )

    def fake_run(
        command: list[str], *, check: bool, env: dict[str, str] | None = None
    ) -> SimpleNamespace:
        staged_candidate = Path(command[command.index("--candidate-output") + 1])
        staged_candidate.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "slopes": [{"slope": 0.0}],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["solve_reset_poses.py"])

    with pytest.raises(RuntimeError, match="saving 1/19.*--reuse-candidates"):
        _run_pipeline_parent(args)

    assert json.loads(candidate_output.read_text(encoding="utf-8"))["slopes"] == [
        {"slope": 0.0}
    ]
    assert json.loads(args.report_output.read_text(encoding="utf-8"))[
        "stage_a_completed_count"
    ] == 1


def test_publication_rolls_back_evidence_when_library_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reset_poses.yaml"
    alignment = tmp_path / "alignment.json"
    report = tmp_path / "report.json"
    output.write_text("old-library\n", encoding="utf-8")
    alignment.write_text("old-alignment\n", encoding="utf-8")
    report.write_text("old-report\n", encoding="utf-8")
    prepared_library = tmp_path / "prepared-library"
    prepared_alignment = tmp_path / "prepared-alignment"
    prepared_report = tmp_path / "prepared-report"
    prepared_library.write_text("new-library\n", encoding="utf-8")
    prepared_alignment.write_text("new-alignment\n", encoding="utf-8")
    prepared_report.write_text("new-report\n", encoding="utf-8")
    real_replace = solver_module.os.replace

    def fail_library_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).resolve() == output.resolve():
            raise OSError("simulated final commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(solver_module.os, "replace", fail_library_replace)

    with pytest.raises(OSError, match="simulated final commit failure"):
        _commit_pipeline_publications(
            [
                (prepared_report, report),
                (prepared_alignment, alignment),
            ],
            prepared_library,
            output,
        )

    assert output.read_text(encoding="utf-8") == "old-library\n"
    assert alignment.read_text(encoding="utf-8") == "old-alignment\n"
    assert report.read_text(encoding="utf-8") == "old-report\n"
    assert not prepared_library.exists()


def test_atomic_artifacts_are_world_readable_and_cleaned_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"
    prepared = _prepare_atomic_text(destination, "prepared\n")
    assert stat.S_IMODE(prepared.stat().st_mode) == 0o644
    prepared.unlink()

    def fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(solver_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        _write_text_atomic(destination, "value\n")

    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []


def test_multistart_failure_reaps_the_entire_worker_batch() -> None:
    try:
        multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork multiprocessing is unavailable")
    existing = {process.pid for process in multiprocessing.active_children()}

    def worker(index: int) -> int:
        if index == 0:
            raise RuntimeError("expected worker failure")
        time.sleep(5.0)
        return index

    with pytest.raises(RuntimeError, match="expected worker failure"):
        _run_multistarts(worker, count=3, workers=3)

    leaked = [
        process
        for process in multiprocessing.active_children()
        if process.pid not in existing
    ]
    assert leaked == []


def _valid_metrics() -> dict[str, float | int]:
    return {
        "hard_residual": 1.0e-5,
        "hard_tolerance": 1.0e-3,
        "lower_torque_ratio": 0.4,
        "lower_torque_limit": 0.7,
        "waist_torque_ratio": 0.4,
        "waist_torque_limit": 0.7,
        "arm_torque_ratio": 0.6,
        "arm_torque_limit": 0.7,
        "q_ref_joint_margin": 0.01,
        "joint_margin": 0.06,
        "minimum_dex_forward_dot": 0.6,
        "maximum_dex_forward_lateral": 0.005,
        "zmp_margin": 0.03,
        "minimum_zmp_margin": 0.02,
        "friction_ratio": 0.2,
        "nominal_friction": 0.6,
        "root_equilibrium_residual": 1.0e-7,
        "root_equilibrium_tolerance": 1.0e-5,
        "fat2_error": 0.02,
        "fat2_error_tolerance": 0.12,
        "torso_pitch": 0.2,
        "maximum_torso_pitch": 0.45,
        "continuation_joint_delta": 0.1,
        "maximum_continuation_joint_delta": 0.35,
        "continuation_arm_delta": 0.1,
        "maximum_continuation_arm_delta": 0.30,
        "support_torque_ratio": 0.1,
        "self_collision_count": 0,
    }


def test_candidate_rank_accepts_only_a_complete_hard_constraint_pass() -> None:
    assert _candidate_constraint_violation(**_valid_metrics()) == 0.0


def test_q_ref_margin_is_a_real_hard_gate() -> None:
    metrics = _valid_metrics()
    metrics.update(q_ref_joint_margin=0.04, minimum_q_ref_joint_margin=0.04)
    assert _candidate_constraint_violation(**metrics) == 0.0
    metrics["q_ref_joint_margin"] = 0.0399
    assert _candidate_constraint_violation(**metrics) > 0.0


def test_fat2_gravity_moment_mismatch_is_a_hard_gate() -> None:
    metrics = _valid_metrics()
    metrics.update(fat2_moment_error=1.0, fat2_moment_tolerance=1.0)
    assert _candidate_constraint_violation(**metrics) == 0.0
    metrics["fat2_moment_error"] = 1.001
    assert _candidate_constraint_violation(**metrics) > 0.0


def test_static_load_cases_and_torque_ratios_use_the_worst_endpoint() -> None:
    assert _static_load_scales(0.0) == (1.0,)
    assert _static_load_scales(0.1) == pytest.approx((0.9, 1.0, 1.1))
    limits = np.ones(29)
    required = np.vstack((0.5 * limits, -0.8 * limits, 0.7 * limits))
    per_joint, lower, waist, arm = _worst_case_torque_ratios(
        required, limits, np
    )
    assert per_joint == pytest.approx(np.full(29, 0.8))
    assert (lower, waist, arm) == pytest.approx((0.8, 0.8, 0.8))


def test_joint_cart_statics_closes_all_six_equations_with_lateral_com() -> None:
    solution = solve_fixed_contact_statics(
        mass=90.0,
        gradient=0.06,
        com_from_axle_sln=(0.65, 0.12, 0.30),
        handle_from_axle_sn=(1.80, 0.45),
        hitch_half_width=0.24,
        wheel_track=0.756462,
        pitch_torque_on_robot=3.0,
    )

    assert solution.cart_force_residual_sln == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-10)
    assert solution.cart_moment_residual_sln == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-10)
    left_hand, right_hand = solution.handle_wrenches_sln
    expected_difference = (
        0.12 * 90.0 * 9.81 * math.sin(math.atan(0.06)) / 0.24
    )
    assert left_hand[0] - right_hand[0] == pytest.approx(expected_difference)
    left_wheel, right_wheel = solution.wheel_contact_forces_sln
    expected_wheel_difference = (
        2.0 * 0.12 * 90.0 * 9.81 * math.cos(math.atan(0.06)) / 0.756462
    )
    assert left_wheel[2] - right_wheel[2] == pytest.approx(expected_wheel_difference)


def test_support_torque_is_applied_only_to_load_bearing_feet() -> None:
    total = np.asarray((1.0, 2.0, 3.0))
    allocated = _allocate_support_torques(total, np.asarray((1.0, 0.0)), np)
    assert allocated[0] == pytest.approx(total)
    assert allocated[1] == pytest.approx(np.zeros(3))
    assert np.sum(allocated, axis=0) == pytest.approx(total)


def test_per_foot_wrench_gate_checks_cop_friction_and_unilateral_contact() -> None:
    forces = np.asarray(((0.0, 0.0, 100.0), (0.0, 0.0, 100.0)))
    torques = np.zeros((2, 3))
    points = np.asarray(((0.0, 0.1, 0.0), (0.0, -0.1, 0.0)))
    origins = points.copy()
    ratios, components = _per_foot_support_wrench_ratios(
        forces=forces,
        free_torques=torques,
        contact_points=points,
        foot_origins=origins,
        contact_bounds=(-0.05, 0.05, -0.03, 0.03),
        friction=0.5,
        np=np,
    )
    assert ratios == pytest.approx((0.0, 0.0))
    assert components == pytest.approx(np.zeros((2, 4)))

    forces[0, 0] = 50.0
    torques[1, 1] = -5.0
    ratios, components = _per_foot_support_wrench_ratios(
        forces=forces,
        free_torques=torques,
        contact_points=points,
        foot_origins=origins,
        contact_bounds=(-0.05, 0.05, -0.03, 0.03),
        friction=0.5,
        np=np,
    )
    assert components[0, 2] == pytest.approx(1.0)
    assert components[1, 0] == pytest.approx(1.0)
    assert ratios == pytest.approx((1.0, 1.0))

    forces[1] = 0.0
    assert math.isinf(
        _per_foot_support_wrench_ratios(
            forces=forces,
            free_torques=torques,
            contact_points=points,
            foot_origins=origins,
            contact_bounds=(-0.05, 0.05, -0.03, 0.03),
            friction=0.5,
            np=np,
        )[0][1]
    )


def test_auto_worker_count_respects_the_configured_core_budget() -> None:
    assert _solver_worker_count(20, 64) == 20
    assert _solver_worker_count(20, 8) == 8


def _rank_metrics() -> dict[str, float]:
    return {
        "violation": 0.0,
        "root_height": 0.68,
        "fat2_error": 0.01,
        "continuation_arm_delta": 0.08,
        "continuation_joint_delta": 0.10,
        "arm_posture_error": 0.20,
        "arm_torque_ratio": 0.60,
        "waist_torque_ratio": 0.20,
        "lower_torque_ratio": 0.20,
        "zmp_margin": 0.04,
        "hand_x_error": 0.01,
        "hard_residual": 1.0e-6,
        "cost": 1.0,
    }


def test_candidate_rank_does_not_trade_stability_for_root_height() -> None:
    stable_continuation = _rank_metrics()
    target_root_higher_torque = {
        **stable_continuation,
        "root_height": 0.72,
        "fat2_error": 0.04,
        "continuation_arm_delta": 0.28,
        "continuation_joint_delta": 0.30,
        "arm_posture_error": 0.80,
        "arm_torque_ratio": 0.84,
        "cost": 2.0,
    }

    assert _candidate_rank_key(stable_continuation, 0.72) < _candidate_rank_key(
        target_root_higher_torque, 0.72
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("zmp_margin", 0.019),
        ("friction_ratio", 0.61),
        ("root_equilibrium_residual", 1.1e-5),
        ("fat2_error", 0.121),
        ("torso_pitch", 0.451),
        ("continuation_joint_delta", 0.351),
        ("continuation_arm_delta", 0.301),
        ("support_torque_ratio", 1.01),
        ("self_collision_count", 1),
    ),
)
def test_candidate_rank_includes_post_solve_static_gates(
    field: str, value: float | int
) -> None:
    metrics = _valid_metrics()
    metrics[field] = value
    assert _candidate_constraint_violation(**metrics) > 0.0


@pytest.mark.parametrize(
    "field",
    ("lower_torque_ratio", "waist_torque_ratio", "arm_torque_ratio"),
)
def test_hardware_torque_gate_uses_the_relaxed_085_limit(field: str) -> None:
    metrics = _valid_metrics()
    metrics.update(
        lower_torque_limit=DEFAULT_HARDWARE_TORQUE_LIMIT,
        waist_torque_limit=DEFAULT_HARDWARE_TORQUE_LIMIT,
        arm_torque_limit=DEFAULT_HARDWARE_TORQUE_LIMIT,
    )
    metrics[field] = DEFAULT_HARDWARE_TORQUE_LIMIT
    assert _candidate_constraint_violation(**metrics) == 0.0
    metrics[field] += 1.0e-4
    assert _candidate_constraint_violation(**metrics) > 0.0


def test_foot_contact_height_is_derived_from_coplanar_urdf_spheres() -> None:
    centers = np.asarray(
        [
            (-0.05, 0.025, -0.03),
            (-0.05, -0.025, -0.03),
            (0.12, 0.03, -0.03),
            (0.12, -0.03, -0.03),
        ]
        * 2,
        dtype=np.float64,
    )
    model = SimpleNamespace(
        geom_bodyid=np.asarray((1,) * 4 + (2,) * 4),
        geom_contype=np.ones(8, dtype=np.int32),
        geom_conaffinity=np.ones(8, dtype=np.int32),
        geom_type=np.full(8, 2, dtype=np.int32),
        geom_pos=centers,
        geom_size=np.asarray(((0.005, 0.0, 0.0),) * 8),
    )
    mujoco = SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_SPHERE=2))

    assert _foot_contact_geometry(model, (1, 2), mujoco, np) == pytest.approx(
        (0.035, -0.05, 0.12, -0.03, 0.03)
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--waist-stiffness", "0"), "waist-stiffness"),
        (
            ("--arm-torque-limit-fraction", "0.5"),
            "ik-arm-torque-target-fraction",
        ),
    ),
)
def test_solver_rejects_nonphysical_preload_parameters_before_loading_mujoco(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["solve_reset_poses.py", *arguments])
    with pytest.raises(ValueError, match=message):
        main()
