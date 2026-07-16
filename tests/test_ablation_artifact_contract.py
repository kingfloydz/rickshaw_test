"""Regression tests for the content-bound policy-ablation artifact chain."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_policy_ablations import _effective_defaults

from g1_rickshaw_lab import policy_evaluation
from g1_rickshaw_lab import training_contract
from g1_rickshaw_lab.policy_evaluation import (
    ABLATION_DEFAULTS,
    ABLATION_VARIANTS,
    FORMAL_EVALUATION_COMMAND_PROTOCOL,
    FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
)
from g1_rickshaw_lab.provenance import sha256_file


def _matrix_runs() -> list[dict]:
    return [
        {
            "id": f"{group}_{value}",
            "group": group,
            "value": value,
            "checkpoint": f"artifacts/{group}_{value}.pt",
            "teacher_checkpoint": f"artifacts/{group}_{value}.teacher.pt",
            "s1_baseline_report": f"artifacts/{group}_{value}.s1.json",
        }
        for group, values in ABLATION_VARIANTS.items()
        for value in values
    ]


def test_global_teacher_and_s1_defaults_are_forbidden(tmp_path: Path) -> None:
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("schema_version: 1\nthresholds: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown ablation defaults"):
        _effective_defaults(
            {
                "thresholds": str(thresholds),
                "teacher_checkpoint": "shared-teacher.pt",
                "s1_baseline_report": "shared-s1.json",
            },
            matrix_directory=tmp_path,
        )


def test_ablation_preflight_requires_exhaustive_factorial_environment_count(
    tmp_path: Path,
) -> None:
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("schema_version: 1\nthresholds: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="multiple of 19"):
        _effective_defaults(
            {"thresholds": str(thresholds), "num_envs": 13},
            matrix_directory=tmp_path,
        )
    assert _effective_defaults(
        {"thresholds": str(thresholds), "num_envs": 380},
        matrix_directory=tmp_path,
    )["num_envs"] == 380
    with pytest.raises(ValueError, match="divisible"):
        _effective_defaults(
            {"thresholds": str(thresholds), "episodes_per_slope": 101},
            matrix_directory=tmp_path,
        )


def test_loader_reparses_matrix_yaml_instead_of_trusting_manifest_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    defaults = {
        "seeds": [42, 43, 44, 45, 46],
        "episodes_per_slope": 100,
        "thresholds": "thresholds.yaml",
    }
    (tmp_path / "thresholds.yaml").write_text("thresholds\n", encoding="utf-8")
    matrix = {"schema_version": 3, "defaults": defaults, "runs": _matrix_runs()}
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    matrix_digest = sha256_file(matrix_path)

    evidence_by_checkpoint: dict[str, dict] = {}
    bindings: list[dict] = []
    for index, run in enumerate(matrix["runs"]):
        checkpoint = (tmp_path / run["checkpoint"]).resolve()
        teacher = (tmp_path / run["teacher_checkpoint"]).resolve()
        s1_report = (tmp_path / run["s1_baseline_report"]).resolve()
        s1_checkpoint = artifacts / f"s1_{index}.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode("ascii"))
        teacher.write_bytes(f"teacher-{index}".encode("ascii"))
        s1_checkpoint.write_bytes(f"s1-{index}".encode("ascii"))
        s1_report.write_text("{}\n", encoding="utf-8")
        expected_values = {**ABLATION_DEFAULTS, run["group"]: run["value"]}
        evidence = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_stage": "s2_student_ppo",
            "checkpoint_lineage": {
                "teacher_checkpoint_sha256": sha256_file(teacher),
                "context_checkpoint_sha256": sha256_file(s1_checkpoint),
            },
            "checkpoint_provenance": {"runtime": "fixed"},
            "training_configuration": {
                "stage": "s2_student_ppo",
                "formal": True,
                "task": "Isaac-G1-Rickshaw-Directional-Slope-v0",
                "seed": 42,
                "ablation_values": expected_values,
            },
            "training_throughput": {"samples_per_second": 1.0},
            "teacher_checkpoint_sha256": sha256_file(teacher),
            "teacher_checkpoint_provenance": {"runtime": "fixed"},
            "teacher_training_configuration": {"ablation_values": expected_values},
            "s1_baseline_report_sha256": sha256_file(s1_report),
            "s1_checkpoint": str(s1_checkpoint.resolve()),
            "s1_checkpoint_sha256": sha256_file(s1_checkpoint),
            "s1_checkpoint_stage": "s1_context_distillation",
            "s1_checkpoint_provenance": {"runtime": "fixed"},
            "s1_training_configuration": {"ablation_values": expected_values},
        }
        evidence_by_checkpoint[str(checkpoint)] = evidence
        report_path = artifacts / f"report_{index}.json"
        report = {
            "checkpoint": {
                "lineage": evidence["checkpoint_lineage"],
                "provenance": evidence["checkpoint_provenance"],
            },
            "teacher_checkpoint": {"path": str(teacher)},
            "s1_baseline_acceptance": {
                "path": str(s1_report),
                "sha256": evidence["s1_baseline_report_sha256"],
            },
            "ablation": {
                "id": run["id"],
                "group": run["group"],
                "matrix_sha256": matrix_digest,
            },
            "evaluation": {
                "fixed_seeds": defaults["seeds"],
                "episodes_per_slope_per_stage": defaults["episodes_per_slope"],
                "num_envs": 380,
                "curriculum_stages": ["training"],
                "command_protocol": FORMAL_EVALUATION_COMMAND_PROTOCOL,
                "cross_case_protocol": FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
            },
            "inputs": {
                "thresholds_sha256": sha256_file(tmp_path / "thresholds.yaml"),
            },
            "thresholds": {},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        variant_flag = {
            "fat2_weight": "--fat2-weight",
            "rollout_steps": "--rollout-steps",
            "latent_dim": "--latent-dim",
        }[run["group"]]
        bindings.append(
            {
                "id": run["id"],
                "group": run["group"],
                "value": run["value"],
                "checkpoint": str(checkpoint),
                "teacher_checkpoint": str(teacher),
                "s1_baseline_report": str(s1_report),
                **evidence,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "command": [
                    sys.executable,
                    str(SCRIPTS_ROOT / "evaluate_policy.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(report_path.resolve()),
                    "--ablation-id",
                    run["id"],
                    "--ablation-group",
                    run["group"],
                    "--ablation-matrix-sha256",
                    matrix_digest,
                    "--teacher-checkpoint",
                    str(teacher),
                    "--s1-baseline-report",
                    str(s1_report),
                    "--task",
                    "Isaac-G1-Rickshaw-Directional-Slope-v0",
                    "--num-envs",
                    "380",
                    "--episodes-per-slope",
                    "100",
                    "--seeds",
                    "42",
                    "43",
                    "44",
                    "45",
                    "46",
                    "--curriculum-stages",
                    "training",
                    "--max-policy-steps-per-seed",
                    "6000",
                    "--thresholds",
                    str((tmp_path / "thresholds.yaml").resolve()),
                    variant_flag,
                    str(run["value"]),
                ],
                "status": "passed",
            }
        )

    selected = next(run for run in bindings if run["id"] == "fat2_weight_0.1")
    manifest = {
        "schema_version": 3,
        "report_type": "g1_rickshaw_policy_ablation_matrix",
        "status": "passed",
        "created_utc": "2026-07-13T00:00:00Z",
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": matrix_digest,
        "defaults": defaults,
        "selected_run_id": selected["id"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_evidence": {"recomputed": True},
        "runs": bindings,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(training_contract, "load_stage_checkpoint", lambda *args, **kwargs: {})

    def fake_lineage(**kwargs):
        return evidence_by_checkpoint[str(Path(kwargs["checkpoint_path"]).resolve())]

    monkeypatch.setattr(
        training_contract,
        "validate_policy_ablation_run_lineage",
        fake_lineage,
    )
    monkeypatch.setattr(
        policy_evaluation,
        "validate_final_student_acceptance_report",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(policy_evaluation, "load_thresholds", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        policy_evaluation,
        "evaluate_ablation_selection",
        lambda *args, **kwargs: {"recomputed": True},
    )

    result = training_contract.load_policy_ablation_artifact(
        manifest_path,
        checkpoint_path=selected["checkpoint"],
    )
    assert result["ablation_manifest_sha256"] == sha256_file(manifest_path)

    alternate = artifacts / "forged.pt"
    alternate.write_bytes(b"forged")
    manifest["runs"][0]["checkpoint"] = str(alternate.resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from matrix YAML"):
        training_contract.load_policy_ablation_artifact(
            manifest_path,
            checkpoint_path=selected["checkpoint"],
        )
