#!/usr/bin/env python3
"""Run the three policy ablation sweeps from one bound matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from _isaaclab_wrappers import SOURCE_ROOT, require_existing_file

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    ABLATION_DEFAULTS,
    FORMAL_EVALUATION_NUM_ENVS_MULTIPLE,
    GUIDE_POLICY_EVALUATION_TASK,
    POLICY_ABLATION_MANIFEST_SCHEMA_VERSION,
    evaluate_ablation_selection,
    validate_ablation_matrix,
    validate_final_student_acceptance_report,
)
from g1_rickshaw_lab.provenance import sha256_file  # noqa: E402
from g1_rickshaw_lab.slope_contract import FORMAL_EVALUATION_NUM_ENVS  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    validate_policy_ablation_run_lineage,
)
from g1_rickshaw_lab.validation import utc_timestamp, write_json_atomic  # noqa: E402


ALLOWED_DEFAULTS = {
    "task",
    "num_envs",
    "episodes_per_slope",
    "seeds",
    "curriculum_stages",
    "max_policy_steps_per_seed",
    "thresholds",
    "device",
    "headless",
}

EVALUATOR_DEFAULTS = {
    "task": GUIDE_POLICY_EVALUATION_TASK,
    "num_envs": FORMAL_EVALUATION_NUM_ENVS,
    "episodes_per_slope": 100,
    "seeds": [42, 43, 44, 45, 46],
    "curriculum_stages": ["training"],
    "max_policy_steps_per_seed": 6000,
}


def _default_arguments(defaults: dict) -> list[str]:
    result: list[str] = []
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            result.append(flag)
            result.extend(str(item) for item in value)
        else:
            result.extend((flag, str(value)))
    return result


def _variant_arguments(group: str, value) -> list[str]:
    if group == "fat2_weight":
        return ["--fat2-weight", str(value)]
    if group == "rollout_steps":
        return ["--rollout-steps", str(value)]
    if group == "latent_dim":
        return ["--latent-dim", str(value)]
    raise ValueError(f"unknown ablation group {group!r}")


def _effective_defaults(defaults: dict, *, matrix_directory: Path) -> dict:
    unknown = set(defaults) - ALLOWED_DEFAULTS
    if unknown:
        raise ValueError(f"unknown ablation defaults: {sorted(unknown)}")
    result = {**EVALUATOR_DEFAULTS, **defaults}
    if result["task"] != GUIDE_POLICY_EVALUATION_TASK:
        raise ValueError("formal ablation evaluation requires the Guide training task")
    num_envs = result["num_envs"]
    episodes = result["episodes_per_slope"]
    max_steps = result["max_policy_steps_per_seed"]
    seeds = result["seeds"]
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or num_envs % FORMAL_EVALUATION_NUM_ENVS_MULTIPLE != 0
    ):
        raise ValueError(
            "ablation num_envs must be a positive multiple of "
            f"{FORMAL_EVALUATION_NUM_ENVS_MULTIPLE}"
        )
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 100:
        raise ValueError("ablation evaluation requires at least 100 episodes per slope")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("ablation evaluation step limit must be positive")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("ablation evaluation seeds must be a non-empty unique integer list")
    if episodes % (len(seeds) * 4) != 0:
        raise ValueError(
            "ablation episodes_per_slope must be divisible by fixed seeds times four "
            "cross cases"
        )
    if result["curriculum_stages"] != ["training"]:
        raise ValueError("formal ablation evaluation requires exactly TRAINING")
    threshold_value = result.get("thresholds")
    if not isinstance(threshold_value, str) or not threshold_value:
        raise ValueError("formal ablation evaluation requires an explicit thresholds YAML")
    threshold_path = Path(threshold_value)
    if not threshold_path.is_absolute():
        threshold_path = matrix_directory / threshold_path
    result["thresholds"] = os.fspath(
        require_existing_file(threshold_path, "ablation thresholds").resolve()
    )
    if "headless" in result and type(result["headless"]) is not bool:
        raise ValueError("ablation headless default must be boolean")
    if "device" in result and (
        not isinstance(result["device"], str) or not result["device"]
    ):
        raise ValueError("ablation device default must be a non-empty string")
    return result


def _matrix_artifact_path(matrix_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else matrix_path.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--selected-run-id",
        default=None,
        help="Run whose checkpoint is selected from the independent validation sweep.",
    )
    args = parser.parse_args()

    matrix_path = require_existing_file(args.matrix, "ablation matrix").resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    runs = validate_ablation_matrix(matrix)
    if not args.dry_run and not args.selected_run_id:
        raise ValueError("completed ablation requires --selected-run-id")
    if args.selected_run_id is not None and args.selected_run_id not in {
        run["id"] for run in runs
    }:
        raise ValueError("--selected-run-id is not present in the ablation matrix")
    defaults = matrix.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("ablation defaults must be a mapping")
    effective_defaults = _effective_defaults(defaults, matrix_directory=matrix_path.parent)
    common_args = _default_arguments(effective_defaults)
    matrix_digest = sha256_file(matrix_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("evaluate_policy.py")

    manifest_runs = []
    for run in runs:
        checkpoint_path = require_existing_file(
            _matrix_artifact_path(matrix_path, run["checkpoint"]),
            "ablation checkpoint",
        ).resolve()
        teacher_path = require_existing_file(
            _matrix_artifact_path(matrix_path, run["teacher_checkpoint"]),
            "ablation teacher checkpoint",
        ).resolve()
        s1_baseline_path = require_existing_file(
            _matrix_artifact_path(matrix_path, run["s1_baseline_report"]),
            "ablation S1 baseline report",
        ).resolve()
        expected_values = {**ABLATION_DEFAULTS, run["group"]: run["value"]}
        lineage_evidence = validate_policy_ablation_run_lineage(
            checkpoint_path=checkpoint_path,
            teacher_checkpoint_path=teacher_path,
            s1_baseline_report_path=s1_baseline_path,
            expected_ablation_values=expected_values,
            fixed_seeds=effective_defaults["seeds"],
            episodes_per_slope=effective_defaults["episodes_per_slope"],
            validate_runtime=True,
        )
        report_path = output_dir / f"{run['id']}.json"
        command = [
            sys.executable,
            os.fspath(evaluator),
            "--checkpoint",
            os.fspath(checkpoint_path),
            "--output",
            os.fspath(report_path),
            "--ablation-id",
            run["id"],
            "--ablation-group",
            run["group"],
            "--ablation-matrix-sha256",
            matrix_digest,
            "--teacher-checkpoint",
            os.fspath(teacher_path),
            "--s1-baseline-report",
            os.fspath(s1_baseline_path),
            *common_args,
            *_variant_arguments(run["group"], run["value"]),
        ]
        binding = {
            "id": run["id"],
            "group": run["group"],
            "value": run["value"],
            "checkpoint": os.fspath(checkpoint_path),
            "teacher_checkpoint": os.fspath(teacher_path),
            "s1_baseline_report": os.fspath(s1_baseline_path),
            **lineage_evidence,
            "report": os.fspath(report_path),
            "command": command,
        }
        if not args.dry_run:
            subprocess.run(command, check=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_checkpoint = report.get("checkpoint")
            report_teacher = report.get("teacher_checkpoint")
            report_s1 = report.get("s1_baseline_acceptance")
            report_ablation = report.get("ablation")
            report_inputs = report.get("inputs")
            validate_final_student_acceptance_report(
                report,
                expected_checkpoint_sha256=lineage_evidence["checkpoint_sha256"],
                expected_teacher_sha256=lineage_evidence["teacher_checkpoint_sha256"],
                expected_s1_checkpoint_sha256=lineage_evidence["s1_checkpoint_sha256"],
            )
            if (
                report_checkpoint.get("lineage") != lineage_evidence["checkpoint_lineage"]
                or report_checkpoint.get("provenance")
                != lineage_evidence["checkpoint_provenance"]
                or report_teacher.get("path") != os.fspath(teacher_path)
                or report_s1.get("path") != os.fspath(s1_baseline_path)
                or report_s1.get("sha256")
                != lineage_evidence["s1_baseline_report_sha256"]
                or not isinstance(report_ablation, dict)
                or report_ablation.get("id") != run["id"]
                or report_ablation.get("group") != run["group"]
                or report_ablation.get("matrix_sha256") != matrix_digest
                or not isinstance(report_inputs, dict)
                or report_inputs.get("ablation_matrix_sha256") != matrix_digest
            ):
                raise RuntimeError(f"ablation report is incomplete, failed, or misbound: {report_path}")
            binding["report_sha256"] = sha256_file(report_path)
            binding["status"] = "passed"
            binding["report_content"] = report
        manifest_runs.append(binding)

    selection_evidence = None
    if not args.dry_run:
        selection_evidence = evaluate_ablation_selection(
            manifest_runs,
            selected_run_id=args.selected_run_id,
        )
        for binding in manifest_runs:
            binding.pop("report_content", None)

    manifest = {
        "schema_version": POLICY_ABLATION_MANIFEST_SCHEMA_VERSION,
        "report_type": "g1_rickshaw_policy_ablation_matrix",
        "status": "planned" if args.dry_run else "passed",
        "created_utc": utc_timestamp(),
        "matrix": os.fspath(matrix_path),
        "matrix_sha256": matrix_digest,
        "defaults": defaults,
        "selected_run_id": args.selected_run_id,
        "selected_checkpoint_sha256": (
            None
            if args.selected_run_id is None
            else next(
                run["checkpoint_sha256"]
                for run in manifest_runs
                if run["id"] == args.selected_run_id
            )
        ),
        "selection_evidence": selection_evidence,
        "runs": manifest_runs,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "runs": len(manifest_runs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
