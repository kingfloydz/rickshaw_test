#!/usr/bin/env python3
"""Export evaluation metrics and record the selected policy after training."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from _isaaclab_wrappers import REPOSITORY_ROOT, add_project_source_to_path, require_existing_file

add_project_source_to_path()

from g1_rickshaw_lab.provenance import sha256_file  # noqa: E402
from g1_rickshaw_lab.training_contract import load_policy_ablation_artifact  # noqa: E402
from g1_rickshaw_lab.validation import utc_timestamp, write_json_atomic  # noqa: E402


RESULTS_SCHEMA_VERSION = 1


def _numeric_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, int | float]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(value[key], child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _numeric_leaves(item, child)
    elif not isinstance(value, bool) and isinstance(value, (int, float)):
        yield prefix, value


def _artifact_binding(path: str | Path) -> dict[str, Any]:
    artifact = require_existing_file(path, "result artifact").resolve()
    size_bytes = artifact.stat().st_size
    if size_bytes <= 0:
        raise RuntimeError(f"result artifact is empty: {artifact}")
    return {
        "path": os.fspath(artifact),
        "sha256": sha256_file(artifact),
        "size_bytes": size_bytes,
    }


def _known_artifact_binding(path: str | Path, sha256: str) -> dict[str, Any]:
    artifact = Path(path).resolve()
    return {
        "path": os.fspath(artifact),
        "sha256": sha256,
        "size_bytes": artifact.stat().st_size,
    }


def _selected_binding(manifest: Mapping[str, Any], selected_run_id: str | None) -> dict[str, Any]:
    manifest_selected = manifest.get("selected_run_id")
    selected = manifest_selected if selected_run_id is None else selected_run_id
    if not isinstance(selected, str) or not selected:
        raise ValueError("evaluation manifest has no selected run")
    if selected != manifest_selected:
        raise ValueError("--selected-run-id differs from the evaluated selection")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("evaluation manifest has no run bindings")
    matches = [run for run in runs if isinstance(run, dict) and run.get("id") == selected]
    if len(matches) != 1:
        raise ValueError("selected run is missing or duplicated in the evaluation manifest")
    return matches[0]


def _write_metrics_csv(path: Path, runs: Sequence[Mapping[str, Any]]) -> int:
    rows: list[tuple[str, str, Any, str, int | float]] = []
    for run in runs:
        report_path = Path(run["report"]).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for metric, value in _numeric_leaves(report.get("stages", {}), "stages"):
            rows.append((run["id"], run["group"], run["value"], metric, value))
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run_id", "ablation_group", "ablation_value", "metric", "value"))
        writer.writerows(rows)
    temporary.replace(path)
    return len(rows)


def _existing_results_are_valid(
    path: Path,
    *,
    checkpoint_sha256: str,
    acceptance_sha256: str,
    manifest_sha256: str,
    video_required: bool,
    video_length: int,
    video_num_envs: int,
) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != RESULTS_SCHEMA_VERSION
            or value.get("status") != "passed"
            or value.get("inputs", {}).get("checkpoint", {}).get("sha256")
            != checkpoint_sha256
            or value.get("inputs", {}).get("acceptance_report", {}).get("sha256")
            != acceptance_sha256
            or value.get("inputs", {}).get("evaluation_manifest", {}).get("sha256")
            != manifest_sha256
            or bool(value.get("video", {}).get("recorded")) != video_required
            or (
                video_required
                and (
                    value.get("video", {}).get("length_steps") != video_length
                    or value.get("video", {}).get("num_envs") != video_num_envs
                )
            )
        ):
            return False
        bindings = [value["metrics_csv"], *value.get("exports", {}).values()]
        if video_required:
            bindings.append(value["video"]["artifact"])
        return all(
            Path(binding["path"]).is_file()
            and Path(binding["path"]).stat().st_size == binding["size_bytes"]
            and sha256_file(binding["path"]) == binding["sha256"]
            for binding in bindings
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _record_video(
    *,
    checkpoint: Path,
    acceptance_report: Path,
    evaluation_manifest: Path,
    validation_dir: Path,
    output_dir: Path,
    video_length: int,
    video_num_envs: int,
    device: str,
) -> Path:
    recording_dir = output_dir / "video_recording"
    recording_dir.mkdir(parents=True, exist_ok=True)
    for stale_video in recording_dir.rglob("*.mp4"):
        stale_video.unlink()
    command = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts" / "play_student.py"),
        "--checkpoint",
        os.fspath(checkpoint),
        "--acceptance-report",
        os.fspath(acceptance_report),
        "--ablation-manifest",
        os.fspath(evaluation_manifest),
        "--validation-dir",
        os.fspath(validation_dir),
        "--video-dir",
        os.fspath(recording_dir),
        "--headless",
        "--video",
        "--video_length",
        str(video_length),
        "--num_envs",
        str(video_num_envs),
        "--seed",
        "42",
        "--device",
        device,
    ]
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    recordings = [path for path in recording_dir.rglob("*.mp4") if path.stat().st_size > 0]
    if not recordings:
        raise RuntimeError(f"policy playback produced no non-empty MP4 in {recording_dir}")
    recorded = max(recordings, key=lambda path: path.stat().st_mtime_ns)
    destination = output_dir / "policy.mp4"
    recorded.replace(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-run-id", default=None)
    parser.add_argument("--validation-dir", type=Path, default=None)
    parser.add_argument("--video-length", type=int, default=1000)
    parser.add_argument("--video-num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.skip_video and (args.video_length <= 0 or args.video_num_envs <= 0):
        raise ValueError("video length and environment count must be positive")
    manifest_path = require_existing_file(
        args.evaluation_manifest, "policy evaluation manifest"
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("policy evaluation manifest must be a JSON object")
    selected = _selected_binding(manifest, args.selected_run_id)
    checkpoint = require_existing_file(selected.get("checkpoint", ""), "selected checkpoint").resolve()
    acceptance = Path(selected["report"]).resolve()
    ablation_binding = load_policy_ablation_artifact(
        manifest_path,
        checkpoint_path=checkpoint,
    )

    validation_dir = None
    if not args.skip_video:
        if args.validation_dir is None:
            raise ValueError("video recording requires validation-dir")
        validation_dir = args.validation_dir.resolve()
        if not validation_dir.is_dir():
            raise FileNotFoundError(
                f"selected validation directory does not exist: {validation_dir}"
            )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    input_bindings = {
        "checkpoint": _known_artifact_binding(
            checkpoint,
            selected["checkpoint_sha256"],
        ),
        "acceptance_report": _known_artifact_binding(
            acceptance,
            selected["report_sha256"],
        ),
        "evaluation_manifest": _known_artifact_binding(
            manifest_path,
            ablation_binding["ablation_manifest_sha256"],
        ),
    }
    if args.resume and _existing_results_are_valid(
        result_path,
        checkpoint_sha256=input_bindings["checkpoint"]["sha256"],
        acceptance_sha256=input_bindings["acceptance_report"]["sha256"],
        manifest_sha256=input_bindings["evaluation_manifest"]["sha256"],
        video_required=not args.skip_video,
        video_length=args.video_length,
        video_num_envs=args.video_num_envs,
    ):
        print(f"reused verified training results: {result_path}")
        return 0

    runs = manifest["runs"]
    metrics_path = output_dir / "metrics.csv"
    metric_rows = _write_metrics_csv(metrics_path, runs)
    video_path = None
    if not args.skip_video:
        video_path = _record_video(
            checkpoint=checkpoint,
            acceptance_report=acceptance,
            evaluation_manifest=manifest_path,
            validation_dir=validation_dir,
            output_dir=output_dir,
            video_length=args.video_length,
            video_num_envs=args.video_num_envs,
            device=args.device,
        )

    export_dir = checkpoint.parent / "exported"
    exports = {}
    if not args.skip_video:
        for name in (
            "policy.pt",
            "policy.onnx",
            "deployment_controller.pt",
            "manifest.json",
            "SHA256SUMS",
        ):
            exports[name] = _artifact_binding(export_dir / name)
    result = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "report_type": "g1_rickshaw_training_results",
        "status": "passed",
        "created_utc": utc_timestamp(),
        "selected_run_id": selected["id"],
        "inputs": input_bindings,
        "metrics_csv": {**_artifact_binding(metrics_path), "rows": metric_rows},
        "evaluation_reports": {
            run["id"]: _known_artifact_binding(
                run["report"],
                run["report_sha256"],
            )
            for run in runs
        },
        "video": {
            "recorded": video_path is not None,
            "length_steps": None if video_path is None else args.video_length,
            "num_envs": None if video_path is None else args.video_num_envs,
            "artifact": None if video_path is None else _artifact_binding(video_path),
        },
        "exports": exports,
    }
    write_json_atomic(result_path, result)
    print(f"wrote training results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
