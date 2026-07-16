from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import generate_training_artifacts as artifacts

from g1_rickshaw_lab.provenance import sha256_file


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def test_numeric_leaves_and_metrics_csv_flatten_reports(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    first.write_text(
        json.dumps(
            {
                "stages": {
                    "training": {
                        "fall_rate": 0.125,
                        "passed": True,
                        "per_seed": [1, 2],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    second = tmp_path / "second.json"
    second.write_text(
        json.dumps({"stages": {"training": {"return_mean": 7.5}}}),
        encoding="utf-8",
    )
    runs = [
        {
            "id": "fat2_weight_0.0",
            "group": "fat2_weight",
            "value": 0.0,
            "report": str(first),
            "report_sha256": sha256_file(first),
        },
        {
            "id": "fat2_weight_0.1",
            "group": "fat2_weight",
            "value": 0.1,
            "report": str(second),
            "report_sha256": sha256_file(second),
        },
    ]

    output = tmp_path / "results" / "metrics.csv"
    assert artifacts._write_metrics_csv(output, runs) == 4
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["metric"] for row in rows] == [
        "stages.training.fall_rate",
        "stages.training.per_seed[0]",
        "stages.training.per_seed[1]",
        "stages.training.return_mean",
    ]
    assert all(row["metric"] != "stages.training.passed" for row in rows)
    assert rows[-1] == {
        "run_id": "fat2_weight_0.1",
        "ablation_group": "fat2_weight",
        "ablation_value": "0.1",
        "metric": "stages.training.return_mean",
        "value": "7.5",
    }


def test_metrics_csv_consumes_reports_validated_by_manifest_loader(tmp_path: Path) -> None:
    report = tmp_path / "acceptance.json"
    report.write_text('{"stages":{"training":{"fall_rate":0.1}}}\n', encoding="utf-8")
    recorded_hash = sha256_file(report)
    report.write_text('{"stages":{"training":{"fall_rate":0.2}}}\n', encoding="utf-8")

    output = tmp_path / "metrics.csv"
    assert artifacts._write_metrics_csv(
        output,
        [
            {
                "id": "fat2_weight_0.1",
                "group": "fat2_weight",
                "value": 0.1,
                "report": str(report),
                "report_sha256": recorded_hash,
            }
        ],
    ) == 1
    assert "0.2" in output.read_text(encoding="utf-8")


def test_selected_binding_is_exactly_bound_to_manifest_selection() -> None:
    manifest = {
        "selected_run_id": "fat2_weight_0.1",
        "runs": [
            {"id": "fat2_weight_0.0"},
            {"id": "fat2_weight_0.1", "checkpoint": "selected.pt"},
        ],
    }
    assert artifacts._selected_binding(manifest, None)["checkpoint"] == "selected.pt"
    assert (
        artifacts._selected_binding(manifest, "fat2_weight_0.1")["checkpoint"]
        == "selected.pt"
    )
    with pytest.raises(ValueError, match="differs from the evaluated selection"):
        artifacts._selected_binding(manifest, "fat2_weight_0.0")

    manifest["runs"].append({"id": "fat2_weight_0.1"})
    with pytest.raises(ValueError, match="missing or duplicated"):
        artifacts._selected_binding(manifest, None)


def test_resume_requires_unchanged_input_and_output_hashes(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("metric,value\nfall_rate,0.1\n", encoding="utf-8")
    exported = tmp_path / "policy.onnx"
    exported.write_bytes(b"onnx")
    result = {
        "schema_version": artifacts.RESULTS_SCHEMA_VERSION,
        "status": "passed",
        "inputs": {
            "checkpoint": {"sha256": "checkpoint-hash"},
            "acceptance_report": {"sha256": "acceptance-hash"},
            "evaluation_manifest": {"sha256": "manifest-hash"},
        },
        "metrics_csv": _binding(metrics),
        "exports": {"policy.onnx": _binding(exported)},
        "video": {"recorded": False},
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    valid_kwargs = {
        "checkpoint_sha256": "checkpoint-hash",
        "acceptance_sha256": "acceptance-hash",
        "manifest_sha256": "manifest-hash",
        "video_required": False,
        "video_length": 1000,
        "video_num_envs": 1,
    }
    assert artifacts._existing_results_are_valid(result_path, **valid_kwargs)
    exported.write_bytes(b"changed")
    assert not artifacts._existing_results_are_valid(result_path, **valid_kwargs)


def test_record_video_builds_playback_command_without_launching_isaac_sim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    output_dir = tmp_path / "output"
    stale_dir = output_dir / "video_recording"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.mp4").write_bytes(b"old")
    captured: dict[str, object] = {}

    def fake_run(command, *, check, cwd):
        captured.update(command=list(command), check=check, cwd=cwd)
        assert not (stale_dir / "stale.mp4").exists()
        (stale_dir / "fresh.mp4").write_bytes(b"new video")

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    result = artifacts._record_video(
        checkpoint=checkpoint,
        acceptance_report=acceptance,
        evaluation_manifest=manifest,
        validation_dir=validation_dir,
        output_dir=output_dir,
        video_length=321,
        video_num_envs=2,
        device="cpu",
    )

    command = captured["command"]
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--acceptance-report") + 1] == str(acceptance)
    assert command[command.index("--ablation-manifest") + 1] == str(manifest)
    assert command[command.index("--video-dir") + 1] == str(stale_dir)
    assert command[command.index("--video_length") + 1] == "321"
    assert command[command.index("--num_envs") + 1] == "2"
    assert command[command.index("--device") + 1] == "cpu"
    assert "--video" in command
    assert "--headless" in command
    assert captured["check"] is True
    assert captured["cwd"] == artifacts.REPOSITORY_ROOT
    assert result == output_dir / "policy.mp4"
    assert result.read_bytes() == b"new video"


def test_skip_video_needs_no_validation_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "acceptance.json"
    report.write_text('{"stages":{"training":{"return":1.0}}}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "selected_run_id": "fat2_weight_0.1",
                "runs": [
                    {
                        "id": "fat2_weight_0.1",
                        "group": "fat2_weight",
                        "value": 0.1,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "report": str(report),
                        "report_sha256": sha256_file(report),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        artifacts,
        "load_policy_ablation_artifact",
        lambda *args, **kwargs: {
            "ablation_manifest_path": str(manifest),
            "ablation_manifest_sha256": sha256_file(manifest),
        },
    )

    output_dir = tmp_path / "results"
    assert artifacts.main(
        [
            "--evaluation-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--skip-video",
        ]
    ) == 0
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert result["video"] == {
        "recorded": False,
        "length_steps": None,
        "num_envs": None,
        "artifact": None,
    }
