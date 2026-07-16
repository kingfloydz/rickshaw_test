from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_ablation_pipeline as pipeline


def test_run_command_streams_output_and_keeps_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cpu = min(os.sched_getaffinity(0))
    log_path = tmp_path / "stream.log"

    pipeline._run_command(
        [sys.executable, "-c", "print('live-output', flush=True)"],
        environment=os.environ,
        cpu_set=(cpu,),
        log_path=log_path,
        label="stream-test",
    )

    assert "[stream-test] live-output" in capsys.readouterr().out
    assert "live-output" in log_path.read_text(encoding="utf-8")


def test_copy_validation_gate_copies_only_current_training_evidence(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "canonical_validation"
    base_dir.mkdir()
    (base_dir / "asset_inspection.json").write_text("asset\n", encoding="utf-8")
    (base_dir / "feasibility_report.json").write_text("obsolete\n", encoding="utf-8")
    (base_dir / "dynamics_report.json").write_text("obsolete\n", encoding="utf-8")

    destination = pipeline._copy_validation_gate(base_dir, tmp_path / "run")

    assert (destination / "asset_inspection.json").read_text(encoding="utf-8") == "asset\n"
    assert not (destination / "reset_alignment_1000.json").exists()
    assert not (destination / "feasibility_report.json").exists()
    assert not (destination / "dynamics_report.json").exists()


def test_unique_training_runs_expand_to_exact_formal_matrix() -> None:
    assert len(pipeline.UNIQUE_RUNS) == 6
    matrix_runs = pipeline._matrix_run_specs()
    assert len(matrix_runs) == 8
    assert len({identifier for identifier, *_ in matrix_runs}) == 8
    baseline_entries = [entry for entry in matrix_runs if entry[3] == "baseline"]
    assert len(baseline_entries) == 3
    assert ("latent_dim_24", "latent_dim", 24, "latent_dim_24") in matrix_runs


def test_shared_storage_modes_require_unambiguous_run_selection() -> None:
    parser = pipeline._parser()
    worker = parser.parse_args(
        [
            "--final-thresholds",
            "thresholds.yaml",
            "--worker-only",
            "--runs",
            "baseline",
        ]
    )
    assert worker.worker_only is True
    assert worker.finalize_only is False
    assert worker.runs == ["baseline"]

    finalizer = parser.parse_args(
        ["--final-thresholds", "thresholds.yaml", "--finalize-only"]
    )
    assert finalizer.finalize_only is True
    assert finalizer.runs is None

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--final-thresholds",
                "thresholds.yaml",
                "--worker-only",
                "--finalize-only",
            ]
        )


def test_shared_storage_lock_rejects_duplicate_configuration(tmp_path: Path) -> None:
    spec = pipeline.RUNS_BY_NAME["baseline"]

    with pipeline._exclusive_run_lock(tmp_path, spec):
        with pytest.raises(RuntimeError, match="already owned"):
            with pipeline._exclusive_run_lock(tmp_path, spec):
                pass


def test_worker_only_exits_without_scanning_or_assembling_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained: list[str] = []
    monkeypatch.setattr(pipeline, "_validate_inputs", lambda _args: None)
    monkeypatch.setattr(
        pipeline,
        "_select_gpus",
        lambda _requested: [pipeline.GpuInfo(0, "test GPU", 1, 0)],
    )
    monkeypatch.setattr(
        pipeline,
        "_run_one_pipeline",
        lambda spec, **_kwargs: trained.append(spec.name),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_completed_run",
        lambda *_args, **_kwargs: pytest.fail("worker scanned the formal matrix"),
    )

    result = pipeline.main(
        [
            "--final-thresholds",
            "thresholds.yaml",
            "--output-dir",
            os.fspath(tmp_path),
            "--runs",
            "baseline",
            "--worker-only",
            "--gpus",
            "0",
        ]
    )

    assert result == 0
    assert trained == ["baseline"]


def test_finalize_only_fails_when_shared_run_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "_validate_inputs", lambda _args: None)
    monkeypatch.setattr(
        pipeline,
        "_select_gpus",
        lambda _requested: [pipeline.GpuInfo(0, "test GPU", 1, 0)],
    )
    monkeypatch.setattr(pipeline, "_load_completed_run", lambda *_args: None)

    with pytest.raises(RuntimeError, match="cannot finalize; shared output is missing"):
        pipeline.main(
            [
                "--final-thresholds",
                "thresholds.yaml",
                "--output-dir",
                os.fspath(tmp_path),
                "--finalize-only",
                "--gpus",
                "0",
            ]
        )


def test_gpu_selection_uses_requested_devices_without_model_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = [
        pipeline.GpuInfo(0, "GPU A", 81_559, 512),
        pipeline.GpuInfo(1, "GPU B", 81_559, 4096),
        pipeline.GpuInfo(2, "NVIDIA RTX 4090", 24_564, 0),
    ]
    monkeypatch.setattr(pipeline, "_discover_gpus", lambda: inventory)
    assert [gpu.index for gpu in pipeline._select_gpus(None)] == [0, 1, 2]
    assert [gpu.index for gpu in pipeline._select_gpus([2])] == [2]
    with pytest.raises(RuntimeError, match="do not exist"):
        pipeline._select_gpus([7])


def test_teacher_command_binds_all_ablation_values(tmp_path: Path) -> None:
    spec = pipeline.RUNS_BY_NAME["latent_dim_24"]
    commands = pipeline._pipeline_commands(
        spec,
        run_dir=tmp_path / spec.name,
        validation_dir=tmp_path / "validation",
    )
    teacher = commands["teacher"]
    assert teacher[teacher.index("--latent-dim") + 1] == "24"
    assert teacher[teacher.index("--rollout-steps") + 1] == "48"
    assert teacher[teacher.index("--fat2-weight") + 1] == "0.1"


def test_resolve_checkpoint_skips_config_stale_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = pipeline.RUNS_BY_NAME["baseline"]
    stale = tmp_path / "model_999.pt"
    current = tmp_path / "model_100.pt"
    stale.write_bytes(b"stale")
    current.write_bytes(b"current")
    checkpoints = {
        stale: {
            pipeline.CHECKPOINT_STAGE_KEY: "s0_teacher",
            pipeline.TRAINING_CONFIGURATION_KEY: {},
            pipeline.TRAINING_THROUGHPUT_KEY: {"iterations": 1000},
            "runtime_config": "stale",
        },
        current: {
            pipeline.CHECKPOINT_STAGE_KEY: "s0_teacher",
            pipeline.TRAINING_CONFIGURATION_KEY: {},
            pipeline.TRAINING_THROUGHPUT_KEY: {"iterations": 100},
            "runtime_config": "current",
        },
    }
    monkeypatch.setattr(pipeline, "_torch_load", lambda path: checkpoints[path])
    monkeypatch.setattr(
        pipeline,
        "_checkpoint_matches_current_config",
        lambda checkpoint: checkpoint["runtime_config"] == "current",
    )
    monkeypatch.setattr(
        pipeline,
        "validate_guide_training_configuration",
        lambda _configuration, *, expected_stage: {
            "ablation_values": spec.ablation_values
        },
    )
    monkeypatch.setattr(
        pipeline,
        "validate_training_throughput",
        lambda throughput: throughput,
    )

    resolved = pipeline._resolve_checkpoint(
        tmp_path,
        stage="s0_teacher",
        expected_values=spec.ablation_values,
    )

    assert resolved == current.resolve()


def test_s0_adapter_rejects_missing_evaluator_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "report.json"
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    with pytest.raises(RuntimeError, match="without writing"):
        pipeline._evaluate_s0(
            [
                "--checkpoint",
                os.fspath(checkpoint),
                "--output",
                os.fspath(output),
                "--stage",
                "training",
            ]
        )
