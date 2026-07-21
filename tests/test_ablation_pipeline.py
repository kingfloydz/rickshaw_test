from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_ablation_pipeline as pipeline
from finetune_student import (
    _validate_resume_lineage,
    _validate_resume_training_configuration,
)


RUN_NAMES = [
    "baseline",
    "fat2_weight_0.0",
    "fat2_weight_0.2",
    "rollout_steps_24",
    "rollout_steps_64",
    "latent_dim_8",
    "latent_dim_24",
    "latent_dim_32",
]
STABILITY_RUN_NAMES = [
    f"latent_dim_{latent_dim}_stability_curriculum"
    for latent_dim in (6, 8, 10, 12, 14, 16, 18, 20)
]
TCN_HISTORY_RUN_NAMES = [
    f"tcn_history_{history_length}_latent_dim_{latent_dim}"
    for history_length in (61, 91)
    for latent_dim in (8, 12, 16, 24)
]


def _exact_arguments(output_dir: Path, *, plan_only: bool = False) -> list[str]:
    arguments = [
        "--output-dir",
        os.fspath(output_dir),
        "--runs",
        *RUN_NAMES,
        "--gpus",
        *(str(index) for index in range(8)),
        "--resume",
    ]
    if plan_only:
        arguments.append("--plan-only")
    return arguments


def test_unique_runs_are_the_requested_controlled_matrix() -> None:
    assert [spec.name for spec in pipeline.UNIQUE_RUNS] == RUN_NAMES
    assert [spec.training_parameters for spec in pipeline.UNIQUE_RUNS] == [
        {
            "fat2_weight": 0.0,
                "rollout_steps": 48,
                "latent_dim": 16,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.0,
                "rollout_steps": 48,
                "latent_dim": 16,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.2,
                "rollout_steps": 48,
                "latent_dim": 16,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.0,
                "rollout_steps": 24,
                "latent_dim": 16,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.0,
                "rollout_steps": 64,
                "latent_dim": 16,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.0,
                "rollout_steps": 48,
                "latent_dim": 8,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.0,
                "rollout_steps": 48,
                "latent_dim": 24,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
        {
            "fat2_weight": 0.0,
                "rollout_steps": 48,
                "latent_dim": 32,
                "history_length": 61,
                "stability_reward_curriculum": False,
        },
    ]


def test_stability_curriculum_matrix_covers_all_latent_dimensions() -> None:
    specs = [pipeline.RUNS_BY_NAME[name] for name in STABILITY_RUN_NAMES]

    assert [spec.latent_dim for spec in specs] == [6, 8, 10, 12, 14, 16, 18, 20]
    assert all(spec.fat2_weight == 0.0 for spec in specs)
    assert all(spec.stability_reward_curriculum for spec in specs)


def test_latent_dim_4_is_registered() -> None:
    spec = pipeline.RUNS_BY_NAME["latent_dim_4"]

    assert spec.latent_dim == 4
    assert spec.fat2_weight == 0.0
    assert spec.rollout_steps == 48
    assert spec.stability_reward_curriculum is False


def test_tcn_history_matrix_covers_requested_lengths_and_latents() -> None:
    specs = [pipeline.RUNS_BY_NAME[name] for name in TCN_HISTORY_RUN_NAMES]

    assert [(spec.history_length, spec.latent_dim) for spec in specs] == [
        (history_length, latent_dim)
        for history_length in (61, 91)
        for latent_dim in (8, 12, 16, 24)
    ]
    assert all(spec.fat2_weight == 0.0 for spec in specs)
    assert all(spec.rollout_steps == 48 for spec in specs)
    assert all(not spec.stability_reward_curriculum for spec in specs)


def test_stability_curriculum_teacher_command_enables_switch(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(task="task", num_envs=4096, seed=42)
    command = pipeline._teacher_command(
        pipeline.RUNS_BY_NAME[STABILITY_RUN_NAMES[0]],
        args,
        tmp_path / STABILITY_RUN_NAMES[0],
        None,
    )

    assert "--stability-reward-curriculum" in command


def test_exact_eight_gpu_plan_has_no_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "pipeline"

    assert pipeline.main(_exact_arguments(output_dir, plan_only=True)) == 0

    plan = json.loads(capsys.readouterr().out)
    assert [(item["run"], item["gpu"]) for item in plan["workers"]] == list(
        zip(RUN_NAMES, range(8), strict=True)
    )
    assert plan["tensorboard_logdir"] == str((output_dir / "runs").resolve())
    assert plan["num_envs"] == 8192
    assert plan["reward_weight_overrides"] == {}
    assert not output_dir.exists()


def test_exact_cli_plan_runs_in_a_clean_subprocess(tmp_path: Path) -> None:
    command = [
        sys.executable,
        os.fspath(SCRIPTS_ROOT / "run_ablation_pipeline.py"),
        *_exact_arguments(tmp_path / "pipeline", plan_only=True),
    ]

    result = subprocess.run(
        command,
        cwd=SCRIPTS_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["workers"]) == 8


@pytest.mark.parametrize(
    ("run_name", "fat2_weight", "rollout_steps", "latent_dim"),
    (
        ("fat2_weight_0.0", "0.0", "48", "16"),
        ("fat2_weight_0.2", "0.2", "48", "16"),
        ("rollout_steps_24", "0.0", "24", "16"),
        ("rollout_steps_64", "0.0", "64", "16"),
        ("latent_dim_8", "0.0", "48", "8"),
        ("latent_dim_32", "0.0", "48", "32"),
    ),
)
def test_teacher_command_binds_every_controlled_parameter(
    tmp_path: Path,
    run_name: str,
    fat2_weight: str,
    rollout_steps: str,
    latent_dim: str,
) -> None:
    args = argparse.Namespace(task="task", num_envs=4096, seed=42)
    command = pipeline._teacher_command(
        pipeline.RUNS_BY_NAME[run_name], args, tmp_path / run_name, None
    )

    assert command[command.index("--fat2-weight") + 1] == fat2_weight
    assert command[command.index("--rollout-steps") + 1] == rollout_steps
    assert command[command.index("--latent-dim") + 1] == latent_dim
    assert command[command.index("--device") + 1] == "cuda:0"
    assert any(value.startswith("hydra.run.dir=") for value in command)


def test_tcn_history_command_binds_history_length(tmp_path: Path) -> None:
    args = argparse.Namespace(task="task", num_envs=8192, seed=42)
    command = pipeline._teacher_command(
        pipeline.RUNS_BY_NAME["tcn_history_91_latent_dim_24"],
        args,
        tmp_path / "tcn_history_91_latent_dim_24",
        None,
    )

    assert command[command.index("--history-length") + 1] == "91"


def test_ablation_freezes_reward_profile_in_teacher_command(tmp_path: Path) -> None:
    args = argparse.Namespace(
        task="task",
        num_envs=4096,
        seed=42,
        reward_weight_overrides={
            "zmp_margin_barrier": -3.0,
            "feet_slide": -0.2,
        },
    )

    command = pipeline._teacher_command(
        pipeline.RUNS_BY_NAME["baseline"], args, tmp_path / "baseline", None
    )

    values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--reward-weight"
    ]
    assert values == ["zmp_margin_barrier=-3.0", "feet_slide=-0.2"]


def test_worker_environment_isolates_the_selected_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    environment = pipeline._gpu_environment(4)

    assert environment["CUDA_VISIBLE_DEVICES"] == "4"
    assert pipeline.LOGICAL_CUDA_DEVICE == "cuda:0"


def test_ablation_worker_combines_physical_isolation_with_logical_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = pipeline.CheckpointRecord(tmp_path / "teacher.pt", 6000, True)
    student = pipeline.CheckpointRecord(tmp_path / "student.pt", 6000, True)
    s0_calls = 0

    def checkpoint(_root, _spec, *, stage, **_identity):
        nonlocal s0_calls
        if stage == "s0_teacher":
            s0_calls += 1
            return None if s0_calls == 1 else teacher
        return student

    launches: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(pipeline, "_ppo_checkpoint", checkpoint)
    monkeypatch.setattr(pipeline, "_valid_diagnostic", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        pipeline, "_valid_s1_checkpoint", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda command, *, environment, **_kwargs: launches.append(
            (list(command), dict(environment))
        ),
    )
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        task="task",
        seed=42,
        num_envs=4096,
        reward_weight_overrides={},
        evaluation_num_envs=380,
        episodes_per_slope=100,
        evaluation_seeds=(42, 43, 44, 45, 46),
    )

    pipeline._run_one_pipeline(
        pipeline.RUNS_BY_NAME["baseline"],
        gpu=pipeline.GpuInfo(4, "H200", 140_000),
        args=args,
        registry=pipeline.ProcessRegistry(),
        stop_event=threading.Event(),
    )

    assert len(launches) == 1
    command, environment = launches[0]
    assert environment["CUDA_VISIBLE_DEVICES"] == "4"
    assert command[command.index("--device") + 1] == "cuda:0"


def test_s0_diagnostic_failure_does_not_block_s1_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    teacher = pipeline.CheckpointRecord(tmp_path / "teacher.pt", 6000, True)
    student = pipeline.CheckpointRecord(tmp_path / "student.pt", 2000, True)
    s1_validation_calls = 0
    launches: list[list[str]] = []

    def checkpoint(_root, _spec, *, stage, **_identity):
        return teacher if stage == "s0_teacher" else student

    def valid_s1(*_args, **_kwargs):
        nonlocal s1_validation_calls
        s1_validation_calls += 1
        return s1_validation_calls > 1

    def valid_diagnostic(path, *_args, **_kwargs):
        return path.name != "s0.json"

    def run_command(command, **_kwargs):
        bound = list(command)
        launches.append(bound)
        if "evaluate_policy.py" in bound[1] and "s0.json" in " ".join(bound):
            raise RuntimeError("S0 diagnostic did not pass")

    monkeypatch.setattr(pipeline, "_ppo_checkpoint", checkpoint)
    monkeypatch.setattr(pipeline, "_valid_s1_checkpoint", valid_s1)
    monkeypatch.setattr(
        pipeline, "_rollout_manifest_matches", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(pipeline, "_valid_diagnostic", valid_diagnostic)
    monkeypatch.setattr(pipeline, "_run_command", run_command)
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        task="task",
        seed=42,
        num_envs=4096,
        reward_weight_overrides={},
        evaluation_num_envs=380,
        episodes_per_slope=100,
        evaluation_seeds=(42, 43, 44, 45, 46),
    )

    result = pipeline._run_one_pipeline(
        pipeline.RUNS_BY_NAME["baseline"],
        gpu=pipeline.GpuInfo(0, "H200", 140_000),
        args=args,
        registry=pipeline.ProcessRegistry(),
        stop_event=threading.Event(),
    )

    assert any(command[1].endswith("train_context.py") for command in launches)
    assert result["diagnostics"]["s0_recorded"] is False
    assert "continuing with the next training stage" in capsys.readouterr().out


def test_requested_gpu_inventory_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "_discover_gpus",
        lambda: [pipeline.GpuInfo(0, "H200", 140_000)],
    )

    assert pipeline._select_gpus([0])[0].name == "H200"
    with pytest.raises(RuntimeError, match="do not exist"):
        pipeline._select_gpus([0, 1])


def test_eight_workers_use_the_planned_gpu_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpus = [pipeline.GpuInfo(index, "H200", 140_000) for index in range(8)]
    monkeypatch.setattr(pipeline, "_select_gpus", lambda _indices: gpus)
    monkeypatch.setattr(pipeline, "_validate_runtime_inputs", lambda _args: None)
    barrier = threading.Barrier(8)
    calls: list[tuple[str, int]] = []
    lock = threading.Lock()

    def run_one(spec, *, gpu, **_kwargs):
        barrier.wait(timeout=5)
        with lock:
            calls.append((spec.name, gpu.index))
        return {"run": spec.name, "gpu": gpu.index}

    monkeypatch.setattr(pipeline, "_run_one_pipeline", run_one)

    output_dir = tmp_path / "pipeline"
    assert pipeline.main(_exact_arguments(output_dir)) == 0
    assert sorted(calls) == sorted(zip(RUN_NAMES, range(8), strict=True))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert [result["run"] for result in summary["runs"]] == RUN_NAMES


def test_one_run_failure_does_not_cancel_an_independent_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpus = [pipeline.GpuInfo(index, "H200", 140_000) for index in range(2)]
    monkeypatch.setattr(pipeline, "_select_gpus", lambda _indices: gpus)
    monkeypatch.setattr(pipeline, "_validate_runtime_inputs", lambda _args: None)
    completed: list[str] = []

    def run_one(spec, **_kwargs):
        if spec.name == "baseline":
            raise RuntimeError("expected failure")
        completed.append(spec.name)
        return {"run": spec.name}

    monkeypatch.setattr(pipeline, "_run_one_pipeline", run_one)
    arguments = [
        "--output-dir",
        os.fspath(tmp_path / "pipeline"),
        "--runs",
        "baseline",
        "fat2_weight_0.0",
        "--gpus",
        "0",
        "1",
        "--resume",
    ]

    with pytest.raises(RuntimeError, match="baseline: expected failure"):
        pipeline.main(arguments)

    assert completed == ["fat2_weight_0.0"]


def test_non_resume_rejects_a_nonempty_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pipeline"
    output.mkdir()
    (output / "existing").write_text("data", encoding="utf-8")
    monkeypatch.setenv("ISAACLAB_PATH", os.fspath(tmp_path))
    args = pipeline._parser().parse_args(
        ["--output-dir", os.fspath(output), "--runs", "baseline", "--gpus", "0"]
    )
    args.output_dir = output

    with pytest.raises(RuntimeError, match="not empty"):
        pipeline._validate_runtime_inputs(args)


def test_evaluation_environment_count_must_cover_all_slopes_evenly() -> None:
    args = pipeline._parser().parse_args(
        ["--runs", "baseline", "--gpus", "0", "--evaluation-num-envs", "20"]
    )

    with pytest.raises(ValueError, match="multiple of 19"):
        pipeline._validate_args(args)


def test_ppo_checkpoint_selects_complete_matching_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / "model_10.pt"
    complete = tmp_path / "model_2599.pt"
    wrong = tmp_path / "model_wrong.pt"
    for path in (partial, complete, wrong):
        path.write_bytes(b"checkpoint")
    spec = pipeline.RUNS_BY_NAME["baseline"]

    def load(path, **_kwargs):
        parameters = (
            pipeline.RUNS_BY_NAME["latent_dim_8"].training_parameters
            if Path(path) == wrong
            else spec.training_parameters
        )
        iteration = 10 if Path(path) == partial else 2599
        return {
            "iter": iteration,
            pipeline.TRAINING_CONFIGURATION_KEY: {
                "training_parameters": parameters,
                "max_iterations": 2600,
                "task": "task",
                "seed": 42,
                "num_envs": 4096,
            },
        }

    monkeypatch.setattr(pipeline, "load_stage_checkpoint", load)

    record = pipeline._ppo_checkpoint(
        tmp_path,
        spec,
        stage="s0_teacher",
        task="task",
        seed=42,
        num_envs=4096,
    )

    assert record is not None
    assert record.path == complete.resolve()
    assert record.complete is True
    assert (
        pipeline._ppo_checkpoint(
            tmp_path,
            spec,
            stage="s0_teacher",
            task="task",
            seed=43,
            num_envs=4096,
        )
        is None
    )


def test_s2_checkpoint_must_match_teacher_and_context_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model_1599.pt"
    checkpoint.write_bytes(b"checkpoint")
    teacher = tmp_path / "teacher.pt"
    context = tmp_path / "context.pt"
    other = tmp_path / "other.pt"
    spec = pipeline.RUNS_BY_NAME["baseline"]

    monkeypatch.setattr(
        pipeline,
        "load_stage_checkpoint",
        lambda *_args, **_kwargs: {
            "iter": 1599,
            pipeline.TRAINING_CONFIGURATION_KEY: {
                "training_parameters": spec.training_parameters,
                "max_iterations": 1600,
                "task": "task",
                "seed": 42,
                "num_envs": 4096,
            },
            pipeline.CHECKPOINT_LINEAGE_KEY: {
                "teacher_checkpoint": str(teacher.resolve()),
                "context_checkpoint": str(context.resolve()),
            },
        },
    )

    assert (
        pipeline._ppo_checkpoint(
            tmp_path,
            spec,
            stage="s2_student_ppo",
            task="task",
            seed=42,
            num_envs=4096,
            teacher=teacher,
            context=context,
        )
        is not None
    )
    assert (
        pipeline._ppo_checkpoint(
            tmp_path,
            spec,
            stage="s2_student_ppo",
            task="task",
            seed=42,
            num_envs=4096,
            teacher=other,
            context=context,
        )
        is None
    )


def test_s2_launcher_rejects_a_different_resume_lineage(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.pt"
    context = tmp_path / "context.pt"
    checkpoint = {
        pipeline.CHECKPOINT_LINEAGE_KEY: {
            "teacher_checkpoint": str(teacher.resolve()),
            "context_checkpoint": str(context.resolve()),
        }
    }

    _validate_resume_lineage(checkpoint, teacher, context)
    with pytest.raises(ValueError, match="different S0/S1 lineage"):
        _validate_resume_lineage(checkpoint, tmp_path / "other.pt", context)


def test_s2_launcher_rejects_a_different_resume_reward_profile() -> None:
    parameters = {"fat2_weight": 0.1, "rollout_steps": 48, "latent_dim": 16}
    source = {
        "training_parameters": parameters,
        "resolved_parameters": {
            "reward_weight_overrides": {"zmp_margin_barrier": -3.0}
        },
    }
    matching = json.loads(json.dumps(source))
    different = json.loads(json.dumps(source))
    different["resolved_parameters"]["reward_weight_overrides"] = {
        "zmp_margin_barrier": -2.0
    }

    _validate_resume_training_configuration(matching, source)
    with pytest.raises(ValueError, match="cannot change reward weights"):
        _validate_resume_training_configuration(different, source)


def test_rollout_resume_requires_a_complete_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    spec = pipeline.RUNS_BY_NAME["baseline"]
    configuration = {
        "training_parameters": spec.training_parameters,
        "task": "task",
        "seed": 42,
        "num_envs": 4096,
    }
    monkeypatch.setattr(
        pipeline,
        "load_stage_checkpoint",
        lambda *_args, **_kwargs: {pipeline.TRAINING_CONFIGURATION_KEY: configuration},
    )
    monkeypatch.setattr(
        pipeline,
        "validate_rollout_stage_coverage",
        lambda _manifest: {
            "TRAINING": pipeline.ROLLOUT_DEFAULT_NUM_ENVS
            * pipeline.DISTILLATION_ROLLOUT_STEPS
        },
    )

    assert not pipeline._rollout_manifest_matches(
        rollout_dir,
        teacher,
        spec,
        task="task",
        seed=42,
        num_envs=4096,
    )

    shard = rollout_dir / "rollout_00000.pt"
    shard.write_bytes(b"shard")
    manifest = {
        "schema_version": pipeline.ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "teacher_checkpoint": str(teacher.resolve()),
        "teacher_training_configuration": configuration,
        "num_envs": pipeline.ROLLOUT_DEFAULT_NUM_ENVS,
        "num_steps_per_stage": pipeline.DISTILLATION_ROLLOUT_STEPS,
        "shards": [shard.name],
    }
    (rollout_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert pipeline._rollout_manifest_matches(
        rollout_dir,
        teacher,
        spec,
        task="task",
        seed=42,
        num_envs=4096,
    )
    extra = rollout_dir / "rollout_extra.pt"
    extra.write_bytes(b"extra")
    assert not pipeline._rollout_manifest_matches(
        rollout_dir,
        teacher,
        spec,
        task="task",
        seed=42,
        num_envs=4096,
    )
    extra.unlink()
    shard.unlink()
    assert not pipeline._rollout_manifest_matches(
        rollout_dir,
        teacher,
        spec,
        task="task",
        seed=42,
        num_envs=4096,
    )


def test_s1_resume_matches_the_training_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "s1.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    teacher = tmp_path / "teacher.pt"
    spec = pipeline.RUNS_BY_NAME["baseline"]
    monkeypatch.setattr(
        pipeline,
        "load_stage_checkpoint",
        lambda *_args, **_kwargs: {
            pipeline.TRAINING_CONFIGURATION_KEY: {
                "training_parameters": spec.training_parameters,
                "max_iterations": 2000,
                "task": "task",
                "seed": 42,
                "num_envs": None,
                "resolved_parameters": {"deterministic_algorithms": False},
            },
            pipeline.CHECKPOINT_LINEAGE_KEY: {
                "teacher_checkpoint": str(teacher.resolve())
            },
            "training": {"completed_iterations": 2000},
        },
    )

    assert pipeline._valid_s1_checkpoint(
        checkpoint_path, spec, teacher, task="task", seed=42
    )
    assert not pipeline._valid_s1_checkpoint(
        checkpoint_path, spec, teacher, task="task", seed=43
    )


def test_diagnostic_resume_matches_protocol_and_newest_dependency(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    teacher = tmp_path / "teacher.pt"
    baseline = tmp_path / "s1.json"
    report_path = tmp_path / "s2.json"
    for path in (checkpoint, teacher, baseline):
        path.write_bytes(b"artifact")
    report = {
        "status": "recorded",
        "task": "task",
        "checkpoint": {"path": str(checkpoint.resolve())},
        "teacher_checkpoint": {"path": str(teacher.resolve())},
        "s1_baseline": {"path": str(baseline.resolve())},
        "evaluation": {
            "deterministic_actions": True,
            "num_envs": 380,
            "episodes_per_slope_per_stage": 100,
            "fixed_seeds": [42, 43, 44, 45, 46],
            "curriculum_stages": ["training"],
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    newest = report_path.stat().st_mtime_ns + 1_000_000
    os.utime(report_path, ns=(newest, newest))

    arguments = {
        "task": "task",
        "evaluation_num_envs": 380,
        "episodes_per_slope": 100,
        "evaluation_seeds": (42, 43, 44, 45, 46),
        "teacher": teacher,
        "s1_baseline": baseline,
    }
    assert pipeline._valid_diagnostic(report_path, checkpoint, **arguments)
    assert not pipeline._valid_diagnostic(
        report_path, checkpoint, **{**arguments, "episodes_per_slope": 200}
    )
    baseline_mtime = newest + 1_000_000
    os.utime(baseline, ns=(baseline_mtime, baseline_mtime))
    assert not pipeline._valid_diagnostic(report_path, checkpoint, **arguments)


def test_run_command_streams_output_and_reports_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "command.log"
    pipeline._run_command(
        [sys.executable, "-c", "print('summary-output', flush=True)"],
        environment=os.environ,
        log_path=log,
        label="test",
    )
    assert "[test] summary-output" in capsys.readouterr().out
    assert "summary-output" in log.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="exit code 3"):
        pipeline._run_command(
            [sys.executable, "-c", "raise SystemExit(3)"],
            environment=os.environ,
            log_path=log,
            label="test",
        )


def test_process_registry_rejects_launches_after_shutdown() -> None:
    registry = pipeline.ProcessRegistry()

    registry.terminate_all()

    assert registry.add(object()) is False


def test_tensorboard_event_paths_are_discovered_recursively(tmp_path: Path) -> None:
    first = tmp_path / "s0" / "run" / "events.out.tfevents.1"
    second = tmp_path / "s2" / "run" / "events.out.tfevents.2"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")

    assert pipeline._tensorboard_files(tmp_path) == [
        str(first.resolve()),
        str(second.resolve()),
    ]
