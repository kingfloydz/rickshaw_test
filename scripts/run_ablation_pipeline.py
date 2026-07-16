#!/usr/bin/env python3
"""Train the complete policy-ablation matrix on the selected GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import sys
import threading
from typing import Any, Iterable, Mapping

from _isaaclab_wrappers import REPOSITORY_ROOT, add_project_source_to_path

add_project_source_to_path()

from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    ABLATION_DEFAULTS,
    ABLATION_VARIANTS,
    GUIDE_POLICY_EVALUATION_TASK,
    POLICY_ABLATION_MATRIX_SCHEMA_VERSION,
    load_thresholds,
    validate_ablation_matrix,
    validate_final_acceptance_thresholds,
    validate_s1_baseline_acceptance_report,
)
from g1_rickshaw_lab.provenance import (  # noqa: E402
    extract_checkpoint_metadata,
    hash_config_files,
    sha256_file,
)
from g1_rickshaw_lab.slope_contract import FORMAL_EVALUATION_NUM_ENVS  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_STAGE_KEY,
    GUIDE_MAX_ITERATIONS,
    TRAINING_CONFIGURATION_KEY,
    TRAINING_THROUGHPUT_KEY,
    load_policy_ablation_artifact,
    load_reward_calibration_report,
    runtime_config_files,
    training_checkpoint_complete,
    validate_guide_training_configuration,
    validate_training_throughput,
)
from g1_rickshaw_lab.validation import (  # noqa: E402
    write_json_atomic,
    write_yaml_atomic,
)

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "ablation_pipeline"
DEFAULT_FEASIBILITY = REPOSITORY_ROOT / "config" / "feasibility_envelope.yaml"
DEFAULT_RESET_POSES = REPOSITORY_ROOT / "config" / "reset_poses.yaml"
FIXED_SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    fat2_weight: float
    rollout_steps: int
    latent_dim: int

    @property
    def ablation_values(self) -> dict[str, Any]:
        return {
            "fat2_weight": self.fat2_weight,
            "rollout_steps": self.rollout_steps,
            "latent_dim": self.latent_dim,
        }


@dataclass(frozen=True, slots=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int


UNIQUE_RUNS = (
    RunSpec("baseline", 0.1, 48, 16),
    RunSpec("fat2_weight_0.0", 0.0, 48, 16),
    RunSpec("fat2_weight_0.2", 0.2, 48, 16),
    RunSpec("rollout_steps_24", 0.1, 24, 16),
    RunSpec("rollout_steps_64", 0.1, 64, 16),
    RunSpec("latent_dim_8", 0.1, 48, 8),
    RunSpec("latent_dim_24", 0.1, 48, 24),
    RunSpec("latent_dim_32", 0.1, 48, 32),
)
RUNS_BY_NAME = {run.name: run for run in UNIQUE_RUNS}


@contextmanager
def _exclusive_run_lock(output_dir: Path, spec: RunSpec):
    """Prevent two shared-storage workers from training the same configuration."""

    lock_dir = output_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{spec.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.seek(0)
            owner = lock.read().strip() or "unknown worker"
            raise RuntimeError(
                f"configuration {spec.name!r} is already owned by {owner}"
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(
            json.dumps(
                {"hostname": socket.gethostname(), "pid": os.getpid()},
                sort_keys=True,
            )
        )
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _matrix_run_specs() -> list[tuple[str, str, Any, str]]:
    result: list[tuple[str, str, Any, str]] = []
    non_default_runs = {
        ("fat2_weight", 0.0): "fat2_weight_0.0",
        ("fat2_weight", 0.2): "fat2_weight_0.2",
        ("rollout_steps", 24): "rollout_steps_24",
        ("rollout_steps", 64): "rollout_steps_64",
        ("latent_dim", 8): "latent_dim_8",
        ("latent_dim", 24): "latent_dim_24",
        ("latent_dim", 32): "latent_dim_32",
    }
    for group, values in ABLATION_VARIANTS.items():
        for value in values:
            run_name = (
                "baseline"
                if value == ABLATION_DEFAULTS[group]
                else non_default_runs[(group, value)]
            )
            result.append((f"{group}_{value}", group, value, run_name))
    return result


def _parse_gpu_rows(output: str) -> list[GpuInfo]:
    result: list[GpuInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        result.append(
            GpuInfo(
                index=int(fields[0]),
                name=fields[1],
                memory_total_mib=int(fields[2]),
                memory_used_mib=int(fields[3]),
            )
        )
    return result


def _discover_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi failed; no usable GPU inventory is available") from exc
    return _parse_gpu_rows(result.stdout)


def _select_gpus(
    requested: Iterable[int] | None,
) -> list[GpuInfo]:
    inventory = _discover_gpus()
    requested_ids = None if requested is None else list(requested)
    selected = (
        inventory
        if requested_ids is None
        else [gpu for gpu in inventory if gpu.index in requested_ids]
    )
    if requested_ids is not None and {gpu.index for gpu in selected} != set(requested_ids):
        raise RuntimeError("one or more requested GPU indices do not exist")
    if not selected:
        available = ", ".join(
            f"GPU {gpu.index}: {gpu.name} ({gpu.memory_total_mib} MiB)"
            for gpu in inventory
        )
        raise RuntimeError(f"no requested GPU is available; detected: {available or 'none'}")
    return selected


def _thread_environment(gpu_index: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "WANDB_MODE": "offline",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _run_command(
    command: list[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    bound_command = command
    print(f"[{label}] {shlex.join(bound_command)}", flush=True)
    run_environment = dict(environment)
    run_environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + shlex.join(bound_command) + "\n")
        log.flush()
        process = subprocess.Popen(
            bound_command,
            cwd=REPOSITORY_ROOT,
            env=run_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            process.kill()
            raise RuntimeError(f"{label} did not expose a readable output stream")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{label}] {line}", end="", flush=True)
        returncode = process.wait()
    if returncode != 0:
        recent = _tail(log_path)
        raise RuntimeError(
            f"{label} failed with exit code {returncode}; log={log_path}\n{recent}"
        )


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint is not a mapping")
    return value


def _checkpoint_score(value: Mapping[str, Any], path: Path) -> tuple[int, int]:
    throughput = value.get(TRAINING_THROUGHPUT_KEY)
    iterations = -1
    if isinstance(throughput, Mapping):
        raw_iterations = throughput.get("iterations")
        if isinstance(raw_iterations, int) and not isinstance(raw_iterations, bool):
            iterations = raw_iterations
    return iterations, path.stat().st_mtime_ns


_RUNTIME_CONFIG_SHA256: dict[str, str] | None = None


def _runtime_config_sha256() -> dict[str, str]:
    global _RUNTIME_CONFIG_SHA256
    if _RUNTIME_CONFIG_SHA256 is None:
        _RUNTIME_CONFIG_SHA256 = dict(hash_config_files(runtime_config_files()))
    return dict(_RUNTIME_CONFIG_SHA256)


def _checkpoint_matches_current_config(value: Mapping[str, Any]) -> bool:
    metadata = extract_checkpoint_metadata(value)
    return dict(metadata.config_sha256) == _runtime_config_sha256()


def _matching_checkpoint(
    path: Path,
    *,
    stage: str,
    expected_values: Mapping[str, Any],
) -> tuple[tuple[int, int], Path] | None:
    try:
        checkpoint = _torch_load(path)
        if not _checkpoint_matches_current_config(checkpoint):
            return None
        if checkpoint.get(CHECKPOINT_STAGE_KEY) != stage:
            return None
        configuration = validate_guide_training_configuration(
            checkpoint.get(TRAINING_CONFIGURATION_KEY), expected_stage=stage
        )
        if configuration["ablation_values"] != dict(expected_values):
            return None
        if stage in {"s0_teacher", "s2_student_ppo"}:
            validate_training_throughput(checkpoint.get(TRAINING_THROUGHPUT_KEY))
        return _checkpoint_score(checkpoint, path), path.resolve()
    except (OSError, RuntimeError, ValueError, KeyError):
        return None


def _resolve_checkpoint(
    directory: Path,
    *,
    stage: str,
    expected_values: Mapping[str, Any],
    minimum_iterations: int = 0,
    require_complete: bool = False,
) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(
        directory.rglob("*.pt"), key=lambda path: path.stat().st_mtime_ns, reverse=True
    )
    matches: list[tuple[tuple[int, int], Path]] = []
    for path in candidates:
        match = _matching_checkpoint(
            path,
            stage=stage,
            expected_values=expected_values,
        )
        if match is not None:
            complete = not require_complete or stage not in {
                "s0_teacher",
                "s2_student_ppo",
            }
            if not complete:
                complete = training_checkpoint_complete(_torch_load(path), stage=stage)
            if match[0][0] >= minimum_iterations and complete:
                matches.append(match)
            target_iterations = GUIDE_MAX_ITERATIONS[stage]
            if match[0][0] >= max(minimum_iterations, target_iterations):
                break
    return max(matches, default=None, key=lambda item: item[0])[1] if matches else None


def _resolve_reward_report(directory: Path, teacher: Path) -> Path | None:
    for path in sorted(
        directory.glob("reward_calibration.*.json"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    ):
        try:
            load_reward_calibration_report(path, teacher_checkpoint_path=teacher)
        except (OSError, RuntimeError, ValueError):
            continue
        return path.resolve()
    return None


def _valid_s1_checkpoint(path: Path, spec: RunSpec) -> bool:
    match = _matching_checkpoint(
        path,
        stage="s1_context_distillation",
        expected_values=spec.ablation_values,
    )
    return match is not None


def _valid_s1_report(path: Path, s1_checkpoint: Path) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_s1_baseline_acceptance_report(
            report,
            expected_checkpoint_sha256=sha256_file(s1_checkpoint),
            fixed_seeds=FIXED_SEEDS,
            episodes_per_slope=100,
        )
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def _write_state(run_dir: Path, spec: RunSpec, artifacts: Mapping[str, Path]) -> None:
    payload = {
        "schema_version": 1,
        "run": spec.name,
        "ablation_values": spec.ablation_values,
        "artifacts": {
            name: {"path": os.fspath(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    write_json_atomic(_state_path(run_dir), payload)


def _pipeline_commands(
    spec: RunSpec,
    *,
    run_dir: Path,
) -> dict[str, list[str]]:
    teacher_dir = run_dir / "teacher"
    validation_command = [
        sys.executable,
        os.fspath(Path(__file__).resolve()),
        "_evaluate-s0",
        "--checkpoint",
        "{checkpoint}",
        "--output",
        "{output}",
        "--stage",
        "{stage}",
        "--device",
        "cuda:0",
    ]
    teacher = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts/train_teacher.py"),
        "--experiment-dir",
        os.fspath(teacher_dir),
        "--fat2-weight",
        str(spec.fat2_weight),
        "--rollout-steps",
        str(spec.rollout_steps),
        "--latent-dim",
        str(spec.latent_dim),
        "--device",
        "cuda:0",
        "--headless",
    ]
    return {
        "teacher": teacher,
        "teacher_validation": validation_command,
    }


def _run_one_pipeline(
    spec: RunSpec,
    *,
    gpu: GpuInfo,
    args: argparse.Namespace,
) -> None:
    run_dir = args.output_dir / "runs" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = _thread_environment(gpu.index)
    environment.update(
        {
            "G1_RICKSHAW_ABLATION_RUN": spec.name,
            "G1_RICKSHAW_PHYSICAL_GPU": str(gpu.index),
        }
    )
    commands = _pipeline_commands(
        spec,
        run_dir=run_dir,
    )
    environment["G1_RICKSHAW_S0_VALIDATION_COMMAND"] = shlex.join(
        commands["teacher_validation"]
    )
    label = f"{spec.name}/gpu{gpu.index}"
    logs = run_dir / "logs"

    teacher_dir = run_dir / "teacher"
    teacher = _resolve_checkpoint(
        teacher_dir,
        stage="s0_teacher",
        expected_values=spec.ablation_values,
        require_complete=True,
    )
    if teacher is None:
        partial = _resolve_checkpoint(
            teacher_dir,
            stage="s0_teacher",
            expected_values=spec.ablation_values,
        )
        command = list(commands["teacher"])
        if partial is not None:
            experiment_index = command.index("--experiment-dir")
            del command[experiment_index : experiment_index + 2]
            command.extend(("--resume-checkpoint", os.fspath(partial)))
        command.extend(("--run_name", f"{spec.name}-s0"))
        _run_command(
            command,
            environment=environment,
            log_path=logs / "01_train_teacher.log",
            label=label,
        )
        teacher = _resolve_checkpoint(
            teacher_dir,
            stage="s0_teacher",
            expected_values=spec.ablation_values,
            require_complete=True,
        )
    if teacher is None:
        raise RuntimeError(f"{label} produced no complete S0 checkpoint")

    reward_dir = run_dir / "reward_calibration"
    reward_dir.mkdir(parents=True, exist_ok=True)
    reward_report = _resolve_reward_report(reward_dir, teacher)
    if reward_report is None:
        _run_command(
            [
                sys.executable,
                os.fspath(REPOSITORY_ROOT / "scripts/calibrate_rewards.py"),
                "--checkpoint",
                os.fspath(teacher),
                "--policy-kind",
                "teacher",
                "--output-dir",
                os.fspath(reward_dir),
                "--device",
                "cuda:0",
                "--headless",
            ],
            environment=environment,
            log_path=logs / "02_calibrate_rewards.log",
            label=label,
        )
        reward_report = _resolve_reward_report(reward_dir, teacher)
    if reward_report is None:
        raise RuntimeError(f"{label} produced no passed reward calibration report")

    s1_checkpoint = run_dir / "s1_context.pt"
    if not _valid_s1_checkpoint(s1_checkpoint, spec):
        command = [
            sys.executable,
            os.fspath(REPOSITORY_ROOT / "scripts/train_context.py"),
            "--teacher",
            os.fspath(teacher),
            "--reward-calibration-report",
            os.fspath(reward_report),
            "--latent-dim",
            str(spec.latent_dim),
            "--output",
            os.fspath(s1_checkpoint),
            "--device",
            "cuda:0",
        ]
        _run_command(
            command,
            environment=environment,
            log_path=logs / "03_train_context.log",
            label=label,
        )
    if not _valid_s1_checkpoint(s1_checkpoint, spec):
        raise RuntimeError(f"{label} produced no valid S1 checkpoint")

    s1_report = run_dir / "s1_acceptance.json"
    if not _valid_s1_report(s1_report, s1_checkpoint):
        _run_command(
            [
                sys.executable,
                os.fspath(REPOSITORY_ROOT / "scripts/evaluate_policy.py"),
                "--checkpoint",
                os.fspath(s1_checkpoint),
                "--teacher-checkpoint",
                os.fspath(teacher),
                "--output",
                os.fspath(s1_report),
                "--num-envs",
                str(FORMAL_EVALUATION_NUM_ENVS),
                "--episodes-per-slope",
                "100",
                "--seeds",
                *(str(seed) for seed in FIXED_SEEDS),
                "--curriculum-stages",
                "training",
                "--device",
                "cuda:0",
                "--headless",
            ],
            environment=environment,
            log_path=logs / "04_evaluate_s1.log",
            label=label,
        )
    if not _valid_s1_report(s1_report, s1_checkpoint):
        raise RuntimeError(f"{label} produced no valid S1 baseline report")

    s2_dir = run_dir / "s2"
    s2_checkpoint = _resolve_checkpoint(
        s2_dir,
        stage="s2_student_ppo",
        expected_values=spec.ablation_values,
        require_complete=True,
    )
    if s2_checkpoint is None:
        partial = _resolve_checkpoint(
            s2_dir,
            stage="s2_student_ppo",
            expected_values=spec.ablation_values,
        )
        command = [
            sys.executable,
            os.fspath(REPOSITORY_ROOT / "scripts/finetune_student.py"),
            "--teacher",
            os.fspath(teacher),
            "--context",
            os.fspath(s1_checkpoint),
            "--bootstrap-dir",
            os.fspath(s2_dir),
            "--device",
            "cuda:0",
            "--headless",
        ]
        if partial is not None:
            command.extend(("--resume-checkpoint", os.fspath(partial)))
        command.extend(("--run_name", f"{spec.name}-s2"))
        _run_command(
            command,
            environment=environment,
            log_path=logs / "05_finetune_student.log",
            label=label,
        )
        s2_checkpoint = _resolve_checkpoint(
            s2_dir,
            stage="s2_student_ppo",
            expected_values=spec.ablation_values,
            require_complete=True,
        )
    if s2_checkpoint is None:
        raise RuntimeError(f"{label} produced no complete S2 checkpoint")

    artifacts = {
        "teacher_checkpoint": teacher,
        "reward_calibration_report": reward_report,
        "s1_checkpoint": s1_checkpoint.resolve(),
        "s1_baseline_report": s1_report.resolve(),
        "s2_checkpoint": s2_checkpoint,
    }
    _write_state(run_dir, spec, artifacts)
    print(f"[{label}] complete", flush=True)


def _load_completed_run(output_dir: Path, spec: RunSpec) -> dict[str, Path] | None:
    state_path = _state_path(output_dir / "runs" / spec.name)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("ablation_values") != spec.ablation_values:
            return None
        artifacts = {
            name: Path(binding["path"]).resolve()
            for name, binding in state["artifacts"].items()
        }
        required = {
            "teacher_checkpoint",
            "reward_calibration_report",
            "s1_checkpoint",
            "s1_baseline_report",
            "s2_checkpoint",
        }
        if set(artifacts) != required:
            return None
        for name, path in artifacts.items():
            if not path.is_file() or sha256_file(path) != state["artifacts"][name]["sha256"]:
                return None
        teacher_match = _matching_checkpoint(
            artifacts["teacher_checkpoint"],
            stage="s0_teacher",
            expected_values=spec.ablation_values,
        )
        s2_match = _matching_checkpoint(
            artifacts["s2_checkpoint"],
            stage="s2_student_ppo",
            expected_values=spec.ablation_values,
        )
        load_reward_calibration_report(
            artifacts["reward_calibration_report"],
            teacher_checkpoint_path=artifacts["teacher_checkpoint"],
        )
        if (
            teacher_match is None
            or s2_match is None
            or not training_checkpoint_complete(
                _torch_load(artifacts["teacher_checkpoint"]),
                stage="s0_teacher",
            )
            or not training_checkpoint_complete(
                _torch_load(artifacts["s2_checkpoint"]),
                stage="s2_student_ppo",
            )
            or not _valid_s1_checkpoint(artifacts["s1_checkpoint"], spec)
            or not _valid_s1_report(
                artifacts["s1_baseline_report"], artifacts["s1_checkpoint"]
            )
        ):
            return None
        return artifacts
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _write_matrix(
    output_dir: Path,
    final_thresholds: Path,
    completed: Mapping[str, Mapping[str, Path]],
) -> Path:
    runs = []
    for identifier, group, value, run_name in _matrix_run_specs():
        artifacts = completed[run_name]
        runs.append(
            {
                "id": identifier,
                "group": group,
                "value": value,
                "checkpoint": os.path.relpath(artifacts["s2_checkpoint"], output_dir),
                "teacher_checkpoint": os.path.relpath(
                    artifacts["teacher_checkpoint"], output_dir
                ),
                "s1_baseline_report": os.path.relpath(
                    artifacts["s1_baseline_report"], output_dir
                ),
            }
        )
    matrix = {
        "schema_version": POLICY_ABLATION_MATRIX_SCHEMA_VERSION,
        "defaults": {
            "task": GUIDE_POLICY_EVALUATION_TASK,
            "num_envs": FORMAL_EVALUATION_NUM_ENVS,
            "episodes_per_slope": 100,
            "seeds": list(FIXED_SEEDS),
            "curriculum_stages": ["training"],
            "max_policy_steps_per_seed": 6000,
            "thresholds": os.fspath(final_thresholds.resolve()),
            "device": "cuda:0",
            "headless": True,
        },
        "runs": runs,
    }
    validate_ablation_matrix(matrix)
    return write_yaml_atomic(output_dir / "ablation_matrix.yaml", matrix)


def _evaluate_s0(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Internal S0 fixed-validation adapter.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("training",), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    command = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts/evaluate_policy.py"),
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--num-envs",
        str(FORMAL_EVALUATION_NUM_ENVS),
        "--episodes-per-slope",
        "100",
        "--seeds",
        *(str(seed) for seed in FIXED_SEEDS),
        "--curriculum-stages",
        args.stage,
        "--device",
        args.device,
        "--headless",
    ]
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    if not Path(args.output).is_file():
        raise RuntimeError(
            "S0 evaluator exited successfully without writing its acceptance report: "
            f"{args.output}"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feasibility", type=Path, default=DEFAULT_FEASIBILITY)
    parser.add_argument("--reset-poses", type=Path, default=DEFAULT_RESET_POSES)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=tuple(RUNS_BY_NAME),
        default=None,
        help="Unique configurations to train; all eight are required for a formal matrix.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--worker-only",
        action="store_true",
        help=(
            "Train only the explicitly selected --runs and exit. Use this on "
            "independent compute nodes sharing --output-dir."
        ),
    )
    mode.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Train nothing; verify all eight shared runs, assemble the matrix, "
            "and perform the requested postprocessing."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--selected-run-id",
        default="fat2_weight_0.1",
        help="Formal matrix run to export and record after all evaluations pass.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Stop after writing the matrix instead of generating evaluation/data/video artifacts.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Generate evaluation JSON/CSV results without recording policy playback.",
    )
    parser.add_argument("--video-length", type=int, default=1000)
    parser.add_argument("--video-num-envs", type=int, default=1)
    return parser


def _validate_inputs(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    args.feasibility = args.feasibility.resolve()
    args.reset_poses = args.reset_poses.resolve()
    args.final_thresholds = args.final_thresholds.resolve()
    for label, path in (
        ("final thresholds", args.final_thresholds),
        ("feasibility envelope", args.feasibility),
        ("reset poses", args.reset_poses),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    final = load_thresholds(args.final_thresholds)
    validate_final_acceptance_thresholds(final, curriculum_stages=("training",))
    if args.worker_only and args.runs is None:
        raise ValueError("--worker-only requires an explicit --runs selection")
    if args.finalize_only and args.runs is not None:
        raise ValueError("--finalize-only cannot be combined with --runs")
    postprocess_enabled = not args.skip_postprocess and not args.worker_only
    matrix_ids = {identifier for identifier, *_ in _matrix_run_specs()}
    if postprocess_enabled and args.selected_run_id not in matrix_ids:
        raise ValueError("--selected-run-id is not a formal matrix run ID")
    if (
        postprocess_enabled
        and not args.skip_video
        and (args.video_length <= 0 or args.video_num_envs <= 0)
    ):
        raise ValueError("video length and environment count must be positive")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_evaluate-s0":
        return _evaluate_s0(arguments[1:])
    args = _parser().parse_args(arguments)
    _validate_inputs(args)
    gpus = _select_gpus(args.gpus)
    requested_names = tuple(RUNS_BY_NAME) if args.runs is None else args.runs
    requested_runs = (
        []
        if args.finalize_only
        else [RUNS_BY_NAME[name] for name in dict.fromkeys(requested_names)]
    )
    mode = "worker" if args.worker_only else "finalize" if args.finalize_only else "all"
    postprocess_enabled = not args.skip_postprocess and not args.worker_only

    print(
        json.dumps(
            {
                "mode": mode,
                "runs": [run.name for run in requested_runs],
                "workers": [
                    {
                        "gpu": gpu.index,
                        "name": gpu.name,
                        "memory_mib": gpu.memory_total_mib,
                    }
                    for gpu in gpus
                ],
                "output_dir": os.fspath(args.output_dir),
                "postprocess": {
                    "enabled": postprocess_enabled,
                    "selected_run_id": args.selected_run_id,
                    "record_video": not args.skip_video,
                    "video_length": args.video_length,
                    "video_num_envs": args.video_num_envs,
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if args.plan_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (
        not args.resume
        and not args.worker_only
        and not args.finalize_only
        and any(args.output_dir.iterdir())
    ):
        raise RuntimeError("output directory is not empty; use --resume to reuse verified stages")

    work_queue: queue.Queue[RunSpec] = queue.Queue()
    for spec in requested_runs:
        work_queue.put(spec)
    failures_lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    stop_event = threading.Event()

    def worker(gpu: GpuInfo) -> None:
        while not stop_event.is_set():
            try:
                spec = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                with _exclusive_run_lock(args.output_dir, spec):
                    _run_one_pipeline(
                        spec,
                        gpu=gpu,
                        args=args,
                    )
            except Exception as exc:  # noqa: BLE001
                with failures_lock:
                    failures.append((spec.name, str(exc)))
                stop_event.set()
                return
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(worker, gpu)
            for gpu in gpus
        ]
        for future in futures:
            future.result()
    if failures:
        details = "\n".join(f"- {name}: {failure}" for name, failure in failures)
        raise RuntimeError(f"one or more ablation pipelines failed:\n{details}")
    if args.worker_only:
        print("worker-only training complete; matrix assembly was not attempted", flush=True)
        return 0

    completed = {
        spec.name: artifacts
        for spec in UNIQUE_RUNS
        if (artifacts := _load_completed_run(args.output_dir, spec)) is not None
    }
    missing = [spec.name for spec in UNIQUE_RUNS if spec.name not in completed]
    if missing:
        if args.finalize_only:
            raise RuntimeError(
                "cannot finalize; shared output is missing verified runs: "
                + ", ".join(missing)
            )
        print(
            "partial training complete; formal matrix awaits: " + ", ".join(missing),
            flush=True,
        )
        return 0

    matrix_path = _write_matrix(args.output_dir, args.final_thresholds, completed)
    print(f"wrote formal ablation matrix: {matrix_path}", flush=True)
    if postprocess_enabled:
        results_dir = args.output_dir / "results"
        evaluation_dir = results_dir / "evaluation"
        evaluation_manifest = evaluation_dir / "manifest.json"
        selected_entry = next(
            entry for entry in _matrix_run_specs() if entry[0] == args.selected_run_id
        )
        selected_run_name = selected_entry[3]
        selected_checkpoint = completed[selected_run_name]["s2_checkpoint"]
        gpu = gpus[0]
        environment = _thread_environment(gpu.index)
        evaluation_valid = False
        if args.resume and evaluation_manifest.is_file():
            try:
                load_policy_ablation_artifact(
                    evaluation_manifest,
                    checkpoint_path=selected_checkpoint,
                )
                evaluation_valid = True
                print(f"reused verified policy evaluations: {evaluation_manifest}", flush=True)
            except (OSError, RuntimeError, TypeError, ValueError):
                evaluation_valid = False
        if not evaluation_valid:
            _run_command(
                [
                    sys.executable,
                    os.fspath(REPOSITORY_ROOT / "scripts/run_policy_ablations.py"),
                    "--matrix",
                    os.fspath(matrix_path),
                    "--output-dir",
                    os.fspath(evaluation_dir),
                    "--selected-run-id",
                    args.selected_run_id,
                ],
                environment=environment,
                log_path=results_dir / "evaluation.log",
                label=f"evaluation/gpu{gpu.index}",
            )
        artifact_command = [
            sys.executable,
            os.fspath(REPOSITORY_ROOT / "scripts/generate_training_artifacts.py"),
            "--evaluation-manifest",
            os.fspath(evaluation_manifest),
            "--output-dir",
            os.fspath(results_dir),
            "--selected-run-id",
            args.selected_run_id,
            "--video-length",
            str(args.video_length),
            "--video-num-envs",
            str(args.video_num_envs),
            "--device",
            "cuda:0",
        ]
        if args.skip_video:
            artifact_command.append("--skip-video")
        if args.resume:
            artifact_command.append("--resume")
        _run_command(
            artifact_command,
            environment=environment,
            log_path=results_dir / "artifacts.log",
            label=f"artifacts/gpu{gpu.index}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
