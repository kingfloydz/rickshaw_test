#!/usr/bin/env python3
"""Run the controlled S0 -> S1 -> S2 training matrix on selected GPUs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from _isaaclab_wrappers import REPOSITORY_ROOT, add_project_source_to_path

add_project_source_to_path()

from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_COUNT,
)
from g1_rickshaw_lab.reward_profile import (  # noqa: E402
    parse_reward_weight_arguments,
    reward_weight_overrides_from_configuration,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_LINEAGE_KEY,
    DISTILLATION_ROLLOUT_STEPS,
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_TASK,
    ROLLOUT_DEFAULT_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    S1_DETERMINISTIC_ALGORITHMS,
    SUPPORTED_FAT2_WEIGHTS,
    TRAINING_CONFIGURATION_KEY,
    guide_max_iterations,
    load_stage_checkpoint,
    validate_rollout_stage_coverage,
)
from g1_rickshaw_lab.validation import write_json_atomic  # noqa: E402


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "ablation_pipeline"
EVALUATION_SEEDS = (42, 43, 44, 45, 46)
LOGICAL_CUDA_DEVICE = "cuda:0"


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    fat2_weight: float
    rollout_steps: int
    latent_dim: int
    stability_reward_curriculum: bool = False
    history_length: int = 61

    @property
    def training_parameters(self) -> dict[str, int | float | bool]:
        return {
            "fat2_weight": self.fat2_weight,
            "rollout_steps": self.rollout_steps,
            "latent_dim": self.latent_dim,
            "history_length": self.history_length,
            "stability_reward_curriculum": self.stability_reward_curriculum,
        }


UNIQUE_RUNS = (
    RunSpec("baseline", 0.0, 48, 16),
    RunSpec("fat2_weight_0.0", 0.0, 48, 16),
    RunSpec("fat2_weight_0.2", 0.2, 48, 16),
    RunSpec("rollout_steps_24", 0.0, 24, 16),
    RunSpec("rollout_steps_64", 0.0, 64, 16),
    RunSpec("latent_dim_8", 0.0, 48, 8),
    RunSpec("latent_dim_24", 0.0, 48, 24),
    RunSpec("latent_dim_32", 0.0, 48, 32),
)
LATENT_DIM_RUNS = tuple(
    RunSpec(f"latent_dim_{latent_dim}", 0.0, 48, latent_dim)
    for latent_dim in (4, 6, 10, 12, 14, 16, 18, 20)
)
STABILITY_CURRICULUM_RUNS = tuple(
    RunSpec(
        f"latent_dim_{latent_dim}_stability_curriculum",
        0.0,
        48,
        latent_dim,
        True,
    )
    for latent_dim in (6, 8, 10, 12, 14, 16, 18, 20)
)
TCN_HISTORY_RUNS = tuple(
    RunSpec(
        f"tcn_history_{history_length}_latent_dim_{latent_dim}",
        0.0,
        48,
        latent_dim,
        False,
        history_length,
    )
    for history_length in (61, 91)
    for latent_dim in (8, 12, 16, 24)
)
RUNS_BY_NAME = {
    spec.name: spec
    for spec in (
        *UNIQUE_RUNS,
        *LATENT_DIM_RUNS,
        *STABILITY_CURRICULUM_RUNS,
        *TCN_HISTORY_RUNS,
    )
}


@dataclass(frozen=True, slots=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    path: Path
    completed_updates: int
    complete: bool


class PipelineCancelled(RuntimeError):
    pass


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[str]] = set()
        self._stopping = False

    def add(self, process: subprocess.Popen[str]) -> bool:
        with self._lock:
            if self._stopping:
                return False
            self._processes.add(process)
            return True

    def remove(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self) -> None:
        with self._lock:
            self._stopping = True
            processes = tuple(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
            process.poll() is None for process in processes
        ):
            time.sleep(0.05)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=tuple(RUNS_BY_NAME),
        default=None,
        help="Training configurations; defaults to the original eight-run matrix.",
    )
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the static run/GPU plan without touching GPUs or output files.",
    )
    parser.add_argument("--task", default=GUIDE_TRAINING_TASK)
    parser.add_argument("--num-envs", type=int, default=GUIDE_TRAINING_NUM_ENVS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fat2-weight",
        type=float,
        choices=SUPPORTED_FAT2_WEIGHTS,
        default=None,
        help="Override FAT2 weight for every selected run.",
    )
    parser.add_argument(
        "--reward-weight",
        action="append",
        default=[],
        metavar="TERM=WEIGHT",
        help="Frozen non-FAT2 reward weight; repeat once per term.",
    )
    parser.add_argument(
        "--evaluation-num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS
    )
    parser.add_argument("--episodes-per-slope", type=int, default=100)
    parser.add_argument(
        "--evaluation-seeds", nargs="+", type=int, default=EVALUATION_SEEDS
    )
    return parser


def _validate_args(args: argparse.Namespace) -> list[RunSpec]:
    args.reward_weight_overrides = parse_reward_weight_arguments(args.reward_weight)
    run_names = (
        [spec.name for spec in UNIQUE_RUNS] if args.runs is None else list(args.runs)
    )
    if len(run_names) != len(set(run_names)):
        raise ValueError("--runs must not contain duplicates")
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or min(args.gpus) < 0:
        raise ValueError("--gpus must contain unique non-negative indices")
    if args.num_envs <= 0 or args.evaluation_num_envs <= 0:
        raise ValueError("training and evaluation environment counts must be positive")
    if args.evaluation_num_envs % SLOPE_COUNT:
        raise ValueError(
            f"--evaluation-num-envs must be a positive multiple of {SLOPE_COUNT}"
        )
    if args.seed < 0 or not args.evaluation_seeds:
        raise ValueError("training/evaluation seeds must be non-negative and non-empty")
    if any(seed < 0 for seed in args.evaluation_seeds):
        raise ValueError("evaluation seeds must be non-negative")
    if len(args.evaluation_seeds) != len(set(args.evaluation_seeds)):
        raise ValueError("evaluation seeds must be unique")
    quota = len(args.evaluation_seeds) * 4
    if args.episodes_per_slope <= 0 or args.episodes_per_slope % quota:
        raise ValueError(
            f"--episodes-per-slope must be positive and divisible by {quota}"
        )
    specs = [RUNS_BY_NAME[name] for name in run_names]
    if args.fat2_weight is not None:
        specs = [replace(spec, fat2_weight=args.fat2_weight) for spec in specs]
    return specs


def _plan(args: argparse.Namespace, specs: Sequence[RunSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": args.task,
        "output_dir": os.fspath(args.output_dir.resolve()),
        "tensorboard_logdir": os.fspath((args.output_dir / "runs").resolve()),
        "resume": bool(args.resume),
        "num_envs": args.num_envs,
        "reward_weight_overrides": args.reward_weight_overrides,
        "evaluation": {
            "num_envs": args.evaluation_num_envs,
            "episodes_per_slope": args.episodes_per_slope,
            "seeds": list(args.evaluation_seeds),
        },
        "workers": [
            {
                "run": spec.name,
                "gpu": args.gpus[index % len(args.gpus)],
                "training_parameters": spec.training_parameters,
            }
            for index, spec in enumerate(specs)
        ],
    }


def _discover_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi failed; GPU inventory is unavailable") from exc
    inventory: list[GpuInfo] = []
    for row in result.stdout.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
        inventory.append(GpuInfo(int(fields[0]), fields[1], int(fields[2])))
    return inventory


def _select_gpus(indices: Sequence[int]) -> list[GpuInfo]:
    inventory = {gpu.index: gpu for gpu in _discover_gpus()}
    missing = [index for index in indices if index not in inventory]
    if missing:
        raise RuntimeError(f"requested GPU indices do not exist: {missing}")
    return [inventory[index] for index in indices]


def _gpu_environment(index: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(index),
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "offline",
        }
    )
    return environment


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )
    except OSError:
        return ""


def _run_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    label: str,
    registry: ProcessRegistry | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    if stop_event is not None and stop_event.is_set():
        raise PipelineCancelled(f"{label} cancelled before launch")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    bound_command = list(command)
    print(f"[{label}] {shlex.join(bound_command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + shlex.join(bound_command) + "\n")
        log.flush()
        process = subprocess.Popen(
            bound_command,
            cwd=REPOSITORY_ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if registry is not None and not registry.add(process):
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise PipelineCancelled(f"{label} cancelled during launch")
        try:
            if process.stdout is None:
                raise RuntimeError(f"{label} has no output stream")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{label}] {line}", end="", flush=True)
            returncode = process.wait()
        finally:
            if registry is not None:
                registry.remove(process)
    if returncode != 0:
        if stop_event is not None and stop_event.is_set():
            raise PipelineCancelled(f"{label} cancelled")
        raise RuntimeError(
            f"{label} failed with exit code {returncode}; log={log_path}\n"
            + _tail(log_path)
        )


def _run_optional_diagnostic(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    label: str,
    stage: str,
    is_current: Callable[[], bool],
    registry: ProcessRegistry,
    stop_event: threading.Event,
) -> bool:
    """Record a diagnostic when possible without gating later training stages."""

    if is_current():
        return True
    try:
        _run_command(
            command,
            environment=environment,
            log_path=log_path,
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
    except PipelineCancelled:
        raise
    except RuntimeError as exc:
        print(
            f"[WARNING] {label} {stage} diagnostic failed; "
            f"continuing with the next training stage: {exc}",
            flush=True,
        )
        return False
    if is_current():
        return True
    print(
        f"[WARNING] {label} {stage} diagnostic produced no current report; "
        "continuing with the next training stage",
        flush=True,
    )
    return False


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _training_invocation_matches(
    configuration: Mapping[str, Any],
    *,
    task: str,
    seed: int,
    num_envs: int | None,
) -> bool:
    return (
        configuration.get("task") == task
        and configuration.get("seed") == seed
        and configuration.get("num_envs") == num_envs
    )


def _ppo_checkpoint(
    directory: Path,
    spec: RunSpec,
    *,
    stage: str,
    task: str,
    seed: int,
    num_envs: int,
    reward_weight_overrides: Mapping[str, float] | None = None,
    teacher: Path | None = None,
    context: Path | None = None,
) -> CheckpointRecord | None:
    if not directory.is_dir():
        return None
    target = guide_max_iterations(stage, spec.rollout_steps)
    records: list[CheckpointRecord] = []
    for path in directory.rglob("*.pt"):
        try:
            checkpoint = load_stage_checkpoint(
                path, expected_stage=stage, validate_runtime=False
            )
            configuration = checkpoint[TRAINING_CONFIGURATION_KEY]
            if configuration["training_parameters"] != spec.training_parameters:
                continue
            if not _training_invocation_matches(
                configuration, task=task, seed=seed, num_envs=num_envs
            ):
                continue
            if (
                reward_weight_overrides is not None
                and reward_weight_overrides_from_configuration(configuration)
                != reward_weight_overrides
            ):
                continue
            if configuration["max_iterations"] != target:
                continue
            if stage == "s2_student_ppo":
                lineage = checkpoint.get(CHECKPOINT_LINEAGE_KEY)
                if not isinstance(lineage, Mapping):
                    continue
                if teacher is None or not _same_path(
                    lineage.get("teacher_checkpoint"), teacher
                ):
                    continue
                if context is None or not _same_path(
                    lineage.get("context_checkpoint"), context
                ):
                    continue
            iteration = checkpoint["iter"]
            completed = int(iteration) + 1
            records.append(
                CheckpointRecord(path.resolve(), completed, completed >= target)
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            continue
    return max(records, key=lambda record: record.completed_updates, default=None)


def _valid_s1_checkpoint(
    path: Path,
    spec: RunSpec,
    teacher: Path,
    *,
    task: str,
    seed: int,
    reward_weight_overrides: Mapping[str, float] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint = load_stage_checkpoint(
            path,
            expected_stage="s1_context_distillation",
            validate_runtime=False,
        )
        configuration = checkpoint[TRAINING_CONFIGURATION_KEY]
        lineage = checkpoint[CHECKPOINT_LINEAGE_KEY]
        return (
            isinstance(lineage, Mapping)
            and isinstance(checkpoint.get("training"), Mapping)
            and configuration["training_parameters"] == spec.training_parameters
            and _training_invocation_matches(
                configuration, task=task, seed=seed, num_envs=None
            )
            and (
                reward_weight_overrides is None
                or reward_weight_overrides_from_configuration(configuration)
                == reward_weight_overrides
            )
            and configuration["max_iterations"]
            == GUIDE_MAX_ITERATIONS["s1_context_distillation"]
            and checkpoint["training"]["completed_iterations"]
            == GUIDE_MAX_ITERATIONS["s1_context_distillation"]
            and configuration["resolved_parameters"]["deterministic_algorithms"]
            is S1_DETERMINISTIC_ALGORITHMS
            and _same_path(lineage.get("teacher_checkpoint"), teacher)
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def _valid_diagnostic(
    path: Path,
    checkpoint: Path,
    *,
    task: str,
    evaluation_num_envs: int,
    episodes_per_slope: int,
    evaluation_seeds: Sequence[int],
    reward_weight_overrides: Mapping[str, float] | None = None,
    teacher: Path | None = None,
    s1_baseline: Path | None = None,
) -> bool:
    dependencies = [checkpoint]
    if teacher is not None:
        dependencies.append(teacher)
    if s1_baseline is not None:
        dependencies.append(s1_baseline)
    if (
        not path.is_file()
        or any(not dependency.is_file() for dependency in dependencies)
        or path.stat().st_mtime_ns
        < max(dependency.stat().st_mtime_ns for dependency in dependencies)
    ):
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        evaluation = report.get("evaluation")
        if (
            report.get("status") != "recorded"
            or report.get("task") != task
            or not _same_path(report["checkpoint"].get("path"), checkpoint)
            or not isinstance(evaluation, Mapping)
            or evaluation.get("deterministic_actions") is not True
            or evaluation.get("num_envs") != evaluation_num_envs
            or evaluation.get("episodes_per_slope_per_stage") != episodes_per_slope
            or evaluation.get("fixed_seeds") != list(evaluation_seeds)
            or evaluation.get("curriculum_stages") != ["training"]
            or (
                reward_weight_overrides is not None
                and evaluation.get("reward_weight_overrides", {})
                != reward_weight_overrides
            )
        ):
            return False
        if teacher is not None:
            binding = report.get("teacher_checkpoint")
            if not isinstance(binding, Mapping) or not _same_path(
                binding.get("path"), teacher
            ):
                return False
        if s1_baseline is not None:
            binding = report.get("s1_baseline")
            if not isinstance(binding, Mapping) or not _same_path(
                binding.get("path"), s1_baseline
            ):
                return False
        return True
    except (OSError, AttributeError, TypeError, ValueError, KeyError):
        return False


def _tensorboard_files(directory: Path) -> list[str]:
    return [
        os.fspath(path.resolve())
        for path in sorted(directory.rglob("events.out.tfevents.*"))
        if path.is_file()
    ]


def _rollout_manifest_matches(
    rollout_dir: Path,
    teacher: Path,
    spec: RunSpec,
    *,
    task: str,
    seed: int,
    num_envs: int,
    reward_weight_overrides: Mapping[str, float] | None = None,
) -> bool:
    manifest_path = rollout_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = manifest["teacher_training_configuration"]
        shards = manifest["shards"]
        teacher_checkpoint = load_stage_checkpoint(
            teacher, expected_stage="s0_teacher", validate_runtime=False
        )
        actual_shards = {
            path.name
            for pattern in ("*.pt", "*.pth")
            for path in rollout_dir.glob(pattern)
            if path.is_file()
        }
        if (
            not isinstance(shards, list)
            or not shards
            or any(
                not isinstance(name, str) or Path(name).name != name for name in shards
            )
            or len(shards) != len(set(shards))
            or set(shards) != actual_shards
        ):
            return False
        validate_rollout_stage_coverage(manifest)
        return (
            manifest.get("schema_version") == ROLLOUT_MANIFEST_SCHEMA_VERSION
            and _same_path(manifest.get("teacher_checkpoint"), teacher)
            and configuration == teacher_checkpoint[TRAINING_CONFIGURATION_KEY]
            and configuration["training_parameters"] == spec.training_parameters
            and _training_invocation_matches(
                configuration, task=task, seed=seed, num_envs=num_envs
            )
            and (
                reward_weight_overrides is None
                or reward_weight_overrides_from_configuration(configuration)
                == reward_weight_overrides
            )
            and manifest.get("num_envs") == ROLLOUT_DEFAULT_NUM_ENVS
            and manifest.get("num_steps_per_stage") == DISTILLATION_ROLLOUT_STEPS
        )
    except (OSError, TypeError, ValueError, KeyError):
        return False


def _teacher_command(
    spec: RunSpec,
    args: argparse.Namespace,
    run_dir: Path,
    resume: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts" / "train_teacher.py"),
        "--task",
        args.task,
        "--num-envs",
        str(args.num_envs),
        "--fat2-weight",
        str(spec.fat2_weight),
        "--rollout-steps",
        str(spec.rollout_steps),
        "--latent-dim",
        str(spec.latent_dim),
        "--history-length",
        str(spec.history_length),
        "--seed",
        str(args.seed),
    ]
    if spec.stability_reward_curriculum:
        command.append("--stability-reward-curriculum")
    for name, weight in getattr(args, "reward_weight_overrides", {}).items():
        command.extend(("--reward-weight", f"{name}={weight!r}"))
    command.extend(
        [
            "--run_name",
            f"{spec.name}-s0",
            "--device",
            LOGICAL_CUDA_DEVICE,
            "--headless",
            f"hydra.run.dir={run_dir / 'hydra' / 's0'}",
        ]
    )
    if resume is None:
        command.extend(("--experiment-dir", os.fspath(run_dir / "s0")))
    else:
        command.extend(("--resume-checkpoint", os.fspath(resume)))
    return command


def _evaluation_command(
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    *,
    teacher: Path | None = None,
    s1_baseline: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts" / "evaluate_policy.py"),
        "--task",
        args.task,
        "--checkpoint",
        os.fspath(checkpoint),
        "--output",
        os.fspath(output),
        "--num-envs",
        str(args.evaluation_num_envs),
        "--episodes-per-slope",
        str(args.episodes_per_slope),
        "--seeds",
        *(str(seed) for seed in args.evaluation_seeds),
        "--device",
        LOGICAL_CUDA_DEVICE,
        "--headless",
    ]
    if teacher is not None:
        command.extend(("--teacher-checkpoint", os.fspath(teacher)))
    if s1_baseline is not None:
        command.extend(("--s1-baseline-report", os.fspath(s1_baseline)))
    return command


def _run_one_pipeline(
    spec: RunSpec,
    *,
    gpu: GpuInfo,
    args: argparse.Namespace,
    registry: ProcessRegistry,
    stop_event: threading.Event,
) -> dict[str, Any]:
    run_dir = args.output_dir / "runs" / spec.name
    logs = run_dir / "logs"
    diagnostics = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = _gpu_environment(gpu.index)
    label = f"{spec.name}/gpu{gpu.index}"
    training_identity = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "reward_weight_overrides": args.reward_weight_overrides,
    }
    s1_identity = {
        "task": args.task,
        "seed": args.seed,
        "reward_weight_overrides": args.reward_weight_overrides,
    }
    diagnostic_identity = {
        "task": args.task,
        "evaluation_num_envs": args.evaluation_num_envs,
        "episodes_per_slope": args.episodes_per_slope,
        "evaluation_seeds": args.evaluation_seeds,
        "reward_weight_overrides": args.reward_weight_overrides,
    }

    teacher_record = _ppo_checkpoint(
        run_dir / "s0", spec, stage="s0_teacher", **training_identity
    )
    if teacher_record is None or not teacher_record.complete:
        resume = None if teacher_record is None else teacher_record.path
        _run_command(
            _teacher_command(spec, args, run_dir, resume),
            environment=environment,
            log_path=logs / "01_s0_teacher.log",
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
        teacher_record = _ppo_checkpoint(
            run_dir / "s0", spec, stage="s0_teacher", **training_identity
        )
    if teacher_record is None or not teacher_record.complete:
        raise RuntimeError(f"{label} produced no complete S0 checkpoint")
    teacher = teacher_record.path

    s0_report = diagnostics / "s0.json"
    s0_diagnostic_recorded = _run_optional_diagnostic(
        _evaluation_command(args, teacher, s0_report),
        environment=environment,
        log_path=logs / "02_s0_diagnostic.log",
        label=label,
        stage="S0",
        is_current=lambda: _valid_diagnostic(s0_report, teacher, **diagnostic_identity),
        registry=registry,
        stop_event=stop_event,
    )

    context = run_dir / "s1_context.pt"
    if not _valid_s1_checkpoint(context, spec, teacher, **s1_identity):
        rollout_dir = run_dir / "rollouts"
        if not _rollout_manifest_matches(
            rollout_dir, teacher, spec, **training_identity
        ):
            collect_command = [
                sys.executable,
                os.fspath(REPOSITORY_ROOT / "scripts" / "collect_teacher_rollouts.py"),
                "--task",
                args.task,
                "--teacher",
                os.fspath(teacher),
                "--output-dir",
                os.fspath(rollout_dir),
                "--seed",
                str(args.seed),
                "--device",
                LOGICAL_CUDA_DEVICE,
                "--headless",
            ]
            if rollout_dir.exists() and any(rollout_dir.iterdir()):
                collect_command.append("--overwrite")
            _run_command(
                collect_command,
                environment=environment,
                log_path=logs / "03_s1_rollouts.log",
                label=label,
                registry=registry,
                stop_event=stop_event,
            )
        if not _rollout_manifest_matches(
            rollout_dir, teacher, spec, **training_identity
        ):
            raise RuntimeError(f"{label} produced no complete S1 rollout manifest")
        _run_command(
            [
                sys.executable,
                os.fspath(REPOSITORY_ROOT / "scripts" / "train_context.py"),
                "--task",
                args.task,
                "--teacher",
                os.fspath(teacher),
                "--rollout-dir",
                os.fspath(rollout_dir),
                "--output",
                os.fspath(context),
                "--device",
                LOGICAL_CUDA_DEVICE,
                "--training-seed",
                str(args.seed),
            ],
            environment=environment,
            log_path=logs / "04_s1_context.log",
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
    if not _valid_s1_checkpoint(context, spec, teacher, **s1_identity):
        raise RuntimeError(f"{label} produced no complete S1 checkpoint")

    s1_report = diagnostics / "s1.json"
    s1_diagnostic_recorded = _run_optional_diagnostic(
        _evaluation_command(args, context, s1_report, teacher=teacher),
        environment=environment,
        log_path=logs / "05_s1_diagnostic.log",
        label=label,
        stage="S1",
        is_current=lambda: _valid_diagnostic(
            s1_report,
            context,
            teacher=teacher,
            **diagnostic_identity,
        ),
        registry=registry,
        stop_event=stop_event,
    )

    s2_record = _ppo_checkpoint(
        run_dir / "s2",
        spec,
        stage="s2_student_ppo",
        **training_identity,
        teacher=teacher,
        context=context,
    )
    if s2_record is None or not s2_record.complete:
        command = [
            sys.executable,
            os.fspath(REPOSITORY_ROOT / "scripts" / "finetune_student.py"),
            "--task",
            args.task,
            "--teacher",
            os.fspath(teacher),
            "--context",
            os.fspath(context),
            "--bootstrap-dir",
            os.fspath(run_dir / "s2"),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--run_name",
            f"{spec.name}-s2",
            "--device",
            LOGICAL_CUDA_DEVICE,
            "--headless",
            f"hydra.run.dir={run_dir / 'hydra' / 's2'}",
        ]
        if s2_record is not None:
            command.extend(("--resume-checkpoint", os.fspath(s2_record.path)))
        _run_command(
            command,
            environment=environment,
            log_path=logs / "06_s2_student.log",
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
        s2_record = _ppo_checkpoint(
            run_dir / "s2",
            spec,
            stage="s2_student_ppo",
            **training_identity,
            teacher=teacher,
            context=context,
        )
    if s2_record is None or not s2_record.complete:
        raise RuntimeError(f"{label} produced no complete S2 checkpoint")

    s2_report = diagnostics / "s2.json"
    s1_baseline = s1_report if s1_diagnostic_recorded else None
    s2_diagnostic_recorded = _run_optional_diagnostic(
        _evaluation_command(
            args,
            s2_record.path,
            s2_report,
            teacher=teacher,
            s1_baseline=s1_baseline,
        ),
        environment=environment,
        log_path=logs / "07_s2_diagnostic.log",
        label=label,
        stage="S2",
        is_current=lambda: _valid_diagnostic(
            s2_report,
            s2_record.path,
            teacher=teacher,
            s1_baseline=s1_baseline,
            **diagnostic_identity,
        ),
        registry=registry,
        stop_event=stop_event,
    )

    result = {
        "run": spec.name,
        "gpu": gpu.index,
        "training_parameters": spec.training_parameters,
        "reward_weight_overrides": args.reward_weight_overrides,
        "diagnostics": {
            "s0_recorded": s0_diagnostic_recorded,
            "s1_recorded": s1_diagnostic_recorded,
            "s2_recorded": s2_diagnostic_recorded,
        },
        "artifacts": {
            "s0_checkpoint": os.fspath(teacher),
            "s0_diagnostic": os.fspath(s0_report.resolve()),
            "s1_checkpoint": os.fspath(context.resolve()),
            "s1_diagnostic": os.fspath(s1_report.resolve()),
            "s2_checkpoint": os.fspath(s2_record.path),
            "s2_diagnostic": os.fspath(s2_report.resolve()),
        },
        "tensorboard": {
            "logdir": os.fspath((args.output_dir / "runs").resolve()),
            "s0_root": os.fspath((run_dir / "s0").resolve()),
            "s0_event_files": _tensorboard_files(run_dir / "s0"),
            "s2_root": os.fspath((run_dir / "s2").resolve()),
            "s2_event_files": _tensorboard_files(run_dir / "s2"),
        },
    }
    write_json_atomic(run_dir / "summary.json", result)
    return result


def _validate_runtime_inputs(args: argparse.Namespace) -> None:
    isaaclab = os.environ.get("ISAACLAB_PATH")
    if not isaaclab or not Path(isaaclab).is_dir():
        raise RuntimeError("ISAACLAB_PATH must name the existing IsaacLab checkout")
    if args.output_dir.exists() and not args.resume and any(args.output_dir.iterdir()):
        raise RuntimeError("output directory is not empty; use --resume")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    specs = _validate_args(args)
    plan = _plan(args, specs)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    _validate_runtime_inputs(args)
    gpus = _select_gpus(args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "plan.json", plan)

    stop_event = threading.Event()
    registry = ProcessRegistry()
    result_lock = threading.Lock()
    results: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []

    def worker(gpu: GpuInfo, assigned_specs: Sequence[RunSpec]) -> None:
        for spec in assigned_specs:
            if stop_event.is_set():
                return
            try:
                result = _run_one_pipeline(
                    spec,
                    gpu=gpu,
                    args=args,
                    registry=registry,
                    stop_event=stop_event,
                )
                with result_lock:
                    results[spec.name] = result
            except PipelineCancelled:
                return
            except Exception as exc:  # noqa: BLE001
                with result_lock:
                    failures.append((spec.name, str(exc)))
                continue

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate_from_signal(signum, _frame) -> None:
        stop_event.set()
        registry.terminate_all()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_from_signal)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(worker, gpu, specs[index :: len(gpus)])
                for index, gpu in enumerate(gpus)
            ]
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        stop_event.set()
        registry.terminate_all()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if stop_event.is_set():
            registry.terminate_all()

    if failures:
        details = "\n".join(f"- {name}: {message}" for name, message in failures)
        raise RuntimeError(f"ablation pipeline failed:\n{details}")
    if len(results) != len(specs):
        raise RuntimeError(
            "ablation pipeline stopped before every requested run completed"
        )
    ordered_results = [results[spec.name] for spec in specs]
    summary = {"schema_version": 1, "plan": plan, "runs": ordered_results}
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(f"completed {len(results)} training pipelines: {args.output_dir}")
    print(f"TensorBoard logdir: {(args.output_dir / 'runs').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
