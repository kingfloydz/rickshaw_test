#!/usr/bin/env python3
"""Run and rank the controlled 2^3 reward-weight experiment on selected GPUs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Any

from _isaaclab_wrappers import REPOSITORY_ROOT, add_project_source_to_path

add_project_source_to_path()

import run_ablation_pipeline as multi_gpu  # noqa: E402
from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    FORMAL_EVALUATION_COMMAND_PROTOCOL,
    FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
    POLICY_DIAGNOSTIC_SCHEMA_VERSION,
    SIGNED_SLOPES,
)
from g1_rickshaw_lab.reward_calibration import (  # noqa: E402
    load_and_recompute_reward_calibration_report,
)
from g1_rickshaw_lab.reward_profile import (  # noqa: E402
    reward_weight_overrides_from_configuration,
)
from g1_rickshaw_lab.reward_tuning import (  # noqa: E402
    aggregate_profile_results,
    factorial_effects,
    factorial_reward_profiles,
    load_reward_tuning_config,
    policy_diagnostic_rank_metrics,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
)
from g1_rickshaw_lab.validation import utc_timestamp, write_json_atomic  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "reward_tuning.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "reward_tuning_screen"
DEFAULT_EVALUATION_SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True, slots=True)
class RewardJob:
    profile: Mapping[str, Any]
    training_seed: int
    gpu_index: int

    @property
    def name(self) -> str:
        return f"{self.profile['name']}/seed_{self.training_seed:05d}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--top-from", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=DEFAULT_EVALUATION_SEEDS)
    parser.add_argument("--evaluation-num-envs", type=int, default=380)
    parser.add_argument("--episodes-per-slope", type=int, default=100)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _selected_profiles(
    args: argparse.Namespace,
    profiles: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    by_name = {str(profile["name"]): profile for profile in profiles}
    if args.profiles is not None and args.top_from is not None:
        raise ValueError("--profiles and --top-from are mutually exclusive")
    if args.top_from is not None:
        source = json.loads(args.top_from.read_text(encoding="utf-8"))
        if source.get("config") != config:
            raise ValueError("--top-from used a different reward tuning configuration")
        ranking = source.get("ranking") if isinstance(source, Mapping) else None
        if not isinstance(ranking, list) or args.top_k <= 0 or args.top_k > len(ranking):
            raise ValueError("--top-from has no complete ranking for --top-k")
        names = [str(item["profile"]) for item in ranking[: args.top_k]]
    elif args.profiles is None:
        names = list(by_name)
    else:
        names = list(args.profiles)
    if len(names) != len(set(names)):
        raise ValueError("reward profiles must be unique")
    unknown = set(names) - set(by_name)
    if unknown:
        raise ValueError(f"unknown reward profiles: {sorted(unknown)}")
    return [by_name[name] for name in names]


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("training_seeds", "evaluation_seeds", "gpus"):
        values = list(getattr(args, name))
        if not values or len(values) != len(set(values)) or min(values) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must contain unique non-negative values")
    if args.evaluation_num_envs <= 0 or args.evaluation_num_envs % 19:
        raise ValueError("--evaluation-num-envs must be a positive multiple of 19")
    if (
        args.episodes_per_slope <= 0
        or args.episodes_per_slope % len(args.evaluation_seeds)
    ):
        raise ValueError("--episodes-per-slope must be divisible by evaluation seed count")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")


def _jobs(
    profiles: Sequence[Mapping[str, Any]],
    training_seeds: Sequence[int],
    gpus: Sequence[int],
) -> list[RewardJob]:
    jobs: list[RewardJob] = []
    for profile in profiles:
        for seed in training_seeds:
            jobs.append(
                RewardJob(
                    profile=profile,
                    training_seed=int(seed),
                    gpu_index=int(gpus[len(jobs) % len(gpus)]),
                )
            )
    return jobs


def _plan(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    jobs: Sequence[RewardJob],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": config["task"],
        "config": os.fspath(args.config.resolve()),
        "output_dir": os.fspath(args.output_dir.resolve()),
        "tensorboard_logdir": os.fspath((args.output_dir / "runs").resolve()),
        "fixed": config["fixed"],
        "evaluation": {
            "num_envs": args.evaluation_num_envs,
            "episodes_per_slope": args.episodes_per_slope,
            "seeds": list(args.evaluation_seeds),
        },
        "jobs": [
            {
                "name": job.name,
                "profile": job.profile["name"],
                "training_seed": job.training_seed,
                "gpu": job.gpu_index,
                "reward_weight_overrides": job.profile["reward_weight_overrides"],
            }
            for job in jobs
        ],
    }


def _checkpoint(
    directory: Path,
    job: RewardJob,
    config: Mapping[str, Any],
) -> multi_gpu.CheckpointRecord | None:
    if not directory.is_dir():
        return None
    fixed = config["fixed"]
    expected_parameters = {
        "fat2_weight": fixed["fat2_weight"],
        "rollout_steps": fixed["rollout_steps"],
        "latent_dim": fixed["latent_dim"],
    }
    records: list[multi_gpu.CheckpointRecord] = []
    for path in directory.rglob("model_*.pt"):
        try:
            checkpoint = load_stage_checkpoint(
                path, expected_stage="s0_teacher", validate_runtime=False
            )
            training = checkpoint[TRAINING_CONFIGURATION_KEY]
            if (
                training["training_parameters"] != expected_parameters
                or training["task"] != config["task"]
                or training["num_envs"] != fixed["num_envs"]
                or training["seed"] != job.training_seed
                or training["max_iterations"] != fixed["max_iterations"]
                or reward_weight_overrides_from_configuration(training)
                != job.profile["reward_weight_overrides"]
            ):
                continue
            completed = int(checkpoint["iter"]) + 1
            records.append(
                multi_gpu.CheckpointRecord(
                    path.resolve(),
                    completed,
                    completed >= fixed["max_iterations"],
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            continue
    return max(records, key=lambda item: item.completed_updates, default=None)


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _valid_diagnostic(
    path: Path,
    checkpoint: Path,
    job: RewardJob,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        evaluation = report["evaluation"]
        return (
            report["schema_version"] == POLICY_DIAGNOSTIC_SCHEMA_VERSION
            and report["status"] == "recorded"
            and report["task"] == config["task"]
            and _same_path(report["checkpoint"]["path"], checkpoint)
            and report["checkpoint"]["stage"] == "s0_teacher"
            and evaluation["deterministic_actions"] is True
            and evaluation["fixed_seeds"] == list(args.evaluation_seeds)
            and evaluation["signed_slopes"] == list(SIGNED_SLOPES)
            and evaluation["num_envs"] == args.evaluation_num_envs
            and evaluation["episodes_per_slope_per_stage"] == args.episodes_per_slope
            and evaluation["curriculum_stages"] == ["training"]
            and evaluation["command_protocol"] == FORMAL_EVALUATION_COMMAND_PROTOCOL
            and evaluation["cross_case_protocol"] == FORMAL_EVALUATION_CROSS_CASE_PROTOCOL
            and evaluation["fat2_weight"] == config["fixed"]["fat2_weight"]
            and evaluation["rollout_steps"] == config["fixed"]["rollout_steps"]
            and evaluation["latent_dim"] == config["fixed"]["latent_dim"]
            and evaluation["reward_weight_overrides"]
            == job.profile["reward_weight_overrides"]
            and path.stat().st_mtime_ns >= checkpoint.stat().st_mtime_ns
            and bool(policy_diagnostic_rank_metrics(report))
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _valid_calibration(
    path: Path,
    checkpoint: Path,
    job: RewardJob,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> bool:
    try:
        loaded = load_and_recompute_reward_calibration_report(
            path, teacher_checkpoint_path=checkpoint
        )
        report = loaded["report"]
        source = report["source"]
        term_weights = source["term_weights"]
        calibration = config["calibration"]
        expected_counts = {
            f"{slope:+.2f}": calibration["samples_per_slope"]
            for slope in SIGNED_SLOPES
        }
        return (
            report["status"] in {"passed", "failed"}
            and _same_path(source["checkpoint"]["path"], checkpoint)
            and source["fixed_seed"] == args.evaluation_seeds[0]
            and source["fixed_slopes"] == list(SIGNED_SLOPES)
            and source["task"] == config["task"]
            and source["num_envs"] == args.evaluation_num_envs
            and source["policy_kind"] == "teacher"
            and source["slope_sample_counts"] == expected_counts
            and source["policy_steps"] <= calibration["max_policy_steps"]
            and float(term_weights["fat2_prior_exp"])
            == float(config["fixed"]["fat2_weight"])
            and all(
                float(term_weights[name]) == float(weight)
                for name, weight in job.profile["reward_weight_overrides"].items()
            )
            and path.stat().st_mtime_ns >= checkpoint.stat().st_mtime_ns
        )
    except (OSError, TypeError, ValueError, KeyError):
        return False


def _teacher_command(
    job: RewardJob,
    config: Mapping[str, Any],
    run_dir: Path,
    resume: Path | None,
) -> list[str]:
    fixed = config["fixed"]
    command = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts" / "train_teacher.py"),
        "--task",
        config["task"],
        "--num-envs",
        str(fixed["num_envs"]),
        "--fat2-weight",
        str(fixed["fat2_weight"]),
        "--rollout-steps",
        str(fixed["rollout_steps"]),
        "--latent-dim",
        str(fixed["latent_dim"]),
        "--seed",
        str(job.training_seed),
        "--max_iterations",
        str(fixed["max_iterations"]),
    ]
    for name, weight in job.profile["reward_weight_overrides"].items():
        command.extend(("--reward-weight", f"{name}={weight!r}"))
    command.extend(
        (
            "--run_name",
            f"{job.profile['name']}-seed{job.training_seed}-s0",
            "--device",
            multi_gpu.LOGICAL_CUDA_DEVICE,
            "--headless",
            f"hydra.run.dir={run_dir / 'hydra'}",
        )
    )
    if resume is None:
        command.extend(("--experiment-dir", os.fspath(run_dir / "s0")))
    else:
        command.extend(("--resume-checkpoint", os.fspath(resume)))
    return command


def _evaluation_command(
    checkpoint: Path,
    output: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> list[str]:
    return [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts" / "evaluate_policy.py"),
        "--task",
        config["task"],
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
        multi_gpu.LOGICAL_CUDA_DEVICE,
        "--headless",
    ]


def _calibration_command(
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> list[str]:
    calibration = config["calibration"]
    return [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "scripts" / "calibrate_rewards.py"),
        "--checkpoint",
        os.fspath(checkpoint),
        "--task",
        config["task"],
        "--policy-kind",
        "teacher",
        "--seed",
        str(args.evaluation_seeds[0]),
        "--num-envs",
        str(args.evaluation_num_envs),
        "--samples-per-slope",
        str(calibration["samples_per_slope"]),
        "--max-policy-steps",
        str(calibration["max_policy_steps"]),
        "--output-dir",
        os.fspath(output_dir),
        "--device",
        multi_gpu.LOGICAL_CUDA_DEVICE,
        "--headless",
    ]


def _run_job(
    job: RewardJob,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    registry: multi_gpu.ProcessRegistry,
    stop_event: threading.Event,
) -> dict[str, Any]:
    run_dir = args.output_dir / "runs" / job.profile["name"] / f"seed_{job.training_seed:05d}"
    logs = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = multi_gpu._gpu_environment(job.gpu_index)
    label = f"{job.name}/gpu{job.gpu_index}"

    record = _checkpoint(run_dir / "s0", job, config)
    if record is None or not record.complete:
        multi_gpu._run_command(
            _teacher_command(
                job,
                config,
                run_dir,
                None if record is None else record.path,
            ),
            environment=environment,
            log_path=logs / "01_train_teacher.log",
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
        record = _checkpoint(run_dir / "s0", job, config)
    if record is None or not record.complete:
        raise RuntimeError(f"{label} produced no complete S0 checkpoint")

    diagnostic = run_dir / "policy_diagnostic.json"
    if not _valid_diagnostic(diagnostic, record.path, job, args, config):
        multi_gpu._run_command(
            _evaluation_command(record.path, diagnostic, args, config),
            environment=environment,
            log_path=logs / "02_evaluate_policy.log",
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
    if not _valid_diagnostic(diagnostic, record.path, job, args, config):
        raise RuntimeError(f"{label} produced no valid policy diagnostic")

    calibration_dir = run_dir / "reward_calibration"
    calibration_report = calibration_dir / "reward_calibration.json"
    if not _valid_calibration(
        calibration_report, record.path, job, args, config
    ):
        multi_gpu._run_command(
            _calibration_command(record.path, calibration_dir, args, config),
            environment=environment,
            log_path=logs / "03_calibrate_rewards.log",
            label=label,
            registry=registry,
            stop_event=stop_event,
        )
    if not _valid_calibration(
        calibration_report, record.path, job, args, config
    ):
        raise RuntimeError(f"{label} produced no valid reward calibration")

    report = json.loads(diagnostic.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_report.read_text(encoding="utf-8"))
    result = {
        "job": job.name,
        "profile": job.profile["name"],
        "training_seed": job.training_seed,
        "gpu": job.gpu_index,
        "reward_weight_overrides": job.profile["reward_weight_overrides"],
        "checkpoint": os.fspath(record.path),
        "diagnostic": os.fspath(diagnostic.resolve()),
        "calibration": os.fspath(calibration_report.resolve()),
        "calibration_status": calibration["status"],
        "metrics": policy_diagnostic_rank_metrics(report),
        "tensorboard_event_files": multi_gpu._tensorboard_files(run_dir / "s0"),
    }
    write_json_atomic(run_dir / "summary.json", result)
    return result


def _validate_runtime(args: argparse.Namespace) -> None:
    isaaclab = os.environ.get("ISAACLAB_PATH")
    if not isaaclab or not Path(isaaclab).is_dir():
        raise RuntimeError("ISAACLAB_PATH must name the existing IsaacLab checkout")
    if args.output_dir.exists() and not args.resume and any(args.output_dir.iterdir()):
        raise RuntimeError("output directory is not empty; use --resume")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    _validate_args(args)
    config = load_reward_tuning_config(args.config)
    profiles = factorial_reward_profiles(config)
    selected = _selected_profiles(args, profiles, config)
    jobs = _jobs(selected, args.training_seeds, args.gpus)
    plan = _plan(args, config, jobs)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    _validate_runtime(args)
    selected_gpus = multi_gpu._select_gpus(args.gpus)
    gpu_indices = {gpu.index for gpu in selected_gpus}
    if any(job.gpu_index not in gpu_indices for job in jobs):
        raise RuntimeError("reward job plan references an unavailable GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "plan.json", plan)

    registry = multi_gpu.ProcessRegistry()
    stop_event = threading.Event()
    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(gpu_index: int, assigned: Sequence[RewardJob]) -> None:
        for job in assigned:
            if stop_event.is_set():
                return
            try:
                result = _run_job(job, args, config, registry, stop_event)
                with lock:
                    results.append(result)
            except multi_gpu.PipelineCancelled:
                return
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failures.append((job.name, str(exc)))

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, _frame) -> None:
        stop_event.set()
        registry.terminate_all()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [
                executor.submit(
                    worker,
                    gpu,
                    [job for job in jobs if job.gpu_index == gpu],
                )
                for gpu in args.gpus
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
        detail = "\n".join(f"- {name}: {message}" for name, message in failures)
        raise RuntimeError(f"reward tuning failed:\n{detail}")
    if len(results) != len(jobs):
        raise RuntimeError("reward tuning stopped before every job completed")
    results.sort(key=lambda item: (item["profile"], item["training_seed"]))
    ranking = aggregate_profile_results(results)
    profiles_by_name = {str(profile["name"]): profile for profile in selected}
    best_profile = str(ranking[0]["profile"])
    summary = {
        "schema_version": 1,
        "created_utc": utc_timestamp(),
        "plan": plan,
        "config": config,
        "results": results,
        "ranking": ranking,
        "recommendation": {
            "profile": best_profile,
            "reward_weight_overrides": profiles_by_name[best_profile][
                "reward_weight_overrides"
            ],
        },
        "factorial_analysis": factorial_effects(ranking, selected, config["factors"]),
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(f"completed {len(results)} reward-tuning jobs: {args.output_dir}")
    print(f"best profile: {best_profile}")
    print(f"TensorBoard logdir: {(args.output_dir / 'runs').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
