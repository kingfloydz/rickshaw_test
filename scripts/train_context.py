#!/usr/bin/env python3
"""Train the S1 causal-TCN context encoder from teacher rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

from _mjlab_wrappers import add_project_source_to_path, require_existing_file

add_project_source_to_path()

import torch  # noqa: E402
import numpy as np  # noqa: E402
from torch.distributions import Independent, Normal  # noqa: E402

from g1_rickshaw_lab.provenance import (  # noqa: E402
    extract_checkpoint_metadata,
    save_checkpoint_atomic,
)
from g1_rickshaw_lab.policy_schema import ACTOR_OBSERVATION_DIM  # noqa: E402
from g1_rickshaw_lab.reward_profile import (  # noqa: E402
    REWARD_WEIGHT_OVERRIDES_KEY,
    reward_weight_overrides_from_configuration,
)
from g1_rickshaw_lab.rl import G1RickshawStudentActor, StudentDistillationLoss  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_LINEAGE_KEY,
    CHECKPOINT_STAGE_KEY,
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_PARAMETERS,
    S1_DETERMINISTIC_ALGORITHMS,
    TRAINING_CONFIGURATION_KEY as TRAINING_CONFIGURATION_CHECKPOINT_KEY,
    build_training_configuration,
    extract_gaussian_actor_state,
    load_stage_checkpoint,
    validate_guide_training_configuration,
    validate_rollout_stage_coverage,
)

from _rollout_audit import (  # noqa: E402
    AUDIT_TENSOR_NAMES,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    normalize_audit_tensors,
    validate_rollout_sample_audit,
)

CHECKPOINT_SCHEMA_VERSION = 1
S1_GUIDE_PARAMETERS = GUIDE_TRAINING_PARAMETERS["s1_context_distillation"]
REQUIRED_TENSORS = (
    "current",
    "history",
    "teacher_action_mean",
    "teacher_action_std",
    "z_star",
)


def validate_s1_arguments(args: argparse.Namespace) -> None:
    """Validate only the value-domain requirements used by S1 training."""

    positive_integers = (
        "max_iterations",
        "batch_size",
        "mini_batch_size",
        "log_interval",
        "validation_interval",
    )
    for name in positive_integers:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.mini_batch_size > args.batch_size:
        raise ValueError("mini_batch_size cannot exceed batch_size")
    for name in ("context_lr", "max_grad_norm"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    for name in ("training_seed", "validation_seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or value < 0 or value > 2**32 - 1:
            raise ValueError(f"{name} must lie in [0, 2**32-1]")


def seed_s1_training(seed: int) -> torch.Generator:
    """Seed model initialization and every training-batch draw."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > 2**32 - 1
    ):
        raise ValueError("S1 training seed must lie in [0, 2**32-1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(S1_DETERMINISTIC_ALGORITHMS)
    torch.backends.cudnn.deterministic = S1_DETERMINISTIC_ALGORITHMS
    torch.backends.cudnn.benchmark = not S1_DETERMINISTIC_ALGORITHMS
    return torch.Generator(device="cpu").manual_seed(seed)


def _torch_load(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"rollout shard must contain a mapping: {path}")
    if value.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"rollout shard has an unsupported schema: {path}")
    rollout = value.get("rollout")
    if not isinstance(rollout, Mapping):
        raise ValueError(
            f"rollout shard is missing its canonical rollout mapping: {path}"
        )
    return rollout


def _as_tensor(value: Any, name: str, path: Path) -> torch.Tensor:
    if value is None:
        raise KeyError(f"{path} is missing rollout tensor {name!r}")
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor


def _normalize_shard(
    path: Path,
    latent_dim: int = 16,
    history_length: int = 61,
) -> dict[str, torch.Tensor]:
    raw = _torch_load(path)
    current = _as_tensor(raw.get("current"), "current", path).float()
    history = _as_tensor(raw.get("history"), "history", path).float()
    teacher_mean = _as_tensor(
        raw.get("teacher_action_mean"), "teacher_action_mean", path
    ).float()
    if current.ndim != 2 or current.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise ValueError(
            f"current must have shape [N,{ACTOR_OBSERVATION_DIM}], got {tuple(current.shape)}"
        )
    batch_size = current.shape[0]
    normalized = {
        "current": current,
        "history": history,
        "teacher_action_mean": teacher_mean,
        "teacher_action_std": _as_tensor(
            raw.get("teacher_action_std"), "teacher_action_std", path
        ).float(),
        "z_star": _as_tensor(raw.get("z_star"), "z_star", path).float(),
    }
    expected_shapes = {
        "history": (batch_size, history_length, ACTOR_OBSERVATION_DIM),
        "teacher_action_mean": (batch_size, 29),
        "teacher_action_std": (batch_size, 29),
        "z_star": (batch_size, latent_dim),
    }
    for name, shape in expected_shapes.items():
        if tuple(normalized[name].shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(normalized[name].shape)}"
            )
    if torch.any(normalized["teacher_action_std"] <= 0.0):
        raise ValueError("teacher_action_std must be strictly positive")
    normalized.update(normalize_audit_tensors(raw, batch_size=batch_size))
    return normalized


def load_rollout_dataset(
    rollout_dir: Path,
    manifest: Mapping[str, Any],
    latent_dim: int,
    history_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load shards and independently reconstruct their formal sample audit."""

    shards = sorted(
        path for pattern in ("*.pt", "*.pth") for path in rollout_dir.glob(pattern)
    )
    if not shards:
        raise FileNotFoundError(f"no .pt/.pth rollout shards found in {rollout_dir}")
    loaded = [_normalize_shard(path, latent_dim, history_length) for path in shards]
    tensors = {
        name: torch.cat([shard[name] for shard in loaded], dim=0)
        for name in (*REQUIRED_TENSORS, *AUDIT_TENSOR_NAMES)
    }
    stage_coverage = validate_rollout_sample_audit(manifest, tensors)
    return tensors, stage_coverage


def validate_rollout_manifest(
    rollout_dir: Path,
    teacher_path: Path,
    teacher_metadata: Any,
) -> Mapping[str, Any]:
    manifest_path = rollout_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"verified rollout manifest is missing: {manifest_path}"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"invalid rollout manifest schema: {manifest_path}")
    if manifest.get("on_policy") is not True:
        raise ValueError("S1 accepts only teacher on-policy rollout manifests")
    collection_seed = manifest.get("collection_seed")
    if (
        isinstance(collection_seed, bool)
        or not isinstance(collection_seed, int)
        or collection_seed < 0
        or collection_seed > 2**32 - 1
    ):
        raise ValueError("rollout manifest is missing its valid collection seed")
    if (
        Path(str(manifest.get("teacher_checkpoint"))).resolve()
        != teacher_path.resolve()
    ):
        raise ValueError(
            "rollout manifest was generated by a different teacher checkpoint"
        )
    if manifest.get("teacher_provenance") != teacher_metadata.to_mapping():
        raise ValueError(
            "rollout manifest teacher provenance differs from the S0 checkpoint"
        )
    shard_names = manifest.get("shards")
    if not isinstance(shard_names, list) or not shard_names:
        raise ValueError("rollout manifest has no shard paths")
    actual_names = {
        path.name for pattern in ("*.pt", "*.pth") for path in rollout_dir.glob(pattern)
    }
    if actual_names != set(shard_names):
        raise ValueError("rollout directory files differ from the manifest shard set")
    validate_rollout_stage_coverage(manifest)
    return manifest


def initialize_actor_from_checkpoint(
    student: G1RickshawStudentActor, checkpoint: Mapping[str, Any]
) -> None:
    student.actor.load_state_dict(extract_gaussian_actor_state(checkpoint), strict=True)


def _batch(
    dataset: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    indices = indices.to(device=device, non_blocking=True)
    return {name: dataset[name][indices] for name in REQUIRED_TENSORS}


def _validation_action_kl(
    student: G1RickshawStudentActor,
    dataset: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
    mini_batch_size: int,
) -> float:
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    student.eval()
    with torch.no_grad():
        for start in range(0, indices.numel(), mini_batch_size):
            sample = _batch(dataset, indices[start : start + mini_batch_size], device)
            teacher = Independent(
                Normal(sample["teacher_action_mean"], sample["teacher_action_std"]), 1
            )
            student_distribution = student(sample["current"], sample["history"])
            divergence = torch.distributions.kl_divergence(
                teacher, student_distribution
            )
            total += divergence.sum().to(dtype=torch.float64)
            count += divergence.numel()
    student.train()
    student.actor.eval()
    return float(total.cpu()) / count


def train(args: argparse.Namespace) -> Path:
    batch_generator = seed_s1_training(args.training_seed)
    teacher_path = require_existing_file(args.teacher, "teacher checkpoint")
    teacher_checkpoint = load_stage_checkpoint(
        teacher_path,
        expected_stage="s0_teacher",
        validate_runtime=True,
    )
    teacher_training_configuration = dict(
        teacher_checkpoint[TRAINING_CONFIGURATION_CHECKPOINT_KEY]
    )
    training_parameters = teacher_training_configuration["training_parameters"]
    reward_weight_overrides = reward_weight_overrides_from_configuration(
        teacher_training_configuration
    )
    latent_dim = int(training_parameters["latent_dim"])
    history_length = int(training_parameters["history_length"])
    rollout_steps = int(training_parameters["rollout_steps"])
    if teacher_training_configuration["task"] != args.task:
        raise ValueError("S1 task differs from the S0 teacher training task")
    metadata = extract_checkpoint_metadata(teacher_checkpoint)
    teacher_curriculum_iteration = teacher_checkpoint.get(
        CHECKPOINT_CURRICULUM_ITERATION_KEY
    )
    if isinstance(teacher_curriculum_iteration, bool) or not isinstance(
        teacher_curriculum_iteration, int
    ):
        raise RuntimeError(
            "S0 teacher checkpoint is missing its audited curriculum iteration"
        )
    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.is_dir():
        raise FileNotFoundError(f"rollout directory does not exist: {rollout_dir}")
    rollout_manifest = validate_rollout_manifest(
        rollout_dir,
        teacher_path,
        metadata,
    )
    if (
        rollout_manifest.get("teacher_training_configuration")
        != teacher_training_configuration
    ):
        raise ValueError(
            "rollout manifest training configuration differs from the S0 teacher"
        )
    dataset, stage_coverage = load_rollout_dataset(
        rollout_dir,
        rollout_manifest,
        latent_dim,
        history_length,
    )
    if args.device == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    for name in REQUIRED_TENSORS:
        dataset[name] = dataset[name].to(device=device)
    student = G1RickshawStudentActor(latent_dim, history_length).to(device)
    initialize_actor_from_checkpoint(student, teacher_checkpoint)

    student.actor.eval()
    for parameter in student.actor.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        student.context_encoder.parameters(), lr=args.context_lr
    )
    criterion = StudentDistillationLoss()
    num_samples = dataset["current"].shape[0]
    if num_samples < 2:
        raise ValueError("rollout dataset must contain at least two samples")
    split_generator = torch.Generator(device="cpu").manual_seed(args.validation_seed)
    permutation = torch.randperm(num_samples, generator=split_generator)
    validation_size = max(1, int(round(num_samples * args.validation_fraction)))
    validation_size = min(validation_size, num_samples - 1)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    batch_size = args.batch_size
    mini_batch_size = args.mini_batch_size
    training_configuration = build_training_configuration(
        stage="s1_context_distillation",
        task=args.task,
        num_envs=None,
        seed=args.training_seed,
        max_iterations=args.max_iterations,
        guide_parameters=S1_GUIDE_PARAMETERS,
        resolved_parameters={
            "optimizer": "adam",
            "context_learning_rate": args.context_lr,
            "batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "gradient_clip": args.max_grad_norm,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "validation_interval": args.validation_interval,
            "teacher_rollout_num_envs": int(rollout_manifest["num_envs"]),
            "teacher_rollout_steps_per_stage": int(
                rollout_manifest["num_steps_per_stage"]
            ),
            "teacher_rollout_samples": int(rollout_manifest["num_samples"]),
            "deterministic_algorithms": S1_DETERMINISTIC_ALGORITHMS,
            REWARD_WEIGHT_OVERRIDES_KEY: reward_weight_overrides,
        },
        actor_initialized_from_teacher=True,
        stage_coverage=stage_coverage,
        fat2_weight=float(training_parameters["fat2_weight"]),
        latent_dim=latent_dim,
        history_length=history_length,
        rollout_steps=rollout_steps,
        stability_reward_curriculum=bool(
            training_parameters["stability_reward_curriculum"]
        ),
    )
    validate_guide_training_configuration(
        training_configuration,
        expected_stage="s1_context_distillation",
    )
    output = Path(args.output)
    last_metrics: dict[str, torch.Tensor] = {}
    validation_history: list[dict[str, float | int | bool]] = []
    best_validation_kl = float("inf")
    best_iteration = 0
    best_state: dict[str, torch.Tensor] | None = None
    completed_iterations = 0
    student.train()
    student.actor.eval()
    for iteration in range(1, args.max_iterations + 1):
        completed_iterations = iteration
        if args.batch_size <= training_indices.numel():
            order = torch.randperm(training_indices.numel(), generator=batch_generator)[
                :batch_size
            ]
            batch_indices = training_indices[order]
        else:
            sampled = torch.randint(
                0,
                training_indices.numel(),
                (args.batch_size,),
                generator=batch_generator,
            )
            batch_indices = training_indices[sampled]
        for start in range(0, batch_indices.numel(), mini_batch_size):
            indices = batch_indices[start : start + mini_batch_size]
            sample = _batch(dataset, indices, device)
            teacher_distribution = Independent(
                Normal(sample["teacher_action_mean"], sample["teacher_action_std"]), 1
            )
            student_distribution, z_hat = student.forward_with_context(
                sample["current"], sample["history"]
            )
            loss, metrics = criterion(
                teacher_distribution,
                student_distribution,
                z_hat,
                sample["z_star"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.context_encoder.parameters(), args.max_grad_norm
            )
            optimizer.step()
            last_metrics = {name: value.detach() for name, value in metrics.items()}
        if args.log_interval > 0 and iteration % args.log_interval == 0:
            print(
                f"iter={iteration} loss={float(last_metrics['loss']):.6f} "
                f"action_kl={float(last_metrics['action_kl']):.6f}"
            )
        if (
            iteration % args.validation_interval == 0
            or iteration == args.max_iterations
        ):
            validation_kl = _validation_action_kl(
                student,
                dataset,
                validation_indices,
                device,
                args.mini_batch_size,
            )
            improved = validation_kl < best_validation_kl
            if improved:
                best_validation_kl = validation_kl
                best_iteration = iteration
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in student.state_dict().items()
                }
            validation_history.append(
                {
                    "iteration": iteration,
                    "validation_action_kl": validation_kl,
                    "improved": improved,
                }
            )

    if best_state is None:
        raise RuntimeError("S1 validation did not produce a best state")
    student.load_state_dict(best_state, strict=True)

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STAGE_KEY: "s1_context_distillation",
        CHECKPOINT_CURRICULUM_ITERATION_KEY: teacher_curriculum_iteration,
        "model_state_dict": best_state,
        TRAINING_CONFIGURATION_CHECKPOINT_KEY: training_configuration,
        "rollout_schema": {
            "required_tensors": list(REQUIRED_TENSORS),
            "num_samples": int(num_samples),
            "training_samples": int(training_indices.numel()),
            "validation_samples": int(validation_indices.numel()),
        },
        "training": {
            "max_iterations": args.max_iterations,
            "completed_iterations": completed_iterations,
            "validation_interval": args.validation_interval,
            "validation_history": validation_history,
            "batch_size": int(batch_indices.numel()),
            "mini_batch_size": mini_batch_size,
            "context_lr": args.context_lr,
            "max_grad_norm": args.max_grad_norm,
            "training_seed": args.training_seed,
            "actor_initialized_from_teacher": True,
            "best_iteration": best_iteration,
        },
        "metrics": {
            "best_validation_action_kl": best_validation_kl,
        },
        CHECKPOINT_LINEAGE_KEY: {
            "teacher_checkpoint": os.fspath(teacher_path.resolve()),
            "rollout_manifest": os.fspath((rollout_dir / "manifest.json").resolve()),
            "rollout_shards": [
                os.fspath((rollout_dir / name).resolve())
                for name in rollout_manifest["shards"]
            ],
        },
    }
    save_checkpoint_atomic(checkpoint, output, metadata=metadata)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        required=True,
        help="S0 teacher checkpoint with provenance metadata.",
    )
    parser.add_argument("--task", default="Mjlab-G1-Rickshaw-Directional-Slope-Teacher")
    parser.add_argument(
        "--rollout-dir",
        required=False,
        help="Directory of on-policy teacher rollouts containing policy/history/teacher outputs.",
    )
    parser.add_argument(
        "--output", default="logs/rsl_rl/g1_rickshaw_context/s1_context.pt"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=GUIDE_MAX_ITERATIONS["s1_context_distillation"],
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--mini-batch-size", type=int, default=8192)
    parser.add_argument("--context-lr", type=float, default=3.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=S1_GUIDE_PARAMETERS["validation_interval"],
    )
    parser.add_argument("--recollect-rollouts", action="store_true")
    args = parser.parse_args()
    validate_s1_arguments(args)

    teacher_path = require_existing_file(args.teacher, "teacher checkpoint").resolve()
    if args.rollout_dir is None:
        output_parent = Path(args.output).resolve().parent
        args.rollout_dir = os.fspath(
            output_parent / f"teacher_rollouts_{teacher_path.stem}"
        )
        manifest = Path(args.rollout_dir) / "manifest.json"
        if args.recollect_rollouts or not manifest.is_file():
            collector = (
                Path(__file__).resolve().with_name("collect_teacher_rollouts.py")
            )
            command = [
                sys.executable,
                os.fspath(collector),
                "--task",
                args.task,
                "--teacher",
                os.fspath(teacher_path),
                "--output-dir",
                args.rollout_dir,
                "--seed",
                str(args.training_seed),
                "--headless",
            ]
            if args.recollect_rollouts:
                command.append("--overwrite")
            subprocess.run(command, check=True)
    output = train(args)
    print(f"saved S1 context checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
