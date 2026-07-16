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

from _isaaclab_wrappers import add_project_source_to_path, require_existing_file

add_project_source_to_path()

import torch  # noqa: E402
import numpy as np  # noqa: E402
from torch.distributions import Independent, Normal  # noqa: E402

from g1_rickshaw_lab.provenance import (  # noqa: E402
    extract_checkpoint_metadata,
    save_checkpoint_atomic,
    sha256_file,
)
from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    validate_s1_candidate_selection_report,
)
from g1_rickshaw_lab.rl import G1RickshawStudentActor, StudentDistillationLoss  # noqa: E402
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_COUNT,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_HASH_HISTORY_KEY,
    CHECKPOINT_LINEAGE_KEY,
    CHECKPOINT_STAGE_KEY,
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_PARAMETERS,
    checkpoint_hash_history,
    extract_gaussian_actor_state,
    load_reward_calibration_report,
    load_stage_checkpoint,
    validate_guide_training_configuration,
    validate_rollout_stage_coverage,
)

from _rollout_audit import (  # noqa: E402
    AUDIT_TENSOR_NAMES,
    FORMAL_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    normalize_audit_tensors,
    validate_rollout_sample_audit,
)
from _training_configuration import (  # noqa: E402
    TRAINING_CONFIGURATION_CHECKPOINT_KEY,
    build_training_configuration,
    finalize_training_configuration,
)


CHECKPOINT_SCHEMA_VERSION = 1
S1_GUIDE_PARAMETERS = GUIDE_TRAINING_PARAMETERS["s1_context_distillation"]
REQUIRED_TENSORS = (
    "current",
    "history",
    "teacher_action_mean",
    "teacher_action_std",
    "z_star",
    "phase_target",
    "frequency_target",
    "contact_target",
    "cart_lag_target",
    "gait_mask",
    "lag_mask",
)
def validate_formal_s1_arguments(args: argparse.Namespace) -> None:
    """Reject every debug switch or guide-critical hyperparameter deviation."""

    deviations: list[str] = []
    exact = {
        "max_iterations": S1_GUIDE_PARAMETERS["max_iterations"],
        "batch_size": S1_GUIDE_PARAMETERS["batch_size"],
        "mini_batch_size": S1_GUIDE_PARAMETERS["mini_batch_size"],
        "context_lr": S1_GUIDE_PARAMETERS["context_learning_rate"],
        "actor_lr": S1_GUIDE_PARAMETERS["actor_learning_rate"],
        "max_grad_norm": S1_GUIDE_PARAMETERS["gradient_clip"],
        "validation_interval": S1_GUIDE_PARAMETERS["validation_interval"],
        "max_validation_candidates": S1_GUIDE_PARAMETERS["validation_candidate_count"],
    }
    for name, expected in exact.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            matches = float(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            deviations.append(f"{name}={actual!r} (required {expected!r})")
    for name in (
        "skip_task_return_evaluation",
        "allow_random_actor_init",
    ):
        if bool(getattr(args, name)):
            deviations.append(f"{name}=true")
    if (
        isinstance(args.training_seed, bool)
        or args.training_seed < 0
        or args.training_seed > 2**32 - 1
    ):
        deviations.append("training_seed must lie in [0, 2**32-1]")
    if not 0.0 < args.validation_fraction < 1.0:
        deviations.append("validation_fraction must lie strictly between 0 and 1")
    if args.eval_episodes_per_slope < 100:
        deviations.append("eval_episodes_per_slope must be at least 100")
    if not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
        deviations.append("eval_seeds must be non-empty and unique")
    if args.eval_num_envs <= 0 or args.eval_num_envs % SLOPE_COUNT != 0:
        deviations.append(
            f"eval_num_envs must be a positive multiple of {SLOPE_COUNT}"
        )
    if (
        args.collect_num_envs <= 0
        or args.collect_num_envs != FORMAL_NUM_ENVS
    ):
        deviations.append(
            f"collect_num_envs must equal the Guide-fixed {FORMAL_NUM_ENVS}"
        )
    if args.collect_num_steps <= 0:
        deviations.append("collect_num_steps must be positive")
    if deviations:
        raise ValueError(
            "non-guide S1 settings cannot produce a formal checkpoint: " + "; ".join(deviations)
        )


def seed_s1_training(seed: int) -> torch.Generator:
    """Seed model initialization and every training-batch draw deterministically."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > 2**32 - 1
    ):
        raise ValueError("S1 training seed must lie in [0, 2**32-1]")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.Generator(device="cpu").manual_seed(seed)


def _torch_load(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"rollout shard must contain a mapping: {path}")
    if value.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"rollout shard has an unsupported schema: {path}")
    rollout = value.get("rollout")
    if not isinstance(rollout, Mapping):
        raise ValueError(f"rollout shard is missing its canonical rollout mapping: {path}")
    return rollout


def _as_tensor(value: Any, name: str, path: Path) -> torch.Tensor:
    if value is None:
        raise KeyError(f"{path} is missing rollout tensor {name!r}")
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() and name not in {"gait_mask", "lag_mask", "contact_target"}:
        tensor = tensor.float()
    return tensor


def _normalize_shard(path: Path) -> dict[str, torch.Tensor]:
    raw = _torch_load(path)
    current = _as_tensor(raw.get("current"), "current", path).float()
    history = _as_tensor(raw.get("history"), "history", path).float()
    teacher_mean = _as_tensor(
        raw.get("teacher_action_mean"), "teacher_action_mean", path
    ).float()
    if current.ndim != 2 or current.shape[-1] != 96:
        raise ValueError(f"current must have shape [N,96], got {tuple(current.shape)}")
    batch_size = current.shape[0]
    normalized = {
        "current": current,
        "history": history,
        "teacher_action_mean": teacher_mean,
        "teacher_action_std": _as_tensor(
            raw.get("teacher_action_std"), "teacher_action_std", path
        ).float(),
        "z_star": _as_tensor(raw.get("z_star"), "z_star", path).float(),
        "phase_target": _as_tensor(
            raw.get("phase_target"), "phase_target", path
        ).float(),
        "frequency_target": _as_tensor(
            raw.get("frequency_target"), "frequency_target", path
        ).float(),
        "contact_target": _as_tensor(
            raw.get("contact_target"), "contact_target", path
        ).float(),
        "cart_lag_target": _as_tensor(
            raw.get("cart_lag_target"), "cart_lag_target", path
        ).float(),
        "gait_mask": _as_tensor(raw.get("gait_mask"), "gait_mask", path).bool(),
        "lag_mask": _as_tensor(raw.get("lag_mask"), "lag_mask", path).bool(),
    }
    expected_shapes = {
        "history": (batch_size, 61, 96),
        "teacher_action_mean": (batch_size, 29),
        "teacher_action_std": (batch_size, 29),
        "z_star": (batch_size, 16),
        "phase_target": (batch_size, 2),
        "frequency_target": (batch_size, 1),
        "contact_target": (batch_size, 2),
        "cart_lag_target": (batch_size, 1),
        "gait_mask": (batch_size, 1),
        "lag_mask": (batch_size, 1),
    }
    for name, shape in expected_shapes.items():
        if tuple(normalized[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(normalized[name].shape)}")
    if torch.any(normalized["teacher_action_std"] <= 0.0):
        raise ValueError("teacher_action_std must be strictly positive")
    normalized.update(normalize_audit_tensors(raw, batch_size=batch_size))
    return normalized


def load_rollout_dataset(
    rollout_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load shards and independently reconstruct their formal sample audit."""

    shards = sorted(
        path for pattern in ("*.pt", "*.pth") for path in rollout_dir.glob(pattern)
    )
    if not shards:
        raise FileNotFoundError(f"no .pt/.pth rollout shards found in {rollout_dir}")
    loaded = [_normalize_shard(path) for path in shards]
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
        raise ValueError(f"verified rollout manifest is missing: {manifest_path}") from exc
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
    if manifest.get("teacher_checkpoint_sha256") != sha256_file(teacher_path):
        raise ValueError("rollout manifest was generated by a different teacher checkpoint")
    if manifest.get("teacher_provenance") != teacher_metadata.to_mapping():
        raise ValueError("rollout manifest teacher provenance differs from the S0 checkpoint")
    hashes = manifest.get("shards_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("rollout manifest has no content-addressed shards")
    actual_names = {path.name for pattern in ("*.pt", "*.pth") for path in rollout_dir.glob(pattern)}
    if actual_names != set(hashes):
        raise ValueError("rollout directory files differ from the manifest shard set")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or sha256_file(rollout_dir / name) != digest:
            raise ValueError(f"rollout shard hash mismatch: {name}")
    validate_rollout_stage_coverage(manifest)
    return manifest


def initialize_actor_from_checkpoint(student: G1RickshawStudentActor, checkpoint: Mapping[str, Any]) -> bool:
    try:
        candidate = extract_gaussian_actor_state(checkpoint)
    except ValueError:
        return False
    if set(candidate) != set(student.actor.state_dict()):
        return False
    student.actor.load_state_dict(candidate, strict=True)
    return True


def _batch(dataset: Mapping[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: dataset[name][indices].to(device=device, non_blocking=True)
        for name in REQUIRED_TENSORS
    }


def _validation_action_kl(
    student: G1RickshawStudentActor,
    dataset: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
    mini_batch_size: int,
) -> float:
    total = 0.0
    count = 0
    student.eval()
    with torch.no_grad():
        for start in range(0, indices.numel(), mini_batch_size):
            sample = _batch(dataset, indices[start : start + mini_batch_size], device)
            teacher = Independent(Normal(sample["teacher_action_mean"], sample["teacher_action_std"]), 1)
            student_distribution = student(sample["current"], sample["history"])
            divergence = torch.distributions.kl_divergence(teacher, student_distribution)
            total += float(divergence.sum().cpu())
            count += divergence.numel()
    student.train()
    return total / max(count, 1)


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
    if teacher_training_configuration["task"] != args.task:
        raise ValueError("S1 task differs from the S0 teacher training task")
    teacher_ablation_values = teacher_training_configuration["ablation_values"]
    metadata = extract_checkpoint_metadata(teacher_checkpoint)
    reward_calibration_path = require_existing_file(
        args.reward_calibration_report,
        "passed reward calibration report",
    ).resolve()
    reward_calibration_binding = load_reward_calibration_report(
        reward_calibration_path,
        teacher_checkpoint_path=teacher_path,
    )
    teacher_curriculum_iteration = teacher_checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    if isinstance(teacher_curriculum_iteration, bool) or not isinstance(teacher_curriculum_iteration, int):
        raise RuntimeError("S0 teacher checkpoint is missing its audited curriculum iteration")
    teacher_lineage = teacher_checkpoint.get(CHECKPOINT_LINEAGE_KEY)
    if not isinstance(teacher_lineage, Mapping):
        raise RuntimeError("S1 teacher checkpoint lineage must be a mapping")
    teacher_checkpoint_hashes = checkpoint_hash_history(
        teacher_checkpoint,
        checkpoint_path=teacher_path,
    )
    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.is_dir():
        raise FileNotFoundError(f"rollout directory does not exist: {rollout_dir}")
    rollout_manifest = validate_rollout_manifest(
        rollout_dir,
        teacher_path,
        metadata,
    )
    if rollout_manifest.get("teacher_training_configuration") != teacher_training_configuration:
        raise ValueError(
            "rollout manifest training configuration differs from the S0 teacher"
        )
    dataset, stage_coverage = load_rollout_dataset(rollout_dir, rollout_manifest)
    if args.device == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    student = G1RickshawStudentActor(latent_dim=args.latent_dim).to(device)
    actor_initialized = initialize_actor_from_checkpoint(student, teacher_checkpoint)
    if not actor_initialized:
        raise RuntimeError(
            "teacher checkpoint does not expose an exact Gaussian actor state_dict. "
            "Formal S1 requires exact actor initialization from the S0 teacher."
        )

    optimizer = torch.optim.Adam(
        (
            {
                "params": [
                    *student.context_encoder.parameters(),
                    *student.context_projection.parameters(),
                ],
                "lr": args.context_lr,
            },
            {"params": student.actor.parameters(), "lr": args.actor_lr},
        )
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
        formal=True,
        task=args.task,
        num_envs=None,
        seed=args.training_seed,
        max_iterations=args.max_iterations,
        argv=args.argv,
        hydra_overrides=(),
        guide_parameters={**S1_GUIDE_PARAMETERS, "latent_dim": args.latent_dim},
        resolved_parameters={
            "optimizer": "adam",
            "context_learning_rate": args.context_lr,
            "actor_learning_rate": args.actor_lr,
            "batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "gradient_clip": args.max_grad_norm,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "validation_interval": args.validation_interval,
            "max_validation_candidates": args.max_validation_candidates,
            "evaluation_num_envs": args.eval_num_envs,
            "evaluation_episodes_per_slope": args.eval_episodes_per_slope,
            "evaluation_seeds": list(args.eval_seeds),
            "teacher_rollout_num_envs": int(rollout_manifest["num_envs"]),
            "teacher_rollout_steps_per_stage": int(
                rollout_manifest["num_steps_per_stage"]
            ),
            "teacher_rollout_samples": int(rollout_manifest["num_samples"]),
        },
        actor_initialized_from_teacher=True,
        stage_coverage=stage_coverage,
        latent_dim=args.latent_dim,
        rollout_steps=int(teacher_ablation_values["rollout_steps"]),
        fat2_weight=float(teacher_ablation_values["fat2_weight"]),
        inputs_sha256={
            "teacher_checkpoint": sha256_file(teacher_path),
            "reward_calibration_report": sha256_file(reward_calibration_path),
            "rollout_manifest": sha256_file(rollout_dir / "manifest.json"),
            **{
                f"rollout_shard:{name}": str(digest)
                for name, digest in rollout_manifest["shards_sha256"].items()
            },
        },
    )
    validate_guide_training_configuration(
        training_configuration,
        expected_stage="s1_context_distillation",
    )
    last_metrics: dict[str, float] = {}
    candidates: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    student.train()
    for iteration in range(1, args.max_iterations + 1):
        if args.batch_size <= training_indices.numel():
            order = torch.randperm(
                training_indices.numel(), generator=batch_generator
            )[:batch_size]
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
            student_distribution, z_hat, auxiliary = student.forward_with_context(
                sample["current"], sample["history"]
            )
            loss, metrics = criterion(
                teacher_distribution,
                student_distribution,
                z_hat,
                sample["z_star"],
                auxiliary,
                sample["phase_target"],
                sample["frequency_target"],
                sample["contact_target"],
                sample["cart_lag_target"],
                sample["gait_mask"],
                sample["lag_mask"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            last_metrics = {
                name: float(value.detach().cpu()) for name, value in metrics.items()
            }
        if args.log_interval > 0 and iteration % args.log_interval == 0:
            print(
                f"iter={iteration} loss={last_metrics['loss']:.6f} "
                f"action_kl={last_metrics['action_kl']:.6f}"
            )
        if iteration % args.validation_interval == 0 or iteration == args.max_iterations:
            validation_kl = _validation_action_kl(
                student,
                dataset,
                validation_indices,
                device,
                args.mini_batch_size,
            )
            state = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
            candidates.append((validation_kl, iteration, state))
            candidates.sort(key=lambda item: item[0])
            del candidates[args.max_validation_candidates :]

    output = Path(args.output)
    selection_report: Mapping[str, Any] | None = None
    selection_report_digest: str | None = None
    selected_candidate_digest: str | None = None
    if not args.skip_task_return_evaluation:
        candidate_dir = output.resolve().parent / "s1_candidates"
        candidate_paths: list[Path] = []
        candidate_hashes: dict[int, str] = {}
        base_lineage = {
            "teacher_checkpoint_sha256": sha256_file(teacher_path),
            "rollout_manifest_sha256": sha256_file(rollout_dir / "manifest.json"),
            "rollout_shards_sha256": dict(rollout_manifest["shards_sha256"]),
            **reward_calibration_binding,
        }
        for validation_kl, iteration, state in candidates:
            path = candidate_dir / f"candidate_{iteration:05d}.pt"
            candidate_training_configuration = finalize_training_configuration(
                {
                    **training_configuration,
                    "stage": "s1_context_candidate",
                }
            )
            save_checkpoint_atomic(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    CHECKPOINT_STAGE_KEY: "s1_context_candidate",
                    CHECKPOINT_CURRICULUM_ITERATION_KEY: teacher_curriculum_iteration,
                    CHECKPOINT_HASH_HISTORY_KEY: {
                        str(iteration): digest for iteration, digest in teacher_checkpoint_hashes.items()
                    },
                    "candidate_iteration": iteration,
                    "validation_action_kl": validation_kl,
                    "model_state_dict": state,
                    TRAINING_CONFIGURATION_CHECKPOINT_KEY: candidate_training_configuration,
                    CHECKPOINT_LINEAGE_KEY: base_lineage,
                },
                path,
                metadata=metadata,
            )
            candidate_paths.append(path)
            candidate_hashes[iteration] = sha256_file(path)
        report_path = output.resolve().parent / "s1_selection_report.json"
        evaluator = Path(__file__).resolve().with_name("evaluate_context_candidates.py")
        command = [
            sys.executable,
            os.fspath(evaluator),
            "--task",
            args.task,
            "--candidates",
            *(os.fspath(path) for path in candidate_paths),
            "--output",
            os.fspath(report_path),
            "--num-envs",
            str(args.eval_num_envs),
            "--episodes-per-slope",
            str(args.eval_episodes_per_slope),
            "--seeds",
            *(str(seed) for seed in args.eval_seeds),
            "--headless",
        ]
        if args.device != "auto":
            command.extend(("--device", args.device))
        subprocess.run(command, check=True)
        selection_report = json.loads(report_path.read_text(encoding="utf-8"))
        results = validate_s1_candidate_selection_report(
            selection_report,
            expected_candidate_sha256=candidate_hashes,
            fixed_seeds=args.eval_seeds,
            episodes_per_slope=args.eval_episodes_per_slope,
        )
        selection_report_digest = sha256_file(report_path)
        kl_order = {
            int(item["iteration"]): rank
            for rank, item in enumerate(sorted(results, key=lambda item: item["validation_action_kl"]))
        }
        return_order = {
            int(item["iteration"]): rank
            for rank, item in enumerate(sorted(results, key=lambda item: item["task_return_mean"], reverse=True))
        }
        selected_result = min(
            results,
            key=lambda item: (
                kl_order[int(item["iteration"])] + return_order[int(item["iteration"])],
                -float(item["task_return_mean"]),
                float(item["validation_action_kl"]),
            ),
        )
        selected_iteration = int(selected_result["iteration"])
        selected_candidate_digest = candidate_hashes[selected_iteration]
        best_kl, best_iteration, best_state = next(
            candidate for candidate in candidates if candidate[1] == selected_iteration
        )
        selected_task_return = float(selected_result["task_return_mean"])
    else:
        best_kl, best_iteration, best_state = candidates[0]
        selected_task_return = None
    student.load_state_dict(best_state, strict=True)

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STAGE_KEY: "s1_context_distillation",
        CHECKPOINT_CURRICULUM_ITERATION_KEY: teacher_curriculum_iteration,
        CHECKPOINT_HASH_HISTORY_KEY: {
            str(iteration): digest for iteration, digest in teacher_checkpoint_hashes.items()
        },
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        TRAINING_CONFIGURATION_CHECKPOINT_KEY: training_configuration,
        "rollout_schema": {
            "required_tensors": list(REQUIRED_TENSORS),
            "num_samples": int(num_samples),
            "training_samples": int(training_indices.numel()),
            "validation_samples": int(validation_indices.numel()),
        },
        "training": {
            "max_iterations": args.max_iterations,
            "batch_size": int(batch_indices.numel()),
            "mini_batch_size": mini_batch_size,
            "context_lr": args.context_lr,
            "actor_lr": args.actor_lr,
            "max_grad_norm": args.max_grad_norm,
            "training_seed": args.training_seed,
            "actor_initialized_from_teacher": actor_initialized,
            "selected_iteration": best_iteration,
        },
        "metrics": {
            **last_metrics,
            "validation_action_kl": best_kl,
            "validation_task_return": selected_task_return,
        },
        "model_selection": {
            "criteria": ["fixed_validation_action_kl", "fixed_seed_task_return"],
            "rank_rule": "minimum equal-weight rank sum; task return then action KL tie-break",
            "task_return_evaluation_skipped": args.skip_task_return_evaluation,
            "report": None if selection_report is None else selection_report,
            "report_path": (
                None if selection_report is None else os.fspath(report_path.resolve())
            ),
            "report_sha256": selection_report_digest,
            "selected_candidate_checkpoint_sha256": selected_candidate_digest,
        },
        "reward_calibration": {
            "path": os.fspath(reward_calibration_path),
            **reward_calibration_binding,
        },
        CHECKPOINT_LINEAGE_KEY: {
            "teacher_checkpoint_sha256": sha256_file(teacher_path),
            "rollout_manifest_sha256": sha256_file(rollout_dir / "manifest.json"),
            "rollout_shards_sha256": dict(rollout_manifest["shards_sha256"]),
            **reward_calibration_binding,
            **(
                {}
                if selection_report_digest is None or selected_candidate_digest is None
                else {
                    "s1_selection_report_sha256": selection_report_digest,
                    "selected_candidate_checkpoint_sha256": selected_candidate_digest,
                }
            ),
        },
    }
    save_checkpoint_atomic(checkpoint, output, metadata=metadata)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="S0 teacher checkpoint with provenance metadata.")
    parser.add_argument(
        "--reward-calibration-report",
        required=True,
        help="Passed content-addressed guide section 11.2 reward calibration report.",
    )
    parser.add_argument("--task", default="Isaac-G1-Rickshaw-Directional-Slope-v0")
    parser.add_argument(
        "--rollout-dir",
        required=False,
        help="Directory of on-policy teacher rollouts containing policy/history/extrinsics/auxiliary labels.",
    )
    parser.add_argument("--output", default="logs/rsl_rl/g1_rickshaw_context/s1_context.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, choices=(8, 16, 24), default=16)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=GUIDE_MAX_ITERATIONS["s1_context_distillation"],
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--mini-batch-size", type=int, default=8192)
    parser.add_argument("--context-lr", type=float, default=3.0e-4)
    parser.add_argument("--actor-lr", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=S1_GUIDE_PARAMETERS["validation_interval"],
    )
    parser.add_argument("--max-validation-candidates", type=int, default=40)
    parser.add_argument(
        "--eval-num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS
    )
    parser.add_argument("--eval-episodes-per-slope", type=int, default=100)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=(42, 43, 44, 45, 46),
    )
    parser.add_argument(
        "--skip-task-return-evaluation",
        action="store_true",
        help="Debug only: select on action KL without the guide-required fixed-seed task return.",
    )
    parser.add_argument("--collect-num-envs", type=int, default=FORMAL_NUM_ENVS)
    parser.add_argument("--collect-num-steps", type=int, default=64)
    parser.add_argument("--recollect-rollouts", action="store_true")
    parser.add_argument(
        "--allow-random-actor-init",
        action="store_true",
        help="Debug only: skip strict S0 teacher actor initialization.",
    )
    args = parser.parse_args()
    args.argv = list(sys.argv[1:])
    validate_formal_s1_arguments(args)

    teacher_path = require_existing_file(args.teacher, "teacher checkpoint").resolve()
    reward_calibration_path = require_existing_file(
        args.reward_calibration_report,
        "passed reward calibration report",
    ).resolve()
    load_reward_calibration_report(
        reward_calibration_path,
        teacher_checkpoint_path=teacher_path,
    )

    if args.rollout_dir is None:
        output_parent = Path(args.output).resolve().parent
        args.rollout_dir = os.fspath(output_parent / f"teacher_rollouts_{sha256_file(teacher_path)[:12]}")
        manifest = Path(args.rollout_dir) / "manifest.json"
        if args.recollect_rollouts or not manifest.is_file():
            collector = Path(__file__).resolve().with_name("collect_teacher_rollouts.py")
            command = [
                sys.executable,
                os.fspath(collector),
                "--task",
                args.task,
                "--teacher",
                os.fspath(teacher_path),
                "--output-dir",
                args.rollout_dir,
                "--num-envs",
                str(args.collect_num_envs),
                "--num-steps",
                str(args.collect_num_steps),
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
