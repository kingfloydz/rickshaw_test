"""Training-stage checkpoint ABI and deployment bundle helpers.

This module intentionally has no Isaac Lab imports.  The command wrappers use
it before Kit starts, while the runner hook imports RSL-RL only after the
upstream training script has initialized its Python environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import importlib.metadata
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any

from .slope_contract import (
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_GRADIENTS,
    SLOPE_LABELS,
    balanced_slope_counts,
)

import torch
import yaml

from .configuration import FIXED_G1_JOINT_ORDER
from .provenance import (
    RSL_RL_COMMIT,
    RSL_RL_VERSION,
    CheckpointMetadata,
    attach_checkpoint_metadata,
    atomic_torch_save,
    collect_checkpoint_metadata,
    extract_checkpoint_metadata,
    load_checkpoint_with_validation,
    sha256_file,
    validate_checkpoint,
)


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STAGE_KEY = "g1_rickshaw_stage"
CHECKPOINT_LINEAGE_KEY = "g1_rickshaw_lineage"
CHECKPOINT_CURRICULUM_ITERATION_KEY = "g1_rickshaw_curriculum_iteration"
CHECKPOINT_HASH_HISTORY_KEY = "g1_rickshaw_checkpoint_hashes"
S0_VALIDATION_STATE_KEY = "g1_rickshaw_s0_validation"
S2_VALIDATION_STATE_KEY = "g1_rickshaw_s2_validation"
TRAINING_CONFIGURATION_KEY = "g1_rickshaw_training_configuration"
TRAINING_THROUGHPUT_KEY = "g1_rickshaw_training_throughput"
TRAINING_CONFIGURATION_SCHEMA_VERSION = 3
EXPECTED_RSL_RL_DISTRIBUTION_VERSION = RSL_RL_VERSION.removeprefix("v")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = REPOSITORY_ROOT.parent / "IsaacLab"
DEFAULT_FEASIBILITY_PATH = REPOSITORY_ROOT / "config" / "feasibility_envelope.yaml"
DEFAULT_RESET_POSES_PATH = REPOSITORY_ROOT / "config" / "reset_poses.yaml"
GUIDE_TRAINING_TASK = "Isaac-G1-Rickshaw-Directional-Slope-v0"
GUIDE_TRAINING_NUM_ENVS = 4096
TRAINING_ARTIFACT_INTERVAL = 200

SIGNED_SLOPE_LABELS = SLOPE_LABELS
ROLLOUT_MANIFEST_SCHEMA_VERSION = 4
ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION = 3
ROLLOUT_FORMAL_NUM_ENVS = GUIDE_TRAINING_NUM_ENVS
ROLLOUT_STAGE_SEQUENCE = ("TRAINING",)
ROLLOUT_PHYSICS_PARAMETER_NAMES = (
    "payload.mass",
    "payload.com.x",
    "payload.com.y",
    "payload.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
    "d6.linear_stiffness",
    "d6.linear_damping",
    "d6.angular_stiffness",
    "d6.angular_damping",
    "d6.max_force",
    "d6.max_torque",
    "d6.linear_limit",
    "d6.angular_limit",
    "motor.strength",
    "control.delay",
    "observation.delay",
    "joint_model_error",
)
ABLATION_VALUE_KEYS = (
    "fat2_weight",
    "rollout_steps",
    "latent_dim",
)
ABLATION_VALUE_OPTIONS = {
    "fat2_weight": (0.0, 0.1, 0.2),
    "rollout_steps": (24, 48, 64),
    "latent_dim": (8, 16, 24, 32),
}
TRAINING_CONFIGURATION_FIELDS = {
    "schema_version",
    "stage",
    "formal",
    "task",
    "num_envs",
    "seed",
    "max_iterations",
    "argv",
    "hydra_overrides",
    "guide_parameters",
    "resolved_parameters",
    "actor_initialized_from_teacher",
    "stage_coverage",
    "ablation_values",
    "inputs_sha256",
    "content_sha256",
}
GUIDE_TRAINING_PARAMETERS = {
    "s0_teacher": {
        "validation_interval": TRAINING_ARTIFACT_INTERVAL,
        "validation_patience": 5,
        "validation_episodes_per_slope": 100,
    },
    "s1_context_distillation": {
        "context_learning_rate": 3.0e-4,
        "actor_learning_rate": 1.0e-4,
        "batch_size": 65536,
        "mini_batch_size": 8192,
        "gradient_clip": 1.0,
        "latent_dim": 16,
        "actor_initialized_from_teacher": True,
        "teacher_actor_initialization": True,
        "rollout_stage_sequence": list(ROLLOUT_STAGE_SEQUENCE),
        "validation_interval": TRAINING_ARTIFACT_INTERVAL,
        "validation_patience": 5,
        "validation_candidate_count": 40,
        "model_selection": [
            "fixed_validation_action_kl",
            "fixed_seed_task_return",
        ],
    },
    "s2_student_ppo": {
        "context_learning_rate": 1.0e-4,
        "actor_learning_rate": 3.0e-4,
        "critic_learning_rate": 3.0e-4,
        "context_encoder_frozen": False,
        "distillation_loss": False,
        "validation_interval": TRAINING_ARTIFACT_INTERVAL,
        "validation_patience": 5,
        "validation_episodes_per_slope": 100,
    },
}
GUIDE_MAX_ITERATIONS = {
    "s0_teacher": 6000,
    "s1_context_distillation": 4000,
    "s2_student_ppo": 2000,
}


def _canonical_training_configuration_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def training_configuration_sha256(value: Mapping[str, Any]) -> str:
    """Hash a training configuration without its self-describing digest."""

    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_training_configuration_json(payload)).hexdigest()


def finalize_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and content-address a JSON-only training configuration."""

    payload = dict(value)
    payload.pop("content_sha256", None)
    normalized = json.loads(_canonical_training_configuration_json(payload).decode("ascii"))
    normalized["content_sha256"] = training_configuration_sha256(normalized)
    return normalized


def validate_training_configuration(
    value: Any,
    *,
    expected_stage: str | None = None,
    require_formal: bool = True,
) -> dict[str, Any]:
    """Validate the replayable CLI/Hydra and ablation configuration in a checkpoint."""

    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != TRAINING_CONFIGURATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "training configuration requires "
            f"schema_version: {TRAINING_CONFIGURATION_SCHEMA_VERSION}"
        )
    if set(value) != TRAINING_CONFIGURATION_FIELDS:
        raise ValueError("training configuration has missing or unknown fields")
    claimed_digest = value.get("content_sha256")
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:
        raise ValueError("training configuration content_sha256 is missing or malformed")
    try:
        int(claimed_digest, 16)
    except ValueError as exc:
        raise ValueError("training configuration content_sha256 is malformed") from exc
    if training_configuration_sha256(value) != claimed_digest.lower():
        raise ValueError("training configuration content_sha256 mismatch")
    stage = value.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("training configuration is missing its stage")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(
            f"training configuration stage {stage!r} differs from {expected_stage!r}"
        )
    seed = value.get("seed")
    iterations = value.get("max_iterations")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training configuration seed must be a non-negative integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("training configuration max_iterations must be positive")
    if type(value.get("formal")) is not bool:
        raise ValueError("training configuration formal must be boolean")
    if require_formal and value["formal"] is not True:
        raise ValueError("formal checkpoint lineage requires formal=true")
    if not isinstance(value.get("task"), str) or not value["task"]:
        raise ValueError("training configuration task must be non-empty")
    num_envs = value.get("num_envs")
    if num_envs is not None and (
        isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0
    ):
        raise ValueError("training configuration num_envs must be a positive integer or null")
    for name in ("argv", "hydra_overrides"):
        sequence = value.get(name)
        if not isinstance(sequence, list) or any(not isinstance(item, str) for item in sequence):
            raise ValueError(f"training configuration {name} must be a string list")
    if not isinstance(value.get("guide_parameters"), Mapping):
        raise ValueError("training configuration requires guide_parameters")
    if not isinstance(value.get("resolved_parameters"), Mapping):
        raise ValueError("training configuration requires resolved_parameters")
    actor_initialized = value.get("actor_initialized_from_teacher")
    if actor_initialized is not None and type(actor_initialized) is not bool:
        raise ValueError("actor_initialized_from_teacher must be boolean or null")
    stage_coverage = value.get("stage_coverage")
    if stage_coverage is not None and not isinstance(stage_coverage, Mapping):
        raise ValueError("training configuration stage_coverage must be a mapping or null")
    inputs = value.get("inputs_sha256")
    if (
        not isinstance(inputs, Mapping)
        or not inputs
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            for name, digest in inputs.items()
        )
    ):
        raise ValueError("training configuration inputs_sha256 is malformed")
    try:
        for digest in inputs.values():
            int(digest, 16)
    except ValueError as exc:
        raise ValueError("training configuration inputs_sha256 is malformed") from exc
    ablation_values = value.get("ablation_values")
    if not isinstance(ablation_values, Mapping) or set(ablation_values) != set(
        ABLATION_VALUE_KEYS
    ):
        raise ValueError(
            f"training configuration ablation_values must contain exactly {ABLATION_VALUE_KEYS}"
        )
    normalized_ablation = dict(ablation_values)
    normalized_ablation["fat2_weight"] = float(normalized_ablation["fat2_weight"])
    normalized_ablation["rollout_steps"] = int(normalized_ablation["rollout_steps"])
    normalized_ablation["latent_dim"] = int(normalized_ablation["latent_dim"])
    for name, supported in ABLATION_VALUE_OPTIONS.items():
        if normalized_ablation[name] not in supported:
            raise ValueError(
                f"unsupported training ablation value {name}={normalized_ablation[name]!r}"
            )
    result = dict(value)
    result["ablation_values"] = normalized_ablation
    return finalize_training_configuration(result)


def validate_guide_training_configuration(
    value: Any,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    """Reject debug or hyperparameter-divergent configurations from formal lineage."""

    result = validate_training_configuration(value, expected_stage=expected_stage)
    guide_stage = (
        "s1_context_distillation"
        if expected_stage == "s1_context_candidate"
        else expected_stage
    )
    if guide_stage not in GUIDE_TRAINING_PARAMETERS:
        raise ValueError(f"no Guide training contract exists for stage {expected_stage!r}")
    if result["max_iterations"] != GUIDE_MAX_ITERATIONS[guide_stage]:
        raise ValueError(
            f"{expected_stage} requires max_iterations={GUIDE_MAX_ITERATIONS[guide_stage]}"
        )
    if result["task"] != GUIDE_TRAINING_TASK:
        raise ValueError(
            f"formal {guide_stage} requires task={GUIDE_TRAINING_TASK!r}"
        )
    if guide_stage in {"s0_teacher", "s2_student_ppo"}:
        if result["num_envs"] != GUIDE_TRAINING_NUM_ENVS:
            raise ValueError(
                f"formal {guide_stage} requires num_envs={GUIDE_TRAINING_NUM_ENVS}"
            )
    parameters = result["guide_parameters"]
    for name, expected in GUIDE_TRAINING_PARAMETERS[guide_stage].items():
        if guide_stage == "s1_context_distillation" and name == "latent_dim":
            expected = result["ablation_values"]["latent_dim"]
        if parameters.get(name) != expected:
            raise ValueError(
                f"{expected_stage} guide parameter {name!r} differs from {expected!r}"
            )
    forbidden = {
        "--allow-random-actor-init",
        "--skip-task-return-eval",
    }
    used_forbidden = sorted(forbidden.intersection(result["argv"]))
    if used_forbidden:
        raise ValueError(
            f"formal {expected_stage} configuration contains debug flags: {used_forbidden}"
        )
    if result["hydra_overrides"]:
        raise ValueError(
            f"formal {expected_stage} configuration cannot contain unverified Hydra overrides"
        )
    if guide_stage in {"s1_context_distillation", "s2_student_ppo"}:
        if result["actor_initialized_from_teacher"] is not True:
            raise ValueError(
                f"formal {guide_stage} requires exact actor initialization from the teacher"
            )
        coverage = result["stage_coverage"]
        if (
            not isinstance(coverage, Mapping)
            or coverage.get("manifest_schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION
            or coverage.get("sample_audit_schema_version")
            != ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION
            or not isinstance(coverage.get("sample_audit_sha256"), str)
            or coverage.get("signed_slopes")
            != list(SLOPE_GRADIENTS)
            or set(coverage.get("stages", {})) != set(ROLLOUT_STAGE_SEQUENCE)
        ):
            raise ValueError(
                f"formal {guide_stage} requires validated sample-level TRAINING coverage"
            )
        for stage_name, stage_coverage in coverage["stages"].items():
            if not isinstance(stage_coverage, Mapping):
                raise ValueError(f"formal {guide_stage} has invalid {stage_name} rollout coverage")
            slope_environments = stage_coverage.get("slope_environment_distribution")
            expected_environments = {
                label: count
                for label, count in zip(
                    SIGNED_SLOPE_LABELS,
                    balanced_slope_counts(ROLLOUT_FORMAL_NUM_ENVS),
                    strict=True,
                )
            }
            if slope_environments != expected_environments:
                raise ValueError(
                    f"formal {guide_stage} {stage_name} lacks the balanced slope allocation"
                )
            samples = stage_coverage.get("samples")
            if (
                isinstance(samples, bool)
                or not isinstance(samples, int)
                or samples <= 0
                or samples % ROLLOUT_FORMAL_NUM_ENVS != 0
            ):
                raise ValueError(
                    f"formal {guide_stage} {stage_name} has an invalid rollout sample count"
                )
            samples_per_environment = samples // ROLLOUT_FORMAL_NUM_ENVS
            expected_samples = {
                label: count * samples_per_environment
                for label, count in expected_environments.items()
            }
            if stage_coverage.get("slope_sample_distribution") != expected_samples:
                raise ValueError(
                    f"formal {guide_stage} {stage_name} lacks the exact slope sample quotas"
                )
            episodes = stage_coverage.get("slope_episode_distribution")
            if (
                not isinstance(episodes, Mapping)
                or set(episodes) != set(SIGNED_SLOPE_LABELS)
                or any(
                    isinstance(count, bool) or not isinstance(count, int) or count <= 0
                    for count in episodes.values()
                )
            ):
                raise ValueError(
                    f"formal {guide_stage} {stage_name} lacks "
                    f"{len(SLOPE_GRADIENTS)}-slope episode evidence"
                )
    return result


def validate_training_throughput(value: Any) -> dict[str, float | int]:
    """Validate recomputable wall-clock sample throughput stored by the runner hook."""

    if not isinstance(value, Mapping) or set(value) != {
        "iterations",
        "transitions",
        "wall_time_s",
        "samples_per_second",
        "num_envs",
        "num_steps_per_env",
    }:
        raise ValueError("training throughput evidence has an invalid schema")
    integer_fields = ("iterations", "transitions", "num_envs", "num_steps_per_env")
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], int)
        for name in integer_fields
    ):
        raise ValueError("training throughput count fields must be integers")
    iterations = value["iterations"]
    transitions = value["transitions"]
    num_envs = value["num_envs"]
    num_steps = value["num_steps_per_env"]
    try:
        wall_time_s = float(value["wall_time_s"])
        samples_per_second = float(value["samples_per_second"])
    except (TypeError, ValueError) as exc:
        raise ValueError("training throughput fields must be numeric") from exc
    if (
        iterations <= 0
        or transitions <= 0
        or num_envs <= 0
        or num_steps <= 0
        or wall_time_s <= 0.0
        or samples_per_second <= 0.0
        or not math.isfinite(wall_time_s)
        or not math.isfinite(samples_per_second)
        or not math.isclose(
            samples_per_second,
            transitions / wall_time_s,
            rel_tol=1.0e-9,
            abs_tol=0.0,
        )
        or transitions != iterations * num_envs * num_steps
    ):
        raise ValueError("training throughput evidence is inconsistent")
    return {
        "iterations": iterations,
        "transitions": transitions,
        "wall_time_s": wall_time_s,
        "samples_per_second": samples_per_second,
        "num_envs": num_envs,
        "num_steps_per_env": num_steps,
    }


def s2_remaining_learning_iterations(
    *,
    requested_iterations: int,
    completed_iterations: int,
    early_stopped: bool,
) -> int:
    """Return the additional S2 iterations, treating patience stop as terminal."""

    if type(early_stopped) is not bool:
        raise ValueError("early_stopped must be boolean")
    for name, value in (
        ("requested_iterations", requested_iterations),
        ("completed_iterations", completed_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    target = GUIDE_MAX_ITERATIONS["s2_student_ppo"]
    if requested_iterations != target:
        raise ValueError(f"formal S2 learn must request the {target}-iteration Guide target")
    if completed_iterations > target:
        raise ValueError("S2 checkpoint already exceeds the 2000-iteration Guide target")
    if early_stopped:
        return 0
    return target - completed_iterations


def s0_remaining_learning_iterations(
    *,
    requested_iterations: int,
    completed_iterations: int,
    early_stopped: bool,
) -> int:
    """Return the additional S0 iterations, treating patience stop as terminal."""

    if type(early_stopped) is not bool:
        raise ValueError("early_stopped must be boolean")
    for name, value in (
        ("requested_iterations", requested_iterations),
        ("completed_iterations", completed_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    target = GUIDE_MAX_ITERATIONS["s0_teacher"]
    if requested_iterations != target:
        raise ValueError(f"formal S0 learn must request the {target}-iteration Guide target")
    if completed_iterations > target:
        raise ValueError("S0 checkpoint already exceeds the 6000-iteration Guide target")
    if early_stopped:
        return 0
    return target - completed_iterations


def fixed_seed_validation_early_stopped(state: "FixedSeedValidationState") -> bool:
    """Whether a persisted fixed-seed patience state represents a terminal stop."""

    return (
        state.last_evaluated_iteration > state.warmup_iterations
        and state.no_improvement_count >= state.patience
    )


def s0_validation_early_stopped(state: "FixedSeedValidationState") -> bool:
    """Backward-compatible name for the S0 validation terminal-state check."""

    return fixed_seed_validation_early_stopped(state)


def _validation_state_key(stage: str) -> str:
    try:
        return {
            "s0_teacher": S0_VALIDATION_STATE_KEY,
            "s2_student_ppo": S2_VALIDATION_STATE_KEY,
        }[stage]
    except KeyError as exc:
        raise ValueError(f"stage {stage!r} does not use fixed-seed early stopping") from exc


def training_checkpoint_complete(checkpoint: Mapping[str, Any], *, stage: str) -> bool:
    """Return whether a PPO checkpoint reached its cap or terminal patience stop."""

    throughput = validate_training_throughput(checkpoint.get(TRAINING_THROUGHPUT_KEY))
    raw_state = checkpoint.get(_validation_state_key(stage))
    if not isinstance(raw_state, Mapping):
        return False
    state = FixedSeedValidationState.from_mapping(raw_state)
    _validate_fixed_seed_validation_iteration_alignment(
        state,
        int(throughput["iterations"]),
    )
    return (
        throughput["iterations"] == GUIDE_MAX_ITERATIONS[stage]
        or fixed_seed_validation_early_stopped(state)
    )


def validate_s1_training_completion(checkpoint: Mapping[str, Any]) -> None:
    """Require S1 to reach its cap or carry auditable terminal patience state."""

    training = checkpoint.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("S1 checkpoint is missing training completion evidence")
    completed = training.get("completed_iterations")
    target = GUIDE_MAX_ITERATIONS["s1_context_distillation"]
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 < completed <= target:
        raise ValueError(f"S1 completed_iterations must lie in [1, {target}]")
    early_stopped = training.get("early_stopped")
    if type(early_stopped) is not bool:
        raise ValueError("S1 early_stopped must be boolean")
    history = training.get("validation_history")
    if not isinstance(history, list) or not history:
        raise ValueError("S1 checkpoint has no validation history")
    last = history[-1]
    patience = GUIDE_TRAINING_PARAMETERS["s1_context_distillation"][
        "validation_patience"
    ]
    terminal_patience = (
        isinstance(last, Mapping)
        and last.get("iteration") == completed
        and last.get("no_improvement_count") == patience
    )
    if completed < target and (not early_stopped or not terminal_patience):
        raise ValueError("incomplete S1 checkpoint lacks terminal patience evidence")
    if completed == target and early_stopped:
        raise ValueError("S1 checkpoint at the iteration cap cannot be marked early-stopped")


def _validate_fixed_seed_validation_iteration_alignment(
    state: "FixedSeedValidationState",
    completed_iterations: int,
) -> None:
    expected = (completed_iterations // state.interval) * state.interval
    if state.last_evaluated_iteration != expected:
        raise ValueError(
            "fixed-seed validation state is not aligned with completed training iterations: "
            f"last={state.last_evaluated_iteration}, expected={expected}"
        )


def validate_student_checkpoint_architecture(
    checkpoint: Mapping[str, Any],
    training_configuration: Mapping[str, Any],
) -> None:
    """Cross-check the recorded latent ablation against the actual model tensors."""

    latent_dim = int(training_configuration["ablation_values"]["latent_dim"])
    state = next(iter(_state_dicts(checkpoint)), None)
    if not isinstance(state, Mapping):
        raise ValueError("student checkpoint has no model state_dict")

    def tensor(*names: str) -> torch.Tensor | None:
        for name in names:
            value = state.get(name)
            if torch.is_tensor(value):
                return value
        return None

    latent_weight = tensor(
        "context_encoder.context.2.weight",
        "encoder.context.2.weight",
    )
    policy_weight = tensor(
        "actor.network.0.weight",
        "policy.network.0.weight",
    )
    if (
        latent_weight is None
        or latent_weight.ndim != 2
        or latent_weight.shape[0] != latent_dim
    ):
        raise ValueError("checkpoint context encoder differs from its latent ablation value")
    if policy_weight is None or policy_weight.ndim != 2 or policy_weight.shape[1] != 112:
        raise ValueError("student actor must preserve the exact 96+16 teacher interface")
    projection = tensor("context_projection.1.weight")
    legacy_linear_projection = tensor("context_projection.weight")
    if latent_dim == 16:
        if projection is not None or legacy_linear_projection is not None:
            raise ValueError("16-D student must use the identity context projection")
    elif projection is None or tuple(projection.shape) != (16, latent_dim):
        raise ValueError(
            "latent ablation checkpoint requires the nonlinear ELU context projection"
        )


@dataclass
class FixedSeedValidationState:
    """Pure scheduling and patience state for fixed-seed PPO validation."""

    interval: int = TRAINING_ARTIFACT_INTERVAL
    patience: int = 5
    warmup_iterations: int = 0
    minimum_improvement: float = 0.0
    stage: str | None = None
    best_score: float | None = None
    no_improvement_count: int = 0
    last_evaluated_iteration: int = 0
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            self.interval != TRAINING_ARTIFACT_INTERVAL
            or self.patience != 5
            or self.warmup_iterations != 0
        ):
            raise ValueError(
                "fixed-seed validation is required every "
                f"{TRAINING_ARTIFACT_INTERVAL} iterations with patience 5"
            )
        if self.minimum_improvement < 0.0:
            raise ValueError("minimum_improvement cannot be negative")

    def should_evaluate(self, iteration: int) -> bool:
        return (
            iteration > 0
            and iteration % self.interval == 0
            and iteration > self.last_evaluated_iteration
        )

    def record(
        self,
        *,
        iteration: int,
        stage: str,
        score: float,
        report_sha256: str,
    ) -> bool:
        if not self.should_evaluate(iteration):
            raise ValueError(f"iteration {iteration} is not the next scheduled validation")
        if not isinstance(stage, str) or not stage or not math.isfinite(score):
            raise ValueError("fixed-seed validation stage/score is invalid")
        if not isinstance(report_sha256, str) or len(report_sha256) != 64:
            raise ValueError("fixed-seed validation report SHA256 is malformed")
        int(report_sha256, 16)
        if self.stage != stage:
            self.stage = stage
            self.best_score = None
            self.no_improvement_count = 0
        improved = self.best_score is None or score > self.best_score + self.minimum_improvement
        if improved:
            self.best_score = score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        self.last_evaluated_iteration = iteration
        self.evaluations.append(
            {
                "iteration": iteration,
                "stage": stage,
                "score": score,
                "improved": improved,
                "report_sha256": report_sha256.lower(),
            }
        )
        return (
            iteration > self.warmup_iterations
            and self.no_improvement_count >= self.patience
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "patience": self.patience,
            "warmup_iterations": self.warmup_iterations,
            "minimum_improvement": self.minimum_improvement,
            "stage": self.stage,
            "best_score": self.best_score,
            "no_improvement_count": self.no_improvement_count,
            "last_evaluated_iteration": self.last_evaluated_iteration,
            "evaluations": list(self.evaluations),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixedSeedValidationState":
        if not isinstance(value, Mapping):
            raise ValueError("fixed-seed validation checkpoint state must be a mapping")
        state = cls(
            interval=value.get("interval"),
            patience=value.get("patience"),
            warmup_iterations=value.get("warmup_iterations"),
            minimum_improvement=value.get("minimum_improvement"),
            stage=value.get("stage"),
            best_score=value.get("best_score"),
            no_improvement_count=value.get("no_improvement_count"),
            last_evaluated_iteration=value.get("last_evaluated_iteration"),
            evaluations=list(value.get("evaluations", ())),
        )
        if state.best_score is not None and not math.isfinite(float(state.best_score)):
            raise ValueError("fixed-seed validation best_score must be finite")
        if state.no_improvement_count < 0 or state.last_evaluated_iteration < 0:
            raise ValueError("fixed-seed validation counters cannot be negative")
        return state


S0FixedSeedValidationState = FixedSeedValidationState


class _EarlyStopSignal(RuntimeError):
    pass


def load_fixed_seed_validation_score(
    report_path: str | Path,
    *,
    checkpoint_path: str | Path,
    curriculum_stage: str,
    fixed_seeds: Iterable[int],
) -> float:
    """Validate an acceptance report and return its mean fixed-seed episode return."""

    from .policy_evaluation import (
        evaluation_runtime_sources_sha256,
    )

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("status") not in {"recorded", "passed"} or report.get("failures"):
        raise RuntimeError("fixed-seed evaluation report is incomplete or failed")
    inputs = report.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("evaluation_runtime_sources_sha256")
        != evaluation_runtime_sources_sha256()
    ):
        raise RuntimeError("fixed-seed evaluation report is stale for evaluator sources")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("fixed-seed validation report checkpoint SHA256 mismatch")
    evaluation = report.get("evaluation")
    seeds = list(fixed_seeds)
    if not isinstance(evaluation, Mapping) or evaluation.get("fixed_seeds") != seeds:
        raise RuntimeError("fixed-seed validation report fixed seeds mismatch")
    if evaluation.get("curriculum_stages") != [curriculum_stage]:
        raise RuntimeError("fixed-seed validation report curriculum stage mismatch")
    if evaluation.get("episodes_per_slope_per_stage", 0) < 100:
        raise RuntimeError("fixed-seed validation must evaluate at least 100 episodes per slope")
    signed_slopes = evaluation.get("signed_slopes")
    try:
        slope_labels = {f"{float(value):+.2f}" for value in signed_slopes}
    except (TypeError, ValueError):
        slope_labels = set()
    if slope_labels != set(SIGNED_SLOPE_LABELS):
        raise RuntimeError(
            f"fixed-seed validation report must cover all {len(SLOPE_GRADIENTS)} slopes"
        )
    stage_report = report.get("stages", {}).get(curriculum_stage)
    if not isinstance(stage_report, Mapping):
        raise RuntimeError("fixed-seed validation report is missing its stage metrics")
    baseline = stage_report.get("context_interventions", {}).get("baseline_return")
    minimum_episodes = 100 * len(SIGNED_SLOPE_LABELS)
    if not isinstance(baseline, Mapping) or baseline.get("episodes", 0) < minimum_episodes:
        raise RuntimeError("fixed-seed validation report did not complete the per-slope episode quota")
    stage_per_slope = stage_report.get("per_slope")
    if not isinstance(stage_per_slope, Mapping) or set(stage_per_slope) != set(SIGNED_SLOPE_LABELS):
        raise RuntimeError("fixed-seed validation report is missing exact per-slope episode metrics")
    for label, slope_report in stage_per_slope.items():
        episodes = slope_report.get("episodes") if isinstance(slope_report, Mapping) else None
        count = episodes.get("completed") if isinstance(episodes, Mapping) else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 100:
            raise RuntimeError(f"fixed-seed validation slope {label} did not complete 100 episodes")
    per_slope = baseline.get("per_slope_mean")
    if not isinstance(per_slope, Mapping) or set(per_slope) != set(SIGNED_SLOPE_LABELS):
        raise RuntimeError("fixed-seed validation return is missing one or more slopes")
    score = float(baseline.get("mean"))
    if not math.isfinite(score):
        raise RuntimeError("fixed-seed validation mean return is not finite")
    return score


load_s0_fixed_seed_validation_score = load_fixed_seed_validation_score


def validate_rollout_stage_coverage(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Validate the single reset-separated TRAINING rollout segment."""

    if manifest.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "formal rollout manifest requires schema_version "
            f"{ROLLOUT_MANIFEST_SCHEMA_VERSION}"
        )
    segments = manifest.get("stage_segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError("rollout manifest requires exactly one TRAINING segment")
    segment = segments[0]
    if not isinstance(segment, Mapping) or segment.get("global_stage") != "TRAINING":
        raise ValueError("rollout stage sequence must be exactly ('TRAINING',)")
    num_envs = manifest.get("num_envs")
    num_steps = manifest.get("num_steps_per_stage")
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs != ROLLOUT_FORMAL_NUM_ENVS
        or isinstance(num_steps, bool)
        or not isinstance(num_steps, int)
        or num_steps <= 0
    ):
        raise ValueError(
            f"formal rollout requires num_envs={ROLLOUT_FORMAL_NUM_ENVS} and positive steps"
        )
    expected_slopes = list(SLOPE_GRADIENTS)
    if manifest.get("signed_slopes") != expected_slopes:
        raise ValueError(
            f"formal rollout manifest must contain exactly all {len(SLOPE_GRADIENTS)} slopes"
        )
    expected_samples = num_envs * num_steps
    if (
        segment.get("valid_samples") != expected_samples
        or segment.get("target_valid_samples") != expected_samples
        or segment.get("full_environment_reset") is not True
        or segment.get("reset_policy_steps") != 0
    ):
        raise ValueError("TRAINING rollout segment did not meet its reset/sample quota")

    expected_environments = {
        label: count
        for label, count in zip(
            SIGNED_SLOPE_LABELS,
            balanced_slope_counts(num_envs),
            strict=True,
        )
    }
    expected_slope_samples = {
        label: count * num_steps for label, count in expected_environments.items()
    }
    if segment.get("slope_environment_distribution") != expected_environments:
        raise ValueError("TRAINING rollout lacks the balanced slope allocation")
    if segment.get("slope_sample_distribution") != expected_slope_samples:
        raise ValueError("TRAINING rollout lacks the exact slope sample quotas")
    episodes = segment.get("slope_episode_distribution")
    if (
        not isinstance(episodes, Mapping)
        or set(episodes) != set(SIGNED_SLOPE_LABELS)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in episodes.values()
        )
    ):
        raise ValueError(
            f"TRAINING rollout lacks episode evidence for all {len(SLOPE_GRADIENTS)} slopes"
        )

    physics = segment.get("physics_distribution")
    if not isinstance(physics, Mapping) or set(ROLLOUT_PHYSICS_PARAMETER_NAMES) - set(physics):
        raise ValueError("TRAINING rollout lacks its physical-parameter distribution")
    for name in ROLLOUT_PHYSICS_PARAMETER_NAMES:
        summary = physics[name]
        if not isinstance(summary, Mapping) or not {"minimum", "mean", "maximum"}.issubset(summary):
            raise ValueError(f"rollout physical summary {name!r} is incomplete")
        values = tuple(float(summary[key]) for key in ("minimum", "mean", "maximum"))
        if not all(math.isfinite(value) for value in values) or not values[0] <= values[1] <= values[2]:
            raise ValueError(f"rollout physical summary {name!r} is invalid")

    environment_stages = segment.get("per_environment_stage_distribution")
    sample_stages = segment.get("valid_sample_stage_distribution")
    if environment_stages != {"TRAINING": num_envs}:
        raise ValueError("rollout environments must all use TRAINING")
    if sample_stages != {"TRAINING": expected_samples}:
        raise ValueError("rollout samples must all use TRAINING")
    if manifest.get("stage_sample_distribution") != sample_stages:
        raise ValueError("rollout aggregate stage distribution differs from its segment")
    if manifest.get("num_samples") != expected_samples:
        raise ValueError("rollout manifest num_samples differs from its segment")
    for name, expected in (
        ("slope_sample_distribution", expected_slope_samples),
        ("slope_environment_distribution", expected_environments),
        ("slope_episode_distribution", dict(episodes)),
    ):
        if manifest.get(name) != expected:
            raise ValueError(f"rollout aggregate {name} differs from its segment")
    return {"TRAINING": expected_samples}


def feasibility_config_path() -> Path:
    """Resolve and publish the canonical feasibility-envelope path."""

    resolved = Path(
        os.environ.get("G1_RICKSHAW_FEASIBILITY_ENVELOPE", DEFAULT_FEASIBILITY_PATH)
    ).resolve()
    os.environ["G1_RICKSHAW_FEASIBILITY_ENVELOPE"] = os.fspath(resolved)
    return resolved


def runtime_config_files() -> dict[str, Path]:
    """Return the relocatable set of files forming the policy ABI."""

    package = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab" / "g1_rickshaw_lab"
    task = package / "tasks" / "manager_based" / "rickshaw_velocity"
    mdp = task / "mdp"
    scripts = REPOSITORY_ROOT / "scripts"
    return {
        "implementation_guide": REPOSITORY_ROOT / "G1_Rickshaw_IsaacLab_Implementation_Guide.md",
        "feasibility_envelope": feasibility_config_path(),
        "reset_poses": Path(os.environ.get("G1_RICKSHAW_RESET_POSES", DEFAULT_RESET_POSES_PATH)),
        "training_configuration_cli": scripts / "_training_configuration.py",
        "rollout_audit_cli": scripts / "_rollout_audit.py",
        "teacher_training_cli": scripts / "train_teacher.py",
        "context_training_cli": scripts / "train_context.py",
        "teacher_rollout_cli": scripts / "collect_teacher_rollouts.py",
        "student_finetune_cli": scripts / "finetune_student.py",
        "context_candidate_evaluator": scripts / "evaluate_context_candidates.py",
        "policy_acceptance_cli": scripts / "evaluate_policy.py",
        "student_play_cli": scripts / "play_student.py",
        "policy_ablation_cli": scripts / "run_policy_ablations.py",
        "package_configuration": package / "configuration.py",
        "checkpoint_provenance": package / "provenance.py",
        "policy_evaluation_contract": package / "policy_evaluation.py",
        "reward_calibration_contract": package / "reward_calibration.py",
        "validation_contract": package / "validation.py",
        "g1_asset_configuration": package / "assets" / "g1_dex1.py",
        "rickshaw_asset_configuration": package / "assets" / "rickshaw.py",
        "task_registration": task / "__init__.py",
        "environment_cfg": task / "env_cfg.py",
        "terrain_cfg": task / "terrain_cfg.py",
        "runner_cfg": task / "agents" / "rsl_rl_cfg.py",
        "actor_critic": package / "rl" / "actor_critic.py",
        "context_encoder": package / "rl" / "context_encoder.py",
        "teacher_model": package / "rl" / "teacher_model.py",
        "distillation": package / "rl" / "distillation.py",
        "rollout_labels": package / "rl" / "rollout_labels.py",
        "rsl_rl_adapter": package / "rl" / "rsl_rl_models.py",
        "mdp_exports": mdp / "__init__.py",
        "action_processing": mdp / "actions.py",
        "curriculum_processing": mdp / "curricula.py",
        "dynamics_processing": mdp / "dynamics.py",
        "event_processing": mdp / "events.py",
        "observation_processing": mdp / "observations.py",
        "reward_processing": mdp / "rewards.py",
        "termination_processing": mdp / "terminations.py",
        "training_contract": package / "training_contract.py",
    }


def reward_calibration_runtime_input_hashes() -> dict[str, str]:
    """Recompute the complete collection/runtime closure used by calibration."""

    from .reward_calibration import (
        reward_calibration_runtime_input_hashes as strict_runtime_input_hashes,
    )

    return strict_runtime_input_hashes(
        repository_root=REPOSITORY_ROOT,
        feasibility_path=feasibility_config_path(),
        reset_pose_path=Path(
            os.environ.get("G1_RICKSHAW_RESET_POSES", DEFAULT_RESET_POSES_PATH)
        ).resolve(),
        isaaclab_root=Path(os.environ.get("ISAACLAB_PATH", ISAACLAB_ROOT)),
    )


def load_reward_calibration_report(
    report_path: str | Path,
    *,
    teacher_checkpoint_path: str | Path,
) -> dict[str, str]:
    """Reload raw samples and recompute the complete Guide 11.2 gate."""

    from .reward_calibration import (
        RewardCalibrationError,
        load_and_recompute_reward_calibration_report,
        reward_calibration_runtime_versions,
    )

    path = Path(report_path).resolve()
    try:
        validated = load_and_recompute_reward_calibration_report(
            path,
            expected_runtime_hashes=reward_calibration_runtime_input_hashes(),
            expected_runtime_versions=reward_calibration_runtime_versions(),
            teacher_checkpoint_path=teacher_checkpoint_path,
        )
    except (OSError, RewardCalibrationError) as exc:
        raise ValueError(f"reward calibration report failed strict recomputation: {path}") from exc
    calibration = validated["calibration"]
    if calibration.get("status") != "passed" or calibration.get("failures") != []:
        raise ValueError("reward calibration analysis contains failed reward terms/slopes")
    return {
        "reward_calibration_report_sha256": validated["report_sha256"],
        "reward_calibration_content_sha256": validated["content_sha256"],
        "reward_calibration_raw_sample_sha256": validated["raw_sample_sha256"],
    }


def require_pinned_rsl_rl() -> str:
    """Fail before simulation when the pinned RSL-RL 5.x ABI is unavailable."""

    try:
        installed = importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "rsl-rl-lib is not installed in this Python environment; install "
            f"rsl-rl-lib=={EXPECTED_RSL_RL_DISTRIBUTION_VERSION} from commit {RSL_RL_COMMIT}"
        ) from exc
    if installed != EXPECTED_RSL_RL_DISTRIBUTION_VERSION:
        raise RuntimeError(
            "incompatible RSL-RL runtime: installed "
            f"{installed}, required exactly {EXPECTED_RSL_RL_DISTRIBUTION_VERSION} "
            f"(commit {RSL_RL_COMMIT})"
        )
    try:
        from rsl_rl.algorithms import PPO  # noqa: F401
        from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config  # noqa: F401
        from rsl_rl.models import MLPModel  # noqa: F401
        from rsl_rl.runners import OnPolicyRunner
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "the installed rsl-rl-lib does not expose the pinned 5.0.1 actor/critic API"
        ) from exc
    load_parameters = inspect.signature(OnPolicyRunner.load).parameters
    if "load_cfg" not in load_parameters or "strict" not in load_parameters:
        raise RuntimeError("RSL-RL OnPolicyRunner.load does not match the pinned 5.0.1 ABI")
    return installed


def collect_runtime_metadata() -> CheckpointMetadata:
    """Collect complete provenance after validating the pinned runtime version."""

    require_pinned_rsl_rl()
    return collect_checkpoint_metadata(
        runtime_config_files(),
        isaaclab_root=Path(os.environ.get("ISAACLAB_PATH", ISAACLAB_ROOT)),
        # A release install has no .git directory.  Exact distribution-version
        # validation above makes the guide-pinned release/commit mapping the authority.
        rsl_rl_commit=RSL_RL_COMMIT,
    )


def _torch_load(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return dict(value)


def checkpoint_stage(checkpoint: Mapping[str, Any], expected: str | Iterable[str] | None = None) -> str:
    stage = checkpoint.get(CHECKPOINT_STAGE_KEY)
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"checkpoint is missing {CHECKPOINT_STAGE_KEY!r}")
    if expected is not None:
        allowed = {expected} if isinstance(expected, str) else set(expected)
        if stage not in allowed:
            raise ValueError(f"checkpoint stage is {stage!r}; expected one of {sorted(allowed)}")
    return stage


def checkpoint_hash_history(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[int, str]:
    """Validate persisted hashes of previously saved curriculum checkpoints."""

    raw = checkpoint.get(CHECKPOINT_HASH_HISTORY_KEY, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"{CHECKPOINT_HASH_HISTORY_KEY} must be a mapping")
    result: dict[int, str] = {}
    for raw_iteration, digest in raw.items():
        try:
            iteration = int(raw_iteration)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint hash-history iterations must be integers") from exc
        if iteration < 0 or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("checkpoint hash history contains an invalid iteration/SHA256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("checkpoint hash history contains a malformed SHA256") from exc
        result[iteration] = digest.lower()
    if checkpoint_path is not None and checkpoint_stage(checkpoint) == "s0_teacher":
        iteration = checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise ValueError("checkpoint is missing its curriculum iteration")
        result[iteration] = sha256_file(checkpoint_path)
    return result


def load_stage_checkpoint(
    path: str | Path,
    *,
    expected_stage: str | Iterable[str] | None = None,
    validate_runtime: bool = False,
    allow_incomplete: bool = False,
) -> Mapping[str, Any]:
    kwargs: dict[str, Any] = {"config_files": runtime_config_files()}
    if validate_runtime:
        metadata = collect_runtime_metadata()
        kwargs["expected"] = metadata
        kwargs["validate_torch_runtime"] = True
    checkpoint = dict(load_checkpoint_with_validation(path, **kwargs))
    loaded_stage = checkpoint_stage(checkpoint, expected_stage)
    if loaded_stage in GUIDE_TRAINING_PARAMETERS or loaded_stage == "s1_context_candidate":
        training_configuration = validate_guide_training_configuration(
            checkpoint.get(TRAINING_CONFIGURATION_KEY),
            expected_stage=loaded_stage,
        )
        checkpoint[TRAINING_CONFIGURATION_KEY] = training_configuration
        if loaded_stage in {"s0_teacher", "s2_student_ppo"}:
            throughput = validate_training_throughput(
                checkpoint.get(TRAINING_THROUGHPUT_KEY)
            )
            if throughput["num_envs"] != training_configuration["num_envs"]:
                raise ValueError(
                    "checkpoint throughput environment count differs from training configuration"
                )
            if throughput["num_steps_per_env"] != training_configuration[
                "ablation_values"
            ]["rollout_steps"]:
                raise ValueError(
                    "checkpoint throughput rollout length differs from training configuration"
                )
            target = GUIDE_MAX_ITERATIONS[loaded_stage]
            if not 0 < throughput["iterations"] <= target:
                raise ValueError(
                    f"formal {loaded_stage} checkpoint iterations must lie in [1, {target}]"
                )
            if checkpoint.get("iter") != throughput["iterations"] - 1:
                raise ValueError(
                    f"formal {loaded_stage} checkpoint iter must be the last completed "
                    "zero-based iteration"
                )
            if not allow_incomplete and not training_checkpoint_complete(
                checkpoint, stage=loaded_stage
            ):
                raise ValueError(
                    f"formal {loaded_stage} checkpoint has neither reached {target} "
                    "iterations nor terminal early stop"
                )
        if loaded_stage in {
            "s1_context_candidate",
            "s1_context_distillation",
            "s2_student_ppo",
        }:
            validate_student_checkpoint_architecture(
                checkpoint,
                training_configuration,
            )
        if loaded_stage == "s1_context_distillation":
            validate_s1_training_completion(checkpoint)
    elif loaded_stage == "s2_bootstrap":
        training_configuration = validate_training_configuration(
            checkpoint.get(TRAINING_CONFIGURATION_KEY),
            expected_stage=loaded_stage,
        )
        checkpoint[TRAINING_CONFIGURATION_KEY] = training_configuration
        validate_student_checkpoint_architecture(checkpoint, training_configuration)
    return checkpoint


def load_s2_resume_checkpoint(
    path: str | Path,
    *,
    validate_runtime: bool = False,
) -> Mapping[str, Any]:
    """Load a provenance-valid S2 checkpoint without requiring it to be final."""

    kwargs: dict[str, Any] = {"config_files": runtime_config_files()}
    if validate_runtime:
        kwargs["expected"] = collect_runtime_metadata()
        kwargs["validate_torch_runtime"] = True
    checkpoint = dict(load_checkpoint_with_validation(path, **kwargs))
    checkpoint_stage(checkpoint, "s2_student_ppo")
    training_configuration = validate_guide_training_configuration(
        checkpoint.get(TRAINING_CONFIGURATION_KEY),
        expected_stage="s2_student_ppo",
    )
    checkpoint[TRAINING_CONFIGURATION_KEY] = training_configuration
    throughput = validate_training_throughput(
        checkpoint.get(TRAINING_THROUGHPUT_KEY)
    )
    if throughput["num_envs"] != training_configuration["num_envs"]:
        raise ValueError(
            "checkpoint throughput environment count differs from training configuration"
        )
    if throughput["num_steps_per_env"] != training_configuration["ablation_values"][
        "rollout_steps"
    ]:
        raise ValueError(
            "checkpoint throughput rollout length differs from training configuration"
        )
    iterations = int(throughput["iterations"])
    target = GUIDE_MAX_ITERATIONS["s2_student_ppo"]
    if not 0 < iterations <= target:
        raise ValueError(
            f"resumable S2 checkpoint iterations must lie in [1, {target}]"
        )
    if checkpoint.get("iter") != iterations - 1:
        raise ValueError(
            "resumable S2 checkpoint iter must be the last completed zero-based iteration"
        )
    if S2_VALIDATION_STATE_KEY not in checkpoint:
        raise ValueError("resumable S2 checkpoint lacks fixed-validation state")
    validation_state = FixedSeedValidationState.from_mapping(
        checkpoint[S2_VALIDATION_STATE_KEY]
    )
    _validate_fixed_seed_validation_iteration_alignment(validation_state, iterations)
    reports = checkpoint.get("g1_rickshaw_s2_validation_reports", {})
    if not isinstance(reports, Mapping):
        raise ValueError("resumable S2 checkpoint validation report history is malformed")
    validate_student_checkpoint_architecture(checkpoint, training_configuration)
    return checkpoint


def load_s0_resume_checkpoint(
    path: str | Path,
    *,
    validate_runtime: bool = False,
) -> Mapping[str, Any]:
    """Load an intermediate, complete, or terminal-early-stop S0 checkpoint."""

    kwargs: dict[str, Any] = {"config_files": runtime_config_files()}
    if validate_runtime:
        kwargs["expected"] = collect_runtime_metadata()
        kwargs["validate_torch_runtime"] = True
    checkpoint = dict(load_checkpoint_with_validation(path, **kwargs))
    checkpoint_stage(checkpoint, "s0_teacher")
    training_configuration = validate_guide_training_configuration(
        checkpoint.get(TRAINING_CONFIGURATION_KEY),
        expected_stage="s0_teacher",
    )
    checkpoint[TRAINING_CONFIGURATION_KEY] = training_configuration
    throughput = validate_training_throughput(
        checkpoint.get(TRAINING_THROUGHPUT_KEY)
    )
    if throughput["num_envs"] != training_configuration["num_envs"]:
        raise ValueError(
            "checkpoint throughput environment count differs from training configuration"
        )
    if throughput["num_steps_per_env"] != training_configuration["ablation_values"][
        "rollout_steps"
    ]:
        raise ValueError(
            "checkpoint throughput rollout length differs from training configuration"
        )
    iterations = int(throughput["iterations"])
    target = GUIDE_MAX_ITERATIONS["s0_teacher"]
    if not 0 < iterations <= target:
        raise ValueError(
            f"resumable S0 checkpoint iterations must lie in [1, {target}]"
        )
    if checkpoint.get("iter") != iterations - 1:
        raise ValueError(
            "resumable S0 checkpoint iter must be the last completed zero-based iteration"
        )
    if checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY) != iterations:
        raise ValueError(
            "resumable S0 checkpoint curriculum iteration differs from completed iterations"
        )
    if S0_VALIDATION_STATE_KEY not in checkpoint:
        raise ValueError("resumable S0 checkpoint lacks fixed-validation state")
    validation_state = FixedSeedValidationState.from_mapping(
        checkpoint[S0_VALIDATION_STATE_KEY]
    )
    _validate_fixed_seed_validation_iteration_alignment(validation_state, iterations)
    reports = checkpoint.get("g1_rickshaw_s0_validation_reports", {})
    if not isinstance(reports, Mapping):
        raise ValueError("resumable S0 checkpoint validation report history is malformed")
    checkpoint_hash_history(checkpoint, checkpoint_path=path)
    return checkpoint


def load_final_policy_acceptance_artifact(
    report_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> dict[str, str]:
    """Validate and bind the sole S2 report allowed to unlock play/export."""

    from .policy_evaluation import validate_final_student_acceptance_report

    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = load_stage_checkpoint(
        checkpoint_file,
        expected_stage="s2_student_ppo",
    )
    lineage = checkpoint.get(CHECKPOINT_LINEAGE_KEY)
    teacher_digest = (
        lineage.get("teacher_checkpoint_sha256") if isinstance(lineage, Mapping) else None
    )
    context_digest = (
        lineage.get("context_checkpoint_sha256") if isinstance(lineage, Mapping) else None
    )
    for label, digest in (("teacher", teacher_digest), ("S1 context", context_digest)):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"S2 checkpoint is missing its {label} SHA256 lineage")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"S2 checkpoint has a malformed {label} SHA256 lineage") from exc

    report_file = Path(report_path).resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    validate_final_student_acceptance_report(
        report,
        expected_checkpoint_sha256=sha256_file(checkpoint_file),
        expected_teacher_sha256=teacher_digest,
        expected_s1_checkpoint_sha256=context_digest,
    )
    checkpoint_binding = report.get("checkpoint")
    if (
        not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("lineage") != lineage
        or checkpoint_binding.get("provenance")
        != extract_checkpoint_metadata(checkpoint).to_mapping()
    ):
        raise ValueError("final acceptance report does not preserve exact S2 lineage/provenance")
    return {
        "final_acceptance_report_path": os.fspath(report_file),
        "final_acceptance_report_sha256": sha256_file(report_file),
    }


def validate_policy_ablation_run_lineage(
    *,
    checkpoint_path: str | Path,
    teacher_checkpoint_path: str | Path,
    s1_baseline_report_path: str | Path,
    expected_ablation_values: Mapping[str, Any],
    fixed_seeds: Iterable[int],
    episodes_per_slope: int,
    validate_runtime: bool = False,
) -> dict[str, Any]:
    """Validate the complete S0/S1/S2 lineage for one independent sweep run."""

    from .policy_evaluation import validate_s1_baseline_acceptance_report

    expected_values = dict(expected_ablation_values)
    if set(expected_values) != set(ABLATION_VALUE_KEYS):
        raise ValueError("ablation run expectation does not contain the exact variant fields")

    checkpoint_file = Path(checkpoint_path).resolve()
    teacher_file = Path(teacher_checkpoint_path).resolve()
    s1_report_file = Path(s1_baseline_report_path).resolve()
    checkpoint = load_stage_checkpoint(
        checkpoint_file,
        expected_stage="s2_student_ppo",
        validate_runtime=validate_runtime,
    )
    teacher = load_stage_checkpoint(
        teacher_file,
        expected_stage="s0_teacher",
        validate_runtime=validate_runtime,
    )
    checkpoint_digest = sha256_file(checkpoint_file)
    teacher_digest = sha256_file(teacher_file)
    lineage = checkpoint.get(CHECKPOINT_LINEAGE_KEY)
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("teacher_checkpoint_sha256") != teacher_digest
    ):
        raise ValueError("ablation S0 checkpoint differs from the S2 teacher lineage")
    context_digest = lineage.get("context_checkpoint_sha256")
    if not isinstance(context_digest, str) or len(context_digest) != 64:
        raise ValueError("ablation S2 checkpoint lacks its S1 context lineage")

    s1_report = json.loads(s1_report_file.read_text(encoding="utf-8"))
    validate_s1_baseline_acceptance_report(
        s1_report,
        expected_checkpoint_sha256=context_digest,
        fixed_seeds=tuple(fixed_seeds),
        episodes_per_slope=episodes_per_slope,
    )
    s1_checkpoint_binding = s1_report.get("checkpoint")
    s1_checkpoint_path_value = (
        s1_checkpoint_binding.get("path")
        if isinstance(s1_checkpoint_binding, Mapping)
        else None
    )
    if not isinstance(s1_checkpoint_path_value, str):
        raise ValueError("S1 baseline report does not retain its checkpoint path")
    s1_checkpoint_file = Path(s1_checkpoint_path_value).resolve()
    if (
        not s1_checkpoint_file.is_file()
        or sha256_file(s1_checkpoint_file) != context_digest
    ):
        raise ValueError("S1 baseline checkpoint no longer matches the S2 lineage")
    s1_checkpoint = load_stage_checkpoint(
        s1_checkpoint_file,
        expected_stage="s1_context_distillation",
        validate_runtime=validate_runtime,
    )

    configurations = {
        "S0": teacher[TRAINING_CONFIGURATION_KEY],
        "S1": s1_checkpoint[TRAINING_CONFIGURATION_KEY],
        "S2": checkpoint[TRAINING_CONFIGURATION_KEY],
    }
    for stage_name, configuration in configurations.items():
        if configuration["ablation_values"] != expected_values:
            raise ValueError(
                f"ablation {stage_name} configuration is not the exact one-factor variant"
            )
    task_seed = {
        (configuration["task"], configuration["seed"])
        for configuration in configurations.values()
    }
    if len(task_seed) != 1:
        raise ValueError("ablation S0/S1/S2 checkpoints do not share task and training seed")

    checkpoint_provenance = extract_checkpoint_metadata(checkpoint).to_mapping()
    teacher_provenance = extract_checkpoint_metadata(teacher).to_mapping()
    s1_provenance = extract_checkpoint_metadata(s1_checkpoint).to_mapping()
    if teacher_provenance != checkpoint_provenance or s1_provenance != checkpoint_provenance:
        raise ValueError("ablation S0/S1/S2 checkpoints do not share runtime provenance")
    s1_lineage = s1_checkpoint.get(CHECKPOINT_LINEAGE_KEY)
    if (
        not isinstance(s1_lineage, Mapping)
        or s1_lineage.get("teacher_checkpoint_sha256") != teacher_digest
    ):
        raise ValueError("ablation S1 checkpoint differs from the supplied teacher lineage")
    inherited_lineage_fields = {
        "rollout_manifest_sha256",
        "rollout_shards_sha256",
        "s1_selection_report_sha256",
        "selected_candidate_checkpoint_sha256",
        "reward_calibration_report_sha256",
        "reward_calibration_content_sha256",
        "reward_calibration_raw_sample_sha256",
    }
    if any(
        name not in s1_lineage or lineage.get(name) != s1_lineage.get(name)
        for name in inherited_lineage_fields
    ):
        raise ValueError("ablation S2 checkpoint does not preserve the complete S1 lineage")
    if (
        s1_checkpoint_binding.get("stage") != "s1_context_distillation"
        or Path(s1_checkpoint_binding.get("path", "")).resolve() != s1_checkpoint_file
        or s1_checkpoint_binding.get("lineage") != s1_lineage
        or s1_checkpoint_binding.get("provenance") != s1_provenance
    ):
        raise ValueError("S1 baseline report checkpoint binding differs from its artifact")

    return {
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_stage": "s2_student_ppo",
        "checkpoint_lineage": dict(lineage),
        "checkpoint_provenance": checkpoint_provenance,
        "training_configuration": configurations["S2"],
        "training_throughput": validate_training_throughput(
            checkpoint.get(TRAINING_THROUGHPUT_KEY)
        ),
        "teacher_checkpoint_sha256": teacher_digest,
        "teacher_checkpoint_provenance": teacher_provenance,
        "teacher_training_configuration": configurations["S0"],
        "s1_baseline_report_sha256": sha256_file(s1_report_file),
        "s1_checkpoint": os.fspath(s1_checkpoint_file),
        "s1_checkpoint_sha256": context_digest,
        "s1_checkpoint_stage": checkpoint_stage(s1_checkpoint),
        "s1_checkpoint_provenance": s1_provenance,
        "s1_training_configuration": configurations["S1"],
    }


def load_policy_ablation_artifact(
    manifest_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> dict[str, str]:
    """Validate all three independent sweeps and the selected deployment checkpoint."""

    from .policy_evaluation import (
        ABLATION_DEFAULTS,
        FORMAL_EVALUATION_COMMAND_PROTOCOL,
        FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
        FORMAL_EVALUATION_NUM_ENVS_MULTIPLE,
        GUIDE_POLICY_EVALUATION_TASK,
        POLICY_ABLATION_MANIFEST_SCHEMA_VERSION,
        evaluate_ablation_selection,
        load_thresholds,
        serialize_thresholds,
        validate_ablation_matrix,
        validate_final_student_acceptance_report,
    )

    selected_checkpoint = Path(checkpoint_path).resolve()
    selected_digest = sha256_file(selected_checkpoint)
    load_stage_checkpoint(
        selected_checkpoint,
        expected_stage="s2_student_ppo",
        validate_runtime=True,
    )
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "schema_version",
        "report_type",
        "status",
        "created_utc",
        "matrix",
        "matrix_sha256",
        "defaults",
        "selected_run_id",
        "selected_checkpoint_sha256",
        "selection_evidence",
        "runs",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != POLICY_ABLATION_MANIFEST_SCHEMA_VERSION
        or manifest.get("report_type") != "g1_rickshaw_policy_ablation_matrix"
        or manifest.get("status") != "passed"
        or manifest.get("selected_checkpoint_sha256") != selected_digest
    ):
        raise ValueError("ablation manifest is not passed or selects a different checkpoint")
    matrix_path_value = manifest.get("matrix")
    matrix_digest = manifest.get("matrix_sha256")
    if not isinstance(matrix_path_value, str) or not isinstance(matrix_digest, str):
        raise ValueError("ablation manifest lacks its matrix binding")
    matrix_path = Path(matrix_path_value).resolve()
    if not matrix_path.is_file() or sha256_file(matrix_path) != matrix_digest:
        raise ValueError("ablation matrix no longer matches its manifest")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    normalized = validate_ablation_matrix(matrix)
    matrix_defaults = matrix.get("defaults", {}) if isinstance(matrix, Mapping) else None
    if not isinstance(matrix_defaults, Mapping) or manifest.get("defaults") != matrix_defaults:
        raise ValueError("ablation manifest defaults differ from the bound matrix YAML")
    allowed_defaults = {
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
    unknown_defaults = set(matrix_defaults) - allowed_defaults
    if unknown_defaults:
        raise ValueError(f"ablation matrix has unknown defaults: {sorted(unknown_defaults)}")
    effective_defaults = {
        "task": GUIDE_POLICY_EVALUATION_TASK,
        "num_envs": FORMAL_EVALUATION_NUM_ENVS,
        "episodes_per_slope": 100,
        "seeds": [42, 43, 44, 45, 46],
        "curriculum_stages": ["training"],
        "max_policy_steps_per_seed": 6000,
        **dict(matrix_defaults),
    }
    threshold_path_value = effective_defaults.get("thresholds")
    if (
        effective_defaults["task"] != GUIDE_POLICY_EVALUATION_TASK
        or effective_defaults["curriculum_stages"] != ["training"]
        or not isinstance(threshold_path_value, str)
        or not threshold_path_value
    ):
        raise ValueError("ablation matrix does not define the formal Guide evaluation defaults")
    fixed_seeds = effective_defaults["seeds"]
    episodes_per_slope = effective_defaults["episodes_per_slope"]
    num_envs = effective_defaults["num_envs"]
    max_policy_steps = effective_defaults["max_policy_steps_per_seed"]
    if (
        not isinstance(fixed_seeds, list)
        or not fixed_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in fixed_seeds)
        or len(set(fixed_seeds)) != len(fixed_seeds)
        or isinstance(episodes_per_slope, bool)
        or not isinstance(episodes_per_slope, int)
        or episodes_per_slope < 100
        or episodes_per_slope % len(fixed_seeds) != 0
        or isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or num_envs % FORMAL_EVALUATION_NUM_ENVS_MULTIPLE != 0
        or isinstance(max_policy_steps, bool)
        or not isinstance(max_policy_steps, int)
        or max_policy_steps <= 0
    ):
        raise ValueError("ablation matrix has malformed formal evaluation defaults")
    raw_threshold_path = Path(threshold_path_value)
    threshold_path = (
        raw_threshold_path
        if raw_threshold_path.is_absolute()
        else matrix_path.parent / raw_threshold_path
    ).resolve()
    if not threshold_path.is_file():
        raise ValueError("ablation threshold authority no longer exists")
    serialized_thresholds = serialize_thresholds(load_thresholds(threshold_path))
    threshold_digest = sha256_file(threshold_path)

    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != len(normalized):
        raise ValueError("ablation manifest runs must be a list")
    selected_run_id = manifest.get("selected_run_id")
    selected_seen = False
    selection_runs: list[dict[str, Any]] = []
    expected_binding_fields = {
        "id",
        "group",
        "value",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_stage",
        "checkpoint_lineage",
        "checkpoint_provenance",
        "training_configuration",
        "training_throughput",
        "teacher_checkpoint",
        "teacher_checkpoint_sha256",
        "teacher_checkpoint_provenance",
        "teacher_training_configuration",
        "s1_baseline_report",
        "s1_baseline_report_sha256",
        "s1_checkpoint",
        "s1_checkpoint_sha256",
        "s1_checkpoint_stage",
        "s1_checkpoint_provenance",
        "s1_training_configuration",
        "report",
        "report_sha256",
        "command",
        "status",
    }

    def matrix_artifact_path(raw: str) -> Path:
        path = Path(raw)
        return (path if path.is_absolute() else matrix_path.parent / path).resolve()

    for expected, binding in zip(normalized, runs, strict=True):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != expected_binding_fields
            or binding.get("status") != "passed"
        ):
            raise ValueError(f"ablation run {expected['id']!r} is not passed")
        if any(binding.get(name) != expected[name] for name in ("id", "group", "value")):
            raise ValueError("ablation manifest run order/content differs from matrix YAML")
        run_checkpoint_path = matrix_artifact_path(expected["checkpoint"])
        teacher_checkpoint_path = matrix_artifact_path(expected["teacher_checkpoint"])
        s1_baseline_report_path = matrix_artifact_path(expected["s1_baseline_report"])
        for name, path in (
            ("checkpoint", run_checkpoint_path),
            ("teacher_checkpoint", teacher_checkpoint_path),
            ("s1_baseline_report", s1_baseline_report_path),
        ):
            if binding.get(name) != os.fspath(path):
                raise ValueError(
                    f"ablation run {expected['id']!r} {name} differs from matrix YAML"
                )
        command = binding.get("command")
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise ValueError(f"ablation run {expected['id']!r} command is malformed")

        def command_value(flag: str) -> str | None:
            locations = [index for index, item in enumerate(command) if item == flag]
            if len(locations) != 1 or locations[0] + 1 >= len(command):
                return None
            return command[locations[0] + 1]

        def command_values(flag: str) -> list[str] | None:
            locations = [index for index, item in enumerate(command) if item == flag]
            if len(locations) != 1:
                return None
            start = locations[0] + 1
            stop = start
            while stop < len(command) and not command[stop].startswith("--"):
                stop += 1
            return command[start:stop]

        variant_flag = {
            "fat2_weight": "--fat2-weight",
            "rollout_steps": "--rollout-steps",
            "latent_dim": "--latent-dim",
        }[expected["group"]]
        if (
            len(command) < 2
            or Path(command[1]).name != "evaluate_policy.py"
            or command_value("--checkpoint") != os.fspath(run_checkpoint_path)
            or command_value("--teacher-checkpoint") != os.fspath(teacher_checkpoint_path)
            or command_value("--s1-baseline-report") != os.fspath(s1_baseline_report_path)
            or command_value("--ablation-id") != expected["id"]
            or command_value("--ablation-group") != expected["group"]
            or command_value("--ablation-matrix-sha256") != matrix_digest
            or command_value(variant_flag) != str(expected["value"])
            or command_value("--output") != binding.get("report")
            or command_value("--task") != effective_defaults["task"]
            or command_value("--num-envs") != str(num_envs)
            or command_value("--episodes-per-slope") != str(episodes_per_slope)
            or command_values("--seeds") != [str(seed) for seed in fixed_seeds]
            or command_values("--curriculum-stages") != ["training"]
            or command_value("--max-policy-steps-per-seed") != str(max_policy_steps)
            or command_value("--thresholds") != os.fspath(threshold_path)
            or (
                "device" in effective_defaults
                and command_value("--device") != str(effective_defaults["device"])
            )
            or (
                "device" not in effective_defaults
                and "--device" in command
            )
            or (
                bool(effective_defaults.get("headless", False))
                != (command.count("--headless") == 1)
            )
        ):
            raise ValueError(f"ablation run {expected['id']!r} command is misbound")
        expected_values = {**ABLATION_DEFAULTS, expected["group"]: expected["value"]}
        lineage_evidence = validate_policy_ablation_run_lineage(
            checkpoint_path=run_checkpoint_path,
            teacher_checkpoint_path=teacher_checkpoint_path,
            s1_baseline_report_path=s1_baseline_report_path,
            expected_ablation_values=expected_values,
            fixed_seeds=fixed_seeds,
            episodes_per_slope=episodes_per_slope,
        )
        if any(binding.get(name) != value for name, value in lineage_evidence.items()):
            raise ValueError(
                f"ablation run {expected['id']!r} lineage evidence changed"
            )
        run_checkpoint_digest = lineage_evidence["checkpoint_sha256"]

        report_path_value = binding.get("report")
        report_digest = binding.get("report_sha256")
        if not isinstance(report_path_value, str) or not isinstance(report_digest, str):
            raise ValueError(f"ablation run {expected['id']!r} lacks its report binding")
        report_path = Path(report_path_value)
        if not report_path.is_file() or sha256_file(report_path) != report_digest:
            raise ValueError(f"ablation run {expected['id']!r} report hash changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_final_student_acceptance_report(
            report,
            expected_checkpoint_sha256=run_checkpoint_digest,
            expected_teacher_sha256=lineage_evidence["teacher_checkpoint_sha256"],
            expected_s1_checkpoint_sha256=lineage_evidence["s1_checkpoint_sha256"],
        )
        ablation = report.get("ablation")
        report_checkpoint = report.get("checkpoint")
        report_teacher = report.get("teacher_checkpoint")
        report_s1 = report.get("s1_baseline_acceptance")
        report_evaluation = report.get("evaluation")
        report_inputs = report.get("inputs")
        if (
            not isinstance(ablation, Mapping)
            or ablation.get("id") != expected["id"]
            or ablation.get("group") != expected["group"]
            or ablation.get("matrix_sha256") != matrix_digest
            or not isinstance(report_checkpoint, Mapping)
            or report_checkpoint.get("lineage") != lineage_evidence["checkpoint_lineage"]
            or report_checkpoint.get("provenance")
            != lineage_evidence["checkpoint_provenance"]
            or not isinstance(report_teacher, Mapping)
            or report_teacher.get("path") != os.fspath(teacher_checkpoint_path)
            or not isinstance(report_s1, Mapping)
            or report_s1.get("path") != os.fspath(s1_baseline_report_path)
            or report_s1.get("sha256")
            != lineage_evidence["s1_baseline_report_sha256"]
            or not isinstance(report_evaluation, Mapping)
            or report_evaluation.get("fixed_seeds") != fixed_seeds
            or report_evaluation.get("episodes_per_slope_per_stage")
            != episodes_per_slope
            or report_evaluation.get("num_envs") != effective_defaults["num_envs"]
            or report_evaluation.get("curriculum_stages") != ["training"]
            or report_evaluation.get("command_protocol")
            != FORMAL_EVALUATION_COMMAND_PROTOCOL
            or report_evaluation.get("cross_case_protocol")
            != FORMAL_EVALUATION_CROSS_CASE_PROTOCOL
            or not isinstance(report_inputs, Mapping)
            or report_inputs.get("thresholds_sha256") != threshold_digest
            or report.get("thresholds") != serialized_thresholds
        ):
            raise ValueError(f"ablation run {expected['id']!r} report is misbound")
        if expected["id"] == selected_run_id:
            selected_seen = run_checkpoint_digest == selected_digest
        selection_runs.append({**dict(binding), "report_content": report})
    if not selected_seen:
        raise ValueError("selected ablation run does not bind the deployment checkpoint")
    selection_evidence = evaluate_ablation_selection(
        selection_runs,
        selected_run_id=selected_run_id,
    )
    if manifest.get("selection_evidence") != selection_evidence:
        raise ValueError("ablation selection evidence does not recompute from bound reports")
    return {
        "ablation_manifest_path": os.fspath(manifest_file),
        "ablation_manifest_sha256": sha256_file(manifest_file),
    }


def _state_dicts(checkpoint: Mapping[str, Any]) -> Iterable[Mapping[str, torch.Tensor]]:
    for key in (
        "actor_state_dict",
        "student_actor_state_dict",
        "teacher_actor_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            yield value


def _select_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
    }


def extract_gaussian_actor_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract ``GaussianActor`` weights from native S0 or project checkpoints."""

    required_suffixes = {"network.0.weight", "network.6.bias", "log_std"}
    prefixes = (
        "policy.",
        "actor.",
        "module.actor.",
        "policy.actor.",
        "",
    )
    for state in _state_dicts(checkpoint):
        for prefix in prefixes:
            candidate = _select_prefix(state, prefix) if prefix else dict(state)
            if required_suffixes.issubset(candidate):
                return {
                    key: value
                    for key, value in candidate.items()
                    if key.startswith("network.") or key == "log_std"
                }
    raise ValueError("checkpoint does not contain a complete fixed Gaussian actor state_dict")


def extract_student_rsl_actor_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert an S1 student checkpoint to the native RSL actor adapter layout."""

    for state in _state_dicts(checkpoint):
        keys = set(state)
        if "encoder.input.weight" in keys and "policy.network.0.weight" in keys:
            return {key: value for key, value in state.items() if torch.is_tensor(value)}
        if "context_encoder.input.weight" not in keys or "actor.network.0.weight" not in keys:
            continue
        result: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if key.startswith("context_encoder."):
                result["encoder." + key.removeprefix("context_encoder.")] = value
            elif key.startswith("context_projection."):
                result[key] = value
            elif key.startswith("actor."):
                result["policy." + key.removeprefix("actor.")] = value
        return result
    raise ValueError("S1 checkpoint does not contain a complete student actor/context state_dict")


def build_s2_bootstrap_checkpoint(
    teacher_path: str | Path,
    context_path: str | Path,
) -> dict[str, Any]:
    """Build the load-only S2 checkpoint: S1 actor/context plus S0 critic."""

    teacher = load_stage_checkpoint(teacher_path, expected_stage="s0_teacher")
    context = load_stage_checkpoint(context_path, expected_stage="s1_context_distillation")
    context_training_configuration = context[TRAINING_CONFIGURATION_KEY]
    teacher_metadata = extract_checkpoint_metadata(teacher)
    context_metadata = extract_checkpoint_metadata(context)
    if teacher_metadata.to_mapping() != context_metadata.to_mapping():
        raise ValueError("S0 and S1 provenance differ; refusing to mix training lineages")
    critic_state = teacher.get("critic_state_dict")
    if not isinstance(critic_state, Mapping) or not critic_state:
        raise ValueError("S0 checkpoint does not contain the privileged critic state_dict")
    curriculum_iteration = teacher.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    if isinstance(curriculum_iteration, bool) or not isinstance(curriculum_iteration, int):
        raise ValueError("S0 checkpoint is missing its audited curriculum iteration")
    teacher_lineage = teacher.get(CHECKPOINT_LINEAGE_KEY)
    if not isinstance(teacher_lineage, Mapping):
        raise ValueError("S0 checkpoint is missing its training lineage")
    context_lineage = context.get(CHECKPOINT_LINEAGE_KEY)
    if not isinstance(context_lineage, Mapping):
        raise ValueError("S1 checkpoint is missing its training lineage")
    teacher_digest = sha256_file(teacher_path)
    if context_lineage.get("teacher_checkpoint_sha256") != teacher_digest:
        raise ValueError("S1 lineage teacher SHA256 differs from the supplied S0 checkpoint")
    reward_calibration = context.get("reward_calibration")
    reward_report_path = (
        reward_calibration.get("path") if isinstance(reward_calibration, Mapping) else None
    )
    if not isinstance(reward_report_path, str):
        raise ValueError("S1 checkpoint is missing its reward calibration report path")
    current_reward_binding = load_reward_calibration_report(
        reward_report_path,
        teacher_checkpoint_path=teacher_path,
    )
    if any(
        reward_calibration.get(name) != digest or context_lineage.get(name) != digest
        for name, digest in current_reward_binding.items()
    ):
        raise ValueError("S1 reward calibration hashes differ from its current report/lineage")
    model_selection = context.get("model_selection")
    if (
        not isinstance(model_selection, Mapping)
        or model_selection.get("task_return_evaluation_skipped") is not False
    ):
        raise ValueError("S2 requires guide-compliant S1 action-KL/task-return model selection")
    selection_report = model_selection.get("report")
    if (
        not isinstance(selection_report, Mapping)
        or selection_report.get("report_type") != "g1_rickshaw_s1_candidate_selection"
        or selection_report.get("status") != "recorded"
    ):
        raise ValueError("S1 checkpoint does not embed a recorded candidate-selection report")
    selection_report_digest = model_selection.get("report_sha256")
    selection_report_path_value = model_selection.get("report_path")
    selected_candidate_digest = model_selection.get("selected_candidate_checkpoint_sha256")
    for name, digest in (
        ("selection report", selection_report_digest),
        ("selected candidate", selected_candidate_digest),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"S1 {name} SHA256 is missing or malformed")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"S1 {name} SHA256 is missing or malformed") from exc
    if not isinstance(selection_report_path_value, str):
        raise ValueError("S1 selection report path is missing")
    selection_report_path = Path(selection_report_path_value).resolve()
    if (
        not selection_report_path.is_file()
        or sha256_file(selection_report_path) != selection_report_digest
        or json.loads(selection_report_path.read_text(encoding="utf-8"))
        != selection_report
    ):
        raise ValueError("S1 embedded selection report differs from its bound artifact")
    if (
        context_lineage.get("s1_selection_report_sha256") != selection_report_digest
        or context_lineage.get("selected_candidate_checkpoint_sha256")
        != selected_candidate_digest
    ):
        raise ValueError("S1 model-selection hashes differ from checkpoint lineage")
    training = context.get("training")
    selected_iteration = training.get("selected_iteration") if isinstance(training, Mapping) else None
    selection_results = selection_report.get("results")
    selected_result = next(
        (
            result
            for result in selection_results
            if isinstance(result, Mapping)
            and result.get("iteration") == selected_iteration
            and result.get("checkpoint_sha256") == selected_candidate_digest
        ),
        None,
    ) if isinstance(selection_results, list) else None
    if (
        isinstance(selected_iteration, bool)
        or not isinstance(selected_iteration, int)
        or not isinstance(selection_results, list)
        or not isinstance(selected_result, Mapping)
    ):
        raise ValueError("S1 selected candidate is not bound to its selection report")
    selected_candidate_path_value = selected_result.get("checkpoint")
    if not isinstance(selected_candidate_path_value, str):
        raise ValueError("S1 selection report does not retain the selected candidate path")
    selected_candidate_path = Path(selected_candidate_path_value).resolve()
    if (
        not selected_candidate_path.is_file()
        or sha256_file(selected_candidate_path) != selected_candidate_digest
    ):
        raise ValueError("S1 selected candidate checkpoint hash changed")
    selected_candidate = load_stage_checkpoint(
        selected_candidate_path,
        expected_stage="s1_context_candidate",
    )
    if selected_candidate.get("candidate_iteration") != selected_iteration:
        raise ValueError("S1 selected candidate iteration differs from its report")
    selected_state = selected_candidate.get("model_state_dict")
    context_state = context.get("model_state_dict")
    if (
        not isinstance(selected_state, Mapping)
        or not isinstance(context_state, Mapping)
        or set(selected_state) != set(context_state)
        or any(
            not torch.is_tensor(selected_state[name])
            or not torch.is_tensor(context_state[name])
            or not torch.equal(selected_state[name].cpu(), context_state[name].cpu())
            for name in selected_state
        )
    ):
        raise ValueError("final S1 model weights differ from the selected candidate")
    rollout_manifest_digest = context_lineage.get("rollout_manifest_sha256")
    rollout_shards = context_lineage.get("rollout_shards_sha256")
    if not isinstance(rollout_manifest_digest, str) or len(rollout_manifest_digest) != 64:
        raise ValueError("S1 lineage is missing the rollout manifest SHA256")
    try:
        int(rollout_manifest_digest, 16)
    except ValueError as exc:
        raise ValueError("S1 rollout manifest lineage contains a malformed SHA256") from exc
    if not isinstance(rollout_shards, Mapping) or not rollout_shards:
        raise ValueError("S1 lineage is missing content-addressed rollout shards")
    for name, digest in rollout_shards.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("S1 rollout shard lineage contains a malformed entry")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("S1 rollout shard lineage contains a malformed SHA256") from exc
    context_iteration = context.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    if context_iteration != curriculum_iteration:
        raise ValueError("S1 and S0 curriculum lineages differ")
    teacher_hashes = checkpoint_hash_history(teacher, checkpoint_path=teacher_path)
    if checkpoint_hash_history(context) != teacher_hashes:
        raise ValueError("S1 checkpoint does not preserve the exact S0 validation hash history")
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STAGE_KEY: "s2_bootstrap",
        CHECKPOINT_CURRICULUM_ITERATION_KEY: curriculum_iteration,
        CHECKPOINT_HASH_HISTORY_KEY: {
            str(iteration): digest
            for iteration, digest in teacher_hashes.items()
        },
        "actor_state_dict": extract_student_rsl_actor_state(context),
        "critic_state_dict": dict(critic_state),
        "iter": 0,
        "infos": {"load_optimizer": False, "load_iteration": False},
        TRAINING_CONFIGURATION_KEY: finalize_training_configuration({
            "schema_version": TRAINING_CONFIGURATION_SCHEMA_VERSION,
            "stage": "s2_bootstrap",
            "formal": True,
            "task": GUIDE_TRAINING_TASK,
            "num_envs": GUIDE_TRAINING_NUM_ENVS,
            "seed": context_training_configuration["seed"],
            "max_iterations": GUIDE_MAX_ITERATIONS["s2_student_ppo"],
            "argv": [],
            "hydra_overrides": [],
            "guide_parameters": {
                "source_stage": "s1_context_distillation",
                "source_checkpoint_sha256": sha256_file(context_path),
            },
            "resolved_parameters": {},
            "actor_initialized_from_teacher": True,
            "stage_coverage": context_training_configuration["stage_coverage"],
            "ablation_values": dict(
                context_training_configuration["ablation_values"]
            ),
            "inputs_sha256": {
                "teacher_checkpoint": teacher_digest,
                "context_checkpoint": sha256_file(context_path),
            },
        }),
        CHECKPOINT_LINEAGE_KEY: {
            "teacher_checkpoint_sha256": teacher_digest,
            "context_checkpoint_sha256": sha256_file(context_path),
            "rollout_manifest_sha256": rollout_manifest_digest,
            "rollout_shards_sha256": dict(rollout_shards),
            "s1_selection_report_sha256": selection_report_digest,
            "selected_candidate_checkpoint_sha256": selected_candidate_digest,
            **current_reward_binding,
        },
    }
    attach_checkpoint_metadata(checkpoint, context_metadata)
    return checkpoint


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _deployment_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    from .validation import asset_hashes, validation_input_assets

    training_configuration = validate_guide_training_configuration(
        checkpoint.get(TRAINING_CONFIGURATION_KEY),
        expected_stage="s2_student_ppo",
    )
    latent_dim = int(training_configuration["ablation_values"]["latent_dim"])

    with Path(os.environ.get("G1_RICKSHAW_RESET_POSES", DEFAULT_RESET_POSES_PATH)).open(
        "r", encoding="utf-8"
    ) as stream:
        reset = yaml.safe_load(stream)
    with feasibility_config_path().open(
        "r", encoding="utf-8"
    ) as stream:
        feasibility = yaml.safe_load(stream)
    calibration = feasibility["calibration"]
    ranges = feasibility["ranges"]
    command_ranges: dict[str, dict[str, Any]] = {}
    for name, source_name, unit in (
        ("acceleration_limit", "command.acceleration_limit", "m/s^2"),
        ("jerk_limit", "command.jerk_limit", "m/s^3"),
    ):
        interval = ranges.get(source_name) if isinstance(ranges, Mapping) else None
        if not isinstance(interval, Mapping) or set(interval) != {"min", "max"}:
            raise ValueError(f"feasibility envelope is missing ranges.{source_name}")
        minimum = float(interval["min"])
        maximum = float(interval["max"])
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0.0 or minimum > maximum:
            raise ValueError(f"feasibility envelope has invalid ranges.{source_name}")
        command_ranges[name] = {
            "min": minimum,
            "max": maximum,
            "unit": unit,
            "source": f"ranges.{source_name}",
        }
    safety = {key.removeprefix("safety."): value for key, value in calibration.items() if key.startswith("safety.")}
    action_scales = [0.40] * 12 + [0.20] * 3
    for _ in range(2):
        action_scales.extend([0.25] * 3 + [0.30] + [0.15] * 3)
    return {
        "schema_version": 1,
        "policy": {
            "type": "deterministic_student_mean",
            "inputs": {"current": [None, 96], "history": [None, 61, 96]},
            "history_bottleneck_dim": latent_dim,
            "actor_context_dim": 16,
            "output": {"normalized_action": [None, 29], "clip": [-1.0, 1.0]},
            "forbidden_components": ["teacher_encoder", "critic", "privileged_observations", "auxiliary_heads"],
        },
        "deployment_controller": {
            "artifact": "deployment_controller.pt",
            "stateless": True,
            "inputs": {
                "current": [None, 96],
                "history": [None, 61, 96],
                "q_ref": [None, 29],
                "x_prev": [None, 29],
                "y_prev": [None, 29],
            },
            "outputs": ["clipped_normalized_action", "filtered_joint_target", "x_next", "y_next"],
        },
        "observation": {
            "history_excludes_current": True,
            "policy_hz": 50.0,
            "runtime_empirical_normalization": False,
            "clip": None,
            "layout": [
                {"name": "base_angular_velocity", "slice": [0, 3], "scale": [0.25, 0.25, 0.25]},
                {"name": "projected_gravity", "slice": [3, 6], "scale": [1.0, 1.0, 1.0]},
                {"name": "task_signal", "fields": ["v_ref", "e_y", "e_psi"], "slice": [6, 9], "scale": [2.0, 2.0, 1.0]},
                {"name": "joint_position_minus_q_ref", "slice": [9, 38], "scale": [1.0] * 29},
                {"name": "joint_velocity", "slice": [38, 67], "scale": [0.05] * 29},
                {"name": "previous_processed_action", "slice": [67, 96], "scale": [1.0] * 29},
            ],
        },
        "command": command_ranges,
        "action": {
            "joint_order": list(FIXED_G1_JOINT_ORDER),
            "scale_rad_per_normalized_action": action_scales,
            "q_ref_by_gradient": [
                {"gradient": pose["gradient"], "q_ref": pose["q_ref"]} for pose in reset["poses"]
            ],
            "butterworth": {
                "sample_rate_hz": 50.0,
                "cutoff_hz": 4.0,
                "b0": 0.20430082,
                "b1": 0.20430082,
                "a1": -0.59139835,
                "equation": "y=b0*x+b1*x_prev-a1*y_prev",
                "reset_x_prev_and_y_prev_to_q_ref": True,
            },
        },
        "safety": {"persistent_steps": 10, "root_height_min_m": 0.31, **safety},
        "source_assets_sha256": asset_hashes(validation_input_assets()),
        "provenance": extract_checkpoint_metadata(checkpoint).to_mapping(),
        "training_configuration": training_configuration,
    }


def write_deployment_manifest(export_dir: str | Path, checkpoint_path: str | Path) -> Path:
    """Write a hash-complete manifest next to JIT/ONNX deployment artifacts."""

    export_path = Path(export_dir)
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stage="s2_student_ppo",
        validate_runtime=True,
    )
    allowed_files = {
        "policy.pt",
        "policy.onnx",
        "deployment_controller.pt",
        "manifest.json",
        "SHA256SUMS",
    }
    unexpected = sorted(
        path.name
        for path in export_path.iterdir()
        if path.is_file() and path.name not in allowed_files
    )
    if unexpected:
        raise RuntimeError(f"deployment directory contains non-deployable artifacts: {unexpected}")
    required_files = {"policy.pt", "policy.onnx", "deployment_controller.pt"}
    missing = sorted(name for name in required_files if not (export_path / name).is_file())
    if missing:
        raise RuntimeError(
            "deployment directory is missing required controller/policy artifacts: "
            + ", ".join(missing)
        )
    acceptance_path = os.environ.get("G1_RICKSHAW_FINAL_ACCEPTANCE_REPORT")
    if not acceptance_path:
        raise RuntimeError("deployment export requires G1_RICKSHAW_FINAL_ACCEPTANCE_REPORT")
    acceptance_binding = load_final_policy_acceptance_artifact(
        acceptance_path,
        checkpoint_path=checkpoint_path,
    )
    ablation_path = os.environ.get("G1_RICKSHAW_ABLATION_MANIFEST")
    if not ablation_path:
        raise RuntimeError("deployment export requires G1_RICKSHAW_ABLATION_MANIFEST")
    ablation_binding = load_policy_ablation_artifact(
        ablation_path,
        checkpoint_path=checkpoint_path,
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(export_path.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    manifest = _deployment_contract(checkpoint)
    manifest["source_checkpoint_sha256"] = sha256_file(checkpoint_path)
    manifest["validation"] = {**acceptance_binding, **ablation_binding}
    manifest["artifacts_sha256"] = artifacts
    destination = export_path / "manifest.json"
    _atomic_json(manifest, destination)
    all_hashes = {**artifacts, destination.name: sha256_file(destination)}
    sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items()))
    descriptor, temporary = tempfile.mkstemp(dir=export_path, prefix=".SHA256SUMS.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(sums)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, export_path / "SHA256SUMS")
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _unwrap_env(env: Any) -> Any:
    visited: set[int] = set()
    while id(env) not in visited:
        visited.add(id(env))
        next_env = getattr(env, "unwrapped", None)
        if next_env is None or next_env is env:
            next_env = getattr(env, "env", None)
        if next_env is None or next_env is env:
            break
        env = next_env
    return env


def reset_runner_environment_for_curriculum(env: Any) -> int:
    """Full-reset all environments without injecting policy steps."""

    env.reset()
    return 0


def install_runner_hooks_from_environment() -> None:
    """Install provenance/curriculum/export hooks into pinned RSL's runner."""

    if os.environ.get("G1_RICKSHAW_RUNNER_HOOK") != "1":
        return
    require_pinned_rsl_rl()
    from rsl_rl.runners import OnPolicyRunner

    if getattr(OnPolicyRunner, "_g1_rickshaw_hook_installed", False):
        return
    metadata = collect_runtime_metadata()
    stage = os.environ.get("G1_RICKSHAW_CHECKPOINT_STAGE")
    if not stage:
        raise RuntimeError("G1_RICKSHAW_CHECKPOINT_STAGE is required when runner hooks are enabled")
    lineage_raw = os.environ.get("G1_RICKSHAW_CHECKPOINT_LINEAGE", "{}")
    lineage = json.loads(lineage_raw)
    if not isinstance(lineage, Mapping):
        raise RuntimeError("G1_RICKSHAW_CHECKPOINT_LINEAGE must encode a JSON mapping")
    training_configuration: dict[str, Any] | None = None
    training_configuration_raw = os.environ.get("G1_RICKSHAW_TRAINING_CONFIGURATION")
    if training_configuration_raw is not None:
        try:
            raw_configuration = json.loads(training_configuration_raw)
            if raw_configuration.get("formal") is True:
                training_configuration = validate_guide_training_configuration(
                    raw_configuration,
                    expected_stage=stage,
                )
            else:
                training_configuration = validate_training_configuration(
                    raw_configuration,
                    expected_stage=stage,
                    require_formal=False,
                )
        except ValueError as exc:
            raise RuntimeError(
                "G1_RICKSHAW_TRAINING_CONFIGURATION is not a valid audited configuration"
            ) from exc

    original_init = OnPolicyRunner.__init__
    original_learn = OnPolicyRunner.learn
    original_load = OnPolicyRunner.load
    original_export_jit = OnPolicyRunner.export_policy_to_jit
    original_export_onnx = OnPolicyRunner.export_policy_to_onnx

    def set_curriculum(runner: Any, iteration: int) -> None:
        env = _unwrap_env(runner.env)
        callback = getattr(env, "set_curriculum_iteration", None)
        if not callable(callback):
            raise RuntimeError("environment does not expose set_curriculum_iteration")
        callback(int(iteration))

    def run_periodic_validation(runner: Any) -> None:
        iteration = int(runner._g1_training_iterations)
        validation_state: FixedSeedValidationState = runner._g1_validation_state
        if not validation_state.should_evaluate(iteration):
            return
        base_env = _unwrap_env(runner.env)
        runtime_stage = base_env.curriculum_runtime_state.stage.name
        stage_alias = {"TRAINING": "training"}.get(runtime_stage)
        if stage_alias is None:
            raise RuntimeError(f"unsupported validation curriculum stage {runtime_stage!r}")
        log_dir = getattr(getattr(runner, "logger", None), "log_dir", None)
        if not log_dir:
            raise RuntimeError("periodic validation requires an RSL runner log directory")
        validation_dir = Path(log_dir) / "fixed_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = validation_dir / f"model_input_{iteration:06d}.pt"
        report_path = validation_dir / f"report_{iteration:06d}.json"
        previous_learning_iteration = int(runner.current_learning_iteration)
        runner.current_learning_iteration = iteration - 1
        try:
            runner.save(os.fspath(checkpoint_path))
        finally:
            runner.current_learning_iteration = previous_learning_iteration

        prefix = "S0" if stage == "s0_teacher" else "S2"
        seeds_variable = f"G1_RICKSHAW_{prefix}_VALIDATION_SEEDS"
        raw_seeds = os.environ.get(seeds_variable, "42,43,44,45,46")
        try:
            fixed_seeds = [int(value.strip()) for value in raw_seeds.split(",") if value.strip()]
        except ValueError as exc:
            raise RuntimeError(f"{seeds_variable} must be comma-separated integers") from exc
        if not fixed_seeds or len(set(fixed_seeds)) != len(fixed_seeds):
            raise RuntimeError("fixed validation seeds must be non-empty and unique")
        command_variable = f"G1_RICKSHAW_{prefix}_VALIDATION_COMMAND"
        command_template = os.environ.get(command_variable)
        if command_template:
            command = shlex.split(
                command_template.format(
                    checkpoint=os.fspath(checkpoint_path),
                    output=os.fspath(report_path),
                    stage=stage_alias,
                    iteration=iteration,
                )
            )
        else:
            task = os.environ.get("G1_RICKSHAW_TASK")
            if not task:
                raise RuntimeError("G1_RICKSHAW_TASK is required for periodic validation")
            command = [
                sys.executable,
                os.fspath(REPOSITORY_ROOT / "scripts" / "evaluate_policy.py"),
                "--task",
                task,
                "--checkpoint",
                os.fspath(checkpoint_path),
                "--output",
                os.fspath(report_path),
                "--num-envs",
                str(FORMAL_EVALUATION_NUM_ENVS),
                "--episodes-per-slope",
                "100",
                "--seeds",
                *(str(seed) for seed in fixed_seeds),
                "--curriculum-stages",
                stage_alias,
                "--training-monitor",
                "--headless",
            ]
            if stage == "s2_student_ppo":
                command.extend(("--allow-missing-teacher", "--no-context-interventions"))
            validation_device = os.environ.get(
                f"G1_RICKSHAW_{prefix}_VALIDATION_DEVICE"
            )
            if validation_device:
                command.extend(("--device", validation_device))
        subprocess.run(command, check=True)
        score = load_fixed_seed_validation_score(
            report_path,
            checkpoint_path=checkpoint_path,
            curriculum_stage=stage_alias,
            fixed_seeds=fixed_seeds,
        )
        report_digest = sha256_file(report_path)
        should_stop = validation_state.record(
            iteration=iteration,
            stage=stage_alias,
            score=score,
            report_sha256=report_digest,
        )
        runner._g1_validation_reports[str(iteration)] = report_digest
        if should_stop:
            runner.current_learning_iteration = max(0, iteration - 1)
            final_path = validation_dir / f"model_early_stop_{iteration:06d}.pt"
            runner.save(os.fspath(final_path))
            raise _EarlyStopSignal(
                f"{stage} fixed validation did not improve for "
                f"{validation_state.patience} evaluations"
            )
        validated_path = validation_dir / f"model_validated_{iteration:06d}.pt"
        previous_learning_iteration = int(runner.current_learning_iteration)
        runner.current_learning_iteration = iteration - 1
        try:
            runner.save(os.fspath(validated_path))
        finally:
            runner.current_learning_iteration = previous_learning_iteration

    def hooked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if training_configuration is not None:
            configured_num_envs = training_configuration.get("num_envs")
            actual_num_envs = int(self.env.num_envs)
            if configured_num_envs != actual_num_envs:
                raise RuntimeError(
                    "formal training environment count differs from the published "
                    f"configuration: actual={actual_num_envs}, configured={configured_num_envs}"
                )
            configured_steps = training_configuration["ablation_values"]["rollout_steps"]
            actual_steps = int(self.cfg["num_steps_per_env"])
            if configured_steps != actual_steps:
                raise RuntimeError(
                    "formal training rollout length differs from the published "
                    f"configuration: actual={actual_steps}, configured={configured_steps}"
                )
        self._g1_training_wall_time_s = 0.0
        self._g1_training_transitions = 0
        self._g1_training_iterations = 0
        original_log = self.logger.log

        def log_with_throughput(*log_args, **log_kwargs):
            collect_time = float(log_kwargs.get("collect_time", 0.0))
            learn_time = float(log_kwargs.get("learn_time", 0.0))
            wall_time = collect_time + learn_time
            if not math.isfinite(wall_time) or wall_time <= 0.0:
                raise RuntimeError("RSL training iteration reported invalid wall-clock timing")
            self._g1_training_wall_time_s += wall_time
            self._g1_training_transitions += int(self.env.num_envs) * int(
                self.cfg["num_steps_per_env"]
            )
            self._g1_training_iterations += 1
            if stage in {"s0_teacher", "s2_student_ppo"}:
                run_periodic_validation(self)
            return original_log(*log_args, **log_kwargs)

        self.logger.log = log_with_throughput
        self._g1_checkpoint_hash_by_iteration = {}
        if stage in {"s0_teacher", "s2_student_ppo"}:
            self._g1_validation_state = FixedSeedValidationState()
            self._g1_validation_reports = {}
        if stage == "s2_student_ppo":
            raw_start = os.environ.get("G1_RICKSHAW_CURRICULUM_START_ITERATION")
            if raw_start is None:
                raise RuntimeError("S2/Play requires G1_RICKSHAW_CURRICULUM_START_ITERATION from checkpoint lineage")
            try:
                curriculum_start = int(raw_start)
            except ValueError as exc:
                raise RuntimeError("G1_RICKSHAW_CURRICULUM_START_ITERATION must be an integer") from exc
            self._g1_curriculum_iteration = curriculum_start
        else:
            self._g1_curriculum_iteration = int(self.current_learning_iteration)
        set_curriculum(self, self._g1_curriculum_iteration)
        original_update = self.alg.update

        def update_with_curriculum(*update_args, **update_kwargs):
            result = original_update(*update_args, **update_kwargs)
            self._g1_curriculum_iteration += 1
            set_curriculum(self, self._g1_curriculum_iteration)
            return result

        self.alg.update = update_with_curriculum

    def hooked_learn(self, *args, **kwargs):
        if (
            stage in {"s0_teacher", "s2_student_ppo"}
            and training_configuration is not None
            and training_configuration["formal"] is True
        ):
            call_args = list(args)
            call_kwargs = dict(kwargs)
            if call_args:
                requested = call_args[0]
            elif "num_learning_iterations" in call_kwargs:
                requested = call_kwargs["num_learning_iterations"]
            else:
                raise RuntimeError("RSL learn call is missing num_learning_iterations")
            if isinstance(requested, bool) or not isinstance(requested, int):
                raise RuntimeError("RSL num_learning_iterations must be an integer")
            if stage == "s0_teacher":
                remaining = s0_remaining_learning_iterations(
                    requested_iterations=requested,
                    completed_iterations=int(self._g1_training_iterations),
                    early_stopped=s0_validation_early_stopped(
                        self._g1_validation_state
                    ),
                )
            else:
                remaining = s2_remaining_learning_iterations(
                    requested_iterations=requested,
                    completed_iterations=int(self._g1_training_iterations),
                    early_stopped=fixed_seed_validation_early_stopped(
                        self._g1_validation_state
                    ),
                )
            if remaining == 0:
                reason = (
                    "reached the iteration target or terminal early stop"
                    if stage == "s0_teacher"
                    else "reached the iteration target or terminal early stop"
                )
                print(f"[INFO] {stage} {reason}")
                return None
            if call_args:
                call_args[0] = remaining
            else:
                call_kwargs["num_learning_iterations"] = remaining
            args = tuple(call_args)
            kwargs = call_kwargs
        try:
            return original_learn(self, *args, **kwargs)
        except _EarlyStopSignal as exc:
            print(f"[INFO] {exc}")
            logger = getattr(self, "logger", None)
            if logger is not None and getattr(logger, "writer", None) is not None:
                logger.stop_logging_writer()
            return None

    def hooked_save(self, path: str, infos: dict | None = None):
        if training_configuration is None:
            raise RuntimeError(
                "training checkpoint save requires G1_RICKSHAW_TRAINING_CONFIGURATION"
            )
        if self._g1_training_iterations <= 0 or self._g1_training_wall_time_s <= 0.0:
            raise RuntimeError("training checkpoint has no measured wall-clock throughput evidence")
        saved = self.alg.save()
        if not isinstance(saved, Mapping):
            raise RuntimeError("RSL algorithm save payload must be a mapping")
        checkpoint = dict(saved)
        checkpoint["iter"] = self.current_learning_iteration
        checkpoint["infos"] = infos
        checkpoint["schema_version"] = CHECKPOINT_SCHEMA_VERSION
        checkpoint[CHECKPOINT_STAGE_KEY] = stage
        checkpoint[CHECKPOINT_CURRICULUM_ITERATION_KEY] = int(self._g1_curriculum_iteration)
        checkpoint[TRAINING_CONFIGURATION_KEY] = dict(training_configuration)
        checkpoint[TRAINING_THROUGHPUT_KEY] = {
            "iterations": int(self._g1_training_iterations),
            "transitions": int(self._g1_training_transitions),
            "wall_time_s": float(self._g1_training_wall_time_s),
            "samples_per_second": float(
                self._g1_training_transitions / self._g1_training_wall_time_s
            ),
            "num_envs": int(self.env.num_envs),
            "num_steps_per_env": int(self.cfg["num_steps_per_env"]),
        }
        checkpoint[CHECKPOINT_HASH_HISTORY_KEY] = {
            str(iteration): digest
            for iteration, digest in sorted(self._g1_checkpoint_hash_by_iteration.items())
        }
        if stage in {"s0_teacher", "s2_student_ppo"}:
            checkpoint[_validation_state_key(stage)] = self._g1_validation_state.to_mapping()
            checkpoint[f"g1_rickshaw_{'s0' if stage == 's0_teacher' else 's2'}_validation_reports"] = dict(
                self._g1_validation_reports
            )
        checkpoint[CHECKPOINT_LINEAGE_KEY] = dict(lineage)
        attach_checkpoint_metadata(checkpoint, metadata, replace=True)
        atomic_torch_save(checkpoint, path)
        self.logger.save_model(path, self.current_learning_iteration)
        self._g1_last_checkpoint_sha256 = sha256_file(path)
        if stage == "s0_teacher":
            self._g1_checkpoint_hash_by_iteration[
                int(self._g1_curriculum_iteration)
            ] = self._g1_last_checkpoint_sha256

    def hooked_load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        checkpoint = _torch_load(path)
        validate_checkpoint(
            checkpoint,
            expected=metadata,
            config_files=runtime_config_files(),
            validate_torch_runtime=True,
        )
        loaded_stage = checkpoint_stage(checkpoint)
        if loaded_stage == "s2_bootstrap":
            loaded_training_configuration = validate_training_configuration(
                checkpoint.get(TRAINING_CONFIGURATION_KEY),
                expected_stage=loaded_stage,
            )
        else:
            loaded_training_configuration = validate_guide_training_configuration(
                checkpoint.get(TRAINING_CONFIGURATION_KEY),
                expected_stage=loaded_stage,
            )
        if loaded_stage in {"s2_bootstrap", "s2_student_ppo"}:
            validate_student_checkpoint_architecture(
                checkpoint,
                loaded_training_configuration,
            )
        allowed_load_stages = (
            {"s0_teacher"} if stage == "s0_teacher" else {"s2_bootstrap", "s2_student_ppo"}
        )
        if loaded_stage not in allowed_load_stages:
            raise RuntimeError(
                f"runner stage {stage!r} cannot load checkpoint stage {loaded_stage!r}"
            )
        normalized_throughput: dict[str, float | int] | None = None
        loaded_validation_state: FixedSeedValidationState | None = None
        if loaded_stage != "s2_bootstrap":
            try:
                normalized_throughput = validate_training_throughput(
                    checkpoint.get(TRAINING_THROUGHPUT_KEY)
                )
            except ValueError as exc:
                raise RuntimeError("loaded checkpoint has malformed throughput evidence") from exc
            if normalized_throughput["num_envs"] != loaded_training_configuration["num_envs"]:
                raise RuntimeError(
                    "loaded checkpoint environment throughput differs from training configuration"
                )
            if normalized_throughput["num_steps_per_env"] != loaded_training_configuration[
                "ablation_values"
            ]["rollout_steps"]:
                raise RuntimeError(
                    "loaded checkpoint rollout throughput differs from training configuration"
                )
            if (
                training_configuration is not None
                and loaded_stage == stage
                and loaded_training_configuration["content_sha256"]
                != training_configuration["content_sha256"]
            ):
                raise RuntimeError(
                    f"resumed {loaded_stage} checkpoint training configuration differs "
                    "from the active run"
                )
            iterations = int(normalized_throughput["iterations"])
            if loaded_stage in {"s0_teacher", "s2_student_ppo"}:
                state_key = _validation_state_key(loaded_stage)
                if state_key not in checkpoint:
                    raise RuntimeError(
                        f"loaded {loaded_stage} checkpoint lacks fixed-validation state"
                    )
                try:
                    loaded_validation_state = FixedSeedValidationState.from_mapping(
                        checkpoint[state_key]
                    )
                    _validate_fixed_seed_validation_iteration_alignment(
                        loaded_validation_state,
                        iterations,
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"loaded {loaded_stage} checkpoint has invalid or misaligned "
                        "fixed-validation state"
                    ) from exc
            if loaded_stage == "s0_teacher":
                try:
                    s0_remaining_learning_iterations(
                        requested_iterations=GUIDE_MAX_ITERATIONS["s0_teacher"],
                        completed_iterations=iterations,
                        early_stopped=fixed_seed_validation_early_stopped(
                            loaded_validation_state
                        ),
                    )
                except ValueError as exc:
                    raise RuntimeError("loaded S0 checkpoint exceeds the Guide target") from exc
                if checkpoint.get("iter") != iterations - 1:
                    raise RuntimeError(
                        "loaded S0 checkpoint iter is not the last completed zero-based iteration"
                    )
            if loaded_stage == "s2_student_ppo":
                try:
                    s2_remaining_learning_iterations(
                        requested_iterations=GUIDE_MAX_ITERATIONS["s2_student_ppo"],
                        completed_iterations=iterations,
                        early_stopped=fixed_seed_validation_early_stopped(
                            loaded_validation_state
                        ),
                    )
                except ValueError as exc:
                    raise RuntimeError("loaded S2 checkpoint exceeds the Guide target") from exc
                if checkpoint.get("iter") != iterations - 1:
                    raise RuntimeError(
                        "loaded S2 checkpoint iter is not the last completed zero-based iteration"
                    )
        if loaded_stage == "s2_bootstrap":
            load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
            strict = True
        result = original_load(self, path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        self._g1_rickshaw_checkpoint_path = os.fspath(path)
        curriculum_iteration = checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
        if isinstance(curriculum_iteration, bool) or not isinstance(curriculum_iteration, int):
            raise RuntimeError("loaded checkpoint is missing an audited curriculum iteration")
        self._g1_curriculum_iteration = curriculum_iteration
        if (
            loaded_stage == "s0_teacher"
            and normalized_throughput is not None
            and curriculum_iteration != normalized_throughput["iterations"]
        ):
            raise RuntimeError(
                "loaded S0 curriculum iteration differs from completed training iterations"
            )
        if normalized_throughput is not None:
            self._g1_training_iterations = int(normalized_throughput["iterations"])
            self._g1_training_transitions = int(normalized_throughput["transitions"])
            self._g1_training_wall_time_s = float(normalized_throughput["wall_time_s"])
            if loaded_stage in {"s0_teacher", "s2_student_ppo"}:
                # RSL stores the final completed 0-based index, while learn(n)
                # starts at current_learning_iteration and executes n more loops.
                self.current_learning_iteration = self._g1_training_iterations
        self._g1_checkpoint_hash_by_iteration = checkpoint_hash_history(
            checkpoint,
            checkpoint_path=path,
        )
        if loaded_validation_state is not None:
            self._g1_validation_state = loaded_validation_state
            report_key = (
                "g1_rickshaw_s0_validation_reports"
                if loaded_stage == "s0_teacher"
                else "g1_rickshaw_s2_validation_reports"
            )
            reports = checkpoint.get(report_key, {})
            if not isinstance(reports, Mapping):
                raise RuntimeError("fixed-seed validation report history must be a mapping")
            self._g1_validation_reports = dict(reports)
        set_curriculum(self, self._g1_curriculum_iteration)
        self._g1_curriculum_reset_steps = reset_runner_environment_for_curriculum(
            self.env
        )
        return result

    def hooked_export_jit(self, path: str, filename: str = "policy.pt"):
        result = original_export_jit(self, path, filename)
        checkpoint_path = getattr(self, "_g1_rickshaw_checkpoint_path", None)
        if checkpoint_path is None:
            raise RuntimeError("deployment export requires a provenance-validated loaded checkpoint")
        controller_factory = getattr(self.alg.get_policy(), "as_deployment_controller", None)
        if not callable(controller_factory):
            raise RuntimeError("student policy does not expose the deployment controller contract")
        controller = controller_factory().to("cpu").eval()
        torch.jit.script(controller).save(os.fspath(Path(path) / "deployment_controller.pt"))
        return result

    def hooked_export_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False):
        result = original_export_onnx(self, path, filename, verbose)
        checkpoint_path = getattr(self, "_g1_rickshaw_checkpoint_path", None)
        if checkpoint_path is None:
            raise RuntimeError("deployment export requires a provenance-validated loaded checkpoint")
        write_deployment_manifest(path, checkpoint_path)
        return result

    OnPolicyRunner.__init__ = hooked_init
    OnPolicyRunner.learn = hooked_learn
    OnPolicyRunner.save = hooked_save
    OnPolicyRunner.load = hooked_load
    OnPolicyRunner.export_policy_to_jit = hooked_export_jit
    OnPolicyRunner.export_policy_to_onnx = hooked_export_onnx
    OnPolicyRunner._g1_rickshaw_hook_installed = True


__all__ = [
    "ABLATION_VALUE_OPTIONS",
    "SIGNED_SLOPE_LABELS",
    "ROLLOUT_FORMAL_NUM_ENVS",
    "ROLLOUT_MANIFEST_SCHEMA_VERSION",
    "ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION",
    "ROLLOUT_STAGE_SEQUENCE",
    "ROLLOUT_PHYSICS_PARAMETER_NAMES",
    "FixedSeedValidationState",
    "S0FixedSeedValidationState",
    "S0_VALIDATION_STATE_KEY",
    "S2_VALIDATION_STATE_KEY",
    "TRAINING_CONFIGURATION_KEY",
    "TRAINING_CONFIGURATION_SCHEMA_VERSION",
    "TRAINING_ARTIFACT_INTERVAL",
    "TRAINING_THROUGHPUT_KEY",
    "CHECKPOINT_CURRICULUM_ITERATION_KEY",
    "CHECKPOINT_HASH_HISTORY_KEY",
    "CHECKPOINT_LINEAGE_KEY",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STAGE_KEY",
    "GUIDE_TRAINING_NUM_ENVS",
    "GUIDE_MAX_ITERATIONS",
    "GUIDE_TRAINING_PARAMETERS",
    "GUIDE_TRAINING_TASK",
    "build_s2_bootstrap_checkpoint",
    "checkpoint_stage",
    "checkpoint_hash_history",
    "collect_runtime_metadata",
    "extract_gaussian_actor_state",
    "extract_student_rsl_actor_state",
    "finalize_training_configuration",
    "feasibility_config_path",
    "install_runner_hooks_from_environment",
    "load_reward_calibration_report",
    "load_final_policy_acceptance_artifact",
    "load_policy_ablation_artifact",
    "load_s0_resume_checkpoint",
    "load_s2_resume_checkpoint",
    "load_stage_checkpoint",
    "load_fixed_seed_validation_score",
    "load_s0_fixed_seed_validation_score",
    "require_pinned_rsl_rl",
    "reset_runner_environment_for_curriculum",
    "reward_calibration_runtime_input_hashes",
    "runtime_config_files",
    "s0_remaining_learning_iterations",
    "s0_validation_early_stopped",
    "s2_remaining_learning_iterations",
    "fixed_seed_validation_early_stopped",
    "training_checkpoint_complete",
    "training_configuration_sha256",
    "validate_rollout_stage_coverage",
    "validate_guide_training_configuration",
    "validate_policy_ablation_run_lineage",
    "validate_training_configuration",
    "validate_training_throughput",
    "validate_s1_training_completion",
    "write_deployment_manifest",
]
