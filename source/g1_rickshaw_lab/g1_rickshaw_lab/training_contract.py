"""Training-stage checkpoint ABI and deployment bundle helpers.

This module intentionally has no Isaac Lab imports.  The command wrappers use
it before Kit starts, while the runner hook imports RSL-RL only after the
upstream training script has initialized its Python environment.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .artifact_io import write_json_atomic
from .configuration import FIXED_G1_JOINT_ORDER
from .policy_schema import (
    ACTION_DIM,
    ACTION_SCALE,
    ACTOR_OBSERVATION_DIM,
    BUTTERWORTH_A1,
    BUTTERWORTH_B0,
    BUTTERWORTH_B1,
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    SUPPORTED_CONTEXT_DIMS,
    validate_context_dim,
)
from .project_paths import CONFIG_ROOT, PROJECT_ROOT
from .provenance import (
    RSL_RL_VERSION,
    CheckpointMetadata,
    attach_checkpoint_metadata,
    collect_checkpoint_metadata,
    extract_checkpoint_metadata,
    load_checkpoint_with_validation,
)
from .reward_profile import (
    REWARD_WEIGHT_OVERRIDES_KEY,
    reward_weight_overrides_from_configuration,
)
from .slope_contract import (
    SLOPE_GRADIENTS,
    SLOPE_LABELS,
    balanced_slope_counts,
)
from .task_artifacts import load_task_artifacts

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STAGE_KEY = "g1_rickshaw_stage"
CHECKPOINT_LINEAGE_KEY = "g1_rickshaw_lineage"
CHECKPOINT_CURRICULUM_ITERATION_KEY = "g1_rickshaw_curriculum_iteration"
CHECKPOINT_STABILITY_REWARDS_ACTIVE_KEY = "g1_rickshaw_stability_rewards_active"
TRAINING_CONFIGURATION_KEY = "g1_rickshaw_training_configuration"
TRAINING_CONFIGURATION_SCHEMA_VERSION = 8
EXPECTED_RSL_RL_DISTRIBUTION_VERSION = RSL_RL_VERSION.removeprefix("v")

REPOSITORY_ROOT = PROJECT_ROOT
DEFAULT_FEASIBILITY_PATH = CONFIG_ROOT / "feasibility_envelope.yaml"
DEFAULT_RESET_POSES_PATH = CONFIG_ROOT / "reset_poses.yaml"
GUIDE_TRAINING_TASK = "Isaac-G1-Rickshaw-Directional-Slope-v0"
GUIDE_TRAINING_NUM_ENVS = 8192
TRAINING_ARTIFACT_INTERVAL = 200
S1_DETERMINISTIC_ALGORITHMS = False

SIGNED_SLOPE_LABELS = SLOPE_LABELS
ROLLOUT_MANIFEST_SCHEMA_VERSION = 4
ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION = 4
ROLLOUT_DEFAULT_NUM_ENVS = GUIDE_TRAINING_NUM_ENVS
DISTILLATION_ROLLOUT_STEPS = 64
ROLLOUT_STAGE_SEQUENCE = ("TRAINING",)
TRAINING_PARAMETER_KEYS = (
    "fat2_weight",
    "rollout_steps",
    "latent_dim",
    "stability_reward_curriculum",
)
DEFAULT_TRAINING_PARAMETERS = {
    "fat2_weight": 0.1,
    "rollout_steps": 48,
    "latent_dim": DEFAULT_CONTEXT_DIM,
    "stability_reward_curriculum": False,
}
SUPPORTED_FAT2_WEIGHTS = (0.0, 0.1, 0.2)
SUPPORTED_ROLLOUT_STEPS = (24, 48, 64)
TRAINING_CONFIGURATION_FIELDS = {
    "schema_version",
    "stage",
    "task",
    "num_envs",
    "seed",
    "max_iterations",
    "guide_parameters",
    "resolved_parameters",
    "actor_initialized_from_teacher",
    "stage_coverage",
    "training_parameters",
}
GUIDE_TRAINING_PARAMETERS = {
    "s0_teacher": {
        "domain_randomization": "startup_fixed",
        "terrain_slopes": "startup_balanced_fixed",
        "observation_noise": "unitree_g1_uniform",
    },
    "s1_context_distillation": {
        "context_learning_rate": 3.0e-4,
        "batch_size": 65536,
        "mini_batch_size": 8192,
        "gradient_clip": 1.0,
        "actor_initialized_from_teacher": True,
        "teacher_actor_initialization": True,
        "rollout_stage_sequence": list(ROLLOUT_STAGE_SEQUENCE),
        "validation_interval": TRAINING_ARTIFACT_INTERVAL,
        "deterministic_algorithms": S1_DETERMINISTIC_ALGORITHMS,
    },
    "s2_student_ppo": {
        "context_learning_rate": 1.0e-4,
        "actor_learning_rate": 3.0e-4,
        "critic_learning_rate": 3.0e-4,
        "context_encoder_frozen": False,
        "distillation_loss": False,
    },
}
GUIDE_MAX_ITERATIONS = {
    "s0_teacher": 4000,
    "s1_context_distillation": 3000,
    "s2_student_ppo": 2000,
}
BASELINE_ROLLOUT_STEPS = DEFAULT_TRAINING_PARAMETERS["rollout_steps"]


def rollout_scaled_iterations(baseline_iterations: int, rollout_steps: int) -> int:
    """Preserve the baseline per-environment transition budget."""

    if isinstance(baseline_iterations, bool) or not isinstance(baseline_iterations, int) or baseline_iterations <= 0:
        raise ValueError("baseline_iterations must be a positive integer")
    if type(rollout_steps) is not int or rollout_steps not in SUPPORTED_ROLLOUT_STEPS:
        raise ValueError(f"rollout_steps must be one of {SUPPORTED_ROLLOUT_STEPS}")
    iterations, remainder = divmod(baseline_iterations * BASELINE_ROLLOUT_STEPS, rollout_steps)
    if remainder:
        raise ValueError("rollout length does not divide the baseline transition budget")
    return iterations


def guide_max_iterations(stage: str, rollout_steps: int = BASELINE_ROLLOUT_STEPS) -> int:
    """Return the stage cap with equal PPO transition budgets."""

    try:
        baseline = GUIDE_MAX_ITERATIONS[stage]
    except KeyError as exc:
        raise ValueError(f"unknown training stage {stage!r}") from exc
    if stage == "s1_context_distillation":
        return baseline
    return rollout_scaled_iterations(baseline, rollout_steps)


def training_artifact_interval(rollout_steps: int) -> int:
    """Return the checkpoint interval at a fixed transition cadence."""

    return rollout_scaled_iterations(TRAINING_ARTIFACT_INTERVAL, rollout_steps)


def _canonical_training_configuration_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def finalize_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a JSON-only training configuration."""

    return json.loads(_canonical_training_configuration_json(value).decode("ascii"))


def build_training_configuration(
    *,
    stage: str,
    task: str,
    num_envs: int | None,
    seed: int,
    max_iterations: int,
    guide_parameters: Mapping[str, Any],
    resolved_parameters: Mapping[str, Any],
    actor_initialized_from_teacher: bool | None,
    stage_coverage: Mapping[str, Any] | None,
    fat2_weight: float = float(DEFAULT_TRAINING_PARAMETERS["fat2_weight"]),
    latent_dim: int = int(DEFAULT_TRAINING_PARAMETERS["latent_dim"]),
    rollout_steps: int = int(DEFAULT_TRAINING_PARAMETERS["rollout_steps"]),
    stability_reward_curriculum: bool = bool(DEFAULT_TRAINING_PARAMETERS["stability_reward_curriculum"]),
) -> dict[str, Any]:
    """Build the canonical configuration shared by every training stage."""

    return finalize_training_configuration(
        {
            "schema_version": TRAINING_CONFIGURATION_SCHEMA_VERSION,
            "stage": stage,
            "task": str(task),
            "num_envs": num_envs,
            "seed": seed,
            "max_iterations": max_iterations,
            "guide_parameters": dict(guide_parameters),
            "resolved_parameters": dict(resolved_parameters),
            "actor_initialized_from_teacher": actor_initialized_from_teacher,
            "stage_coverage": None if stage_coverage is None else dict(stage_coverage),
            "training_parameters": {
                "fat2_weight": fat2_weight,
                "rollout_steps": rollout_steps,
                "latent_dim": latent_dim,
                "stability_reward_curriculum": stability_reward_curriculum,
            },
        }
    )


def cli_value(
    arguments: Sequence[str],
    flag: str,
    *,
    hydra_keys: Sequence[str] = (),
    default: Any,
    cast,
) -> Any:
    """Resolve one scalar from CLI or Hydra-style overrides."""

    result = default
    index = 0
    while index < len(arguments):
        token = str(arguments[index])
        if token == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"{flag} requires a value")
            result = cast(arguments[index + 1])
            index += 2
            continue
        if token.startswith(flag + "="):
            result = cast(token.split("=", 1)[1])
        for key in hydra_keys:
            if token.startswith(key + "="):
                result = cast(token.split("=", 1)[1])
        index += 1
    return result


def validate_training_configuration(
    value: Any,
    *,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    """Validate the replayable CLI/Hydra configuration in a checkpoint."""

    if not isinstance(value, Mapping) or value.get("schema_version") != TRAINING_CONFIGURATION_SCHEMA_VERSION:
        raise ValueError(f"training configuration requires schema_version: {TRAINING_CONFIGURATION_SCHEMA_VERSION}")
    if set(value) != TRAINING_CONFIGURATION_FIELDS:
        raise ValueError("training configuration has missing or unknown fields")
    stage = value.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("training configuration is missing its stage")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(f"training configuration stage {stage!r} differs from {expected_stage!r}")
    seed = value.get("seed")
    iterations = value.get("max_iterations")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training configuration seed must be a non-negative integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("training configuration max_iterations must be positive")
    if not isinstance(value.get("task"), str) or not value["task"]:
        raise ValueError("training configuration task must be non-empty")
    num_envs = value.get("num_envs")
    if num_envs is not None and (isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0):
        raise ValueError("training configuration num_envs must be a positive integer or null")
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
    training_parameters = value.get("training_parameters")
    if not isinstance(training_parameters, Mapping) or set(training_parameters) != set(TRAINING_PARAMETER_KEYS):
        raise ValueError(f"training configuration training_parameters must contain exactly {TRAINING_PARAMETER_KEYS}")
    if type(training_parameters["rollout_steps"]) is not int:
        raise ValueError("rollout_steps must be an integer")
    if type(training_parameters["latent_dim"]) is not int:
        raise ValueError("latent_dim must be an integer")
    if type(training_parameters["stability_reward_curriculum"]) is not bool:
        raise ValueError("stability_reward_curriculum must be boolean")
    raw_fat2_weight = training_parameters["fat2_weight"]
    if isinstance(raw_fat2_weight, bool) or not isinstance(raw_fat2_weight, (int, float)):
        raise ValueError("fat2_weight must be numeric")
    normalized_parameters = {
        "fat2_weight": float(raw_fat2_weight),
        "rollout_steps": int(training_parameters["rollout_steps"]),
        "latent_dim": int(training_parameters["latent_dim"]),
        "stability_reward_curriculum": training_parameters["stability_reward_curriculum"],
    }
    if normalized_parameters["fat2_weight"] not in SUPPORTED_FAT2_WEIGHTS:
        raise ValueError(f"fat2_weight must be one of {SUPPORTED_FAT2_WEIGHTS}")
    if normalized_parameters["rollout_steps"] not in SUPPORTED_ROLLOUT_STEPS:
        raise ValueError(f"rollout_steps must be one of {SUPPORTED_ROLLOUT_STEPS}")
    validate_context_dim(normalized_parameters["latent_dim"])
    result = dict(value)
    result["training_parameters"] = normalized_parameters
    return finalize_training_configuration(result)


def validate_guide_training_configuration(
    value: Any,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    """Validate stage identity while preserving recorded training parameters."""

    result = validate_training_configuration(value, expected_stage=expected_stage)
    if expected_stage not in GUIDE_TRAINING_PARAMETERS:
        raise ValueError(f"no Guide training contract exists for stage {expected_stage!r}")
    if result["guide_parameters"] != GUIDE_TRAINING_PARAMETERS[expected_stage]:
        raise ValueError(f"training configuration guide parameters differ for stage {expected_stage!r}")
    return result


def s2_remaining_learning_iterations(
    *,
    requested_iterations: int,
    completed_iterations: int,
) -> int:
    """Return the additional S2 iterations needed to reach the training cap."""

    for name, value in (
        ("requested_iterations", requested_iterations),
        ("completed_iterations", completed_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if completed_iterations > requested_iterations:
        raise ValueError("S2 checkpoint iteration exceeds the requested training target")
    return requested_iterations - completed_iterations


def s0_remaining_learning_iterations(
    *,
    requested_iterations: int,
    completed_iterations: int,
) -> int:
    """Return the additional S0 iterations needed to reach the training cap."""

    for name, value in (
        ("requested_iterations", requested_iterations),
        ("completed_iterations", completed_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if completed_iterations > requested_iterations:
        raise ValueError("S0 checkpoint iteration exceeds the requested training target")
    return requested_iterations - completed_iterations


def validate_student_checkpoint_architecture(
    checkpoint: Mapping[str, Any],
    training_configuration: Mapping[str, Any],
) -> None:
    """Cross-check the recorded student latent width against model tensors."""

    latent_dim = int(training_configuration["training_parameters"]["latent_dim"])
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
        "context_encoder.context.weight",
        "encoder.context.weight",
    )
    policy_weight = tensor(
        "actor.network.0.weight",
        "policy.network.0.weight",
    )
    if latent_weight is None or latent_weight.ndim != 2 or latent_weight.shape[0] != latent_dim:
        raise ValueError("checkpoint context encoder differs from its recorded latent width")
    if policy_weight is None or policy_weight.ndim != 2 or policy_weight.shape[1] != ACTOR_OBSERVATION_DIM + latent_dim:
        raise ValueError("student actor input differs from its recorded latent width")


def validate_teacher_checkpoint_architecture(
    checkpoint: Mapping[str, Any],
    training_configuration: Mapping[str, Any],
) -> None:
    """Cross-check the recorded S0 encoder and policy widths."""

    latent_dim = int(training_configuration["training_parameters"]["latent_dim"])
    state = next(iter(_state_dicts(checkpoint)), None)
    if not isinstance(state, Mapping):
        raise ValueError("teacher checkpoint has no actor state_dict")
    encoder_weight = state.get("encoder.context.weight")
    policy_weight = state.get("policy.network.0.weight")
    if not torch.is_tensor(encoder_weight) or encoder_weight.ndim != 2 or encoder_weight.shape[0] != latent_dim:
        raise ValueError("teacher encoder differs from its recorded latent width")
    if (
        not torch.is_tensor(policy_weight)
        or policy_weight.ndim != 2
        or policy_weight.shape[1] != ACTOR_OBSERVATION_DIM + latent_dim
    ):
        raise ValueError("teacher actor input differs from its recorded latent width")


def validate_rollout_stage_coverage(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Validate the single reset-separated TRAINING rollout segment."""

    if manifest.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"rollout manifest requires schema_version {ROLLOUT_MANIFEST_SCHEMA_VERSION}")
    segments = manifest.get("stage_segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError("rollout manifest requires exactly one TRAINING segment")
    segment = segments[0]
    if not isinstance(segment, Mapping) or segment.get("global_stage") != "TRAINING":
        raise ValueError("rollout stage sequence must be exactly ('TRAINING',)")
    num_envs = manifest.get("num_envs")
    num_steps = manifest.get("num_steps_per_stage")
    if num_envs != ROLLOUT_DEFAULT_NUM_ENVS or num_steps != DISTILLATION_ROLLOUT_STEPS:
        raise ValueError(
            "S1 rollout budget must be exactly "
            f"{ROLLOUT_DEFAULT_NUM_ENVS} environments x "
            f"{DISTILLATION_ROLLOUT_STEPS} steps"
        )
    expected_slopes = list(SLOPE_GRADIENTS)
    if manifest.get("signed_slopes") != expected_slopes:
        raise ValueError(f"rollout manifest must contain exactly all {len(SLOPE_GRADIENTS)} slopes")
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
    expected_slope_samples = {label: count * num_steps for label, count in expected_environments.items()}
    if segment.get("slope_environment_distribution") != expected_environments:
        raise ValueError("TRAINING rollout lacks the balanced slope allocation")
    if segment.get("slope_sample_distribution") != expected_slope_samples:
        raise ValueError("TRAINING rollout lacks the exact slope sample quotas")
    episodes = segment.get("slope_episode_distribution")
    if (
        not isinstance(episodes, Mapping)
        or set(episodes) != set(SIGNED_SLOPE_LABELS)
        or any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in episodes.values())
    ):
        raise ValueError(f"TRAINING rollout lacks episode evidence for all {len(SLOPE_GRADIENTS)} slopes")

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
    """Return the canonical feasibility-envelope path."""

    return DEFAULT_FEASIBILITY_PATH.resolve()


def require_pinned_rsl_rl() -> str:
    """Fail before simulation when the pinned RSL-RL 5.x ABI is unavailable."""

    try:
        installed = importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "rsl-rl-lib is not installed in this Python environment; install "
            f"rsl-rl-lib=={EXPECTED_RSL_RL_DISTRIBUTION_VERSION}"
        ) from exc
    if installed != EXPECTED_RSL_RL_DISTRIBUTION_VERSION:
        raise RuntimeError(
            "incompatible RSL-RL runtime: installed "
            f"{installed}, required exactly {EXPECTED_RSL_RL_DISTRIBUTION_VERSION}"
        )
    try:
        from rsl_rl.algorithms import PPO  # noqa: F401
        from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config  # noqa: F401
        from rsl_rl.models import MLPModel  # noqa: F401
        from rsl_rl.runners import OnPolicyRunner
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("the installed rsl-rl-lib does not expose the pinned 5.0.1 actor/critic API") from exc
    load_parameters = inspect.signature(OnPolicyRunner.load).parameters
    if "load_cfg" not in load_parameters or "strict" not in load_parameters:
        raise RuntimeError("RSL-RL OnPolicyRunner.load does not match the pinned 5.0.1 ABI")
    return installed


def normalize_rsl_rl_runner_configuration(agent_cfg: Any) -> Any:
    """Apply Isaac Lab's required RSL-RL 5 configuration migration."""

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    return handle_deprecated_rsl_rl_cfg(agent_cfg, require_pinned_rsl_rl())


def collect_runtime_metadata() -> CheckpointMetadata:
    """Collect complete provenance after validating the pinned runtime version."""

    require_pinned_rsl_rl()
    return collect_checkpoint_metadata(joint_order=FIXED_G1_JOINT_ORDER)


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


def load_stage_checkpoint(
    path: str | Path,
    *,
    expected_stage: str | Iterable[str] | None = None,
    validate_runtime: bool = False,
) -> Mapping[str, Any]:
    kwargs: dict[str, Any] = {}
    if validate_runtime:
        metadata = collect_runtime_metadata()
        kwargs["expected"] = metadata
        kwargs["validate_torch_runtime"] = True
    checkpoint = dict(load_checkpoint_with_validation(path, **kwargs))
    loaded_stage = checkpoint_stage(checkpoint, expected_stage)
    curriculum_iteration = checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    if isinstance(curriculum_iteration, bool) or not isinstance(curriculum_iteration, int) or curriculum_iteration < 0:
        raise ValueError("checkpoint is missing a non-negative curriculum iteration")
    if loaded_stage in {"s0_teacher", "s2_student_ppo"}:
        iteration = checkpoint.get("iter")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("PPO checkpoint is missing a non-negative iteration")
    if loaded_stage in GUIDE_TRAINING_PARAMETERS:
        training_configuration = validate_guide_training_configuration(
            checkpoint.get(TRAINING_CONFIGURATION_KEY),
            expected_stage=loaded_stage,
        )
        checkpoint[TRAINING_CONFIGURATION_KEY] = training_configuration
        if loaded_stage in {
            "s1_context_distillation",
            "s2_student_ppo",
        }:
            validate_student_checkpoint_architecture(
                checkpoint,
                training_configuration,
            )
        if loaded_stage == "s0_teacher":
            validate_teacher_checkpoint_architecture(
                checkpoint,
                training_configuration,
            )
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

    return load_stage_checkpoint(
        path,
        expected_stage="s2_student_ppo",
        validate_runtime=validate_runtime,
    )


def load_s0_resume_checkpoint(
    path: str | Path,
    *,
    validate_runtime: bool = False,
) -> Mapping[str, Any]:
    """Load an intermediate or complete S0 checkpoint."""

    return load_stage_checkpoint(
        path,
        expected_stage="s0_teacher",
        validate_runtime=validate_runtime,
    )


def _state_dicts(checkpoint: Mapping[str, Any]) -> Iterable[Mapping[str, torch.Tensor]]:
    for key in ("actor_state_dict", "model_state_dict"):
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
    """Extract ``GaussianActor`` weights from the native S0 actor state."""

    required_suffixes = {"network.0.weight", "network.6.bias", "log_std"}
    state = checkpoint.get("actor_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("S0 checkpoint is missing actor_state_dict")
    candidate = _select_prefix(state, "policy.")
    if not required_suffixes.issubset(candidate):
        raise ValueError("S0 checkpoint does not contain the fixed Gaussian actor")
    return {key: value for key, value in candidate.items() if key.startswith("network.") or key == "log_std"}


def extract_student_rsl_actor_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert an S1 student checkpoint to the native RSL actor adapter layout."""

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("S1 checkpoint is missing model_state_dict")
    if "context_encoder.input.weight" not in state or "actor.network.0.weight" not in state:
        raise ValueError("S1 checkpoint does not contain the fixed student actor")
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("context_encoder."):
            result["encoder." + key.removeprefix("context_encoder.")] = value
        elif key.startswith("actor."):
            result["policy." + key.removeprefix("actor.")] = value
    return result


def build_s2_bootstrap_checkpoint(
    teacher_path: str | Path,
    context_path: str | Path,
) -> dict[str, Any]:
    """Build the load-only S2 checkpoint: S1 actor/context plus S0 critic."""

    teacher = load_stage_checkpoint(teacher_path, expected_stage="s0_teacher")
    context = load_stage_checkpoint(context_path, expected_stage="s1_context_distillation")
    teacher_training_configuration = teacher[TRAINING_CONFIGURATION_KEY]
    context_training_configuration = context[TRAINING_CONFIGURATION_KEY]
    training_parameters = dict(context_training_configuration["training_parameters"])
    if teacher_training_configuration["training_parameters"] != training_parameters:
        raise ValueError("S0 and S1 training parameters differ")
    teacher_reward_overrides = reward_weight_overrides_from_configuration(teacher_training_configuration)
    context_reward_overrides = reward_weight_overrides_from_configuration(context_training_configuration)
    if teacher_reward_overrides != context_reward_overrides:
        raise ValueError("S0 and S1 reward profiles differ")
    teacher_metadata = extract_checkpoint_metadata(teacher)
    context_metadata = extract_checkpoint_metadata(context)
    if teacher_metadata.to_mapping() != context_metadata.to_mapping():
        raise ValueError("S0 and S1 provenance differ; refusing to mix training lineages")
    critic_state = teacher.get("critic_state_dict")
    if not isinstance(critic_state, Mapping) or not critic_state:
        raise ValueError("S0 checkpoint does not contain the privileged critic state_dict")
    curriculum_iteration = teacher.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    context_lineage = context.get(CHECKPOINT_LINEAGE_KEY)
    if not isinstance(context_lineage, Mapping):
        raise ValueError("S1 checkpoint is missing its training lineage")
    teacher_file = Path(teacher_path).resolve()
    context_file = Path(context_path).resolve()
    if Path(str(context_lineage.get("teacher_checkpoint"))).resolve() != teacher_file:
        raise ValueError("S1 lineage teacher path differs from the supplied S0 checkpoint")
    context_iteration = context.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    if context_iteration != curriculum_iteration:
        raise ValueError("S1 and S0 curriculum lineages differ")
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STAGE_KEY: "s2_bootstrap",
        CHECKPOINT_CURRICULUM_ITERATION_KEY: curriculum_iteration,
        "actor_state_dict": extract_student_rsl_actor_state(context),
        "critic_state_dict": dict(critic_state),
        "iter": 0,
        "infos": {"load_optimizer": False, "load_iteration": False},
        TRAINING_CONFIGURATION_KEY: finalize_training_configuration(
            {
                "schema_version": TRAINING_CONFIGURATION_SCHEMA_VERSION,
                "stage": "s2_bootstrap",
                "task": GUIDE_TRAINING_TASK,
                "num_envs": GUIDE_TRAINING_NUM_ENVS,
                "seed": context_training_configuration["seed"],
                "max_iterations": guide_max_iterations("s2_student_ppo", int(training_parameters["rollout_steps"])),
                "guide_parameters": {
                    "source_stage": "s1_context_distillation",
                    "source_checkpoint": os.fspath(context_file),
                },
                "resolved_parameters": {
                    REWARD_WEIGHT_OVERRIDES_KEY: context_reward_overrides,
                },
                "actor_initialized_from_teacher": True,
                "stage_coverage": context_training_configuration["stage_coverage"],
                "training_parameters": training_parameters,
            }
        ),
        CHECKPOINT_LINEAGE_KEY: {
            "teacher_checkpoint": os.fspath(teacher_file),
            "context_checkpoint": os.fspath(context_file),
        },
    }
    attach_checkpoint_metadata(checkpoint, context_metadata)
    return checkpoint


def _deployment_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    raw_training_configuration = checkpoint.get(TRAINING_CONFIGURATION_KEY)
    if not isinstance(raw_training_configuration, Mapping):
        raise ValueError("deployment checkpoint has no training configuration")
    if raw_training_configuration.get("stage") != "s2_student_ppo":
        raise ValueError("deployment training configuration must be S2")
    training_configuration = dict(raw_training_configuration)
    latent_dim = int(training_configuration["training_parameters"]["latent_dim"])
    artifacts = load_task_artifacts(
        os.fspath(feasibility_config_path()),
        os.fspath(DEFAULT_RESET_POSES_PATH.resolve()),
    )
    calibration = artifacts.feasibility.calibration
    ranges = artifacts.feasibility.ranges
    command_ranges: dict[str, dict[str, Any]] = {}
    for name, source_name, unit in (
        ("acceleration_limit", "command.acceleration_limit", "m/s^2"),
        ("jerk_limit", "command.jerk_limit", "m/s^3"),
    ):
        interval = ranges.get(source_name)
        if interval is None:
            raise ValueError(f"feasibility envelope is missing ranges.{source_name}")
        minimum = float(interval.minimum)
        maximum = float(interval.maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0.0 or minimum > maximum:
            raise ValueError(f"feasibility envelope has invalid ranges.{source_name}")
        command_ranges[name] = {
            "min": minimum,
            "max": maximum,
            "unit": unit,
            "source": f"ranges.{source_name}",
        }
    safety = {key.removeprefix("safety."): value for key, value in calibration.items() if key.startswith("safety.")}
    return {
        "schema_version": 1,
        "policy": {
            "type": "deterministic_student_mean",
            "inputs": {
                "current": [None, ACTOR_OBSERVATION_DIM],
                "history": [None, HISTORY_LENGTH, ACTOR_OBSERVATION_DIM],
            },
            "context_dim": latent_dim,
            "output": {"normalized_action": [None, ACTION_DIM], "clip": [-1.0, 1.0]},
            "forbidden_components": ["teacher_encoder", "critic", "privileged_observations", "auxiliary_heads"],
        },
        "deployment_controller": {
            "artifact": "deployment_controller.pt",
            "stateless": True,
            "inputs": {
                "current": [None, ACTOR_OBSERVATION_DIM],
                "history": [None, HISTORY_LENGTH, ACTOR_OBSERVATION_DIM],
                "q_ref": [None, ACTION_DIM],
                "x_prev": [None, ACTION_DIM],
                "y_prev": [None, ACTION_DIM],
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
                {"name": "joint_position_minus_q_ref", "slice": [9, 38], "scale": [1.0] * ACTION_DIM},
                {"name": "joint_velocity", "slice": [38, 67], "scale": [0.05] * ACTION_DIM},
                {
                    "name": "previous_processed_action",
                    "slice": [67, ACTOR_OBSERVATION_DIM],
                    "scale": [1.0] * ACTION_DIM,
                },
            ],
        },
        "command": command_ranges,
        "action": {
            "joint_order": list(FIXED_G1_JOINT_ORDER),
            "scale_rad_per_normalized_action": list(ACTION_SCALE),
            "q_ref_by_gradient": [
                {"gradient": pose.gradient, "q_ref": list(pose.q_ref)} for pose in artifacts.reset_poses.poses
            ],
            "butterworth": {
                "sample_rate_hz": 50.0,
                "cutoff_hz": 4.0,
                "b0": BUTTERWORTH_B0,
                "b1": BUTTERWORTH_B1,
                "a1": BUTTERWORTH_A1,
                "equation": "y=b0*x+b1*x_prev-a1*y_prev",
                "reset_x_prev_and_y_prev_to_q_ref": True,
            },
        },
        "safety": {"persistent_steps": 10, "root_height_min_m": 0.31, **safety},
        "training_configuration": training_configuration,
    }


def write_deployment_manifest(export_dir: str | Path, checkpoint_path: str | Path) -> Path:
    """Write the runtime contract next to JIT/ONNX deployment artifacts."""

    export_path = Path(export_dir)
    checkpoint = _torch_load(checkpoint_path)
    checkpoint_stage(checkpoint, "s2_student_ppo")
    allowed_files = {
        "policy.pt",
        "policy.onnx",
        "deployment_controller.pt",
        "manifest.json",
    }
    unexpected = sorted(
        path.name for path in export_path.iterdir() if path.is_file() and path.name not in allowed_files
    )
    if unexpected:
        raise RuntimeError(f"deployment directory contains non-deployable artifacts: {unexpected}")
    required_files = {"policy.pt", "policy.onnx", "deployment_controller.pt"}
    missing = sorted(name for name in required_files if not (export_path / name).is_file())
    if missing:
        raise RuntimeError(
            "deployment directory is missing required controller/policy artifacts: " + ", ".join(missing)
        )
    manifest = _deployment_contract(checkpoint)
    manifest["source_checkpoint"] = os.fspath(Path(checkpoint_path).resolve())
    manifest["artifacts"] = sorted(required_files)
    destination = export_path / "manifest.json"
    write_json_atomic(destination, manifest)
    return destination


__all__ = [
    "BASELINE_ROLLOUT_STEPS",
    "DEFAULT_TRAINING_PARAMETERS",
    "DISTILLATION_ROLLOUT_STEPS",
    "SIGNED_SLOPE_LABELS",
    "ROLLOUT_DEFAULT_NUM_ENVS",
    "ROLLOUT_MANIFEST_SCHEMA_VERSION",
    "ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION",
    "S1_DETERMINISTIC_ALGORITHMS",
    "ROLLOUT_STAGE_SEQUENCE",
    "TRAINING_CONFIGURATION_KEY",
    "TRAINING_CONFIGURATION_SCHEMA_VERSION",
    "TRAINING_ARTIFACT_INTERVAL",
    "SUPPORTED_CONTEXT_DIMS",
    "SUPPORTED_FAT2_WEIGHTS",
    "SUPPORTED_ROLLOUT_STEPS",
    "CHECKPOINT_CURRICULUM_ITERATION_KEY",
    "CHECKPOINT_STABILITY_REWARDS_ACTIVE_KEY",
    "CHECKPOINT_LINEAGE_KEY",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STAGE_KEY",
    "GUIDE_TRAINING_NUM_ENVS",
    "GUIDE_MAX_ITERATIONS",
    "GUIDE_TRAINING_PARAMETERS",
    "GUIDE_TRAINING_TASK",
    "build_training_configuration",
    "build_s2_bootstrap_checkpoint",
    "checkpoint_stage",
    "cli_value",
    "collect_runtime_metadata",
    "extract_gaussian_actor_state",
    "extract_student_rsl_actor_state",
    "finalize_training_configuration",
    "feasibility_config_path",
    "guide_max_iterations",
    "load_s0_resume_checkpoint",
    "load_s2_resume_checkpoint",
    "load_stage_checkpoint",
    "normalize_rsl_rl_runner_configuration",
    "require_pinned_rsl_rl",
    "rollout_scaled_iterations",
    "s0_remaining_learning_iterations",
    "s2_remaining_learning_iterations",
    "training_artifact_interval",
    "validate_rollout_stage_coverage",
    "validate_guide_training_configuration",
    "validate_training_configuration",
    "validate_student_checkpoint_architecture",
    "validate_teacher_checkpoint_architecture",
    "write_deployment_manifest",
]
