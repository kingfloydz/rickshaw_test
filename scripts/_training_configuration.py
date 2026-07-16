"""Canonical training-configuration records shared by the project launchers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from g1_rickshaw_lab.training_contract import (
    TRAINING_CONFIGURATION_KEY,
    TRAINING_CONFIGURATION_SCHEMA_VERSION,
    finalize_training_configuration,
    training_configuration_sha256,
    validate_training_configuration as _validate_training_configuration,
)

TRAINING_CONFIGURATION_ENV = "G1_RICKSHAW_TRAINING_CONFIGURATION"
TRAINING_CONFIGURATION_CHECKPOINT_KEY = TRAINING_CONFIGURATION_KEY

_FORMAL_OPERATIONAL_FLAGS = {
    "--headless",
    "--enable_cameras",
    "--verbose",
    "--info",
    "--video",
    "--export_io_descriptors",
}
_FORMAL_OPERATIONAL_OPTIONS = {
    "--livestream",
    "--device",
    "--rendering_mode",
    "--video_length",
    "--video_interval",
    "--run_name",
    "--experiment_name",
    "--seed",
    "--max_iterations",
}


def validate_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_training_configuration(value, require_formal=False)


def publish_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish the canonical mapping for the runner checkpoint hook."""

    normalized = validate_training_configuration(value)
    os.environ[TRAINING_CONFIGURATION_ENV] = json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return normalized


def build_training_configuration(
    *,
    stage: str,
    formal: bool,
    task: str,
    num_envs: int | None,
    seed: int,
    max_iterations: int,
    argv: Sequence[str],
    hydra_overrides: Sequence[str],
    guide_parameters: Mapping[str, Any],
    resolved_parameters: Mapping[str, Any],
    actor_initialized_from_teacher: bool | None,
    stage_coverage: Mapping[str, Any] | None,
    latent_dim: int,
    rollout_steps: int,
    fat2_weight: float,
    inputs_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the stable schema consumed by S2/bootstrap/ablation gates."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    if num_envs is not None and (
        isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0
    ):
        raise ValueError("training num_envs must be a positive integer or null")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if latent_dim <= 0 or rollout_steps <= 0:
        raise ValueError("latent_dim and rollout_steps must be positive")
    return finalize_training_configuration(
        {
            "schema_version": TRAINING_CONFIGURATION_SCHEMA_VERSION,
            "stage": stage,
            "formal": bool(formal),
            "task": str(task),
            "num_envs": num_envs,
            "seed": seed,
            "max_iterations": max_iterations,
            "argv": [str(item) for item in argv],
            "hydra_overrides": [str(item) for item in hydra_overrides],
            "guide_parameters": dict(guide_parameters),
            "resolved_parameters": dict(resolved_parameters),
            "actor_initialized_from_teacher": actor_initialized_from_teacher,
            "stage_coverage": None if stage_coverage is None else dict(stage_coverage),
            "ablation_values": {
                "fat2_weight": float(fat2_weight),
                "rollout_steps": int(rollout_steps),
                "latent_dim": int(latent_dim),
            },
            "inputs_sha256": dict(inputs_sha256),
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
    """Resolve a simple CLI/Hydra scalar for the pre-launch audit record."""

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


def validate_formal_launcher_arguments(arguments: Sequence[str]) -> None:
    """Reject untracked simulator, distributed, policy, and environment overrides."""

    index = 0
    while index < len(arguments):
        token = str(arguments[index])
        if token in _FORMAL_OPERATIONAL_FLAGS:
            index += 1
            continue
        matched = next(
            (
                option
                for option in _FORMAL_OPERATIONAL_OPTIONS
                if token == option or token.startswith(option + "=")
            ),
            None,
        )
        if matched is None:
            raise ValueError(f"formal training rejects unverified launcher argument {token!r}")
        if token == matched:
            if index + 1 >= len(arguments):
                raise ValueError(f"formal training option {matched} requires a value")
            value = str(arguments[index + 1])
            index += 2
        else:
            value = token.split("=", 1)[1]
            index += 1
        if matched == "--device" and not (
            value == "cuda" or value.startswith("cuda:")
        ):
            raise ValueError("formal training with 4096 environments requires a CUDA device")


__all__ = [
    "TRAINING_CONFIGURATION_ENV",
    "TRAINING_CONFIGURATION_CHECKPOINT_KEY",
    "TRAINING_CONFIGURATION_SCHEMA_VERSION",
    "build_training_configuration",
    "cli_value",
    "finalize_training_configuration",
    "publish_training_configuration",
    "training_configuration_sha256",
    "validate_formal_launcher_arguments",
    "validate_training_configuration",
]
