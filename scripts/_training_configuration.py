"""Canonical training-configuration records shared by the project launchers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from g1_rickshaw_lab.training_contract import (
    DEFAULT_TRAINING_PARAMETERS,
    TRAINING_CONFIGURATION_KEY,
    TRAINING_CONFIGURATION_SCHEMA_VERSION,
    finalize_training_configuration,
    validate_training_configuration as _validate_training_configuration,
)

TRAINING_CONFIGURATION_ENV = "G1_RICKSHAW_TRAINING_CONFIGURATION"
TRAINING_CONFIGURATION_CHECKPOINT_KEY = TRAINING_CONFIGURATION_KEY

def validate_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_training_configuration(value)


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
) -> dict[str, Any]:
    """Build the stable configuration shared by every mainline stage."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    if num_envs is not None and (
        isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0
    ):
        raise ValueError("training num_envs must be a positive integer or null")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
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


__all__ = [
    "TRAINING_CONFIGURATION_ENV",
    "TRAINING_CONFIGURATION_CHECKPOINT_KEY",
    "TRAINING_CONFIGURATION_SCHEMA_VERSION",
    "build_training_configuration",
    "cli_value",
    "finalize_training_configuration",
    "publish_training_configuration",
    "validate_training_configuration",
]
