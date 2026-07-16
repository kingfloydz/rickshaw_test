"""Canonical training-configuration records shared by the project launchers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from typing import Any

from g1_rickshaw_lab.training_contract import (
    TRAINING_CONFIGURATION_KEY,
    TRAINING_CONFIGURATION_SCHEMA_VERSION,
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


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def training_configuration_sha256(value: Mapping[str, Any]) -> str:
    """Hash a training record while excluding its self-describing digest."""

    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def finalize_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-only, content-addressed training configuration."""

    payload = dict(value)
    payload.pop("content_sha256", None)
    if payload.get("schema_version") != TRAINING_CONFIGURATION_SCHEMA_VERSION:
        raise ValueError("training configuration has an unsupported schema_version")
    if not isinstance(payload.get("stage"), str) or not payload["stage"]:
        raise ValueError("training configuration requires a stage")
    if type(payload.get("formal")) is not bool:
        raise ValueError("training configuration formal must be boolean")
    # The round trip rejects tensors, Path objects, NaN/Inf, and other unstable values.
    normalized = json.loads(_canonical_json(payload).decode("ascii"))
    normalized["content_sha256"] = training_configuration_sha256(normalized)
    return normalized


def validate_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("training configuration must be a mapping")
    expected = value.get("content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("training configuration is missing content_sha256")
    try:
        int(expected, 16)
    except ValueError as exc:
        raise ValueError("training configuration content_sha256 is malformed") from exc
    if training_configuration_sha256(value) != expected.lower():
        raise ValueError("training configuration content_sha256 mismatch")
    required = {
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
    if set(value) != required:
        raise ValueError("training configuration has missing or unknown fields")
    if not isinstance(value.get("argv"), list) or not all(
        isinstance(item, str) for item in value["argv"]
    ):
        raise ValueError("training configuration argv must be a string list")
    if not isinstance(value.get("hydra_overrides"), list) or not all(
        isinstance(item, str) for item in value["hydra_overrides"]
    ):
        raise ValueError("training configuration hydra_overrides must be a string list")
    if not isinstance(value.get("guide_parameters"), Mapping):
        raise ValueError("training configuration guide_parameters must be a mapping")
    if not isinstance(value.get("resolved_parameters"), Mapping):
        raise ValueError("training configuration resolved_parameters must be a mapping")
    num_envs = value.get("num_envs")
    if num_envs is not None and (
        isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0
    ):
        raise ValueError("training configuration num_envs must be a positive integer or null")
    ablations = value.get("ablation_values")
    if not isinstance(ablations, Mapping) or set(ablations) != {
        "fat2_weight",
        "rollout_steps",
        "latent_dim",
    }:
        raise ValueError(
            "training configuration ablation_values must contain the exact three variants"
        )
    return finalize_training_configuration(value)


def publish_training_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish the canonical mapping for the runner checkpoint hook."""

    normalized = validate_training_configuration(value)
    os.environ[TRAINING_CONFIGURATION_ENV] = _canonical_json(normalized).decode("ascii")
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
