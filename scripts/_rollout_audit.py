"""Pure CPU contracts for content-audited S1 teacher rollouts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

import torch

from g1_rickshaw_lab.slope_contract import (
    SLOPE_GRADIENTS,
    SLOPE_LABELS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)
from g1_rickshaw_lab.training_contract import (
    GUIDE_TRAINING_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
)


ACTION_DIM = 29
SIGNED_SLOPES = SLOPE_GRADIENTS
FORMAL_NUM_ENVS = GUIDE_TRAINING_NUM_ENVS

PHYSICS_VALUE_NAMES = (
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
)
AUDIT_TENSOR_NAMES = (
    "teacher_extrinsics",
    "curriculum_stage",
    "collection_segment",
    "environment_id",
    "episode_id",
    "slope",
    "terrain_level",
    "terrain_type",
    "physics_values",
    "joint_model_error",
    "observation_noise_scale",
)
INTEGER_AUDIT_TENSORS = frozenset(
    {
        "curriculum_stage",
        "collection_segment",
        "environment_id",
        "episode_id",
        "terrain_level",
        "terrain_type",
    }
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def formal_slope_environment_assignment(
    num_envs: int, *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Return the deterministic formal assignment over configured slopes."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs != FORMAL_NUM_ENVS:
        raise ValueError(f"formal rollout num_envs must equal {FORMAL_NUM_ENVS}")
    environment_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    slope_index = environment_ids.remainder(len(SIGNED_SLOPES))
    return {
        "slope_index": slope_index,
        "slope": torch.tensor(SIGNED_SLOPES, device=device)[slope_index],
        "terrain_level": torch.tensor(SLOPE_TERRAIN_LEVELS, device=device)[slope_index],
        "terrain_type": torch.tensor(SLOPE_TERRAIN_TYPES, device=device)[slope_index],
    }


def _column(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"rollout audit tensor {name} must have shape [N,1]")
    return value


def normalize_audit_tensors(
    raw: Mapping[str, Any], *, batch_size: int
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in AUDIT_TENSOR_NAMES:
        if name not in raw:
            raise ValueError(f"rollout shard is missing audit tensor {name!r}")
        tensor = torch.as_tensor(raw[name]).detach().cpu()
        if name in INTEGER_AUDIT_TENSORS:
            tensor = _column(tensor, name)
            if tensor.is_floating_point() and not torch.all(tensor == torch.round(tensor)):
                raise ValueError(f"rollout audit tensor {name} contains non-integer values")
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        if tensor.shape[0] != batch_size:
            raise ValueError(f"rollout audit tensor {name} has the wrong sample count")
        result[name] = tensor
    expected = {
        "teacher_extrinsics": (batch_size, len(PHYSICS_VALUE_NAMES)),
        "slope": (batch_size, 1),
        "physics_values": (batch_size, len(PHYSICS_VALUE_NAMES)),
        "joint_model_error": (batch_size, ACTION_DIM),
        "observation_noise_scale": (batch_size, 1),
    }
    for name, shape in expected.items():
        if tuple(result[name].shape) != shape or torch.any(~torch.isfinite(result[name])):
            raise ValueError(f"rollout audit tensor {name} must be finite with shape {shape}")
    return result


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.double().reshape(-1)
    if values.numel() == 0 or torch.any(~torch.isfinite(values)):
        raise ValueError("cannot summarize empty or non-finite rollout values")
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
    }


def _bounds(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain [minimum, maximum]")
    low, high = map(float, value)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{label} is invalid")
    return low, high


def _check_range(values: torch.Tensor, bounds: tuple[float, float], label: str) -> None:
    low, high = bounds
    tolerance = max(1.0e-6, abs(low) * 1.0e-6, abs(high) * 1.0e-6)
    if torch.any(values < low - tolerance) or torch.any(values > high + tolerance):
        raise ValueError(f"actual rollout samples for {label} lie outside configured range")


def _slope_indices(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    slopes = tensors["slope"].reshape(-1).double()
    levels = tensors["terrain_level"].reshape(-1)
    types = tensors["terrain_type"].reshape(-1)
    matches = (
        torch.isclose(slopes[:, None], torch.tensor(SIGNED_SLOPES)[None, :], atol=1.0e-7, rtol=0.0)
        & (levels[:, None] == torch.tensor(SLOPE_TERRAIN_LEVELS)[None, :])
        & (types[:, None] == torch.tensor(SLOPE_TERRAIN_TYPES)[None, :])
    )
    if torch.any(matches.sum(dim=1) != 1):
        raise ValueError("rollout slope evidence is not a canonical slope/terrain triple")
    return matches.long().argmax(dim=1)


def _slope_histogram(indices: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(indices, minlength=len(SIGNED_SLOPES))
    return {label: int(count) for label, count in zip(SLOPE_LABELS, counts, strict=True)}


def _episode_histogram(episode_ids: torch.Tensor, slope_indices: torch.Tensor) -> dict[str, int]:
    order = torch.argsort(episode_ids.reshape(-1), stable=True)
    episodes = episode_ids.reshape(-1)[order]
    slopes = slope_indices[order]
    first = torch.ones_like(episodes, dtype=torch.bool)
    first[1:] = episodes[1:] != episodes[:-1]
    if torch.any(~first[1:] & (slopes[1:] != slopes[:-1])):
        raise ValueError("one rollout episode spans more than one slope")
    return _slope_histogram(slopes[first])


def _validate_episode_binding(tensors: Mapping[str, torch.Tensor]) -> int:
    episodes = tensors["episode_id"].reshape(-1)
    if episodes.numel() == 0 or torch.any(episodes < 0):
        raise ValueError("rollout episode_id must be non-negative")
    order = torch.argsort(episodes, stable=True)
    same = episodes[order][1:] == episodes[order][:-1]
    for name in AUDIT_TENSOR_NAMES:
        if name == "episode_id":
            continue
        values = tensors[name][order].reshape(episodes.numel(), -1)
        equal = (
            torch.isclose(values[1:], values[:-1], atol=1.0e-6, rtol=1.0e-6).all(dim=1)
            if values.is_floating_point()
            else (values[1:] == values[:-1]).all(dim=1)
        )
        if torch.any(same & ~equal):
            raise ValueError(f"one episode changes {name} without a reset")
    return int(torch.unique(episodes).numel())


def summarize_segment_samples(
    tensors: Mapping[str, torch.Tensor],
    *,
    segment_index: int,
    num_envs: int,
    samples_per_environment: int,
    physics_bounds: Mapping[str, Any],
    joint_model_error_bounds: Any,
) -> dict[str, Any]:
    """Validate and summarize the single TRAINING segment from stored samples."""

    if segment_index != 0:
        raise ValueError("collection segment must be 0")
    assignment = formal_slope_environment_assignment(num_envs)
    expected_samples = num_envs * samples_per_environment
    if tensors["environment_id"].shape[0] != expected_samples:
        raise ValueError("TRAINING segment has the wrong sample count")
    if not torch.all(tensors["collection_segment"] == 0):
        raise ValueError("rollout contains a non-TRAINING collection segment")
    if not torch.all(tensors["curriculum_stage"] == 1):
        raise ValueError("rollout contains a non-TRAINING curriculum stage")

    environment_ids = tensors["environment_id"].reshape(-1)
    if torch.any(environment_ids < 0) or torch.any(environment_ids >= num_envs):
        raise ValueError("rollout environment_id is out of range")
    if not torch.all(torch.bincount(environment_ids, minlength=num_envs) == samples_per_environment):
        raise ValueError("every rollout environment must contribute the same quota")
    slope_indices = _slope_indices(tensors)
    expected_environment_slopes = _slope_histogram(assignment["slope_index"])
    expected_sample_slopes = {
        label: count * samples_per_environment
        for label, count in expected_environment_slopes.items()
    }
    if _slope_histogram(slope_indices) != expected_sample_slopes:
        raise ValueError("rollout lacks the deterministic balanced slope allocation")

    raw = tensors["physics_values"]
    normalized = []
    for index, name in enumerate(PHYSICS_VALUE_NAMES):
        low, high = _bounds(physics_bounds.get(name), f"physics_bounds.{name}")
        _check_range(raw[:, index], (low, high), name)
        normalized.append(
            torch.zeros_like(raw[:, index])
            if high == low
            else torch.clamp(2.0 * (raw[:, index] - low) / (high - low) - 1.0, -1.0, 1.0)
        )
    expected_extrinsics = torch.stack(normalized, dim=-1)
    if not torch.allclose(tensors["teacher_extrinsics"], expected_extrinsics, atol=2.0e-3, rtol=1.0e-3):
        raise ValueError("teacher_extrinsics do not match raw episode physics")
    _check_range(tensors["joint_model_error"], _bounds(joint_model_error_bounds, "joint_model_error_bounds"), "joint.model_error")
    episodes = _validate_episode_binding(tensors)

    summary = {
        "segment_index": 0,
        "samples": expected_samples,
        "episodes": episodes,
        "stage_distribution": {"TRAINING": expected_samples},
        "environment_stage_distribution": {"TRAINING": num_envs},
        "slope_sample_distribution": expected_sample_slopes,
        "slope_environment_distribution": expected_environment_slopes,
        "slope_episode_distribution": _episode_histogram(tensors["episode_id"], slope_indices),
        "terrain_level_distribution": {
            str(int(value)): int(count)
            for value, count in zip(*torch.unique(tensors["terrain_level"], return_counts=True), strict=True)
        },
        "terrain_type_distribution": {
            str(int(value)): int(count)
            for value, count in zip(*torch.unique(tensors["terrain_type"], return_counts=True), strict=True)
        },
        "physics_distribution": {
            name: _distribution(raw[:, index]) for index, name in enumerate(PHYSICS_VALUE_NAMES)
        },
        "observation_noise_scale": _distribution(tensors["observation_noise_scale"]),
    }
    summary["content_sha256"] = canonical_sha256(summary)
    return summary


def validate_rollout_sample_audit(
    manifest: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Recompute and bind the single-segment rollout audit."""

    if manifest.get("schema_version") != ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"formal S1 requires manifest schema {ROLLOUT_MANIFEST_SCHEMA_VERSION}")
    audit = manifest.get("sample_audit")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION:
        raise ValueError("rollout manifest is missing its sample audit")
    if tuple(audit.get("physics_value_names", ())) != PHYSICS_VALUE_NAMES:
        raise ValueError("sample audit physics ABI differs")
    if (
        tuple(float(value) for value in audit.get("signed_slopes", ())) != SIGNED_SLOPES
        or tuple(int(value) for value in audit.get("terrain_levels", ())) != SLOPE_TERRAIN_LEVELS
        or tuple(int(value) for value in audit.get("terrain_types", ())) != SLOPE_TERRAIN_TYPES
        or audit.get("formal_num_envs") != FORMAL_NUM_ENVS
    ):
        raise ValueError("sample audit slope/terrain ABI differs")
    physics_bounds = audit.get("physics_bounds")
    if not isinstance(physics_bounds, Mapping) or set(physics_bounds) != set(PHYSICS_VALUE_NAMES):
        raise ValueError("sample audit has incomplete physics bounds")
    segments = manifest.get("stage_segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError("formal S1 requires exactly one TRAINING segment")
    for name in AUDIT_TENSOR_NAMES:
        if name not in tensors:
            raise ValueError(f"rollout is missing audit tensor {name!r}")
    summary = summarize_segment_samples(
        tensors,
        segment_index=0,
        num_envs=int(manifest.get("num_envs", 0)),
        samples_per_environment=int(manifest.get("num_steps_per_stage", 0)),
        physics_bounds=physics_bounds,
        joint_model_error_bounds=audit.get("joint_model_error_bounds"),
    )
    segment = segments[0]
    if segment.get("global_stage") != "TRAINING" or segment.get("actual_sample_audit") != summary:
        raise ValueError("TRAINING segment manifest differs from shard samples")
    bindings = {
        "slope_sample_distribution": summary["slope_sample_distribution"],
        "slope_environment_distribution": summary["slope_environment_distribution"],
        "slope_episode_distribution": summary["slope_episode_distribution"],
    }
    if any(manifest.get(name) != value for name, value in bindings.items()):
        raise ValueError("rollout slope aggregates differ from shard samples")
    compact = {
        "manifest_schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "sample_audit_schema_version": ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
        "sample_audit_sha256": canonical_sha256({"sample_audit": dict(audit), "segments": [summary]}),
        "signed_slopes": list(SIGNED_SLOPES),
        **bindings,
        "stages": {
            "TRAINING": {
                "samples": summary["samples"],
                "episodes": summary["episodes"],
                "stage_distribution": summary["stage_distribution"],
                **bindings,
                "content_sha256": summary["content_sha256"],
            }
        },
    }
    if manifest.get("sample_audit_sha256") != compact["sample_audit_sha256"]:
        raise ValueError("sample_audit_sha256 differs from shard samples")
    return compact


__all__ = [
    "ACTION_DIM",
    "AUDIT_TENSOR_NAMES",
    "FORMAL_NUM_ENVS",
    "INTEGER_AUDIT_TENSORS",
    "PHYSICS_VALUE_NAMES",
    "ROLLOUT_MANIFEST_SCHEMA_VERSION",
    "ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION",
    "SIGNED_SLOPES",
    "SLOPE_LABELS",
    "SLOPE_TERRAIN_LEVELS",
    "SLOPE_TERRAIN_TYPES",
    "canonical_sha256",
    "formal_slope_environment_assignment",
    "normalize_audit_tensors",
    "summarize_segment_samples",
    "validate_rollout_sample_audit",
]
