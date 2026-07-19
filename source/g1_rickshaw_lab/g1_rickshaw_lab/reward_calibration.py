"""Guide section 11.2 reward calibration statistics and artifact helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .slope_contract import SLOPE_GRADIENTS


REWARD_CALIBRATION_SCHEMA_VERSION = 5
RAW_REWARD_SAMPLE_SCHEMA_VERSION = 5
RAW_REWARD_SAMPLE_KIND = "isaaclab_reward_manager_unweighted_terms"
SPEED_REFERENCE_TERM = "track_speed_exp"
SPEED_TERMS = (
    "track_speed_exp",
    "track_speed_precise_exp",
    "speed_error_pseudo_huber",
)
TERMINATION_TERM = "termination"
BALANCE_LIMIT_RATIO = 0.5
NORMAL_SAMPLE_DEFINITION = "post-RewardManager step; terminated=false; timeout=false"
SIGNED_C1_SLOPES = SLOPE_GRADIENTS

GUIDE_REWARD_TERMS = (
    "track_speed_exp",
    "track_speed_precise_exp",
    "speed_error_pseudo_huber",
    "lateral_error_l2",
    "heading_error_l2",
    "zmp_margin_barrier",
    "hitch_height_exp",
    "hitch_height_recovery_l2",
    "fat2_prior_exp",
    "feet_landing",
    "feet_air_time_excess_l2",
    "feet_slide",
    "terrain_normal_velocity_l2",
    "joint_power_l1",
    "processed_action_rate_l2",
    "hip_yaw_roll_reference_l2",
    "pelvis_height_limits_l2",
    "joint_position_limits",
    "termination",
)

GUIDE_PHYSICAL_SCALES = {
    "speed_error_sigma_mps": 0.5,
    "speed_precise_error_sigma_mps": 0.25,
    "speed_pseudo_huber_scale_mps": 0.5,
    "lateral_error_scale_m": 0.30,
    "heading_error_scale_rad": 0.30,
    "zmp_margin_m": 0.02,
    "hitch_height_error_sigma_m": 0.02,
    "hitch_height_recovery_deadband_m": 0.05,
    "hitch_height_recovery_scale_m": 0.05,
    "fat2_sigma_rad": 0.12,
    "processed_action_rate_normalizer": 1.0,
    "hip_yaw_roll_reference_scale_rad": 0.20,
    "pelvis_height_bounds_m": [0.58, 0.87],
    "pelvis_height_error_scale_m": 0.05,
    "feet_landing_target_air_time_s": 0.30,
    "feet_landing_sigma_s": 0.12,
    "feet_max_air_time_s": 0.50,
    "feet_air_time_excess_scale_s": 0.20,
    "feet_slide_normalizer_mps": 1.0,
    "terrain_normal_velocity_scale_mps": 0.25,
    "joint_power_normalizer_w": 1.0,
    "joint_limit_normalizer_rad": 1.0,
}

GUIDE_REWARD_NORMALIZATION_SCALES = {
    "track_speed_exp": {"scale": 0.5, "unit": "m/s"},
    "track_speed_precise_exp": {"scale": 0.25, "unit": "m/s"},
    "speed_error_pseudo_huber": {"scale": 0.5, "unit": "m/s"},
    "lateral_error_l2": {"scale": 0.30, "unit": "m"},
    "heading_error_l2": {"scale": 0.30, "unit": "rad"},
    "zmp_margin_barrier": {"scale": 0.02, "unit": "m"},
    "hitch_height_exp": {"scale": 0.02, "unit": "m"},
    "hitch_height_recovery_l2": {"scale": 0.05, "unit": "m"},
    "fat2_prior_exp": {"scale": 0.12, "unit": "rad"},
    "feet_landing": {"scale": 0.12, "unit": "s"},
    "feet_air_time_excess_l2": {"scale": 0.20, "unit": "s"},
    "feet_slide": {"scale": 1.0, "unit": "m/s"},
    "terrain_normal_velocity_l2": {"scale": 0.25, "unit": "m/s"},
    "joint_power_l1": {"scale": 1.0, "unit": "W"},
    "processed_action_rate_l2": {"scale": 1.0, "unit": "normalized_action"},
    "hip_yaw_roll_reference_l2": {"scale": 0.20, "unit": "rad"},
    "pelvis_height_limits_l2": {"scale": 0.05, "unit": "m"},
    "joint_position_limits": {"scale": 1.0, "unit": "rad"},
    "termination": {"scale": 1.0, "unit": "binary"},
}

C1_NOMINAL_PHYSICS_FIELDS = (
    "torso.mass_delta",
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
)


class RewardCalibrationError(ValueError):
    """Raised when reward samples do not satisfy the calibration contract."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def reward_calibration_runtime_versions() -> dict[str, str]:
    """Return the runtime package versions that affect collection semantics."""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency
        raise RewardCalibrationError("PyTorch is required for reward calibration") from error
    try:
        rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
        isaaclab_version = importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError as error:
        raise RewardCalibrationError(
            "Isaac Lab and RSL-RL distributions are required for reward calibration"
        ) from error
    return {
        "torch": str(torch.__version__),
        "rsl_rl": rsl_rl_version,
        "isaaclab": isaaclab_version,
    }


def validate_c1_physics_snapshot(
    snapshot: Mapping[str, Any],
    nominal_values: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1.0e-6,
) -> None:
    """Require every fixed-C1 runtime physics field to equal its declared nominal."""

    expected = set(C1_NOMINAL_PHYSICS_FIELDS)
    if not isinstance(snapshot, Mapping) or set(snapshot) != expected:
        actual = set(snapshot) if isinstance(snapshot, Mapping) else set()
        raise RewardCalibrationError(
            f"C1 physics snapshot fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    if not isinstance(nominal_values, Mapping) or set(nominal_values) != expected:
        actual = set(nominal_values) if isinstance(nominal_values, Mapping) else set()
        raise RewardCalibrationError(
            f"C1 nominal physics fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    tolerance = _finite_number(absolute_tolerance, "absolute_tolerance")
    if tolerance < 0.0:
        raise RewardCalibrationError("absolute_tolerance must be non-negative")
    for name in C1_NOMINAL_PHYSICS_FIELDS:
        bounds = snapshot[name]
        if not isinstance(bounds, Mapping) or set(bounds) != {"minimum", "maximum"}:
            raise RewardCalibrationError(f"c1_physics.{name} must contain exactly minimum/maximum")
        nominal = _finite_number(nominal_values[name], f"c1_nominal_values.{name}")
        minimum = _finite_number(bounds["minimum"], f"c1_physics.{name}.minimum")
        maximum = _finite_number(bounds["maximum"], f"c1_physics.{name}.maximum")
        if abs(minimum - nominal) > tolerance or abs(maximum - nominal) > tolerance:
            raise RewardCalibrationError(f"fixed C1 physical value {name!r} differs from nominal {nominal}")


def write_reward_calibration_json(
    output_dir: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically write the current reward-calibration report."""

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "reward_calibration.json"
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".reward_calibration.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RewardCalibrationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RewardCalibrationError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise RewardCalibrationError(f"{label} must be a finite number")
    return result


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    rank = (len(sorted_values) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def summarize_unweighted_samples(samples: Sequence[Any], *, term_name: str) -> dict[str, float | int]:
    if hasattr(samples, "detach") and hasattr(samples, "reshape"):
        samples = samples.detach().cpu().reshape(-1).tolist()
    values = sorted(_finite_number(value, f"samples.{term_name}[{index}]") for index, value in enumerate(samples))
    if not values:
        raise RewardCalibrationError(f"samples.{term_name} must not be empty")
    return {
        "count": len(values),
        "minimum": values[0],
        "p50": _linear_quantile(values, 0.50),
        "p90": _linear_quantile(values, 0.90),
        "p99": _linear_quantile(values, 0.99),
        "maximum": values[-1],
    }


def _validate_term_contract(
    raw_terms: Mapping[str, Sequence[Any]], term_weights: Mapping[str, Any]
) -> tuple[dict[str, float], int]:
    expected = set(GUIDE_REWARD_TERMS)
    sample_names = set(raw_terms)
    weight_names = set(term_weights)
    if sample_names != expected:
        raise RewardCalibrationError(
            f"raw reward terms differ from guide section 11.1: "
            f"missing={sorted(expected - sample_names)}, unknown={sorted(sample_names - expected)}"
        )
    if weight_names != expected:
        raise RewardCalibrationError(
            f"reward weights differ from guide section 11.1 terms: "
            f"missing={sorted(expected - weight_names)}, unknown={sorted(weight_names - expected)}"
        )
    weights = {name: _finite_number(term_weights[name], f"term_weights.{name}") for name in GUIDE_REWARD_TERMS}
    counts = {name: len(raw_terms[name]) for name in GUIDE_REWARD_TERMS}
    if any(count <= 0 for count in counts.values()):
        raise RewardCalibrationError("every reward term must contain normal C1 samples")
    if len(set(counts.values())) != 1:
        raise RewardCalibrationError(f"reward term sample counts differ: {counts}")
    return weights, next(iter(counts.values()))


def _calibrate_reward_terms_unstratified(
    raw_terms: Mapping[str, Sequence[Any]],
    term_weights: Mapping[str, Any],
    *,
    limit_ratio: float = BALANCE_LIMIT_RATIO,
) -> dict[str, Any]:
    """Compute guide quantiles and enforce the section 11.2 weighted-p90 cap."""

    ratio = _finite_number(limit_ratio, "limit_ratio")
    if not 0.0 < ratio < 1.0:
        raise RewardCalibrationError("limit_ratio must lie in (0, 1)")
    weights, sample_count = _validate_term_contract(raw_terms, term_weights)
    if weights[SPEED_REFERENCE_TERM] <= 0.0:
        raise RewardCalibrationError("track_speed_exp must retain a positive reference weight")
    summaries = {name: summarize_unweighted_samples(raw_terms[name], term_name=name) for name in GUIDE_REWARD_TERMS}
    speed_weighted_p90 = weights[SPEED_REFERENCE_TERM] * float(summaries[SPEED_REFERENCE_TERM]["p90"])
    if speed_weighted_p90 <= 0.0:
        raise RewardCalibrationError("track_speed_exp p90 must provide a positive calibration reference")
    speed_reference = abs(speed_weighted_p90)
    cap = ratio * speed_reference
    terms: dict[str, Any] = {}
    failures: list[str] = []
    for name in GUIDE_REWARD_TERMS:
        weight = weights[name]
        raw_p90 = float(summaries[name]["p90"])
        weighted_p90 = weight * raw_p90
        exempt_reason: str | None = None
        if name in SPEED_TERMS:
            exempt_reason = "speed_term"
        elif name == TERMINATION_TERM:
            exempt_reason = "guide_termination_exception"
        passed = True if exempt_reason is not None else abs(weighted_p90) <= cap + 1.0e-12
        if not passed:
            failures.append(name)
        raw_p90_abs = abs(raw_p90)
        max_abs_weight = None if raw_p90_abs == 0.0 else cap / raw_p90_abs
        recommended_weight = weight
        if exempt_reason is None and not passed and max_abs_weight is not None:
            recommended_weight = math.copysign(max_abs_weight, weight)
        terms[name] = {
            "weight": weight,
            "unweighted": summaries[name],
            "weighted_p90": weighted_p90,
            "weighted_abs_p90": abs(weighted_p90),
            "maximum_allowed_abs_p90": None if exempt_reason is not None else cap,
            "maximum_allowed_abs_weight": None if exempt_reason is not None else max_abs_weight,
            "recommended_weight_if_failed": recommended_weight,
            "exempt_reason": exempt_reason,
            "passed": passed,
        }
    return {
        "status": "passed" if not failures else "failed",
        "normal_sample_count": sample_count,
        "balance_rule": {
            "reference_term": SPEED_REFERENCE_TERM,
            "reference_weighted_p90": speed_weighted_p90,
            "reference_abs_p90": speed_reference,
            "limit_ratio": ratio,
            "maximum_other_term_abs_p90": cap,
            "exceptions": [TERMINATION_TERM],
        },
        "terms": terms,
        "failures": failures,
    }


def _normalized_slope_indices(sample_slope_indices: Sequence[Any], sample_count: int) -> list[int]:
    if hasattr(sample_slope_indices, "detach"):
        sample_slope_indices = sample_slope_indices.detach().cpu().reshape(-1).tolist()
    elif hasattr(sample_slope_indices, "reshape") and hasattr(sample_slope_indices, "tolist"):
        sample_slope_indices = sample_slope_indices.reshape(-1).tolist()
    if not isinstance(sample_slope_indices, Sequence) or isinstance(sample_slope_indices, (str, bytes)):
        raise RewardCalibrationError("sample_slope_indices must be a one-dimensional sequence")
    if len(sample_slope_indices) != sample_count:
        raise RewardCalibrationError("sample_slope_indices length differs from reward samples")
    result: list[int] = []
    for position, value in enumerate(sample_slope_indices):
        if isinstance(value, bool) or not isinstance(value, int):
            raise RewardCalibrationError(f"sample_slope_indices[{position}] must be an integer")
        if not 0 <= value < len(SIGNED_C1_SLOPES):
            raise RewardCalibrationError(
                f"sample_slope_indices[{position}] lies outside [0, {len(SIGNED_C1_SLOPES) - 1}]"
            )
        result.append(value)
    missing = [f"{SIGNED_C1_SLOPES[index]:+.2f}" for index in range(len(SIGNED_C1_SLOPES)) if index not in result]
    if missing:
        raise RewardCalibrationError(f"sample_slope_indices do not cover all fixed slopes: missing={missing}")
    return result


def calibrate_reward_terms(
    raw_terms: Mapping[str, Sequence[Any]],
    term_weights: Mapping[str, Any],
    *,
    sample_slope_indices: Sequence[Any] | None = None,
    limit_ratio: float = BALANCE_LIMIT_RATIO,
) -> dict[str, Any]:
    """Calibrate globally and, when labels are supplied, independently per slope."""

    result = _calibrate_reward_terms_unstratified(raw_terms, term_weights, limit_ratio=limit_ratio)
    if sample_slope_indices is None:
        return result

    sample_count = int(result["normal_sample_count"])
    indices = _normalized_slope_indices(sample_slope_indices, sample_count)
    normalized_terms = {
        name: (values.detach().cpu().reshape(-1).tolist() if hasattr(values, "detach") else list(values))
        for name, values in raw_terms.items()
    }
    per_slope: dict[str, dict[str, Any]] = {}
    slope_failures: dict[str, list[str]] = {}
    for slope_index, slope in enumerate(SIGNED_C1_SLOPES):
        selected = [position for position, value in enumerate(indices) if value == slope_index]
        slope_terms = {name: [values[position] for position in selected] for name, values in normalized_terms.items()}
        label = f"{slope:+.2f}"
        slope_result = _calibrate_reward_terms_unstratified(slope_terms, term_weights, limit_ratio=limit_ratio)
        per_slope[label] = slope_result
        if slope_result["failures"]:
            slope_failures[label] = list(slope_result["failures"])

    global_failures = list(result["failures"])
    all_failures = sorted(set(global_failures).union(term for failures in slope_failures.values() for term in failures))
    for name, term in result["terms"].items():
        slope_terms = {label: slope_result["terms"][name] for label, slope_result in per_slope.items()}
        allowed_weights = {
            label: slope_term["maximum_allowed_abs_weight"]
            for label, slope_term in slope_terms.items()
            if slope_term["maximum_allowed_abs_weight"] is not None
        }
        strictest_weight = min(allowed_weights.values()) if allowed_weights else None
        limiting_slope = min(allowed_weights, key=allowed_weights.__getitem__) if allowed_weights else None
        exceedance_ratios: dict[str, float] = {}
        for label, slope_term in slope_terms.items():
            cap = slope_term["maximum_allowed_abs_p90"]
            weighted = float(slope_term["weighted_abs_p90"])
            if cap is None:
                continue
            cap_value = float(cap)
            exceedance_ratios[label] = (
                weighted / cap_value if cap_value > 0.0 else (math.inf if weighted > 0.0 else 0.0)
            )
        worst_slope = max(exceedance_ratios, key=exceedance_ratios.__getitem__) if exceedance_ratios else None
        failing_slopes = [label for label, slope_term in slope_terms.items() if not slope_term["passed"]]
        term["global_passed"] = bool(term["passed"])
        term["passed"] = bool(term["passed"]) and not failing_slopes
        term["failing_slopes"] = failing_slopes
        term["worst_slope"] = worst_slope
        term["worst_slope_weighted_abs_p90"] = (
            None if worst_slope is None else float(slope_terms[worst_slope]["weighted_abs_p90"])
        )
        term["worst_slope_cap_exceedance_ratio"] = None if worst_slope is None else exceedance_ratios[worst_slope]
        term["limiting_slope"] = limiting_slope
        term["stratified_maximum_allowed_abs_weight"] = strictest_weight
        if strictest_weight is not None and abs(float(term["weight"])) > strictest_weight:
            term["recommended_weight_if_failed"] = math.copysign(strictest_weight, float(term["weight"]))

    result["global_status"] = result["status"]
    result["global_failures"] = global_failures
    result["per_slope"] = per_slope
    result["slope_failures"] = slope_failures
    result["failures"] = all_failures
    result["status"] = "passed" if not all_failures else "failed"
    result["stratified_balance"] = {
        "required_slopes": [f"{slope:+.2f}" for slope in SIGNED_C1_SLOPES],
        "all_slopes_passed": not slope_failures,
    }
    return result


def reward_manager_term_weights(reward_manager: Any) -> dict[str, float]:
    names = tuple(reward_manager.active_terms)
    if names != GUIDE_REWARD_TERMS:
        raise RewardCalibrationError(f"RewardManager active term order differs from guide: actual={names}")
    return {
        name: _finite_number(reward_manager.get_term_cfg(name).weight, f"RewardManager.{name}.weight") for name in names
    }


def collect_reward_manager_unweighted_step(reward_manager: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Recover current unweighted terms from Isaac Lab's RewardManager.

    The pinned Isaac Lab ABI computes ``value = raw * weight * dt`` and then
    stores ``value / dt`` in ``_step_reward``. Thus this buffer is a per-second
    weighted term, ``raw * weight``. The identity is checked against the
    integrated ``_reward_buf`` every step before division by weight. A
    zero-weight term is skipped by the manager, so its configured callable is
    evaluated directly against the same environment state.
    """

    names = tuple(reward_manager.active_terms)
    if names != GUIDE_REWARD_TERMS:
        raise RewardCalibrationError(f"RewardManager active term order differs from guide: actual={names}")
    step_reward = getattr(reward_manager, "_step_reward", None)
    if step_reward is None or getattr(step_reward, "ndim", None) != 2:
        raise RewardCalibrationError("RewardManager does not expose a two-dimensional _step_reward")
    if step_reward.shape[1] != len(names):
        raise RewardCalibrationError("RewardManager _step_reward width differs from active terms")
    reward_buffer = getattr(reward_manager, "_reward_buf", None)
    step_dt = _finite_number(
        getattr(getattr(reward_manager, "_env", None), "step_dt", None),
        "RewardManager._env.step_dt",
    )
    if step_dt <= 0.0 or reward_buffer is None or getattr(reward_buffer, "ndim", None) != 1:
        raise RewardCalibrationError("RewardManager integrated reward ABI is unavailable")
    if reward_buffer.shape[0] != step_reward.shape[0]:
        raise RewardCalibrationError("RewardManager reward buffers have inconsistent environment counts")
    expected_per_second = reward_buffer / step_dt
    residual = step_reward.sum(dim=1) - expected_per_second
    max_residual = float(residual.detach().abs().max().cpu())
    reference = float(expected_per_second.detach().abs().max().cpu())
    if max_residual > 1.0e-5 * (1.0 + reference):
        raise RewardCalibrationError("RewardManager _step_reward is not the pinned per-second weighted-term ABI")
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for index, name in enumerate(names):
        term_cfg = reward_manager.get_term_cfg(name)
        weight = _finite_number(term_cfg.weight, f"RewardManager.{name}.weight")
        if weight == 0.0:
            value = term_cfg.func(reward_manager._env, **term_cfg.params)
            source = "configured_term_callable_zero_weight"
        else:
            value = step_reward[:, index] / weight
            source = "RewardManager._step_reward_per_second_divided_by_weight"
        if getattr(value, "ndim", None) != 1 or value.shape[0] != step_reward.shape[0]:
            raise RewardCalibrationError(f"reward term {name!r} did not produce one scalar per environment")
        values[name] = value
        sources[name] = source
    return values, sources


def validate_raw_sample_artifact(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RAW_REWARD_SAMPLE_SCHEMA_VERSION:
        raise RewardCalibrationError("unsupported raw reward sample schema_version")
    if value.get("kind") != RAW_REWARD_SAMPLE_KIND:
        raise RewardCalibrationError("raw samples were not collected from Isaac Lab RewardManager")
    if value.get("reward_normalization_scales") != GUIDE_REWARD_NORMALIZATION_SCALES:
        raise RewardCalibrationError("raw samples use unknown reward normalization scales")
    if value.get("curriculum_stage") != "TRAINING":
        raise RewardCalibrationError("reward calibration samples must use TRAINING")
    fixed_seed = value.get("fixed_seed")
    if isinstance(fixed_seed, bool) or not isinstance(fixed_seed, int) or fixed_seed < 0:
        raise RewardCalibrationError("raw samples must record one fixed non-negative integer seed")
    slopes = value.get("fixed_slopes")
    if not isinstance(slopes, Sequence) or isinstance(slopes, (str, bytes)):
        raise RewardCalibrationError("raw samples must record the configured slope sequence")
    try:
        normalized_slopes = tuple(float(item) for item in slopes)
    except (TypeError, ValueError) as error:
        raise RewardCalibrationError("raw samples contain malformed fixed slopes") from error
    if normalized_slopes != SIGNED_C1_SLOPES:
        raise RewardCalibrationError("raw samples do not cover the exact configured training slopes")
    if value.get("normal_sample_definition") != NORMAL_SAMPLE_DEFINITION:
        raise RewardCalibrationError("raw samples use an unsupported normal-sample definition")
    raw_terms = value.get("raw_terms")
    weights = value.get("term_weights")
    if not isinstance(raw_terms, Mapping) or not isinstance(weights, Mapping):
        raise RewardCalibrationError("raw sample artifact must contain raw_terms and term_weights mappings")
    normalized_weights, sample_count = _validate_term_contract(raw_terms, weights)
    slope_counts = value.get("slope_sample_counts")
    expected_labels = {f"{slope:+.2f}" for slope in SIGNED_C1_SLOPES}
    if not isinstance(slope_counts, Mapping) or set(slope_counts) != expected_labels:
        raise RewardCalibrationError("raw samples must report a count for each fixed C1 slope")
    counts: list[int] = []
    for label in expected_labels:
        count = slope_counts[label]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RewardCalibrationError(f"slope_sample_counts.{label} must be a positive integer")
        counts.append(count)
    if len(set(counts)) != 1 or sum(counts) != sample_count:
        raise RewardCalibrationError("raw reward samples must be equally balanced over all configured slopes")
    sample_slope_indices = _normalized_slope_indices(value.get("sample_slope_indices"), sample_count)
    actual_counts = {f"{slope:+.2f}": sample_slope_indices.count(index) for index, slope in enumerate(SIGNED_C1_SLOPES)}
    if actual_counts != dict(slope_counts):
        raise RewardCalibrationError("sample_slope_indices do not match slope_sample_counts")
    term_sources = value.get("term_sources")
    if not isinstance(term_sources, Mapping) or set(term_sources) != set(GUIDE_REWARD_TERMS):
        raise RewardCalibrationError("raw samples must identify the extraction source of every reward term")
    for name in GUIDE_REWARD_TERMS:
        expected_source = (
            "configured_term_callable_zero_weight"
            if normalized_weights[name] == 0.0
            else "RewardManager._step_reward_per_second_divided_by_weight"
        )
        if term_sources[name] != expected_source:
            raise RewardCalibrationError(f"reward term {name!r} has an invalid extraction source")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RewardCalibrationError("raw samples must bind one policy checkpoint")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise RewardCalibrationError("raw samples must record the bound checkpoint path")
    runtime_versions = value.get("runtime_versions")
    if (
        not isinstance(runtime_versions, Mapping)
        or set(runtime_versions) != {"torch", "rsl_rl", "isaaclab"}
        or not all(isinstance(item, str) and item for item in runtime_versions.values())
    ):
        raise RewardCalibrationError("raw samples contain malformed runtime versions")
    validate_c1_physics_snapshot(value.get("c1_physics"), value.get("c1_nominal_values"))


def validate_sample_checkpoint_binding(value: Mapping[str, Any]) -> Path:
    """Require the raw-sample checkpoint path to remain available."""

    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RewardCalibrationError("raw samples must bind one policy checkpoint")
    raw_path = checkpoint.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RewardCalibrationError("raw samples must record the bound checkpoint path")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise RewardCalibrationError(f"bound reward-calibration checkpoint no longer exists: {path}")
    return path


def load_raw_reward_sample_artifact(path: str | Path, *, require_checkpoint_binding: bool = True) -> Mapping[str, Any]:
    """Load and fully validate one raw reward sample artifact."""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency
        raise RewardCalibrationError("PyTorch is required to load raw reward samples") from error
    sample_path = Path(path).resolve()
    if not sample_path.is_file():
        raise RewardCalibrationError(f"raw reward sample artifact does not exist: {sample_path}")
    value = torch.load(sample_path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise RewardCalibrationError("raw reward sample artifact must contain a mapping")
    validate_raw_sample_artifact(value)
    if require_checkpoint_binding:
        validate_sample_checkpoint_binding(value)
    return value


def recompute_reward_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly recompute all global and per-slope statistics from raw samples."""

    validate_raw_sample_artifact(value)
    return calibrate_reward_terms(
        value["raw_terms"],
        value["term_weights"],
        sample_slope_indices=value["sample_slope_indices"],
    )


_REPORT_SOURCE_FIELDS = (
    "created_at_utc",
    "curriculum_stage",
    "fixed_seed",
    "fixed_slopes",
    "slope_sample_counts",
    "normal_sample_definition",
    "task",
    "num_envs",
    "policy_steps",
    "step_dt_s",
    "rejected_samples",
    "policy_kind",
    "checkpoint",
    "c1_physics",
    "c1_nominal_values",
    "term_weights",
    "term_sources",
    "reward_normalization_scales",
    "runtime_versions",
)


def reward_sample_report_source(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the exact audited raw-sample metadata embedded in the JSON report."""

    validate_raw_sample_artifact(value)
    return {name: value[name] for name in _REPORT_SOURCE_FIELDS}


def reward_calibration_guide_contract() -> dict[str, Any]:
    return {
        "section": "11.2",
        "physical_scales": GUIDE_PHYSICAL_SCALES,
        "reward_normalization_scales": GUIDE_REWARD_NORMALIZATION_SCALES,
        "rule": (
            "global and every fixed slope independently require "
            "abs(weight * unweighted_p90) <= "
            "0.5 * abs(speed_weight * speed_unweighted_p90)"
        ),
        "termination_exempt": True,
    }


def load_and_recompute_reward_calibration_report(
    report_path: str | Path,
    *,
    teacher_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a report, reload its raw samples, and reject any unrecomputed claim."""

    path = Path(report_path).resolve()
    try:
        report = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise RewardCalibrationError(f"invalid reward calibration report: {path}") from error
    if not isinstance(report, Mapping):
        raise RewardCalibrationError("reward calibration report must contain a mapping")
    if report.get("schema_version") != REWARD_CALIBRATION_SCHEMA_VERSION or report.get("tool") != "calibrate_rewards":
        raise RewardCalibrationError("reward calibration report has an unsupported schema")
    raw_binding = report.get("raw_sample_artifact")
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != {"path"}:
        raise RewardCalibrationError("reward calibration report lacks its raw sample binding")
    raw_path = Path(str(raw_binding["path"])).resolve()
    if not raw_path.is_file():
        raise RewardCalibrationError("reward calibration raw sample artifact is missing")
    artifact = load_raw_reward_sample_artifact(raw_path)
    artifact_source = {name: artifact[name] for name in _REPORT_SOURCE_FIELDS}
    if report.get("source") != artifact_source:
        raise RewardCalibrationError("reward calibration report source differs from raw samples")
    if report.get("guide_contract") != reward_calibration_guide_contract():
        raise RewardCalibrationError("reward calibration report guide contract is malformed")
    calibration = calibrate_reward_terms(
        artifact["raw_terms"],
        artifact["term_weights"],
        sample_slope_indices=artifact["sample_slope_indices"],
    )
    if report.get("calibration") != calibration or report.get("status") != calibration["status"]:
        raise RewardCalibrationError("reward calibration statistics differ from raw recomputation")
    if teacher_checkpoint_path is not None:
        teacher_path = Path(teacher_checkpoint_path).resolve()
        checkpoint = artifact["checkpoint"]
        if (
            artifact.get("policy_kind") != "teacher"
            or checkpoint.get("stage") != "s0_teacher"
            or not teacher_path.is_file()
            or Path(str(checkpoint.get("path", ""))).resolve() != teacher_path
        ):
            raise RewardCalibrationError("reward calibration samples are not bound to the supplied S0 teacher")
    return {
        "report": report,
        "artifact": artifact,
        "calibration": calibration,
        "report_path": str(path),
        "raw_sample_path": str(raw_path),
    }


__all__ = [
    "C1_NOMINAL_PHYSICS_FIELDS",
    "BALANCE_LIMIT_RATIO",
    "GUIDE_PHYSICAL_SCALES",
    "GUIDE_REWARD_NORMALIZATION_SCALES",
    "GUIDE_REWARD_TERMS",
    "NORMAL_SAMPLE_DEFINITION",
    "RAW_REWARD_SAMPLE_KIND",
    "RAW_REWARD_SAMPLE_SCHEMA_VERSION",
    "REWARD_CALIBRATION_SCHEMA_VERSION",
    "RewardCalibrationError",
    "SIGNED_C1_SLOPES",
    "calibrate_reward_terms",
    "collect_reward_manager_unweighted_step",
    "load_and_recompute_reward_calibration_report",
    "load_raw_reward_sample_artifact",
    "recompute_reward_calibration",
    "reward_calibration_guide_contract",
    "reward_calibration_runtime_versions",
    "reward_manager_term_weights",
    "reward_sample_report_source",
    "utc_timestamp",
    "validate_raw_sample_artifact",
    "validate_c1_physics_snapshot",
    "validate_sample_checkpoint_binding",
    "write_reward_calibration_json",
]
