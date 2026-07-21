"""Pure configuration and ranking logic for reward-weight tuning."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import yaml

from .reward_profile import (
    FAT2_REWARD_TERM,
    GUIDE_REWARD_TERMS,
    validate_reward_weight_overrides,
)
from .slope_contract import SLOPE_GRADIENTS

REWARD_TUNING_SCHEMA_VERSION = 1
COST_METRICS = (
    "worst_slope_fall_rate",
    "fall_rate",
    "speed_rmse_mps",
    "overspeed_rate",
    "lateral_error_rms_m",
    "heading_error_rms_rad",
    "hitch_height_error_rms_m",
    "foot_slip_p90_mps",
    "processed_action_rate_p90_radps",
    "power_p90_w",
)
BENEFIT_METRICS = (
    "zmp_margin_p01_m",
    "two_wheel_contact_rate",
)
RANK_METRICS = (
    "worst_slope_fall_rate",
    "fall_rate",
    "speed_rmse_mps",
    "overspeed_rate",
    "lateral_error_rms_m",
    "heading_error_rms_rad",
    "zmp_margin_p01_m",
    "two_wheel_contact_rate",
    "hitch_height_error_rms_m",
    "foot_slip_p90_mps",
    "processed_action_rate_p90_radps",
    "power_p90_w",
)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_reward_tuning_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != REWARD_TUNING_SCHEMA_VERSION:
        raise ValueError("reward tuning configuration has an unsupported schema")
    if not isinstance(raw.get("task"), str) or not raw["task"]:
        raise ValueError("reward tuning task must be non-empty")
    fixed = raw.get("fixed")
    if not isinstance(fixed, Mapping) or set(fixed) != {
        "num_envs",
        "rollout_steps",
        "latent_dim",
        "fat2_weight",
        "max_iterations",
    }:
        raise ValueError("reward tuning fixed configuration is incomplete")
    parsed_fixed = {
        "num_envs": _positive_integer(fixed["num_envs"], "fixed.num_envs"),
        "rollout_steps": _positive_integer(
            fixed["rollout_steps"], "fixed.rollout_steps"
        ),
        "latent_dim": _positive_integer(fixed["latent_dim"], "fixed.latent_dim"),
        "fat2_weight": _number(fixed["fat2_weight"], "fixed.fat2_weight"),
        "max_iterations": _positive_integer(
            fixed["max_iterations"], "fixed.max_iterations"
        ),
    }
    factors = raw.get("factors")
    if not isinstance(factors, Mapping) or len(factors) != 3:
        raise ValueError("reward tuning requires exactly three two-level factors")
    parsed_factors: dict[str, dict[str, Any]] = {}
    terms: set[str] = set()
    labels: set[str] = set()
    for factor_name, factor in factors.items():
        if not isinstance(factor_name, str) or not isinstance(factor, Mapping) or set(factor) != {
            "term",
            "low",
            "high",
            "high_label",
        }:
            raise ValueError(f"invalid reward factor {factor_name!r}")
        term = factor["term"]
        label = factor["high_label"]
        if term not in GUIDE_REWARD_TERMS or term == FAT2_REWARD_TERM:
            raise ValueError(f"factor {factor_name!r} has an unsupported reward term")
        if term in terms:
            raise ValueError("reward tuning factors must use distinct terms")
        if (
            not isinstance(label, str)
            or not re.fullmatch(r"[a-z0-9_]+", label)
            or label in labels
        ):
            raise ValueError(f"factor {factor_name!r} has an invalid high_label")
        low = _number(factor["low"], f"factors.{factor_name}.low")
        high = _number(factor["high"], f"factors.{factor_name}.high")
        if low == high:
            raise ValueError("factor levels must differ")
        parsed_factors[factor_name] = {
            "term": term,
            "low": low,
            "high": high,
            "high_label": label,
        }
        terms.add(term)
        labels.add(label)

    fixed_reward_weights = validate_reward_weight_overrides(
        raw.get("fixed_reward_weights", {})
    )
    overlap = terms.intersection(fixed_reward_weights)
    if overlap:
        raise ValueError(
            f"fixed reward weights overlap tuned factors: {sorted(overlap)}"
        )

    calibration = raw.get("calibration")
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "samples_per_slope",
        "max_policy_steps",
    }:
        raise ValueError("reward tuning calibration configuration is incomplete")
    parsed_calibration = {
        "samples_per_slope": _positive_integer(
            calibration["samples_per_slope"], "calibration.samples_per_slope"
        ),
        "max_policy_steps": _positive_integer(
            calibration["max_policy_steps"], "calibration.max_policy_steps"
        ),
    }
    parsed = {
        "schema_version": REWARD_TUNING_SCHEMA_VERSION,
        "task": raw["task"],
        "fixed": parsed_fixed,
        "factors": parsed_factors,
        "calibration": parsed_calibration,
    }
    if "fixed_reward_weights" in raw:
        parsed["fixed_reward_weights"] = fixed_reward_weights
    return parsed


def factorial_reward_profiles(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    factors = list(config["factors"].items())
    fixed_reward_weights = config.get("fixed_reward_weights", {})
    profiles: list[dict[str, Any]] = []
    for high_count in range(len(factors) + 1):
        for high_indices in combinations(range(len(factors)), high_count):
            high_set = set(high_indices)
            name = (
                "baseline"
                if not high_set
                else "_".join(factors[index][1]["high_label"] for index in high_indices)
            )
            levels = {
                factor_name: ("high" if index in high_set else "low")
                for index, (factor_name, _) in enumerate(factors)
            }
            overrides = dict(fixed_reward_weights)
            overrides.update(
                {
                    factor["term"]: factor[levels[factor_name]]
                    for factor_name, factor in factors
                }
            )
            profiles.append(
                {
                    "name": name,
                    "levels": levels,
                    "reward_weight_overrides": validate_reward_weight_overrides(
                        overrides
                    ),
                }
            )
    return profiles


def _path(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"diagnostic report is missing {'.'.join(keys)}")
        value = value[key]
    return value


def policy_diagnostic_rank_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    if report.get("status") != "recorded":
        raise ValueError("policy diagnostic report is incomplete")
    stage = _path(report, "stages", "training")
    metrics = _path(stage, "metrics")
    non_finite = _path(metrics, "non_finite_sample_counts")
    if not isinstance(non_finite, Mapping) or any(int(value) != 0 for value in non_finite.values()):
        raise ValueError("policy diagnostic contains non-finite samples")
    per_slope = _path(stage, "per_slope")
    expected_slopes = {f"{slope:+.2f}" for slope in SLOPE_GRADIENTS}
    if not isinstance(per_slope, Mapping) or set(per_slope) != expected_slopes:
        raise ValueError("policy diagnostic does not contain the exact training slopes")
    worst_fall = max(
        _number(_path(value, "episodes", "fall_rate"), "per-slope fall rate")
        for value in per_slope.values()
    )
    result = {
        "worst_slope_fall_rate": worst_fall,
        "fall_rate": _number(_path(metrics, "episodes", "fall_rate"), "fall rate"),
        "speed_rmse_mps": _number(_path(metrics, "tracking", "speed_rmse_mps"), "speed RMSE"),
        "overspeed_rate": _number(_path(metrics, "tracking", "overspeed_rate"), "overspeed rate"),
        "lateral_error_rms_m": _number(_path(metrics, "tracking", "lateral_error", "rms_m"), "lateral RMS"),
        "heading_error_rms_rad": _number(_path(metrics, "tracking", "heading_error", "rms_rad"), "heading RMS"),
        "zmp_margin_p01_m": _number(_path(metrics, "stability", "zmp_margin_m", "p01"), "ZMP p01"),
        "two_wheel_contact_rate": _number(_path(metrics, "rickshaw", "two_wheel_contact_rate"), "two-wheel contact rate"),
        "hitch_height_error_rms_m": _number(_path(metrics, "rickshaw", "hitch_height_error", "rms_m"), "hitch-height RMS"),
        "foot_slip_p90_mps": _number(_path(metrics, "locomotion", "foot_slip_mps", "p90"), "foot-slip p90"),
        "processed_action_rate_p90_radps": _number(_path(metrics, "actions", "processed_rate_radps", "p90"), "action-rate p90"),
        "power_p90_w": _number(_path(metrics, "actuation", "power_w", "p90"), "power p90"),
    }
    return result


def aggregate_profile_results(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["profile"]), []).append(record)
    aggregates: list[dict[str, Any]] = []
    for profile, items in grouped.items():
        metric_values = {
            name: [_number(item["metrics"][name], name) for item in items]
            for name in RANK_METRICS
        }
        summary = {
            name: {
                "mean": fmean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in metric_values.items()
        }
        robust = {
            name: (max(values) if name in COST_METRICS else min(values))
            for name, values in metric_values.items()
        }
        calibration_passed = all(item.get("calibration_status") == "passed" for item in items)
        rank_key = [0.0 if calibration_passed else 1.0]
        rank_key.extend(
            -robust[name] if name in BENEFIT_METRICS else robust[name]
            for name in RANK_METRICS
        )
        aggregates.append(
            {
                "profile": profile,
                "training_seeds": sorted(int(item["training_seed"]) for item in items),
                "calibration_passed": calibration_passed,
                "metrics": summary,
                "robust_metrics": robust,
                "rank_key": rank_key,
            }
        )
    aggregates.sort(key=lambda item: item["rank_key"])
    for index, item in enumerate(aggregates, start=1):
        item["rank"] = index
    return aggregates


def factorial_effects(
    aggregates: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    factors: Mapping[str, Any],
) -> dict[str, Any]:
    profile_names = {str(profile["name"]) for profile in profiles}
    aggregate_names = {str(item["profile"]) for item in aggregates}
    if (
        len(profiles) != 2 ** len(factors)
        or len(aggregates) != len(profiles)
        or aggregate_names != profile_names
        or any(len(item["training_seeds"]) != 1 for item in aggregates)
    ):
        return {}
    aggregate_by_name = {str(item["profile"]): item for item in aggregates}
    factor_names = list(factors)
    profile_by_name = {item["name"]: item for item in profiles}
    effects: dict[str, Any] = {}
    for size in range(1, len(factor_names) + 1):
        for selected in combinations(factor_names, size):
            label = ":".join(selected)
            metric_effects: dict[str, float] = {}
            for metric in RANK_METRICS:
                signed_sum = 0.0
                for profile_name, profile in profile_by_name.items():
                    sign = math.prod(
                        1.0 if profile["levels"][factor] == "high" else -1.0
                        for factor in selected
                    )
                    signed_sum += sign * aggregate_by_name[profile_name]["metrics"][metric]["mean"]
                metric_effects[metric] = signed_sum / (2 ** (len(factor_names) - 1))
            effects[label] = metric_effects
    return {
        "definition": "high-level mean minus low-level mean; negative improves cost metrics and positive improves benefit metrics",
        "cost_metrics": list(COST_METRICS),
        "benefit_metrics": list(BENEFIT_METRICS),
        "effects": effects,
    }


__all__ = [
    "BENEFIT_METRICS",
    "COST_METRICS",
    "RANK_METRICS",
    "REWARD_TUNING_SCHEMA_VERSION",
    "aggregate_profile_results",
    "factorial_effects",
    "factorial_reward_profiles",
    "load_reward_tuning_config",
    "policy_diagnostic_rank_metrics",
]
