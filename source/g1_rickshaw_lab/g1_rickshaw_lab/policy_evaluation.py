"""Pure policy-acceptance aggregation and artifact contracts.

The simulator-facing runner lives in ``scripts/evaluate_policy.py``.  Keeping
the reductions here free of Isaac Lab imports makes every reported acceptance
number independently testable on CPU.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Final

import numpy as np
import torch
import yaml

from .slope_contract import SLOPE_GRADIENTS


POLICY_ACCEPTANCE_SCHEMA_VERSION: Final[int] = 1
POLICY_ABLATION_MATRIX_SCHEMA_VERSION: Final[int] = 3
POLICY_ABLATION_MANIFEST_SCHEMA_VERSION: Final[int] = 3
GUIDE_POLICY_EVALUATION_TASK: Final[str] = (
    "Isaac-G1-Rickshaw-Directional-Slope-v0"
)
SIGNED_SLOPES: Final[tuple[float, ...]] = SLOPE_GRADIENTS
COMMAND_PHASE_LABELS: Final[tuple[str, ...]] = (
    "standing",
    "accelerating",
    "cruising",
    "decelerating",
)
CROSS_CASE_LABELS: Final[tuple[str, ...]] = (
    "RANDOM",
)
FORMAL_EVALUATION_NUM_ENVS_MULTIPLE: Final[int] = len(SIGNED_SLOPES) * len(
    CROSS_CASE_LABELS
)
FORMAL_EVALUATION_COMMAND_PROTOCOL: Final[str] = "deterministic_0_to_1_to_0_mps"
FORMAL_EVALUATION_CROSS_CASE_PROTOCOL: Final[str] = (
    "single_training_distribution"
)
THRESHOLD_OPERATORS: Final[dict[str, Any]] = {
    "<=": lambda value, limit: value <= limit,
    ">=": lambda value, limit: value >= limit,
    "<": lambda value, limit: value < limit,
    ">": lambda value, limit: value > limit,
}
ABLATION_VARIANTS: Final[dict[str, tuple[Any, ...]]] = {
    "fat2_weight": (0.0, 0.1),
    "rollout_steps": (24, 48, 64),
    "latent_dim": (8, 16, 24),
}
ABLATION_DEFAULTS: Final[dict[str, Any]] = {
    "fat2_weight": 0.1,
    "rollout_steps": 48,
    "latent_dim": 16,
}

FINAL_ACCEPTANCE_STAGE_THRESHOLDS: Final[dict[str, str]] = {
    "metrics.tracking.speed_rmse_mps": "<=",
    "metrics.episodes.fall_rate": "<=",
    "metrics.tracking.lateral_error.rms_m": "<=",
    "metrics.tracking.heading_error.rms_rad": "<=",
    "metrics.rickshaw.two_wheel_contact_rate": ">=",
    "metrics.d6.residual_m_or_rad.p99": "<=",
    "metrics.stability.zmp_margin_m.p01": ">=",
    "metrics.actuation.arm_torque_margin.p01": ">=",
    "metrics.actuation.leg_torque_margin.p01": ">=",
    "context_interventions.zero_return_drop.fraction_of_abs_baseline": ">=",
    "context_interventions.shuffle_return_drop.fraction_of_abs_baseline": ">=",
}


METRIC_DEFINITIONS: Final[dict[str, str]] = {
    "tracking.speed_rmse_mps": "sqrt(mean((v_s-v_ref)^2)) over policy samples",
    "episodes.fall_rate": "non-timeout terminated episodes / completed episodes",
    "tracking.overspeed_rate": "samples with v_s > v_ref + configured safety margin / samples",
    "tracking.lateral_error": "RMS and maximum absolute path lateral error",
    "tracking.heading_error": "wrapped heading RMS and maximum absolute error",
    "rickshaw.pitch_error": "actual slope-relative pitch minus alpha_target",
    "rickshaw.hitch_height_error": "actual slope-normal hitch height minus target",
    "rickshaw.two_wheel_contact_rate": "samples where both wheel normal forces pass the safety threshold",
    "rickshaw.wheel_normal_force": "per-wheel force percentiles",
    "locomotion.foot_slip": "summed slope-plane speed of contacting feet",
    "actions.processed_rate": "RMS joint norm of (q_t-q_t-1)/policy_dt",
    "actions.processed_jerk": "RMS joint norm of (q_t-2q_t-1+q_t-2)/policy_dt^2",
    "actuation.power": "sum(abs(applied_torque * joint_velocity)) over the 29 policy joints",
    "d6.residual": "maximum of the two D6 spatial residual norms",
    "d6.force/torque": "maximum left/right retained-link incoming D6 reaction force or torque norm",
    "d6.asymmetry": "absolute left/right norm difference divided by their sum",
    "analytic_force.relative_error": "instantaneous symmetric relative error after projecting robot-on-cart D6 force",
    "analytic_force.fat_window_consistency": "0.5 s impulse-bias gate normalized by mean absolute analytic force",
    "stability.zmp_margin": "signed ZMP support-polygon margin for valid samples",
    "actuation.arm/leg_torque_margin": (
        "minimum per-environment 1-|applied_torque|/current actuator.effort_limit"
    ),
    "distillation.teacher_student_action_kl": "KL(teacher Gaussian || student Gaussian), summed over 29 actions",
    "context.zero/shuffle_return_drop": "baseline mean episode return minus intervention mean return",
    "curriculum.distribution": "policy-sample histogram for training stage and slope",
    "stratified": (
        "full metric reductions under the deterministic 0->1->0 m/s command protocol "
        "over the single training distribution"
    ),
}


def d6_wrench_channels(wrench: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reduce the two retained-link D6 reaction wrenches without discarding torque."""

    if not torch.is_tensor(wrench) or wrench.ndim != 3 or wrench.shape[1:] != (2, 6):
        raise ValueError("D6 reaction wrench must have shape [N, 2, 6]")
    if not torch.isfinite(wrench).all():
        raise ValueError("D6 reaction wrench contains non-finite values")
    force_norm = torch.linalg.vector_norm(wrench[..., :3], dim=-1)
    torque_norm = torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
    return {
        "force": torch.amax(force_norm, dim=-1),
        "torque": torch.amax(torque_norm, dim=-1),
        "force_asymmetry": torch.abs(force_norm[:, 0] - force_norm[:, 1])
        / torch.clamp(force_norm[:, 0] + force_norm[:, 1], min=1.0e-6),
        "torque_asymmetry": torch.abs(torque_norm[:, 0] - torque_norm[:, 1])
        / torch.clamp(torque_norm[:, 0] + torque_norm[:, 1], min=1.0e-6),
    }


def slope_label(slope: float) -> str:
    """Return an unambiguous, stable JSON key for one signed gradient."""

    return f"{float(slope):+.2f}"


def command_phase_labels(
    v_ref: Any,
    a_ref: Any,
    *,
    velocity_epsilon: float = 1.0e-3,
    acceleration_epsilon: float = 1.0e-3,
) -> list[str]:
    """Classify deployable speed-reference samples without using ``v_sample``."""

    if velocity_epsilon < 0.0 or acceleration_epsilon < 0.0:
        raise ValueError("command phase epsilons must be non-negative")
    velocity = np.asarray(v_ref, dtype=np.float64)
    acceleration = np.asarray(a_ref, dtype=np.float64)
    if velocity.ndim != 1 or acceleration.shape != velocity.shape:
        raise ValueError("v_ref and a_ref must be one-dimensional arrays with equal shape")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration)):
        raise ValueError("v_ref and a_ref must be finite")
    labels = np.full(velocity.shape, "cruising", dtype=object)
    standing = (np.abs(velocity) <= velocity_epsilon) & (
        np.abs(acceleration) <= acceleration_epsilon
    )
    labels[standing] = "standing"
    labels[acceleration > acceleration_epsilon] = "accelerating"
    labels[acceleration < -acceleration_epsilon] = "decelerating"
    return labels.tolist()


def validate_stratified_summary(value: Any, *, label: str = "stratified") -> None:
    """Require the complete phase/cross-case evaluation grid in an artifact."""

    if not isinstance(value, Mapping) or set(value) != {
        "by_phase",
        "by_cross_case",
        "by_slope_phase",
        "by_slope_cross_case",
    }:
        raise ValueError(f"{label} must contain the exact stratified reductions")

    def validate_leaves(
        raw: Any, expected: Sequence[str], *, leaf_label: str
    ) -> None:
        if not isinstance(raw, Mapping) or set(raw) != set(expected):
            raise ValueError(f"{leaf_label} has an incomplete label set")
        for name, summary in raw.items():
            samples = summary.get("samples") if isinstance(summary, Mapping) else None
            if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
                raise ValueError(f"{leaf_label}.{name} has an invalid sample count")
            episodes = summary.get("episodes")
            completed = episodes.get("completed") if isinstance(episodes, Mapping) else None
            fall_rate = episodes.get("fall_rate") if isinstance(episodes, Mapping) else None
            causes = (
                episodes.get("termination_cause_histogram")
                if isinstance(episodes, Mapping)
                else None
            )
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or completed <= 0
                or isinstance(fall_rate, bool)
                or not isinstance(fall_rate, (int, float))
                or not math.isfinite(float(fall_rate))
                or not isinstance(causes, Mapping)
            ):
                raise ValueError(f"{leaf_label}.{name} has incomplete episode evidence")

    validate_leaves(value["by_phase"], COMMAND_PHASE_LABELS, leaf_label=f"{label}.by_phase")
    validate_leaves(
        value["by_cross_case"],
        CROSS_CASE_LABELS,
        leaf_label=f"{label}.by_cross_case",
    )
    slope_labels = tuple(slope_label(slope) for slope in SIGNED_SLOPES)
    for key, inner_labels in (
        ("by_slope_phase", COMMAND_PHASE_LABELS),
        ("by_slope_cross_case", CROSS_CASE_LABELS),
    ):
        raw = value[key]
        if not isinstance(raw, Mapping) or set(raw) != set(slope_labels):
            raise ValueError(
                f"{label}.{key} does not contain the exact {len(SIGNED_SLOPES)} slopes"
            )
        for slope in slope_labels:
            validate_leaves(
                raw[slope], inner_labels, leaf_label=f"{label}.{key}.{slope}"
            )


def evaluation_runtime_sources_sha256() -> dict[str, str]:
    """Hash the complete source closure that gives acceptance metrics meaning."""

    repository = Path(__file__).resolve().parents[3]
    package = repository / "source" / "g1_rickshaw_lab" / "g1_rickshaw_lab"
    task = package / "tasks" / "manager_based" / "rickshaw_velocity"
    paths = {
        "implementation_guide": repository / "G1_Rickshaw_IsaacLab_Implementation_Guide.md",
        "policy_acceptance_cli": repository / "scripts" / "evaluate_policy.py",
        "policy_acceptance_contract": Path(__file__).resolve(),
        "s1_candidate_evaluator": repository / "scripts" / "evaluate_context_candidates.py",
        "training_contract": package / "training_contract.py",
        "environment_config": task / "env_cfg.py",
        "terrain_config": task / "terrain_cfg.py",
        "curriculum": task / "mdp" / "curricula.py",
        "events": task / "mdp" / "events.py",
        "dynamics": task / "mdp" / "dynamics.py",
        "rewards": task / "mdp" / "rewards.py",
        "terminations": task / "mdp" / "terminations.py",
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def validate_s1_baseline_acceptance_report(
    report: Any,
    *,
    expected_checkpoint_sha256: str,
    fixed_seeds: Sequence[int],
    episodes_per_slope: int,
) -> dict[str, float]:
    """Validate the S1 return floor used by final S2 acceptance."""

    if not isinstance(report, Mapping):
        raise ValueError("S1 baseline acceptance report must be a mapping")
    if report.get("schema_version") != POLICY_ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError("S1 baseline acceptance report schema is unsupported")
    if report.get("report_type") != "g1_rickshaw_policy_acceptance":
        raise ValueError("S1 baseline report has the wrong report_type")
    if report.get("task") != GUIDE_POLICY_EVALUATION_TASK:
        raise ValueError("S1 baseline report does not use the Guide training task")
    if report.get("status") not in {"recorded", "passed"} or report.get("failures") != []:
        raise ValueError("S1 baseline acceptance report is incomplete or failed")
    inputs = report.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("evaluation_runtime_sources_sha256")
        != evaluation_runtime_sources_sha256()
    ):
        raise ValueError("S1 baseline acceptance report is stale for evaluator sources")
    if not isinstance(expected_checkpoint_sha256, str) or len(expected_checkpoint_sha256) != 64:
        raise ValueError("expected S1 checkpoint SHA256 is malformed")
    try:
        int(expected_checkpoint_sha256, 16)
    except ValueError as exc:
        raise ValueError("expected S1 checkpoint SHA256 is malformed") from exc
    checkpoint = report.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("stage") not in {"s1_context_candidate", "s1_context_distillation"}
        or checkpoint.get("sha256") != expected_checkpoint_sha256.lower()
    ):
        raise ValueError("S1 baseline report checkpoint binding differs from S2 lineage")
    evaluation = report.get("evaluation")
    expected_seeds = list(fixed_seeds)
    if (
        isinstance(episodes_per_slope, bool)
        or not isinstance(episodes_per_slope, int)
        or episodes_per_slope < 100
        or not expected_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in expected_seeds
        )
        or len(set(expected_seeds)) != len(expected_seeds)
        or episodes_per_slope
        % (len(expected_seeds) * len(CROSS_CASE_LABELS))
        != 0
    ):
        raise ValueError(
            "S1/S2 acceptance episode quota must be at least 100 and divisible "
            "by fixed seeds times four cross cases"
        )
    expected_slopes = list(SIGNED_SLOPES)
    curriculum_stages = evaluation.get("curriculum_stages") if isinstance(evaluation, Mapping) else None
    num_envs = evaluation.get("num_envs") if isinstance(evaluation, Mapping) else None
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("deterministic_actions") is not True
        or evaluation.get("fixed_seeds") != expected_seeds
        or evaluation.get("signed_slopes") != expected_slopes
        or evaluation.get("episodes_per_slope_per_stage") != episodes_per_slope
        or isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or num_envs % FORMAL_EVALUATION_NUM_ENVS_MULTIPLE != 0
        or evaluation.get("command_protocol")
        != FORMAL_EVALUATION_COMMAND_PROTOCOL
        or evaluation.get("cross_case_protocol")
        != FORMAL_EVALUATION_CROSS_CASE_PROTOCOL
        or not isinstance(curriculum_stages, (list, tuple))
        or list(curriculum_stages) != ["training"]
    ):
        raise ValueError("S1 baseline report does not use the exact S2 seeds/slopes/episode quota")

    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("S1 baseline report is missing curriculum-stage results")
    expected_labels = {slope_label(slope) for slope in SIGNED_SLOPES}
    returns: dict[str, float] = {}
    for stage_name in ("training",):
        stage_report = stages.get(stage_name)
        if not isinstance(stage_report, Mapping):
            raise ValueError(f"S1 baseline report is missing {stage_name} results")
        validate_stratified_summary(
            stage_report.get("stratified"), label=f"S1 {stage_name}.stratified"
        )
        per_slope = stage_report.get("per_slope")
        if not isinstance(per_slope, Mapping) or set(per_slope) != expected_labels:
            raise ValueError(
                f"S1 {stage_name} report does not contain the exact {len(SIGNED_SLOPES)} slopes"
            )
        for label, slope_report in per_slope.items():
            episodes = slope_report.get("episodes") if isinstance(slope_report, Mapping) else None
            count = episodes.get("completed") if isinstance(episodes, Mapping) else None
            if isinstance(count, bool) or not isinstance(count, int) or count < episodes_per_slope:
                raise ValueError(
                    f"S1 {stage_name} slope {label} has fewer than {episodes_per_slope} episodes"
                )
        interventions = stage_report.get("context_interventions")
        baseline = interventions.get("baseline_return") if isinstance(interventions, Mapping) else None
        if not isinstance(baseline, Mapping):
            raise ValueError(f"S1 {stage_name} report has no baseline return")
        baseline_episodes = baseline.get("episodes")
        if (
            isinstance(baseline_episodes, bool)
            or not isinstance(baseline_episodes, int)
            or baseline_episodes < len(SIGNED_SLOPES) * episodes_per_slope
        ):
            raise ValueError(f"S1 {stage_name} baseline return episode quota is incomplete")
        per_slope_mean = baseline.get("per_slope_mean")
        if not isinstance(per_slope_mean, Mapping) or set(per_slope_mean) != expected_labels:
            raise ValueError(f"S1 {stage_name} baseline return is missing per-slope means")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in per_slope_mean.values()
        ):
            raise ValueError(f"S1 {stage_name} baseline contains a non-finite per-slope return")
        mean = baseline.get("mean")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
            raise ValueError(f"S1 {stage_name} baseline mean return is not finite")
        returns[stage_name] = float(mean)
    return returns


def evaluate_s2_return_floor(
    stage_reports: Mapping[str, Any],
    s1_baseline_returns: Mapping[str, float],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Require fixed-seed S2 TRAINING return to be no lower than S1."""

    comparisons: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for stage_name in ("training",):
        stage_report = stage_reports.get(stage_name)
        interventions = (
            stage_report.get("context_interventions") if isinstance(stage_report, Mapping) else None
        )
        baseline = interventions.get("baseline_return") if isinstance(interventions, Mapping) else None
        s2_return = baseline.get("mean") if isinstance(baseline, Mapping) else None
        s1_return = s1_baseline_returns.get(stage_name)
        if (
            isinstance(s1_return, bool)
            or not isinstance(s1_return, (int, float))
            or not math.isfinite(s1_return)
            or isinstance(s2_return, bool)
            or not isinstance(s2_return, (int, float))
            or not math.isfinite(s2_return)
        ):
            raise ValueError(f"S1/S2 {stage_name} baseline return comparison is incomplete")
        passed = float(s2_return) >= float(s1_return)
        comparisons[stage_name] = {
            "s1_baseline_mean": float(s1_return),
            "s2_baseline_mean": float(s2_return),
            "passed": passed,
        }
        if not passed:
            failures.append(f"s2_{stage_name}_return_below_s1")
    return comparisons, failures


def validate_s1_candidate_selection_report(
    report: Any,
    *,
    expected_candidate_sha256: Mapping[int, str],
    fixed_seeds: Sequence[int],
    episodes_per_slope: int,
) -> list[dict[str, Any]]:
    """Validate the fixed TRAINING task-return half of S1 model selection."""

    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != 1
        or report.get("report_type") != "g1_rickshaw_s1_candidate_selection"
        or report.get("status") != "recorded"
    ):
        raise ValueError("S1 candidate selection report schema/status is invalid")
    if report.get("task") != GUIDE_POLICY_EVALUATION_TASK:
        raise ValueError("S1 candidate selection does not use the Guide training task")
    if report.get("evaluation_runtime_sources_sha256") != evaluation_runtime_sources_sha256():
        raise ValueError("S1 candidate selection report is stale for evaluator sources")
    evaluation = report.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("curriculum_stage") != "training"
        or evaluation.get("fixed_seeds") != [int(seed) for seed in fixed_seeds]
        or evaluation.get("signed_slopes") != list(SIGNED_SLOPES)
        or evaluation.get("episodes_per_slope") != episodes_per_slope
        or episodes_per_slope < 100
    ):
        raise ValueError(
            "S1 candidate selection does not use the configured-slope evaluation ABI"
        )
    expected_labels = {slope_label(slope) for slope in SIGNED_SLOPES}
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(expected_candidate_sha256):
        raise ValueError("S1 candidate selection report has an incomplete candidate set")
    normalized: list[dict[str, Any]] = []
    observed: set[int] = set()
    for raw in results:
        if not isinstance(raw, Mapping):
            raise ValueError("S1 candidate result must be a mapping")
        iteration = raw.get("iteration")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration in observed
            or iteration not in expected_candidate_sha256
            or raw.get("checkpoint_sha256") != expected_candidate_sha256[iteration]
        ):
            raise ValueError("S1 candidate result checkpoint/iteration binding differs")
        observed.add(iteration)
        action_kl = raw.get("validation_action_kl")
        task_return = raw.get("task_return_mean")
        if (
            isinstance(action_kl, bool)
            or not isinstance(action_kl, (int, float))
            or not math.isfinite(action_kl)
            or action_kl < 0.0
            or isinstance(task_return, bool)
            or not isinstance(task_return, (int, float))
            or not math.isfinite(task_return)
        ):
            raise ValueError("S1 candidate selection metrics must be finite and KL non-negative")
        expected_episodes = len(SIGNED_SLOPES) * episodes_per_slope
        if raw.get("episodes") != expected_episodes:
            raise ValueError("S1 candidate result total episode quota is incomplete")
        per_slope = raw.get("per_slope")
        if not isinstance(per_slope, Mapping) or set(per_slope) != expected_labels:
            raise ValueError("S1 candidate result is missing exact per-slope returns")
        means: list[float] = []
        for label, value in per_slope.items():
            count = value.get("episodes") if isinstance(value, Mapping) else None
            mean = value.get("mean") if isinstance(value, Mapping) else None
            if (
                count != episodes_per_slope
                or isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or not math.isfinite(mean)
            ):
                raise ValueError(f"S1 candidate slope {label} has an invalid count/mean")
            means.append(float(mean))
        if not math.isclose(
            float(task_return),
            sum(means) / len(means),
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise ValueError("S1 candidate global return differs from equal-quota per-slope means")
        normalized.append(dict(raw))
    return normalized


def _as_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"sample {name!r} must be one-dimensional, got {result.shape}")
    if result.dtype == np.bool_:
        return result.astype(np.float32)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"sample {name!r} must be numeric")
    return result.astype(np.float32, copy=False)


def _rms(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values, dtype=np.float64))


def _maximum_absolute(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.max(np.abs(values)))


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("p01", "p05", "p50", "p90", "p95", "p99", "max")}
    quantiles = np.quantile(values, (0.01, 0.05, 0.50, 0.90, 0.95, 0.99))
    return {
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(np.max(values)),
    }


@dataclass
class MetricStore:
    """Chunked sample/episode store used by global and per-slope reductions."""

    chunks: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    non_finite_counts: Counter[str] = field(default_factory=Counter)
    episode_returns: list[float] = field(default_factory=list)
    falls: int = 0
    termination_causes: Counter[str] = field(default_factory=Counter)
    curriculum: dict[str, Counter[str]] = field(
        default_factory=lambda: {
            "stage": Counter(),
            "cross_case": Counter(),
            "slope": Counter(),
        }
    )

    def add_samples(self, samples: Mapping[str, Any]) -> None:
        """Append an equally-sized batch, excluding but accounting for NaN/Inf."""

        expected: int | None = None
        for name, raw_value in samples.items():
            values = _as_vector(raw_value, name)
            if expected is None:
                expected = values.size
            elif values.size != expected:
                raise ValueError("all sample arrays in one batch must have equal length")
            finite = np.isfinite(values)
            self.non_finite_counts[name] += int(np.sum(~finite))
            if np.any(finite):
                self.chunks[name].append(values[finite].copy())

    def add_episode(self, episode_return: float, *, fell: bool, causes: Sequence[str]) -> None:
        value = float(episode_return)
        if not math.isfinite(value):
            raise ValueError("episode return must be finite")
        self.episode_returns.append(value)
        self.falls += int(fell)
        self.termination_causes.update(str(cause) for cause in causes)

    def add_curriculum(self, kind: str, labels: Sequence[str]) -> None:
        if kind not in self.curriculum:
            raise KeyError(f"unsupported curriculum histogram {kind!r}")
        self.curriculum[kind].update(str(label) for label in labels)

    def values(self, name: str) -> np.ndarray:
        chunks = self.chunks.get(name, ())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def summary(self) -> dict[str, Any]:  # noqa: C901 - mirrors the guide's metric list.
        speed_error = self.values("speed_error")
        lateral = self.values("lateral_error")
        heading = self.values("heading_error")
        pitch = self.values("pitch_error")
        hitch = self.values("hitch_height_error")
        returns = np.asarray(self.episode_returns, dtype=np.float32)
        episodes = len(self.episode_returns)

        def distribution(name: str) -> dict[str, Any]:
            values = self.values(name)
            return {"mean": _mean(values), **_percentiles(values)}

        summary = {
            "samples": int(speed_error.size),
            "non_finite_sample_counts": dict(sorted(self.non_finite_counts.items())),
            "episodes": {
                "completed": episodes,
                "falls": self.falls,
                "fall_rate": None if episodes == 0 else self.falls / episodes,
                "return": {"mean": _mean(returns), **_percentiles(returns)},
                "termination_cause_histogram": dict(sorted(self.termination_causes.items())),
            },
            "tracking": {
                "speed_rmse_mps": _rms(speed_error),
                "overspeed_rate": _mean(self.values("overspeed")),
                "lateral_error": {"rms_m": _rms(lateral), "max_abs_m": _maximum_absolute(lateral)},
                "heading_error": {"rms_rad": _rms(heading), "max_abs_rad": _maximum_absolute(heading)},
            },
            "rickshaw": {
                "pitch_error": {"rms_rad": _rms(pitch), "max_abs_rad": _maximum_absolute(pitch)},
                "hitch_height_error": {"rms_m": _rms(hitch), "max_abs_m": _maximum_absolute(hitch)},
                "two_wheel_contact_rate": _mean(self.values("two_wheel_contact")),
                "wheel_normal_force_n": {
                    "left": distribution("wheel_normal_force_left"),
                    "right": distribution("wheel_normal_force_right"),
                },
            },
            "locomotion": {"foot_slip_mps": distribution("foot_slip")},
            "actions": {
                "processed_rate_radps": distribution("processed_action_rate"),
                "processed_jerk_radps2": distribution("processed_action_jerk"),
            },
            "actuation": {
                "power_w": distribution("power"),
                "arm_torque_margin": distribution("arm_torque_margin"),
                "leg_torque_margin": distribution("leg_torque_margin"),
            },
            "d6": {
                "residual_m_or_rad": distribution("d6_residual"),
                "force_n": distribution("d6_force"),
                "torque_nm": distribution("d6_torque"),
                "force_asymmetry": distribution("d6_force_asymmetry"),
                "torque_asymmetry": distribution("d6_torque_asymmetry"),
            },
            "analytic_force": {
                "t_s_relative_error": distribution("t_s_relative_error"),
                "t_n_relative_error": distribution("t_n_relative_error"),
                "t_s_sign_agreement_rate": _mean(self.values("t_s_sign_agreement")),
                "t_n_sign_agreement_rate": _mean(self.values("t_n_sign_agreement")),
                "valid_rate": _mean(self.values("analytic_force_valid")),
                "fat_window_consistency_rate": _mean(
                    self.values("fat_wrench_consistent")
                ),
                "fat_window_t_s_relative_error": distribution(
                    "fat_wrench_t_s_relative_error"
                ),
                "fat_window_t_n_relative_error": distribution(
                    "fat_wrench_t_n_relative_error"
                ),
            },
            "stability": {
                "zmp_margin_m": distribution("zmp_margin"),
                "zmp_valid_rate": _mean(self.values("zmp_valid")),
            },
            "distillation": {
                "teacher_student_action_kl": distribution("teacher_student_kl"),
            },
            "curriculum": {
                "distribution": {
                    kind: dict(sorted(counter.items()))
                    for kind, counter in self.curriculum.items()
                }
            },
        }
        return summary


@dataclass
class PolicyEvaluationAccumulator:
    """Global plus complete per-slope metric aggregation."""

    global_store: MetricStore = field(default_factory=MetricStore)
    slope_stores: tuple[MetricStore, ...] = field(
        default_factory=lambda: tuple(MetricStore() for _ in SIGNED_SLOPES)
    )
    phase_stores: dict[str, MetricStore] = field(default_factory=dict)
    cross_case_stores: dict[str, MetricStore] = field(default_factory=dict)
    slope_phase_stores: tuple[dict[str, MetricStore], ...] = field(
        default_factory=lambda: tuple({} for _ in SIGNED_SLOPES)
    )
    slope_cross_case_stores: tuple[dict[str, MetricStore], ...] = field(
        default_factory=lambda: tuple({} for _ in SIGNED_SLOPES)
    )

    @staticmethod
    def _add_labeled_samples(
        stores: dict[str, MetricStore],
        labels: Sequence[str],
        vectors: Mapping[str, np.ndarray],
    ) -> None:
        label_array = np.asarray(labels, dtype=object)
        for label in dict.fromkeys(str(value) for value in labels):
            selected = label_array == label
            stores.setdefault(label, MetricStore()).add_samples(
                {name: value[selected] for name, value in vectors.items()}
            )

    def add_step(
        self,
        samples: Mapping[str, Any],
        slope_indices: Any,
        *,
        stage_labels: Sequence[str] | None = None,
        cross_case_labels: Sequence[str] | None = None,
        phase_labels: Sequence[str] | None = None,
    ) -> None:
        indices = np.asarray(slope_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(SIGNED_SLOPES)):
            raise ValueError("slope_indices must be a 1-D array in [0, 12]")
        vectors = {name: _as_vector(value, name) for name, value in samples.items()}
        if any(value.size != indices.size for value in vectors.values()):
            raise ValueError("sample arrays and slope_indices must have equal length")
        self.global_store.add_samples(vectors)
        slope_names = [slope_label(SIGNED_SLOPES[index]) for index in indices]
        self.global_store.add_curriculum("slope", slope_names)
        if stage_labels is not None:
            if len(stage_labels) != indices.size:
                raise ValueError("stage labels have the wrong length")
            self.global_store.add_curriculum("stage", stage_labels)
        if cross_case_labels is not None:
            if len(cross_case_labels) != indices.size:
                raise ValueError("cross-case labels have the wrong length")
            self.global_store.add_curriculum("cross_case", cross_case_labels)
            unknown = sorted(set(cross_case_labels) - set(CROSS_CASE_LABELS))
            if unknown:
                raise ValueError(f"unknown cross-case labels: {unknown}")
            self._add_labeled_samples(self.cross_case_stores, cross_case_labels, vectors)
        if phase_labels is not None:
            if len(phase_labels) != indices.size:
                raise ValueError("command-phase labels have the wrong length")
            unknown = sorted(set(phase_labels) - set(COMMAND_PHASE_LABELS))
            if unknown:
                raise ValueError(f"unknown command-phase labels: {unknown}")
            self._add_labeled_samples(self.phase_stores, phase_labels, vectors)
        for slope_index, store in enumerate(self.slope_stores):
            selected = indices == slope_index
            if np.any(selected):
                selected_vectors = {
                    name: value[selected] for name, value in vectors.items()
                }
                store.add_samples(selected_vectors)
                store.add_curriculum("slope", [slope_label(SIGNED_SLOPES[slope_index])] * int(selected.sum()))
                if stage_labels is not None:
                    labels = np.asarray(stage_labels, dtype=object)[selected].tolist()
                    store.add_curriculum("stage", labels)
                if cross_case_labels is not None:
                    labels = np.asarray(cross_case_labels, dtype=object)[selected].tolist()
                    store.add_curriculum("cross_case", labels)
                    self._add_labeled_samples(
                        self.slope_cross_case_stores[slope_index],
                        labels,
                        selected_vectors,
                    )
                if phase_labels is not None:
                    labels = np.asarray(phase_labels, dtype=object)[selected].tolist()
                    self._add_labeled_samples(
                        self.slope_phase_stores[slope_index],
                        labels,
                        selected_vectors,
                    )

    def add_episode(
        self,
        slope_index: int,
        episode_return: float,
        *,
        fell: bool,
        causes: Sequence[str],
        phase_labels: Sequence[str],
        cross_case_label: str,
    ) -> None:
        if not 0 <= slope_index < len(SIGNED_SLOPES):
            raise ValueError("invalid slope index")
        observed_phases = tuple(dict.fromkeys(str(label) for label in phase_labels))
        if not observed_phases:
            raise ValueError("episode must contain at least one command phase")
        unknown_phases = sorted(set(observed_phases) - set(COMMAND_PHASE_LABELS))
        if unknown_phases:
            raise ValueError(f"unknown command-phase labels {unknown_phases}")
        if cross_case_label not in CROSS_CASE_LABELS:
            raise ValueError(f"unknown cross-case label {cross_case_label!r}")
        self.global_store.add_episode(episode_return, fell=fell, causes=causes)
        self.slope_stores[slope_index].add_episode(episode_return, fell=fell, causes=causes)
        for phase_label in observed_phases:
            self.phase_stores.setdefault(phase_label, MetricStore()).add_episode(
                episode_return, fell=fell, causes=causes
            )
            self.slope_phase_stores[slope_index].setdefault(
                phase_label, MetricStore()
            ).add_episode(episode_return, fell=fell, causes=causes)
        self.cross_case_stores.setdefault(cross_case_label, MetricStore()).add_episode(
            episode_return, fell=fell, causes=causes
        )
        self.slope_cross_case_stores[slope_index].setdefault(
            cross_case_label, MetricStore()
        ).add_episode(episode_return, fell=fell, causes=causes)

    def summary(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        return self.global_store.summary(), {
            slope_label(slope): self.slope_stores[index].summary()
            for index, slope in enumerate(SIGNED_SLOPES)
        }

    def stratified_summary(self) -> dict[str, Any]:
        def summarize(
            stores: Mapping[str, MetricStore], labels: Sequence[str]
        ) -> dict[str, Any]:
            return {
                label: stores.get(label, MetricStore()).summary()
                for label in labels
            }

        return {
            "by_phase": summarize(self.phase_stores, COMMAND_PHASE_LABELS),
            "by_cross_case": summarize(self.cross_case_stores, CROSS_CASE_LABELS),
            "by_slope_phase": {
                slope_label(slope): summarize(
                    self.slope_phase_stores[index], COMMAND_PHASE_LABELS
                )
                for index, slope in enumerate(SIGNED_SLOPES)
            },
            "by_slope_cross_case": {
                slope_label(slope): summarize(
                    self.slope_cross_case_stores[index], CROSS_CASE_LABELS
                )
                for index, slope in enumerate(SIGNED_SLOPES)
            },
        }


def validate_final_student_acceptance_report(
    report: Any,
    *,
    expected_checkpoint_sha256: str,
    expected_teacher_sha256: str,
    expected_s1_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate the complete, deployable S2 fixed-seed acceptance artifact."""

    if not isinstance(report, Mapping):
        raise ValueError("final S2 acceptance report must be a mapping")
    if (
        report.get("schema_version") != POLICY_ACCEPTANCE_SCHEMA_VERSION
        or report.get("report_type") != "g1_rickshaw_policy_acceptance"
        or report.get("status") != "passed"
        or report.get("failures") != []
    ):
        raise ValueError("final S2 acceptance report is incomplete or failed")
    if report.get("task") != GUIDE_POLICY_EVALUATION_TASK:
        raise ValueError("final S2 acceptance does not use the Guide training task")
    checkpoint = report.get("checkpoint")
    teacher = report.get("teacher_checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("stage") != "s2_student_ppo"
        or checkpoint.get("sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("final acceptance checkpoint binding differs from the S2 checkpoint")
    if (
        not isinstance(teacher, Mapping)
        or teacher.get("stage") != "s0_teacher"
        or teacher.get("sha256") != expected_teacher_sha256
    ):
        raise ValueError("final acceptance teacher binding differs from S2 lineage")
    inputs = report.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("evaluation_runtime_sources_sha256")
        != evaluation_runtime_sources_sha256()
    ):
        raise ValueError("final acceptance report is stale for evaluator sources")

    evaluation = report.get("evaluation")
    stages_requested = (
        evaluation.get("curriculum_stages") if isinstance(evaluation, Mapping) else None
    )
    quota = (
        evaluation.get("episodes_per_slope_per_stage")
        if isinstance(evaluation, Mapping)
        else None
    )
    seeds = evaluation.get("fixed_seeds") if isinstance(evaluation, Mapping) else None
    num_envs = evaluation.get("num_envs") if isinstance(evaluation, Mapping) else None
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("deterministic_actions") is not True
        or evaluation.get("signed_slopes") != list(SIGNED_SLOPES)
        or not isinstance(stages_requested, (list, tuple))
        or list(stages_requested) != ["training"]
        or isinstance(quota, bool)
        or not isinstance(quota, int)
        or quota < 100
        or not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or quota % (len(seeds) * len(CROSS_CASE_LABELS)) != 0
        or isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or num_envs % FORMAL_EVALUATION_NUM_ENVS_MULTIPLE != 0
        or evaluation.get("command_protocol")
        != FORMAL_EVALUATION_COMMAND_PROTOCOL
        or evaluation.get("cross_case_protocol")
        != FORMAL_EVALUATION_CROSS_CASE_PROTOCOL
    ):
        raise ValueError("final acceptance does not prove the fixed TRAINING evaluation quota")

    stages = report.get("stages")
    slope_labels = {slope_label(slope) for slope in SIGNED_SLOPES}
    if not isinstance(stages, Mapping):
        raise ValueError("final acceptance report has no stage results")
    for stage_name in ("training",):
        stage_report = stages.get(stage_name)
        per_slope = (
            stage_report.get("per_slope") if isinstance(stage_report, Mapping) else None
        )
        if not isinstance(per_slope, Mapping) or set(per_slope) != slope_labels:
            raise ValueError(f"final acceptance {stage_name} lacks exact per-slope metrics")
        for label, slope_report in per_slope.items():
            episodes = (
                slope_report.get("episodes")
                if isinstance(slope_report, Mapping)
                else None
            )
            if not isinstance(episodes, Mapping) or episodes.get("completed") != quota:
                raise ValueError(
                    f"final acceptance {stage_name}/{label} episode count differs from quota"
                )
        validate_stratified_summary(
            stage_report.get("stratified"),
            label=f"stages.{stage_name}.stratified",
        )
        interventions = stage_report.get("context_interventions")
        if not isinstance(interventions, Mapping):
            raise ValueError(f"final acceptance {stage_name} lacks context interventions")
        for intervention in ("zero", "shuffle"):
            drop = interventions.get(f"{intervention}_return_drop")
            fraction = (
                drop.get("fraction_of_abs_baseline") if isinstance(drop, Mapping) else None
            )
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(float(fraction))
            ):
                raise ValueError(
                    f"final acceptance {stage_name} lacks a finite {intervention} return drop"
                )

    serialized_thresholds = report.get("thresholds")
    thresholds = _parse_threshold_mapping(serialized_thresholds)
    validate_final_acceptance_thresholds(
        thresholds,
        curriculum_stages=stages_requested,
    )
    recomputed, failures = evaluate_thresholds(report, thresholds)
    if failures or report.get("threshold_results") != recomputed or not all(recomputed.values()):
        raise ValueError("final acceptance threshold results do not recompute as passed")

    baseline = report.get("s1_baseline_acceptance")
    return_floor = baseline.get("s2_return_floor") if isinstance(baseline, Mapping) else None
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("checkpoint_sha256") != expected_s1_checkpoint_sha256
        or not isinstance(return_floor, Mapping)
        or set(return_floor) != {"training"}
        or any(
            not isinstance(value, Mapping) or value.get("passed") is not True
            for value in return_floor.values()
        )
    ):
        raise ValueError("final acceptance does not prove the S2 >= S1 TRAINING return floor")
    return {
        "fixed_seeds": list(seeds),
        "episodes_per_slope": quota,
        "curriculum_stages": list(stages_requested),
    }


@dataclass(frozen=True)
class Threshold:
    operator: str
    value: float

    def __post_init__(self) -> None:
        if self.operator not in THRESHOLD_OPERATORS:
            raise ValueError(f"unsupported threshold operator {self.operator!r}")
        if not math.isfinite(self.value):
            raise ValueError("threshold values must be finite")


def _parse_threshold_mapping(value: Any) -> dict[str, Threshold]:
    if not isinstance(value, Mapping):
        raise ValueError("thresholds must be a metric-path mapping")
    result: dict[str, Threshold] = {}
    for path, specification in value.items():
        if not isinstance(path, str) or not path or path.startswith(".") or path.endswith("."):
            raise ValueError("threshold metric paths must be non-empty dotted strings")
        if not isinstance(specification, Mapping) or set(specification) != {"operator", "value"}:
            raise ValueError(f"threshold {path!r} must contain exactly operator/value")
        result[path] = Threshold(str(specification["operator"]), float(specification["value"]))
    return result


def load_thresholds(
    path: str | Path | None = None,
    cli_values: Sequence[str] = (),
) -> dict[str, Threshold]:
    """Load only evaluator-supplied thresholds; this function has no defaults."""

    result: dict[str, Threshold] = {}
    if path is not None:
        threshold_path = Path(path)
        value = yaml.safe_load(threshold_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise ValueError("threshold YAML requires schema_version: 1")
        if set(value) != {"schema_version", "thresholds"}:
            raise ValueError("threshold YAML may contain only schema_version and thresholds")
        result.update(_parse_threshold_mapping(value["thresholds"]))
    pattern = re.compile(r"^([^<>=]+?)(<=|>=|<|>)([^<>=]+)$")
    for raw in cli_values:
        match = pattern.fullmatch(raw.strip())
        if match is None:
            raise ValueError(f"invalid CLI threshold {raw!r}; expected metric.path<=number")
        metric_path, operator, raw_limit = match.groups()
        metric_path = metric_path.strip()
        if metric_path in result:
            raise ValueError(f"duplicate threshold {metric_path!r}")
        result[metric_path] = Threshold(operator, float(raw_limit))
    return result


def _read_metric(root: Mapping[str, Any], path: str) -> float:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"threshold references missing metric {path!r}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"threshold metric {path!r} is not a finite scalar")
    return float(value)


def evaluate_thresholds(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Threshold]
) -> tuple[dict[str, bool], list[str]]:
    outcomes: dict[str, bool] = {}
    for path, threshold in thresholds.items():
        value = _read_metric(metrics, path)
        outcomes[path] = bool(THRESHOLD_OPERATORS[threshold.operator](value, threshold.value))
    return outcomes, [path for path, passed in outcomes.items() if not passed]


def validate_final_acceptance_thresholds(
    thresholds: Mapping[str, Threshold],
    *,
    curriculum_stages: Sequence[str],
) -> None:
    """Require an explicit authority for every final TRAINING acceptance metric."""

    stages = tuple(curriculum_stages)
    if stages != ("training",):
        raise ValueError("final student acceptance requires exactly TRAINING")
    required = {
        f"stages.{stage}.{suffix}": operator
        for stage in ("training",)
        for suffix, operator in FINAL_ACCEPTANCE_STAGE_THRESHOLDS.items()
    }
    missing = sorted(set(required) - set(thresholds))
    if missing:
        raise ValueError(
            "final student acceptance thresholds are incomplete: " + ", ".join(missing)
        )
    for path, operator in required.items():
        threshold = thresholds[path]
        if threshold.operator != operator:
            raise ValueError(
                f"final student threshold {path!r} must use {operator}, "
                f"got {threshold.operator}"
            )
        if "return_drop" in path and threshold.value <= 0.0:
            raise ValueError(
                f"context intervention threshold {path!r} must require a positive decline"
            )


def serialize_thresholds(thresholds: Mapping[str, Threshold]) -> dict[str, dict[str, Any]]:
    return {
        path: {"operator": threshold.operator, "value": threshold.value}
        for path, threshold in sorted(thresholds.items())
    }


def validate_ablation_matrix(value: Any) -> list[dict[str, Any]]:
    """Validate the three independent policy ablation sweeps."""

    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != POLICY_ABLATION_MATRIX_SCHEMA_VERSION
    ):
        raise ValueError(
            "ablation matrix requires "
            f"schema_version: {POLICY_ABLATION_MATRIX_SCHEMA_VERSION}"
        )
    if set(value) - {"schema_version", "defaults", "runs"}:
        raise ValueError("ablation matrix contains unknown top-level keys")
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise ValueError("ablation matrix runs must be a list")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    observed: dict[str, list[Any]] = defaultdict(list)
    for raw in runs:
        required_fields = {
            "id",
            "group",
            "value",
            "checkpoint",
            "teacher_checkpoint",
            "s1_baseline_report",
        }
        if not isinstance(raw, Mapping) or set(raw) != required_fields:
            raise ValueError(
                "each ablation run requires exactly id/group/value/checkpoint/"
                "teacher_checkpoint/s1_baseline_report"
            )
        identifier = raw["id"]
        group = raw["group"]
        paths = {
            name: raw[name]
            for name in ("checkpoint", "teacher_checkpoint", "s1_baseline_report")
        }
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("ablation run IDs must be unique non-empty strings")
        if group not in ABLATION_VARIANTS:
            raise ValueError(f"unknown ablation group {group!r}")
        if any(not isinstance(path, str) or not path for path in paths.values()):
            raise ValueError("ablation checkpoint/report bindings must be non-empty paths")
        raw_variant = raw["value"]
        if group == "fat2_weight":
            variant: Any = float(raw_variant)
        else:
            if isinstance(raw_variant, bool):
                raise ValueError(f"{group} variants must be integers")
            variant = int(raw_variant)
        if variant not in ABLATION_VARIANTS[group]:
            raise ValueError(f"unsupported {group} variant {variant!r}")
        ids.add(identifier)
        observed[group].append(variant)
        normalized.append(
            {"id": identifier, "group": group, "value": variant, **paths}
        )
    for group, required in ABLATION_VARIANTS.items():
        if sorted(observed[group], key=str) != sorted(required, key=str):
            raise ValueError(
                f"ablation group {group!r} must contain each of {required} exactly once"
            )
    return normalized


def evaluate_ablation_selection(
    runs: Sequence[Mapping[str, Any]],
    *,
    selected_run_id: str,
) -> dict[str, Any]:
    """Prove that every selected non-default variant improves fixed validation."""

    defaults = ABLATION_DEFAULTS
    by_id = {run.get("id"): run for run in runs}
    if len(by_id) != len(runs):
        raise ValueError("ablation evidence contains duplicate run IDs")
    selected = by_id.get(selected_run_id)
    if not isinstance(selected, Mapping):
        raise ValueError("selected ablation run does not exist")
    selected_digest = selected.get("checkpoint_sha256")
    selected_training = selected.get("training_configuration")
    selected_values = (
        selected_training.get("ablation_values")
        if isinstance(selected_training, Mapping)
        else None
    )
    if not isinstance(selected_digest, str) or not isinstance(selected_values, Mapping):
        raise ValueError("selected ablation run lacks checkpoint/training bindings")

    default_digests: set[str] = set()
    common_task_seed: tuple[Any, Any] | None = None
    common_report_contract: dict[str, Any] | None = None
    common_provenance: dict[str, Any] | None = None
    for run in runs:
        group = run.get("group")
        value = run.get("value")
        training = run.get("training_configuration")
        values = training.get("ablation_values") if isinstance(training, Mapping) else None
        if group not in ABLATION_VARIANTS or value not in ABLATION_VARIANTS[group]:
            raise ValueError(f"ablation run {run.get('id')!r} has an invalid group/value")
        expected_values = {**defaults, group: value}
        if values != expected_values:
            raise ValueError(
                f"ablation run {run.get('id')!r} is not an independent one-factor sweep"
            )
        if (
            training.get("stage") != "s2_student_ppo"
            or training.get("formal") is not True
            or training.get("task") != GUIDE_POLICY_EVALUATION_TASK
            or isinstance(training.get("seed"), bool)
            or not isinstance(training.get("seed"), int)
        ):
            raise ValueError(
                f"ablation run {run.get('id')!r} is not a formal Guide S2 training run"
            )
        task_seed = (training.get("task"), training.get("seed"))
        if common_task_seed is None:
            common_task_seed = task_seed
        elif task_seed != common_task_seed:
            raise ValueError("ablation runs do not share one task and training seed")
        digest = run.get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"ablation run {run.get('id')!r} lacks a checkpoint SHA256")
        if value == defaults[group]:
            default_digests.add(digest)

        provenance = run.get("checkpoint_provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError(f"ablation run {run.get('id')!r} lacks checkpoint provenance")
        normalized_provenance = dict(provenance)
        if common_provenance is None:
            common_provenance = normalized_provenance
        elif normalized_provenance != common_provenance:
            raise ValueError("ablation runs do not share one runtime/config provenance")

        report = run.get("report_content")
        checkpoint_binding = report.get("checkpoint") if isinstance(report, Mapping) else None
        report_ablation = report.get("ablation") if isinstance(report, Mapping) else None
        evaluation = report.get("evaluation") if isinstance(report, Mapping) else None
        inputs = report.get("inputs") if isinstance(report, Mapping) else None
        thresholds = report.get("thresholds") if isinstance(report, Mapping) else None
        if (
            not isinstance(report, Mapping)
            or report.get("status") != "passed"
            or report.get("failures") != []
            or report.get("task") != GUIDE_POLICY_EVALUATION_TASK
            or not isinstance(checkpoint_binding, Mapping)
            or checkpoint_binding.get("stage") != "s2_student_ppo"
            or checkpoint_binding.get("sha256") != digest
            or not isinstance(report_ablation, Mapping)
            or report_ablation.get("id") != run.get("id")
            or report_ablation.get("group") != group
            or not isinstance(report_ablation.get("matrix_sha256"), str)
            or not isinstance(evaluation, Mapping)
            or not isinstance(inputs, Mapping)
            or inputs.get("ablation_matrix_sha256")
            != report_ablation.get("matrix_sha256")
        ):
            raise ValueError(f"ablation run {run.get('id')!r} has a misbound evaluation report")
        serialized_thresholds = dict(thresholds) if isinstance(thresholds, Mapping) else None
        if serialized_thresholds is None:
            raise ValueError(f"ablation run {run.get('id')!r} lacks explicit thresholds")
        parsed_thresholds = _parse_threshold_mapping(serialized_thresholds)
        validate_final_acceptance_thresholds(
            parsed_thresholds,
            curriculum_stages=evaluation.get("curriculum_stages", ()),
        )
        fixed_seeds = evaluation.get("fixed_seeds")
        quota = evaluation.get("episodes_per_slope_per_stage")
        num_envs = evaluation.get("num_envs")
        if (
            evaluation.get("deterministic_actions") is not True
            or not isinstance(fixed_seeds, list)
            or not fixed_seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in fixed_seeds)
            or len(set(fixed_seeds)) != len(fixed_seeds)
            or evaluation.get("signed_slopes") != list(SIGNED_SLOPES)
            or isinstance(quota, bool)
            or not isinstance(quota, int)
            or quota < 100
            or quota % (len(fixed_seeds) * len(CROSS_CASE_LABELS)) != 0
            or isinstance(num_envs, bool)
            or not isinstance(num_envs, int)
            or num_envs <= 0
            or num_envs % FORMAL_EVALUATION_NUM_ENVS_MULTIPLE != 0
            or evaluation.get("command_protocol")
            != FORMAL_EVALUATION_COMMAND_PROTOCOL
            or evaluation.get("cross_case_protocol")
            != FORMAL_EVALUATION_CROSS_CASE_PROTOCOL
            or evaluation.get("curriculum_stages") != ["training"]
            or evaluation.get("fat2_weight_override") != expected_values["fat2_weight"]
            or evaluation.get("rollout_steps_training_variant")
            != expected_values["rollout_steps"]
            or evaluation.get("latent_dim_variant") != expected_values["latent_dim"]
        ):
            raise ValueError(
                f"ablation run {run.get('id')!r} does not use the formal fixed evaluation ABI"
            )
        report_contract = {
            "task": report.get("task"),
            "matrix_sha256": report_ablation.get("matrix_sha256"),
            "evaluation": {
                name: evaluation[name]
                for name in (
                    "deterministic_actions",
                    "fixed_seeds",
                    "signed_slopes",
                    "episodes_per_slope_per_stage",
                    "num_envs",
                    "curriculum_stages",
                    "command_protocol",
                    "cross_case_protocol",
                )
            },
            "inputs": dict(inputs),
            "thresholds": serialized_thresholds,
        }
        if common_report_contract is None:
            common_report_contract = report_contract
        elif report_contract != common_report_contract:
            raise ValueError(
                "ablation runs do not share one fixed evaluation setting/threshold authority"
            )
    if len(default_digests) != 1:
        raise ValueError("all three default ablation entries must bind one baseline checkpoint")

    selected_group = selected.get("group")
    selected_value = selected.get("value")

    def find(group: str, value: Any) -> Mapping[str, Any]:
        matches = [
            run
            for run in runs
            if run.get("group") == group and run.get("value") == value
        ]
        if len(matches) != 1:
            raise ValueError(f"ablation evidence for {group}={value!r} is not unique")
        return matches[0]

    def metrics(run: Mapping[str, Any], stage: str) -> tuple[float, float, float]:
        report = run.get("report_content")
        try:
            stage_metrics = report["stages"][stage]["metrics"]
            values = (
                float(stage_metrics["tracking"]["speed_rmse_mps"]),
                float(stage_metrics["episodes"]["fall_rate"]),
                float(stage_metrics["stability"]["zmp_margin_m"]["p01"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"ablation run {run.get('id')!r} lacks independent {stage} metrics"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"ablation run {run.get('id')!r} has non-finite metrics")
        return values

    evidence: dict[str, Any] = {}
    for group, default_value in defaults.items():
        baseline = find(group, default_value)
        variants: list[dict[str, Any]] = []
        for value in ABLATION_VARIANTS[group]:
            candidate = find(group, value)
            comparisons: dict[str, Any] = {}
            improved = False
            for stage in ("training",):
                candidate_speed, candidate_fall, candidate_zmp = metrics(candidate, stage)
                baseline_speed, baseline_fall, baseline_zmp = metrics(baseline, stage)
                stage_passed = (
                    candidate_speed <= baseline_speed
                    and candidate_fall <= baseline_fall
                    and candidate_zmp >= baseline_zmp
                )
                improved = improved or (
                    candidate_speed < baseline_speed
                    or candidate_fall < baseline_fall
                    or candidate_zmp > baseline_zmp
                )
                comparisons[stage] = {
                    "speed_rmse_delta": candidate_speed - baseline_speed,
                    "fall_rate_delta": candidate_fall - baseline_fall,
                    "zmp_margin_delta": candidate_zmp - baseline_zmp,
                    "passed": stage_passed,
                }
            throughput_ratio = None
            eligible = value == default_value or (
                improved and all(item["passed"] for item in comparisons.values())
            )
            if group == "rollout_steps" and value == 64:
                candidate_throughput = float(
                    candidate["training_throughput"]["samples_per_second"]
                )
                baseline_throughput = float(
                    baseline["training_throughput"]["samples_per_second"]
                )
                throughput_ratio = candidate_throughput / baseline_throughput
                training = comparisons["training"]
                eligible = eligible and (
                    throughput_ratio >= 0.85
                    and training["speed_rmse_delta"] < 0.0
                    and training["fall_rate_delta"] < 0.0
                    and training["zmp_margin_delta"] > 0.0
                )
            variants.append(
                {
                    "value": value,
                    "comparisons": comparisons,
                    "throughput_ratio_vs_48": throughput_ratio,
                    "eligible_for_selection": eligible,
                }
            )
            if (
                group == selected_group
                and value == selected_value
                and not eligible
            ):
                raise ValueError(
                    f"selected non-default {group}={value!r} does not improve independent validation"
                )
        evidence[group] = {
            "selected_value": selected_values[group],
            "default_value": default_value,
            "variants": variants,
            "passed": True,
        }
    return evidence


__all__ = [
    "ABLATION_DEFAULTS",
    "ABLATION_VARIANTS",
    "COMMAND_PHASE_LABELS",
    "CROSS_CASE_LABELS",
    "FORMAL_EVALUATION_COMMAND_PROTOCOL",
    "FORMAL_EVALUATION_CROSS_CASE_PROTOCOL",
    "FORMAL_EVALUATION_NUM_ENVS_MULTIPLE",
    "GUIDE_POLICY_EVALUATION_TASK",
    "METRIC_DEFINITIONS",
    "POLICY_ABLATION_MANIFEST_SCHEMA_VERSION",
    "POLICY_ABLATION_MATRIX_SCHEMA_VERSION",
    "POLICY_ACCEPTANCE_SCHEMA_VERSION",
    "SIGNED_SLOPES",
    "MetricStore",
    "PolicyEvaluationAccumulator",
    "Threshold",
    "evaluation_runtime_sources_sha256",
    "command_phase_labels",
    "d6_wrench_channels",
    "evaluate_s2_return_floor",
    "evaluate_ablation_selection",
    "evaluate_thresholds",
    "load_thresholds",
    "serialize_thresholds",
    "slope_label",
    "validate_s1_baseline_acceptance_report",
    "validate_final_acceptance_thresholds",
    "validate_final_student_acceptance_report",
    "validate_stratified_summary",
    "validate_s1_candidate_selection_report",
    "validate_ablation_matrix",
]
