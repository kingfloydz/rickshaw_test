"""CPU-only tests for guide section 11.2 reward calibration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.reward_calibration import (
    C1_NOMINAL_PHYSICS_FIELDS,
    GUIDE_REWARD_NORMALIZATION_SCALES,
    GUIDE_REWARD_TERMS,
    NORMAL_SAMPLE_DEFINITION,
    RAW_REWARD_SAMPLE_KIND,
    RAW_REWARD_SAMPLE_SCHEMA_VERSION,
    REQUIRED_RUNTIME_DEPENDENCY_LABELS,
    REQUIRED_RUNTIME_INPUT_HASHES,
    REQUIRED_RUNTIME_INPUT_LABELS,
    REWARD_CALIBRATION_SCHEMA_VERSION,
    REWARD_RUNTIME_INPUT_CLOSURE_VERSION,
    SIGNED_C1_SLOPES,
    RewardCalibrationError,
    calibrate_reward_terms,
    collect_reward_manager_unweighted_step,
    load_and_recompute_reward_calibration_report,
    recompute_reward_calibration,
    reward_calibration_guide_contract,
    reward_calibration_runtime_input_hashes,
    reward_sample_report_source,
    sha256_file,
    validate_raw_sample_artifact,
    validate_c1_physics_snapshot,
    validate_sample_checkpoint_binding,
    verify_content_addressed_report,
    write_content_addressed_json,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


def _weights() -> dict[str, float]:
    result = {name: 0.1 for name in GUIDE_REWARD_TERMS}
    result["track_speed_exp"] = 2.0
    result["termination"] = -200.0
    return result


def _raw_terms(samples=(0.0, 1.0, 2.0)) -> dict[str, list[float]]:
    return {name: list(samples) for name in GUIDE_REWARD_TERMS}


def _artifact() -> dict:
    weights = _weights()
    count_per_slope = 2
    total = len(SIGNED_C1_SLOPES) * count_per_slope
    nominal_values = {
        name: float(index) for index, name in enumerate(C1_NOMINAL_PHYSICS_FIELDS)
    }
    runtime_labels = REQUIRED_RUNTIME_INPUT_HASHES | REQUIRED_RUNTIME_DEPENDENCY_LABELS
    return {
        "schema_version": RAW_REWARD_SAMPLE_SCHEMA_VERSION,
        "kind": RAW_REWARD_SAMPLE_KIND,
        "runtime_input_closure_version": REWARD_RUNTIME_INPUT_CLOSURE_VERSION,
        "reward_normalization_scales": GUIDE_REWARD_NORMALIZATION_SCALES,
        "curriculum_stage": "TRAINING",
        "created_at_utc": "2026-07-13T00:00:00Z",
        "fixed_seed": 42,
        "fixed_slopes": list(SIGNED_C1_SLOPES),
        "slope_sample_counts": {
            f"{slope:+.2f}": count_per_slope for slope in SIGNED_C1_SLOPES
        },
        "normal_sample_definition": NORMAL_SAMPLE_DEFINITION,
        "task": "test-task-v0",
        "num_envs": len(SIGNED_C1_SLOPES),
        "policy_steps": 2,
        "step_dt_s": 0.02,
        "rejected_samples": {"terminated_or_timeout": 0},
        "policy_kind": "teacher",
        "checkpoint": {
            "path": "/tmp/reward-calibration-checkpoint.pt",
            "sha256": "a" * 64,
            "stage": "s0_teacher",
            "curriculum_iteration": 2000,
        },
        "runtime_inputs_sha256": {name: "a" * 64 for name in runtime_labels},
        "runtime_versions": {"torch": "test", "rsl_rl": "test", "isaaclab": "test"},
        "c1_physics": {
            name: {"minimum": value, "maximum": value}
            for name, value in nominal_values.items()
        },
        "c1_nominal_values": nominal_values,
        "term_weights": weights,
        "term_sources": {
            name: "RewardManager._step_reward_per_second_divided_by_weight"
            for name in GUIDE_REWARD_TERMS
        },
        "raw_terms": {name: [0.25] * total for name in GUIDE_REWARD_TERMS},
        "sample_slope_indices": [
            index
            for index in range(len(SIGNED_C1_SLOPES))
            for _ in range(count_per_slope)
        ],
    }


def test_calibration_records_quantiles_and_hard_fails_oversized_term() -> None:
    passing = calibrate_reward_terms(_raw_terms(), _weights())
    assert passing["status"] == "passed"
    assert passing["failures"] == []

    weights = _weights()
    weights["zmp_margin_barrier"] = -2.0
    result = calibrate_reward_terms(_raw_terms(), weights)

    assert result["status"] == "failed"
    assert result["failures"] == ["zmp_margin_barrier"]
    assert result["terms"]["track_speed_exp"]["unweighted"]["p90"] == pytest.approx(1.8)
    assert result["balance_rule"]["maximum_other_term_abs_p90"] == pytest.approx(1.8)
    assert result["terms"]["zmp_margin_barrier"]["weighted_abs_p90"] == pytest.approx(3.6)
    assert result["terms"]["zmp_margin_barrier"]["recommended_weight_if_failed"] == pytest.approx(-1.0)
    assert result["terms"]["termination"]["passed"] is True
    assert result["terms"]["termination"]["exempt_reason"] == "guide_termination_exception"


def test_calibration_rejects_missing_nonfinite_or_misaligned_samples() -> None:
    missing = _raw_terms()
    del missing["feet_slide"]
    with pytest.raises(RewardCalibrationError, match="missing"):
        calibrate_reward_terms(missing, _weights())

    misaligned = _raw_terms()
    misaligned["feet_slide"].append(3.0)
    with pytest.raises(RewardCalibrationError, match="counts differ"):
        calibrate_reward_terms(misaligned, _weights())

    nonfinite = _raw_terms()
    nonfinite["feet_slide"][0] = float("nan")
    with pytest.raises(RewardCalibrationError, match="finite"):
        calibrate_reward_terms(nonfinite, _weights())

    zero_reference = _weights()
    zero_reference["track_speed_exp"] = 0.0
    with pytest.raises(RewardCalibrationError, match="positive reference weight"):
        calibrate_reward_terms(_raw_terms(), zero_reference)


def test_stratified_calibration_catches_one_slope_hidden_below_global_p90() -> None:
    count_per_slope = 10
    sample_count = len(SIGNED_C1_SLOPES) * count_per_slope
    indices = [
        index
        for index in range(len(SIGNED_C1_SLOPES))
        for _ in range(count_per_slope)
    ]
    raw = {name: [0.0] * sample_count for name in GUIDE_REWARD_TERMS}
    raw["track_speed_exp"] = [1.0] * sample_count
    raw["track_speed_exp"][count_per_slope : 2 * count_per_slope] = [
        0.1
    ] * count_per_slope
    raw["zmp_margin_barrier"][:count_per_slope] = [9.0] * count_per_slope
    raw["zmp_margin_barrier"][count_per_slope : 2 * count_per_slope] = [
        2.0
    ] * count_per_slope

    result = calibrate_reward_terms(
        raw, _weights(), sample_slope_indices=indices
    )

    assert result["global_status"] == "passed"
    assert result["status"] == "failed"
    assert result["slope_failures"] == {
        f"{SIGNED_C1_SLOPES[1]:+.2f}": ["zmp_margin_barrier"]
    }
    term = result["terms"]["zmp_margin_barrier"]
    assert term["global_passed"] is True
    assert term["passed"] is False
    expected_slope = f"{SIGNED_C1_SLOPES[1]:+.2f}"
    assert term["failing_slopes"] == [expected_slope]
    assert term["worst_slope"] == expected_slope
    assert term["worst_slope_cap_exceedance_ratio"] == pytest.approx(2.0)
    assert term["limiting_slope"] == expected_slope


def test_reward_manager_extraction_divides_weight_and_calls_zero_weight_term() -> None:
    raw = torch.arange(1, 1 + 2 * len(GUIDE_REWARD_TERMS), dtype=torch.float32).reshape(
        2, len(GUIDE_REWARD_TERMS)
    )
    weights = _weights()
    weights["fat2_prior_exp"] = 0.0
    step_dt = 0.02
    cfgs = {}
    weighted = torch.empty_like(raw)
    for index, name in enumerate(GUIDE_REWARD_TERMS):
        weight = weights[name]
        weighted[:, index] = raw[:, index] * weight
        if name == "fat2_prior_exp":
            func = lambda env: env.zero_weight_raw
        else:
            func = lambda env: torch.zeros_like(env.zero_weight_raw)
        cfgs[name] = SimpleNamespace(weight=weight, func=func, params={})
    manager = SimpleNamespace(
        active_terms=list(GUIDE_REWARD_TERMS),
        _step_reward=weighted,
        _reward_buf=weighted.sum(dim=1) * step_dt,
        _env=SimpleNamespace(
            step_dt=step_dt,
            zero_weight_raw=raw[:, GUIDE_REWARD_TERMS.index("fat2_prior_exp")],
        ),
        get_term_cfg=lambda name: cfgs[name],
    )

    values, sources = collect_reward_manager_unweighted_step(manager)

    for index, name in enumerate(GUIDE_REWARD_TERMS):
        assert torch.allclose(values[name], raw[:, index], rtol=1.0e-6, atol=1.0e-6)
    assert sources["track_speed_exp"] == (
        "RewardManager._step_reward_per_second_divided_by_weight"
    )
    assert sources["fat2_prior_exp"] == "configured_term_callable_zero_weight"


def test_raw_artifact_requires_balanced_fixed_c1_reward_manager_samples() -> None:
    artifact = _artifact()
    validate_raw_sample_artifact(artifact)

    artifact["slope_sample_counts"]["+0.00"] = 1
    with pytest.raises(RewardCalibrationError, match="equally balanced"):
        validate_raw_sample_artifact(artifact)

    artifact = _artifact()
    artifact["sample_slope_indices"][0] = 1
    with pytest.raises(RewardCalibrationError, match="do not match"):
        validate_raw_sample_artifact(artifact)

    artifact = _artifact()
    artifact["term_weights"]["fat2_prior_exp"] = 0.0
    with pytest.raises(RewardCalibrationError, match="invalid extraction source"):
        validate_raw_sample_artifact(artifact)


def test_raw_artifact_recomputes_bound_checkpoint_sha256(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"fixed C1 policy")
    artifact = _artifact()
    artifact["checkpoint"] = {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
    }

    assert validate_sample_checkpoint_binding(artifact) == checkpoint.resolve()
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RewardCalibrationError, match="SHA256 changed"):
        validate_sample_checkpoint_binding(artifact)


def test_c1_snapshot_requires_every_field_at_its_nominal_value() -> None:
    artifact = _artifact()
    validate_c1_physics_snapshot(
        artifact["c1_physics"], artifact["c1_nominal_values"]
    )

    artifact["c1_physics"]["terrain.friction"]["maximum"] += 0.01
    with pytest.raises(RewardCalibrationError, match="differs from nominal"):
        validate_c1_physics_snapshot(
            artifact["c1_physics"], artifact["c1_nominal_values"]
        )

    artifact = _artifact()
    del artifact["c1_physics"]["d6.max_force"]
    with pytest.raises(RewardCalibrationError, match="snapshot fields differ"):
        validate_c1_physics_snapshot(
            artifact["c1_physics"], artifact["c1_nominal_values"]
        )


def test_reward_normalization_is_explicit_and_numerically_compatible() -> None:
    assert rewards.REWARD_NORMALIZATION_SCALES == GUIDE_REWARD_NORMALIZATION_SCALES
    assert set(rewards.REWARD_WEIGHTS) == set(GUIDE_REWARD_TERMS)
    assert rewards.REWARD_WEIGHTS == {
        "track_speed_exp": 2.0,
        "lateral_error_l2": -0.5,
        "heading_error_l2": -0.5,
        "zmp_margin_barrier": -2.0,
        "hitch_height_exp": 0.5,
        "fat2_prior_exp": 0.1,
        "feet_air_time": 0.1,
        "feet_slide": -0.1,
        "terrain_normal_velocity_l2": -0.5,
        "joint_power_l1": -1.0e-4,
        "processed_action_rate_l2": -0.01,
        "processed_action_jerk_l2": -0.005,
        "joint_position_limits": -1.0,
        "termination": -200.0,
    }
    assert rewards.FEET_AIR_TIME_NORMALIZER_S == 1.0
    assert rewards.FEET_SLIDE_NORMALIZER_MPS == 1.0
    assert rewards.JOINT_POWER_NORMALIZER_W == 1.0
    assert rewards.JOINT_LIMIT_NORMALIZER_RAD == 1.0
    power = rewards.joint_power_l1_value(
        torch.tensor([[2.0, -3.0]]), torch.tensor([[4.0, 5.0]])
    )
    torch.testing.assert_close(power, torch.tensor([23.0]))


def test_runtime_input_closure_covers_transitive_code_assets_and_isaaclab() -> None:
    hashes = reward_calibration_runtime_input_hashes()
    assert REQUIRED_RUNTIME_INPUT_LABELS == REQUIRED_RUNTIME_INPUT_HASHES
    assert REQUIRED_RUNTIME_INPUT_HASHES.issubset(hashes)
    assert REQUIRED_RUNTIME_DEPENDENCY_LABELS.issubset(hashes)
    assert any(name.startswith("asset:assets/rickshaw/") for name in hashes)
    assert any(name.endswith("mdp/observations.py") for name in hashes)
    assert all(len(digest) == 64 for digest in hashes.values())


def _write_raw_artifact(tmp_path: Path, artifact: dict) -> Path:
    temporary = tmp_path / "reward_samples.tmp"
    torch.save(artifact, temporary)
    digest = sha256_file(temporary)
    destination = tmp_path / f"reward_samples.{digest}.pt"
    temporary.replace(destination)
    return destination


def test_report_must_recompute_all_statistics_from_bound_raw_samples(tmp_path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"fixed C1 teacher")
    artifact = _artifact()
    artifact["checkpoint"]["path"] = str(checkpoint.resolve())
    artifact["checkpoint"]["sha256"] = sha256_file(checkpoint)
    raw_path = _write_raw_artifact(tmp_path, artifact)
    calibration = recompute_reward_calibration(artifact)
    payload = {
        "schema_version": REWARD_CALIBRATION_SCHEMA_VERSION,
        "tool": "calibrate_rewards",
        "created_at_utc": "2026-07-13T00:00:01Z",
        "status": calibration["status"],
        "guide_contract": reward_calibration_guide_contract(),
        "raw_sample_artifact": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_file(raw_path),
        },
        "analysis_runtime_inputs_sha256": artifact["runtime_inputs_sha256"],
        "source": reward_sample_report_source(artifact),
        "calibration": calibration,
    }
    report_path = write_content_addressed_json(
        tmp_path, "reward_calibration", payload
    )

    loaded = load_and_recompute_reward_calibration_report(
        report_path,
        expected_runtime_hashes=artifact["runtime_inputs_sha256"],
        expected_runtime_versions=artifact["runtime_versions"],
        teacher_checkpoint_path=checkpoint,
    )
    assert loaded["calibration"] == calibration

    forged = dict(payload)
    forged["calibration"] = {
        "status": "passed",
        "failures": [],
        "per_slope": {
            f"{slope:+.2f}": {"status": "passed", "failures": []}
            for slope in SIGNED_C1_SLOPES
        },
    }
    forged_path = write_content_addressed_json(
        tmp_path, "reward_calibration", forged
    )
    with pytest.raises(RewardCalibrationError, match="differ from raw recomputation"):
        load_and_recompute_reward_calibration_report(
            forged_path,
            expected_runtime_hashes=artifact["runtime_inputs_sha256"],
            expected_runtime_versions=artifact["runtime_versions"],
            teacher_checkpoint_path=checkpoint,
        )


def test_content_addressed_report_embeds_and_uses_full_digest(tmp_path) -> None:
    path = write_content_addressed_json(
        tmp_path,
        "reward_calibration",
        {"schema_version": 1, "status": "passed", "terms": {}},
    )
    report = json.loads(path.read_text(encoding="ascii"))

    assert verify_content_addressed_report(report)
    assert path.name == f"reward_calibration.{report['content_sha256']}.json"
    report["status"] = "failed"
    assert not verify_content_addressed_report(report)
    load_and_recompute_reward_calibration_report,
    recompute_reward_calibration,
    reward_calibration_guide_contract,
    reward_calibration_runtime_input_hashes,
    reward_sample_report_source,
