"""CPU-only tests for reward calibration diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from g1_rickshaw_lab.reward_calibration import (
    C1_NOMINAL_PHYSICS_FIELDS,
    GUIDE_REWARD_NORMALIZATION_SCALES,
    GUIDE_REWARD_TERMS,
    NORMAL_SAMPLE_DEFINITION,
    RAW_REWARD_SAMPLE_KIND,
    RAW_REWARD_SAMPLE_SCHEMA_VERSION,
    REWARD_CALIBRATION_SCHEMA_VERSION,
    SIGNED_C1_SLOPES,
    RewardCalibrationError,
    calibrate_reward_terms,
    collect_reward_manager_unweighted_step,
    load_and_recompute_reward_calibration_report,
    recompute_reward_calibration,
    reward_calibration_guide_contract,
    reward_sample_report_source,
    validate_c1_physics_snapshot,
    validate_raw_sample_artifact,
    validate_sample_checkpoint_binding,
    write_reward_calibration_json,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from calibrate_rewards import _physics_snapshot  # noqa: E402


def _weights() -> dict[str, float]:
    result = {name: 0.1 for name in GUIDE_REWARD_TERMS}
    result["track_speed_exp"] = 2.0
    result["termination"] = -200.0
    return result


def _raw_terms(samples=(0.0, 1.0, 2.0)) -> dict[str, list[float]]:
    return {name: list(samples) for name in GUIDE_REWARD_TERMS}


def _artifact(checkpoint: Path | None = None) -> dict:
    count_per_slope = 2
    total = len(SIGNED_C1_SLOPES) * count_per_slope
    nominal = {
        name: float(index + 1) for index, name in enumerate(C1_NOMINAL_PHYSICS_FIELDS)
    }
    return {
        "schema_version": RAW_REWARD_SAMPLE_SCHEMA_VERSION,
        "kind": RAW_REWARD_SAMPLE_KIND,
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
            "path": str((checkpoint or Path("/tmp/missing-teacher.pt")).resolve()),
            "stage": "s0_teacher",
            "curriculum_iteration": 2000,
        },
        "runtime_versions": {
            "torch": "test",
            "rsl_rl": "test",
            "isaaclab": "test",
        },
        "c1_physics": {
            name: {"minimum": value, "maximum": value}
            for name, value in nominal.items()
        },
        "c1_nominal_values": nominal,
        "term_weights": _weights(),
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


def _write_artifacts(tmp_path: Path, artifact: dict) -> tuple[Path, Path, dict]:
    raw_path = tmp_path / "reward_samples.pt"
    torch.save(artifact, raw_path)
    calibration = recompute_reward_calibration(artifact)
    report = {
        "schema_version": REWARD_CALIBRATION_SCHEMA_VERSION,
        "tool": "calibrate_rewards",
        "created_at_utc": "2026-07-13T00:00:01Z",
        "status": calibration["status"],
        "guide_contract": reward_calibration_guide_contract(),
        "raw_sample_artifact": {"path": str(raw_path.resolve())},
        "source": reward_sample_report_source(artifact),
        "calibration": calibration,
    }
    report_path = write_reward_calibration_json(tmp_path, report)
    return raw_path, report_path, calibration


def test_calibration_records_quantiles_and_reports_oversized_term() -> None:
    passing = calibrate_reward_terms(_raw_terms(), _weights())
    assert passing["status"] == "passed"

    weights = _weights()
    weights["zmp_margin_barrier"] = -2.0
    result = calibrate_reward_terms(_raw_terms(), weights)
    assert result["status"] == "failed"
    assert result["failures"] == ["zmp_margin_barrier"]
    assert result["terms"]["track_speed_exp"]["unweighted"]["p90"] == pytest.approx(1.8)
    assert result["terms"]["zmp_margin_barrier"]["weighted_abs_p90"] == pytest.approx(
        3.6
    )
    assert (
        result["terms"]["termination"]["exempt_reason"] == "guide_termination_exception"
    )


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


def test_stratified_calibration_reports_hidden_slope_imbalance() -> None:
    count = 10
    sample_count = len(SIGNED_C1_SLOPES) * count
    indices = [index for index in range(len(SIGNED_C1_SLOPES)) for _ in range(count)]
    raw = {name: [0.0] * sample_count for name in GUIDE_REWARD_TERMS}
    raw["track_speed_exp"] = [1.0] * sample_count
    raw["track_speed_exp"][count : 2 * count] = [0.1] * count
    raw["zmp_margin_barrier"][:count] = [9.0] * count
    raw["zmp_margin_barrier"][count : 2 * count] = [2.0] * count

    result = calibrate_reward_terms(raw, _weights(), sample_slope_indices=indices)
    expected_slope = f"{SIGNED_C1_SLOPES[1]:+.2f}"
    assert result["global_status"] == "passed"
    assert result["status"] == "failed"
    assert result["slope_failures"] == {expected_slope: ["zmp_margin_barrier"]}
    assert result["stratified_balance"]["all_slopes_passed"] is False


def test_reward_manager_extraction_divides_weight_and_calls_zero_weight_term() -> None:
    raw = torch.arange(1, 1 + 2 * len(GUIDE_REWARD_TERMS), dtype=torch.float32).reshape(
        2, len(GUIDE_REWARD_TERMS)
    )
    weights = _weights()
    weights["fat2_prior_exp"] = 0.0
    weighted = torch.empty_like(raw)
    configs = {}
    for index, name in enumerate(GUIDE_REWARD_TERMS):
        weighted[:, index] = raw[:, index] * weights[name]
        configs[name] = SimpleNamespace(
            weight=weights[name],
            func=lambda env: env.zero_weight_raw,
            params={},
        )
    manager = SimpleNamespace(
        active_terms=list(GUIDE_REWARD_TERMS),
        _step_reward=weighted,
        _reward_buf=weighted.sum(dim=1) * 0.02,
        _env=SimpleNamespace(
            step_dt=0.02,
            zero_weight_raw=raw[:, GUIDE_REWARD_TERMS.index("fat2_prior_exp")],
        ),
        get_term_cfg=lambda name: configs[name],
    )

    values, sources = collect_reward_manager_unweighted_step(manager)
    for index, name in enumerate(GUIDE_REWARD_TERMS):
        torch.testing.assert_close(values[name], raw[:, index])
    assert sources["fat2_prior_exp"] == "configured_term_callable_zero_weight"


def test_raw_artifact_requires_balanced_fixed_slope_samples() -> None:
    artifact = _artifact()
    validate_raw_sample_artifact(artifact)
    artifact["slope_sample_counts"]["+0.00"] = 1
    with pytest.raises(RewardCalibrationError, match="equally balanced"):
        validate_raw_sample_artifact(artifact)


def test_checkpoint_binding_is_path_based(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    artifact = _artifact(checkpoint)
    assert validate_sample_checkpoint_binding(artifact) == checkpoint.resolve()
    checkpoint.write_bytes(b"updated teacher")
    assert validate_sample_checkpoint_binding(artifact) == checkpoint.resolve()
    checkpoint.unlink()
    with pytest.raises(RewardCalibrationError, match="no longer exists"):
        validate_sample_checkpoint_binding(artifact)


def test_c1_snapshot_requires_every_field_at_its_nominal_value() -> None:
    artifact = _artifact()
    validate_c1_physics_snapshot(artifact["c1_physics"], artifact["c1_nominal_values"])
    artifact["c1_physics"]["terrain.friction"]["maximum"] += 0.01
    with pytest.raises(RewardCalibrationError, match="differs from nominal"):
        validate_c1_physics_snapshot(
            artifact["c1_physics"], artifact["c1_nominal_values"]
        )


def test_runtime_snapshot_reads_fixed_d6_nominals_from_constraint_cfg() -> None:
    d6_fields = (
        "linear_stiffness",
        "linear_damping",
        "angular_stiffness",
        "angular_damping",
        "max_force",
        "max_torque",
        "linear_limit",
        "angular_limit",
    )
    d6_values = {name: float(index + 2) for index, name in enumerate(d6_fields)}
    domain_nominal = {
        "torso.mass_delta": 0.0,
        "payload.mass": 1.0,
        "payload.com.x": 1.0,
        "payload.com.y": 1.0,
        "payload.com.z": 1.0,
        "rolling_resistance.c_rr": 1.0,
        "terrain.friction": 1.0,
        "wheel.left_damping": 1.0,
        "wheel.right_damping": 1.0,
    }
    ones = torch.ones(2)
    base_env = SimpleNamespace(
        cfg=SimpleNamespace(
            domain_randomization=SimpleNamespace(nominal=domain_nominal)
        ),
        num_envs=2,
        device="cpu",
        d6_constraint_manager=SimpleNamespace(cfg=SimpleNamespace(**d6_values)),
        torso_mass_delta=torch.zeros(2),
        _payload_mass=ones,
        _payload_com=torch.ones(2, 3),
        c_rr=ones,
        terrain_friction=ones,
        _wheel_damping=torch.ones(2, 2),
    )

    _, nominal = _physics_snapshot(base_env)

    assert {name: nominal[f"d6.{name}"] for name in d6_fields} == d6_values


def test_report_recomputes_statistics_from_bound_raw_samples(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    artifact = _artifact(checkpoint)
    _, report_path, calibration = _write_artifacts(tmp_path, artifact)
    loaded = load_and_recompute_reward_calibration_report(
        report_path, teacher_checkpoint_path=checkpoint
    )
    assert loaded["calibration"] == calibration

    report = json.loads(report_path.read_text(encoding="ascii"))
    report["calibration"] = {"status": "passed"}
    report_path.write_text(json.dumps(report), encoding="ascii")
    with pytest.raises(RewardCalibrationError, match="differ from raw recomputation"):
        load_and_recompute_reward_calibration_report(
            report_path, teacher_checkpoint_path=checkpoint
        )


def test_failed_diagnostic_report_remains_usable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    artifact = _artifact(checkpoint)
    artifact["term_weights"]["zmp_margin_barrier"] = 2.0
    _, report_path, calibration = _write_artifacts(tmp_path, artifact)
    assert calibration["status"] == "failed"

    loaded = load_and_recompute_reward_calibration_report(
        report_path, teacher_checkpoint_path=checkpoint
    )
    assert loaded["calibration"]["status"] == "failed"
    assert loaded["report_path"] == str(report_path.resolve())


def test_reward_report_uses_one_stable_path(tmp_path: Path) -> None:
    first = write_reward_calibration_json(tmp_path, {"schema_version": 1, "value": 1})
    second = write_reward_calibration_json(tmp_path, {"schema_version": 1, "value": 2})
    assert first == second == tmp_path.resolve() / "reward_calibration.json"
    assert json.loads(second.read_text(encoding="ascii"))["value"] == 2


def test_reward_normalization_constants_match_runtime() -> None:
    assert rewards.REWARD_NORMALIZATION_SCALES == GUIDE_REWARD_NORMALIZATION_SCALES
    assert set(rewards.REWARD_WEIGHTS) == set(GUIDE_REWARD_TERMS)
    assert rewards.REWARD_WEIGHTS["zmp_margin_barrier"] == pytest.approx(0.0)
    assert rewards.REWARD_WEIGHTS["fat2_prior_exp"] == pytest.approx(0.0)
    power = rewards.joint_power_l1_value(
        torch.tensor([[2.0, -3.0]]), torch.tensor([[4.0, 5.0]])
    )
    torch.testing.assert_close(power, torch.tensor([23.0]))
