from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from g1_rickshaw_lab.reward_profile import (
    apply_reward_weight_overrides,
    parse_reward_weight_arguments,
    reward_weight_hydra_overrides,
    reward_weight_overrides_from_configuration,
    validate_reward_weight_overrides,
)
from g1_rickshaw_lab.reward_tuning import (
    RANK_METRICS,
    aggregate_profile_results,
    factorial_effects,
    factorial_reward_profiles,
    load_reward_tuning_config,
    policy_diagnostic_rank_metrics,
)
from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_reward_tuning as pipeline  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reward_tuning.yaml"
QUALITY_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "reward_tuning_quality.yaml"
)


def test_reward_weight_override_contract_is_canonical() -> None:
    overrides = parse_reward_weight_arguments(
        ["feet_slide=-0.2", "zmp_margin_barrier=-3"]
    )
    configuration = {"resolved_parameters": {"reward_weight_overrides": overrides}}

    assert reward_weight_overrides_from_configuration(configuration) == {
        "zmp_margin_barrier": -3.0,
        "feet_slide": -0.2,
    }
    assert reward_weight_hydra_overrides(overrides) == [
        "env.rewards.zmp_margin_barrier.weight=-3.0",
        "env.rewards.feet_slide.weight=-0.2",
    ]

    env_cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            zmp_margin_barrier=SimpleNamespace(weight=-2.0),
            feet_slide=SimpleNamespace(weight=-0.1),
        )
    )
    apply_reward_weight_overrides(env_cfg, overrides)
    assert env_cfg.rewards.zmp_margin_barrier.weight == -3.0
    assert env_cfg.rewards.feet_slide.weight == -0.2


@pytest.mark.parametrize(
    "values",
    (
        ["unknown=-1"],
        ["fat2_prior_exp=0.2"],
        ["feet_slide=-0.1", "feet_slide=-0.2"],
        ["feet_slide=nan"],
        ["feet_slide"],
    ),
)
def test_reward_weight_arguments_reject_invalid_profiles(values: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_reward_weight_arguments(values)


def test_generic_reward_profile_cannot_duplicate_the_fat2_parameter() -> None:
    with pytest.raises(ValueError, match="dedicated FAT2"):
        validate_reward_weight_overrides({"fat2_prior_exp": 0.2})


def test_factorial_configuration_generates_the_exact_eight_profiles() -> None:
    config = load_reward_tuning_config(CONFIG_PATH)
    profiles = factorial_reward_profiles(config)

    assert [profile["name"] for profile in profiles] == [
        "baseline",
        "zmp_0",
        "hitch_050",
        "slide_020",
        "zmp_0_hitch_050",
        "zmp_0_slide_020",
        "hitch_050_slide_020",
        "zmp_0_hitch_050_slide_020",
    ]
    assert sum(profile["levels"]["zmp"] == "high" for profile in profiles) == 4
    assert {
        profile["reward_weight_overrides"]["zmp_margin_barrier"]
        for profile in profiles
        if profile["levels"]["zmp"] == "high"
    } == {0.0}
    assert {
        profile["reward_weight_overrides"]["zmp_margin_barrier"]
        for profile in profiles
        if profile["levels"]["zmp"] == "low"
    } == {-2.0}
    assert len(
        {
            tuple(profile["reward_weight_overrides"].items())
            for profile in profiles
        }
    ) == 8


def test_quality_configuration_generates_the_exact_second_factorial() -> None:
    config = load_reward_tuning_config(QUALITY_CONFIG_PATH)
    profiles = factorial_reward_profiles(config)

    assert [profile["name"] for profile in profiles] == [
        "baseline",
        "normal_100",
        "power_0002",
        "rate_002",
        "normal_100_power_0002",
        "normal_100_rate_002",
        "power_0002_rate_002",
        "normal_100_power_0002_rate_002",
    ]
    assert len(
        {
            tuple(profile["reward_weight_overrides"].items())
            for profile in profiles
        }
    ) == 8
    assert {
        profile["reward_weight_overrides"]["zmp_margin_barrier"]
        for profile in profiles
    } == {0.0}
    anchor = profiles[0]["reward_weight_overrides"]
    assert anchor == {
        "zmp_margin_barrier": 0.0,
        "terrain_normal_velocity_l2": -0.5,
        "joint_power_l1": -0.0001,
        "processed_action_rate_l2": -0.01,
    }
    strongest = profiles[-1]["reward_weight_overrides"]
    assert strongest == {
        "zmp_margin_barrier": 0.0,
        "terrain_normal_velocity_l2": -1.0,
        "joint_power_l1": -0.0002,
        "processed_action_rate_l2": -0.02,
    }


def test_reward_commands_use_the_isolated_logical_gpu(tmp_path: Path) -> None:
    config = load_reward_tuning_config(CONFIG_PATH)
    profile = factorial_reward_profiles(config)[0]
    job = pipeline.RewardJob(profile, training_seed=42, gpu_index=6)
    args = SimpleNamespace(
        evaluation_seeds=(42, 43, 44, 45, 46),
        evaluation_num_envs=380,
        episodes_per_slope=100,
    )
    checkpoint = tmp_path / "model.pt"
    commands = (
        pipeline._teacher_command(job, config, tmp_path, None),
        pipeline._evaluation_command(
            checkpoint, tmp_path / "diagnostic.json", args, config
        ),
        pipeline._calibration_command(
            checkpoint, tmp_path / "calibration", args, config
        ),
    )

    assert {
        command[command.index("--device") + 1] for command in commands
    } == {"cuda:0"}


def test_reward_worker_combines_physical_isolation_with_logical_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_reward_tuning_config(CONFIG_PATH)
    profile = factorial_reward_profiles(config)[0]
    job = pipeline.RewardJob(profile, training_seed=42, gpu_index=6)
    run_dir = tmp_path / "runs" / profile["name"] / "seed_00042"
    diagnostic = run_dir / "policy_diagnostic.json"
    calibration = run_dir / "reward_calibration" / "reward_calibration.json"
    diagnostic.parent.mkdir(parents=True)
    calibration.parent.mkdir(parents=True)
    diagnostic.write_text(json.dumps(_diagnostic()), encoding="utf-8")
    calibration.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"checkpoint")
    record = pipeline.multi_gpu.CheckpointRecord(checkpoint, 6000, True)
    checkpoint_calls = 0

    def find_checkpoint(*_args, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return None if checkpoint_calls == 1 else record

    launches: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(pipeline, "_checkpoint", find_checkpoint)
    monkeypatch.setattr(pipeline, "_valid_diagnostic", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pipeline, "_valid_calibration", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        pipeline.multi_gpu,
        "_run_command",
        lambda command, *, environment, **_kwargs: launches.append(
            (list(command), dict(environment))
        ),
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        evaluation_seeds=(42, 43, 44, 45, 46),
        evaluation_num_envs=380,
        episodes_per_slope=100,
    )

    pipeline._run_job(
        job,
        args,
        config,
        pipeline.multi_gpu.ProcessRegistry(),
        threading.Event(),
    )

    assert len(launches) == 1
    command, environment = launches[0]
    assert environment["CUDA_VISIBLE_DEVICES"] == "6"
    assert command[command.index("--device") + 1] == "cuda:0"


def _diagnostic(*, fall_rate: float = 0.1, speed: float = 0.2) -> dict:
    metrics = {
        "non_finite_sample_counts": {},
        "episodes": {"fall_rate": fall_rate},
        "tracking": {
            "speed_rmse_mps": speed,
            "overspeed_rate": 0.03,
            "lateral_error": {"rms_m": 0.04},
            "heading_error": {"rms_rad": 0.05},
        },
        "stability": {"zmp_margin_m": {"p01": -0.01}},
        "rickshaw": {
            "two_wheel_contact_rate": 0.98,
            "hitch_height_error": {"rms_m": 0.02},
        },
        "locomotion": {"foot_slip_mps": {"p90": 0.1}},
        "actions": {"processed_rate_radps": {"p90": 0.5}},
        "actuation": {"power_w": {"p90": 500.0}},
    }
    per_slope = {
        f"{slope:+.2f}": {"episodes": {"fall_rate": fall_rate}}
        for slope in SLOPE_GRADIENTS
    }
    per_slope["-0.08"]["episodes"]["fall_rate"] = fall_rate + 0.05
    return {
        "status": "recorded",
        "stages": {
            "training": {
                "metrics": metrics,
                "per_slope": per_slope,
            }
        },
    }


def test_diagnostic_ranking_uses_physical_metrics_not_return() -> None:
    metrics = policy_diagnostic_rank_metrics(_diagnostic())

    assert set(metrics) == set(RANK_METRICS)
    assert metrics["worst_slope_fall_rate"] == pytest.approx(0.15)
    assert "return" not in metrics


def _record(
    profile: str,
    seed: int,
    *,
    fall_rate: float,
    calibration_status: str = "passed",
) -> dict:
    metrics = {name: 1.0 for name in RANK_METRICS}
    metrics["fall_rate"] = fall_rate
    metrics["worst_slope_fall_rate"] = fall_rate
    metrics["zmp_margin_p01_m"] = -0.01
    metrics["two_wheel_contact_rate"] = 0.98
    return {
        "profile": profile,
        "training_seed": seed,
        "calibration_status": calibration_status,
        "metrics": metrics,
    }


def test_ranking_prefers_calibration_pass_then_robust_fall_rate() -> None:
    ranking = aggregate_profile_results(
        [
            _record("safe", 42, fall_rate=0.1),
            _record("safe", 43, fall_rate=0.2),
            _record("unsafe", 42, fall_rate=0.0, calibration_status="failed"),
        ]
    )

    assert [item["profile"] for item in ranking] == ["safe", "unsafe"]
    assert ranking[0]["robust_metrics"]["fall_rate"] == pytest.approx(0.2)


def test_factorial_effects_recover_an_injected_main_effect() -> None:
    config = load_reward_tuning_config(CONFIG_PATH)
    profiles = factorial_reward_profiles(config)
    records = []
    for profile in profiles:
        fall = 0.1 + (0.04 if profile["levels"]["zmp"] == "high" else 0.0)
        records.append(_record(profile["name"], 42, fall_rate=fall))
    ranking = aggregate_profile_results(records)

    effects = factorial_effects(ranking, profiles, config["factors"])

    assert effects["effects"]["zmp"]["fall_rate"] == pytest.approx(0.04)
    assert effects["effects"]["hitch"]["fall_rate"] == pytest.approx(0.0)
    assert effects["effects"]["zmp:hitch"]["fall_rate"] == pytest.approx(0.0)


def test_screen_plan_maps_eight_profiles_to_eight_gpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = pipeline.main(
        [
            "--config",
            os.fspath(CONFIG_PATH),
            "--output-dir",
            os.fspath(tmp_path / "screen"),
            "--gpus",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "--plan-only",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["jobs"]) == 8
    assert [job["gpu"] for job in plan["jobs"]] == list(range(8))
    assert plan["output_dir"] == os.fspath((tmp_path / "screen").resolve())


def test_quality_batch_uses_an_independent_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "reward_tuning_screen_v2"

    assert (
        pipeline.main(
            [
                "--config",
                os.fspath(QUALITY_CONFIG_PATH),
                "--output-dir",
                os.fspath(output_dir),
                "--training-seeds",
                "43",
                "--gpus",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "--plan-only",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["output_dir"] == os.fspath(output_dir.resolve())
    assert plan["tensorboard_logdir"] == os.fspath(
        (output_dir / "runs").resolve()
    )
    assert len(plan["jobs"]) == 8
    assert [job["gpu"] for job in plan["jobs"]] == list(range(8))
    assert {job["training_seed"] for job in plan["jobs"]} == {43}
    assert plan["jobs"][0]["profile"] == "baseline"
    assert plan["jobs"][0]["reward_weight_overrides"] == {
        "zmp_margin_barrier": 0.0,
        "terrain_normal_velocity_l2": -0.5,
        "joint_power_l1": -0.0001,
        "processed_action_rate_l2": -0.01,
    }
    assert not output_dir.exists()


def test_confirmation_plan_expands_top_two_over_four_training_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps(
            {
                "config": load_reward_tuning_config(CONFIG_PATH),
                "ranking": [
                    {"profile": "zmp_0"},
                    {"profile": "baseline"},
                    {"profile": "slide_020"},
                ]
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        "--config",
        os.fspath(CONFIG_PATH),
        "--output-dir",
        os.fspath(tmp_path / "confirm"),
        "--top-from",
        os.fspath(screen),
        "--top-k",
        "2",
        "--training-seeds",
        "101",
        "102",
        "103",
        "104",
        "--gpus",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "--plan-only",
    ]

    assert pipeline.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["jobs"]) == 8
    assert {job["profile"] for job in plan["jobs"]} == {"zmp_0", "baseline"}
    assert {job["training_seed"] for job in plan["jobs"]} == {101, 102, 103, 104}


def test_diagnostic_resume_binds_the_complete_evaluation_protocol(
    tmp_path: Path,
) -> None:
    config = load_reward_tuning_config(CONFIG_PATH)
    profile = factorial_reward_profiles(config)[0]
    job = pipeline.RewardJob(profile, training_seed=42, gpu_index=0)
    args = SimpleNamespace(
        evaluation_seeds=(42, 43, 44, 45, 46),
        evaluation_num_envs=380,
        episodes_per_slope=100,
    )
    checkpoint = tmp_path / "model_5999.pt"
    checkpoint.write_bytes(b"checkpoint")
    report_path = tmp_path / "diagnostic.json"
    report = _diagnostic()
    report.update(
        {
            "schema_version": 1,
            "task": config["task"],
            "checkpoint": {
                "path": os.fspath(checkpoint.resolve()),
                "stage": "s0_teacher",
            },
            "evaluation": {
                "deterministic_actions": True,
                "fixed_seeds": list(args.evaluation_seeds),
                "signed_slopes": list(SLOPE_GRADIENTS),
                "num_envs": args.evaluation_num_envs,
                "episodes_per_slope_per_stage": args.episodes_per_slope,
                "curriculum_stages": ["training"],
                "command_protocol": pipeline.FORMAL_EVALUATION_COMMAND_PROTOCOL,
                "cross_case_protocol": pipeline.FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
                "fat2_weight": config["fixed"]["fat2_weight"],
                "rollout_steps": config["fixed"]["rollout_steps"],
                "latent_dim": config["fixed"]["latent_dim"],
                "reward_weight_overrides": profile["reward_weight_overrides"],
            },
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    timestamp = checkpoint.stat().st_mtime_ns + 1_000_000
    os.utime(report_path, ns=(timestamp, timestamp))

    assert pipeline._valid_diagnostic(
        report_path, checkpoint, job, args, config
    )
    report["evaluation"]["curriculum_stages"] = ["nominal"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not pipeline._valid_diagnostic(
        report_path, checkpoint, job, args, config
    )


def test_calibration_resume_recomputes_and_binds_collection_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_reward_tuning_config(CONFIG_PATH)
    profile = factorial_reward_profiles(config)[0]
    job = pipeline.RewardJob(profile, training_seed=42, gpu_index=0)
    args = SimpleNamespace(
        evaluation_seeds=(42, 43, 44, 45, 46),
        evaluation_num_envs=380,
    )
    checkpoint = tmp_path / "model_5999.pt"
    checkpoint.write_bytes(b"checkpoint")
    report_path = tmp_path / "reward_calibration.json"
    report_path.write_text("{}", encoding="utf-8")
    timestamp = checkpoint.stat().st_mtime_ns + 1_000_000
    os.utime(report_path, ns=(timestamp, timestamp))
    source = {
        "checkpoint": {"path": os.fspath(checkpoint.resolve())},
        "fixed_seed": args.evaluation_seeds[0],
        "fixed_slopes": list(SLOPE_GRADIENTS),
        "task": config["task"],
        "num_envs": args.evaluation_num_envs,
        "policy_kind": "teacher",
        "slope_sample_counts": {
            f"{slope:+.2f}": config["calibration"]["samples_per_slope"]
            for slope in SLOPE_GRADIENTS
        },
        "policy_steps": config["calibration"]["max_policy_steps"],
        "term_weights": {
            **profile["reward_weight_overrides"],
            "fat2_prior_exp": config["fixed"]["fat2_weight"],
        },
    }
    report = {"status": "passed", "source": source}
    calls: list[tuple[Path, Path]] = []

    def load_report(path: Path, *, teacher_checkpoint_path: Path) -> dict:
        calls.append((path, teacher_checkpoint_path))
        return {"report": report}

    monkeypatch.setattr(
        pipeline, "load_and_recompute_reward_calibration_report", load_report
    )

    assert pipeline._valid_calibration(
        report_path, checkpoint, job, args, config
    )
    assert calls == [(report_path, checkpoint)]
    source["fixed_seed"] = 999
    assert not pipeline._valid_calibration(
        report_path, checkpoint, job, args, config
    )
