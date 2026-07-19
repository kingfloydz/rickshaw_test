#!/usr/bin/env python3
"""Calibrate every unweighted reward term on a fixed-seed C1 policy rollout."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from _isaaclab_wrappers import add_isaaclab_sources_to_path, add_project_source_to_path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
add_project_source_to_path()

from g1_rickshaw_lab.reward_calibration import (  # noqa: E402
    C1_NOMINAL_PHYSICS_FIELDS,
    GUIDE_REWARD_NORMALIZATION_SCALES,
    GUIDE_REWARD_TERMS,
    NORMAL_SAMPLE_DEFINITION,
    RAW_REWARD_SAMPLE_KIND,
    RAW_REWARD_SAMPLE_SCHEMA_VERSION,
    REWARD_CALIBRATION_SCHEMA_VERSION,
    SIGNED_C1_SLOPES,
    RewardCalibrationError,
    collect_reward_manager_unweighted_step,
    load_and_recompute_reward_calibration_report,
    load_raw_reward_sample_artifact,
    recompute_reward_calibration,
    reward_calibration_guide_contract,
    reward_calibration_runtime_versions,
    reward_manager_term_weights,
    reward_sample_report_source,
    utc_timestamp,
    validate_raw_sample_artifact,
    validate_c1_physics_snapshot,
    validate_sample_checkpoint_binding,
    write_reward_calibration_json,
)
from g1_rickshaw_lab.reward_profile import (  # noqa: E402
    apply_reward_weight_overrides,
    reward_weight_overrides_from_configuration,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_STAGE_KEY,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
    normalize_rsl_rl_runner_configuration,
)
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)


DEFAULT_TASK = "Isaac-G1-Rickshaw-Directional-Slope-v0"
SIGNED_SLOPES = SIGNED_C1_SLOPES


def _parser() -> argparse.ArgumentParser:
    add_isaaclab_sources_to_path()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--samples",
        type=Path,
        help="Previously exported .pt raw RewardManager samples.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--policy-kind", choices=("auto", "teacher", "student"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS)
    parser.add_argument("--samples-per-slope", type=int, default=10_000)
    parser.add_argument("--max-policy-steps", type=int, default=5_000)
    parser.add_argument(
        "--feasibility-envelope",
        type=Path,
        default=REPOSITORY_ROOT / "config/feasibility_envelope.yaml",
    )
    parser.add_argument(
        "--reset-poses",
        type=Path,
        default=REPOSITORY_ROOT / "config/reset_poses.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/reward_calibration",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.samples is not None:
        if not args.samples.is_file():
            parser.error(f"raw sample artifact does not exist: {args.samples}")
    else:
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
    for label, path in (
        ("feasibility envelope", args.feasibility_envelope),
        ("reset poses", args.reset_poses),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.samples is not None:
        return
    if args.seed < 0:
        parser.error("--seed must be non-negative so the C1 rollout is reproducible")
    if args.num_envs <= 0 or args.num_envs % len(SIGNED_SLOPES) != 0:
        parser.error(f"--num-envs must be a positive multiple of {len(SIGNED_SLOPES)}")
    if args.samples_per_slope <= 0 or args.max_policy_steps <= 0:
        parser.error("sample quota and maximum policy steps must be positive")


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    return dict(load_raw_reward_sample_artifact(path))


def _write_raw_samples(output_dir: Path, artifact: dict[str, Any]) -> Path:
    import torch

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".reward_samples.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(artifact, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        destination = output_dir / "reward_samples.pt"
        os.replace(temporary, destination)
        return destination
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint_header(
    path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    checkpoint = load_stage_checkpoint(
        path,
        expected_stage={"s0_teacher", "s2_student_ppo"},
        validate_runtime=True,
    )
    stage = checkpoint[CHECKPOINT_STAGE_KEY]
    training_configuration = dict(checkpoint[TRAINING_CONFIGURATION_KEY])
    iteration = checkpoint.get("g1_rickshaw_curriculum_iteration")
    header = {
        "path": os.fspath(path.resolve()),
        "stage": stage,
        "curriculum_iteration": (
            iteration
            if isinstance(iteration, int) and not isinstance(iteration, bool)
            else None
        ),
        "training_parameters": dict(training_configuration["training_parameters"]),
    }
    return header, checkpoint, training_configuration


def _resolve_policy_kind(requested: str, checkpoint: dict[str, Any]) -> str:
    stage = checkpoint["stage"]
    expected = {
        "s0_teacher": "teacher",
        "s2_student_ppo": "student",
    }[stage]
    if requested != "auto" and requested != expected:
        raise RewardCalibrationError(
            f"--policy-kind {requested} is incompatible with checkpoint stage {stage}"
        )
    return expected


def _configure_training(env_cfg: Any) -> None:
    env_cfg.curriculum = None
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.domain_randomization = replace(
        env_cfg.domain_randomization,
        enabled=False,
    )
    env_cfg.events.initialize_domain.params = {"cfg": env_cfg.domain_randomization}


def _assign_fixed_slopes(base_env: Any):
    import torch
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    slots = torch.arange(base_env.num_envs, device=base_env.device) % len(SIGNED_SLOPES)
    levels = torch.tensor(
        SLOPE_TERRAIN_LEVELS, device=base_env.device, dtype=torch.long
    )[slots]
    columns = torch.tensor(
        SLOPE_TERRAIN_TYPES, device=base_env.device, dtype=torch.long
    )[slots]
    terrain = base_env.scene.terrain
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
    mdp.update_slope_frame(base_env)
    expected = torch.tensor(SIGNED_SLOPES, device=base_env.device)[slots]
    if not torch.allclose(base_env.slope, expected, atol=1.0e-7, rtol=0.0):
        raise RewardCalibrationError(
            "fixed terrain assignment did not resolve to every configured slope"
        )
    return slots


def _select_quota_ids(valid: Any, slope_slots: Any, counts: Any, quota: int):
    import torch

    selected: list[Any] = []
    for slope_index in range(len(SIGNED_SLOPES)):
        remaining = quota - int(counts[slope_index].item())
        if remaining <= 0:
            continue
        candidates = torch.nonzero(
            valid & (slope_slots == slope_index), as_tuple=False
        ).flatten()
        chosen = candidates[:remaining]
        if chosen.numel() > 0:
            selected.append(chosen)
            counts[slope_index] += chosen.numel()
    if not selected:
        return torch.empty(0, dtype=torch.long, device=valid.device)
    return torch.cat(selected)


def _physics_snapshot(
    base_env: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    import torch

    nominal_source = base_env.cfg.domain_randomization.nominal
    if not isinstance(nominal_source, Mapping):
        raise RewardCalibrationError("C1 environment has no fixed physics values")
    num_envs = base_env.num_envs
    device = base_env.device
    d6_cfg = base_env.d6_constraint_manager.cfg
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
    actual_values = {
        "torso.mass_delta": base_env.torso_mass_delta,
        "payload.mass": base_env._payload_mass,
        "payload.com.x": base_env._payload_com[:, 0],
        "payload.com.y": base_env._payload_com[:, 1],
        "payload.com.z": base_env._payload_com[:, 2],
        "rolling_resistance.c_rr": base_env.c_rr,
        "terrain.friction": base_env.terrain_friction,
        "wheel.left_damping": base_env._wheel_damping[:, 0],
        "wheel.right_damping": base_env._wheel_damping[:, 1],
    }
    for field in d6_fields:
        actual_values[f"d6.{field}"] = torch.full(
            (num_envs,), float(getattr(d6_cfg, field)), device=device
        )
    if set(actual_values) != set(C1_NOMINAL_PHYSICS_FIELDS):
        raise RewardCalibrationError("C1 runtime physical fields are incomplete")
    nominal_values = dict(nominal_source)
    nominal_values.update(
        {f"d6.{field}": float(getattr(d6_cfg, field)) for field in d6_fields}
    )
    missing_nominal = sorted(set(C1_NOMINAL_PHYSICS_FIELDS) - set(nominal_values))
    if missing_nominal:
        raise RewardCalibrationError(
            f"C1 runtime nominal physics fields are incomplete: missing={missing_nominal}"
        )
    nominal_values = {
        name: float(nominal_values[name]) for name in C1_NOMINAL_PHYSICS_FIELDS
    }
    result: dict[str, dict[str, float]] = {}
    for name, value in sorted(actual_values.items()):
        tensor = value.detach().float()
        if tensor.numel() == 0 or torch.any(~torch.isfinite(tensor)):
            raise RewardCalibrationError(
                f"C1 physical value {name!r} is empty or non-finite"
            )
        minimum = float(torch.amin(tensor).cpu())
        maximum = float(torch.amax(tensor).cpu())
        result[name] = {"minimum": minimum, "maximum": maximum}
    validate_c1_physics_snapshot(result, nominal_values)
    return result, nominal_values


def _collect_runtime_samples(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401

    checkpoint, _checkpoint_payload, training_configuration = _checkpoint_header(
        args.checkpoint
    )
    policy_kind = _resolve_policy_kind(args.policy_kind, checkpoint)
    if training_configuration["task"] != args.task:
        raise RewardCalibrationError(
            "reward calibration task differs from the checkpoint training task"
        )
    device = args.device
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    _configure_training(env_cfg)
    training_parameters = training_configuration["training_parameters"]
    apply_reward_weight_overrides(
        env_cfg,
        reward_weight_overrides_from_configuration(training_configuration),
    )
    env_cfg.rewards.fat2_prior_exp.weight = float(training_parameters["fat2_weight"])
    registry_key = (
        "rsl_rl_cfg_entry_point"
        if policy_kind == "teacher"
        else "rsl_rl_student_cfg_entry_point"
    )
    agent_cfg = load_cfg_from_registry(args.task, registry_key)
    agent_cfg.device = device
    agent_cfg.actor.latent_dim = int(training_parameters["latent_dim"])
    agent_cfg = normalize_rsl_rl_runner_configuration(agent_cfg)
    raw_env = gym.make(args.task, cfg=env_cfg)
    env = None
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = raw_env.unwrapped
        env.seed(args.seed)
        slope_slots = _assign_fixed_slopes(base_env)
        observation, _ = env.reset()
        c1_physics, c1_nominal_values = _physics_snapshot(base_env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(
            os.fspath(args.checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        policy = runner.get_inference_policy(device=device)
        weights = reward_manager_term_weights(base_env.reward_manager)
        pending: dict[str, list[Any]] = {name: [] for name in GUIDE_REWARD_TERMS}
        pending_slope_indices: list[Any] = []
        sources: dict[str, str] | None = None
        counts = torch.zeros(len(SIGNED_SLOPES), dtype=torch.long, device=device)
        done_rejected = 0
        policy_steps = 0
        while torch.any(counts < args.samples_per_slope):
            if policy_steps >= args.max_policy_steps:
                raise RewardCalibrationError(
                    "fixed C1 rollout exceeded --max-policy-steps before reaching per-slope quota; "
                    f"counts={counts.detach().cpu().tolist()}"
                )
            with torch.inference_mode():
                actions = policy(observation)
                next_observation, _, dones, extras = env.step(actions)
            time_outs = extras["time_outs"]
            if torch.any(
                dones.to(dtype=torch.bool)
                != (base_env.reset_terminated | time_outs.to(dtype=torch.bool))
            ):
                raise RewardCalibrationError(
                    "RSL wrapper done mask differs from termination/timeout buffers"
                )
            raw_step, step_sources = collect_reward_manager_unweighted_step(
                base_env.reward_manager
            )
            if sources is None:
                sources = step_sources
            elif sources != step_sources:
                raise RewardCalibrationError(
                    "RewardManager term extraction source changed during rollout"
                )
            done_mask = dones.to(dtype=torch.bool)
            valid = ~done_mask
            done_rejected += int(torch.sum(done_mask).item())
            selected = _select_quota_ids(
                valid, slope_slots, counts, args.samples_per_slope
            )
            pending_slope_indices.append(
                slope_slots[selected].detach().to(dtype=torch.long).cpu()
            )
            for name in GUIDE_REWARD_TERMS:
                pending[name].append(raw_step[name][selected].detach().float().cpu())
            observation = next_observation
            policy.reset(dones)
            policy_steps += 1
        raw_terms = {name: torch.cat(values) for name, values in pending.items()}
        sample_slope_indices = torch.cat(pending_slope_indices)
        slope_counts = {
            f"{slope:+.2f}": int(count)
            for slope, count in zip(
                SIGNED_SLOPES, counts.detach().cpu().tolist(), strict=True
            )
        }
        artifact = {
            "schema_version": RAW_REWARD_SAMPLE_SCHEMA_VERSION,
            "kind": RAW_REWARD_SAMPLE_KIND,
            "created_at_utc": utc_timestamp(),
            "curriculum_stage": "TRAINING",
            "fixed_seed": args.seed,
            "fixed_slopes": list(SIGNED_SLOPES),
            "slope_sample_counts": slope_counts,
            "normal_sample_definition": NORMAL_SAMPLE_DEFINITION,
            "task": args.task,
            "num_envs": args.num_envs,
            "policy_steps": policy_steps,
            "step_dt_s": float(base_env.step_dt),
            "rejected_samples": {
                "terminated_or_timeout": done_rejected,
            },
            "policy_kind": policy_kind,
            "checkpoint": checkpoint,
            "c1_physics": c1_physics,
            "c1_nominal_values": c1_nominal_values,
            "term_weights": weights,
            "term_sources": sources,
            "reward_normalization_scales": GUIDE_REWARD_NORMALIZATION_SCALES,
            "runtime_versions": reward_calibration_runtime_versions(),
            "raw_terms": raw_terms,
            "sample_slope_indices": sample_slope_indices,
        }
        validate_raw_sample_artifact(artifact)
        path = _write_raw_samples(args.output_dir, artifact)
        return artifact, path
    finally:
        if env is not None:
            env.close()
        else:
            raw_env.close()


def _report_from_artifact(
    artifact: dict[str, Any],
    sample_path: Path,
) -> dict[str, Any]:
    validate_raw_sample_artifact(artifact)
    validate_sample_checkpoint_binding(artifact)
    calibration = recompute_reward_calibration(artifact)
    source = reward_sample_report_source(artifact)
    return {
        "schema_version": REWARD_CALIBRATION_SCHEMA_VERSION,
        "tool": "calibrate_rewards",
        "created_at_utc": utc_timestamp(),
        "status": calibration["status"],
        "guide_contract": reward_calibration_guide_contract(),
        "raw_sample_artifact": {
            "path": os.fspath(sample_path.resolve()),
        },
        "source": source,
        "calibration": calibration,
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    simulation_app = None
    try:
        if args.samples is not None:
            artifact = _load_torch_mapping(args.samples)
            sample_path = args.samples
        else:
            os.environ["G1_RICKSHAW_FEASIBILITY_ENVELOPE"] = os.fspath(
                args.feasibility_envelope.resolve()
            )
            os.environ["G1_RICKSHAW_RESET_POSES"] = os.fspath(
                args.reset_poses.resolve()
            )
            from isaaclab.app import AppLauncher

            args.headless = True
            app_launcher = AppLauncher(args)
            simulation_app = app_launcher.app
            artifact, sample_path = _collect_runtime_samples(args)
        report = _report_from_artifact(
            artifact,
            sample_path,
        )
        report_path = write_reward_calibration_json(args.output_dir, report)
        load_and_recompute_reward_calibration_report(
            report_path,
            teacher_checkpoint_path=(
                args.checkpoint if artifact.get("policy_kind") == "teacher" else None
            ),
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": os.fspath(report_path),
                    "raw_samples": os.fspath(sample_path.resolve()),
                    "failures": report["calibration"]["failures"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except BaseException:
        # Process teardown releases Kit resources. Closing SimulationApp here
        # can terminate the interpreter and hide the original exception.
        raise
    if simulation_app is not None:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RewardCalibrationError as error:
        print(f"reward calibration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
