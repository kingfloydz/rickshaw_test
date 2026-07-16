#!/usr/bin/env python3
"""Collect verified S0 teacher on-policy rollouts for S1 distillation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import numpy as np

from _isaaclab_wrappers import (
    add_isaaclab_sources_to_path,
    add_project_source_to_path,
    require_existing_file,
)

add_project_source_to_path()

from g1_rickshaw_lab.provenance import atomic_torch_save, extract_checkpoint_metadata, sha256_file  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
    require_pinned_rsl_rl,
    validate_rollout_stage_coverage,
)
from g1_rickshaw_lab.validation import write_json_atomic  # noqa: E402

from _rollout_audit import (  # noqa: E402
    AUDIT_TENSOR_NAMES,
    FORMAL_NUM_ENVS,
    INTEGER_AUDIT_TENSORS,
    PHYSICS_VALUE_NAMES,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
    SIGNED_SLOPES,
    SLOPE_LABELS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    canonical_sha256,
    formal_slope_environment_assignment,
    summarize_segment_samples,
    validate_rollout_sample_audit,
)


DEFAULT_TASK = "Isaac-G1-Rickshaw-Directional-Slope-v0"


def main() -> int:  # noqa: C901
    add_isaaclab_sources_to_path()
    require_pinned_rsl_rl()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=FORMAL_NUM_ENVS)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="Required valid policy transitions per environment.",
    )
    parser.add_argument(
        "--shard-steps",
        type=int,
        default=4,
        help="Policy steps buffered per shard; four bounds host memory at 4096 environments.",
    )
    parser.add_argument("--max-collection-multiplier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--training-iteration", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (
        args.num_steps <= 0
        or args.shard_steps <= 0
        or args.max_collection_multiplier <= 0
    ):
        raise ValueError(
            "--num-steps/--shard-steps/--max-collection-multiplier must be positive"
        )
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise ValueError("formal rollout collection seed must lie in [0, 2**32-1]")
    if args.num_envs != FORMAL_NUM_ENVS:
        raise ValueError(
            "formal TRAINING rollout collection requires the Guide-fixed "
            f"--num-envs={FORMAL_NUM_ENVS}"
        )
    if args.training_iteration < 0:
        raise ValueError("--training-iteration must be non-negative")

    teacher_path = require_existing_file(args.teacher, "teacher checkpoint").resolve()
    teacher_checkpoint = load_stage_checkpoint(
        teacher_path,
        expected_stage="s0_teacher",
        validate_runtime=True,
    )
    teacher_training_configuration = dict(
        teacher_checkpoint[TRAINING_CONFIGURATION_KEY]
    )
    if teacher_training_configuration["task"] != args.task:
        raise ValueError("rollout task differs from the S0 teacher training task")
    teacher_ablation_values = teacher_training_configuration["ablation_values"]
    metadata = extract_checkpoint_metadata(teacher_checkpoint)
    output_dir = Path(args.output_dir).resolve()
    existing = list(output_dir.glob("rollout_*.pt")) + list(output_dir.glob("manifest.json"))
    if existing and not args.overwrite:
        raise FileExistsError(f"rollout output already contains artifacts: {output_dir}")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401
        from g1_rickshaw_lab.rl.rollout_labels import RolloutLabelTracker
        from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = args.device
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        # Formal rollouts bind immutable slopes.  This prevents auto-resets
        # from advancing terrain difficulty during a short segment.
        env_cfg.curriculum = None
        env_cfg.rewards.fat2_prior_exp.weight = float(
            teacher_ablation_values["fat2_weight"]
        )
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.device = device
        raw_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(
            os.fspath(teacher_path),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        actor = runner.alg.actor.eval()
        base_env = raw_env.unwrapped
        fixed_assignment = formal_slope_environment_assignment(
            args.num_envs, device=agent_cfg.device
        )
        curriculum_callback = getattr(base_env, "set_curriculum_iteration", None)
        if not callable(curriculum_callback):
            raise RuntimeError("environment does not expose set_curriculum_iteration")
        storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
        pending: dict[str, list[torch.Tensor]] = {}
        shard_hashes: dict[str, str] = {}
        stage_segments: list[dict] = []
        stage_sample_distribution: dict[str, int] = {}
        all_audit_chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in AUDIT_TENSOR_NAMES
        }
        shard_index = 0
        collected_steps = 0
        collected_samples = 0
        next_episode_id = 0

        def append(name: str, value: torch.Tensor) -> None:
            if name in INTEGER_AUDIT_TENSORS:
                stored = value.detach().to(device="cpu", dtype=torch.long)
            elif name in AUDIT_TENSOR_NAMES:
                stored = value.detach().to(device="cpu", dtype=torch.float32)
            else:
                stored = value.detach().to(device="cpu", dtype=storage_dtype)
            pending.setdefault(name, []).append(stored)

        def flush() -> None:
            nonlocal shard_index, collected_samples
            if not pending:
                return
            tensors = {name: torch.cat(values, dim=0) for name, values in pending.items()}
            samples = tensors["current"].shape[0]
            if samples == 0:
                pending.clear()
                return
            name = f"rollout_{shard_index:05d}.pt"
            path = output_dir / name
            atomic_torch_save(
                {
                    "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
                    "rollout": tensors,
                    "teacher_checkpoint_sha256": sha256_file(teacher_path),
                    "g1_rickshaw_provenance": metadata.to_mapping(),
                },
                path,
            )
            shard_hashes[name] = sha256_file(path)
            shard_index += 1
            collected_samples += samples
            pending.clear()

        def stage_name(value: int) -> str:
            enum_type = type(base_env.curriculum_runtime_state.stage)
            for item in enum_type:
                if int(item) == int(value):
                    return item.name
            raise RuntimeError(f"unknown curriculum stage value {value}")

        def install_fixed_rollout_assignment() -> None:
            """Install the deterministic configured-slope allocation before reset."""

            terrain = base_env.scene.terrain
            levels = fixed_assignment["terrain_level"]
            terrain_types = fixed_assignment["terrain_type"]
            terrain.terrain_levels.copy_(levels)
            terrain.terrain_types.copy_(terrain_types)
            terrain.env_origins.copy_(terrain.terrain_origins[levels, terrain_types])
            mdp.update_slope_frame(base_env)
            state = base_env.curriculum_runtime_state
            base_env.curriculum_stage_per_env.copy_(state.stage_per_environment())
            curriculum_log = getattr(base_env, "extras", {}).get("curriculum")
            if isinstance(curriculum_log, dict):
                curriculum_log["distribution"] = state.distribution()

        def assert_fixed_rollout_assignment() -> None:
            terrain = base_env.scene.terrain
            if not torch.equal(terrain.terrain_levels, fixed_assignment["terrain_level"]):
                raise RuntimeError("formal rollout terrain levels changed during collection")
            if not torch.equal(terrain.terrain_types, fixed_assignment["terrain_type"]):
                raise RuntimeError("formal rollout terrain types changed during collection")
            if not torch.allclose(
                base_env.slope,
                fixed_assignment["slope"],
                atol=1.0e-7,
                rtol=0.0,
            ):
                raise RuntimeError("formal rollout slopes changed during collection")
            if not torch.equal(
                base_env.curriculum_stage_per_env,
                base_env.curriculum_runtime_state.stage_per_environment(),
            ):
                raise RuntimeError("formal rollout per-environment curriculum stage changed")

        def reset_at_physx_fixed_point() -> object:
            observation, _ = env.reset()
            return observation.to(agent_cfg.device)

        runtime_cfg = base_env.cfg.runtime_randomization
        if tuple(runtime_cfg.teacher_extrinsic_names) != PHYSICS_VALUE_NAMES:
            raise RuntimeError("runtime teacher-extrinsic order differs from the rollout audit ABI")
        physics_bounds = {
            name: [float(runtime_cfg.ranges[name][0]), float(runtime_cfg.ranges[name][1])]
            for name in PHYSICS_VALUE_NAMES
        }
        sample_audit_contract = {
            "schema_version": ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
            "physics_value_names": list(PHYSICS_VALUE_NAMES),
            "signed_slopes": list(SIGNED_SLOPES),
            "terrain_levels": list(SLOPE_TERRAIN_LEVELS),
            "terrain_types": list(SLOPE_TERRAIN_TYPES),
            "formal_num_envs": FORMAL_NUM_ENVS,
            "physics_bounds": physics_bounds,
            "joint_model_error_bounds": [
                float(runtime_cfg.ranges["joint.model_error"][0]),
                float(runtime_cfg.ranges["joint.model_error"][1]),
            ],
            "collection_seed": args.seed,
            "episode_binding": "environment_id + monotonically assigned reset episode_id",
            "source": "per-transition shard tensors",
        }

        stage_specs = [("TRAINING", args.training_iteration)]

        target_samples = args.num_envs * args.num_steps
        for segment_index, (expected_stage, curriculum_iteration) in enumerate(stage_specs):
            curriculum_callback(curriculum_iteration)
            actual_global_stage = base_env.curriculum_runtime_state.stage.name
            if actual_global_stage != expected_stage:
                raise RuntimeError(
                    f"requested rollout stage {expected_stage} resolved to {actual_global_stage}"
                )

            install_fixed_rollout_assignment()
            observation = reset_at_physx_fixed_point()
            assert_fixed_rollout_assignment()
            tracker = RolloutLabelTracker(args.num_envs, device=agent_cfg.device)
            stage_valid_samples = 0
            stage_collection_steps = 0
            per_environment_samples = torch.zeros(
                args.num_envs, device=agent_cfg.device, dtype=torch.long
            )
            sample_distribution: dict[str, int] = {}
            stage_audit_chunks: dict[str, list[torch.Tensor]] = {
                name: [] for name in AUDIT_TENSOR_NAMES
            }
            episode_ids = torch.arange(
                next_episode_id,
                next_episode_id + args.num_envs,
                device=agent_cfg.device,
                dtype=torch.long,
            )
            next_episode_id += args.num_envs
            per_env_stage: dict[str, int] = {}
            for value, count in zip(
                *torch.unique(base_env.curriculum_stage_per_env, return_counts=True),
                strict=True,
            ):
                per_env_stage[stage_name(int(value))] = int(count)

            while stage_valid_samples < target_samples:
                if stage_collection_steps >= args.num_steps * args.max_collection_multiplier:
                    raise RuntimeError(
                        f"stage {expected_stage} failed to collect {target_samples} valid samples"
                    )
                assert_fixed_rollout_assignment()
                valid = torch.ones(args.num_envs, dtype=torch.bool, device=agent_cfg.device)
                valid &= per_environment_samples < args.num_steps
                valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                selected = valid_ids
                contact_sensor = base_env.scene["robot_contacts"]
                contact = contact_sensor.data.current_contact_time[:, base_env.foot_sensor_ids] > 0.0
                cart_velocity = base_env.scene["rickshaw"].data.root_lin_vel_w
                cart_speed = torch.sum(cart_velocity * base_env.path_tangent_w, dim=-1)
                labels = tracker.update(contact, base_env.command_state.v_ref, cart_speed)

                audit_values = None
                if selected.numel() > 0:
                    # Snapshot reset-scoped values before env.step() can auto-reset
                    # a completed episode and replace its randomization.
                    raw_physics = torch.stack(
                        [
                            base_env.teacher_extrinsic_values[name][selected]
                            for name in PHYSICS_VALUE_NAMES
                        ],
                        dim=-1,
                    )
                    audit_values = {
                        "teacher_extrinsics": observation["teacher_extrinsics"][selected].clone(),
                        "curriculum_stage": base_env.curriculum_stage_per_env[selected]
                        .unsqueeze(-1)
                        .clone(),
                        "collection_segment": torch.full(
                            (selected.numel(), 1),
                            segment_index,
                            device=agent_cfg.device,
                            dtype=torch.long,
                        ),
                        "environment_id": selected.unsqueeze(-1).clone(),
                        "episode_id": episode_ids[selected].unsqueeze(-1).clone(),
                        "slope": base_env.slope[selected].unsqueeze(-1).clone(),
                        "terrain_level": base_env.scene.terrain.terrain_levels[
                            selected
                        ]
                        .unsqueeze(-1)
                        .clone(),
                        "terrain_type": base_env.scene.terrain.terrain_types[selected]
                        .unsqueeze(-1)
                        .clone(),
                        "physics_values": raw_physics.clone(),
                        "joint_model_error": base_env.joint_model_error[selected].clone(),
                        "observation_noise_scale": base_env.observation_noise_scale[
                            selected
                        ]
                        .unsqueeze(-1)
                        .clone(),
                    }

                with torch.inference_mode():
                    actions = actor(observation, stochastic_output=True)
                    teacher_mean = actor.output_mean.clone()
                    teacher_std = actor.output_std.clone()
                    z_star = actor.encode(observation).clone()
                    next_observation, reward, dones, _ = env.step(actions)

                if selected.numel() > 0:
                    assert audit_values is not None
                    append("current", observation["policy"][selected])
                    append("history", observation["history"][selected])
                    append("teacher_action_mean", teacher_mean[selected])
                    append("teacher_action_std", teacher_std[selected])
                    append("z_star", z_star[selected])
                    for name, value in labels.items():
                        append(name, value[selected])
                    append("reward", reward.reshape(-1, 1)[selected])
                    for name, value in audit_values.items():
                        append(name, value)
                        cpu_value = value.detach().to(
                            device="cpu",
                            dtype=torch.long if name in INTEGER_AUDIT_TENSORS else torch.float32,
                        )
                        stage_audit_chunks[name].append(cpu_value)
                        all_audit_chunks[name].append(cpu_value)
                    selected_stages = audit_values["curriculum_stage"].reshape(-1)
                    for value, count in zip(
                        *torch.unique(selected_stages, return_counts=True),
                        strict=True,
                    ):
                        name = stage_name(int(value))
                        amount = int(count)
                        sample_distribution[name] = sample_distribution.get(name, 0) + amount
                        stage_sample_distribution[name] = stage_sample_distribution.get(name, 0) + amount
                    per_environment_samples[selected] += 1
                    stage_valid_samples += int(selected.numel())
                observation = next_observation.to(agent_cfg.device)
                tracker.reset(dones)
                done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
                if done_ids.numel() > 0:
                    episode_ids[done_ids] = torch.arange(
                        next_episode_id,
                        next_episode_id + done_ids.numel(),
                        device=agent_cfg.device,
                        dtype=torch.long,
                    )
                    next_episode_id += int(done_ids.numel())
                stage_collection_steps += 1
                collected_steps += 1
                if collected_steps % args.shard_steps == 0:
                    flush()

            if not torch.all(per_environment_samples == args.num_steps):
                raise RuntimeError(
                    f"stage {expected_stage} did not collect the exact per-environment quota"
                )
            segment_audit_tensors = {
                name: torch.cat(values, dim=0) for name, values in stage_audit_chunks.items()
            }
            actual_sample_audit = summarize_segment_samples(
                segment_audit_tensors,
                segment_index=segment_index,
                num_envs=args.num_envs,
                samples_per_environment=args.num_steps,
                physics_bounds=physics_bounds,
                joint_model_error_bounds=sample_audit_contract["joint_model_error_bounds"],
            )
            stage_segments.append(
                {
                    "global_stage": actual_global_stage,
                    "curriculum_iteration": curriculum_iteration,
                    "full_environment_reset": True,
                    "reset_policy_steps": 0,
                    "collection_policy_steps": stage_collection_steps,
                    "target_valid_samples": target_samples,
                    "valid_samples": stage_valid_samples,
                    "per_environment_stage_distribution": per_env_stage,
                    "valid_sample_stage_distribution": sample_distribution,
                    "slope_sample_distribution": actual_sample_audit[
                        "slope_sample_distribution"
                    ],
                    "slope_environment_distribution": actual_sample_audit[
                        "slope_environment_distribution"
                    ],
                    "slope_episode_distribution": actual_sample_audit[
                        "slope_episode_distribution"
                    ],
                    "terrain_level_distribution": actual_sample_audit[
                        "terrain_level_distribution"
                    ],
                    "terrain_type_distribution": actual_sample_audit[
                        "terrain_type_distribution"
                    ],
                    "episode_count": actual_sample_audit["episodes"],
                    "physics_distribution": actual_sample_audit["physics_distribution"],
                    "observation_noise_scale": actual_sample_audit[
                        "observation_noise_scale"
                    ],
                    "actual_sample_audit": actual_sample_audit,
                }
            )
            # Keep every shard within one reset-separated curriculum segment.
            flush()
        flush()

        global_slope_samples = {
            label: sum(
                segment["slope_sample_distribution"][label]
                for segment in stage_segments
            )
            for label in SLOPE_LABELS
        }
        global_slope_environments = {
            label: sum(
                segment["slope_environment_distribution"][label]
                for segment in stage_segments
            )
            for label in SLOPE_LABELS
        }
        global_slope_episodes = {
            label: sum(
                segment["slope_episode_distribution"][label]
                for segment in stage_segments
            )
            for label in SLOPE_LABELS
        }
        manifest = {
            "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
            "task": args.task,
            "on_policy": True,
            "teacher_checkpoint": os.fspath(teacher_path),
            "teacher_checkpoint_sha256": sha256_file(teacher_path),
            "teacher_provenance": metadata.to_mapping(),
            "teacher_training_configuration": teacher_training_configuration,
            "training_iteration": args.training_iteration,
            "num_envs": args.num_envs,
            "collection_seed": args.seed,
            "num_steps_per_stage": args.num_steps,
            "num_policy_steps": collected_steps,
            "num_samples": collected_samples,
            "signed_slopes": list(SIGNED_SLOPES),
            "slope_sample_distribution": global_slope_samples,
            "slope_environment_distribution": global_slope_environments,
            "slope_episode_distribution": global_slope_episodes,
            "stage_segments": stage_segments,
            "stage_sample_distribution": stage_sample_distribution,
            "storage_dtype": args.storage_dtype,
            "shards_sha256": shard_hashes,
            "sample_audit": sample_audit_contract,
        }
        manifest["sample_audit_sha256"] = canonical_sha256(
            {
                "sample_audit": sample_audit_contract,
                "segments": [segment["actual_sample_audit"] for segment in stage_segments],
            }
        )
        validate_rollout_stage_coverage(manifest)
        validate_rollout_sample_audit(
            manifest,
            {name: torch.cat(values, dim=0) for name, values in all_audit_chunks.items()},
        )
        write_json_atomic(output_dir / "manifest.json", manifest)
        print(f"collected {collected_samples} verified on-policy samples in {output_dir}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
