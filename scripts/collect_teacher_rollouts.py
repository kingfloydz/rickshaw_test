#!/usr/bin/env python3
"""Collect verified S0 teacher on-policy rollouts for S1 distillation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from _mjlab_wrappers import (
    add_mjlab_sources_to_path,
    add_project_source_to_path,
    load_mjlab_configs,
    require_existing_file,
)

add_project_source_to_path()

from g1_rickshaw_lab.provenance import atomic_torch_save, extract_checkpoint_metadata  # noqa: E402
from g1_rickshaw_lab.rl.runner import RunnerContext, create_rickshaw_runner_type  # noqa: E402
from g1_rickshaw_lab.reward_profile import (  # noqa: E402
    apply_reward_weight_overrides,
    reward_weight_overrides_from_configuration,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    DISTILLATION_ROLLOUT_STEPS,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
    normalize_rsl_rl_runner_configuration,
    require_pinned_rsl_rl,
    validate_rollout_stage_coverage,
)
from g1_rickshaw_lab.artifact_io import write_json_atomic  # noqa: E402
from _rollout_audit import (  # noqa: E402
    AUDIT_TENSOR_NAMES,
    DEFAULT_NUM_ENVS,
    INTEGER_AUDIT_TENSORS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
    SIGNED_SLOPES,
    SLOPE_LABELS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    slope_environment_assignment,
    summarize_segment_samples,
    validate_rollout_sample_audit,
)


DEFAULT_TASK = "Mjlab-G1-Rickshaw-Directional-Slope-Teacher"


def _step_teacher_policy(
    actor: Any,
    observation: Any,
    env: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Run one teacher/environment transition without creating inference tensors."""

    with torch.no_grad():
        z_star = actor.encode(observation).clone()
        distribution = actor.policy.distribution(observation["policy"], z_star)
        actions = distribution.sample()
        teacher_mean = distribution.mean.clone()
        teacher_std = distribution.base_dist.scale.clone()
        next_observation, reward, dones, extras = env.step(actions)
    return z_star, teacher_mean, teacher_std, next_observation, reward, dones, extras


def main() -> int:  # noqa: C901
    add_mjlab_sources_to_path()
    require_pinned_rsl_rl()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--shard-steps",
        type=int,
        default=4,
        help="Policy steps buffered per shard; four bounds host memory at 8192 environments.",
    )
    parser.add_argument("--max-collection-multiplier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--storage-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--training-iteration",
        type=int,
        default=None,
        help="Override the S0 checkpoint domain-randomization iteration.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.num_envs = DEFAULT_NUM_ENVS
    args.num_steps = DISTILLATION_ROLLOUT_STEPS
    if args.shard_steps <= 0 or args.max_collection_multiplier <= 0:
        raise ValueError(
            "--shard-steps and --max-collection-multiplier must be positive"
        )
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise ValueError("rollout collection seed must lie in [0, 2**32-1]")
    if args.training_iteration is not None and args.training_iteration < 0:
        raise ValueError("--training-iteration must be non-negative")

    teacher_path = require_existing_file(args.teacher, "teacher checkpoint").resolve()
    teacher_checkpoint = load_stage_checkpoint(
        teacher_path,
        expected_stage="s0_teacher",
        validate_runtime=True,
    )
    if args.training_iteration is None:
        checkpoint_iteration = teacher_checkpoint.get(
            CHECKPOINT_CURRICULUM_ITERATION_KEY
        )
        if (
            isinstance(checkpoint_iteration, bool)
            or not isinstance(checkpoint_iteration, int)
            or checkpoint_iteration < 0
        ):
            raise RuntimeError("S0 checkpoint is missing a valid curriculum iteration")
        args.training_iteration = checkpoint_iteration
    teacher_training_configuration = dict(
        teacher_checkpoint[TRAINING_CONFIGURATION_KEY]
    )
    training_parameters = teacher_training_configuration["training_parameters"]
    latent_dim = int(training_parameters["latent_dim"])
    history_length = int(training_parameters["history_length"])
    if teacher_training_configuration["task"] != args.task:
        raise ValueError("rollout task differs from the S0 teacher training task")
    metadata = extract_checkpoint_metadata(teacher_checkpoint)
    output_dir = Path(args.output_dir).resolve()
    existing = list(output_dir.glob("rollout_*.pt")) + list(
        output_dir.glob("manifest.json")
    )
    if existing and not args.overwrite:
        raise FileExistsError(
            f"rollout output already contains artifacts: {output_dir}"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401
        from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mjlab_events import (
            assign_mjlab_slope_slots,
        )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        env_cfg, agent_cfg = load_mjlab_configs(
            args.task,
            play=False,
            num_envs=args.num_envs,
            seed=args.seed,
            history_length=history_length,
        )
        # Rollout samples bind immutable slopes. This prevents auto-resets
        # from advancing terrain difficulty during a short segment.
        env_cfg.curriculum = {}
        apply_reward_weight_overrides(
            env_cfg,
            reward_weight_overrides_from_configuration(teacher_training_configuration),
        )
        env_cfg.rewards["fat2_prior_exp"].weight = float(
            training_parameters["fat2_weight"]
        )
        agent_cfg.actor.latent_dim = latent_dim
        agent_cfg = normalize_rsl_rl_runner_configuration(agent_cfg)
        raw_env = ManagerBasedRlEnv(env_cfg, device=device)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner_type = create_rickshaw_runner_type(
            RunnerContext.playback(
                stage="s0_teacher",
                curriculum_start_iteration=args.training_iteration,
                metadata=metadata,
            ),
            base_runner_type=MjlabOnPolicyRunner,
        )
        runner = runner_type(
            env, asdict(agent_cfg), log_dir=None, device=device
        )
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
        fixed_assignment = slope_environment_assignment(
            args.num_envs, device=device
        )
        storage_dtype = (
            torch.float16 if args.storage_dtype == "float16" else torch.float32
        )
        pending: dict[str, list[torch.Tensor]] = {}
        stage_segments: list[dict] = []
        stage_sample_distribution: dict[str, int] = {}
        all_audit_chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in AUDIT_TENSOR_NAMES
        }
        shard_index = 0
        shard_names: list[str] = []
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
            tensors = {
                name: torch.cat(values, dim=0) for name, values in pending.items()
            }
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
                    "g1_rickshaw_provenance": metadata.to_mapping(),
                },
                path,
            )
            shard_names.append(name)
            shard_index += 1
            collected_samples += samples
            pending.clear()

        def install_fixed_rollout_assignment() -> None:
            """Install the deterministic configured-slope allocation before reset."""

            assign_mjlab_slope_slots(base_env, fixed_assignment["slope_index"])

        def assert_fixed_rollout_assignment() -> None:
            terrain = base_env.scene.terrain
            if not torch.equal(
                terrain.terrain_levels, fixed_assignment["terrain_level"]
            ):
                raise RuntimeError("rollout terrain levels changed during collection")
            if not torch.equal(terrain.terrain_types, fixed_assignment["terrain_type"]):
                raise RuntimeError("rollout terrain types changed during collection")
            if not torch.allclose(
                base_env.slope,
                fixed_assignment["slope"],
                atol=1.0e-7,
                rtol=0.0,
            ):
                raise RuntimeError("rollout slopes changed during collection")

        def reset_at_static_fixed_point() -> object:
            observation, _ = env.reset()
            return observation.to(device)

        sample_audit_contract = {
            "schema_version": ROLLOUT_SAMPLE_AUDIT_SCHEMA_VERSION,
            "signed_slopes": list(SIGNED_SLOPES),
            "terrain_levels": list(SLOPE_TERRAIN_LEVELS),
            "terrain_types": list(SLOPE_TERRAIN_TYPES),
            "collection_seed": args.seed,
            "episode_binding": "environment_id + monotonically assigned reset episode_id",
            "source": "per-transition shard tensors",
        }

        stage_specs = [("TRAINING", args.training_iteration)]

        target_samples = args.num_envs * args.num_steps
        for segment_index, (expected_stage, curriculum_iteration) in enumerate(
            stage_specs
        ):
            actual_global_stage = expected_stage

            install_fixed_rollout_assignment()
            observation = reset_at_static_fixed_point()
            assert_fixed_rollout_assignment()
            stage_valid_samples = 0
            stage_collection_steps = 0
            per_environment_samples = torch.zeros(
                args.num_envs, device=device, dtype=torch.long
            )
            sample_distribution: dict[str, int] = {}
            stage_audit_chunks: dict[str, list[torch.Tensor]] = {
                name: [] for name in AUDIT_TENSOR_NAMES
            }
            episode_ids = torch.arange(
                next_episode_id,
                next_episode_id + args.num_envs,
                device=device,
                dtype=torch.long,
            )
            next_episode_id += args.num_envs
            per_env_stage = {"TRAINING": args.num_envs}

            while stage_valid_samples < target_samples:
                if (
                    stage_collection_steps
                    >= args.num_steps * args.max_collection_multiplier
                ):
                    raise RuntimeError(
                        f"stage {expected_stage} failed to collect {target_samples} valid samples"
                    )
                assert_fixed_rollout_assignment()
                valid = torch.ones(
                    args.num_envs, dtype=torch.bool, device=device
                )
                valid &= per_environment_samples < args.num_steps
                valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                selected = valid_ids
                audit_values = None
                if selected.numel() > 0:
                    audit_values = {
                        "curriculum_stage": torch.ones(
                            (selected.numel(), 1),
                            device=device,
                            dtype=torch.long,
                        ),
                        "collection_segment": torch.full(
                            (selected.numel(), 1),
                            segment_index,
                            device=device,
                            dtype=torch.long,
                        ),
                        "environment_id": selected.unsqueeze(-1).clone(),
                        "episode_id": episode_ids[selected].unsqueeze(-1).clone(),
                        "slope": base_env.slope[selected].unsqueeze(-1).clone(),
                        "terrain_level": base_env.scene.terrain.terrain_levels[selected]
                        .unsqueeze(-1)
                        .clone(),
                        "terrain_type": base_env.scene.terrain.terrain_types[selected]
                        .unsqueeze(-1)
                        .clone(),
                    }

                (
                    z_star,
                    teacher_mean,
                    teacher_std,
                    next_observation,
                    _,
                    dones,
                    _,
                ) = _step_teacher_policy(actor, observation, env)

                if selected.numel() > 0:
                    assert audit_values is not None
                    append("current", observation["policy"][selected])
                    append("history", observation["history"][selected])
                    append("teacher_action_mean", teacher_mean[selected])
                    append("teacher_action_std", teacher_std[selected])
                    append("z_star", z_star[selected])
                    for name, value in audit_values.items():
                        append(name, value)
                        cpu_value = value.detach().to(
                            device="cpu",
                            dtype=torch.long
                            if name in INTEGER_AUDIT_TENSORS
                            else torch.float32,
                        )
                        stage_audit_chunks[name].append(cpu_value)
                        all_audit_chunks[name].append(cpu_value)
                    amount = int(selected.numel())
                    sample_distribution["TRAINING"] = (
                        sample_distribution.get("TRAINING", 0) + amount
                    )
                    stage_sample_distribution["TRAINING"] = (
                        stage_sample_distribution.get("TRAINING", 0) + amount
                    )
                    per_environment_samples[selected] += 1
                    stage_valid_samples += int(selected.numel())
                observation = next_observation.to(device)
                done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
                if done_ids.numel() > 0:
                    episode_ids[done_ids] = torch.arange(
                        next_episode_id,
                        next_episode_id + done_ids.numel(),
                        device=device,
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
                name: torch.cat(values, dim=0)
                for name, values in stage_audit_chunks.items()
            }
            actual_sample_audit = summarize_segment_samples(
                segment_audit_tensors,
                segment_index=segment_index,
                num_envs=args.num_envs,
                samples_per_environment=args.num_steps,
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
                    "actual_sample_audit": actual_sample_audit,
                }
            )
            # Keep every shard within one reset-separated training segment.
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
            "teacher_provenance": metadata.to_mapping(),
            "teacher_training_configuration": teacher_training_configuration,
            "training_iteration": args.training_iteration,
            "num_envs": args.num_envs,
            "collection_seed": args.seed,
            "num_steps_per_stage": args.num_steps,
            "teacher_latent_dim": latent_dim,
            "num_policy_steps": collected_steps,
            "num_samples": collected_samples,
            "signed_slopes": list(SIGNED_SLOPES),
            "slope_sample_distribution": global_slope_samples,
            "slope_environment_distribution": global_slope_environments,
            "slope_episode_distribution": global_slope_episodes,
            "stage_segments": stage_segments,
            "stage_sample_distribution": stage_sample_distribution,
            "storage_dtype": args.storage_dtype,
            "shards": shard_names,
            "sample_audit": sample_audit_contract,
        }
        validate_rollout_stage_coverage(manifest)
        validate_rollout_sample_audit(
            manifest,
            {
                name: torch.cat(values, dim=0)
                for name, values in all_audit_chunks.items()
            },
        )
        write_json_atomic(output_dir / "manifest.json", manifest)
        print(
            f"collected {collected_samples} verified on-policy samples in {output_dir}"
        )
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
