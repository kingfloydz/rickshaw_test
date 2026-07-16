#!/usr/bin/env python3
"""Evaluate S1 candidates on fixed-seed environment task return in one Kit process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import math
import os
from pathlib import Path
from typing import Any

from _isaaclab_wrappers import add_isaaclab_sources_to_path, add_project_source_to_path

add_project_source_to_path()

from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    SIGNED_SLOPES,
    evaluation_runtime_sources_sha256,
    slope_label,
)
from g1_rickshaw_lab.provenance import sha256_file  # noqa: E402
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    TRAINING_CONFIGURATION_KEY,
    checkpoint_hash_history,
    feasibility_config_path,
    load_stage_checkpoint,
    require_pinned_rsl_rl,
)
from g1_rickshaw_lab.validation import write_json_atomic  # noqa: E402


def _configure_training(env_cfg: Any) -> None:
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    env_cfg.curriculum = None
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.runtime_randomization = replace(
        env_cfg.runtime_randomization,
        curriculum=mdp.CurriculumScheduleCfg(),
    )
    env_cfg.events.sample_physics.params = {"cfg": env_cfg.runtime_randomization}
    env_cfg.events.initialize_curriculum.params = {"cfg": env_cfg.runtime_randomization}


def _apply_checkpoint_environment(
    env_cfg: Any, training_configuration: Mapping[str, Any]
) -> None:
    ablation = training_configuration["ablation_values"]
    env_cfg.rewards.fat2_prior_exp.weight = float(ablation["fat2_weight"])


def _assign_fixed_slopes(base_env: Any) -> Any:
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
        raise RuntimeError(
            "S1 fixed validation terrain does not resolve to every configured slope"
        )
    cfg = base_env.cfg.runtime_randomization.curriculum
    previous_iteration = int(base_env.curriculum_runtime_state.iteration)
    runtime_state = mdp.CurriculumRuntimeState.create(
        columns, torch.sign(expected).to(dtype=torch.long), cfg
    )
    runtime_state.set_iteration(previous_iteration)
    base_env.curriculum_runtime_state = runtime_state
    base_env.curriculum_stage_per_env[:] = (
        base_env.curriculum_runtime_state.stage_per_environment()
    )
    expected_stage = int(mdp.CurriculumStage.TRAINING)
    if not torch.all(base_env.curriculum_stage_per_env == expected_stage):
        raise RuntimeError("S1 task-return validation must use TRAINING")
    return slots


def _evaluate_candidate(
    env: Any,
    base_env: Any,
    model: Any,
    *,
    seeds: list[int],
    episodes_per_slope: int,
    max_policy_steps_per_seed: int,
) -> dict[str, Any]:
    import torch

    completed = torch.zeros(len(SIGNED_SLOPES), dtype=torch.long, device=base_env.device)
    returns_by_slope: list[list[float]] = [[] for _ in SIGNED_SLOPES]
    episode_return = torch.zeros(base_env.num_envs, device=base_env.device)
    for seed_index, seed in enumerate(seeds):
        milestone = math.ceil((seed_index + 1) * episodes_per_slope / len(seeds))
        env.seed(seed)
        slope_slots = _assign_fixed_slopes(base_env)
        observation, _ = env.reset()
        episode_return.zero_()
        policy_steps = 0
        while torch.any(completed < milestone):
            if policy_steps >= max_policy_steps_per_seed:
                remaining = (milestone - completed).clamp_min(0).detach().cpu().tolist()
                raise RuntimeError(
                    f"seed {seed} exceeded {max_policy_steps_per_seed} policy steps; "
                    f"remaining episodes per slope={remaining}"
                )
            eligible = completed[slope_slots] < milestone
            active = eligible
            with torch.inference_mode():
                actions = model.act(
                    observation["policy"],
                    observation["history"],
                    deterministic=True,
                )
                observation, reward, dones, _ = env.step(actions)
            episode_return += reward * active.to(dtype=reward.dtype)
            done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
            for env_id in done_ids.detach().cpu().tolist():
                if not bool(active[env_id].item()):
                    episode_return[env_id] = 0.0
                    continue
                slope_index = int(slope_slots[env_id].item())
                if int(completed[slope_index].item()) >= milestone:
                    episode_return[env_id] = 0.0
                    continue
                value = float(episode_return[env_id].item())
                if not math.isfinite(value):
                    raise RuntimeError("S1 task-return evaluation produced a non-finite return")
                returns_by_slope[slope_index].append(value)
                completed[slope_index] += 1
                episode_return[env_id] = 0.0
            policy_steps += 1
    if torch.any(completed != episodes_per_slope):
        raise RuntimeError(f"S1 per-slope episode quota drifted: {completed.cpu().tolist()}")
    flattened = [value for values in returns_by_slope for value in values]
    return {
        "episodes": len(flattened),
        "mean": sum(flattened) / len(flattened),
        "per_slope": {
            slope_label(slope): {
                "episodes": len(returns_by_slope[index]),
                "mean": sum(returns_by_slope[index]) / len(returns_by_slope[index]),
            }
            for index, slope in enumerate(SIGNED_SLOPES)
        },
    }


def main() -> int:
    add_isaaclab_sources_to_path()
    feasibility_config_path()
    require_pinned_rsl_rl()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-G1-Rickshaw-Directional-Slope-v0")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS)
    parser.add_argument("--episodes-per-slope", type=int, default=100)
    parser.add_argument("--max-policy-steps-per-seed", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.num_envs % len(SIGNED_SLOPES) != 0:
        raise ValueError(f"--num-envs must be a positive multiple of {len(SIGNED_SLOPES)}")
    if args.episodes_per_slope < 100 or args.max_policy_steps_per_seed <= 0:
        raise ValueError("S1 validation requires at least 100 episodes per slope and a positive step cap")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("S1 fixed validation seeds must be non-empty and unique")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401
        from g1_rickshaw_lab.rl import G1RickshawStudentActor

        device = args.device
        first_checkpoint = load_stage_checkpoint(
            Path(args.candidates[0]).resolve(),
            expected_stage="s1_context_candidate",
            validate_runtime=True,
        )
        first_training_configuration = dict(
            first_checkpoint[TRAINING_CONFIGURATION_KEY]
        )
        if first_training_configuration["task"] != args.task:
            raise RuntimeError(
                "S1 task-return evaluation task differs from the candidate training task"
            )
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        env_cfg.seed = args.seeds[0]
        _configure_training(env_cfg)
        _apply_checkpoint_environment(env_cfg, first_training_configuration)
        raw_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)
        base_env = raw_env.unwrapped
        first_lineage = first_checkpoint.get("g1_rickshaw_lineage")
        if not isinstance(first_lineage, Mapping):
            raise RuntimeError("S1 candidate is missing its teacher/rollout lineage")
        first_checkpoint_hashes = checkpoint_hash_history(first_checkpoint)
        base_env.set_curriculum_iteration(0)
        _assign_fixed_slopes(base_env)

        results = []
        observed_iterations: set[int] = set()
        for candidate_index, candidate_name in enumerate(args.candidates):
            candidate_path = Path(candidate_name).resolve()
            if candidate_index == 0:
                checkpoint = first_checkpoint
                candidate_training_configuration = first_training_configuration
            else:
                checkpoint = load_stage_checkpoint(
                    candidate_path,
                    expected_stage="s1_context_candidate",
                    validate_runtime=True,
                )
                candidate_training_configuration = dict(
                    checkpoint[TRAINING_CONFIGURATION_KEY]
                )
            if candidate_training_configuration != first_training_configuration:
                raise RuntimeError(
                    "S1 candidates do not share one exact training configuration"
                )
            if checkpoint.get("g1_rickshaw_lineage") != first_lineage:
                raise RuntimeError("S1 candidates do not share one teacher/rollout lineage")
            if checkpoint_hash_history(checkpoint) != first_checkpoint_hashes:
                raise RuntimeError("S1 candidates do not preserve the same S0 checkpoint history")
            candidate_iteration = checkpoint.get("candidate_iteration")
            validation_action_kl = checkpoint.get("validation_action_kl")
            if (
                isinstance(candidate_iteration, bool)
                or not isinstance(candidate_iteration, int)
                or candidate_iteration in observed_iterations
                or isinstance(validation_action_kl, bool)
                or not isinstance(validation_action_kl, (int, float))
                or not math.isfinite(validation_action_kl)
                or validation_action_kl < 0.0
            ):
                raise ValueError("S1 candidate iteration/KL metadata is invalid or duplicated")
            observed_iterations.add(candidate_iteration)
            state = checkpoint["model_state_dict"]
            model = G1RickshawStudentActor(
                latent_dim=int(
                    candidate_training_configuration["ablation_values"]["latent_dim"]
                )
            ).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            task_return = _evaluate_candidate(
                env,
                base_env,
                model,
                seeds=list(args.seeds),
                episodes_per_slope=args.episodes_per_slope,
                max_policy_steps_per_seed=args.max_policy_steps_per_seed,
            )
            results.append(
                {
                    "checkpoint": os.fspath(candidate_path),
                    "checkpoint_sha256": sha256_file(candidate_path),
                    "iteration": candidate_iteration,
                    "validation_action_kl": float(validation_action_kl),
                    "task_return_mean": task_return["mean"],
                    "episodes": task_return["episodes"],
                    "per_slope": task_return["per_slope"],
                }
            )
        write_json_atomic(
            args.output,
            {
                "schema_version": 1,
                "report_type": "g1_rickshaw_s1_candidate_selection",
                "status": "recorded",
                "task": args.task,
                "training_configuration_sha256": first_training_configuration[
                    "content_sha256"
                ],
                "ablation_values": dict(
                    first_training_configuration["ablation_values"]
                ),
                "lineage": dict(first_lineage),
                "selection_metrics": ["validation_action_kl", "task_return_mean"],
                "evaluation_runtime_sources_sha256": evaluation_runtime_sources_sha256(),
                "evaluation": {
                    "curriculum_stage": "training",
                    "fixed_seeds": list(args.seeds),
                    "signed_slopes": list(SIGNED_SLOPES),
                    "episodes_per_slope": args.episodes_per_slope,
                    "num_envs": args.num_envs,
                },
                "results": results,
            },
        )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
