#!/usr/bin/env python3
"""Train the S0 privileged teacher with Mjlab RSL-RL."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from _mjlab_wrappers import (
    add_project_source_to_path,
    require_existing_file,
    run_mjlab_rsl_rl,
)

add_project_source_to_path()

from g1_rickshaw_lab.reward_profile import (  # noqa: E402
    REWARD_WEIGHT_OVERRIDES_KEY,
    parse_reward_weight_arguments,
    reward_weight_hydra_overrides,
    reward_weight_overrides_from_configuration,
)
from g1_rickshaw_lab.rl.runner import RunnerContext  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    DEFAULT_TRAINING_PARAMETERS,
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_PARAMETERS,
    GUIDE_TRAINING_TASK,
    SUPPORTED_CONTEXT_DIMS,
    SUPPORTED_FAT2_WEIGHTS,
    SUPPORTED_HISTORY_LENGTHS,
    SUPPORTED_ROLLOUT_STEPS,
    TRAINING_CONFIGURATION_KEY,
    build_training_configuration,
    cli_value,
    guide_max_iterations,
    load_s0_resume_checkpoint,
    require_pinned_rsl_rl,
    training_artifact_interval,
)

DEFAULT_TASK = GUIDE_TRAINING_TASK
S0_GUIDE_PARAMETERS = GUIDE_TRAINING_PARAMETERS["s0_teacher"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Optional isolated RSL-RL experiment root for a fresh S0 run.",
    )
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--fat2-weight", type=float, choices=SUPPORTED_FAT2_WEIGHTS, default=None)
    parser.add_argument("--latent-dim", type=int, choices=SUPPORTED_CONTEXT_DIMS, default=None)
    parser.add_argument("--history-length", type=int, choices=SUPPORTED_HISTORY_LENGTHS, default=None)
    parser.add_argument("--rollout-steps", type=int, choices=SUPPORTED_ROLLOUT_STEPS, default=None)
    parser.add_argument("--stability-reward-curriculum", action="store_true", default=None)
    parser.add_argument(
        "--reward-weight",
        action="append",
        default=[],
        metavar="TERM=WEIGHT",
        help="Override one non-FAT2 reward weight; may be repeated.",
    )
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=GUIDE_TRAINING_NUM_ENVS,
    )
    args, remaining = parser.parse_known_args()
    requested_reward_overrides = parse_reward_weight_arguments(args.reward_weight)
    owned_resume_flags = ("--resume", "--load_run", "--checkpoint", "--agent")
    if args.resume_checkpoint is not None:
        owned_resume_flags += ("--experiment_name",)
    if any(token == flag or token.startswith(flag + "=") for token in remaining for flag in owned_resume_flags):
        raise ValueError("S0 owns its agent and resume selection; use --resume-checkpoint")
    require_pinned_rsl_rl()
    resume_path: Path | None = None
    resume_configuration = None
    if args.resume_checkpoint is not None:
        resume_path = require_existing_file(
            args.resume_checkpoint,
            "S0 resume checkpoint",
        ).resolve()
        loaded = load_s0_resume_checkpoint(
            resume_path,
            validate_runtime=True,
        )
        resume_configuration = loaded[TRAINING_CONFIGURATION_KEY]
    resume_parameters = None if resume_configuration is None else resume_configuration["training_parameters"]
    if resume_configuration is None:
        reward_weight_overrides = requested_reward_overrides
    else:
        resumed_reward_overrides = reward_weight_overrides_from_configuration(resume_configuration)
        if requested_reward_overrides and requested_reward_overrides != resumed_reward_overrides:
            raise ValueError("S0 resume cannot change reward weights")
        reward_weight_overrides = resumed_reward_overrides
    defaults = DEFAULT_TRAINING_PARAMETERS if resume_parameters is None else resume_parameters
    fat2_weight = float(defaults["fat2_weight"] if args.fat2_weight is None else args.fat2_weight)
    latent_dim = int(defaults["latent_dim"] if args.latent_dim is None else args.latent_dim)
    history_length = int(defaults["history_length"] if args.history_length is None else args.history_length)
    rollout_steps = int(defaults["rollout_steps"] if args.rollout_steps is None else args.rollout_steps)
    stability_reward_curriculum = bool(
        defaults["stability_reward_curriculum"]
        if args.stability_reward_curriculum is None
        else args.stability_reward_curriculum
    )
    if resume_parameters is not None and (
        fat2_weight != float(resume_parameters["fat2_weight"])
        or latent_dim != int(resume_parameters["latent_dim"])
        or history_length != int(resume_parameters["history_length"])
        or rollout_steps != int(resume_parameters["rollout_steps"])
        or stability_reward_curriculum != bool(resume_parameters["stability_reward_curriculum"])
    ):
        raise ValueError("S0 resume cannot change FAT2, latent_dim, history_length, rollout_steps, or stability reward curriculum")
    seed = cli_value(
        remaining,
        "--seed",
        hydra_keys=("agent.seed",),
        default=42,
        cast=int,
    )
    max_iterations = cli_value(
        remaining,
        "--max_iterations",
        hydra_keys=("agent.max_iterations",),
        default=guide_max_iterations("s0_teacher", rollout_steps),
        cast=int,
    )
    training_configuration = build_training_configuration(
        stage="s0_teacher",
        task=args.task,
        num_envs=args.num_envs,
        seed=seed,
        max_iterations=max_iterations,
        guide_parameters=S0_GUIDE_PARAMETERS,
        resolved_parameters={
            "seed": seed,
            "max_iterations": max_iterations,
            "num_envs": args.num_envs,
            "fat2_weight": fat2_weight,
            "latent_dim": latent_dim,
            "num_steps_per_env": rollout_steps,
            "save_interval": training_artifact_interval(rollout_steps),
            "stability_reward_curriculum": stability_reward_curriculum,
            REWARD_WEIGHT_OVERRIDES_KEY: reward_weight_overrides,
            "launcher_arguments": list(remaining),
        },
        actor_initialized_from_teacher=None,
        stage_coverage=None,
        fat2_weight=fat2_weight,
        latent_dim=latent_dim,
        history_length=history_length,
        rollout_steps=rollout_steps,
        stability_reward_curriculum=stability_reward_curriculum,
    )
    runner_context = RunnerContext.training(
        stage="s0_teacher",
        training_configuration=training_configuration,
    )
    runtime_overrides = [
        f"agent.num_steps_per_env={rollout_steps}",
        f"agent.save_interval={training_artifact_interval(rollout_steps)}",
        f"agent.actor.latent_dim={latent_dim}",
        f"agent.actor.history_length={history_length}",
        f"env.history_length={history_length}",
        *reward_weight_hydra_overrides(reward_weight_overrides),
        f"env.rewards.fat2_prior_exp.weight={fat2_weight}",
    ]
    experiment_arguments: list[str] = []
    if resume_path is not None:
        experiment_root = resume_path.parent.parent
        if args.experiment_dir is not None and Path(args.experiment_dir).resolve() != experiment_root.resolve():
            raise ValueError("--experiment-dir must match the resume checkpoint experiment root")
        experiment_arguments = [
            "--experiment_name",
            os.fspath(experiment_root),
            "--resume",
            "--load_run",
            "^" + re.escape(resume_path.parent.name) + "$",
            "--checkpoint",
            "^" + re.escape(resume_path.name) + "$",
        ]
    elif args.experiment_dir is not None:
        experiment_arguments = [
            "--experiment_name",
            os.fspath(Path(args.experiment_dir).resolve()),
        ]
    run_mjlab_rsl_rl(
        "train",
        [
            "--task",
            args.task,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(max_iterations),
            "--seed",
            str(seed),
            "--logger",
            "tensorboard",
            *experiment_arguments,
            *remaining,
            *runtime_overrides,
        ],
        runner_context=runner_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
