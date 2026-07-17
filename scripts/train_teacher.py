#!/usr/bin/env python3
"""Train the S0 privileged teacher with Isaac Lab RSL-RL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

from _isaaclab_wrappers import add_project_source_to_path, require_existing_file, run_isaaclab_rsl_rl

add_project_source_to_path()

from g1_rickshaw_lab.training_contract import (  # noqa: E402
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_PARAMETERS,
    GUIDE_TRAINING_TASK,
    MAINLINE_PARAMETERS,
    feasibility_config_path,
    guide_max_iterations,
    load_s0_resume_checkpoint,
    require_pinned_rsl_rl,
    validate_guide_training_configuration,
)
from _training_configuration import (  # noqa: E402
    build_training_configuration,
    cli_value,
    publish_training_configuration,
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
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=GUIDE_TRAINING_NUM_ENVS,
    )
    args, remaining = parser.parse_known_args()
    owned_resume_flags = ("--resume", "--load_run", "--checkpoint", "--agent")
    if args.resume_checkpoint is not None:
        owned_resume_flags += ("--experiment_name",)
    if any(
        token == flag or token.startswith(flag + "=")
        for token in remaining
        for flag in owned_resume_flags
    ):
        raise ValueError(
            "S0 owns its agent and resume selection; use --resume-checkpoint"
        )
    feasibility_path = feasibility_config_path()
    os.environ["G1_RICKSHAW_FEASIBILITY_ENVELOPE"] = os.fspath(feasibility_path)
    require_pinned_rsl_rl()
    os.environ["G1_RICKSHAW_RUNNER_HOOK"] = "1"
    os.environ["G1_RICKSHAW_TASK"] = args.task
    os.environ["G1_RICKSHAW_CHECKPOINT_STAGE"] = "s0_teacher"
    resume_path: Path | None = None
    if args.resume_checkpoint is not None:
        resume_path = require_existing_file(
            args.resume_checkpoint,
            "S0 resume checkpoint",
        ).resolve()
        load_s0_resume_checkpoint(
            resume_path,
            validate_runtime=True,
        )
    os.environ["G1_RICKSHAW_CHECKPOINT_LINEAGE"] = "{}"
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
        default=guide_max_iterations("s0_teacher"),
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
            "launcher_arguments": list(remaining),
        },
        actor_initialized_from_teacher=None,
        stage_coverage=None,
    )
    validate_guide_training_configuration(
        training_configuration,
        expected_stage="s0_teacher",
    )
    publish_training_configuration(training_configuration)
    runtime_overrides = [
        f"agent.num_steps_per_env={MAINLINE_PARAMETERS['rollout_steps']}",
        f"env.rewards.fat2_prior_exp.weight={MAINLINE_PARAMETERS['fat2_weight']}",
    ]
    experiment_arguments: list[str] = []
    if resume_path is not None:
        experiment_root = resume_path.parent.parent
        if (
            args.experiment_dir is not None
            and Path(args.experiment_dir).resolve() != experiment_root.resolve()
        ):
            raise ValueError(
                "--experiment-dir must match the resume checkpoint experiment root"
            )
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
    run_isaaclab_rsl_rl(
        "train.py",
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
