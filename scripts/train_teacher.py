#!/usr/bin/env python3
"""Train the S0 privileged teacher with Isaac Lab RSL-RL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

from _isaaclab_wrappers import add_project_source_to_path, require_existing_file, run_isaaclab_rsl_rl

add_project_source_to_path()

from g1_rickshaw_lab.provenance import sha256_file  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    ABLATION_VALUE_OPTIONS,
    DEFAULT_RESET_POSES_PATH,
    GUIDE_MAX_ITERATIONS,
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_PARAMETERS,
    GUIDE_TRAINING_TASK,
    TRAINING_CONFIGURATION_KEY,
    feasibility_config_path,
    load_s0_resume_checkpoint,
    require_pinned_rsl_rl,
    validate_guide_training_configuration,
)
from _training_configuration import (  # noqa: E402
    build_training_configuration,
    cli_value,
    publish_training_configuration,
    validate_formal_launcher_arguments,
    validate_training_configuration as validate_launcher_training_configuration,
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
    parser.add_argument(
        "--fat2-weight",
        type=float,
        choices=ABLATION_VALUE_OPTIONS["fat2_weight"],
        default=0.1,
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        choices=ABLATION_VALUE_OPTIONS["rollout_steps"],
        default=48,
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        choices=ABLATION_VALUE_OPTIONS["latent_dim"],
        default=16,
        help="Downstream student ablation binding; the S0 teacher architecture remains 16-D.",
    )
    args, remaining = parser.parse_known_args()
    validate_formal_launcher_arguments(remaining)
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
    reset_pose_path = Path(
        os.environ.get("G1_RICKSHAW_RESET_POSES", os.fspath(DEFAULT_RESET_POSES_PATH))
    ).resolve()
    require_pinned_rsl_rl()
    os.environ["G1_RICKSHAW_RUNNER_HOOK"] = "1"
    os.environ["G1_RICKSHAW_TASK"] = args.task
    os.environ["G1_RICKSHAW_CHECKPOINT_STAGE"] = "s0_teacher"
    current_inputs = {
        "feasibility_envelope": sha256_file(feasibility_path),
        "reset_poses": sha256_file(reset_pose_path),
    }
    resume_path: Path | None = None
    resume_training_configuration: dict | None = None
    if args.resume_checkpoint is not None:
        resume_path = require_existing_file(
            args.resume_checkpoint,
            "S0 resume checkpoint",
        ).resolve()
        resume_checkpoint = load_s0_resume_checkpoint(
            resume_path,
            validate_runtime=True,
        )
        resume_training_configuration = dict(
            resume_checkpoint[TRAINING_CONFIGURATION_KEY]
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
        default=GUIDE_MAX_ITERATIONS["s0_teacher"],
        cast=int,
    )
    hydra_overrides = [
        token for token in remaining if "=" in token and not token.startswith("--")
    ]
    expected_ablations = {
        "fat2_weight": args.fat2_weight,
        "rollout_steps": args.rollout_steps,
        "latent_dim": args.latent_dim,
    }
    if resume_training_configuration is None:
        formal_run = (
            max_iterations == GUIDE_MAX_ITERATIONS["s0_teacher"]
            and seed == 42
            and args.task == GUIDE_TRAINING_TASK
            and args.num_envs == GUIDE_TRAINING_NUM_ENVS
        )
        training_configuration = build_training_configuration(
            stage="s0_teacher",
            formal=formal_run,
            task=args.task,
            num_envs=args.num_envs,
            seed=seed,
            max_iterations=max_iterations,
            argv=sys.argv[1:],
            hydra_overrides=hydra_overrides,
            guide_parameters=S0_GUIDE_PARAMETERS,
            resolved_parameters={
                "seed": seed,
                "max_iterations": max_iterations,
                "num_envs": args.num_envs,
                "launcher_arguments": list(remaining),
            },
            actor_initialized_from_teacher=None,
            stage_coverage=None,
            latent_dim=args.latent_dim,
            rollout_steps=args.rollout_steps,
            fat2_weight=args.fat2_weight,
            inputs_sha256=current_inputs,
        )
    else:
        deviations: list[str] = []
        for name, actual, expected in (
            ("task", args.task, resume_training_configuration["task"]),
            ("num_envs", args.num_envs, resume_training_configuration["num_envs"]),
            ("seed", seed, resume_training_configuration["seed"]),
            ("max_iterations", max_iterations, resume_training_configuration["max_iterations"]),
            ("ablation_values", expected_ablations, resume_training_configuration["ablation_values"]),
            ("inputs_sha256", current_inputs, resume_training_configuration["inputs_sha256"]),
        ):
            if actual != expected:
                deviations.append(f"{name}: requested={actual!r}, checkpoint={expected!r}")
        if hydra_overrides:
            deviations.append(f"unverified Hydra overrides: {hydra_overrides!r}")
        if deviations:
            raise ValueError(
                "S0 resume must preserve the checkpoint's formal run configuration: "
                + "; ".join(deviations)
            )
        training_configuration = resume_training_configuration
    if resume_training_configuration is None:
        if training_configuration["formal"]:
            validate_guide_training_configuration(
                training_configuration,
                expected_stage="s0_teacher",
            )
        else:
            validate_launcher_training_configuration(training_configuration)
    publish_training_configuration(training_configuration)
    runtime_overrides = [
        f"agent.num_steps_per_env={args.rollout_steps}",
        f"env.rewards.fat2_prior_exp.weight={args.fat2_weight}",
        "env.observations.history=null",
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
            *runtime_overrides,
            *remaining,
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
