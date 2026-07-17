#!/usr/bin/env python3
"""Fine-tune the S2 student policy from teacher/context checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from _isaaclab_wrappers import add_project_source_to_path, require_existing_file, run_isaaclab_rsl_rl

add_project_source_to_path()

from g1_rickshaw_lab.provenance import atomic_torch_save  # noqa: E402
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    GUIDE_TRAINING_NUM_ENVS,
    GUIDE_TRAINING_PARAMETERS,
    GUIDE_TRAINING_TASK,
    MAINLINE_PARAMETERS,
    build_s2_bootstrap_checkpoint,
    guide_max_iterations,
    load_s2_resume_checkpoint,
    load_stage_checkpoint,
    validate_guide_training_configuration,
)

from _training_configuration import (  # noqa: E402
    TRAINING_CONFIGURATION_CHECKPOINT_KEY,
    build_training_configuration,
    cli_value,
    publish_training_configuration,
)


DEFAULT_TASK = GUIDE_TRAINING_TASK
STUDENT_AGENT_KEY = "rsl_rl_student_cfg_entry_point"
S2_GUIDE_PARAMETERS = GUIDE_TRAINING_PARAMETERS["s2_student_ppo"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--bootstrap-dir", default="logs/rsl_rl/g1_rickshaw_student")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=GUIDE_TRAINING_NUM_ENVS,
    )
    args, remaining = parser.parse_known_args()
    teacher = require_existing_file(args.teacher, "teacher checkpoint")
    context = require_existing_file(args.context, "context checkpoint")
    owned_resume_flags = (
        "--agent",
        "--experiment_name",
        "--resume",
        "--load_run",
        "--checkpoint",
    )
    if any(
        token == flag or token.startswith(flag + "=")
        for token in remaining
        for flag in owned_resume_flags
    ):
        raise ValueError("S2 owns its agent, experiment root, and resume selection")
    context_checkpoint = load_stage_checkpoint(
        context,
        expected_stage="s1_context_distillation",
        validate_runtime=True,
    )
    s1_training_configuration = dict(
        context_checkpoint[TRAINING_CONFIGURATION_CHECKPOINT_KEY]
    )
    resume_checkpoint_path: Path | None = None
    if args.resume_checkpoint is None:
        checkpoint = build_s2_bootstrap_checkpoint(teacher, context)
    else:
        resume_checkpoint_path = require_existing_file(
            args.resume_checkpoint,
            "S2 resume checkpoint",
        ).resolve()
        checkpoint = load_s2_resume_checkpoint(
            resume_checkpoint_path,
            validate_runtime=True,
        )
    curriculum_iteration = checkpoint[CHECKPOINT_CURRICULUM_ITERATION_KEY]
    lineage = checkpoint["g1_rickshaw_lineage"]
    if resume_checkpoint_path is None:
        load_run = "bootstrap_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", context.stem)
        experiment_root = Path(args.bootstrap_dir).resolve()
        load_checkpoint_name = "model_0.pt"
        atomic_torch_save(
            checkpoint,
            experiment_root / load_run / load_checkpoint_name,
        )
    else:
        experiment_root = resume_checkpoint_path.parent.parent
        load_run = "^" + re.escape(resume_checkpoint_path.parent.name) + "$"
        load_checkpoint_name = "^" + re.escape(resume_checkpoint_path.name) + "$"

    os.environ["G1_RICKSHAW_RUNNER_HOOK"] = "1"
    os.environ["G1_RICKSHAW_TASK"] = args.task
    os.environ["G1_RICKSHAW_CHECKPOINT_STAGE"] = "s2_student_ppo"
    os.environ["G1_RICKSHAW_CURRICULUM_START_ITERATION"] = str(
        curriculum_iteration
    )
    os.environ["G1_RICKSHAW_CHECKPOINT_LINEAGE"] = json.dumps(lineage, sort_keys=True)
    seed = cli_value(
        remaining,
        "--seed",
        hydra_keys=("agent.seed",),
        default=42,
        cast=int,
    )
    rollout_steps = int(MAINLINE_PARAMETERS["rollout_steps"])
    max_iterations = cli_value(
        remaining,
        "--max_iterations",
        hydra_keys=("agent.max_iterations",),
        default=guide_max_iterations("s2_student_ppo"),
        cast=int,
    )
    fat2_weight = float(MAINLINE_PARAMETERS["fat2_weight"])
    training_configuration = build_training_configuration(
        stage="s2_student_ppo",
        task=args.task,
        num_envs=args.num_envs,
        seed=seed,
        max_iterations=max_iterations,
        guide_parameters=S2_GUIDE_PARAMETERS,
        resolved_parameters={
            "seed": seed,
            "max_iterations": max_iterations,
            "num_envs": args.num_envs,
            "num_steps_per_env": rollout_steps,
            "launcher_arguments": list(remaining),
            "teacher_checkpoint": os.fspath(teacher.resolve()),
            "context_checkpoint": os.fspath(context.resolve()),
        },
        actor_initialized_from_teacher=bool(
            s1_training_configuration.get("actor_initialized_from_teacher")
        ),
        stage_coverage=s1_training_configuration.get("stage_coverage"),
    )
    validate_guide_training_configuration(
        training_configuration,
        expected_stage="s2_student_ppo",
    )
    publish_training_configuration(training_configuration)
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
            "--agent",
            STUDENT_AGENT_KEY,
            "--experiment_name",
            os.fspath(experiment_root),
            "--resume",
            "--load_run",
            load_run,
            "--checkpoint",
            load_checkpoint_name,
            *remaining,
            f"agent.num_steps_per_env={rollout_steps}",
            f"env.rewards.fat2_prior_exp.weight={fat2_weight}",
            "env.observations.teacher_dynamic_history=null",
            "env.observations.teacher_static=null",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
