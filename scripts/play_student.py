#!/usr/bin/env python3
"""Play/export a trained S2 student checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _isaaclab_wrappers import add_project_source_to_path, require_existing_file, run_isaaclab_rsl_rl

add_project_source_to_path()

from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
)


DEFAULT_TASK = "Isaac-G1-Rickshaw-Directional-Slope-Play-v0"
_OPERATIONAL_FLAGS = {
    "--headless",
    "--enable_cameras",
    "--verbose",
    "--info",
    "--video",
    "--real-time",
}
_OPERATIONAL_OPTIONS = {
    "--livestream",
    "--device",
    "--rendering_mode",
    "--video_length",
    "--num_envs",
    "--seed",
}


def validate_operational_play_arguments(arguments: list[str]) -> None:
    """Allow presentation/runtime sizing flags, but no policy or environment overrides."""

    index = 0
    while index < len(arguments):
        token = str(arguments[index])
        if token in _OPERATIONAL_FLAGS:
            index += 1
            continue
        matched = next(
            (
                option
                for option in _OPERATIONAL_OPTIONS
                if token == option or token.startswith(option + "=")
            ),
            None,
        )
        if matched is None:
            raise ValueError(
                f"play/export rejects policy or environment override {token!r}"
            )
        if token == matched:
            if index + 1 >= len(arguments):
                raise ValueError(f"play/export option {matched} requires a value")
            index += 2
        else:
            index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Write RecordVideo output to this directory instead of the checkpoint log tree.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Validate and export JIT/ONNX plus manifest, then close Kit without entering the play loop.",
    )
    args, remaining = parser.parse_known_args()
    if args.task != DEFAULT_TASK:
        raise ValueError(f"play/export task is fixed to {DEFAULT_TASK}")
    validate_operational_play_arguments(list(remaining))
    checkpoint = require_existing_file(args.checkpoint, "student checkpoint").resolve()
    if args.video_dir:
        os.environ["G1_RICKSHAW_VIDEO_DIR"] = os.fspath(Path(args.video_dir).resolve())
    loaded_checkpoint = load_stage_checkpoint(
        checkpoint,
        expected_stage="s2_student_ppo",
        validate_runtime=True,
    )
    training_parameters = loaded_checkpoint[TRAINING_CONFIGURATION_KEY][
        "training_parameters"
    ]
    fat2_weight = float(training_parameters["fat2_weight"])
    latent_dim = int(training_parameters["latent_dim"])
    os.environ["G1_RICKSHAW_RUNNER_HOOK"] = "1"
    os.environ["G1_RICKSHAW_CHECKPOINT_STAGE"] = "s2_student_ppo"
    curriculum_iteration = loaded_checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
    if isinstance(curriculum_iteration, bool) or not isinstance(curriculum_iteration, int):
        raise RuntimeError("S2 checkpoint is missing its audited curriculum iteration")
    os.environ["G1_RICKSHAW_CURRICULUM_START_ITERATION"] = str(curriculum_iteration)
    os.environ["G1_RICKSHAW_CHECKPOINT_LINEAGE"] = "{}"
    if args.export_only:
        os.environ["G1_RICKSHAW_EXPORT_ONLY"] = "1"
    run_isaaclab_rsl_rl(
        "play.py",
        [
            "--task",
            args.task,
            "--checkpoint",
            str(checkpoint),
            f"agent.actor.latent_dim={latent_dim}",
            f"env.rewards.fat2_prior_exp.weight={fat2_weight}",
            "env.observations.teacher_dynamic_history=null",
            "env.observations.teacher_static=null",
            *remaining,
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
