#!/usr/bin/env python3
"""Render one S0 teacher checkpoint into one video per configured slope."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from _mjlab_wrappers import (
    add_project_source_to_path,
    require_existing_file,
    run_mjlab_rsl_rl,
)

add_project_source_to_path()

from g1_rickshaw_lab.rl.runner import RunnerContext  # noqa: E402
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    SLOPE_COUNT,
    SLOPE_PERCENTAGES,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
)
from g1_rickshaw_lab.workflows.rsl_rl import PlayOptions  # noqa: E402

DEFAULT_TASK = "Mjlab-G1-Rickshaw-Directional-Slope-Teacher"
DEFAULT_FRAMES_PER_SLOPE = 1000
RENDER_SPEED_MPS = 1.0
FIXED_ENVIRONMENT_OVERRIDES = (
    "env.domain_randomization.enabled=false",
)


def slope_index_for_frame(frame_index: int, frames_per_slope: int) -> int:
    """Return the configured slope slot for a zero-based output frame."""

    if frame_index < 0:
        raise ValueError("frame index cannot be negative")
    if frames_per_slope <= 0:
        raise ValueError("frames per slope must be positive")
    return min(frame_index // frames_per_slope, SLOPE_COUNT - 1)


def slope_video_path(output: Path, slope_index: int) -> Path:
    slope_percent = SLOPE_PERCENTAGES[slope_index]
    return output.with_name(
        f"{output.stem}_slope_{slope_index + 1:02d}_{slope_percent:+d}pct{output.suffix}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames-per-slope", type=int, default=DEFAULT_FRAMES_PER_SLOPE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--render-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.frames_per_slope <= 0:
        raise ValueError("--frames-per-slope must be positive")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")


def _run_play_child(
    launcher_arguments: list[str],
    *,
    runner_context: RunnerContext,
    play_options: PlayOptions,
) -> None:
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *launcher_arguments]
    try:
        run_mjlab_rsl_rl(
            "play",
            launcher_arguments,
            runner_context=runner_context,
            play_options=play_options,
        )
    finally:
        sys.argv = previous_argv


def _label_slope_video(
    source: Path,
    output: Path,
    *,
    slope_index: int,
    frames_per_slope: int,
) -> dict[str, object]:
    import cv2

    capture = cv2.VideoCapture(os.fspath(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode raw video: {source}")
    raw_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if raw_frames < frames_per_slope:
        capture.release()
        raise RuntimeError(f"raw video has {raw_frames} frames; expected at least {frames_per_slope}")
    if fps <= 0.0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("raw video has invalid media metadata")

    output.parent.mkdir(parents=True, exist_ok=True)
    destination = slope_video_path(output, slope_index)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    writer = cv2.VideoWriter(
        os.fspath(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create slope video: {temporary}")

    slope_percent = SLOPE_PERCENTAGES[slope_index]
    label = f"Slope {slope_percent:+d}%   {slope_index + 1}/{SLOPE_COUNT}"
    try:
        for frame_index in range(frames_per_slope):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode raw frame {frame_index}")
            cv2.rectangle(frame, (24, 24), (340, 78), (0, 0, 0), thickness=-1)
            cv2.putText(
                frame,
                label,
                (40, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"slope video encoder produced no output: {temporary}")
    os.replace(temporary, destination)
    return {
        "index": slope_index,
        "percent": slope_percent,
        "gradient": slope_percent / 100.0,
        "video": os.fspath(destination),
        "frames": frames_per_slope,
        "fps": fps,
        "duration_seconds": frames_per_slope / fps,
        "resolution": [width, height],
    }


def _write_manifest(
    output: Path,
    *,
    checkpoint: Path,
    checkpoint_iteration: int,
    videos: list[dict[str, object]],
) -> Path:
    manifest = {
        "checkpoint": os.fspath(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "videos": videos,
    }
    destination = output.with_suffix(".json")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def main() -> int:
    args = _parser().parse_args()
    _validate_arguments(args)
    if not args.render_child:
        subprocess.run(
            [
                sys.executable,
                os.fspath(Path(__file__).resolve()),
                *sys.argv[1:],
                "--render-child",
            ],
            check=True,
        )
        return 0

    checkpoint_path = require_existing_file(args.checkpoint, "S0 checkpoint").resolve()
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stage="s0_teacher",
        validate_runtime=True,
    )
    training_parameters = checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]
    latent_dim = int(training_parameters["latent_dim"])
    history_length = int(training_parameters["history_length"])
    checkpoint_iteration = int(checkpoint[CHECKPOINT_CURRICULUM_ITERATION_KEY])

    output = args.output.resolve()
    raw_directory = output.parent / "raw" / output.stem
    raw_directory.mkdir(parents=True, exist_ok=True)
    total_frames = SLOPE_COUNT * args.frames_per_slope
    videos: list[dict[str, object]] = []

    def finalize_segment(raw_video: Path, slope_index: int) -> None:
        video = _label_slope_video(
            raw_video,
            output,
            slope_index=slope_index,
            frames_per_slope=args.frames_per_slope,
        )
        videos.append(video)
        manifest = _write_manifest(
            output,
            checkpoint=checkpoint_path,
            checkpoint_iteration=checkpoint_iteration,
            videos=videos,
        )
        if not args.keep_raw:
            raw_video.unlink()
        print(f"rendered slope {slope_index + 1}/{SLOPE_COUNT}: {video['video']}")
        print(f"manifest: {manifest}")

    runner_context = RunnerContext.playback(
        stage="s0_teacher",
        curriculum_start_iteration=checkpoint_iteration,
    )
    play_options = PlayOptions(
        video_dir=raw_directory,
        export_policy=False,
        follow_robot_camera=True,
        slope_frames=args.frames_per_slope,
        video_name_prefix=output.stem,
        video_segment_callback=finalize_segment,
    )
    launcher_arguments = [
        "--task",
        DEFAULT_TASK,
        "--checkpoint",
        os.fspath(checkpoint_path),
        "--video",
        "--video_length",
        str(total_frames),
        "--num_envs",
        str(SLOPE_COUNT),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        f"agent.actor.latent_dim={latent_dim}",
        f"agent.actor.history_length={history_length}",
        f"env.history_length={history_length}",
        f"env.policy_update.command.minimum={RENDER_SPEED_MPS}",
        f"env.policy_update.command.maximum={RENDER_SPEED_MPS}",
        f"env.policy_update.command.limit_maximum={RENDER_SPEED_MPS}",
        "env.policy_update.command.standing_fraction=0.0",
        *FIXED_ENVIRONMENT_OVERRIDES,
    ]
    if args.headless:
        launcher_arguments.append("--headless")
    try:
        _run_play_child(
            launcher_arguments,
            runner_context=runner_context,
            play_options=play_options,
        )
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    print(f"rendered {len(videos)} slope videos: {output.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
