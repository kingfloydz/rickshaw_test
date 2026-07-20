#!/usr/bin/env python3
"""Render each configured reset pose for a fixed simulation horizon."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from _isaaclab_wrappers import (
    REPOSITORY_ROOT,
    add_isaaclab_sources_to_path,
    add_project_source_to_path,
)

ROOT = REPOSITORY_ROOT
add_isaaclab_sources_to_path()
add_project_source_to_path()

from isaaclab.app import AppLauncher  # noqa: E402
from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reset_video")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.fps <= 0.0 or args.fps > 50.0:
    parser.error("--fps must be in (0, 50]")
if args.width <= 0 or args.height <= 0:
    parser.error("--width and --height must be positive")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def _slope_label(slope: float) -> str:
    return f"{slope:+.2f}".replace("+", "p").replace("-", "m")


def _write_frame(writer, image, *, slope: float, index: int) -> None:
    import cv2
    import numpy as np

    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError("renderer returned an invalid RGB frame")
    frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
    label = f"reset pose  slope {slope:+.2f}  ({index + 1}/{len(SLOPE_GRADIENTS)})"
    cv2.rectangle(frame, (18, 18), (440, 62), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        label,
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    writer.write(frame)


def main() -> None:
    import cv2
    import gymnasium as gym
    import numpy as np
    import torch

    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import (
        G1RickshawDirectionalSlopePlayEnvCfg,
        PLAY_TASK_ID,
    )
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.curricula import (
        assign_terrain_slopes,
    )

    cfg = G1RickshawDirectionalSlopePlayEnvCfg()
    cfg.scene.num_envs = len(SLOPE_GRADIENTS)
    cfg.sim.device = args.device
    cfg.viewer.resolution = (args.width, args.height)
    cfg.domain_randomization = replace(
        cfg.domain_randomization,
        enabled=False,
    )
    cfg.events.initialize_domain.params = {"cfg": cfg.domain_randomization}
    cfg.curriculum = None
    cfg.scene.terrain.terrain_generator.curriculum = True

    env = gym.make(PLAY_TASK_ID, cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_hz = 1.0 / float(base.step_dt)
    sample_interval = max(1, round(policy_hz / args.fps))
    actual_fps = policy_hz / sample_interval
    action_dim = int(base.action_manager.total_action_dim)
    zero_action = torch.zeros((len(SLOPE_GRADIENTS), action_dim), device=base.device)
    output = output_dir / "reset_pose_19_slopes_1000steps.mp4"
    writer = cv2.VideoWriter(
        os.fspath(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {output}")
    manifest: list[dict[str, object]] = []

    try:
        assign_terrain_slopes(base, SLOPE_GRADIENTS)
        for index, slope in enumerate(SLOPE_GRADIENTS):
            env.reset(seed=args.seed)
            assign_terrain_slopes(base, SLOPE_GRADIENTS)
            env.reset(seed=args.seed)

            robot_pos = base.scene["robot"].data.root_pos_w[index].detach().cpu().numpy()
            cart_pos = base.scene["rickshaw"].data.root_pos_w[index].detach().cpu().numpy()
            target = 0.5 * (robot_pos + cart_pos)
            target[2] = max(float(target[2]), 0.85)
            camera_position = target + np.array((0.0, 4.2, 1.4))
            base.sim.set_camera_view(
                tuple(float(value) for value in camera_position),
                tuple(float(value) for value in target),
            )

            frames = 0
            for step in range(1, args.steps + 1):
                env.step(zero_action)
                if step % sample_interval != 0 and step != args.steps:
                    continue
                simulation_app.update()
                image = base.render(recompute=True)
                _write_frame(writer, image, slope=slope, index=index)
                frames += 1
            manifest.append(
                {
                    "slope": slope,
                    "gradient": slope,
                    "steps": args.steps,
                    "frames": frames,
                    "fps": actual_fps,
                    "sample_interval": sample_interval,
                    "path": str(output),
                }
            )
            print(f"rendered slope {slope:+.2f}: {frames} frames", flush=True)

        writer.release()
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"empty reset video: {output}")
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "steps_per_slope": args.steps,
                    "requested_fps": args.fps,
                    "actual_fps": actual_fps,
                    "sample_interval": sample_interval,
                    "videos": manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        writer.release()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
