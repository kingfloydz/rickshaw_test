#!/usr/bin/env python3
"""Render all configured reset poses from multiple camera views."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "g1_rickshaw_lab"
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", ROOT.parent / "IsaacLab"))
for path in (
    SOURCE,
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher
from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS, terrain_index_for_gradient


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir", type=Path, default=ROOT / "outputs" / "reset_render_047"
)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def terrain_indices(slopes: tuple[float, ...]) -> tuple[list[int], list[int]]:
    indices = tuple(terrain_index_for_gradient(slope) for slope in slopes)
    return [level for level, _ in indices], [terrain_type for _, terrain_type in indices]


def main() -> None:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import (
        G1RickshawDirectionalSlopePlayEnvCfg,
        PLAY_TASK_ID,
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
    try:
        levels, columns = terrain_indices(SLOPE_GRADIENTS)
        terrain = base.scene.terrain
        level_tensor = torch.tensor(levels, device=base.device, dtype=torch.long)
        column_tensor = torch.tensor(columns, device=base.device, dtype=torch.long)
        terrain.terrain_levels.copy_(level_tensor)
        terrain.terrain_types.copy_(column_tensor)
        terrain.env_origins.copy_(terrain.terrain_origins[level_tensor, column_tensor])
        env.reset(seed=args.seed)

        expected = torch.tensor(
            SLOPE_GRADIENTS, device=base.device, dtype=base.slope.dtype
        )
        if not torch.allclose(base.slope, expected, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(f"slope assignment failed: {base.slope.tolist()}")

        robot_pos = base.scene["robot"].data.root_pos_w.detach().cpu().numpy()
        cart_pos = base.scene["rickshaw"].data.root_pos_w.detach().cpu().numpy()
        records: list[dict[str, object]] = []
        views: list[tuple[tuple[float, ...], tuple[float, ...], Path, dict[str, object]]] = []
        for index, slope in enumerate(SLOPE_GRADIENTS):
            target = 0.5 * (robot_pos[index] + cart_pos[index])
            target[2] = max(float(target[2]), 0.85)
            view_offsets = {
                "side": np.array((0.0, 4.2, 1.4)),
                "front_oblique": np.array((3.2, 3.2, 2.0)),
                "top": np.array((0.01, 0.01, 6.0)),
            }
            slope_label = f"{slope:+.2f}".replace("+", "p").replace("-", "m")
            for view_name, offset in view_offsets.items():
                position = target + offset
                camera_position = tuple(float(value) for value in position)
                look_at = tuple(float(value) for value in target)
                path = output_dir / f"slope_{slope_label}_{view_name}.png"
                metadata = {
                    "slope": slope,
                    "view": view_name,
                    "path": str(path),
                    "camera_position_w": position.tolist(),
                    "look_at_w": target.tolist(),
                }
                views.append((camera_position, look_at, path, metadata))

        for camera_position, look_at, path, metadata in views:
            base.sim.set_camera_view(camera_position, look_at)
            image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            for _ in range(8):
                simulation_app.update()
                image = np.asarray(base.render(recompute=True))
                if np.count_nonzero(image) > 0:
                    break
            if image.size == 0 or image.ndim != 3 or image.shape[2] < 3:
                raise RuntimeError(f"camera returned an invalid image for {path.name}")
            Image.fromarray(image[..., :3].astype(np.uint8), mode="RGB").save(path)
            metadata["resolution"] = [int(image.shape[1]), int(image.shape[0])]
            metadata["nonzero_pixels"] = int(np.count_nonzero(image[..., :3]))
            if metadata["nonzero_pixels"] == 0:
                raise RuntimeError(f"camera returned a blank image for {path.name}")
            records.append(metadata)

        manifest = {
            "reset_pose_path": str((ROOT / "config" / "reset_poses.yaml").resolve()),
            "hitch_spacing_m": 0.47,
            "slopes": list(SLOPE_GRADIENTS),
            "images": records,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"rendered {len(records)} reset images: {output_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
