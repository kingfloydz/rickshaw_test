#!/usr/bin/env python3
"""Create a Mjlab environment, run zero-action steps, and render initialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=19)
    parser.add_argument(
        "--render-env-index",
        type=int,
        default=0,
        help="Environment slot to render; play mode maps slots 0-18 to the 19 slopes.",
    )
    parser.add_argument(
        "--render-env-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional set of slots to render into suffixed output images.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    render_indices = (
        [args.render_env_index]
        if args.render_env_indices is None
        else list(args.render_env_indices)
    )
    if any(index < 0 or index >= args.num_envs for index in render_indices):
        parser.error("rendered environment indices must lie in [0, --num-envs)")
    if len(render_indices) > 1 and args.output is None:
        parser.error("--render-env-indices with multiple slots requires --output")

    import torch

    import g1_rickshaw_lab.tasks  # noqa: F401
    from g1_rickshaw_lab.configuration import (
        ARM_HARDWARE_EFFORT_LIMITS,
        G1_JOINT_ORDER,
        LOWER_HARDWARE_EFFORT_LIMITS,
        WAIST_HARDWARE_EFFORT_LIMITS,
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_env_cfg(
        "Mjlab-G1-Rickshaw-Directional-Slope-Student", play=True
    )
    cfg.scene.num_envs = args.num_envs
    cfg.viewer.env_idx = render_indices[0]
    env = ManagerBasedRlEnv(
        cfg,
        device=device,
        render_mode="rgb_array" if args.output is not None else None,
    )
    try:
        observations, _ = env.reset(seed=42)

        def physical_metrics() -> dict[str, object]:
            robot = env.scene["robot"]
            cart = env.scene["rickshaw"]
            wheel_centers = cart.data.body_link_pos_w[:, env.wheel_body_ids]
            wheel_plane_distance = torch.sum(
                (wheel_centers - env.scene.env_origins[:, None, :])
                * env.path_normal_w[:, None, :],
                dim=-1,
            )
            connection_error = torch.linalg.vector_norm(
                robot.data.site_pos_w[:, env.grasp_site_ids]
                - cart.data.site_pos_w[:, env.hitch_site_ids],
                dim=-1,
            )
            joint_position = env.static_joint_position_table[env.slope_slot]
            q_ref = env.static_q_ref_table[env.slope_slot]
            joint_reset_error = torch.abs(
                robot.data.joint_pos[:, env.policy_joint_ids] - joint_position
            )
            joint_target_error = torch.abs(
                robot.data.joint_pos_target[:, env.policy_joint_ids] - q_ref
            )
            effort_limit = torch.tensor(
                LOWER_HARDWARE_EFFORT_LIMITS
                + WAIST_HARDWARE_EFFORT_LIMITS
                + ARM_HARDWARE_EFFORT_LIMITS,
                device=device,
            )
            static_torque_ratio = torch.abs(
                env.static_actuator_torque_table[env.slope_slot]
            ) / effort_limit
            actuator_torque_ratio = torch.abs(
                robot.data.actuator_force[:, env.policy_actuator_ids]
            ) / effort_limit
            reset_worst = joint_reset_error.argmax(dim=-1).cpu().tolist()
            static_worst = static_torque_ratio.argmax(dim=-1).cpu().tolist()
            current_worst = torch.gather(
                robot.data.joint_pos[:, env.policy_joint_ids],
                1,
                joint_reset_error.argmax(dim=-1, keepdim=True),
            ).squeeze(-1)
            static_position_worst = torch.gather(
                joint_position,
                1,
                joint_reset_error.argmax(dim=-1, keepdim=True),
            ).squeeze(-1)

            def contact_counts(sensor_name: str) -> list[int]:
                found = env.scene[sensor_name].data.found
                if found is None:
                    raise RuntimeError(f"{sensor_name} does not expose contact matches")
                return (found > 0).reshape(args.num_envs, -1).sum(dim=-1).cpu().tolist()

            return {
                "wheel_plane_distance_m": wheel_plane_distance.cpu().tolist(),
                "wheel_radius_error_max_m": float(
                    torch.max(torch.abs(wheel_plane_distance - 0.3)).item()
                ),
                "wheel_contact_counts": contact_counts("wheel_contacts"),
                "foot_contact_counts": contact_counts("robot_contacts"),
                "connection_position_error_m": connection_error.max(dim=-1).values.cpu().tolist(),
                "connection_position_error_max_m": float(torch.max(connection_error).item()),
                "joint_reset_error_max_rad": joint_reset_error.max(dim=-1).values.cpu().tolist(),
                "joint_reset_error_worst_joint": [
                    G1_JOINT_ORDER[index] for index in reset_worst
                ],
                "joint_position_worst_rad": current_worst.cpu().tolist(),
                "joint_static_position_worst_rad": static_position_worst.cpu().tolist(),
                "joint_velocity_max_rad_s": torch.abs(
                    robot.data.joint_vel[:, env.policy_joint_ids]
                ).max(dim=-1).values.cpu().tolist(),
                "joint_target_error_max_rad": joint_target_error.max(dim=-1).values.cpu().tolist(),
                "static_torque_ratio_max": static_torque_ratio.max(dim=-1).values.cpu().tolist(),
                "static_torque_ratio_worst_joint": [
                    G1_JOINT_ORDER[index] for index in static_worst
                ],
                "actuator_torque_ratio_max": actuator_torque_ratio.max(dim=-1).values.cpu().tolist(),
            }

        metrics_before_step = physical_metrics()
        actions = torch.zeros(
            (args.num_envs, env.action_manager.total_action_dim), device=device
        )
        for _ in range(args.steps):
            observations, reward, terminated, truncated, _ = env.step(actions)
        metrics_after_step = physical_metrics()
        output_paths: list[Path] = []
        if args.output is not None:
            import imageio.v3 as iio

            for index in render_indices:
                env.cfg.viewer.env_idx = index
                image = env.render()
                if image is None:
                    raise RuntimeError("Mjlab offscreen renderer returned no image")
                output = (
                    args.output
                    if len(render_indices) == 1
                    else args.output.with_name(
                        f"{args.output.stem}_slot_{index:02d}{args.output.suffix}"
                    )
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(output, image)
                output_paths.append(output.resolve())
        print(
            json.dumps(
                {
                    "num_envs": args.num_envs,
                    "device": device,
                    "observation_shapes": {
                        name: list(value.shape)
                        for name, value in observations.items()
                    },
                    "action_dim": env.action_manager.total_action_dim,
                    "zero_action_steps": args.steps,
                    "static_solution_count": len(env._mujoco_static_equilibria),
                    "slopes": env.slope.detach().cpu().tolist(),
                    "render_env_indices": render_indices,
                    "rendered_slopes": [float(env.slope[index].item()) for index in render_indices],
                    "physical_metrics_before_zero_step": metrics_before_step,
                    "physical_metrics_after_zero_action_steps": metrics_after_step,
                    "reward": reward.detach().cpu().tolist(),
                    "terminated": terminated.detach().cpu().tolist(),
                    "truncated": truncated.detach().cpu().tolist(),
                    "outputs": [str(path) for path in output_paths],
                },
                indent=2,
            )
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
