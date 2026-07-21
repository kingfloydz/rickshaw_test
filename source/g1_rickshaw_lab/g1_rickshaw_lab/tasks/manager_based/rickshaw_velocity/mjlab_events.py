"""mjlab reset events for the rigid robot-rickshaw assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g1_rickshaw_lab.static_equilibrium import solve_mujoco_static_equilibrium

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_from_mujoco_statics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    gradient: float = 0.0,
) -> None:
    """Write a cached MuJoCo static solution with no settling controller."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    cache = getattr(env, "_mujoco_static_equilibria", None)
    if cache is None:
        cache = {}
        env._mujoco_static_equilibria = cache
    key = round(float(gradient), 8)
    if key not in cache:
        cache[key] = solve_mujoco_static_equilibrium(env.sim.mj_model, gradient)
    solution = cache[key]

    robot = env.scene["robot"]
    rickshaw = env.scene["rickshaw"]
    model = env.sim.mj_model
    qpos = solution.qpos

    def qadr(name: str) -> int:
        return int(model.joint(name).qposadr[0])

    robot_root = qadr("robot/floating_base_joint")
    cart_root = qadr("rickshaw/floating_base_joint")
    origins = env.scene.env_origins[env_ids]
    count = len(env_ids)

    robot_pose = torch.as_tensor(qpos[robot_root : robot_root + 7], device=env.device, dtype=torch.float32).repeat(
        count, 1
    )
    robot_pose[:, :3] += origins
    robot.write_root_link_pose_to_sim(robot_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros((count, 6), device=env.device), env_ids=env_ids)

    cart_pose = torch.as_tensor(qpos[cart_root : cart_root + 7], device=env.device, dtype=torch.float32).repeat(
        count, 1
    )
    cart_pose[:, :3] += origins
    rickshaw.write_root_link_pose_to_sim(cart_pose, env_ids=env_ids)
    rickshaw.write_root_link_velocity_to_sim(torch.zeros((count, 6), device=env.device), env_ids=env_ids)

    robot_joint_pos = torch.tensor(
        [qpos[qadr(f"robot/{name}")] for name in robot.joint_names],
        device=env.device,
        dtype=torch.float32,
    ).repeat(count, 1)
    robot.write_joint_state_to_sim(
        robot_joint_pos,
        torch.zeros_like(robot_joint_pos),
        env_ids=env_ids,
    )
    joint_target = torch.as_tensor(
        solution.joint_position_target,
        device=env.device,
        dtype=torch.float32,
    ).repeat(count, 1)
    robot.set_joint_position_target(joint_target, env_ids=env_ids)
    wheel_pos = torch.tensor(
        [qpos[qadr("rickshaw/left_wheel_joint")], qpos[qadr("rickshaw/right_wheel_joint")]],
        device=env.device,
        dtype=torch.float32,
    ).repeat(count, 1)
    rickshaw.write_joint_state_to_sim(wheel_pos, torch.zeros_like(wheel_pos), env_ids=env_ids)

    if not hasattr(env, "fat2_reference_angle"):
        env.fat2_reference_angle = torch.zeros(env.num_envs, device=env.device)
    env.fat2_reference_angle[env_ids] = solution.fat2_reference_angle


__all__ = ["reset_from_mujoco_statics"]
