"""Mjlab observation, reward, and curriculum adapters."""

from __future__ import annotations

from typing import Any

import torch

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)

from .mdp.observations import (
    ACTOR_OBSERVATION_NOISE_SCALE,
    SLOPE_LOWER,
    SLOPE_UPPER,
    assemble_actor_observation,
    gait_phase_observation,
)
from .mdp.rewards import (
    FOOT_SWING_HEIGHT_TARGET_M,
    HIP_YAW_ROLL_POLICY_INDICES,
    HIP_YAW_ROLL_REFERENCE_SCALE_RAD,
    HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    HITCH_HEIGHT_RECOVERY_SCALE_M,
    JOINT_LIMIT_NORMALIZER_RAD,
    PELVIS_HEIGHT_BOUNDS_M,
    PELVIS_HEIGHT_ERROR_SCALE_M,
    action_rate_l2_value,
    fat2_prior_exp_value,
    feet_gait_value,
    feet_slide_value,
    feet_swing_height_value,
    heading_error_l2_value,
    hip_yaw_roll_reference_l2_value,
    hitch_height_exp_value,
    hitch_height_recovery_l2_value,
    joint_power_l1_value,
    lateral_error_l2_value,
    pelvis_height_limits_l2_value,
    terrain_normal_velocity_l2_value,
    track_speed_exp_value,
    zmp_margin_barrier_value,
)
from .mjlab_events import ensure_mjlab_physical_state


def _shape_probe(env: Any, *shape: int) -> torch.Tensor:
    if hasattr(env, "observation_manager"):
        raise RuntimeError("observation state was not initialized by the startup event")
    return torch.empty((env.num_envs, *shape), device=env.device)


def _dynamic_privilege(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]

    def project(vector: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.sum(vector * env.path_tangent_w, dim=-1),
                torch.sum(vector * env.path_lateral_w, dim=-1),
                torch.sum(vector * env.path_normal_w, dim=-1),
            ),
            dim=-1,
        )

    basis = torch.stack((env.path_tangent_w, env.path_lateral_w, env.path_normal_w), dim=1)
    wrench = env.rickshaw_state.connection_truth_wrench_w
    force_sln = torch.einsum("nsw,ncw->nsc", wrench[..., :3], basis)
    torque_sln = torch.einsum("nsw,ncw->nsc", wrench[..., 3:], basis)
    result = torch.cat(
        (
            project(robot.data.root_link_lin_vel_w),
            project(cart.data.root_link_lin_vel_w),
            env.rickshaw_state.pitch[:, None],
            env.rickshaw_state.wheel_normal_force,
            torch.cat((force_sln, torque_sln), dim=-1).reshape(env.num_envs, -1),
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, TEACHER_DYNAMIC_DIM):
        raise RuntimeError("teacher dynamic observation is not 21-D")
    return result


def _update_observation_state(env: Any) -> None:
    ensure_mjlab_physical_state(env)
    step = int(env.common_step_counter)
    if env._mjlab_observation_state_step == step:
        return
    env._mjlab_observation_state_step = step
    robot = env.scene["robot"]
    current = assemble_actor_observation(
        robot.data.root_link_ang_vel_b,
        robot.data.projected_gravity_b,
        env.command_state.v_ref,
        env.path_state.lateral_error,
        env.path_state.heading_error,
        robot.data.joint_pos[:, env.policy_joint_ids],
        env.action_state.q_ref,
        robot.data.joint_vel[:, env.policy_joint_ids],
        env.action_state.target,
        gait_phase_observation(env.episode_length_buf * env.step_dt),
    )
    if env.cfg.observation_noise_enabled:
        noise = torch.tensor(ACTOR_OBSERVATION_NOISE_SCALE, device=env.device)
        current = current + torch.empty_like(current).uniform_(-1.0, 1.0) * noise
    env.observation_history_state.advance(current)
    env.teacher_dynamic_history_state.advance(_dynamic_privilege(env))


def current_actor_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        return _shape_probe(env, ACTOR_OBSERVATION_DIM)
    _update_observation_state(env)
    return env.observation_history_state.current


def actor_observation_history(env: Any, history_length: int = HISTORY_LENGTH) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        return _shape_probe(env, history_length, ACTOR_OBSERVATION_DIM)
    _update_observation_state(env)
    history = env.observation_history_state.history
    if history is None:
        raise RuntimeError("actor history is disabled")
    return history


def teacher_dynamic_history(env: Any, history_length: int = HISTORY_LENGTH) -> torch.Tensor:
    if not hasattr(env, "teacher_dynamic_history_state"):
        return _shape_probe(env, history_length, TEACHER_DYNAMIC_DIM)
    _update_observation_state(env)
    history = env.teacher_dynamic_history_state.history
    if history is None:
        raise RuntimeError("teacher dynamic history is disabled")
    return history


def teacher_static(env: Any) -> torch.Tensor:
    if not hasattr(env, "normalized_teacher_static_domain"):
        return _shape_probe(env, TEACHER_STATIC_DIM)
    slope = torch.clamp(
        2.0 * (env.slope[:, None] - SLOPE_LOWER) / (SLOPE_UPPER - SLOPE_LOWER) - 1.0,
        -1.0,
        1.0,
    )
    return torch.cat((env.normalized_teacher_static_domain, slope), dim=-1)


def critic_privileged_state(env: Any) -> torch.Tensor:
    if not hasattr(env, "teacher_dynamic_history_state"):
        return _shape_probe(env, CRITIC_PRIVILEGED_DIM)
    _update_observation_state(env)
    result = torch.cat(
        (
            teacher_static(env),
            env.teacher_dynamic_history_state.current,
            env.rickshaw_state.connection_residual[:, None],
            env.stability_state.zmp_margin[:, None],
            env.analytic_force_state.a_s[:, None],
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, CRITIC_PRIVILEGED_DIM):
        raise RuntimeError("critic privileged observation is not 34-D")
    return result


def track_speed_exp(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    cfg = env.runtime_cfg.command
    return track_speed_exp_value(
        env.command_state.v_ref,
        env.policy_robot_speed_s,
        env.policy_robot_speed_l,
        cfg.maximum / cfg.limit_maximum,
    )


def lateral_error_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return lateral_error_l2_value(env.path_state.lateral_error)


def heading_error_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return heading_error_l2_value(env.path_state.heading_error)


def zmp_margin_barrier(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return zmp_margin_barrier_value(env.stability_state.zmp_margin)


def hitch_height_exp(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return hitch_height_exp_value(
        env.rickshaw_state.hitch_height,
        env.rickshaw_pose_cfg.hitch_height_target,
        env.rickshaw_state.two_wheel_contact,
    )


def hitch_height_recovery_l2(
    env: Any,
    deadband: float = HITCH_HEIGHT_RECOVERY_DEADBAND_M,
    scale: float = HITCH_HEIGHT_RECOVERY_SCALE_M,
) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return hitch_height_recovery_l2_value(
        env.rickshaw_state.hitch_height,
        env.rickshaw_pose_cfg.hitch_height_target,
        deadband=deadband,
        scale=scale,
    )


def fat2_prior_exp(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return fat2_prior_exp_value(
        env.stability_state.torso_pitch,
        env.stability_state.theta_fat,
        env.stability_state.fat_valid,
    )


def feet_gait(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    sensor = env.scene["robot_contacts"]
    contact = sensor.data.current_contact_time > 0
    return feet_gait_value(env.episode_length_buf * env.step_dt, contact, env.command_state.v_ref)


def feet_swing_height(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    robot = env.scene["robot"]
    sensor = env.scene["robot_contacts"]
    contact = sensor.data.found > 0
    height = torch.sum(
        (robot.data.body_link_pos_w[:, env.foot_body_ids] - env.scene.env_origins[:, None, :])
        * env.path_normal_w[:, None, :],
        dim=-1,
    )
    return feet_swing_height_value(height, contact, target_height=FOOT_SWING_HEIGHT_TARGET_M)


def feet_slide(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    robot = env.scene["robot"]
    sensor = env.scene["robot_contacts"]
    contact = sensor.data.found > 0
    return feet_slide_value(robot.data.body_link_lin_vel_w[:, env.foot_body_ids], contact)


def terrain_normal_velocity_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    return terrain_normal_velocity_l2_value(env.policy_robot_velocity_n)


def joint_power_l1(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    robot = env.scene["robot"]
    return joint_power_l1_value(robot.data.actuator_force, robot.data.joint_vel[:, env.policy_joint_ids])


def joint_acc_l2(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.joint_acc[:, env.policy_joint_ids]), dim=-1)


def action_rate_l2(env: Any) -> torch.Tensor:
    return action_rate_l2_value(env.action_state.raw_action, env.action_state.prev_raw_action)


def hip_yaw_roll_reference_l2(env: Any) -> torch.Tensor:
    index = torch.tensor(HIP_YAW_ROLL_POLICY_INDICES, device=env.device)
    position = env.scene["robot"].data.joint_pos[:, env.policy_joint_ids]
    return hip_yaw_roll_reference_l2_value(
        position[:, index], env.action_state.q_ref[:, index], scale=HIP_YAW_ROLL_REFERENCE_SCALE_RAD
    )


def pelvis_height_limits_l2(env: Any) -> torch.Tensor:
    ensure_mjlab_physical_state(env)
    pelvis = env.scene["robot"].data.body_link_pos_w[:, env.pelvis_body_id]
    height = torch.sum((pelvis - env.scene.env_origins) * env.path_normal_w, dim=-1)
    return pelvis_height_limits_l2_value(height, bounds=PELVIS_HEIGHT_BOUNDS_M, scale=PELVIS_HEIGHT_ERROR_SCALE_M)


def joint_position_limits(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    position = robot.data.joint_pos[:, env.policy_joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, env.policy_joint_ids]
    violation = torch.relu(limits[..., 0] - position) + torch.relu(position - limits[..., 1])
    return torch.sum(violation / JOINT_LIMIT_NORMALIZER_RAD, dim=-1)


def termination(env: Any) -> torch.Tensor:
    return (env.termination_manager.terminated & ~env.termination_manager.time_outs).float()


def speed_command_levels(env: Any, env_ids: torch.Tensor, reward_term_name: str = "track_speed_exp") -> torch.Tensor:
    cfg = env.runtime_cfg.command
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_cfg.weight * 0.8:
        cfg.maximum = min(cfg.maximum + cfg.curriculum_step, cfg.limit_maximum)
    return torch.tensor(cfg.maximum, device=env.device)


__all__ = [
    "actor_observation_history",
    "critic_privileged_state",
    "current_actor_observation",
    "fat2_prior_exp",
    "feet_gait",
    "feet_slide",
    "feet_swing_height",
    "heading_error_l2",
    "hip_yaw_roll_reference_l2",
    "hitch_height_exp",
    "hitch_height_recovery_l2",
    "joint_acc_l2",
    "joint_position_limits",
    "joint_power_l1",
    "lateral_error_l2",
    "pelvis_height_limits_l2",
    "action_rate_l2",
    "speed_command_levels",
    "teacher_dynamic_history",
    "teacher_static",
    "terrain_normal_velocity_l2",
    "termination",
    "track_speed_exp",
    "zmp_margin_barrier",
]
