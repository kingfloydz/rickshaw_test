#!/usr/bin/env python3
"""Produce fixed-seed, configured-slope policy diagnostics in Isaac Lab."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Any

from _isaaclab_wrappers import (
    add_isaaclab_sources_to_path,
    add_project_source_to_path,
    require_existing_file,
)

add_project_source_to_path()

from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    COMMAND_PHASE_LABELS,
    CROSS_CASE_LABELS,
    FORMAL_EVALUATION_COMMAND_PROTOCOL,
    FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
    METRIC_DEFINITIONS,
    POLICY_DIAGNOSTIC_SCHEMA_VERSION,
    SIGNED_SLOPES,
    PolicyEvaluationAccumulator,
    command_phase_labels,
    d6_wrench_channels,
    evaluate_s2_return_floor,
    slope_label,
    validate_s1_baseline_diagnostic_report,
)
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_STAGE_KEY,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
    require_pinned_rsl_rl,
)
from g1_rickshaw_lab.validation import (  # noqa: E402
    utc_timestamp,
    write_json_atomic,
)


DEFAULT_TASK = "Isaac-G1-Rickshaw-Directional-Slope-v0"
SUPPORTED_STAGES = {
    "s0_teacher",
    "s1_context_distillation",
    "s2_student_ppo",
}
CURRICULUM_NAMES = ("training",)


class PolicyHandle:
    """Uniform distribution interface over native RSL and S1 actors."""

    def __init__(self, actor: Any, *, kind: str) -> None:
        if kind not in {"standalone_student", "rsl_student", "rsl_teacher"}:
            raise ValueError(f"unknown policy handle kind {kind!r}")
        self.actor = actor
        self.kind = kind

    def distribution(self, observation: Any):
        if self.kind == "standalone_student":
            context = self.actor.encode(observation["history"])
            policy = self.actor.actor
        else:
            context = self.actor.encode(observation)
            policy = self.actor.policy
        return policy.distribution(observation["policy"], context)


def _parser() -> argparse.ArgumentParser:
    add_isaaclab_sources_to_path()
    require_pinned_rsl_rl()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument(
        "--s1-baseline-report",
        default=None,
        help=(
            "Optional S1 fixed-seed TRAINING report for an S2 return comparison."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS)
    parser.add_argument("--episodes-per-slope", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--curriculum-stages", nargs="+", choices=CURRICULUM_NAMES, default=("training",))
    parser.add_argument("--max-policy-steps-per-seed", type=int, default=6000)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args: argparse.Namespace, stage: str) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if not args.seeds:
        raise ValueError("fixed seeds must be non-empty")
    quota_divisor = len(args.seeds) * len(CROSS_CASE_LABELS)
    if (
        args.episodes_per_slope <= 0
        or args.episodes_per_slope % quota_divisor != 0
        or args.max_policy_steps_per_seed <= 0
    ):
        raise ValueError(
            "diagnostics require a positive episode quota "
            f"divisible by seeds={quota_divisor}, and a positive step limit"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("fixed seeds must be unique")
    if len(set(args.curriculum_stages)) != len(args.curriculum_stages):
        raise ValueError("curriculum stages must be unique")
    if args.s1_baseline_report is not None and stage != "s2_student_ppo":
        raise ValueError("--s1-baseline-report applies only to S2 diagnostics")
    if (
        stage == "s2_student_ppo"
        and args.s1_baseline_report is not None
        and list(args.curriculum_stages) != ["training"]
    ):
        raise ValueError("S2 return comparison requires TRAINING evaluation")
def _configure_fixed_stage(env_cfg: Any, stage_name: str, fat2_weight: float | None) -> None:
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    if stage_name != "training":
        raise ValueError(f"unknown evaluation stage {stage_name!r}")
    env_cfg.curriculum = None
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.domain_randomization = replace(
        env_cfg.domain_randomization,
        enabled=False,
        curriculum=mdp.CurriculumScheduleCfg(static_hand_load_iterations=0),
    )
    env_cfg.events.initialize_domain.params = {"cfg": env_cfg.domain_randomization}
    if fat2_weight is not None:
        env_cfg.rewards.fat2_prior_exp.weight = float(fat2_weight)


def _assign_fixed_slopes(base_env: Any) -> Any:
    """Bind every configured gradient for the single training distribution."""

    import torch
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    slots = torch.arange(base_env.num_envs, device=base_env.device) % len(SIGNED_SLOPES)
    levels_lut = torch.tensor(
        SLOPE_TERRAIN_LEVELS, device=base_env.device, dtype=torch.long
    )
    columns_lut = torch.tensor(
        SLOPE_TERRAIN_TYPES, device=base_env.device, dtype=torch.long
    )
    levels = levels_lut[slots]
    columns = columns_lut[slots]
    terrain = base_env.scene.terrain
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
    mdp.update_slope_frame(base_env)
    expected = torch.tensor(SIGNED_SLOPES, device=base_env.device)[slots]
    if not torch.allclose(base_env.slope, expected, atol=1.0e-7, rtol=0.0):
        raise RuntimeError(
            "fixed evaluation terrain does not resolve to every configured slope"
        )

    cfg = base_env.cfg.domain_randomization.curriculum
    base_env.curriculum_runtime_state = mdp.CurriculumRuntimeState.create(
        columns, torch.sign(expected).to(dtype=torch.long), cfg
    )
    base_env.curriculum_runtime_state.set_iteration(
        cfg.static_hand_load_iterations
    )
    base_env.curriculum_runtime_state.activate(
        torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    )
    base_env.curriculum_stage_per_env = (
        base_env.curriculum_runtime_state.stage_per_environment()
    )
    return slots


def _apply_evaluation_command_protocol(base_env: Any, active: Any) -> None:
    """Drive every counted episode through all four command phases deterministically."""

    import torch

    if active.dtype != torch.bool or active.shape != (base_env.num_envs,):
        raise ValueError("active evaluation mask must be boolean with shape [num_envs]")
    if not torch.any(active):
        return
    elapsed_s = base_env.episode_length_buf.to(dtype=torch.float32) * float(
        base_env.step_dt
    )
    # Hold zero briefly, accelerate to and cruise at 1 m/s, then brake to a
    # final standing interval before the 20 s timeout.
    moving = (elapsed_s >= 1.0) & (elapsed_s < 10.0)
    target = moving.to(dtype=base_env.command_state.v_sample.dtype)
    base_env.command_state.v_sample[active] = target[active]
    # Disable the regular 10 s random resampler without bypassing the deployable
    # acceleration/jerk limiter that advances v_ref and a_ref.
    base_env.command_state.resampling_elapsed_s[active] = 0.0


def _episode_fell(*, timed_out: bool, causes: list[str]) -> bool:
    """Classify safety termination as a fall even on the nominal timeout step."""

    return any(cause != "time_out" for cause in causes) or not timed_out


def _load_policy(
    env: Any,
    checkpoint_path: Path,
    checkpoint: Any,
    stage: str,
    device: str,
    task: str,
) -> tuple[PolicyHandle, list[Any]]:
    keepalive: list[Any] = []
    latent_dim = int(
        checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]["latent_dim"]
    )
    if stage == "s1_context_distillation":
        from g1_rickshaw_lab.rl import G1RickshawStudentActor

        state = checkpoint["model_state_dict"]
        model = G1RickshawStudentActor(latent_dim).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        keepalive.append(model)
        return PolicyHandle(model, kind="standalone_student"), keepalive

    from rsl_rl.runners import OnPolicyRunner
    registry_key = "rsl_rl_cfg_entry_point" if stage == "s0_teacher" else "rsl_rl_student_cfg_entry_point"
    agent_cfg = _load_rsl_runner_cfg(task, registry_key, device, latent_dim)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(
        os.fspath(checkpoint_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        strict=True,
    )
    runner.alg.actor.eval()
    keepalive.append(runner)
    kind = "rsl_teacher" if stage == "s0_teacher" else "rsl_student"
    return PolicyHandle(runner.alg.actor, kind=kind), keepalive


def _load_teacher_policy(
    env: Any, checkpoint_path: Path, device: str, task: str
) -> tuple[PolicyHandle, list[Any]]:
    from rsl_rl.runners import OnPolicyRunner
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stage="s0_teacher",
    )
    latent_dim = int(
        checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]["latent_dim"]
    )
    agent_cfg = _load_rsl_runner_cfg(
        task, "rsl_rl_cfg_entry_point", device, latent_dim
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(
        os.fspath(checkpoint_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        strict=True,
    )
    runner.alg.actor.eval()
    return PolicyHandle(runner.alg.actor, kind="rsl_teacher"), [runner]


def _load_rsl_runner_cfg(
    task: str, registry_key: str, device: str, latent_dim: int
):
    """Load the fixed RSL-RL 5 runner configuration."""

    from isaaclab_tasks.utils import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(task, registry_key)
    agent_cfg.device = device
    agent_cfg.actor.latent_dim = latent_dim
    return agent_cfg


def _sample_metrics(base_env: Any, teacher_kl: Any | None) -> dict[str, Any]:  # noqa: C901
    import torch
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    robot = base_env.scene["robot"]
    state = base_env.rickshaw_state
    stability = base_env.stability_state
    analytic = base_env.analytic_force_state
    actual_speed = torch.sum(robot.data.root_lin_vel_w * base_env.path_tangent_w, dim=-1)
    speed_error = actual_speed - base_env.command_state.v_ref
    persistent_cfg = base_env.cfg.terminations.persistent_safety.params["cfg"]
    overspeed = actual_speed > base_env.command_state.v_ref + persistent_cfg.overspeed_margin
    pitch_error = state.pitch - mdp.target_pitch_from_hitch_height(
        base_env.rickshaw_pose_cfg
    )
    hitch_error = state.hitch_height - float(base_env.rickshaw_pose_cfg.hitch_height_target)

    contact_sensor = base_env.scene["robot_contacts"]
    foot_contact = contact_sensor.data.current_contact_time[:, base_env.foot_sensor_ids] > 0.0
    foot_velocity = robot.data.body_lin_vel_w[:, base_env.foot_body_ids]
    foot_s = torch.sum(foot_velocity * base_env.path_tangent_w[:, None, :], dim=-1)
    foot_y = torch.sum(foot_velocity * base_env.path_lateral_w[:, None, :], dim=-1)
    foot_slip = torch.sum(torch.sqrt(foot_s.square() + foot_y.square()) * foot_contact, dim=-1)

    dt = float(base_env.step_dt)
    action = base_env.action_state
    action_rate = torch.sqrt(torch.mean(((action.target - action.prev_target) / dt).square(), dim=-1))
    action_jerk = torch.sqrt(
        torch.mean(((action.target - 2.0 * action.prev_target + action.prev_prev_target) / (dt * dt)).square(), dim=-1)
    )

    ids = base_env.policy_joint_ids
    torque = robot.data.applied_torque[:, ids]
    velocity = robot.data.joint_vel[:, ids]
    effort = mdp.actuator_effort_limits(robot, ids)
    torque_margin = 1.0 - torch.abs(torque) / effort
    leg_margin = torch.amin(torque_margin[:, :12], dim=-1)
    arm_margin = torch.amin(torque_margin[:, 15:], dim=-1)
    power = torch.sum(torch.abs(torque * velocity), dim=-1)

    wrench = state.d6_wrench_w
    d6_channels = d6_wrench_channels(wrench)
    d6_force = d6_channels["force"]
    d6_torque = d6_channels["torque"]
    force_asymmetry = d6_channels["force_asymmetry"]
    torque_asymmetry = d6_channels["torque_asymmetry"]
    # The adapter stores cart-on-robot reaction; analytic T_s/T_n are
    # robot-on-cart forces.
    force_on_cart_w = -state.hand_force_w
    projected_t_s = torch.sum(force_on_cart_w * base_env.path_tangent_w, dim=-1)
    projected_t_n = torch.sum(force_on_cart_w * base_env.path_normal_w, dim=-1)

    def relative_error(reference: Any, measured: Any) -> Any:
        denominator = torch.maximum(torch.maximum(torch.abs(reference), torch.abs(measured)), torch.ones_like(reference))
        return torch.abs(reference - measured) / denominator

    sign_active_s = (torch.abs(analytic.t_s) > 1.0) | (torch.abs(projected_t_s) > 1.0)
    sign_active_n = (torch.abs(analytic.t_n) > 1.0) | (torch.abs(projected_t_n) > 1.0)
    sign_s = torch.where(sign_active_s, torch.sign(analytic.t_s) == torch.sign(projected_t_s), torch.ones_like(sign_active_s))
    sign_n = torch.where(sign_active_n, torch.sign(analytic.t_n) == torch.sign(projected_t_n), torch.ones_like(sign_active_n))

    result = {
        "speed_error": speed_error,
        "overspeed": overspeed,
        "lateral_error": base_env.path_state.lateral_error,
        "heading_error": torch.atan2(torch.sin(base_env.path_state.heading_error), torch.cos(base_env.path_state.heading_error)),
        "pitch_error": pitch_error,
        "hitch_height_error": hitch_error,
        "two_wheel_contact": state.two_wheel_contact,
        "wheel_normal_force_left": state.wheel_normal_force[:, 0],
        "wheel_normal_force_right": state.wheel_normal_force[:, 1],
        "foot_slip": foot_slip,
        "processed_action_rate": action_rate,
        "processed_action_jerk": action_jerk,
        "power": power,
        "d6_residual": state.d6_residual,
        "d6_force": d6_force,
        "d6_torque": d6_torque,
        "d6_force_asymmetry": force_asymmetry,
        "d6_torque_asymmetry": torque_asymmetry,
        "t_s_relative_error": relative_error(analytic.t_s, projected_t_s),
        "t_n_relative_error": relative_error(analytic.t_n, projected_t_n),
        "t_s_sign_agreement": sign_s,
        "t_n_sign_agreement": sign_n,
        "analytic_force_valid": analytic.valid,
        "fat_wrench_consistent": stability.fat_wrench_consistent,
        "fat_wrench_t_s_relative_error": stability.fat_wrench_relative_error[:, 0],
        "fat_wrench_t_n_relative_error": stability.fat_wrench_relative_error[:, 1],
        "zmp_margin": stability.zmp_margin,
        "zmp_valid": stability.zmp_valid,
        "arm_torque_margin": arm_margin,
        "leg_torque_margin": leg_margin,
    }
    if teacher_kl is not None:
        result["teacher_student_kl"] = teacher_kl
    return result


def _labels(base_env: Any, env_ids: Any) -> tuple[list[str], list[str], list[str]]:
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    stages = base_env.curriculum_stage_per_env[env_ids].detach().cpu().tolist()
    stage_names = {int(item): item.name for item in mdp.CurriculumStage}
    phases = command_phase_labels(
        base_env.command_state.v_ref[env_ids].detach().cpu().numpy(),
        base_env.command_state.a_ref[env_ids].detach().cpu().numpy(),
    )
    return (
        [stage_names[int(value)] for value in stages],
        ["RANDOM"] * len(stages),
        phases,
    )


def _run_mode(
    env: Any,
    base_env: Any,
    slope_slots: Any,
    policy: PolicyHandle,
    teacher: PolicyHandle | None,
    *,
    seeds: list[int],
    episodes_per_slope: int,
    max_steps_per_seed: int,
) -> tuple[PolicyEvaluationAccumulator, dict[str, Any]]:
    import torch

    if not seeds:
        raise ValueError("fixed evaluation seeds must be non-empty")
    quota_divisor = len(seeds) * len(CROSS_CASE_LABELS)
    if episodes_per_slope % quota_divisor != 0:
        raise ValueError(
            "episodes_per_slope must be divisible by the number of seeds"
        )
    accumulator = PolicyEvaluationAccumulator()
    completed = torch.zeros(len(SIGNED_SLOPES), dtype=torch.long, device=base_env.device)
    in_flight = torch.zeros_like(completed)
    completed_by_case = torch.zeros(
        (len(SIGNED_SLOPES), len(CROSS_CASE_LABELS)),
        dtype=torch.long,
        device=base_env.device,
    )
    in_flight_by_case = torch.zeros_like(completed_by_case)
    enrolled = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    returns_by_slope: list[list[float]] = [[] for _ in SIGNED_SLOPES]
    episode_return = torch.zeros(base_env.num_envs, device=base_env.device)
    episode_phases: list[set[str]] = [set() for _ in range(base_env.num_envs)]
    episode_cross_cases: list[str | None] = [None] * base_env.num_envs
    cause_names = tuple(
        __import__(
            "g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.terminations",
            fromlist=["TERMINATION_CAUSES"],
        ).TERMINATION_CAUSES
    )

    for seed_index, seed in enumerate(seeds):
        milestone = (seed_index + 1) * episodes_per_slope // len(seeds)
        case_milestone = milestone // len(CROSS_CASE_LABELS)
        env.seed(seed)
        observation, _ = env.reset()
        observation = observation.to(base_env.device)
        if torch.any(in_flight != 0):
            raise RuntimeError("evaluation seed changed with unfinished reserved episodes")
        enrolled.zero_()

        def reserve_episodes(candidate_ids: Any) -> None:
            """Reserve only episodes that will belong to the exact quota."""

            if candidate_ids.numel() == 0:
                return
            case_slots = torch.zeros(
                base_env.num_envs, device=base_env.device, dtype=torch.long
            )
            for slope_index in range(len(SIGNED_SLOPES)):
                for case_index in range(len(CROSS_CASE_LABELS)):
                    needed = int(
                        (
                            case_milestone
                            - completed_by_case[slope_index, case_index]
                            - in_flight_by_case[slope_index, case_index]
                        ).item()
                    )
                    if needed <= 0:
                        continue
                    candidates = candidate_ids[
                        (slope_slots[candidate_ids] == slope_index)
                        & (case_slots[candidate_ids] == case_index)
                        & ~enrolled[candidate_ids]
                    ]
                    selected = candidates[:needed]
                    enrolled[selected] = True
                    in_flight[slope_index] += selected.numel()
                    in_flight_by_case[slope_index, case_index] += selected.numel()

        reserve_episodes(
            torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
        )
        episode_return.zero_()
        for phases in episode_phases:
            phases.clear()
        episode_cross_cases[:] = [None] * base_env.num_envs
        policy_steps = 0
        while torch.any(completed < milestone):
            if policy_steps >= max_steps_per_seed:
                remaining = (milestone - completed).clamp_min(0).detach().cpu().tolist()
                raise RuntimeError(
                    f"seed {seed} exceeded {max_steps_per_seed} policy steps; "
                    f"remaining episodes per slope={remaining}; in_flight={in_flight.detach().cpu().tolist()}"
                )
            active = enrolled
            _apply_evaluation_command_protocol(base_env, active)
            # IsaacLab lazily allocates articulation state during env.step().
            # inference_mode would turn those buffers into inference tensors,
            # which later resets cannot update outside that mode.
            with torch.no_grad():
                distribution = policy.distribution(observation)
                teacher_kl = None
                if teacher is not None:
                    teacher_distribution = teacher.distribution(observation)
                    teacher_kl = torch.distributions.kl_divergence(teacher_distribution, distribution)
                if torch.any(active):
                    raw_samples = _sample_metrics(base_env, teacher_kl)
                    ids = torch.nonzero(active, as_tuple=False).flatten()
                    samples = {
                        name: value[ids].detach().cpu().numpy()
                        for name, value in raw_samples.items()
                    }
                    stage_labels, case_labels, phase_labels = _labels(base_env, ids)
                    for env_id, phase, case in zip(
                        ids.detach().cpu().tolist(),
                        phase_labels,
                        case_labels,
                        strict=True,
                    ):
                        episode_phases[int(env_id)].add(str(phase))
                        previous_case = episode_cross_cases[int(env_id)]
                        if previous_case is not None and previous_case != str(case):
                            raise RuntimeError(
                                "evaluation cross-case changed within an episode"
                            )
                        episode_cross_cases[int(env_id)] = str(case)
                    accumulator.add_step(
                        samples,
                        slope_slots[ids].detach().cpu().numpy(),
                        stage_labels=stage_labels,
                        cross_case_labels=case_labels,
                        phase_labels=phase_labels,
                    )
                actions = distribution.mean
                observation, reward, dones, extras = env.step(actions)
                observation = observation.to(base_env.device)
            episode_return += reward * active.to(dtype=reward.dtype)
            done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
            if done_ids.numel() > 0:
                time_outs = extras["time_outs"]
                if not torch.is_tensor(time_outs) or time_outs.shape != dones.shape:
                    raise RuntimeError("evaluation step did not expose per-environment timeout flags")
                cause_state = base_env.termination_cause_state
                reusable_ids: list[int] = []
                for env_id in done_ids.detach().cpu().tolist():
                    if not bool(enrolled[env_id].item()) or not bool(active[env_id].item()):
                        episode_return[env_id] = 0.0
                        episode_phases[env_id].clear()
                        episode_cross_cases[env_id] = None
                        continue
                    slope_index = int(slope_slots[env_id].item())
                    case_index = 0
                    value = float(episode_return[env_id].item())
                    cause_mask = cause_state.last_causes[env_id]
                    causes = [
                        name
                        for index, name in enumerate(cause_names)
                        if bool(cause_mask[index].item())
                    ]
                    fell = _episode_fell(
                        timed_out=bool(time_outs[env_id].item()),
                        causes=causes,
                    )
                    returns_by_slope[slope_index].append(value)
                    completed[slope_index] += 1
                    in_flight[slope_index] -= 1
                    completed_by_case[slope_index, case_index] += 1
                    in_flight_by_case[slope_index, case_index] -= 1
                    enrolled[env_id] = False
                    reusable_ids.append(env_id)
                    if not episode_phases[env_id] or episode_cross_cases[env_id] is None:
                        raise RuntimeError(
                            "active completed episode has no phase/cross-case evidence"
                        )
                    accumulator.add_episode(
                        slope_index,
                        value,
                        fell=fell,
                        causes=causes,
                        phase_labels=[
                            label
                            for label in COMMAND_PHASE_LABELS
                            if label in episode_phases[env_id]
                        ],
                        cross_case_label=episode_cross_cases[env_id],
                    )
                    episode_return[env_id] = 0.0
                    episode_phases[env_id].clear()
                    episode_cross_cases[env_id] = None
                if reusable_ids:
                    reserve_episodes(
                        torch.tensor(
                            reusable_ids, device=base_env.device, dtype=torch.long
                        )
                    )
            policy_steps += 1

        if (
            torch.any(in_flight != 0)
            or torch.any(in_flight_by_case != 0)
            or torch.any(enrolled)
            or torch.any(completed_by_case != case_milestone)
        ):
            raise RuntimeError("evaluation milestone completed with reserved episodes still active")

    if torch.any(completed != episodes_per_slope):
        raise RuntimeError(f"episode quota drifted: {completed.detach().cpu().tolist()}")
    expected_case_total = episodes_per_slope // len(CROSS_CASE_LABELS)
    if torch.any(completed_by_case != expected_case_total):
        raise RuntimeError(
            "slope/cross-case episode quota drifted: "
            f"{completed_by_case.detach().cpu().tolist()}"
        )
    flattened = [value for group in returns_by_slope for value in group]
    return_summary = {
        "episodes": len(flattened),
        "mean": sum(flattened) / len(flattened),
        "per_slope_mean": {
            slope_label(slope): sum(returns_by_slope[index]) / len(returns_by_slope[index])
            for index, slope in enumerate(SIGNED_SLOPES)
        },
    }
    return accumulator, return_summary


def _checkpoint_binding(path: Path, checkpoint: Any) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "stage": checkpoint[CHECKPOINT_STAGE_KEY],
        "curriculum_iteration": checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY),
    }


def main() -> int:  # noqa: C901
    parser = _parser()
    args = parser.parse_args()
    checkpoint_path = require_existing_file(args.checkpoint, "policy checkpoint").resolve()
    checkpoint = load_stage_checkpoint(
        checkpoint_path,
        expected_stage=SUPPORTED_STAGES,
        validate_runtime=False,
    )
    stage = checkpoint[CHECKPOINT_STAGE_KEY]
    training_configuration = dict(checkpoint[TRAINING_CONFIGURATION_KEY])
    if training_configuration["task"] != args.task:
        raise ValueError("policy evaluation task differs from checkpoint training task")
    training_parameters = training_configuration["training_parameters"]
    fat2_weight = float(training_parameters["fat2_weight"])
    rollout_steps = int(training_parameters["rollout_steps"])
    latent_dim = int(training_parameters["latent_dim"])
    _validate_args(args, stage)
    is_student = stage != "s0_teacher"

    s1_baseline_path: Path | None = None
    s1_baseline_returns: dict[str, float] | None = None
    s1_baseline_binding: dict[str, Any] | None = None
    if args.s1_baseline_report is not None:
        s1_baseline_path = require_existing_file(
            args.s1_baseline_report, "S1 baseline diagnostic report"
        ).resolve()
        s1_report = json.loads(s1_baseline_path.read_text(encoding="utf-8"))
        s1_baseline_returns = validate_s1_baseline_diagnostic_report(
            s1_report,
            fixed_seeds=args.seeds,
            episodes_per_slope=args.episodes_per_slope,
        )
        baseline_parameters = s1_report["evaluation"]
        if (
            baseline_parameters.get("latent_dim") != latent_dim
            or baseline_parameters.get("rollout_steps") != rollout_steps
            or baseline_parameters.get("fat2_weight") != fat2_weight
        ):
            raise ValueError("S1 baseline uses different training parameters")
        s1_baseline_binding = {
            "path": os.fspath(s1_baseline_path),
            "baseline_return_mean": dict(s1_baseline_returns),
        }

    teacher_path: Path | None = None
    teacher_checkpoint: Any | None = None
    if args.teacher_checkpoint is not None:
        teacher_path = require_existing_file(args.teacher_checkpoint, "teacher checkpoint").resolve()
        teacher_checkpoint = load_stage_checkpoint(
            teacher_path, expected_stage="s0_teacher", validate_runtime=False
        )
        if (
            teacher_checkpoint[TRAINING_CONFIGURATION_KEY]["training_parameters"]
            != training_parameters
        ):
            raise ValueError("teacher checkpoint uses different training parameters")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    stage_reports: dict[str, Any] = {}
    omitted_diagnostics: list[str] = []
    try:
        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401

        device = args.device
        for curriculum_name in args.curriculum_stages:
            raw_env = None
            env = None
            try:
                env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
                env_cfg.seed = args.seeds[0]
                _configure_fixed_stage(env_cfg, curriculum_name, fat2_weight)
                if is_student and teacher_path is None:
                    env_cfg.observations.teacher_dynamic_history = None
                    env_cfg.observations.teacher_static = None
                agent_key = "rsl_rl_cfg_entry_point" if stage == "s0_teacher" else "rsl_rl_student_cfg_entry_point"
                agent_cfg = _load_rsl_runner_cfg(
                    args.task, agent_key, device, latent_dim
                )
                raw_env = gym.make(args.task, cfg=env_cfg)
                env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
                base_env = raw_env.unwrapped
                slope_slots = _assign_fixed_slopes(base_env)

                policy, keepalive = _load_policy(
                    env, checkpoint_path, checkpoint, stage, device, args.task
                )
                teacher: PolicyHandle | None = None
                if is_student and teacher_path is not None:
                    teacher, teacher_keepalive = _load_teacher_policy(
                        env, teacher_path, device, args.task
                    )
                    keepalive.extend(teacher_keepalive)
                del keepalive  # Actors/runners remain reachable through PolicyHandle objects.

                baseline_accumulator, baseline_return = _run_mode(
                    env,
                    base_env,
                    slope_slots,
                    policy,
                    teacher,
                    seeds=list(args.seeds),
                    episodes_per_slope=args.episodes_per_slope,
                    max_steps_per_seed=args.max_policy_steps_per_seed,
                )
                metrics, per_slope = baseline_accumulator.summary()
                stratified = baseline_accumulator.stratified_summary()
                stage_reports[curriculum_name] = {
                    "metrics": metrics,
                    "per_slope": per_slope,
                    "stratified": stratified,
                    "return": baseline_return,
                }
            finally:
                if env is not None:
                    env.close()
                elif raw_env is not None:
                    raw_env.close()
    except BaseException:
        # Process teardown releases Kit resources. Closing SimulationApp here
        # can terminate the interpreter and hide the original exception.
        raise

    if is_student and teacher_checkpoint is None:
        omitted_diagnostics.append("teacher_student_action_kl")
    for name, stage_report in stage_reports.items():
        if stage_report["metrics"]["episodes"]["completed"] != len(SIGNED_SLOPES) * args.episodes_per_slope:
            raise RuntimeError(f"{name} episode quota is incomplete")

    if stage == "s2_student_ppo":
        if s1_baseline_returns is None:
            omitted_diagnostics.append("s1_baseline_return_comparison")
        else:
            comparisons = evaluate_s2_return_floor(stage_reports, s1_baseline_returns)
            assert s1_baseline_binding is not None
            s1_baseline_binding["s2_return_floor"] = comparisons

    report: dict[str, Any] = {
        "schema_version": POLICY_DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": "g1_rickshaw_policy_diagnostics",
        "status": "recorded",
        "created_utc": utc_timestamp(),
        "task": args.task,
        "checkpoint": _checkpoint_binding(checkpoint_path, checkpoint),
        "teacher_checkpoint": (
            None
            if teacher_path is None or teacher_checkpoint is None
            else _checkpoint_binding(teacher_path, teacher_checkpoint)
        ),
        "s1_baseline": s1_baseline_binding,
        "evaluation": {
            "deterministic_actions": True,
            "fixed_seeds": list(args.seeds),
            "signed_slopes": list(SIGNED_SLOPES),
            "episodes_per_slope_per_stage": args.episodes_per_slope,
            "num_envs": args.num_envs,
            "curriculum_stages": list(args.curriculum_stages),
            "command_protocol": FORMAL_EVALUATION_COMMAND_PROTOCOL,
            "cross_case_protocol": FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
            "fat2_weight": fat2_weight,
            "rollout_steps": rollout_steps,
            "latent_dim": latent_dim,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "stages": stage_reports,
        "omitted_diagnostics": omitted_diagnostics,
    }
    write_json_atomic(args.output, report)

    print(f"wrote policy diagnostic report: {Path(args.output).resolve()}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
