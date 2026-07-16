#!/usr/bin/env python3
"""Run fixed-seed, configured-slope policy acceptance in Isaac Lab."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

from _isaaclab_wrappers import SOURCE_ROOT, add_isaaclab_sources_to_path, require_existing_file

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_rickshaw_lab.policy_evaluation import (  # noqa: E402
    COMMAND_PHASE_LABELS,
    CROSS_CASE_LABELS,
    FORMAL_EVALUATION_COMMAND_PROTOCOL,
    FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
    FORMAL_EVALUATION_NUM_ENVS_MULTIPLE,
    METRIC_DEFINITIONS,
    POLICY_ACCEPTANCE_SCHEMA_VERSION,
    SIGNED_SLOPES,
    PolicyEvaluationAccumulator,
    command_phase_labels,
    d6_wrench_channels,
    evaluation_runtime_sources_sha256,
    evaluate_s2_return_floor,
    evaluate_thresholds,
    load_thresholds,
    serialize_thresholds,
    slope_label,
    validate_s1_baseline_acceptance_report,
    validate_final_acceptance_thresholds,
)
from g1_rickshaw_lab.provenance import (  # noqa: E402
    extract_checkpoint_metadata,
    hash_config_files,
    sha256_file,
)
from g1_rickshaw_lab.slope_contract import (  # noqa: E402
    FORMAL_EVALUATION_NUM_ENVS,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)
from g1_rickshaw_lab.training_contract import (  # noqa: E402
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_LINEAGE_KEY,
    CHECKPOINT_STAGE_KEY,
    TRAINING_CONFIGURATION_KEY,
    load_stage_checkpoint,
    require_pinned_rsl_rl,
    runtime_config_files,
)
from g1_rickshaw_lab.validation import (  # noqa: E402
    asset_hashes,
    utc_timestamp,
    validation_input_assets,
    write_json_atomic,
)


DEFAULT_TASK = "Isaac-G1-Rickshaw-Directional-Slope-v0"
SUPPORTED_STAGES = {
    "s0_teacher",
    "s1_context_candidate",
    "s1_context_distillation",
    "s2_student_ppo",
}
CURRICULUM_NAMES = ("training",)


class PolicyHandle:
    """Uniform distribution interface over native RSL and S1 actors."""

    def __init__(self, actor: Any, *, student: bool) -> None:
        self.actor = actor
        self.student = student

    @property
    def latent_dim(self) -> int:
        encoder = self.actor.context_encoder if hasattr(self.actor, "context_encoder") else self.actor.encoder
        value = getattr(encoder, "latent_dim", None)
        if not isinstance(value, int):
            raise RuntimeError("policy encoder does not expose latent_dim")
        return value

    def distribution(self, observation: Any, intervention: str = "baseline"):
        if intervention not in {"baseline", "zero", "shuffle"}:
            raise ValueError(f"unknown context intervention {intervention!r}")
        if hasattr(self.actor, "context_encoder"):
            context = self.actor.context_encoder.encode(observation["history"])
            context = self.actor.context_projection(context)
            policy = self.actor.actor
        else:
            context = self.actor.encode(observation)
            policy = self.actor.policy
        if intervention != "baseline":
            if not self.student:
                raise RuntimeError("context intervention is defined only for student policies")
            if intervention == "zero":
                context = context * 0.0
            else:
                if context.shape[0] < 2:
                    raise RuntimeError("cross-environment context shuffle requires at least two environments")
                context = context.roll(shifts=1, dims=0)
        return policy.distribution(observation["policy"], context)


def _parser() -> argparse.ArgumentParser:
    add_isaaclab_sources_to_path()
    require_pinned_rsl_rl()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument(
        "--s1-baseline-report",
        default=None,
        help=(
            "S1 fixed-seed TRAINING acceptance report. Optional for diagnostics, but required "
            "for an S2 report to pass."
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--num-envs", type=int, default=FORMAL_EVALUATION_NUM_ENVS)
    parser.add_argument("--episodes-per-slope", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--curriculum-stages", nargs="+", choices=CURRICULUM_NAMES, default=("training",))
    parser.add_argument("--max-policy-steps-per-seed", type=int, default=6000)
    parser.add_argument("--thresholds", default=None, help="Explicit schema-v1 YAML; no built-in thresholds exist.")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Additional explicit scalar threshold, e.g. stages.training.metrics.episodes.fall_rate<=0.01.",
    )
    parser.add_argument(
        "--no-context-interventions",
        action="store_true",
        help="Debug only; marks student acceptance incomplete.",
    )
    parser.add_argument(
        "--allow-missing-teacher",
        action="store_true",
        help="Debug only; permits a student report without teacher-student KL and marks it incomplete.",
    )
    parser.add_argument("--fat2-weight", type=float, choices=(0.0, 0.1), default=None)
    parser.add_argument("--rollout-steps", type=int, choices=(24, 48, 64), default=None)
    parser.add_argument("--latent-dim", type=int, choices=(8, 16, 24), default=None)
    parser.add_argument("--ablation-id", default=None)
    parser.add_argument("--ablation-group", choices=("fat2_weight", "rollout_steps", "latent_dim"), default=None)
    parser.add_argument("--ablation-matrix-sha256", default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args: argparse.Namespace, stage: str) -> None:
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"unsupported checkpoint stage {stage!r}")
    if (
        args.num_envs <= 0
        or args.num_envs % FORMAL_EVALUATION_NUM_ENVS_MULTIPLE != 0
    ):
        raise ValueError(
            f"--num-envs must be a positive multiple of "
            f"{FORMAL_EVALUATION_NUM_ENVS_MULTIPLE} so every "
            "slope has an equal number of environments"
        )
    if not args.seeds:
        raise ValueError("fixed seeds must be non-empty")
    quota_divisor = len(args.seeds) * len(CROSS_CASE_LABELS)
    if (
        args.episodes_per_slope < 100
        or args.episodes_per_slope % quota_divisor != 0
        or args.max_policy_steps_per_seed <= 0
    ):
        raise ValueError(
            "acceptance requires at least 100 episodes per slope, an episode quota "
            f"divisible by seeds={quota_divisor}, and a positive step limit"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("fixed seeds must be unique")
    if len(set(args.curriculum_stages)) != len(args.curriculum_stages):
        raise ValueError("curriculum stages must be unique")
    is_student = stage != "s0_teacher"
    if is_student and args.teacher_checkpoint is None and not args.allow_missing_teacher:
        raise ValueError("student acceptance requires --teacher-checkpoint for action KL")
    if args.s1_baseline_report is not None and stage != "s2_student_ppo":
        raise ValueError("--s1-baseline-report applies only to S2 acceptance")
    if (
        stage == "s2_student_ppo"
        and args.s1_baseline_report is not None
        and list(args.curriculum_stages) != ["training"]
    ):
        raise ValueError("S2 return-floor acceptance requires TRAINING evaluation")
    if (args.ablation_id is None) != (args.ablation_group is None):
        raise ValueError("--ablation-id and --ablation-group must be supplied together")
    if args.ablation_group == "fat2_weight" and args.fat2_weight is None:
        raise ValueError("FAT2 ablation requires --fat2-weight")
    if args.ablation_group == "rollout_steps" and args.rollout_steps is None:
        raise ValueError("rollout ablation requires --rollout-steps")
    if args.ablation_group == "latent_dim" and args.latent_dim is None:
        raise ValueError("latent ablation requires --latent-dim")


def _configure_fixed_stage(env_cfg: Any, stage_name: str, fat2_weight: float | None) -> None:
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp

    if stage_name != "training":
        raise ValueError(f"unknown evaluation stage {stage_name!r}")
    env_cfg.curriculum = None
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.runtime_randomization = replace(
        env_cfg.runtime_randomization,
        curriculum=mdp.CurriculumScheduleCfg(),
    )
    env_cfg.events.sample_physics.params = {"cfg": env_cfg.runtime_randomization}
    env_cfg.events.initialize_curriculum.params = {"cfg": env_cfg.runtime_randomization}
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

    cfg = base_env.runtime_randomization.curriculum if hasattr(base_env, "runtime_randomization") else base_env.cfg.runtime_randomization.curriculum
    base_env.curriculum_runtime_state = mdp.CurriculumRuntimeState.create(
        columns, torch.sign(expected).to(dtype=torch.long), cfg
    )
    base_env.curriculum_stage_per_env[:] = base_env.curriculum_runtime_state.stage_per_environment()
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
    import torch

    keepalive: list[Any] = []
    if stage in {"s1_context_candidate", "s1_context_distillation"}:
        from g1_rickshaw_lab.rl import G1RickshawStudentActor

        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("S1 checkpoint has no model_state_dict")
        latent_weight = state.get("context_encoder.context.2.weight")
        if not torch.is_tensor(latent_weight) or latent_weight.ndim != 2:
            raise ValueError("S1 checkpoint does not expose the context latent dimension")
        model = G1RickshawStudentActor(latent_dim=int(latent_weight.shape[0])).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        keepalive.append(model)
        return PolicyHandle(model, student=True), keepalive

    from rsl_rl.runners import OnPolicyRunner
    registry_key = "rsl_rl_cfg_entry_point" if stage == "s0_teacher" else "rsl_rl_student_cfg_entry_point"
    agent_cfg = _load_rsl_runner_cfg(task, registry_key, device)
    if stage == "s2_student_ppo":
        training_configuration = checkpoint[TRAINING_CONFIGURATION_KEY]
        latent_dim = int(training_configuration["ablation_values"]["latent_dim"])
        agent_cfg.actor.latent_dim = latent_dim
        agent_cfg.critic.latent_dim = latent_dim
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(
        os.fspath(checkpoint_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        strict=True,
    )
    runner.alg.actor.eval()
    keepalive.append(runner)
    return PolicyHandle(runner.alg.actor, student=stage != "s0_teacher"), keepalive


def _load_teacher_policy(
    env: Any, checkpoint_path: Path, device: str, task: str
) -> tuple[PolicyHandle, list[Any]]:
    from rsl_rl.runners import OnPolicyRunner
    agent_cfg = _load_rsl_runner_cfg(task, "rsl_rl_cfg_entry_point", device)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(
        os.fspath(checkpoint_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        strict=True,
    )
    runner.alg.actor.eval()
    return PolicyHandle(runner.alg.actor, student=False), [runner]


def _load_rsl_runner_cfg(task: str, registry_key: str, device: str):
    """Load and migrate a runner config exactly as IsaacLab's train entry point does."""

    import importlib.metadata

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(task, registry_key)
    agent_cfg.device = device
    return handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )


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

    wrench = getattr(base_env, "d6_incoming_joint_proxy_w", None)
    if wrench is None:
        raise RuntimeError("acceptance requires retained-link incoming D6 reaction wrenches")
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
    mode: str,
    seeds: list[int],
    episodes_per_slope: int,
    max_steps_per_seed: int,
    collect_metrics: bool,
) -> tuple[PolicyEvaluationAccumulator | None, dict[str, Any]]:
    import torch

    if not seeds:
        raise ValueError("fixed evaluation seeds must be non-empty")
    quota_divisor = len(seeds) * len(CROSS_CASE_LABELS)
    if episodes_per_slope % quota_divisor != 0:
        raise ValueError(
            "episodes_per_slope must be divisible by the number of seeds"
        )
    accumulator = PolicyEvaluationAccumulator() if collect_metrics else None
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
                distribution = policy.distribution(observation, intervention=mode)
                teacher_kl = None
                if teacher is not None and mode == "baseline":
                    teacher_distribution = teacher.distribution(observation)
                    teacher_kl = torch.distributions.kl_divergence(teacher_distribution, distribution)
                if collect_metrics and torch.any(active):
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
                    assert accumulator is not None
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
                cause_state = getattr(base_env, "termination_cause_state", None)
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
                    causes: list[str] = []
                    if cause_state is not None:
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
                    if accumulator is not None:
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
        "sha256": sha256_file(path),
        "stage": checkpoint[CHECKPOINT_STAGE_KEY],
        "curriculum_iteration": checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY),
        "lineage": checkpoint.get(CHECKPOINT_LINEAGE_KEY, {}),
        "provenance": extract_checkpoint_metadata(checkpoint).to_mapping(),
    }


def _input_binding(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        "runtime_config_sha256": dict(hash_config_files(runtime_config_files())),
        "assets_sha256": asset_hashes(validation_input_assets()),
        "evaluation_runtime_sources_sha256": evaluation_runtime_sources_sha256(),
    }
    if args.thresholds is not None:
        result["thresholds_sha256"] = sha256_file(args.thresholds)
    if args.ablation_matrix_sha256 is not None:
        digest = args.ablation_matrix_sha256.lower()
        if len(digest) != 64:
            raise ValueError("--ablation-matrix-sha256 must be a SHA-256 digest")
        int(digest, 16)
        result["ablation_matrix_sha256"] = digest
    return result


def _context_report(
    baseline: dict[str, Any], zero: dict[str, Any] | None, shuffle: dict[str, Any] | None
) -> dict[str, Any]:
    result: dict[str, Any] = {"baseline_return": baseline}
    for name, intervention in (("zero", zero), ("shuffle", shuffle)):
        if intervention is None:
            result[f"{name}_return"] = None
            result[f"{name}_return_drop"] = None
            continue
        drop = baseline["mean"] - intervention["mean"]
        denominator = max(abs(baseline["mean"]), 1.0e-12)
        result[f"{name}_return"] = intervention
        result[f"{name}_return_drop"] = {
            "absolute": drop,
            "fraction_of_abs_baseline": drop / denominator,
        }
    return result


def main() -> int:  # noqa: C901
    parser = _parser()
    args = parser.parse_args()
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required")
    checkpoint_path = require_existing_file(args.checkpoint, "policy checkpoint").resolve()
    checkpoint = load_stage_checkpoint(checkpoint_path, validate_runtime=True)
    stage = checkpoint[CHECKPOINT_STAGE_KEY]
    training_configuration = dict(checkpoint[TRAINING_CONFIGURATION_KEY])
    if training_configuration["task"] != args.task:
        raise ValueError("policy evaluation task differs from checkpoint training task")
    configured_variants = training_configuration["ablation_values"]
    for argument, configured in (
        ("fat2_weight", float(configured_variants["fat2_weight"])),
        ("rollout_steps", int(configured_variants["rollout_steps"])),
        ("latent_dim", int(configured_variants["latent_dim"])),
    ):
        requested = getattr(args, argument)
        if requested is not None and requested != configured:
            raise ValueError(
                f"--{argument.replace('_', '-')}={requested!r} differs from "
                f"checkpoint training value {configured!r}"
            )
        setattr(args, argument, configured)
    _validate_args(args, stage)
    is_student = stage != "s0_teacher"

    s1_baseline_path: Path | None = None
    s1_baseline_returns: dict[str, float] | None = None
    s1_baseline_binding: dict[str, Any] | None = None
    if args.s1_baseline_report is not None:
        s1_baseline_path = require_existing_file(
            args.s1_baseline_report, "S1 baseline acceptance report"
        ).resolve()
        lineage = checkpoint.get(CHECKPOINT_LINEAGE_KEY)
        context_digest = (
            lineage.get("context_checkpoint_sha256") if isinstance(lineage, Mapping) else None
        )
        if not isinstance(context_digest, str):
            raise RuntimeError("S2 checkpoint lineage has no S1 context checkpoint SHA256")
        s1_report = json.loads(s1_baseline_path.read_text(encoding="utf-8"))
        s1_baseline_returns = validate_s1_baseline_acceptance_report(
            s1_report,
            expected_checkpoint_sha256=context_digest,
            fixed_seeds=args.seeds,
            episodes_per_slope=args.episodes_per_slope,
        )
        s1_baseline_binding = {
            "path": os.fspath(s1_baseline_path),
            "sha256": sha256_file(s1_baseline_path),
            "checkpoint_sha256": context_digest,
            "baseline_return_mean": dict(s1_baseline_returns),
        }

    teacher_path: Path | None = None
    teacher_checkpoint: Any | None = None
    if args.teacher_checkpoint is not None:
        teacher_path = require_existing_file(args.teacher_checkpoint, "teacher checkpoint").resolve()
        teacher_checkpoint = load_stage_checkpoint(
            teacher_path, expected_stage="s0_teacher", validate_runtime=True
        )
        if (
            extract_checkpoint_metadata(teacher_checkpoint).to_mapping()
            != extract_checkpoint_metadata(checkpoint).to_mapping()
        ):
            raise RuntimeError("teacher and evaluated policy provenance differ")
        if is_student:
            policy_lineage = checkpoint.get(CHECKPOINT_LINEAGE_KEY)
            expected_teacher_digest = (
                policy_lineage.get("teacher_checkpoint_sha256")
                if isinstance(policy_lineage, Mapping)
                else None
            )
            if expected_teacher_digest != sha256_file(teacher_path):
                raise RuntimeError(
                    "teacher checkpoint SHA256 differs from the evaluated policy lineage"
                )

    thresholds = load_thresholds(args.thresholds, args.threshold)
    if is_student and thresholds:
        validate_final_acceptance_thresholds(
            thresholds,
            curriculum_stages=args.curriculum_stages,
        )
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    stage_reports: dict[str, Any] = {}
    incomplete: list[str] = []
    try:
        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity  # noqa: F401

        device = args.device or "cuda:0"
        for curriculum_name in args.curriculum_stages:
            raw_env = None
            env = None
            try:
                env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
                env_cfg.seed = args.seeds[0]
                _configure_fixed_stage(env_cfg, curriculum_name, args.fat2_weight)
                if not is_student:
                    env_cfg.observations.history = None
                agent_key = "rsl_rl_cfg_entry_point" if stage == "s0_teacher" else "rsl_rl_student_cfg_entry_point"
                agent_cfg = _load_rsl_runner_cfg(args.task, agent_key, device)
                raw_env = gym.make(args.task, cfg=env_cfg)
                env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
                base_env = raw_env.unwrapped
                slope_slots = _assign_fixed_slopes(base_env)

                policy, keepalive = _load_policy(
                    env, checkpoint_path, checkpoint, stage, device, args.task
                )
                if args.latent_dim is not None and policy.latent_dim != args.latent_dim:
                    raise RuntimeError(
                        f"checkpoint latent dimension {policy.latent_dim} != requested ablation {args.latent_dim}"
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
                    mode="baseline",
                    seeds=list(args.seeds),
                    episodes_per_slope=args.episodes_per_slope,
                    max_steps_per_seed=args.max_policy_steps_per_seed,
                    collect_metrics=True,
                )
                assert baseline_accumulator is not None
                metrics, per_slope = baseline_accumulator.summary()
                stratified = baseline_accumulator.stratified_summary()
                zero_return = None
                shuffle_return = None
                if is_student and not args.no_context_interventions:
                    _, zero_return = _run_mode(
                        env,
                        base_env,
                        slope_slots,
                        policy,
                        None,
                        mode="zero",
                        seeds=list(args.seeds),
                        episodes_per_slope=args.episodes_per_slope,
                        max_steps_per_seed=args.max_policy_steps_per_seed,
                        collect_metrics=False,
                    )
                    _, shuffle_return = _run_mode(
                        env,
                        base_env,
                        slope_slots,
                        policy,
                        None,
                        mode="shuffle",
                        seeds=list(args.seeds),
                        episodes_per_slope=args.episodes_per_slope,
                        max_steps_per_seed=args.max_policy_steps_per_seed,
                        collect_metrics=False,
                    )
                stage_reports[curriculum_name] = {
                    "metrics": metrics,
                    "per_slope": per_slope,
                    "stratified": stratified,
                    "context_interventions": _context_report(
                        baseline_return, zero_return, shuffle_return
                    ),
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
        incomplete.append("teacher_student_action_kl_missing")
    if is_student and args.no_context_interventions:
        incomplete.append("student_context_zero_shuffle_skipped")
    for name, stage_report in stage_reports.items():
        if stage_report["metrics"]["episodes"]["completed"] != len(SIGNED_SLOPES) * args.episodes_per_slope:
            incomplete.append(f"{name}_episode_quota_incomplete")

    return_floor_failures: list[str] = []
    if stage == "s2_student_ppo":
        if s1_baseline_returns is None:
            incomplete.append("s1_baseline_acceptance_report_missing")
        else:
            comparisons, return_floor_failures = evaluate_s2_return_floor(
                stage_reports,
                s1_baseline_returns,
            )
            assert s1_baseline_binding is not None
            s1_baseline_binding["s2_return_floor"] = comparisons

    report: dict[str, Any] = {
        "schema_version": POLICY_ACCEPTANCE_SCHEMA_VERSION,
        "report_type": "g1_rickshaw_policy_acceptance",
        "status": (
            "failed"
            if return_floor_failures
            else "incomplete" if incomplete else "recorded"
        ),
        "created_utc": utc_timestamp(),
        "task": args.task,
        "checkpoint": _checkpoint_binding(checkpoint_path, checkpoint),
        "teacher_checkpoint": (
            None
            if teacher_path is None or teacher_checkpoint is None
            else _checkpoint_binding(teacher_path, teacher_checkpoint)
        ),
        "s1_baseline_acceptance": s1_baseline_binding,
        "inputs": _input_binding(args),
        "evaluation": {
            "deterministic_actions": True,
            "fixed_seeds": list(args.seeds),
            "signed_slopes": list(SIGNED_SLOPES),
            "episodes_per_slope_per_stage": args.episodes_per_slope,
            "num_envs": args.num_envs,
            "curriculum_stages": list(args.curriculum_stages),
            "command_protocol": FORMAL_EVALUATION_COMMAND_PROTOCOL,
            "cross_case_protocol": FORMAL_EVALUATION_CROSS_CASE_PROTOCOL,
            "fat2_weight_override": args.fat2_weight,
            "rollout_steps_training_variant": args.rollout_steps,
            "latent_dim_variant": args.latent_dim,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "stages": stage_reports,
        "thresholds": serialize_thresholds(thresholds),
        "threshold_results": {},
        "failures": [*incomplete, *return_floor_failures],
        "ablation": (
            None
            if args.ablation_id is None
            else {
                "id": args.ablation_id,
                "group": args.ablation_group,
                "matrix_sha256": args.ablation_matrix_sha256,
            }
        ),
    }
    if thresholds:
        outcomes, failures = evaluate_thresholds(report, thresholds)
        report["threshold_results"] = outcomes
        report["failures"].extend(failures)
        report["status"] = "passed" if not report["failures"] else "failed"
    write_json_atomic(args.output, report)

    print(f"wrote policy acceptance report: {Path(args.output).resolve()}")
    exit_code = 0 if report["status"] in {"recorded", "passed"} else 1
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
