"""Mjlab configuration for G1 towing the site-connected rickshaw."""

from __future__ import annotations

import math

from g1_rickshaw_lab.assets import get_g1_robot_cfg, get_rickshaw_cfg
from g1_rickshaw_lab.configuration import load_feasibility_envelope
from g1_rickshaw_lab.policy_schema import HISTORY_LENGTH
from g1_rickshaw_lab.project_paths import CONFIG_ROOT


def _runtime_cfg(*, play: bool, history_length: int):
    from .mdp.dynamics import AnalyticForceCfg, FAT2Cfg, SupportPolygonCfg, ZMPCfg
    from .mdp.events import DomainRandomizationCfg, SpeedCommandSamplingCfg
    from .mjlab_events import MjlabTaskRuntimeCfg
    from .task_spec import RickshawPoseTargetCfg

    envelope = load_feasibility_envelope(CONFIG_ROOT / "feasibility_envelope.yaml")
    calibration = dict(envelope.calibration)
    names = (
        "torso.mass_delta",
        "payload.mass",
        "payload.com.x",
        "payload.com.y",
        "payload.com.z",
        "rolling_resistance.c_rr",
        "terrain.friction",
        "wheel.left_damping",
        "wheel.right_damping",
    )
    ranges = {name: (envelope.ranges[name].minimum, envelope.ranges[name].maximum) for name in names}
    nominal = {
        "torso.mass_delta": 0.0,
        "payload.mass": 0.0,
        "payload.com.x": 0.5 * sum(ranges["payload.com.x"]),
        "payload.com.y": 0.0,
        "payload.com.z": 0.5 * sum(ranges["payload.com.z"]),
        "rolling_resistance.c_rr": calibration["rolling_resistance.c_rr_nominal"],
        "terrain.friction": calibration["terrain.friction_nominal"],
        "wheel.left_damping": 0.02,
        "wheel.right_damping": 0.02,
    }
    domain = DomainRandomizationCfg(
        enabled=not play,
        ranges=ranges,
        nominal=nominal,
        calibration=calibration,
    )
    return MjlabTaskRuntimeCfg(
        domain=domain,
        command=SpeedCommandSamplingCfg(maximum=1.0 if play else 0.1),
        speed_acceleration_limit=envelope.ranges["command.acceleration_limit"].maximum,
        speed_jerk_limit=envelope.ranges["command.jerk_limit"].maximum,
        rickshaw_pose=RickshawPoseTargetCfg(
            hitch_height_target=calibration["rickshaw_pose.hitch_height_target"],
            hitch_height_tolerance=calibration["rickshaw_pose.hitch_height_tolerance"],
            hitch_vertical_speed_tolerance=calibration["rickshaw_pose.hitch_vertical_speed_tolerance"],
        ),
        analytic_force=AnalyticForceCfg(minimum_wheel_normal_force=calibration["safety.minimum_wheel_normal_force"]),
        fat2=FAT2Cfg(
            robot_mass=calibration["fat.robot_mass"],
            com_radius=calibration["fat.com_radius"],
            com_radius_bounds=tuple(calibration["fat.com_radius_bounds"]),
            theta_max=calibration["safety.theta_max"],
            wrench_consistency_relative_tolerance=calibration["fat.wrench_consistency_relative_tolerance"],
            wrench_consistency_absolute_floor_n=calibration["fat.wrench_consistency_absolute_floor_n"],
            wrench_consistency_window_steps=calibration["fat.wrench_consistency_window_steps"],
        ),
        support=SupportPolygonCfg(
            foot_half_length=calibration["support.foot_half_length"],
            foot_half_width=calibration["support.foot_half_width"],
            foot_center_offset_x=calibration["support.foot_center_offset_x"],
        ),
        zmp=ZMPCfg(min_ground_reaction=calibration["safety.min_ground_reaction"]),
        history_length=history_length,
        shuffle_slopes=not play,
        play=play,
    )


def g1_rickshaw_env_cfg(*, play: bool = False, history_length: int = HISTORY_LENGTH):
    """Create the full directional-slope task using mjlab 1.2 APIs."""

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig

    from . import mjlab_mdp
    from .closed_chain import add_closed_chain_constraints
    from .mdp.rewards import REWARD_WEIGHTS
    from .mjlab_actions import StaticReferenceJointPositionActionCfg
    from .mjlab_events import (
        advance_mjlab_policy_state,
        initialize_mjlab_domain,
        initialize_mjlab_task,
        reset_from_mujoco_statics,
    )
    from .terrain_cfg import make_mjlab_directional_slopes_cfg

    runtime = _runtime_cfg(play=play, history_length=history_length)
    observations = {
        "policy": ObservationGroupCfg(
            terms={"current": ObservationTermCfg(func=mjlab_mdp.current_actor_observation)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "history": ObservationGroupCfg(
            terms={
                "history": ObservationTermCfg(
                    func=mjlab_mdp.actor_observation_history,
                    params={"history_length": history_length},
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "teacher_dynamic_history": ObservationGroupCfg(
            terms={
                "history": ObservationTermCfg(
                    func=mjlab_mdp.teacher_dynamic_history,
                    params={"history_length": history_length},
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "teacher_static": ObservationGroupCfg(
            terms={"static": ObservationTermCfg(func=mjlab_mdp.teacher_static)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={"privileged": ObservationTermCfg(func=mjlab_mdp.critic_privileged_state)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    actions = {
        "joint_pos": StaticReferenceJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            preserve_order=True,
        )
    }
    events = {
        "initialize_task": EventTermCfg(func=initialize_mjlab_task, mode="startup", params={"cfg": runtime}),
        "initialize_domain": EventTermCfg(func=initialize_mjlab_domain, mode="startup", params={"cfg": runtime}),
        "mujoco_static_reset": EventTermCfg(func=reset_from_mujoco_statics, mode="reset"),
        "policy_state": EventTermCfg(func=advance_mjlab_policy_state, mode="step", params={"cfg": runtime}),
    }
    rewards = {
        "track_speed_exp": RewardTermCfg(func=mjlab_mdp.track_speed_exp, weight=REWARD_WEIGHTS["track_speed_exp"]),
        "lateral_error_l2": RewardTermCfg(func=mjlab_mdp.lateral_error_l2, weight=REWARD_WEIGHTS["lateral_error_l2"]),
        "heading_error_l2": RewardTermCfg(func=mjlab_mdp.heading_error_l2, weight=REWARD_WEIGHTS["heading_error_l2"]),
        "zmp_margin_barrier": RewardTermCfg(
            func=mjlab_mdp.zmp_margin_barrier, weight=REWARD_WEIGHTS["zmp_margin_barrier"]
        ),
        "hitch_height_exp": RewardTermCfg(func=mjlab_mdp.hitch_height_exp, weight=REWARD_WEIGHTS["hitch_height_exp"]),
        "hitch_height_recovery_l2": RewardTermCfg(
            func=mjlab_mdp.hitch_height_recovery_l2,
            weight=REWARD_WEIGHTS["hitch_height_recovery_l2"],
        ),
        "fat2_prior_exp": RewardTermCfg(func=mjlab_mdp.fat2_prior_exp, weight=REWARD_WEIGHTS["fat2_prior_exp"]),
        "feet_gait": RewardTermCfg(func=mjlab_mdp.feet_gait, weight=REWARD_WEIGHTS["feet_gait"]),
        "feet_swing_height": RewardTermCfg(
            func=mjlab_mdp.feet_swing_height, weight=REWARD_WEIGHTS["feet_swing_height"]
        ),
        "feet_slide": RewardTermCfg(func=mjlab_mdp.feet_slide, weight=REWARD_WEIGHTS["feet_slide"]),
        "terrain_normal_velocity_l2": RewardTermCfg(
            func=mjlab_mdp.terrain_normal_velocity_l2,
            weight=REWARD_WEIGHTS["terrain_normal_velocity_l2"],
        ),
        "joint_power_l1": RewardTermCfg(func=mjlab_mdp.joint_power_l1, weight=REWARD_WEIGHTS["joint_power_l1"]),
        "joint_acc_l2": RewardTermCfg(
            func=mjlab_mdp.joint_acc_l2,
            weight=REWARD_WEIGHTS["joint_acc_l2"],
        ),
        "action_rate_l2": RewardTermCfg(
            func=mjlab_mdp.action_rate_l2,
            weight=REWARD_WEIGHTS["action_rate_l2"],
        ),
        "hip_yaw_roll_reference_l2": RewardTermCfg(
            func=mjlab_mdp.hip_yaw_roll_reference_l2,
            weight=REWARD_WEIGHTS["hip_yaw_roll_reference_l2"],
        ),
        "pelvis_height_limits_l2": RewardTermCfg(
            func=mjlab_mdp.pelvis_height_limits_l2,
            weight=REWARD_WEIGHTS["pelvis_height_limits_l2"],
        ),
        "joint_position_limits": RewardTermCfg(
            func=mjlab_mdp.joint_position_limits,
            weight=REWARD_WEIGHTS["joint_position_limits"],
        ),
        "termination": RewardTermCfg(func=mjlab_mdp.termination, weight=REWARD_WEIGHTS["termination"]),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }
    curriculum = {
        "speed_command_levels": CurriculumTermCfg(
            func=mjlab_mdp.speed_command_levels,
            params={"reward_term_name": "track_speed_exp"},
        )
    }
    feet = ContactSensorCfg(
        name="robot_contacts",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    wheels = ContactSensorCfg(
        name="wheel_contacts",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(left_wheel_link|right_wheel_link)$",
            entity="rickshaw",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=10,
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=make_mjlab_directional_slopes_cfg(),
                max_init_terrain_level=0,
            ),
            entities={"robot": get_g1_robot_cfg(), "rickshaw": get_rickshaw_cfg()},
            sensors=(feet, wheels),
            num_envs=19 if play else 8192,
            env_spacing=6.0,
            spec_fn=add_closed_chain_constraints,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={} if play else curriculum,
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=4.0,
            elevation=-5.0,
            # The towing path is the local +x axis, so +y gives a true side view.
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=2400,
            contact_sensor_maxmatch=256,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=100,
                ls_iterations=50,
                ccd_iterations=50,
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
    )
    cfg.history_length = history_length
    cfg.observation_noise_enabled = not play
    cfg.domain_randomization = runtime.domain
    cfg.policy_update = runtime
    return cfg


def G1RickshawDirectionalSlopeEnvCfg():
    return g1_rickshaw_env_cfg(play=False)


def G1RickshawDirectionalSlopePlayEnvCfg():
    return g1_rickshaw_env_cfg(play=True)


__all__ = [
    "G1RickshawDirectionalSlopeEnvCfg",
    "G1RickshawDirectionalSlopePlayEnvCfg",
    "g1_rickshaw_env_cfg",
]
