"""mjlab manager-based configuration for G1 pulling a rigidly grasped rickshaw."""

from __future__ import annotations

import math

from g1_rickshaw_lab.assets import get_g1_robot_cfg, get_rickshaw_cfg


def g1_rickshaw_env_cfg(*, play: bool = False):
    """Create a fresh mjlab task config following unitree_rl_mjlab conventions."""

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs import mdp as envs_mdp
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig

    from . import mjlab_mdp
    from .closed_chain import add_closed_chain_constraints
    from .mjlab_events import reset_from_mujoco_statics

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale={
                r".*_(hip|knee|ankle)_joint": 0.25,
                r"waist_.*_joint": 0.15,
                r".*_(shoulder|elbow|wrist)_.*": 0.20,
            },
            use_default_offset=True,
        )
    }
    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            heading_command=False,
            debug_vis=play,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(0.0, 0.0),
            ),
        )
    }
    actor_terms = {
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "command": ObservationTermCfg(func=envs_mdp.generated_commands, params={"command_name": "twist"}),
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=not play),
        "critic": ObservationGroupCfg(
            terms={
                **actor_terms,
                "cart_forward_velocity": ObservationTermCfg(func=mjlab_mdp.cart_forward_velocity),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=velocity_mdp.track_linear_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": 0.5},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=velocity_mdp.track_angular_velocity,
            weight=0.5,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "fat2_prior": RewardTermCfg(
            func=mjlab_mdp.fat2_prior,
            weight=0.1,
            params={"sigma": 0.12, "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        ),
        "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-10.0),
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(65.0)}),
    }
    events = {
        "mujoco_static_reset": EventTermCfg(
            func=reset_from_mujoco_statics,
            mode="reset",
            params={"gradient": 0.0},
        )
    }
    if play:
        observations["actor"].enable_corruption = False

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg(), "rickshaw": get_rickshaw_cfg()},
            sensors=(),
            num_envs=1 if play else 4096,
            env_spacing=4.0,
            extent=4.0,
            spec_fn=add_closed_chain_constraints,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=4.0,
            elevation=-12.0,
            azimuth=145.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=2000,
            mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=30, ccd_iterations=50),
        ),
        decimation=4,
        episode_length_s=20.0,
    )


def G1RickshawDirectionalSlopeEnvCfg():
    return g1_rickshaw_env_cfg(play=False)


def G1RickshawDirectionalSlopePlayEnvCfg():
    return g1_rickshaw_env_cfg(play=True)


__all__ = [
    "G1RickshawDirectionalSlopeEnvCfg",
    "G1RickshawDirectionalSlopePlayEnvCfg",
    "g1_rickshaw_env_cfg",
]
