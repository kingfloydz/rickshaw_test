"""Manager-based Isaac Lab environment for G1 rickshaw velocity tracking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from g1_rickshaw_lab.assets.g1_dex1 import build_g1_rickshaw_cfg
from g1_rickshaw_lab.assets.rickshaw import (
    HITCH_LINK_NAMES,
    RICKSHAW_URDF_SPEC,
    WHEEL_LINK_NAMES,
    build_rickshaw_cfg,
)
from g1_rickshaw_lab.configuration import (
    G1_JOINT_ORDER,
    load_feasibility_envelope,
    load_reset_pose_library,
)

from . import mdp
from .closed_chain import ReplicatedDualD6SpawnerCfg
from .terrain_cfg import DIRECTIONAL_SLOPES_CFG


REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_FEASIBILITY_PATH = REPOSITORY_ROOT / "config" / "feasibility_envelope.yaml"
DEFAULT_RESET_POSES_PATH = REPOSITORY_ROOT / "config" / "reset_poses.yaml"
LOWER_JOINT_NAMES = G1_JOINT_ORDER[:12]
WAIST_JOINT_NAMES = G1_JOINT_ORDER[12:15]
ARM_JOINT_NAMES = G1_JOINT_ORDER[15:29]
DEX_JOINT_NAMES = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
PELVIS_BODY_NAME = "pelvis"
TORSO_BODY_NAME = "torso_link"
ILLEGAL_CONTACT_BODY_NAMES = (
    "pelvis",
    "imu_in_pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "d435_link",
    "head_link",
    "imu_in_torso",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_dex1_base_link",
    "left_dex1_finger_link_1",
    "left_dex1_finger_link_2",
    "logo_link",
    "mid360_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_dex1_base_link",
    "right_dex1_finger_link_1",
    "right_dex1_finger_link_2",
)
TEACHER_STATIC_DIM = mdp.TEACHER_STATIC_DIM
TEACHER_DYNAMIC_DIM = mdp.TEACHER_DYNAMIC_DIM
CRITIC_PRIVILEGED_DIM = mdp.CRITIC_PRIVILEGED_DIM


def _configured_path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, os.fspath(default)))


def _range_pairs(envelope) -> dict[str, tuple[float, float]]:
    return {
        name: (interval.minimum, interval.maximum)
        for name, interval in envelope.ranges.items()
    }


def _range_upper(envelope, name: str) -> float:
    return float(envelope.ranges[name].maximum)


def _cal(calibration: dict[str, Any], name: str) -> Any:
    return calibration[name]


@configclass
class G1RickshawSceneCfg(InteractiveSceneCfg):
    """Terrain, G1+Dex, rickshaw, contact sensors, and lighting."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=DIRECTIONAL_SLOPES_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = build_g1_rickshaw_cfg(require_usd=True)
    rickshaw: ArticulationCfg = build_rickshaw_cfg(require_usd=True)
    closed_chain: AssetBaseCfg | None = None

    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        track_air_time=True,
    )
    wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Rickshaw/.*_wheel_link",
        history_length=1,
        track_air_time=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class ActionsCfg:
    """Three fixed action groups totaling exactly 29 policy joints."""

    lower = mdp.FilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LOWER_JOINT_NAMES),
        preserve_order=True,
        scale=0.40,
        use_default_offset=False,
        reference_indices=tuple(range(0, 12)),
        physics_hook_owner=True,
    )
    waist = mdp.FilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(WAIST_JOINT_NAMES),
        preserve_order=True,
        scale=0.20,
        use_default_offset=False,
        reference_indices=tuple(range(12, 15)),
    )
    upper = mdp.FilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        preserve_order=True,
        scale={
            ".*_shoulder_.*": 0.25,
            ".*_elbow_joint": 0.30,
            ".*_wrist_.*": 0.15,
        },
        use_default_offset=False,
        reference_indices=tuple(range(15, 29)),
    )


@configclass
class ObservationsCfg:
    """Actor, history, teacher, and critic observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        current = ObsTerm(func=mdp.current_actor_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        history = ObsTerm(func=mdp.actor_observation_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherDynamicHistoryCfg(ObsGroup):
        history = ObsTerm(
            func=mdp.teacher_dynamic_history,
            params={"expected_dim": TEACHER_DYNAMIC_DIM},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherStaticCfg(ObsGroup):
        static = ObsTerm(
            func=mdp.teacher_static,
            params={"expected_dim": TEACHER_STATIC_DIM},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        privileged = ObsTerm(
            func=mdp.critic_privileged_state,
            params={"expected_dim": CRITIC_PRIVILEGED_DIM},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    history: HistoryCfg | None = HistoryCfg()
    teacher_dynamic_history: TeacherDynamicHistoryCfg = TeacherDynamicHistoryCfg()
    teacher_static: TeacherStaticCfg = TeacherStaticCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Startup, reset, and global policy-rate events."""

    initialize_mdp = EventTerm(func=mdp.initialize_mdp_state, mode="startup", params={})
    initialize_domain = EventTerm(
        func=mdp.initialize_domain_randomization, mode="startup", params={}
    )
    reset_closed_chain = EventTerm(func=mdp.reset_closed_chain, mode="reset", params={})
    policy_interval = EventTerm(
        func=mdp.advance_policy_interval,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={},
    )


@configclass
class RewardsCfg:
    """Reward terms from guide section 11.1."""

    track_speed_exp = RewTerm(func=mdp.track_speed_exp, weight=mdp.REWARD_WEIGHTS["track_speed_exp"])
    lateral_error_l2 = RewTerm(func=mdp.lateral_error_l2, weight=mdp.REWARD_WEIGHTS["lateral_error_l2"])
    heading_error_l2 = RewTerm(func=mdp.heading_error_l2, weight=mdp.REWARD_WEIGHTS["heading_error_l2"])
    zmp_margin_barrier = RewTerm(
        func=mdp.zmp_margin_barrier,
        weight=mdp.REWARD_WEIGHTS["zmp_margin_barrier"],
    )
    hitch_height_exp = RewTerm(func=mdp.hitch_height_exp, weight=mdp.REWARD_WEIGHTS["hitch_height_exp"])
    hitch_height_recovery_l2 = RewTerm(
        func=mdp.hitch_height_recovery_l2,
        weight=mdp.REWARD_WEIGHTS["hitch_height_recovery_l2"],
        params={
            "deadband": mdp.HITCH_HEIGHT_RECOVERY_DEADBAND_M,
            "scale": mdp.HITCH_HEIGHT_RECOVERY_SCALE_M,
        },
    )
    fat2_prior_exp = RewTerm(func=mdp.fat2_prior_exp, weight=mdp.REWARD_WEIGHTS["fat2_prior_exp"])
    feet_single_stance = RewTerm(
        func=mdp.feet_single_stance,
        weight=mdp.REWARD_WEIGHTS["feet_single_stance"],
        params={
            "sensor_cfg": SceneEntityCfg(
                "robot_contacts", body_names=list(FOOT_BODY_NAMES), preserve_order=True
            ),
            "cap": mdp.FEET_SINGLE_STANCE_CAP_S,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=mdp.REWARD_WEIGHTS["feet_slide"],
        params={
            "sensor_cfg": SceneEntityCfg("robot_contacts", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
        },
    )
    terrain_normal_velocity_l2 = RewTerm(
        func=mdp.terrain_normal_velocity_l2,
        weight=mdp.REWARD_WEIGHTS["terrain_normal_velocity_l2"],
    )
    joint_power_l1 = RewTerm(func=mdp.joint_power_l1, weight=mdp.REWARD_WEIGHTS["joint_power_l1"])
    processed_action_rate_l2 = RewTerm(
        func=mdp.processed_action_rate_l2,
        weight=mdp.REWARD_WEIGHTS["processed_action_rate_l2"],
    )
    processed_action_jerk_l2 = RewTerm(
        func=mdp.processed_action_jerk_l2,
        weight=mdp.REWARD_WEIGHTS["processed_action_jerk_l2"],
    )
    hip_yaw_roll_reference_l2 = RewTerm(
        func=mdp.hip_yaw_roll_reference_l2,
        weight=mdp.REWARD_WEIGHTS["hip_yaw_roll_reference_l2"],
        params={
            "policy_indices": mdp.HIP_YAW_ROLL_POLICY_INDICES,
            "scale": mdp.HIP_YAW_ROLL_REFERENCE_SCALE_RAD,
        },
    )
    pelvis_height_limits_l2 = RewTerm(
        func=mdp.pelvis_height_limits_l2,
        weight=mdp.REWARD_WEIGHTS["pelvis_height_limits_l2"],
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[PELVIS_BODY_NAME], preserve_order=True
            ),
            "bounds": mdp.PELVIS_HEIGHT_BOUNDS_M,
            "scale": mdp.PELVIS_HEIGHT_ERROR_SCALE_M,
        },
    )
    joint_position_limits = RewTerm(
        func=mdp.joint_position_limits,
        weight=mdp.REWARD_WEIGHTS["joint_position_limits"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_JOINT_ORDER), preserve_order=True)},
    )
    termination = RewTerm(func=mdp.termination, weight=mdp.REWARD_WEIGHTS["termination"])


@configclass
class TerminationsCfg:
    """Immediate and persistent safety checks."""

    refresh_policy_state = DoneTerm(func=mdp.refresh_policy_state, params={})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    immediate_safety = DoneTerm(func=mdp.immediate_safety_violation, params={})
    persistent_safety = DoneTerm(func=mdp.persistent_safety_violation, params={})


@configclass
class CurriculumCfg:
    """Directional-slope terrain curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_level_curriculum)


@configclass
class G1RickshawDirectionalSlopeEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for the registered G1 rickshaw task."""

    scene: G1RickshawSceneCfg = G1RickshawSceneCfg(
        num_envs=4096,
        env_spacing=6.0,
        replicate_physics=True,
        # ContactSensor reporter discovery currently requires USD-visible clones.
        clone_in_fabric=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    feasibility_path: str = os.fspath(DEFAULT_FEASIBILITY_PATH)
    reset_pose_path: str = os.fspath(DEFAULT_RESET_POSES_PATH)
    @property
    def reset_pose_library(self):
        """Validated runtime pose data, intentionally excluded from Hydra serialization."""

        return self.__dict__["__reset_pose_library"]

    @reset_pose_library.setter
    def reset_pose_library(self, value) -> None:
        self.__dict__["__reset_pose_library"] = value

    def __post_init__(self):
        feasibility_path = _configured_path(
            "G1_RICKSHAW_FEASIBILITY_ENVELOPE", Path(self.feasibility_path)
        )
        reset_pose_path = _configured_path("G1_RICKSHAW_RESET_POSES", Path(self.reset_pose_path))
        envelope = load_feasibility_envelope(feasibility_path)
        reset_library = load_reset_pose_library(reset_pose_path)
        calibration = dict(envelope.calibration)
        ranges = _range_pairs(envelope)

        self.feasibility_path = os.fspath(feasibility_path)
        self.reset_pose_path = os.fspath(reset_pose_path)
        self.__dict__["__reset_pose_library"] = reset_library
        self.domain_randomization = mdp.DomainRandomizationCfg(
            enabled=True,
            refresh_interval_iterations=200,
            calibration=calibration,
            ranges={name: ranges[name] for name in mdp.DOMAIN_PARAMETER_NAMES},
            nominal={
                "payload.mass": 0.0,
                "payload.com.x": 0.5 * sum(ranges["payload.com.x"]),
                "payload.com.y": 0.0,
                "payload.com.z": 0.5 * sum(ranges["payload.com.z"]),
                "rolling_resistance.c_rr": _cal(
                    calibration, "rolling_resistance.c_rr_nominal"
                ),
                "terrain.friction": _cal(calibration, "terrain.friction_nominal"),
                "wheel.left_damping": RICKSHAW_URDF_SPEC.wheel_joint_damping,
                "wheel.right_damping": RICKSHAW_URDF_SPEC.wheel_joint_damping,
                "motor.strength": 1.0,
                "joint.model_error": 0.0,
                "control.delay": 0.0,
                "observation.delay": 0.0,
            },
            curriculum=mdp.CurriculumScheduleCfg(),
        )
        self.handle_constraint = mdp.HandleConstraintCfg(
            robot_body_paths=tuple(_cal(calibration, "d6.robot_body_paths")),
            hitch_body_paths=tuple(_cal(calibration, "d6.hitch_body_paths")),
            grasp_local_positions=(
                tuple(_cal(calibration, "dex.left_grasp_center_frame")[:3]),
                tuple(_cal(calibration, "dex.right_grasp_center_frame")[:3]),
            ),
            grasp_local_quaternions_wxyz=(
                tuple(_cal(calibration, "dex.left_grasp_center_frame")[3:]),
                tuple(_cal(calibration, "dex.right_grasp_center_frame")[3:]),
            ),
            linear_stiffness=_cal(calibration, "d6.linear_stiffness_nominal"),
            linear_damping=_cal(calibration, "d6.linear_damping_nominal"),
            angular_stiffness=_cal(calibration, "d6.angular_stiffness_nominal"),
            angular_damping=_cal(calibration, "d6.angular_damping_nominal"),
            max_force=_cal(calibration, "d6.max_force_nominal"),
            max_torque=_cal(calibration, "d6.max_torque_nominal"),
            linear_limit=_cal(calibration, "d6.linear_limit_nominal"),
            angular_limit=_cal(calibration, "d6.angular_limit_nominal"),
            rotation_free_axes=tuple(_cal(calibration, "d6.rotation_free_axes")),
            rotation_driven_axes=tuple(_cal(calibration, "d6.rotation_driven_axes")),
            reaction_is_joint_on_robot=bool(_cal(calibration, "d6.reaction_is_joint_on_robot")),
        )
        self.scene.closed_chain = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Constraints",
            spawn=ReplicatedDualD6SpawnerCfg(
                handle_constraint=self.handle_constraint,
            ),
        )
        self.rickshaw_pose = mdp.RickshawPoseTargetCfg(
            hitch_height_target=_cal(calibration, "rickshaw_pose.hitch_height_target"),
            hitch_height_tolerance=_cal(calibration, "rickshaw_pose.hitch_height_tolerance"),
            hitch_vertical_speed_tolerance=_cal(
                calibration, "rickshaw_pose.hitch_vertical_speed_tolerance"
            ),
        )
        self.rolling_resistance = mdp.RollingResistanceCfg(enabled=True)
        self.reset_validation = mdp.ResetValidationCfg(
            hand_position_tolerance=_cal(
                calibration, "reset.hand_position_tolerance"
            ),
            minimum_wheel_normal_force=_cal(
                calibration, "safety.minimum_wheel_normal_force"
            ),
        )
        self.task_entity_names = mdp.TaskEntityNamesCfg(
            policy_joint_names=G1_JOINT_ORDER,
            arm_joint_names=ARM_JOINT_NAMES,
            dex_joint_names=DEX_JOINT_NAMES,
            wheel_body_names=WHEEL_LINK_NAMES,
            hitch_body_names=HITCH_LINK_NAMES,
            foot_body_names=FOOT_BODY_NAMES,
            torso_body_name=TORSO_BODY_NAME,
        )
        self.robot_mass = _cal(calibration, "fat.robot_mass")
        self.dex_q_grasp = tuple(_cal(calibration, "dex.q_grasp"))

        self.scene.robot.actuators["legs"].stiffness = _cal(
            calibration, "control.leg_stiffness"
        )
        self.scene.robot.actuators["legs"].damping = _cal(
            calibration, "control.leg_damping"
        )
        self.scene.robot.actuators["feet"].stiffness = _cal(
            calibration, "control.foot_stiffness"
        )
        self.scene.robot.actuators["feet"].damping = _cal(
            calibration, "control.foot_damping"
        )
        self.scene.robot.actuators["waist"].stiffness = _cal(
            calibration, "control.waist_stiffness"
        )
        self.scene.robot.actuators["waist"].damping = _cal(
            calibration, "control.waist_damping"
        )
        self.scene.robot.actuators["arms"].stiffness = _cal(
            calibration, "control.arm_stiffness"
        )
        self.scene.robot.actuators["arms"].damping = _cal(
            calibration, "control.arm_damping"
        )

        support = mdp.SupportPolygonCfg(
            foot_half_length=_cal(calibration, "support.foot_half_length"),
            foot_half_width=_cal(calibration, "support.foot_half_width"),
            foot_center_offset_x=_cal(calibration, "support.foot_center_offset_x"),
        )
        analytic_force = mdp.AnalyticForceCfg(
            minimum_wheel_normal_force=_cal(calibration, "safety.minimum_wheel_normal_force")
        )
        fat2 = mdp.FAT2Cfg(
            robot_mass=_cal(calibration, "fat.robot_mass"),
            com_radius=_cal(calibration, "fat.com_radius"),
            com_radius_bounds=tuple(_cal(calibration, "fat.com_radius_bounds")),
            theta_max=_cal(calibration, "safety.theta_max"),
            wrench_consistency_relative_tolerance=_cal(
                calibration, "fat.wrench_consistency_relative_tolerance"
            ),
            wrench_consistency_absolute_floor_n=_cal(
                calibration, "fat.wrench_consistency_absolute_floor_n"
            ),
            wrench_consistency_window_steps=_cal(
                calibration, "fat.wrench_consistency_window_steps"
            ),
        )
        zmp = mdp.ZMPCfg(min_ground_reaction=_cal(calibration, "safety.min_ground_reaction"))
        speed = mdp.SpeedReferenceCfg(
            acceleration_limit=_range_upper(envelope, "command.acceleration_limit"),
            jerk_limit=_range_upper(envelope, "command.jerk_limit"),
        )
        policy_update = mdp.PolicyStateUpdateCfg(
            speed_reference=speed,
            analytic_force=analytic_force,
            support_polygon=support,
            fat2=fat2,
            zmp=zmp,
            command_sampling=mdp.SpeedCommandSamplingCfg(),
        )
        self.policy_update = policy_update

        self.scene.robot.actuators["dex"] = ImplicitActuatorCfg(
            joint_names_expr=list(DEX_JOINT_NAMES),
            effort_limit_sim=_cal(calibration, "dex.effort_limit"),
            velocity_limit_sim=_cal(calibration, "dex.velocity_limit"),
            stiffness=_cal(calibration, "dex.actuator_stiffness"),
            damping=_cal(calibration, "dex.actuator_damping"),
        )
        self.events.initialize_mdp.params = {
            "handle_constraint_cfg": self.handle_constraint,
            "rolling_resistance_cfg": self.rolling_resistance,
            "entity_names_cfg": self.task_entity_names,
            "rickshaw_pose_cfg": self.rickshaw_pose,
            "robot_mass": self.robot_mass,
            "dex_q_grasp": self.dex_q_grasp,
        }
        self.events.initialize_domain.params = {"cfg": self.domain_randomization}
        self.events.policy_interval.params = {"cfg": policy_update}

        self.terminations.refresh_policy_state.params = {"cfg": policy_update}
        self.terminations.immediate_safety.params = {
            "cfg": mdp.ImmediateSafetyCfg(
                illegal_contact_force_threshold=_cal(
                    calibration, "safety.illegal_contact_force_threshold"
                ),
                wheel_lift_normal_force_threshold=_cal(
                    calibration, "safety.minimum_wheel_normal_force"
                ),
                d6_residual_limit=_cal(calibration, "safety.d6_residual_limit"),
                d6_impulse_limit=_cal(calibration, "safety.d6_impulse_limit"),
            ),
            "illegal_contact_sensor_cfg": SceneEntityCfg(
                "robot_contacts",
                body_names=list(ILLEGAL_CONTACT_BODY_NAMES),
                preserve_order=True,
            ),
            "robot_asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(G1_JOINT_ORDER), preserve_order=True
            ),
        }
        self.terminations.persistent_safety.params = {
            "cfg": mdp.PersistentSafetyCfg(
                torso_tilt_max=_cal(calibration, "safety.theta_max"),
                hitch_height_bounds=tuple(_cal(calibration, "safety.hitch_height_bounds")),
                rickshaw_pitch_bounds=tuple(_cal(calibration, "safety.rickshaw_pitch_bounds")),
                lateral_corridor=_cal(calibration, "safety.corridor_half_width"),
                heading_envelope=_cal(calibration, "safety.heading_error_limit"),
                overspeed_margin=_cal(calibration, "safety.overspeed_margin"),
                arm_torque_limit=_cal(calibration, "safety.arm_torque_limit"),
            )
        }

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.enable_external_forces_every_iteration = True
        if hasattr(self.sim.physx, "solve_articulation_contact_last"):
            self.sim.physx.solve_articulation_contact_last = True
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.rickshaw.spawn.activate_contact_sensors = True
        self.scene.robot_contacts.update_period = self.sim.dt
        self.scene.wheel_contacts.update_period = self.sim.dt
        # This flag controls deterministic row/column generation, not whether
        # the runtime terrain-level CurriculumTerm is enabled.
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1RickshawDirectionalSlopePlayEnvCfg(G1RickshawDirectionalSlopeEnvCfg):
    """Play configuration with structured terrain and training physics intact."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.curriculum = None
        self.domain_randomization.enabled = False
        self.domain_randomization.curriculum.static_hand_load_iterations = 0


def _rebind_manager_cfg_references(env_cfg: G1RickshawDirectionalSlopeEnvCfg) -> None:
    """Restore shared manager bindings after configclass deep-copies top-level fields."""

    initialize_params = env_cfg.events.initialize_mdp.params
    env_cfg.scene.closed_chain.spawn.handle_constraint = env_cfg.handle_constraint
    initialize_params["handle_constraint_cfg"] = env_cfg.handle_constraint
    initialize_params["rolling_resistance_cfg"] = env_cfg.rolling_resistance
    initialize_params["entity_names_cfg"] = env_cfg.task_entity_names
    initialize_params["rickshaw_pose_cfg"] = env_cfg.rickshaw_pose
    env_cfg.events.initialize_domain.params["cfg"] = env_cfg.domain_randomization
    env_cfg.events.policy_interval.params["cfg"] = env_cfg.policy_update
    env_cfg.terminations.refresh_policy_state.params["cfg"] = env_cfg.policy_update


def _install_post_init_rebind(config_type) -> None:
    original_post_init = config_type.__post_init__

    def _post_init_with_rebind(self) -> None:
        original_post_init(self)
        _rebind_manager_cfg_references(self)

    config_type.__post_init__ = _post_init_with_rebind


_install_post_init_rebind(G1RickshawDirectionalSlopeEnvCfg)
_install_post_init_rebind(G1RickshawDirectionalSlopePlayEnvCfg)


__all__ = [
    "G1RickshawDirectionalSlopeEnvCfg",
    "G1RickshawDirectionalSlopePlayEnvCfg",
    "configure_rolling_resistance",
]


configure_rolling_resistance = mdp.configure_rolling_resistance
