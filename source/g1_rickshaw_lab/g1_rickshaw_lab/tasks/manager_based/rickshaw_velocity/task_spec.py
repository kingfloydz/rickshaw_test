"""Simulator-independent physical specifications for the rickshaw task."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass

from g1_rickshaw_lab.rickshaw_spec import HITCH_HALF_WIDTH, HITCH_X, HITCH_Z, WHEEL_RADIUS


@dataclass(kw_only=True)
class RickshawPoseTargetCfg:
    """Rickshaw front-lift target and reset acceptance tolerances."""

    wheel_radius: float = WHEEL_RADIUS
    hitch_x: float = HITCH_X
    hitch_z: float = HITCH_Z
    hitch_half_width: float = HITCH_HALF_WIDTH
    hitch_height_target: float = MISSING
    hitch_height_tolerance: float = MISSING
    hitch_vertical_speed_tolerance: float = MISSING


@dataclass(kw_only=True)
class HandleConstraintCfg:
    """Complete calibrated double-D6 constraint specification."""

    robot_body_paths: tuple[str, str] = MISSING
    hitch_body_paths: tuple[str, str] = MISSING
    grasp_local_positions: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    grasp_local_quaternions_wxyz: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] = MISSING
    linear_stiffness: float = MISSING
    linear_damping: float = MISSING
    angular_stiffness: float = MISSING
    angular_damping: float = MISSING
    max_force: float = MISSING
    max_torque: float = MISSING
    linear_limit: float = MISSING
    angular_limit: float = MISSING
    rotation_free_axes: tuple[bool, bool, bool] = MISSING
    rotation_driven_axes: tuple[bool, bool, bool] = MISSING
    reaction_is_joint_on_robot: bool = MISSING
    env_prim_path_template: str = "/World/envs/env_{env_id}"
    joint_prim_path_template: str = "{ENV_NS}/Constraints/{side}_grasp_hitch_d6"

    def validate(self) -> None:
        if len(self.robot_body_paths) != 2 or len(self.hitch_body_paths) != 2:
            raise ValueError("two robot grasp bodies and two hitch bodies are required")
        if len(self.grasp_local_positions) != 2 or len(self.grasp_local_quaternions_wxyz) != 2:
            raise ValueError("left/right calibrated grasp local poses are required")
        values = (
            self.linear_stiffness,
            self.linear_damping,
            self.angular_stiffness,
            self.angular_damping,
            self.max_force,
            self.max_torque,
            self.linear_limit,
            self.angular_limit,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("D6 drive, limit, force, and torque values must be positive and finite")
        if len(self.rotation_free_axes) != 3 or len(self.rotation_driven_axes) != 3:
            raise ValueError("rotation axis modes must have exactly three entries")
        if any(
            free and driven for free, driven in zip(self.rotation_free_axes, self.rotation_driven_axes, strict=True)
        ):
            raise ValueError("a physically free D6 rotation axis cannot have a drive")
        if not isinstance(self.reaction_is_joint_on_robot, bool):
            raise ValueError("reaction_is_joint_on_robot must explicitly define the PhysX sign")
        for quaternion in self.grasp_local_quaternions_wxyz:
            norm = math.sqrt(sum(component * component for component in quaternion))
            if abs(norm - 1.0) > 1.0e-4:
                raise ValueError("calibrated grasp-local quaternion must be unit length")


__all__ = ["HandleConstraintCfg", "RickshawPoseTargetCfg"]
