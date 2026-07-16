"""Reset and policy-step events for the closed-chain rickshaw task."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field
from types import MethodType
from typing import Any, Mapping

import torch

from g1_rickshaw_lab.assets.rickshaw import (
    HITCH_HALF_WIDTH,
    HITCH_X,
    HITCH_Z,
    WHEEL_RADIUS,
    WHEEL_TRACK,
)
from g1_rickshaw_lab.static_equilibrium import fixed_contact_static_components

from .actions import (
    ACTION_DIM,
    ButterworthActionState,
    action_scale_vector,
    canonicalize_action_scale,
)
from .actuation import actuator_effort_limits
from .dynamics import (
    AnalyticForceCfg,
    AnalyticHandleForceState,
    CartInteractionWrenchState,
    FAT2Cfg,
    GRAVITY,
    RickshawMassProperties,
    RollingResistanceCfg,
    SpeedReferenceCfg,
    SupportPolygonCfg,
    ZMPCfg,
    actual_rickshaw_geometry_in_slope_frame,
    adapt_d6_reaction_wrench,
    accumulate_cart_interaction_wrench,
    apply_rolling_resistance,
    cart_system_mass_kinematics,
    combine_mass_properties,
    effective_cart_mass,
    effective_wheel_damping,
    quat_apply_wxyz,
    rickshaw_pitch_from_quaternion,
    robot_system_mass_kinematics,
    update_analytic_rickshaw_force,
    update_cart_interaction_wrench,
    update_fat2_reference,
    update_slope_frame,
    update_speed_reference,
    update_support_polygon,
    update_zmp_stability,
)
from .curricula import CurriculumRuntimeState, CurriculumScheduleCfg, CurriculumStage
from .observations import INDEPENDENT_EXTRINSIC_NAMES


@dataclass
class CommandState:
    v_sample: torch.Tensor
    v_ref: torch.Tensor
    a_ref: torch.Tensor
    resampling_elapsed_s: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "CommandState":
        zeros = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.v_sample[ids] = 0.0
        self.v_ref[ids] = 0.0
        self.a_ref[ids] = 0.0
        self.resampling_elapsed_s[ids] = 0.0


@dataclass
class PathTrackingState:
    lateral_error: torch.Tensor
    heading_error: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "PathTrackingState":
        zeros = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(zeros.clone(), zeros.clone())


@dataclass
class RickshawRuntimeState:
    wheel_normal_force: torch.Tensor
    hitch_height: torch.Tensor
    hitch_vertical_speed: torch.Tensor
    pitch: torch.Tensor
    two_wheel_contact: torch.Tensor
    d6_residual: torch.Tensor
    d6_impulse: torch.Tensor
    d6_wrench_w: torch.Tensor
    hand_force_w: torch.Tensor
    hand_torque_w: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        num_wheels: int = 2,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "RickshawRuntimeState":
        scalar = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(
            wheel_normal_force=torch.zeros(
                (num_envs, num_wheels), device=device, dtype=dtype
            ),
            hitch_height=scalar.clone(),
            hitch_vertical_speed=scalar.clone(),
            pitch=scalar.clone(),
            two_wheel_contact=torch.zeros(num_envs, device=device, dtype=torch.bool),
            d6_residual=scalar.clone(),
            # [linear impulse (N*s), angular impulse (N*m*s)] maxima.
            d6_impulse=torch.zeros((num_envs, 2), device=device, dtype=dtype),
            d6_wrench_w=torch.zeros((num_envs, 2, 6), device=device, dtype=dtype),
            hand_force_w=torch.zeros((num_envs, 3), device=device, dtype=dtype),
            hand_torque_w=torch.zeros((num_envs, 3), device=device, dtype=dtype),
        )


@dataclass
class StabilityState:
    theta_fat: torch.Tensor
    fat_valid: torch.Tensor
    fat_wrench_consistent: torch.Tensor
    fat_wrench_relative_error: torch.Tensor
    torso_pitch: torch.Tensor
    zmp_s: torch.Tensor
    zmp_margin: torch.Tensor
    zmp_valid: torch.Tensor
    ground_reaction_normal: torch.Tensor
    support_center_w: torch.Tensor
    support_points_sy: torch.Tensor
    support_point_mask: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "StabilityState":
        scalar = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(
            theta_fat=scalar.clone(),
            fat_valid=torch.zeros(num_envs, device=device, dtype=torch.bool),
            fat_wrench_consistent=torch.zeros(
                num_envs, device=device, dtype=torch.bool
            ),
            fat_wrench_relative_error=torch.zeros(
                (num_envs, 2), device=device, dtype=dtype
            ),
            torso_pitch=scalar.clone(),
            zmp_s=scalar.clone(),
            zmp_margin=scalar.clone(),
            zmp_valid=torch.zeros(num_envs, device=device, dtype=torch.bool),
            ground_reaction_normal=scalar.clone(),
            support_center_w=torch.zeros((num_envs, 3), device=device, dtype=dtype),
            support_points_sy=torch.zeros((num_envs, 8, 2), device=device, dtype=dtype),
            support_point_mask=torch.zeros((num_envs, 8), device=device, dtype=torch.bool),
        )


@dataclass(kw_only=True)
class RickshawPoseTargetCfg:
    wheel_radius: float = WHEEL_RADIUS
    hitch_x: float = HITCH_X
    hitch_z: float = HITCH_Z
    hitch_half_width: float = HITCH_HALF_WIDTH
    hitch_height_target: float = MISSING
    hitch_height_tolerance: float = MISSING
    hitch_vertical_speed_tolerance: float = MISSING


def target_pitch_from_hitch_height(cfg: RickshawPoseTargetCfg) -> float:
    """Solve the rickshaw front-lift pitch from the target hitch height."""

    radius = math.hypot(cfg.hitch_x, cfg.hitch_z - cfg.wheel_radius)
    phase = math.atan2(cfg.hitch_z - cfg.wheel_radius, cfg.hitch_x)
    ratio = (cfg.hitch_height_target - cfg.wheel_radius) / radius
    if not -1.0 <= ratio <= 1.0:
        raise ValueError("infeasible hitch_height_target")
    return math.asin(ratio) - phase


def wheel_phase_from_path_position(
    path_position: torch.Tensor, wheel_radius: float = 0.374999
) -> torch.Tensor:
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")
    phase = torch.remainder(-path_position / wheel_radius + math.pi, 2.0 * math.pi) - math.pi
    return torch.stack((phase, phase), dim=-1)


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product for batched wxyz quaternions."""

    if lhs.shape != rhs.shape or lhs.shape[-1] != 4:
        raise ValueError("quaternion operands must have identical [...,4] shapes")
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def target_cart_orientation(
    slope_quaternion_wxyz: torch.Tensor, alpha_target: float
) -> torch.Tensor:
    zeros = torch.zeros_like(slope_quaternion_wxyz[..., 0])
    pitch_relative = torch.stack(
        (
            torch.full_like(zeros, math.cos(0.5 * alpha_target)),
            zeros,
            torch.full_like(zeros, -math.sin(0.5 * alpha_target)),
            zeros,
        ),
        dim=-1,
    )
    return quat_multiply_wxyz(slope_quaternion_wxyz, pitch_relative)


def fit_cart_pose_to_hitch_targets(
    target_hitch_positions_w: torch.Tensor,
    nominal_cart_quaternion_wxyz: torch.Tensor,
    path_normal_w: torch.Tensor,
    cfg: RickshawPoseTargetCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one rigid cart pose to the two independently loaded hitch targets."""

    if target_hitch_positions_w.ndim != 3 or target_hitch_positions_w.shape[1:] != (2, 3):
        raise ValueError("target hitch positions must have shape [N,2,3]")
    num_envs = target_hitch_positions_w.shape[0]
    if nominal_cart_quaternion_wxyz.shape != (num_envs, 4):
        raise ValueError("nominal cart quaternion must have shape [N,4]")
    if path_normal_w.shape != (num_envs, 3):
        raise ValueError("path normal must have shape [N,3]")

    dtype = target_hitch_positions_w.dtype
    device = target_hitch_positions_w.device
    local_hitches = torch.tensor(
        (
            (cfg.hitch_x, cfg.hitch_half_width, cfg.hitch_z),
            (cfg.hitch_x, -cfg.hitch_half_width, cfg.hitch_z),
        ),
        device=device,
        dtype=dtype,
    ).expand(num_envs, -1, -1)
    target_delta_w = target_hitch_positions_w[:, 1] - target_hitch_positions_w[:, 0]
    target_length = torch.linalg.vector_norm(target_delta_w, dim=-1, keepdim=True)
    if torch.any(target_length <= 1.0e-8):
        raise RuntimeError("two-point cart fit has a degenerate lateral baseline")
    # Local +y points from the right hitch to the left hitch.
    y_axis_w = -target_delta_w / target_length
    local_forward = torch.zeros((num_envs, 3), device=device, dtype=dtype)
    local_forward[:, 0] = 1.0
    nominal_forward_w = quat_apply_wxyz(
        nominal_cart_quaternion_wxyz, local_forward
    )
    x_axis_w = nominal_forward_w - torch.sum(
        nominal_forward_w * y_axis_w, dim=-1, keepdim=True
    ) * y_axis_w
    x_length = torch.linalg.vector_norm(x_axis_w, dim=-1, keepdim=True)
    if torch.any(x_length <= 1.0e-8):
        raise RuntimeError("two-point cart fit cannot preserve the nominal forward axis")
    x_axis_w = x_axis_w / x_length
    z_axis_w = torch.cross(x_axis_w, y_axis_w, dim=-1)
    normal_alignment = torch.sum(z_axis_w * path_normal_w, dim=-1)
    if torch.any(normal_alignment <= 0.0):
        raise RuntimeError("two-point cart fit would invert the cart normal axis")
    # Columns are the fitted local axes expressed in world coordinates.
    rotation = torch.stack((x_axis_w, y_axis_w, z_axis_w), dim=-1)
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    quaternion_w = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=1.0e-12))
    if torch.any(quaternion_w <= 1.0e-6):
        raise RuntimeError("two-point cart fit reached an unsupported 180-degree rotation")
    denominator = 4.0 * quaternion_w
    fitted_quaternion = torch.stack(
        (
            quaternion_w,
            (rotation[:, 2, 1] - rotation[:, 1, 2]) / denominator,
            (rotation[:, 0, 2] - rotation[:, 2, 0]) / denominator,
            (rotation[:, 1, 0] - rotation[:, 0, 1]) / denominator,
        ),
        dim=-1,
    )
    fitted_quaternion = fitted_quaternion / torch.linalg.vector_norm(
        fitted_quaternion, dim=-1, keepdim=True
    )
    local_midpoint = torch.mean(local_hitches, dim=1)
    target_midpoint = torch.mean(target_hitch_positions_w, dim=1)
    fitted_root = target_midpoint - quat_apply_wxyz(fitted_quaternion, local_midpoint)
    fitted_hitches = fitted_root[:, None, :] + quat_apply_wxyz(
        fitted_quaternion[:, None, :].expand(-1, 2, -1), local_hitches
    )
    fit_error = torch.amax(
        torch.linalg.vector_norm(fitted_hitches - target_hitch_positions_w, dim=-1),
        dim=-1,
    )
    return fitted_root, fitted_quaternion, fit_error


@dataclass
class SpeedCommandSamplingCfg:
    minimum: float = 0.0
    maximum: float = 1.0
    standing_fraction: float = 0.02
    resampling_time_s: float = 10.0

    def validate(self) -> None:
        if self.minimum != 0.0 or self.maximum != 1.0:
            raise ValueError("the fixed task speed sample range is [0, 1] m/s")
        if not 0.0 <= self.standing_fraction <= 1.0:
            raise ValueError("standing_fraction must lie in [0,1]")
        if self.resampling_time_s <= 0.0:
            raise ValueError("resampling_time_s must be positive")


def sample_speed_commands(
    num_samples: int,
    cfg: SpeedCommandSamplingCfg = SpeedCommandSamplingCfg(),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    cfg.validate()
    samples = torch.rand(num_samples, device=device, dtype=dtype, generator=generator)
    samples = cfg.minimum + (cfg.maximum - cfg.minimum) * samples
    if cfg.standing_fraction > 0.0:
        standing = torch.rand(num_samples, device=device, generator=generator) < cfg.standing_fraction
        samples[standing] = 0.0
    return samples


def reset_speed_command(env: Any, env_ids: torch.Tensor) -> None:
    """Reset all command states to zero."""

    env.command_state.reset(env_ids)


def resample_speed_command(
    env: Any,
    env_ids: torch.Tensor,
    cfg: SpeedCommandSamplingCfg = SpeedCommandSamplingCfg(),
) -> None:
    if env_ids.numel() == 0:
        return
    env.command_state.v_sample[env_ids] = sample_speed_commands(
        env_ids.numel(),
        cfg,
        device=env.command_state.v_sample.device,
        dtype=env.command_state.v_sample.dtype,
    )
    env.command_state.resampling_elapsed_s[env_ids] = 0.0


def advance_speed_command_resampling(
    env: Any,
    cfg: SpeedCommandSamplingCfg = SpeedCommandSamplingCfg(),
) -> torch.Tensor:
    """Advance per-environment timers and resample exactly every 10 seconds."""

    cfg.validate()
    elapsed = env.command_state.resampling_elapsed_s
    elapsed += float(env.step_dt)
    due = elapsed >= cfg.resampling_time_s - 1.0e-9
    due_ids = torch.nonzero(due, as_tuple=False).flatten()
    if due_ids.numel() > 0:
        resample_speed_command(env, due_ids, cfg)
    return due_ids


def advance_speed_reference(env: Any, cfg: SpeedReferenceCfg) -> None:
    """Policy-step event; never expose ``v_sample`` to reward or observation."""

    update_speed_reference(env.command_state, env.command_state.v_sample, env.step_dt, cfg)


def compute_path_tracking_errors(
    robot_position_w: torch.Tensor,
    cart_position_w: torch.Tensor,
    robot_quaternion_wxyz: torch.Tensor,
    path_origin_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Midpoint lateral error and wrapped robot heading error."""

    midpoint = 0.5 * (robot_position_w + cart_position_w)
    lateral_error = torch.sum((midpoint - path_origin_w) * path_lateral_w, dim=-1)
    local_x = torch.zeros_like(path_tangent_w)
    local_x[:, 0] = 1.0
    robot_forward_w = quat_apply_wxyz(robot_quaternion_wxyz, local_x)
    heading_error = torch.atan2(
        torch.sum(robot_forward_w * path_lateral_w, dim=-1),
        torch.sum(robot_forward_w * path_tangent_w, dim=-1),
    )
    return lateral_error, heading_error


def update_path_tracking_state(env: Any) -> None:
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    origin = env.scene.terrain.env_origins
    lateral, heading = compute_path_tracking_errors(
        robot.data.root_pos_w,
        cart.data.root_pos_w,
        robot.data.root_quat_w,
        origin,
        env.path_tangent_w,
        env.path_lateral_w,
    )
    env.path_state.lateral_error[:] = lateral
    env.path_state.heading_error[:] = heading


def update_rickshaw_geometry_state(
    env: Any,
    *,
    hitch_position_w: torch.Tensor | None = None,
    hitch_velocity_w: torch.Tensor | None = None,
    pitch: torch.Tensor | None = None,
) -> None:
    """Read actual wheel/hitch poses; do not substitute nominal URDF frames."""

    cart = env.scene["rickshaw"]
    if hitch_position_w is None:
        hitch_position_w = torch.mean(
            cart.data.body_pos_w[:, env.hitch_body_ids], dim=1
        )
    if hitch_velocity_w is None:
        hitch_velocity_w = torch.mean(
            cart.data.body_lin_vel_w[:, env.hitch_body_ids], dim=1
        )
    terrain_origin = env.scene.terrain.env_origins
    env.rickshaw_state.hitch_height[:] = torch.sum(
        (hitch_position_w - terrain_origin) * env.path_normal_w, dim=-1
    )
    env.rickshaw_state.hitch_vertical_speed[:] = torch.sum(
        hitch_velocity_w * env.path_normal_w, dim=-1
    )
    if pitch is None:
        pitch = rickshaw_pitch_from_quaternion(
            cart.data.root_quat_w, env.path_tangent_w, env.path_normal_w
        )
    env.rickshaw_state.pitch[:] = pitch


def sample_rolling_resistance(
    env: Any, env_ids: torch.Tensor, c_rr_range: tuple[float, float]
) -> None:
    """Sample the single tensor shared by physics, critic, analytics, and logs."""

    low, high = c_rr_range
    if low < 0.0 or high < low:
        raise ValueError("c_rr_range must be non-negative and ordered")
    samples = torch.empty(
        env_ids.numel(), device=env.c_rr.device, dtype=env.c_rr.dtype
    ).uniform_(low, high)
    env.c_rr[env_ids] = samples


@dataclass(kw_only=True)
class RuntimeRandomizationCfg:
    """Feasibility-bounded reset-time physical samples.

    The same tensors feed simulation parameters, analytics, teacher
    extrinsics, critic state, and logs.  Ranges are loaded from
    ``feasibility_envelope.yaml`` by the environment config; this class never
    supplies fallback calibration values.
    """

    ranges: Mapping[str, tuple[float, float]] = MISSING
    calibration: Mapping[str, Any] = MISSING
    nominal_values: Mapping[str, float] = MISSING
    curriculum: CurriculumScheduleCfg = MISSING
    sample_ranges: bool = True
    teacher_extrinsic_names: tuple[str, ...] = INDEPENDENT_EXTRINSIC_NAMES

    def validate(self) -> None:
        if not isinstance(self.ranges, Mapping) or not self.ranges:
            raise ValueError("runtime randomization requires feasibility ranges")
        for name, value in self.ranges.items():
            if len(value) != 2:
                raise ValueError(f"range {name!r} must be a two-value interval")
            low, high = float(value[0]), float(value[1])
            if not math.isfinite(low) or not math.isfinite(high) or high < low:
                raise ValueError(f"range {name!r} is not finite and ordered")
        for name in self.teacher_extrinsic_names:
            if name not in self.ranges:
                raise ValueError(f"teacher extrinsic {name!r} lacks a feasibility range")
        required = set(self.teacher_extrinsic_names) | {"joint.model_error"}
        missing_nominal = sorted(required - set(self.nominal_values))
        if missing_nominal:
            raise ValueError(f"nominal values are missing {missing_nominal}")
        if self.sample_ranges:
            non_singleton = sorted(
                name
                for name in required
                if float(self.ranges[name][0]) != float(self.ranges[name][1])
            )
            if non_singleton:
                raise ValueError(
                    "replicated physics only permits singleton scan ranges; "
                    f"non-singleton ranges: {non_singleton}"
                )
        for name in required:
            value = float(self.nominal_values[name])
            low, high = self.ranges[name]
            if not math.isfinite(value) or not float(low) <= value <= float(high):
                raise ValueError(f"nominal {name!r} lies outside its feasibility range")
        if float(self.ranges["joint.model_error"][0]) <= -1.0:
            raise ValueError("joint.model_error must keep actuator gains positive")
        self.curriculum.validate()

def _ensure_named_tensor_dict(env: Any, attribute: str) -> dict[str, torch.Tensor]:
    values = getattr(env, attribute, None)
    if values is None:
        values = {}
        setattr(env, attribute, values)
    return values


def _ensure_wheel_joint_ids(env: Any) -> list[int]:
    if hasattr(env, "wheel_joint_ids"):
        return env.wheel_joint_ids
    from g1_rickshaw_lab.assets.rickshaw import WHEEL_JOINT_NAMES

    cart = env.scene["rickshaw"]
    env.wheel_joint_ids = _exact_name_ids(
        tuple(cart.joint_names), WHEEL_JOINT_NAMES, "wheel joint"
    )
    return env.wheel_joint_ids


def _update_teacher_extrinsics(
    env: Any,
    env_ids: torch.Tensor,
    cfg: RuntimeRandomizationCfg,
    sampled: Mapping[str, torch.Tensor],
) -> None:
    from .observations import assemble_teacher_extrinsics

    values = _ensure_named_tensor_dict(env, "teacher_extrinsic_values")
    bounds = getattr(env, "teacher_extrinsic_bounds", None)
    if bounds is None:
        bounds = {
            name: (
                float(cfg.ranges[name][0]),
                float(cfg.ranges[name][1]),
            )
            for name in cfg.teacher_extrinsic_names
        }
        env.teacher_extrinsic_bounds = bounds
    for name in cfg.teacher_extrinsic_names:
        if name not in values:
            values[name] = torch.zeros(env.num_envs, device=env.device)
        values[name][env_ids] = sampled[name]
    env.normalized_teacher_extrinsics = assemble_teacher_extrinsics(
        values, bounds, cfg.teacher_extrinsic_names
    )


def _write_payload_to_physx(
    env: Any,
    env_ids: torch.Tensor,
    payload_mass: torch.Tensor,
    payload_com: torch.Tensor,
) -> None:
    """Add a point payload to the cart base body's mass properties in PhysX."""

    from g1_rickshaw_lab.assets.rickshaw import BASE_LINK_NAME, RICKSHAW_TOTAL_MASS

    if env_ids.numel() == 0:
        return
    if not hasattr(env, "_rickshaw_payload_mass_written"):
        env._rickshaw_payload_mass_written = torch.zeros(env.num_envs, device=env.device)
        env._rickshaw_payload_com_written = torch.zeros((env.num_envs, 3), device=env.device)
    desired_mass = payload_mass.to(
        device=env._rickshaw_payload_mass_written.device,
        dtype=env._rickshaw_payload_mass_written.dtype,
    )
    desired_com = payload_com.to(
        device=env._rickshaw_payload_com_written.device,
        dtype=env._rickshaw_payload_com_written.dtype,
    )
    previous_mass = env._rickshaw_payload_mass_written[env_ids]
    previous_com = env._rickshaw_payload_com_written[env_ids]
    mass_changed = ~torch.isclose(previous_mass, desired_mass, rtol=0.0, atol=1.0e-6)
    com_relevant = (torch.abs(previous_mass) > 1.0e-6) | (torch.abs(desired_mass) > 1.0e-6)
    com_changed = com_relevant & torch.any(
        ~torch.isclose(previous_com, desired_com, rtol=0.0, atol=1.0e-6),
        dim=-1,
    )
    changed = mass_changed | com_changed
    if not torch.any(changed):
        return
    env_ids = env_ids[changed]
    payload_mass = payload_mass[changed]
    payload_com = payload_com[changed]
    desired_mass = desired_mass[changed]
    desired_com = desired_com[changed]

    cart = env.scene["rickshaw"]
    view = cart.root_physx_view
    ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if not hasattr(env, "_rickshaw_default_masses_cpu"):
        env._rickshaw_base_body_id = _exact_name_ids(
            tuple(cart.body_names), (BASE_LINK_NAME,), "rickshaw base body"
        )[0]
        env._rickshaw_default_masses_cpu = view.get_masses().clone().cpu()
        env._rickshaw_default_coms_cpu = view.get_coms().clone().cpu()
        env._rickshaw_default_inertias_cpu = view.get_inertias().clone().cpu()
        default_total = torch.sum(env._rickshaw_default_masses_cpu, dim=-1)
        if not torch.allclose(
            default_total,
            torch.full_like(default_total, RICKSHAW_TOTAL_MASS),
            rtol=0.0,
            atol=1.0e-4,
        ):
            actual = (float(torch.amin(default_total)), float(torch.amax(default_total)))
            raise RuntimeError(
                f"rickshaw USD mass contract failed: expected {RICKSHAW_TOTAL_MASS}, got range {actual}"
            )

    base_id = env._rickshaw_base_body_id
    default_mass = env._rickshaw_default_masses_cpu[ids_cpu, base_id]
    default_com = env._rickshaw_default_coms_cpu[ids_cpu, base_id, :3]
    default_inertia = env._rickshaw_default_inertias_cpu[ids_cpu, base_id].reshape(-1, 3, 3)
    sampled_mass = payload_mass.detach().to(device="cpu", dtype=default_mass.dtype)
    sampled_com = payload_com.detach().to(device="cpu", dtype=default_com.dtype)
    zero_payload_inertia = torch.zeros_like(default_inertia)
    total_mass, total_com, total_inertia = combine_mass_properties(
        default_mass,
        default_com,
        default_inertia,
        sampled_mass,
        sampled_com,
        zero_payload_inertia,
    )

    masses = view.get_masses().clone()
    coms = view.get_coms().clone()
    inertias = view.get_inertias().clone()
    masses[ids_cpu, base_id] = total_mass
    coms[ids_cpu, base_id, :3] = total_com
    inertias[ids_cpu, base_id] = total_inertia.reshape(-1, 9)
    view.set_masses(masses, ids_cpu)
    view.set_coms(coms, ids_cpu)
    view.set_inertias(inertias, ids_cpu)
    if not hasattr(env, "rickshaw_body_masses"):
        env.rickshaw_body_masses = torch.zeros(
            (env.num_envs, masses.shape[1]), device=env.device, dtype=torch.float32
        )
        env.rickshaw_body_com_pos_b = torch.zeros(
            (env.num_envs, coms.shape[1], 3), device=env.device, dtype=torch.float32
        )
    env.rickshaw_body_masses[env_ids] = masses[ids_cpu].to(
        device=env.device, dtype=env.rickshaw_body_masses.dtype
    )
    env.rickshaw_body_com_pos_b[env_ids] = coms[ids_cpu, :, :3].to(
        device=env.device, dtype=env.rickshaw_body_com_pos_b.dtype
    )
    env._rickshaw_payload_mass_written[env_ids] = desired_mass
    env._rickshaw_payload_com_written[env_ids] = desired_com


def _write_effective_terrain_friction_to_physx(
    env: Any, env_ids: torch.Tensor, friction: torch.Tensor
) -> None:
    """Set per-environment collider friction against the unit-friction terrain.

    Isaac Lab imports the generated terrain as one shared mesh, so its material
    cannot vary by environment.  With the required ``multiply`` combine mode
    and terrain coefficient 1.0, writing the sampled coefficient on both G1 and
    rickshaw colliders is physically equivalent for every terrain contact.
    """

    material_cfg = env.scene.terrain.cfg.physics_material
    if (
        material_cfg is None
        or material_cfg.friction_combine_mode != "multiply"
        or not math.isclose(float(material_cfg.static_friction), 1.0)
        or not math.isclose(float(material_cfg.dynamic_friction), 1.0)
    ):
        raise RuntimeError(
            "per-environment terrain friction requires a unit-friction terrain with multiply combine mode"
        )
    if torch.any(~torch.isfinite(friction)) or torch.any(friction <= 0.0):
        raise ValueError("terrain friction samples must be finite and positive")
    if env_ids.numel() == 0:
        return
    if not hasattr(env, "terrain_friction"):
        env.terrain_friction = torch.ones(env.num_envs, device=env.device)
    desired = friction.to(device=env.terrain_friction.device, dtype=env.terrain_friction.dtype)
    changed = ~torch.isclose(
        env.terrain_friction[env_ids], desired, rtol=0.0, atol=1.0e-6
    )
    env.terrain_friction[env_ids] = desired
    if not torch.any(changed):
        return
    env_ids = env_ids[changed]
    friction = friction[changed]

    ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    samples_cpu = friction.detach().to(device="cpu")
    for asset_name in ("robot", "rickshaw"):
        view = env.scene[asset_name].root_physx_view
        materials = view.get_material_properties().clone()
        values = samples_cpu.to(dtype=materials.dtype)[:, None]
        materials[ids_cpu, :, 0] = values
        materials[ids_cpu, :, 1] = values
        view.set_material_properties(materials, ids_cpu)


def _update_rickshaw_mass_properties(
    env: Any,
    env_ids: torch.Tensor,
    sampled: Mapping[str, torch.Tensor],
    calibration: Mapping[str, Any],
) -> None:
    from g1_rickshaw_lab.assets.rickshaw import (
        HITCH_X,
        HITCH_Z,
        RICKSHAW_CENTER_OF_MASS,
        RICKSHAW_TOTAL_MASS,
        RICKSHAW_URDF_SPEC,
        WHEEL_RADIUS,
    )

    device = env.device
    dtype = torch.float32
    if not hasattr(env, "_payload_mass"):
        env._payload_mass = torch.zeros(env.num_envs, device=device, dtype=dtype)
        env._payload_com = torch.zeros((env.num_envs, 3), device=device, dtype=dtype)
        env._wheel_damping = torch.full(
            (env.num_envs, 2),
            RICKSHAW_URDF_SPEC.wheel_joint_damping,
            device=device,
            dtype=dtype,
        )
    env._payload_mass[env_ids] = sampled["payload.mass"]
    env._payload_com[env_ids, 0] = sampled["payload.com.x"]
    env._payload_com[env_ids, 1] = sampled["payload.com.y"]
    env._payload_com[env_ids, 2] = sampled["payload.com.z"]
    env._wheel_damping[env_ids, 0] = sampled["wheel.left_damping"]
    env._wheel_damping[env_ids, 1] = sampled["wheel.right_damping"]
    _write_payload_to_physx(
        env,
        env_ids,
        env._payload_mass[env_ids],
        env._payload_com[env_ids],
    )

    base_mass = torch.full((env.num_envs,), RICKSHAW_TOTAL_MASS, device=device, dtype=dtype)
    base_com = torch.tensor(RICKSHAW_CENTER_OF_MASS, device=device, dtype=dtype)
    base_com[2] -= WHEEL_RADIUS
    base_pitch_inertia = torch.full(
        (env.num_envs,),
        float(calibration["rickshaw.pitch_inertia_about_axle"]),
        device=device,
        dtype=dtype,
    )
    total_mass = base_mass + env._payload_mass
    payload_com_from_axle = env._payload_com.clone()
    payload_com_from_axle[:, 2] -= WHEEL_RADIUS
    total_com = (
        base_mass[:, None] * base_com[None, :]
        + env._payload_mass[:, None] * payload_com_from_axle
    ) / torch.clamp(total_mass[:, None], min=1.0e-6)
    payload_pitch_offset = torch.square(payload_com_from_axle[:, 0]) + torch.square(
        payload_com_from_axle[:, 2]
    )
    pitch_inertia = base_pitch_inertia + env._payload_mass * payload_pitch_offset
    wheel_radius = torch.full((env.num_envs, 2), WHEEL_RADIUS, device=device, dtype=dtype)
    wheel_spin_inertia = torch.full(
        (env.num_envs, 2),
        RICKSHAW_URDF_SPEC.wheel_inertia_diagonal[1],
        device=device,
        dtype=dtype,
    )
    handle = torch.tensor(
        (HITCH_X, 0.0, HITCH_Z - WHEEL_RADIUS), device=device, dtype=dtype
    )
    env.rickshaw_mass_properties = RickshawMassProperties(
        m_cart=total_mass,
        com_x_from_axle=total_com[:, 0],
        com_z_from_axle=total_com[:, 2],
        pitch_inertia_about_axle=pitch_inertia,
        m_eff=effective_cart_mass(total_mass, wheel_spin_inertia, wheel_radius),
        b_eff=effective_wheel_damping(env._wheel_damping, wheel_radius),
        handle_x_from_axle=handle[0].expand(env.num_envs),
        handle_z_from_axle=handle[2].expand(env.num_envs),
    )


def _publish_curriculum_state(env: Any) -> None:
    state: CurriculumRuntimeState = env.curriculum_runtime_state
    env.curriculum_stage = state.stage
    if hasattr(env, "extras"):
        env.extras["curriculum"] = {
            "iteration": state.iteration,
            "stage": state.stage.name,
            "distribution": state.distribution(),
        }


def initialize_curriculum_runtime(
    env: Any,
    env_ids: Any,
    cfg: RuntimeRandomizationCfg,
) -> None:
    """Startup EventTerm installing explicit schedule and delay state."""

    del env_ids  # Startup state is allocated for every environment at once.
    cfg.validate()
    if hasattr(env, "curriculum_runtime_state"):
        return
    terrain = env.scene.terrain
    terrain_types = terrain.terrain_types.to(device=env.device, dtype=torch.long)
    terrain_direction = torch.sign(env.slope).to(dtype=torch.long)
    state = CurriculumRuntimeState.create(
        terrain_types, terrain_direction, cfg.curriculum
    )
    env.curriculum_runtime_state = state
    env.curriculum_stage_per_env = state.stage_per_environment()
    env._curriculum_iteration_explicit = False

    step_dt = float(env.step_dt)
    max_control_delay = 0
    max_observation_delay = 0
    if cfg.sample_ranges:
        max_control_delay = int(
            math.ceil(float(cfg.ranges["control.delay"][1]) / step_dt - 1.0e-9)
        )
        max_observation_delay = int(
            math.ceil(float(cfg.ranges["observation.delay"][1]) / step_dt - 1.0e-9)
        )
    env.max_control_delay_steps = max_control_delay
    env.control_delay_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.observation_delay_steps = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )
    env.motor_strength = torch.ones(env.num_envs, device=env.device)
    env.joint_model_error = torch.zeros(
        (env.num_envs, ACTION_DIM), device=env.device
    )
    env.observation_noise_scale = torch.zeros(env.num_envs, device=env.device)
    from .observations import ObservationDelayState, ObservationNoiseCfg

    env.observation_delay_state = ObservationDelayState.zeros(
        env.num_envs, max_observation_delay, device=env.device
    )
    env.actor_observation_noise_cfg = ObservationNoiseCfg()

    def set_iteration(self: Any, iteration: int) -> CurriculumStage:
        self._curriculum_iteration_explicit = True
        result = self.curriculum_runtime_state.set_iteration(iteration)
        _publish_curriculum_state(self)
        return result

    def get_distribution(self: Any) -> dict[str, int]:
        return self.curriculum_runtime_state.distribution()

    env.set_curriculum_iteration = MethodType(set_iteration, env)
    env.get_curriculum_distribution = MethodType(get_distribution, env)
    _publish_curriculum_state(env)


def update_curriculum_iteration_from_steps(env: Any) -> CurriculumStage:
    """Fallback synchronization for launchers without the RSL runner hook."""

    state = getattr(env, "curriculum_runtime_state", None)
    if state is None:
        raise RuntimeError("curriculum runtime was not initialized at startup")
    if not env._curriculum_iteration_explicit:
        iteration = int(env.common_step_counter) // state.cfg.rollout_steps_per_iteration
        state.set_iteration(iteration)
        _publish_curriculum_state(env)
    return state.stage


def _quantize_delay_samples(
    sampled_seconds: torch.Tensor, step_dt: float, max_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = torch.round(sampled_seconds / step_dt).to(dtype=torch.long)
    steps = torch.clamp(steps, min=0, max=max_steps)
    actual_seconds = steps.to(dtype=sampled_seconds.dtype) * step_dt
    return steps, actual_seconds


def _write_actuator_parameters(
    env: Any,
    env_ids: torch.Tensor,
    motor_strength: torch.Tensor,
    joint_model_error: torch.Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    robot = env.scene["robot"]
    num_envs = int(getattr(env, "num_envs", robot.data.joint_stiffness.shape[0]))
    action_dim = int(joint_model_error.shape[-1])
    if not hasattr(env, "_motor_strength_written"):
        env._motor_strength_written = torch.full(
            (num_envs,), float("nan"), device=env.device
        )
        env._joint_model_error_written = torch.full(
            (num_envs, action_dim), float("nan"), device=env.device
        )
    desired_motor = motor_strength.to(
        device=env._motor_strength_written.device,
        dtype=env._motor_strength_written.dtype,
    )
    desired_joint_error = joint_model_error.to(
        device=env._joint_model_error_written.device,
        dtype=env._joint_model_error_written.dtype,
    )
    unchanged = torch.isclose(
        env._motor_strength_written[env_ids],
        desired_motor,
        rtol=0.0,
        atol=1.0e-6,
    ) & torch.all(
        torch.isclose(
            env._joint_model_error_written[env_ids],
            desired_joint_error,
            rtol=0.0,
            atol=1.0e-6,
        ),
        dim=-1,
    )
    if torch.all(unchanged):
        return
    changed = ~unchanged
    env_ids = env_ids[changed]
    motor_strength = motor_strength[changed]
    joint_model_error = joint_model_error[changed]
    desired_motor = desired_motor[changed]
    desired_joint_error = desired_joint_error[changed]

    joint_ids = env.policy_joint_ids
    if not hasattr(env, "_nominal_robot_joint_stiffness"):
        # Cache solver-side values, which remain zero for explicit actuators;
        # their PD gains are randomized separately in the actuator model below.
        env._nominal_robot_joint_stiffness = robot.data.joint_stiffness.clone()
        env._nominal_robot_joint_damping = robot.data.joint_damping.clone()
        env._nominal_robot_joint_effort_limit = robot.data.joint_effort_limits.clone()
        # PhysX limits are intentionally permissive for explicit DCMotor groups.
        # Bind policy joints to their actuator-model limits so simulation,
        # randomization, and all torque-ratio safety gates share one denominator.
        env._nominal_robot_joint_effort_limit[:, joint_ids] = actuator_effort_limits(
            robot, joint_ids
        )
        env._nominal_actuator_parameters = {
            name: {
                parameter: getattr(actuator, parameter).clone()
                for parameter in (
                    "stiffness",
                    "damping",
                    "effort_limit",
                    "_saturation_effort",
                )
                if torch.is_tensor(getattr(actuator, parameter, None))
            }
            for name, actuator in robot.actuators.items()
        }
    ids = env_ids[:, None]
    gain = motor_strength[:, None] * (1.0 + joint_model_error)
    stiffness = env._nominal_robot_joint_stiffness[ids, joint_ids] * gain
    damping = env._nominal_robot_joint_damping[ids, joint_ids] * gain
    effort_limit = env._nominal_robot_joint_effort_limit[ids, joint_ids] * gain
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_effort_limit_to_sim(
        effort_limit, joint_ids=joint_ids, env_ids=env_ids
    )
    policy_stiffness = stiffness.clone()

    policy_index_by_joint = {
        int(joint_id): index for index, joint_id in enumerate(env.policy_joint_ids)
    }
    for name, actuator in robot.actuators.items():
        actuator_joint_ids = actuator.joint_indices
        if isinstance(actuator_joint_ids, slice):
            actuator_joint_ids = torch.arange(robot.num_joints, device=env.device)[
                actuator_joint_ids
            ]
        policy_indices = [
            policy_index_by_joint.get(int(joint_id), -1)
            for joint_id in actuator_joint_ids
        ]
        selected_local = [index for index, value in enumerate(policy_indices) if value >= 0]
        if not selected_local:
            continue
        selected_policy = torch.tensor(
            [policy_indices[index] for index in selected_local],
            device=env.device,
            dtype=torch.long,
        )
        local_gain = motor_strength[:, None] * (
            1.0 + joint_model_error[:, selected_policy]
        )
        nominal = env._nominal_actuator_parameters[name]
        for parameter in ("stiffness", "damping", "effort_limit", "_saturation_effort"):
            if parameter not in nominal:
                continue
            target = getattr(actuator, parameter)
            target[ids, selected_local] = nominal[parameter][ids, selected_local] * local_gain
        if "stiffness" in nominal:
            policy_stiffness[:, selected_policy] = actuator.stiffness[
                ids, selected_local
            ]

    if not hasattr(env, "policy_joint_stiffness"):
        env.policy_joint_stiffness = torch.zeros(
            (robot.data.joint_stiffness.shape[0], len(joint_ids)),
            device=env.device,
            dtype=stiffness.dtype,
        )
    env.policy_joint_stiffness[env_ids] = policy_stiffness

    actual_effort_limit = actuator_effort_limits(robot, joint_ids)[env_ids]
    if not torch.allclose(actual_effort_limit, effort_limit, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError("actuator and PhysX effort limits diverged")
    env._motor_strength_written[env_ids] = desired_motor
    env._joint_model_error_written[env_ids] = desired_joint_error


def sample_episode_physics(
    env: Any,
    env_ids: torch.Tensor,
    cfg: RuntimeRandomizationCfg,
) -> None:
    """Reset-time EventTerm for the single training distribution."""

    cfg.validate()
    stage_per_env = env.curriculum_runtime_state.stage_per_environment()
    env.curriculum_stage_per_env[env_ids] = stage_per_env[env_ids]
    configured_values = (
        {name: float(cfg.ranges[name][0]) for name in cfg.teacher_extrinsic_names}
        if cfg.sample_ranges
        else {name: float(cfg.nominal_values[name]) for name in cfg.teacher_extrinsic_names}
    )
    sampled = {
        name: torch.full(
            (env_ids.numel(),), value, device=env.device, dtype=torch.float32
        )
        for name, value in configured_values.items()
    }

    joint_error_value = float(
        cfg.ranges["joint.model_error"][0]
        if cfg.sample_ranges
        else cfg.nominal_values["joint.model_error"]
    )
    joint_error = torch.full(
        (env_ids.numel(), ACTION_DIM),
        joint_error_value,
        device=env.device,
        dtype=torch.float32,
    )

    control_steps, sampled["control.delay"] = _quantize_delay_samples(
        sampled["control.delay"], float(env.step_dt), env.max_control_delay_steps
    )
    observation_steps, sampled["observation.delay"] = _quantize_delay_samples(
        sampled["observation.delay"],
        float(env.step_dt),
        env.observation_delay_state.max_delay_steps,
    )
    env.control_delay_steps[env_ids] = control_steps
    env.observation_delay_steps[env_ids] = observation_steps
    env.observation_noise_scale[env_ids] = 0.0
    env.observation_delay_state.reset(env_ids)

    env.c_rr[env_ids] = sampled["rolling_resistance.c_rr"]
    _update_rickshaw_mass_properties(env, env_ids, sampled, cfg.calibration)
    update_slope_frame(env, env_ids)
    env.motor_strength[env_ids] = sampled["motor.strength"]
    env.joint_model_error[env_ids] = joint_error
    _write_effective_terrain_friction_to_physx(
        env, env_ids, sampled["terrain.friction"]
    )
    _update_teacher_extrinsics(env, env_ids, cfg, sampled)
    _write_actuator_parameters(
        env, env_ids, sampled["motor.strength"], joint_error
    )

    wheel_joint_ids = _ensure_wheel_joint_ids(env)
    wheel_damping = torch.stack(
        (sampled["wheel.left_damping"], sampled["wheel.right_damping"]), dim=-1
    )
    if not hasattr(env, "_wheel_damping_written"):
        env._wheel_damping_written = torch.full(
            (env.num_envs, 2), float("nan"), device=env.device
        )
    desired_wheel_damping = wheel_damping.to(
        device=env._wheel_damping_written.device,
        dtype=env._wheel_damping_written.dtype,
    )
    wheel_changed = torch.any(
        ~torch.isclose(
            env._wheel_damping_written[env_ids],
            desired_wheel_damping,
            rtol=0.0,
            atol=1.0e-6,
        ),
        dim=-1,
    )
    if torch.any(wheel_changed):
        wheel_env_ids = env_ids[wheel_changed]
        env.scene["rickshaw"].write_joint_damping_to_sim(
            wheel_damping[wheel_changed],
            joint_ids=wheel_joint_ids,
            env_ids=wheel_env_ids,
        )
        env._wheel_damping_written[wheel_env_ids] = desired_wheel_damping[wheel_changed]


@dataclass(kw_only=True)
class HandleConstraintCfg:
    """Fully calibrated double-D6 definition.

    No drive, limit, frame, or safety value is inferred here.  Rotation axes
    marked free receive neither a limit nor a drive API.
    """

    robot_body_paths: tuple[str, str] = MISSING
    hitch_body_paths: tuple[str, str] = MISSING
    grasp_local_positions: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    grasp_local_quaternions_wxyz: tuple[
        tuple[float, float, float, float], tuple[float, float, float, float]
    ] = MISSING
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
        for name, value in (
            ("linear_stiffness", self.linear_stiffness),
            ("linear_damping", self.linear_damping),
            ("angular_stiffness", self.angular_stiffness),
            ("angular_damping", self.angular_damping),
            ("max_force", self.max_force),
            ("max_torque", self.max_torque),
            ("linear_limit", self.linear_limit),
            ("angular_limit", self.angular_limit),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a calibrated positive finite value")
        if len(self.rotation_free_axes) != 3 or len(self.rotation_driven_axes) != 3:
            raise ValueError("rotation axis modes must have exactly three entries")
        for free, driven in zip(self.rotation_free_axes, self.rotation_driven_axes, strict=True):
            if free and driven:
                raise ValueError("a physically free D6 rotation axis cannot have a drive")
        if not isinstance(self.reaction_is_joint_on_robot, bool):
            raise ValueError("reaction_is_joint_on_robot must explicitly define the PhysX sign")
        for quaternion in self.grasp_local_quaternions_wxyz:
            norm = math.sqrt(sum(component * component for component in quaternion))
            if abs(norm - 1.0) > 1.0e-4:
                raise ValueError("calibrated grasp-local quaternion must be unit length")


class D6ConstraintManager:
    """Runtime metadata for source-authored, scene-replicated D6 joints."""

    _SIDES = ("left", "right")

    def __init__(self, env: Any, cfg: HandleConstraintCfg):
        cfg.validate()
        self.env = env
        self.cfg = cfg
        self.created = False
        self.joint_paths: list[list[str]] = []

    @staticmethod
    def _stage_from_env(env: Any) -> Any:
        if hasattr(env.scene, "stage"):
            return env.scene.stage
        try:
            import omni.usd
        except ImportError as error:  # pragma: no cover - requires Isaac Sim.
            raise RuntimeError("D6 startup requires Isaac Sim's omni.usd module") from error
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable during D6 startup")
        return stage

    def bind_existing(self) -> None:
        """Bind paths after the source D6 prims have been replicated by the scene."""

        if self.created:
            raise RuntimeError("D6 constraints may only be bound once at startup")
        stage = self._stage_from_env(self.env)
        source_namespace = self.cfg.env_prim_path_template.format(env_id=0)
        source_paths = [
            self.cfg.joint_prim_path_template.format(
                ENV_NS=source_namespace, side=side, env_id=0
            )
            for side in self._SIDES
        ]
        for joint_path in source_paths:
            if not stage.GetPrimAtPath(joint_path).IsValid():
                raise RuntimeError(
                    f"replicated D6 source prim does not exist: {joint_path}"
                )
        self.joint_paths = [
            [
                self.cfg.joint_prim_path_template.format(
                    ENV_NS=self.cfg.env_prim_path_template.format(env_id=env_id),
                    side=side,
                    env_id=env_id,
                )
                for side in self._SIDES
            ]
            for env_id in range(self.env.num_envs)
        ]
        self.created = True

class D6ReactionResidualAdapter:
    """Version-independent adapter around a local D6 constraint proxy reader.

    The provider returns a world-frame proxy wrench ``[N,2,6]``, position
    residual ``[N,2,3]``, rotation residual ``[N,2,3]``, and spatial impulse
    ``[N,2,6]``. The proxy is retained for conservative D6 load/residual safety;
    whole-cart momentum balance independently measures the physical hand force.
    """

    def __init__(self, env: Any, manager: D6ConstraintManager):
        self.env = env
        self.manager = manager

    def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        provider = getattr(self.env, "d6_reaction_residual_provider", None)
        if provider is None:
            raise RuntimeError(
                "set env.d6_reaction_residual_provider to the validated Isaac-Sim-version "
                "D6 tensor adapter before training"
            )
        raw_wrench, position_residual, rotation_residual, impulse = provider(
            self.manager.joint_paths
        )
        if raw_wrench.shape != (self.env.num_envs, 2, 6):
            raise ValueError("D6 provider wrench must have shape [N,2,6]")
        if position_residual.shape != (self.env.num_envs, 2, 3):
            raise ValueError("D6 position residual must have shape [N,2,3]")
        if rotation_residual.shape != (self.env.num_envs, 2, 3):
            raise ValueError("D6 rotation residual must have shape [N,2,3]")
        if impulse.shape != (self.env.num_envs, 2, 6):
            raise ValueError("D6 spatial impulse must have shape [N,2,6]")
        wrench = adapt_d6_reaction_wrench(
            raw_wrench,
            reaction_is_joint_on_body=self.manager.cfg.reaction_is_joint_on_robot,
        )
        residual = torch.maximum(
            torch.linalg.vector_norm(position_residual, dim=-1),
            torch.linalg.vector_norm(rotation_residual, dim=-1),
        ).amax(dim=-1)
        impulse_magnitude = d6_spatial_impulse_magnitudes(impulse)
        state = self.env.rickshaw_state
        # Excluded external D6 joints are not part of either articulation's
        # tensor joint view. The retained-link incoming wrench is kept only as
        # a conservative per-side constraint proxy; whole-cart momentum balance
        # owns the physical hand force used by FAT2, ZMP, and observations.
        self.env.d6_incoming_joint_proxy_w[:] = wrench
        state.d6_residual[:] = residual
        state.d6_impulse[:] = impulse_magnitude
        return wrench, residual, impulse_magnitude


def d6_spatial_impulse_magnitudes(impulse: torch.Tensor) -> torch.Tensor:
    """Reduce ``[N,2,6]`` spatial impulses to linear/angular safety channels."""

    if impulse.ndim != 3 or impulse.shape[1:] != (2, 6):
        raise ValueError("D6 spatial impulse must have shape [N,2,6]")
    linear = torch.linalg.vector_norm(impulse[..., :3], dim=-1).amax(dim=-1)
    angular = torch.linalg.vector_norm(impulse[..., 3:], dim=-1).amax(dim=-1)
    return torch.stack((linear, angular), dim=-1)


def _axis_angle_from_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the shortest batched rotation vector for scalar-first quaternions."""

    quaternion = quaternion / torch.clamp(
        torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1.0e-12
    )
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, torch.clamp(quaternion[..., :1], min=0.0))
    scale = torch.where(sin_half > 1.0e-7, angle / sin_half, torch.full_like(sin_half, 2.0))
    return vector * scale


def recover_d6_wrench_on_robot(
    incoming_force_w: torch.Tensor,
    incoming_torque_w: torch.Tensor,
    mass: torch.Tensor,
    gravity_w: torch.Tensor,
    linear_acceleration_w: torch.Tensor,
    inertial_torque_w: torch.Tensor,
) -> torch.Tensor:
    """Map the retained hitch-link incoming wrench to a robot-side proxy."""

    force_w = incoming_force_w + mass * gravity_w - mass * linear_acceleration_w
    torque_w = incoming_torque_w - inertial_torque_w
    return torch.cat((force_w, torque_w), dim=-1)


class IsaacSimD6ReactionProvider:
    """Isaac Sim 5.1 D6 residual/impulse proxy from retained hitch links.

    PhysX does not expose excluded external D6 joints through a tensor joint
    view. The incoming fixed-joint wrench is useful as a conservative local
    constraint-load proxy, but it does not close whole-cart momentum balance
    and is therefore not used as the physical hand force.
    """

    def __init__(self, env: Any, manager: D6ConstraintManager):
        self.env = env
        self.manager = manager

    def __call__(
        self, joint_paths: list[list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if joint_paths != self.manager.joint_paths:
            raise ValueError("D6 provider received paths outside its constraint manager")

        cart = self.env.scene["rickshaw"]
        robot = self.env.scene["robot"]
        hitch_ids = self.env.hitch_body_ids
        grasp_ids = self.env.grasp_body_ids

        hitch_position_w = cart.data.body_pos_w[:, hitch_ids]
        hitch_quaternion_w = cart.data.body_quat_w[:, hitch_ids]
        parent_quaternion_w = cart.data.body_quat_w[:, :1].expand_as(hitch_quaternion_w)

        incoming_wrench_b = cart.data.body_incoming_joint_wrench_b[:, hitch_ids]
        incoming_force_w = quat_apply_wxyz(parent_quaternion_w, incoming_wrench_b[..., :3])
        # PhysX reports the spatial wrench at the child joint frame.  The
        # fixed hitch joint has identity local rotation, so only rotate it to
        # world; translating it by the base-to-hitch lever arm would double
        # count r x F.
        incoming_torque_hitch_w = quat_apply_wxyz(
            parent_quaternion_w, incoming_wrench_b[..., 3:]
        )

        mass = cart.data.default_mass[:, hitch_ids, None].to(
            device=incoming_force_w.device, dtype=incoming_force_w.dtype
        )
        acceleration_w = cart.data.body_com_acc_w[:, hitch_ids]
        gravity_w = torch.tensor(
            self.env.cfg.sim.gravity,
            device=incoming_force_w.device,
            dtype=incoming_force_w.dtype,
        ).view(1, 1, 3)
        inertia_b = (
            cart.data.default_inertia[:, hitch_ids]
            .to(device=incoming_force_w.device, dtype=incoming_force_w.dtype)
            .reshape(self.env.num_envs, 2, 3, 3)
        )
        angular_velocity_b = quat_apply_wxyz(
            torch.cat((hitch_quaternion_w[..., :1], -hitch_quaternion_w[..., 1:]), dim=-1),
            cart.data.body_com_vel_w[:, hitch_ids, 3:],
        )
        angular_acceleration_b = quat_apply_wxyz(
            torch.cat((hitch_quaternion_w[..., :1], -hitch_quaternion_w[..., 1:]), dim=-1),
            acceleration_w[..., 3:],
        )
        angular_momentum_b = torch.matmul(inertia_b, angular_velocity_b[..., None]).squeeze(-1)
        inertial_torque_b = (
            torch.matmul(inertia_b, angular_acceleration_b[..., None]).squeeze(-1)
            + torch.linalg.cross(angular_velocity_b, angular_momentum_b)
        )
        # Convert the retained-link incoming load to the historical robot-side
        # proxy convention. Whole-cart momentum balance supplies physical force.
        raw_wrench_on_robot_w = recover_d6_wrench_on_robot(
            incoming_force_w,
            incoming_torque_hitch_w,
            mass,
            gravity_w,
            acceleration_w[..., :3],
            quat_apply_wxyz(hitch_quaternion_w, inertial_torque_b),
        )

        grasp_body_position_w = robot.data.body_pos_w[:, grasp_ids]
        grasp_body_quaternion_w = robot.data.body_quat_w[:, grasp_ids]
        grasp_local_position = torch.tensor(
            self.manager.cfg.grasp_local_positions,
            device=grasp_body_position_w.device,
            dtype=grasp_body_position_w.dtype,
        ).view(1, 2, 3)
        grasp_local_quaternion = torch.tensor(
            self.manager.cfg.grasp_local_quaternions_wxyz,
            device=grasp_body_quaternion_w.device,
            dtype=grasp_body_quaternion_w.dtype,
        ).view(1, 2, 4)
        grasp_frame_position_w = grasp_body_position_w + quat_apply_wxyz(
            grasp_body_quaternion_w, grasp_local_position
        )
        grasp_frame_quaternion_w = quat_multiply_wxyz(
            grasp_body_quaternion_w,
            grasp_local_quaternion.expand_as(grasp_body_quaternion_w),
        )
        hitch_conjugate = torch.cat(
            (hitch_quaternion_w[..., :1], -hitch_quaternion_w[..., 1:]), dim=-1
        )
        relative_quaternion = quat_multiply_wxyz(
            hitch_conjugate, grasp_frame_quaternion_w
        )
        rotation_residual = _axis_angle_from_quaternion_wxyz(relative_quaternion)
        free_axes = torch.tensor(
            self.manager.cfg.rotation_free_axes,
            device=rotation_residual.device,
            dtype=torch.bool,
        ).view(1, 1, 3)
        rotation_residual = torch.where(free_axes, torch.zeros_like(rotation_residual), rotation_residual)
        position_residual = grasp_frame_position_w - hitch_position_w

        # PhysX reports a spatial wrench; retain both linear and angular
        # per-substep impulses so neither D6 safety channel can be hidden.
        impulse = raw_wrench_on_robot_w * float(self.env.physics_dt)
        return raw_wrench_on_robot_w, position_residual, rotation_residual, impulse


def bind_d6_runtime_adapters(env: Any) -> D6ReactionResidualAdapter:
    """Startup binding after PhysX handles are active."""

    manager = getattr(env, "d6_constraint_manager", None)
    if manager is None or not manager.created:
        raise RuntimeError(
            "replicated D6 prims must be bound before installing runtime adapters"
        )
    if hasattr(env, "d6_reaction_adapter"):
        raise RuntimeError("D6 runtime adapter is already bound")
    if not hasattr(env, "d6_reaction_residual_provider"):
        env.d6_reaction_residual_provider = IsaacSimD6ReactionProvider(env, manager)
    adapter = D6ReactionResidualAdapter(env, manager)
    env.d6_reaction_adapter = adapter

    def read_reaction(self: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.d6_reaction_adapter.read()
    env.read_d6_reaction_residual = MethodType(read_reaction, env)
    return adapter


@dataclass(kw_only=True)
class ResetValidationCfg:
    """Physical endpoint tolerances used by the direct reset writer."""

    hand_position_tolerance: float = MISSING
    minimum_wheel_normal_force: float = MISSING

    def validate(self) -> None:
        for name in ("hand_position_tolerance", "minimum_wheel_normal_force"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"reset validation {name} must be positive and finite")


@dataclass(kw_only=True)
class TaskEntityNamesCfg:
    """Persisted exact entity order; no regex ordering is performed at runtime."""

    policy_joint_names: tuple[str, ...] = MISSING
    arm_joint_names: tuple[str, ...] = MISSING
    dex_joint_names: tuple[str, ...] = MISSING
    wheel_body_names: tuple[str, str] = MISSING
    hitch_body_names: tuple[str, str] = MISSING
    foot_body_names: tuple[str, str] = MISSING
    torso_body_name: str = MISSING

    def validate(self) -> None:
        expected = {
            "policy_joint_names": (self.policy_joint_names, 29),
            "arm_joint_names": (self.arm_joint_names, 14),
            "dex_joint_names": (self.dex_joint_names, 4),
            "wheel_body_names": (self.wheel_body_names, 2),
            "hitch_body_names": (self.hitch_body_names, 2),
            "foot_body_names": (self.foot_body_names, 2),
        }
        for name, (values, count) in expected.items():
            if len(values) != count or len(set(values)) != count:
                raise ValueError(f"{name} must contain {count} unique persisted names")
        if not set(self.arm_joint_names).issubset(self.policy_joint_names):
            raise ValueError("arm joints must be a subset of the 29 policy joints")
        if set(self.dex_joint_names) & set(self.policy_joint_names):
            raise ValueError("Dex joints must be excluded from the policy action")


def _exact_name_ids(available: list[str] | tuple[str, ...], requested: tuple[str, ...], label: str) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
        raise RuntimeError(f"missing {label} names: {missing}")
    return [index_by_name[name] for name in requested]


def resolve_task_entities(env: Any, cfg: TaskEntityNamesCfg) -> None:
    """Startup event that installs all fixed joint/body/sensor indices."""

    cfg.validate()
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    robot_joint_names = tuple(robot.joint_names)
    robot_body_names = tuple(robot.body_names)
    cart_body_names = tuple(cart.body_names)
    env.policy_joint_names = cfg.policy_joint_names
    env.policy_joint_ids = _exact_name_ids(
        robot_joint_names, cfg.policy_joint_names, "policy joint"
    )
    env.arm_joint_ids = _exact_name_ids(robot_joint_names, cfg.arm_joint_names, "arm joint")
    env.dex_joint_ids = _exact_name_ids(robot_joint_names, cfg.dex_joint_names, "Dex joint")
    env.wheel_body_ids = _exact_name_ids(cart_body_names, cfg.wheel_body_names, "wheel body")
    env.hitch_body_ids = _exact_name_ids(cart_body_names, cfg.hitch_body_names, "hitch body")
    env.foot_body_ids = _exact_name_ids(robot_body_names, cfg.foot_body_names, "foot body")
    env.torso_body_id = _exact_name_ids(
        robot_body_names, (cfg.torso_body_name,), "torso body"
    )[0]
    wheel_sensor = env.scene["wheel_contacts"]
    robot_sensor = env.scene["robot_contacts"]
    env.wheel_sensor_ids = _exact_name_ids(
        tuple(wheel_sensor.body_names), cfg.wheel_body_names, "wheel sensor body"
    )
    env.foot_sensor_ids = _exact_name_ids(
        tuple(robot_sensor.body_names), cfg.foot_body_names, "foot sensor body"
    )
    robot_masses = robot.root_physx_view.get_masses().to(device=env.device)
    if robot_masses.shape != (env.num_envs, len(robot_body_names)):
        raise RuntimeError(
            "robot PhysX mass tensor shape differs from the resolved body-name contract"
        )
    if torch.any(~torch.isfinite(robot_masses)) or torch.any(robot_masses <= 0.0):
        raise RuntimeError("every retained G1 body must have finite positive PhysX mass")
    expected_mass = float(getattr(env.cfg, "robot_mass"))
    actual_mass = torch.sum(robot_masses, dim=-1)
    if not torch.allclose(
        actual_mass,
        torch.full_like(actual_mass, expected_mass),
        rtol=0.0,
        atol=max(1.0e-4, expected_mass * 1.0e-5),
    ):
        actual = (float(torch.amin(actual_mass)), float(torch.amax(actual_mass)))
        raise RuntimeError(
            f"G1 USD mass contract failed: expected {expected_mass}, got range {actual}"
        )
    env.robot_body_masses = robot_masses
    cart_masses = cart.root_physx_view.get_masses().to(device=env.device)
    if cart_masses.shape != (env.num_envs, len(cart_body_names)):
        raise RuntimeError(
            "rickshaw PhysX mass tensor shape differs from the resolved body-name contract"
        )
    if torch.any(~torch.isfinite(cart_masses)) or torch.any(cart_masses <= 0.0):
        raise RuntimeError("every retained rickshaw body must have finite positive PhysX mass")
    env.rickshaw_body_masses = cart_masses
    env.rickshaw_body_com_pos_b = cart.root_physx_view.get_coms()[..., :3].to(
        device=env.device, dtype=cart_masses.dtype
    )


def _body_name_from_prim_path(path: str) -> str:
    name = path.rstrip("/").split("/")[-1]
    if not name:
        raise ValueError(f"invalid body prim path {path!r}")
    return name


def static_reset_contact_allocation(
    env: Any, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right hand wrenches and wheel forces in the SLN frame."""

    properties = env.rickshaw_mass_properties
    mass = properties.m_cart[env_ids]
    alpha = target_pitch_from_hitch_height(env.rickshaw_pose_cfg)
    cosine = math.cos(alpha)
    sine = math.sin(alpha)
    handle_s = (
        cosine * properties.handle_x_from_axle[env_ids]
        - sine * properties.handle_z_from_axle[env_ids]
    )
    handle_n = (
        sine * properties.handle_x_from_axle[env_ids]
        + cosine * properties.handle_z_from_axle[env_ids]
    )
    if torch.any(handle_s <= 0.5):
        bad = env_ids[handle_s <= 0.5].detach().cpu().tolist()
        raise RuntimeError(f"static handle-force geometry is invalid for env_ids={bad[:16]}")
    com_s = (
        cosine * properties.com_x_from_axle[env_ids]
        - sine * properties.com_z_from_axle[env_ids]
    )
    com_n = (
        sine * properties.com_x_from_axle[env_ids]
        + cosine * properties.com_z_from_axle[env_ids]
    )
    com_l = env._payload_mass[env_ids] * env._payload_com[env_ids, 1] / mass
    hand_rows, wheel_rows = fixed_contact_static_components(
        gravity_tangent=mass * GRAVITY * torch.sin(env.gamma[env_ids]),
        gravity_normal=mass * GRAVITY * torch.cos(env.gamma[env_ids]),
        com_s=com_s,
        com_l=com_l,
        com_n=com_n,
        handle_s=handle_s,
        handle_n=handle_n,
        hitch_half_width=env.rickshaw_pose_cfg.hitch_half_width,
        wheel_track=WHEEL_TRACK,
    )
    hand_wrenches_sln = torch.stack(
        tuple(torch.stack(row, dim=-1) for row in hand_rows), dim=1
    )
    wheel_forces_sln = torch.stack(
        tuple(torch.stack(row, dim=-1) for row in wheel_rows), dim=1
    )
    return hand_wrenches_sln, wheel_forces_sln


def install_q_ref_from_reset_library(env: Any, env_ids: torch.Tensor) -> None:
    """Install the nominal MuJoCo static state and controller reference directly."""

    if hasattr(env, "reset_pose_gradients"):
        pose_indices = torch.argmin(
            torch.abs(env.slope[env_ids, None] - env.reset_pose_gradients[None, :]),
            dim=1,
        )
        env.reset_pose_index[env_ids] = pose_indices
        q_reset = env.reset_q_reset_table[pose_indices]
        q_ref = env.reset_q_ref_table[pose_indices]
    else:
        library = getattr(env, "reset_pose_library", None)
        if library is None:
            library = getattr(env.cfg, "reset_pose_library", None)
        if library is None or not hasattr(library, "pose_for_gradient"):
            raise RuntimeError("a schema-v4 reset pose library is required")
        poses = [
            library.pose_for_gradient(round(float(value), 2))
            for value in env.slope[env_ids].detach().cpu()
        ]
        device = env.action_state.q_ref.device
        dtype = env.action_state.q_ref.dtype
        q_reset = torch.tensor(
            [pose.q_reset for pose in poses], device=device, dtype=dtype
        )
        q_ref = torch.tensor(
            [pose.q_ref for pose in poses], device=device, dtype=dtype
        )
    env.action_state.q_ref[env_ids] = q_ref
    env.reset_policy_joint_pos[env_ids] = q_reset


def _compile_reset_pose_tables(env: Any) -> None:
    """Move the immutable configured-slope reset library to the simulation device once."""

    library = getattr(env, "reset_pose_library", None)
    if library is None:
        library = getattr(env.cfg, "reset_pose_library", None)
    poses = getattr(library, "poses", None)
    if poses is None:
        raise RuntimeError("a schema-v4 reset pose library is required")
    device = env.device
    dtype = env.action_state.q_ref.dtype
    env.reset_pose_gradients = torch.tensor(
        [pose.gradient for pose in poses], device=device, dtype=dtype
    )
    env.reset_q_reset_table = torch.tensor(
        [pose.q_reset for pose in poses], device=device, dtype=dtype
    )
    env.reset_q_ref_table = torch.tensor(
        [pose.q_ref for pose in poses], device=device, dtype=dtype
    )
    env.reset_root_pitch_table = torch.tensor(
        [pose.root_pitch for pose in poses], device=device, dtype=dtype
    )
    env.reset_root_height_table = torch.tensor(
        [pose.root_height for pose in poses], device=device, dtype=dtype
    )
    env.reset_handle_wrenches_sln_table = torch.tensor(
        [pose.handle_wrenches_sln for pose in poses], device=device, dtype=dtype
    )
    env.reset_pose_index = torch.zeros(
        env.num_envs, device=device, dtype=torch.long
    )
    env.reset_grasp_positions_b_table = torch.zeros(
        (len(poses), 2, 3), device=device, dtype=dtype
    )
    env._reset_grasp_cached_indices = set()


def initialize_mdp_state(
    env: Any,
    env_ids: Any,
    handle_constraint_cfg: HandleConstraintCfg,
    rolling_resistance_cfg: RollingResistanceCfg,
    entity_names_cfg: TaskEntityNamesCfg,
    rickshaw_pose_cfg: RickshawPoseTargetCfg,
    robot_mass: float,
    dex_q_grasp: tuple[float, float, float, float],
) -> None:
    """Startup event that allocates state, creates D6 joints, and binds hooks.

    All three physical configurations must come from the validated environment
    config.  Missing values fail before the first simulation step.
    """

    del env_ids  # Startup state is allocated for every environment at once.
    if robot_mass <= 0.0:
        raise RuntimeError("validated positive robot_mass is required at startup")
    if len(dex_q_grasp) != 4:
        raise RuntimeError("calibrated four-joint dex_q_grasp is required at startup")
    rolling_resistance_cfg.validate()
    if hasattr(env, "d6_constraint_manager"):
        raise RuntimeError("D6 constraint manager is already installed")
    manager = D6ConstraintManager(env, handle_constraint_cfg)
    manager.bind_existing()
    env.d6_constraint_manager = manager
    if not hasattr(env, "_rickshaw_physics_hook_action_term"):
        raise RuntimeError(
            "exactly one FilteredJointPositionActionCfg must set physics_hook_owner=True"
        )
    resolve_task_entities(env, entity_names_cfg)
    env.nominal_policy_effort_limits = actuator_effort_limits(
        env.scene["robot"], env.policy_joint_ids
    ).clone()
    env.arm_effort_limits = actuator_effort_limits(
        env.scene["robot"], env.arm_joint_ids
    ).clone()
    robot_body_names = tuple(env.scene["robot"].body_names)
    env.grasp_body_ids = _exact_name_ids(
        robot_body_names,
        tuple(_body_name_from_prim_path(path) for path in handle_constraint_cfg.robot_body_paths),
        "robot grasp body",
    )
    action_terms = list(env.action_manager._terms.values())
    if len(action_terms) != 3:
        raise RuntimeError("the task requires exactly three ActionTerms totaling 29 joints")
    if not all(hasattr(term, "filter_state") for term in action_terms):
        raise RuntimeError("all three action groups must use FilteredJointPositionAction")
    reference_indices = tuple(
        index
        for term in action_terms
        for index in (term.cfg.reference_indices or ())
    )
    if reference_indices != tuple(range(ACTION_DIM)):
        raise RuntimeError(
            "ActionTerm reference_indices must partition the persisted q_ref order 0..28"
        )
    resolved_action_names = tuple(
        name for term in action_terms for name in term._joint_names
    )
    if resolved_action_names != entity_names_cfg.policy_joint_names:
        raise RuntimeError("ActionTerm joint concatenation differs from persisted checkpoint order")
    resolved_scales: list[torch.Tensor] = []
    for term in action_terms:
        try:
            resolved_scales.append(
                canonicalize_action_scale(
                    term._scale, term.action_dim, env.num_envs, device=env.device
                )
            )
        except ValueError as error:
            raise RuntimeError(
                "ActionTerm scale tensor does not match the term action dimension"
            ) from error
    scale = torch.cat(resolved_scales)
    expected_scale = action_scale_vector(device=env.device, dtype=scale.dtype)
    if not torch.allclose(scale, expected_scale, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("ActionTerm scales differ from the fixed 29-D action contract")

    device = env.device
    num_envs = env.num_envs
    env.command_state = CommandState.zeros(num_envs, device=device)
    env.path_state = PathTrackingState.zeros(num_envs, device=device)
    env.rickshaw_state = RickshawRuntimeState.zeros(num_envs, device=device)
    env.stability_state = StabilityState.zeros(num_envs, device=device)
    env.action_state = ButterworthActionState.create(
        torch.zeros((num_envs, ACTION_DIM), device=device)
    )
    _compile_reset_pose_tables(env)
    env.reset_policy_joint_pos = torch.zeros(
        (num_envs, ACTION_DIM), device=device
    )
    env.c_rr = torch.zeros(num_envs, device=device)
    env.static_d6_preload_offset_w = torch.zeros((num_envs, 3), device=device)
    env.policy_robot_speed_s = torch.zeros(num_envs, device=device)
    env.policy_robot_velocity_n = torch.zeros(num_envs, device=device)
    env.d6_incoming_joint_proxy_w = torch.zeros(
        (num_envs, 2, 6), device=device
    )
    env.rickshaw_pose_cfg = rickshaw_pose_cfg
    env.robot_mass = torch.full((num_envs,), float(robot_mass), device=device)
    env.dex_q_grasp = torch.tensor(dex_q_grasp, device=device, dtype=torch.float32)
    update_slope_frame(env)
    cart = env.scene["rickshaw"]
    initial_v_s = torch.sum(cart.data.root_lin_vel_w * env.path_tangent_w, dim=-1)
    initial_pitch = rickshaw_pitch_from_quaternion(
        cart.data.root_quat_w, env.path_tangent_w, env.path_normal_w
    )
    env.analytic_force_state = AnalyticHandleForceState.initialized(
        initial_v_s, initial_pitch
    )
    _, cart_com_velocity_w, _ = cart_system_mass_kinematics(env)
    env.cart_interaction_wrench_state = CartInteractionWrenchState.initialized(
        cart_com_velocity_w
    )
    env.cart_interaction_wrench_valid = torch.zeros(
        num_envs, device=device, dtype=torch.bool
    )
    from .curricula import TerrainCurriculumState
    from .observations import ObservationHistoryState
    from .terminations import PersistentTerminationState

    history_enabled = getattr(env.cfg.observations, "history", None) is not None
    env.observation_history_state = ObservationHistoryState.zeros(
        num_envs,
        device=device,
        history_enabled=history_enabled,
    )
    env.curriculum_state = TerrainCurriculumState.zeros(num_envs, device=device)
    env.termination_state = PersistentTerminationState.zeros(num_envs, device=device)
    bind_d6_runtime_adapters(env)
    env._rolling_resistance_cfg = rolling_resistance_cfg

    def pre_physics_step(self: Any) -> None:
        rolling_force_w = apply_rolling_resistance(self, self._rolling_resistance_cfg)
        accumulate_cart_interaction_wrench(self, rolling_force_w)

    env._g1_rickshaw_pre_physics_step = MethodType(pre_physics_step, env)
    env.write_closed_chain_reset_state = MethodType(write_closed_chain_reset_state, env)


def _forward_reset_kinematics(env: Any) -> None:
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def spatial_wrenches_sln_to_world(
    wrenches_sln: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Rotate batched SLN force/torque components into world coordinates."""

    if wrenches_sln.ndim != 3 or wrenches_sln.shape[1:] != (2, 6):
        raise ValueError("static handle wrenches must have shape [N,2,6]")
    expected = (wrenches_sln.shape[0], 3)
    if any(axis.shape != expected for axis in (path_tangent_w, path_lateral_w, path_normal_w)):
        raise ValueError("slope-frame axes must each have shape [N,3]")
    basis = torch.stack((path_tangent_w, path_lateral_w, path_normal_w), dim=1)
    force_w = torch.einsum("nsc,ncw->nsw", wrenches_sln[..., :3], basis)
    torque_w = torch.einsum("nsc,ncw->nsw", wrenches_sln[..., 3:], basis)
    return torch.cat((force_w, torque_w), dim=-1)


def grasp_positions_w(env: Any, env_ids: torch.Tensor) -> torch.Tensor:
    """Return both calibrated grasp centers from current robot FK."""

    robot = env.scene["robot"]
    handle_cfg = env.d6_constraint_manager.cfg
    body_position = robot.data.body_pos_w[env_ids][:, env.grasp_body_ids]
    body_quaternion = robot.data.body_quat_w[env_ids][:, env.grasp_body_ids]
    local_position = torch.tensor(
        handle_cfg.grasp_local_positions,
        device=env.device,
        dtype=body_position.dtype,
    ).expand(env_ids.numel(), -1, -1)
    return body_position + quat_apply_wxyz(body_quaternion, local_position)


def _reset_grasp_positions_w(
    env: Any,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
) -> torch.Tensor:
    """Return reset grasp positions, running FK only for unseen slope poses."""

    pose_indices = env.reset_pose_index[env_ids]
    cached_indices: set[int] = env._reset_grasp_cached_indices
    if len(cached_indices) != env.reset_pose_gradients.numel():
        requested = set(pose_indices.detach().cpu().tolist())
        missing = requested - cached_indices
        if missing:
            _forward_reset_kinematics(env)
            actual_grasp_w = grasp_positions_w(env, env_ids)
            root_quat_inverse = root_quat.clone()
            root_quat_inverse[:, 1:] *= -1.0
            local_grasp = quat_apply_wxyz(
                root_quat_inverse[:, None, :].expand(-1, 2, -1),
                actual_grasp_w - root_pos[:, None, :],
            )
            for pose_index in missing:
                sample = torch.nonzero(pose_indices == pose_index, as_tuple=False)[0, 0]
                env.reset_grasp_positions_b_table[pose_index] = local_grasp[sample]
            cached_indices.update(missing)
    local_grasp = env.reset_grasp_positions_b_table[pose_indices]
    return root_pos[:, None, :] + quat_apply_wxyz(
        root_quat[:, None, :].expand(-1, 2, -1), local_grasp
    )


def write_closed_chain_reset_state(env: Any, env_ids: torch.Tensor) -> None:
    """Write calibrated robot/cart reset states and refresh kinematics.

    The reset-pose library owns the 29-D G1 reference and per-slope root pitch.
    The Dex joints use calibrated ``q_grasp``, and the cart root is solved from
    the actual post-forward grasp midpoint and guide-defined hitch geometry.
    """

    if not hasattr(env, "action_state"):
        raise RuntimeError("initialize_mdp_state must run before closed-chain reset")
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    origin = env.scene.terrain.env_origins[env_ids]
    default_root = robot.data.default_root_state[env_ids]
    default_pos = default_root[:, :3]
    pose_indices = env.reset_pose_index[env_ids]
    root_height = env.reset_root_height_table[pose_indices].to(dtype=default_root.dtype)
    root_pos = (
        origin
        + env.path_tangent_w[env_ids] * default_pos[:, 0:1]
        + env.path_lateral_w[env_ids] * default_pos[:, 1:2]
        + env.path_normal_w[env_ids] * root_height[:, None]
    )
    root_pitch = env.reset_root_pitch_table[pose_indices].to(dtype=default_root.dtype)
    pitch_quat = torch.zeros((env_ids.numel(), 4), device=env.device, dtype=default_root.dtype)
    pitch_quat[:, 0] = torch.cos(0.5 * root_pitch)
    pitch_quat[:, 2] = torch.sin(0.5 * root_pitch)
    root_quat = quat_multiply_wxyz(
        quat_multiply_wxyz(env.slope_quat_w[env_ids], pitch_quat),
        default_root[:, 3:7],
    )
    root_velocity = torch.zeros((env_ids.numel(), 6), device=env.device)
    root_pose = torch.cat((root_pos, root_quat), dim=-1)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    q_reset = env.reset_q_reset_table[pose_indices].to(dtype=joint_pos.dtype)
    env.reset_policy_joint_pos[env_ids] = q_reset
    joint_pos[:, env.policy_joint_ids] = q_reset
    joint_pos[:, env.dex_joint_ids] = env.dex_q_grasp[None, :]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    joint_target = joint_pos.clone()
    policy_target = env.action_state.q_ref[env_ids].clone()
    joint_target[:, env.policy_joint_ids] = policy_target
    robot.set_joint_position_target(joint_target, env_ids=env_ids)
    grasp_position = _reset_grasp_positions_w(
        env, env_ids, root_pos, root_quat
    )
    alpha_target = target_pitch_from_hitch_height(env.rickshaw_pose_cfg)
    cart_quat = target_cart_orientation(env.slope_quat_w[env_ids], alpha_target)
    hand_wrenches_sln = env.reset_handle_wrenches_sln_table[pose_indices]
    handle_wrenches_w = spatial_wrenches_sln_to_world(
        hand_wrenches_sln,
        env.path_tangent_w[env_ids],
        env.path_lateral_w[env_ids],
        env.path_normal_w[env_ids],
    )
    linear_stiffness = float(env.d6_constraint_manager.cfg.linear_stiffness)
    preload_offset_per_hand_w = -handle_wrenches_w[..., :3] / linear_stiffness
    # Keep the mean for scalar hitch-height diagnostics.  Cart placement below
    # fits the two per-hand offsets as a rigid two-point pose, so the differential
    # preload is present before the first physics step.
    preload_offset_w = torch.mean(preload_offset_per_hand_w, dim=1)
    env.static_d6_preload_offset_w[env_ids] = preload_offset_w
    target_hitch_positions_w = grasp_position + preload_offset_per_hand_w
    cart_root, cart_quat, _ = fit_cart_pose_to_hitch_targets(
        target_hitch_positions_w,
        cart_quat,
        env.path_normal_w[env_ids],
        env.rickshaw_pose_cfg,
    )
    path_position = torch.sum((cart_root - origin) * env.path_tangent_w[env_ids], dim=-1)
    wheel_phase = wheel_phase_from_path_position(
        path_position, env.rickshaw_pose_cfg.wheel_radius
    )
    cart.write_root_pose_to_sim(torch.cat((cart_root, cart_quat), dim=-1), env_ids=env_ids)
    cart.write_root_velocity_to_sim(torch.zeros_like(root_velocity), env_ids=env_ids)
    wheel_joint_ids = _ensure_wheel_joint_ids(env)
    cart.write_joint_state_to_sim(
        wheel_phase,
        torch.zeros_like(wheel_phase),
        joint_ids=wheel_joint_ids,
        env_ids=env_ids,
    )


def reset_task_state(env: Any, env_ids: torch.Tensor) -> None:
    """Clear dynamic histories after the closed-chain state has been written."""

    env.command_state.reset(env_ids)
    env.path_state.lateral_error[env_ids] = 0.0
    env.path_state.heading_error[env_ids] = 0.0
    env.rickshaw_state.wheel_normal_force[env_ids] = 0.0
    env.rickshaw_state.two_wheel_contact[env_ids] = False
    env.rickshaw_state.d6_residual[env_ids] = 0.0
    env.rickshaw_state.d6_impulse[env_ids] = 0.0
    env.rickshaw_state.d6_wrench_w[env_ids] = 0.0
    env.rickshaw_state.hand_force_w[env_ids] = 0.0
    env.rickshaw_state.hand_torque_w[env_ids] = 0.0
    env.d6_incoming_joint_proxy_w[env_ids] = 0.0
    env.cart_interaction_wrench_state.reset(
        torch.zeros((env_ids.numel(), 3), device=env.device), env_ids
    )
    env.cart_interaction_wrench_valid[env_ids] = False
    env.stability_state.fat_valid[env_ids] = False
    env.stability_state.fat_wrench_consistent[env_ids] = False
    env.stability_state.fat_wrench_relative_error[env_ids] = 0.0
    env.stability_state.theta_fat[env_ids] = 0.0
    env.stability_state.torso_pitch[env_ids] = 0.0
    env.stability_state.zmp_s[env_ids] = 0.0
    env.stability_state.zmp_margin[env_ids] = 0.0
    env.stability_state.zmp_valid[env_ids] = False
    env.stability_state.ground_reaction_normal[env_ids] = 0.0
    env.stability_state.support_center_w[env_ids] = 0.0
    env.stability_state.support_points_sy[env_ids] = 0.0
    env.stability_state.support_point_mask[env_ids] = False
    env.action_state.reset(env.action_state.q_ref[env_ids], env_ids)
    env.observation_history_state.reset(env_ids)
    env.termination_state.reset(env_ids)
    env.curriculum_state.reset(env_ids)
    cart = env.scene["rickshaw"]
    v_s = torch.sum(cart.data.root_lin_vel_w * env.path_tangent_w, dim=-1)
    pitch = rickshaw_pitch_from_quaternion(
        cart.data.root_quat_w, env.path_tangent_w, env.path_normal_w
    )
    env.analytic_force_state.reset(v_s, pitch, env_ids)
    if hasattr(env, "fat2_wrench_consistency_state"):
        env.fat2_wrench_consistency_state.reset(env_ids)
    if hasattr(env, "fat2_com_radius_state"):
        env.fat2_com_radius_state.reset(env_ids)
        env.fat_com_radius_raw[env_ids] = env.fat2_com_radius_state.reference_radius[env_ids]
    if hasattr(env, "zmp_kinematic_state"):
        robot = env.scene["robot"]
        velocity_s = torch.sum(
            robot.data.root_com_lin_vel_w * env.path_tangent_w, dim=-1
        )
        velocity_n = torch.sum(
            robot.data.root_com_lin_vel_w * env.path_normal_w, dim=-1
        )
        env.zmp_kinematic_state.reset(velocity_s, velocity_n, env_ids)


def prepare_closed_chain_reset(env: Any, env_ids: torch.Tensor) -> None:
    """Install the nominal fully-loaded fixed point before writing its state.

    D6 drives and the existing policy controller are active at their final
    values from the first physics substep.  No reset-only controller or load
    homotopy is introduced.
    """

    update_slope_frame(env, env_ids)
    install_q_ref_from_reset_library(env, env_ids)


def _reset_action_terms_to_current_reference(env: Any, env_ids: torch.Tensor) -> None:
    """Rebind action filters after the reset event installs the new q_ref."""

    for term in env.action_manager._terms.values():
        term.reset(env_ids)


def finish_closed_chain_reset(env: Any, env_ids: torch.Tensor) -> None:
    """Commit the fully-loaded fixed point directly to the normal controller."""

    reset_task_state(env, env_ids)
    _reset_action_terms_to_current_reference(env, env_ids)
    resample_speed_command(env, env_ids)


def reset_closed_chain(env: Any, env_ids: torch.Tensor) -> None:
    """Single reset EventTerm around the calibrated project pose writer."""

    prepare_closed_chain_reset(env, env_ids)
    env.write_closed_chain_reset_state(env_ids)
    finish_closed_chain_reset(env, env_ids)


@dataclass
class PolicyStateUpdateCfg:
    """Validated configurations used by the first termination-manager term."""

    speed_reference: SpeedReferenceCfg
    analytic_force: AnalyticForceCfg
    support_polygon: SupportPolygonCfg
    fat2: FAT2Cfg
    zmp: ZMPCfg
    command_sampling: SpeedCommandSamplingCfg = field(default_factory=SpeedCommandSamplingCfg)


def refresh_policy_state(env: Any, cfg: PolicyStateUpdateCfg) -> torch.Tensor:
    """Refresh current physics state before all termination and reward terms.

    Configure this as the first TerminationTerm.  It always returns false; its
    purpose is to provide current-step path, cart, D6, FAT2, ZMP, curriculum,
    and curriculum state to the remaining terms in unmodified
    :class:`ManagerBasedRLEnv`.
    """

    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    hitch_position_w = torch.mean(
        cart.data.body_pos_w[:, env.hitch_body_ids], dim=1
    )
    hitch_velocity_w = torch.mean(
        cart.data.body_lin_vel_w[:, env.hitch_body_ids], dim=1
    )
    cart_pitch = rickshaw_pitch_from_quaternion(
        cart.data.root_quat_w, env.path_tangent_w, env.path_normal_w
    )
    cart_speed_s = torch.sum(
        cart.data.root_lin_vel_w * env.path_tangent_w, dim=-1
    )
    robot_speed_s = torch.sum(
        robot.data.root_lin_vel_w * env.path_tangent_w, dim=-1
    )
    robot_velocity_n = torch.sum(
        robot.data.root_lin_vel_w * env.path_normal_w, dim=-1
    )
    env.policy_robot_speed_s[:] = robot_speed_s
    env.policy_robot_velocity_n[:] = robot_velocity_n
    cart_kinematics = cart_system_mass_kinematics(env)
    robot_kinematics = robot_system_mass_kinematics(env)
    rickshaw_geometry_sn = actual_rickshaw_geometry_in_slope_frame(
        env,
        cart_com_w=cart_kinematics[0],
        hitch_w=hitch_position_w,
    )

    update_path_tracking_state(env)
    update_rickshaw_geometry_state(
        env,
        hitch_position_w=hitch_position_w,
        hitch_velocity_w=hitch_velocity_w,
        pitch=cart_pitch,
    )
    update_support_polygon(env, cfg.support_polygon)
    update_cart_interaction_wrench(env, cart_kinematics)
    update_analytic_rickshaw_force(
        env,
        cfg.analytic_force,
        cart_speed_s=cart_speed_s,
        pitch=cart_pitch,
        geometry_sn=rickshaw_geometry_sn,
    )
    env.rickshaw_state.two_wheel_contact[:] = torch.all(
        env.rickshaw_state.wheel_normal_force
        >= cfg.analytic_force.minimum_wheel_normal_force,
        dim=-1,
    )
    update_fat2_reference(
        env,
        cfg.fat2,
        robot_kinematics=robot_kinematics,
        hitch_w=hitch_position_w,
    )
    update_zmp_stability(
        env,
        cfg.zmp,
        robot_kinematics=robot_kinematics,
        hitch_w=hitch_position_w,
    )
    from .curricula import record_curriculum_tracking

    record_curriculum_tracking(env, robot_speed_s)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def advance_policy_interval(
    env: Any,
    env_ids: torch.Tensor | None,
    cfg: PolicyStateUpdateCfg,
) -> None:
    """Policy-rate interval EventTerm for the next command/current/history.

    Configure it with ``interval_range_s=(step_dt, step_dt)`` and
    ``is_global_time=True`` so all environments are updated exactly once after
    reward/reset and before observation computation.
    """

    if env_ids is not None and env_ids.numel() != env.num_envs:
        raise RuntimeError("advance_policy_interval must be a global-time interval event")
    update_curriculum_iteration_from_steps(env)
    advance_speed_command_resampling(env, cfg.command_sampling)
    advance_speed_reference(env, cfg.speed_reference)
    from .observations import actor_observation

    observation = actor_observation(env, use_cache=False)
    env._actor_observation_cache = observation
    valid = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    env.observation_history_state.advance(observation, valid)


__all__ = [
    "CommandState",
    "D6ConstraintManager",
    "D6ReactionResidualAdapter",
    "IsaacSimD6ReactionProvider",
    "HandleConstraintCfg",
    "PathTrackingState",
    "PolicyStateUpdateCfg",
    "RickshawPoseTargetCfg",
    "RickshawRuntimeState",
    "RuntimeRandomizationCfg",
    "ResetValidationCfg",
    "SpeedCommandSamplingCfg",
    "StabilityState",
    "TaskEntityNamesCfg",
    "advance_policy_interval",
    "advance_speed_command_resampling",
    "advance_speed_reference",
    "bind_d6_runtime_adapters",
    "compute_path_tracking_errors",
    "d6_spatial_impulse_magnitudes",
    "fit_cart_pose_to_hitch_targets",
    "finish_closed_chain_reset",
    "initialize_mdp_state",
    "initialize_curriculum_runtime",
    "install_q_ref_from_reset_library",
    "prepare_closed_chain_reset",
    "quat_multiply_wxyz",
    "resample_speed_command",
    "refresh_policy_state",
    "reset_speed_command",
    "reset_closed_chain",
    "recover_d6_wrench_on_robot",
    "reset_task_state",
    "resolve_task_entities",
    "sample_episode_physics",
    "sample_rolling_resistance",
    "sample_speed_commands",
    "spatial_wrenches_sln_to_world",
    "static_reset_contact_allocation",
    "target_cart_orientation",
    "target_pitch_from_hitch_height",
    "update_path_tracking_state",
    "update_curriculum_iteration_from_steps",
    "update_rickshaw_geometry_state",
    "wheel_phase_from_path_position",
    "write_closed_chain_reset_state",
]
