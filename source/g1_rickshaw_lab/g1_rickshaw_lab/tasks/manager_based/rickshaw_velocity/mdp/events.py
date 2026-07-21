"""Simulator-independent state and sampling kernels used by Mjlab events."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass
from typing import Any, Mapping

import torch

from g1_rickshaw_lab.assets.rickshaw import (
    RICKSHAW_CENTER_OF_MASS,
    RICKSHAW_TOTAL_MASS,
)

from .dynamics import quat_apply_wxyz


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
    ) -> CommandState:
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
    ) -> PathTrackingState:
        zeros = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(zeros.clone(), zeros.clone())


@dataclass
class RickshawRuntimeState:
    wheel_normal_force: torch.Tensor
    hitch_height: torch.Tensor
    hitch_vertical_speed: torch.Tensor
    pitch: torch.Tensor
    two_wheel_contact: torch.Tensor
    connection_residual: torch.Tensor
    connection_impulse: torch.Tensor
    connection_wrench_w: torch.Tensor
    connection_truth_wrench_w: torch.Tensor
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
    ) -> RickshawRuntimeState:
        scalar = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(
            wheel_normal_force=torch.zeros(
                (num_envs, num_wheels), device=device, dtype=dtype
            ),
            hitch_height=scalar.clone(),
            hitch_vertical_speed=scalar.clone(),
            pitch=scalar.clone(),
            two_wheel_contact=torch.zeros(
                num_envs, device=device, dtype=torch.bool
            ),
            connection_residual=scalar.clone(),
            connection_impulse=torch.zeros((num_envs, 2), device=device, dtype=dtype),
            connection_wrench_w=torch.zeros((num_envs, 2, 6), device=device, dtype=dtype),
            connection_truth_wrench_w=torch.zeros(
                (num_envs, 2, 6), device=device, dtype=dtype
            ),
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
    ) -> StabilityState:
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
            support_points_sy=torch.zeros(
                (num_envs, 8, 2), device=device, dtype=dtype
            ),
            support_point_mask=torch.zeros(
                (num_envs, 8), device=device, dtype=torch.bool
            ),
        )


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
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


@dataclass
class SpeedCommandSamplingCfg:
    minimum: float = 0.0
    maximum: float = 0.1
    limit_maximum: float = 1.0
    curriculum_step: float = 0.1
    standing_fraction: float = 0.02
    resampling_time_s: float = 10.0

    def validate(self) -> None:
        if not 0.0 <= self.minimum <= self.maximum <= self.limit_maximum:
            raise ValueError(
                "speed command range must satisfy 0 <= minimum <= maximum <= limit_maximum"
            )
        if self.curriculum_step <= 0.0:
            raise ValueError("curriculum_step must be positive")
        if not 0.0 <= self.standing_fraction <= 1.0:
            raise ValueError("standing_fraction must lie in [0,1]")
        if self.resampling_time_s <= 0.0:
            raise ValueError("resampling_time_s must be positive")


def sample_speed_commands(
    num_samples: int,
    cfg: SpeedCommandSamplingCfg,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    cfg.validate()
    samples = torch.rand(
        num_samples, device=device, dtype=dtype, generator=generator
    )
    samples = cfg.minimum + (cfg.maximum - cfg.minimum) * samples
    if cfg.standing_fraction > 0.0:
        standing = (
            torch.rand(num_samples, device=device, generator=generator)
            < cfg.standing_fraction
        )
        samples[standing] = 0.0
    return samples


def resample_speed_command(
    env: Any, env_ids: torch.Tensor, cfg: SpeedCommandSamplingCfg
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


def compute_path_tracking_errors(
    robot_position_w: torch.Tensor,
    cart_position_w: torch.Tensor,
    robot_quaternion_wxyz: torch.Tensor,
    path_origin_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = 0.5 * (robot_position_w + cart_position_w)
    lateral_error = torch.sum(
        (midpoint - path_origin_w) * path_lateral_w, dim=-1
    )
    local_x = torch.zeros_like(path_tangent_w)
    local_x[:, 0] = 1.0
    robot_forward_w = quat_apply_wxyz(robot_quaternion_wxyz, local_x)
    heading_error = torch.atan2(
        torch.sum(robot_forward_w * path_lateral_w, dim=-1),
        torch.sum(robot_forward_w * path_tangent_w, dim=-1),
    )
    return lateral_error, heading_error


DOMAIN_RANDOMIZATION_NAMES = (
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
DOMAIN_PARAMETER_NAMES = DOMAIN_RANDOMIZATION_NAMES


@dataclass(kw_only=True)
class DomainRandomizationCfg:
    enabled: bool = True
    ranges: Mapping[str, tuple[float, float]] = MISSING
    nominal: Mapping[str, float] = MISSING
    calibration: Mapping[str, Any] = MISSING

    def validate(self) -> None:
        required = set(DOMAIN_PARAMETER_NAMES)
        for label, values in (("ranges", self.ranges), ("nominal", self.nominal)):
            if not isinstance(values, Mapping) or set(values) != required:
                raise ValueError(
                    f"domain randomization {label} must contain exactly {sorted(required)}"
                )
        for name, interval in self.ranges.items():
            low, high = map(float, interval)
            nominal = float(self.nominal[name])
            if not all(map(math.isfinite, (low, high, nominal))) or not (
                low <= nominal <= high
            ):
                raise ValueError(f"invalid range or nominal value for {name!r}")
        for name in (
            "rolling_resistance.c_rr",
            "wheel.left_damping",
            "wheel.right_damping",
        ):
            if float(self.ranges[name][0]) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if float(self.ranges["terrain.friction"][0]) <= 0.0:
            raise ValueError("terrain friction must stay positive")
        if not isinstance(self.calibration, Mapping):
            raise ValueError("domain randomization calibration must be a mapping")


def sample_domain_parameters(
    cfg: DomainRandomizationCfg,
    batch_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    cfg.validate()
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative")

    def sample(name: str) -> torch.Tensor:
        if not cfg.enabled:
            return torch.full(
                (batch_size,), float(cfg.nominal[name]), device=device, dtype=dtype
            )
        low, high = map(float, cfg.ranges[name])
        if low == high:
            return torch.full((batch_size,), low, device=device, dtype=dtype)
        return torch.empty((batch_size,), device=device, dtype=dtype).uniform_(
            low, high, generator=generator
        )

    return {name: sample(name) for name in DOMAIN_RANDOMIZATION_NAMES}


def effective_cart_mass_com_bounds(
    ranges: Mapping[str, tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mass_low, mass_high = map(float, ranges["payload.mass"])
    lower = [RICKSHAW_TOTAL_MASS + mass_low]
    upper = [RICKSHAW_TOTAL_MASS + mass_high]
    for axis, name in enumerate(
        ("payload.com.x", "payload.com.y", "payload.com.z")
    ):
        payload_low, payload_high = map(float, ranges[name])
        candidates = [
            (
                RICKSHAW_TOTAL_MASS * RICKSHAW_CENTER_OF_MASS[axis]
                + payload_mass * payload_com
            )
            / (RICKSHAW_TOTAL_MASS + payload_mass)
            for payload_mass in (mass_low, mass_high)
            for payload_com in (payload_low, payload_high)
        ]
        lower.append(min(candidates))
        upper.append(max(candidates))
    return tuple(lower), tuple(upper)


def _update_teacher_static_domain(
    env: Any,
    cfg: DomainRandomizationCfg,
    sampled: Mapping[str, torch.Tensor],
) -> None:
    raw = torch.cat(
        (
            env.effective_torso_mass[:, None],
            env.effective_cart_mass_com,
            sampled["rolling_resistance.c_rr"][:, None],
            sampled["terrain.friction"][:, None],
            torch.stack(
                (
                    sampled["wheel.left_damping"],
                    sampled["wheel.right_damping"],
                ),
                dim=-1,
            ),
        ),
        dim=-1,
    )
    cart_lower, cart_upper = effective_cart_mass_com_bounds(cfg.ranges)
    nominal_torso_mass = float(
        env._default_robot_masses_cpu[0, env.torso_body_id]
    )
    lower = torch.tensor(
        (
            nominal_torso_mass + cfg.ranges["torso.mass_delta"][0],
            *cart_lower,
            cfg.ranges["rolling_resistance.c_rr"][0],
            cfg.ranges["terrain.friction"][0],
            cfg.ranges["wheel.left_damping"][0],
            cfg.ranges["wheel.right_damping"][0],
        ),
        device=env.device,
        dtype=raw.dtype,
    )
    upper = torch.tensor(
        (
            nominal_torso_mass + cfg.ranges["torso.mass_delta"][1],
            *cart_upper,
            cfg.ranges["rolling_resistance.c_rr"][1],
            cfg.ranges["terrain.friction"][1],
            cfg.ranges["wheel.left_damping"][1],
            cfg.ranges["wheel.right_damping"][1],
        ),
        device=env.device,
        dtype=raw.dtype,
    )
    from .observations import TEACHER_STATIC_DOMAIN_DIM, normalize_features

    if raw.shape != (env.num_envs, TEACHER_STATIC_DOMAIN_DIM):
        raise RuntimeError(
            f"effective teacher static domain must have shape [N,{TEACHER_STATIC_DOMAIN_DIM}]"
        )
    env.teacher_static_domain_raw = raw
    env.teacher_static_domain_bounds = (lower, upper)
    env.normalized_teacher_static_domain = normalize_features(raw, lower, upper)


__all__ = [
    "CommandState",
    "DOMAIN_PARAMETER_NAMES",
    "DOMAIN_RANDOMIZATION_NAMES",
    "DomainRandomizationCfg",
    "PathTrackingState",
    "RickshawRuntimeState",
    "SpeedCommandSamplingCfg",
    "StabilityState",
    "compute_path_tracking_errors",
    "effective_cart_mass_com_bounds",
    "quat_multiply_wxyz",
    "resample_speed_command",
    "sample_domain_parameters",
    "sample_speed_commands",
]
