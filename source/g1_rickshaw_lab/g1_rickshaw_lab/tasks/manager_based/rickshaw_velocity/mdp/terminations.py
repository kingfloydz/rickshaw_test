"""Immediate and ten-step persistent safety terminations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, dataclass
from typing import Any

import torch

PERSISTENCE_STEPS = 10
ROOT_NORMAL_HEIGHT_MIN = 0.31
PERSISTENT_CAUSES = (
    "low_root_height",
    "torso_tilt",
    "rickshaw_envelope",
    "lateral_corridor",
    "heading_envelope",
    "overspeed",
    "arm_torque",
    "zmp_outside",
)
IMMEDIATE_CAUSES = (
    "non_finite",
    "illegal_body_contact",
    "wheel_lift",
    "connection_constraint_failure",
    "joint_hard_limit",
)
TERMINATION_CAUSES = ("time_out", *IMMEDIATE_CAUSES, *PERSISTENT_CAUSES)


@dataclass(kw_only=True)
class ImmediateSafetyCfg:
    illegal_contact_force_threshold: float = MISSING
    wheel_lift_normal_force_threshold: float = MISSING
    connection_residual_limit: float = MISSING
    connection_impulse_limit: float = MISSING


@dataclass(kw_only=True)
class PersistentSafetyCfg:
    torso_tilt_max: float = MISSING
    hitch_height_bounds: tuple[float, float] = MISSING
    rickshaw_pitch_bounds: tuple[float, float] = MISSING
    lateral_corridor: float = MISSING
    heading_envelope: float = MISSING
    overspeed_margin: float = MISSING
    arm_torque_limit: float = MISSING
    root_normal_height_min: float = ROOT_NORMAL_HEIGHT_MIN
    persistence_steps: int = PERSISTENCE_STEPS

    def validate(self) -> None:
        if self.root_normal_height_min != ROOT_NORMAL_HEIGHT_MIN:
            raise ValueError("the specified root-height threshold is 0.31 m")
        if self.persistence_steps != PERSISTENCE_STEPS:
            raise ValueError("persistent safety conditions must last 10 policy steps")
        if not self.torso_tilt_max > 0.0:
            raise ValueError("torso_tilt_max must be positive")
        if self.hitch_height_bounds[1] <= self.hitch_height_bounds[0]:
            raise ValueError("hitch height bounds are not ordered")
        if self.rickshaw_pitch_bounds[1] <= self.rickshaw_pitch_bounds[0]:
            raise ValueError("rickshaw pitch bounds are not ordered")
        for name, value in (
            ("lateral_corridor", self.lateral_corridor),
            ("heading_envelope", self.heading_envelope),
            ("overspeed_margin", self.overspeed_margin),
            ("arm_torque_limit", self.arm_torque_limit),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.arm_torque_limit > 1.0:
            raise ValueError("arm_torque_limit is a normalized fraction and must not exceed 1")


@dataclass
class PersistentTerminationState:
    counters: torch.Tensor
    last_causes: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
    ) -> PersistentTerminationState:
        shape = (num_envs, len(PERSISTENT_CAUSES))
        return cls(
            counters=torch.zeros(shape, device=device, dtype=torch.long),
            last_causes=torch.zeros(shape, device=device, dtype=torch.bool),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.counters[ids] = 0
        self.last_causes[ids] = False

    def update(
        self, violations: torch.Tensor, persistence_steps: int = PERSISTENCE_STEPS
    ) -> torch.Tensor:
        if violations.shape != self.counters.shape or violations.dtype != torch.bool:
            raise ValueError(
                f"violations must be bool with shape {tuple(self.counters.shape)}"
            )
        if persistence_steps != PERSISTENCE_STEPS:
            raise ValueError("the task requires exactly 10 consecutive policy steps")
        self.counters[:] = torch.where(
            violations, self.counters + 1, torch.zeros_like(self.counters)
        )
        self.last_causes[:] = self.counters >= persistence_steps
        return torch.any(self.last_causes, dim=-1)

    def cause_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: self.last_causes[:, index]
            for index, name in enumerate(PERSISTENT_CAUSES)
        }


@dataclass
class TerminationCauseState:
    """Global cause histogram plus per-environment causes for the last step."""

    counts: torch.Tensor
    last_causes: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
    ) -> TerminationCauseState:
        return cls(
            counts=torch.zeros(len(TERMINATION_CAUSES), device=device, dtype=torch.long),
            last_causes=torch.zeros(
                (num_envs, len(TERMINATION_CAUSES)), device=device, dtype=torch.bool
            ),
        )

    def begin_policy_step(self) -> None:
        self.last_causes[:] = False

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.last_causes[ids] = False

    def record(self, names: Sequence[str], causes: torch.Tensor) -> None:
        if causes.shape != (self.last_causes.shape[0], len(names)) or causes.dtype != torch.bool:
            raise ValueError("termination causes have the wrong shape or dtype")
        for local_index, name in enumerate(names):
            try:
                global_index = TERMINATION_CAUSES.index(name)
            except ValueError as exc:
                raise KeyError(f"unknown termination cause {name!r}") from exc
            self.last_causes[:, global_index] |= causes[:, local_index]
            self.counts[global_index] += torch.sum(causes[:, local_index]).to(torch.long)

    def histogram(self, *, reset: bool = False) -> dict[str, int]:
        result = {
            name: int(self.counts[index].item())
            for index, name in enumerate(TERMINATION_CAUSES)
        }
        if reset:
            self.counts[:] = 0
        return result


def termination_cause_histogram(env: Any, *, reset: bool = False) -> dict[str, int]:
    """Public logging interface for the mandatory termination histogram."""

    return env.termination_cause_state.histogram(reset=reset)


def finite_tensor_violation(*values: torch.Tensor) -> torch.Tensor:
    """Return a per-environment NaN/Inf mask across arbitrary state tensors."""

    if not values:
        raise ValueError("at least one tensor is required")
    num_envs = values[0].shape[0]
    violation = torch.zeros(num_envs, device=values[0].device, dtype=torch.bool)
    for value in values:
        if value.shape[0] != num_envs:
            raise ValueError("all finite-check tensors must share the environment axis")
        flattened = value.reshape(num_envs, -1)
        violation |= ~torch.all(torch.isfinite(flattened), dim=-1)
    return violation


def contact_force_violation(
    contact_force_w: torch.Tensor, threshold: float
) -> torch.Tensor:
    if threshold <= 0.0:
        raise ValueError("contact-force threshold must be positive")
    if contact_force_w.ndim < 3 or contact_force_w.shape[-1] != 3:
        raise ValueError("contact forces must have environment/body/.../3 axes")
    magnitude = torch.linalg.vector_norm(contact_force_w, dim=-1)
    return torch.any(magnitude.reshape(magnitude.shape[0], -1) > threshold, dim=-1)


def wheel_lift_violation(
    wheel_normal_force: torch.Tensor, lift_threshold: float
) -> torch.Tensor:
    if lift_threshold <= 0.0:
        raise ValueError("wheel lift threshold must be positive")
    if wheel_normal_force.ndim != 2 or wheel_normal_force.shape[-1] != 2:
        raise ValueError("wheel normal force must have shape [N,2]")
    return torch.any(wheel_normal_force < lift_threshold, dim=-1)


def connection_safety_violation(
    residual: torch.Tensor,
    impulse: torch.Tensor,
    residual_limit: float,
    impulse_limit: float,
) -> torch.Tensor:
    if residual_limit <= 0.0 or impulse_limit <= 0.0:
        raise ValueError("connection residual and impulse limits must be positive")
    residual_value = torch.amax(residual.reshape(residual.shape[0], -1), dim=-1)
    impulse_value = torch.amax(torch.abs(impulse).reshape(impulse.shape[0], -1), dim=-1)
    return (residual_value > residual_limit) | (impulse_value > impulse_limit)


def hard_joint_limit_violation(
    joint_position: torch.Tensor, hard_limits: torch.Tensor
) -> torch.Tensor:
    if hard_limits.shape != (*joint_position.shape, 2):
        raise ValueError("hard limits must have shape [N,J,2] matching joint positions")
    outside = (joint_position < hard_limits[..., 0]) | (
        joint_position > hard_limits[..., 1]
    )
    return torch.any(outside, dim=-1)


def persistent_condition_matrix(
    root_normal_height: torch.Tensor,
    torso_tilt: torch.Tensor,
    hitch_height: torch.Tensor,
    rickshaw_pitch: torch.Tensor,
    lateral_error: torch.Tensor,
    heading_error: torch.Tensor,
    actual_speed: torch.Tensor,
    v_ref: torch.Tensor,
    arm_torque: torch.Tensor,
    zmp_margin: torch.Tensor,
    zmp_valid: torch.Tensor,
    cfg: PersistentSafetyCfg,
) -> torch.Tensor:
    """Build the ordered eight-cause bool matrix used by persistent counters."""

    cfg.validate()
    hitch_low, hitch_high = cfg.hitch_height_bounds
    pitch_low, pitch_high = cfg.rickshaw_pitch_bounds
    wrapped_heading = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    arm_overload = torch.any(torch.abs(arm_torque) > cfg.arm_torque_limit, dim=-1)
    return torch.stack(
        (
            root_normal_height < cfg.root_normal_height_min,
            torch.abs(torso_tilt) > cfg.torso_tilt_max,
            (hitch_height < hitch_low)
            | (hitch_height > hitch_high)
            | (rickshaw_pitch < pitch_low)
            | (rickshaw_pitch > pitch_high),
            torch.abs(lateral_error) > cfg.lateral_corridor,
            torch.abs(wrapped_heading) > cfg.heading_envelope,
            actual_speed > v_ref + cfg.overspeed_margin,
            arm_overload,
            (~zmp_valid) | (zmp_margin < 0.0),
        ),
        dim=-1,
    )


__all__ = [
    "ImmediateSafetyCfg",
    "IMMEDIATE_CAUSES",
    "PERSISTENCE_STEPS",
    "PERSISTENT_CAUSES",
    "PersistentSafetyCfg",
    "PersistentTerminationState",
    "TERMINATION_CAUSES",
    "TerminationCauseState",
    "ROOT_NORMAL_HEIGHT_MIN",
    "contact_force_violation",
    "connection_safety_violation",
    "finite_tensor_violation",
    "hard_joint_limit_violation",
    "persistent_condition_matrix",
    "termination_cause_histogram",
    "wheel_lift_violation",
]
