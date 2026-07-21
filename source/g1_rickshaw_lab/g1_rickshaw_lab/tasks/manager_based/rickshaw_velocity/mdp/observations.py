"""Fixed-schema actor observations and the exclusive 61-frame history."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
    validate_history_length,
)
from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS

from .actions import ACTION_DIM
from .rewards import GAIT_PERIOD_S

TEACHER_STATIC_DOMAIN_DIM = TEACHER_STATIC_DIM - 1
SLOPE_LOWER = min(SLOPE_GRADIENTS)
SLOPE_UPPER = max(SLOPE_GRADIENTS)

BASE_ANGULAR_VELOCITY_SLICE = slice(0, 3)
PROJECTED_GRAVITY_SLICE = slice(3, 6)
TASK_SIGNAL_SLICE = slice(6, 9)
JOINT_POSITION_SLICE = slice(9, 38)
JOINT_VELOCITY_SLICE = slice(38, 67)
PREVIOUS_ACTION_SLICE = slice(67, 67 + ACTION_DIM)
GAIT_PHASE_SLICE = slice(67 + ACTION_DIM, ACTOR_OBSERVATION_DIM)

BASE_ANGULAR_VELOCITY_SCALE = 0.25
PROJECTED_GRAVITY_SCALE = 1.0
TASK_SIGNAL_SCALE = (2.0, 2.0, 1.0)
JOINT_POSITION_SCALE = 1.0
JOINT_VELOCITY_SCALE = 0.05
PREVIOUS_ACTION_SCALE = 1.0

# Unitree G1-29DoF velocity-policy sensor noise, expressed after this
# project's observation scaling.
ACTOR_OBSERVATION_NOISE_SCALE = (
    (0.2 * BASE_ANGULAR_VELOCITY_SCALE,) * 3
    + (0.05 * PROJECTED_GRAVITY_SCALE,) * 3
    + (0.0,) * 3
    + (0.01 * JOINT_POSITION_SCALE,) * ACTION_DIM
    + (1.5 * JOINT_VELOCITY_SCALE,) * ACTION_DIM
    + (0.0,) * ACTION_DIM
    + (0.0,) * 2
)

TEACHER_STATIC_FEATURE_NAMES = (
    "robot.torso_mass",
    "cart.total_mass",
    "cart.com.x",
    "cart.com.y",
    "cart.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
    "terrain.slope",
)
TEACHER_DYNAMIC_FEATURE_NAMES = (
    "robot.velocity.s",
    "robot.velocity.l",
    "robot.velocity.n",
    "cart.velocity.s",
    "cart.velocity.l",
    "cart.velocity.n",
    "cart.pitch",
    "wheel.left_normal_force",
    "wheel.right_normal_force",
    *(
        f"connection.{side}.{kind}.{axis}"
        for side in ("left", "right")
        for kind in ("force", "torque")
        for axis in ("s", "l", "n")
    ),
)
if len(TEACHER_STATIC_FEATURE_NAMES) != TEACHER_STATIC_DIM:
    raise RuntimeError("teacher static feature schema has the wrong dimension")
if len(TEACHER_DYNAMIC_FEATURE_NAMES) != TEACHER_DYNAMIC_DIM:
    raise RuntimeError("teacher dynamic feature schema is not 21-D")


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def gait_phase_observation(episode_time_s: torch.Tensor) -> torch.Tensor:
    phase = torch.remainder(episode_time_s, GAIT_PERIOD_S) / GAIT_PERIOD_S
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def assemble_actor_observation(
    base_angular_velocity_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    v_ref: torch.Tensor,
    lateral_error: torch.Tensor,
    heading_error: torch.Tensor,
    joint_position: torch.Tensor,
    q_ref: torch.Tensor,
    joint_velocity_value: torch.Tensor,
    previous_processed_action: torch.Tensor,
    gait_phase: torch.Tensor,
) -> torch.Tensor:
    """Assemble the only deployment observation, in the fixed 98-D order."""

    batch_shape = base_angular_velocity_b.shape[:-1]
    if base_angular_velocity_b.shape[-1] != 3:
        raise ValueError("base angular velocity must end in dimension 3")
    if projected_gravity_b.shape != base_angular_velocity_b.shape:
        raise ValueError("projected gravity shape differs from angular velocity")
    for name, value in (
        ("joint_position", joint_position),
        ("q_ref", q_ref),
        ("joint_velocity", joint_velocity_value),
        ("previous_processed_action", previous_processed_action),
    ):
        if value.shape != (*batch_shape, ACTION_DIM):
            raise ValueError(f"{name} must have shape {(*batch_shape, ACTION_DIM)}")
    if gait_phase.shape != (*batch_shape, 2):
        raise ValueError(f"gait_phase must have shape {(*batch_shape, 2)}")
    for name, value in (
        ("v_ref", v_ref),
        ("lateral_error", lateral_error),
        ("heading_error", heading_error),
    ):
        if value.shape != batch_shape:
            raise ValueError(f"{name} must have batch shape {batch_shape}")

    return _assemble_actor_observation(
        base_angular_velocity_b,
        projected_gravity_b,
        v_ref,
        lateral_error,
        heading_error,
        joint_position,
        q_ref,
        joint_velocity_value,
        previous_processed_action,
        gait_phase,
    )


def _assemble_actor_observation(
    base_angular_velocity_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    v_ref: torch.Tensor,
    lateral_error: torch.Tensor,
    heading_error: torch.Tensor,
    joint_position: torch.Tensor,
    q_ref: torch.Tensor,
    joint_velocity_value: torch.Tensor,
    previous_processed_action: torch.Tensor,
    gait_phase: torch.Tensor,
) -> torch.Tensor:
    """Hot-path observation assembly after startup schema validation."""

    position_error = joint_position - q_ref
    task = torch.stack(
        (
            v_ref * TASK_SIGNAL_SCALE[0],
            lateral_error * TASK_SIGNAL_SCALE[1],
            wrap_to_pi(heading_error) * TASK_SIGNAL_SCALE[2],
        ),
        dim=-1,
    )
    return torch.cat(
        (
            base_angular_velocity_b * BASE_ANGULAR_VELOCITY_SCALE,
            projected_gravity_b * PROJECTED_GRAVITY_SCALE,
            task,
            position_error * JOINT_POSITION_SCALE,
            joint_velocity_value * JOINT_VELOCITY_SCALE,
            previous_processed_action * PREVIOUS_ACTION_SCALE,
            gait_phase,
        ),
        dim=-1,
    )


@dataclass
class ObservationHistoryState:
    """History where ``current`` is never included in ``history``."""

    history: torch.Tensor | None
    current: torch.Tensor
    initialized: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        history_length: int = HISTORY_LENGTH,
        observation_dim: int = ACTOR_OBSERVATION_DIM,
        history_enabled: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ObservationHistoryState:
        history_length = validate_history_length(history_length)
        if observation_dim <= 0:
            raise ValueError("feature dimension must be positive")
        return cls(
            history=(
                torch.zeros(
                    (num_envs, history_length, observation_dim),
                    device=device,
                    dtype=dtype,
                )
                if history_enabled
                else None
            ),
            current=torch.zeros((num_envs, observation_dim), device=device, dtype=dtype),
            initialized=torch.zeros(num_envs, device=device, dtype=torch.bool),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        if self.history is not None:
            self.history[ids] = 0.0
        self.current[ids] = 0.0
        self.initialized[ids] = False

    def initialize(self, observation: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Fill all history frames with the first post-reset observation."""

        if env_ids is None:
            ids = torch.arange(self.current.shape[0], device=self.current.device)
        else:
            ids = env_ids.to(device=self.current.device, dtype=torch.long)
        if observation.shape == self.current.shape:
            observation = observation[ids]
        if observation.shape != (ids.numel(), self.current.shape[-1]):
            raise ValueError("initial history observation has the wrong shape")
        if self.history is not None:
            self.history[ids] = observation[:, None, :].expand(-1, self.history.shape[1], -1)
        self.current[ids] = observation
        self.initialized[ids] = True

    def advance(self, new_observation: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
        """Append old current, then replace current with the new observation."""

        if new_observation.shape != self.current.shape:
            raise ValueError("new_observation shape differs from history state")
        if valid_mask is None:
            valid_mask = torch.ones_like(self.initialized)
        if valid_mask.shape != self.initialized.shape:
            raise ValueError("valid_mask must have shape [N]")

        if self.history is None:
            self.current[valid_mask] = new_observation[valid_mask]
            self.initialized[valid_mask] = True
            return

        was_initialized = self.initialized.clone()
        initialize_mask = valid_mask & ~was_initialized
        advance_mask = valid_mask & was_initialized

        next_history = torch.cat((self.history[:, 1:], self.current[:, None, :]), dim=1)
        next_history[~advance_mask] = self.history[~advance_mask]
        initial = new_observation[initialize_mask]
        next_history[initialize_mask] = initial[:, None, :].expand(-1, next_history.shape[1], -1)
        self.history = next_history

        self.current[valid_mask] = new_observation[valid_mask]
        self.initialized[valid_mask] = True


def normalize_features(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Normalize explicit feature bounds to [-1, 1]; singleton bounds map to zero."""

    expected = (values.shape[-1],)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError(f"normalization bounds must both have shape {expected}")
    if not values.is_floating_point() or torch.any(~torch.isfinite(values)):
        raise ValueError("features to normalize must be finite floating-point values")
    lower = lower.to(device=values.device, dtype=values.dtype)
    upper = upper.to(device=values.device, dtype=values.dtype)
    if torch.any(~torch.isfinite(lower)) or torch.any(~torch.isfinite(upper)):
        raise ValueError("normalization bounds must be finite")
    width = upper - lower
    if torch.any(width < 0.0):
        raise ValueError("normalization bounds must be ordered")
    safe_width = torch.where(width > 0.0, width, torch.ones_like(width))
    normalized = 2.0 * (values - lower) / safe_width - 1.0
    normalized = torch.where(width > 0.0, normalized, torch.zeros_like(normalized))
    return torch.clamp(normalized, -1.0, 1.0)


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "GAIT_PHASE_SLICE",
    "HISTORY_LENGTH",
    "TEACHER_DYNAMIC_FEATURE_NAMES",
    "TEACHER_STATIC_DIM",
    "TEACHER_STATIC_DOMAIN_DIM",
    "TEACHER_STATIC_FEATURE_NAMES",
    "ACTOR_OBSERVATION_NOISE_SCALE",
    "ObservationHistoryState",
    "assemble_actor_observation",
    "gait_phase_observation",
    "normalize_features",
    "wrap_to_pi",
]
