"""Fixed-schema actor observations and the exclusive 61-frame history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .actions import ACTION_DIM


ACTOR_OBSERVATION_DIM = 96
HISTORY_LENGTH = 61

BASE_ANGULAR_VELOCITY_SLICE = slice(0, 3)
PROJECTED_GRAVITY_SLICE = slice(3, 6)
TASK_SIGNAL_SLICE = slice(6, 9)
JOINT_POSITION_SLICE = slice(9, 38)
JOINT_VELOCITY_SLICE = slice(38, 67)
PREVIOUS_ACTION_SLICE = slice(67, 96)

BASE_ANGULAR_VELOCITY_SCALE = 0.25
PROJECTED_GRAVITY_SCALE = 1.0
TASK_SIGNAL_SCALE = (2.0, 2.0, 1.0)
JOINT_POSITION_SCALE = 1.0
JOINT_VELOCITY_SCALE = 0.05
PREVIOUS_ACTION_SCALE = 1.0

INDEPENDENT_EXTRINSIC_NAMES = (
    "payload.mass",
    "payload.com.x",
    "payload.com.y",
    "payload.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
    "d6.linear_stiffness",
    "d6.linear_damping",
    "d6.angular_stiffness",
    "d6.angular_damping",
    "d6.max_force",
    "d6.max_torque",
    "d6.linear_limit",
    "d6.angular_limit",
    "motor.strength",
    "control.delay",
    "observation.delay",
)

DERIVED_OR_PRIVILEGED_ONLY_NAMES = frozenset(
    {
        "slope",
        "base_velocity",
        "cart_velocity",
        "rickshaw_pitch",
        "wheel_normal_force",
        "d6_wrench",
        "d6_residual",
        "filtered_acceleration",
        "t_s",
        "t_n",
        "zmp_margin",
    }
)


@dataclass(frozen=True)
class ObservationNoiseCfg:
    base_angular_velocity: float | torch.Tensor = 0.2
    projected_gravity: float | torch.Tensor = 0.05
    joint_position: float | torch.Tensor = 0.01
    joint_velocity: float | torch.Tensor = 1.5

    def scaled(self, scale: torch.Tensor | float) -> "ObservationNoiseCfg":
        return ObservationNoiseCfg(
            base_angular_velocity=self.base_angular_velocity * scale,
            projected_gravity=self.projected_gravity * scale,
            joint_position=self.joint_position * scale,
            joint_velocity=self.joint_velocity * scale,
        )


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _uniform_noise_like(
    value: torch.Tensor,
    magnitude: float | torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    magnitude_value = torch.as_tensor(
        magnitude, device=value.device, dtype=value.dtype
    )
    if magnitude_value.ndim == 1 and magnitude_value.shape[0] == value.shape[0]:
        magnitude_value = magnitude_value.reshape(
            magnitude_value.shape[0], *((1,) * (value.ndim - 1))
        )
    if torch.any(~torch.isfinite(magnitude_value)) or torch.any(magnitude_value < 0.0):
        raise ValueError("observation noise magnitude must be finite and non-negative")
    return (2.0 * torch.rand(
        value.shape,
        device=value.device,
        dtype=value.dtype,
        generator=generator,
    ) - 1.0) * magnitude_value


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
    *,
    noise_cfg: ObservationNoiseCfg | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Assemble the only deployment observation, in the fixed 96-D order."""

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
    for name, value in (
        ("v_ref", v_ref),
        ("lateral_error", lateral_error),
        ("heading_error", heading_error),
    ):
        if value.shape != batch_shape:
            raise ValueError(f"{name} must have batch shape {batch_shape}")

    angular_velocity = base_angular_velocity_b
    gravity = projected_gravity_b
    position_error = joint_position - q_ref
    velocity = joint_velocity_value
    if noise_cfg is not None:
        angular_velocity = angular_velocity + _uniform_noise_like(
            angular_velocity, noise_cfg.base_angular_velocity, generator
        )
        gravity = gravity + _uniform_noise_like(
            gravity, noise_cfg.projected_gravity, generator
        )
        position_error = position_error + _uniform_noise_like(
            position_error, noise_cfg.joint_position, generator
        )
        velocity = velocity + _uniform_noise_like(
            velocity, noise_cfg.joint_velocity, generator
        )

    task_scale = torch.tensor(
        TASK_SIGNAL_SCALE,
        device=base_angular_velocity_b.device,
        dtype=base_angular_velocity_b.dtype,
    )
    task = torch.stack((v_ref, lateral_error, wrap_to_pi(heading_error)), dim=-1) * task_scale
    observation = torch.cat(
        (
            angular_velocity * BASE_ANGULAR_VELOCITY_SCALE,
            gravity * PROJECTED_GRAVITY_SCALE,
            task,
            position_error * JOINT_POSITION_SCALE,
            velocity * JOINT_VELOCITY_SCALE,
            previous_processed_action * PREVIOUS_ACTION_SCALE,
        ),
        dim=-1,
    )
    if observation.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise RuntimeError(f"actor observation is {observation.shape[-1]}-D, expected 96-D")
    return observation


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
    ) -> "ObservationHistoryState":
        if history_length != HISTORY_LENGTH or observation_dim != ACTOR_OBSERVATION_DIM:
            raise ValueError("the task schema is fixed at history [N,61,96]")
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
        """Fill all 61 past frames with the first post-reset observation."""

        if env_ids is None:
            ids = torch.arange(self.current.shape[0], device=self.current.device)
        else:
            ids = env_ids.to(device=self.current.device, dtype=torch.long)
        if observation.shape == self.current.shape:
            observation = observation[ids]
        if observation.shape != (ids.numel(), ACTOR_OBSERVATION_DIM):
            raise ValueError("initial history observation has the wrong shape")
        if self.history is not None:
            self.history[ids] = observation[:, None, :].expand(
                -1, HISTORY_LENGTH, -1
            )
        self.current[ids] = observation
        self.initialized[ids] = True

    def advance(
        self, new_observation: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> None:
        """Append old current, then replace current with the new observation."""

        if new_observation.shape != self.current.shape:
            raise ValueError("new_observation must have shape [N,96]")
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

        # A policy step advances almost every environment. Building the next
        # contiguous history once avoids multiple ~96 MB gather/clone/scatter
        # passes at 4096 environments while preserving oldest-to-newest order.
        if torch.any(advance_mask):
            next_history = torch.cat(
                (self.history[:, 1:], self.current[:, None, :]), dim=1
            )
            preserve_mask = ~advance_mask & ~initialize_mask
            if torch.any(preserve_mask):
                next_history[preserve_mask] = self.history[preserve_mask]
            self.history = next_history

        initialize_ids = torch.nonzero(initialize_mask, as_tuple=False).flatten()
        if initialize_ids.numel() > 0:
            initial = new_observation[initialize_ids]
            self.history[initialize_ids] = initial[:, None, :].expand(
                -1, HISTORY_LENGTH, -1
            )

        self.current[valid_mask] = new_observation[valid_mask]
        self.initialized[valid_mask] = True


@dataclass
class ObservationDelayState:
    """Per-environment integer policy-step observation latency."""

    buffer: torch.Tensor
    initialized: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        max_delay_steps: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ObservationDelayState":
        if max_delay_steps < 0:
            raise ValueError("max_delay_steps cannot be negative")
        return cls(
            buffer=torch.zeros(
                (num_envs, max_delay_steps + 1, ACTOR_OBSERVATION_DIM),
                device=device,
                dtype=dtype,
            ),
            initialized=torch.zeros(num_envs, device=device, dtype=torch.bool),
        )

    @property
    def max_delay_steps(self) -> int:
        return self.buffer.shape[1] - 1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.buffer[ids] = 0.0
        self.initialized[ids] = False

    def apply(
        self,
        observation: torch.Tensor,
        delay_steps: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observation.shape != (self.buffer.shape[0], ACTOR_OBSERVATION_DIM):
            raise ValueError("observation must have shape [N,96]")
        if delay_steps.shape != self.initialized.shape or delay_steps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("delay_steps must be an integer tensor with shape [N]")
        if torch.any((delay_steps < 0) | (delay_steps > self.max_delay_steps)):
            raise ValueError("observation delay exceeds the allocated buffer")
        if active_mask is None:
            active_mask = torch.ones_like(self.initialized)
        if active_mask.shape != self.initialized.shape or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean with shape [N]")

        initialize = active_mask & ~self.initialized
        initialize_ids = torch.nonzero(initialize, as_tuple=False).flatten()
        if initialize_ids.numel() > 0:
            self.buffer[initialize_ids] = observation[initialize_ids, None, :]
            self.initialized[initialize_ids] = True

        advance = active_mask & self.initialized & ~initialize
        advance_ids = torch.nonzero(advance, as_tuple=False).flatten()
        if advance_ids.numel() > 0:
            selected = self.buffer[advance_ids].clone()
            selected[:, :-1] = selected[:, 1:].clone()
            selected[:, -1] = observation[advance_ids]
            self.buffer[advance_ids] = selected

        result = observation.clone()
        ready_ids = torch.nonzero(active_mask & self.initialized, as_tuple=False).flatten()
        if ready_ids.numel() > 0:
            read_index = self.max_delay_steps - delay_steps[ready_ids]
            result[ready_ids] = self.buffer[ready_ids, read_index]
        return result


def normalize_to_minus_one_one(
    value: torch.Tensor, lower: torch.Tensor | float, upper: torch.Tensor | float
) -> torch.Tensor:
    """Map a configured training range to [-1,1], with no runtime normalizer."""

    low = torch.as_tensor(lower, device=value.device, dtype=value.dtype)
    high = torch.as_tensor(upper, device=value.device, dtype=value.dtype)
    if torch.any(high <= low):
        raise ValueError("normalization upper bound must exceed lower bound")
    return torch.clamp(2.0 * (value - low) / (high - low) - 1.0, -1.0, 1.0)


def assemble_teacher_extrinsics(
    values: Mapping[str, torch.Tensor],
    bounds: Mapping[str, tuple[torch.Tensor | float, torch.Tensor | float]],
    ordered_names: Sequence[str],
) -> torch.Tensor:
    """Normalize and concatenate only independent randomized extrinsics."""

    if not ordered_names:
        raise ValueError("teacher extrinsics cannot be empty")
    normalized: list[torch.Tensor] = []
    for name in ordered_names:
        if name in DERIVED_OR_PRIVILEGED_ONLY_NAMES:
            raise ValueError(f"derived quantity {name!r} cannot enter the teacher encoder")
        if name not in values or name not in bounds:
            raise KeyError(f"missing value or bounds for extrinsic {name!r}")
        value = values[name]
        low, high = bounds[name]
        item = normalize_to_minus_one_one(value, low, high)
        if item.ndim == 1:
            item = item.unsqueeze(-1)
        normalized.append(item)
    return torch.cat(normalized, dim=-1)


def _resolve_asset(env: Any, asset_cfg: Any | None) -> Any:
    name = "robot" if asset_cfg is None else getattr(asset_cfg, "name", "robot")
    return env.scene[name]


def _policy_joint_ids(env: Any, asset_cfg: Any | None) -> Any:
    if asset_cfg is not None:
        ids = getattr(asset_cfg, "joint_ids", None)
        if ids is not None and not isinstance(ids, slice):
            return ids
    return env.policy_joint_ids


def _reference(env: Any) -> torch.Tensor:
    result = env.action_state.q_ref
    if result.shape[-1] != ACTION_DIM:
        raise ValueError("q_ref must use the fixed 29-joint checkpoint order")
    return result


def base_angular_velocity(env: Any, asset_cfg: Any | None = None) -> torch.Tensor:
    asset = _resolve_asset(env, asset_cfg)
    return asset.data.root_ang_vel_b * BASE_ANGULAR_VELOCITY_SCALE


def projected_gravity(env: Any, asset_cfg: Any | None = None) -> torch.Tensor:
    asset = _resolve_asset(env, asset_cfg)
    return asset.data.projected_gravity_b


def joint_velocity(env: Any, asset_cfg: Any | None = None) -> torch.Tensor:
    asset = _resolve_asset(env, asset_cfg)
    return asset.data.joint_vel[:, _policy_joint_ids(env, asset_cfg)] * JOINT_VELOCITY_SCALE


def previous_processed_action(env: Any) -> torch.Tensor:
    result = env.action_state.target
    if result.shape[-1] != ACTION_DIM:
        raise ValueError("processed action must be 29-D; raw last_action is not accepted")
    return result


def actor_observation(
    env: Any,
    asset_cfg: Any | None = None,
    noise_cfg: ObservationNoiseCfg | None = None,
    *,
    use_cache: bool = True,
) -> torch.Tensor:
    """Isaac Lab observation-manager adapter for the complete 96-D vector."""

    if use_cache and hasattr(env, "_actor_observation_cache"):
        return env._actor_observation_cache
    asset = _resolve_asset(env, asset_cfg)
    ids = _policy_joint_ids(env, asset_cfg)
    if noise_cfg is None:
        noise_cfg = env.actor_observation_noise_cfg
    if noise_cfg is not None:
        noise_cfg = noise_cfg.scaled(env.observation_noise_scale)
    # Component manager functions above are scaled; this call uses the pure
    # assembler so there is a single schema assertion and no double scaling.
    observation = assemble_actor_observation(
        asset.data.root_ang_vel_b,
        asset.data.projected_gravity_b,
        env.command_state.v_ref,
        env.path_state.lateral_error,
        env.path_state.heading_error,
        asset.data.joint_pos[:, ids],
        _reference(env),
        asset.data.joint_vel[:, ids],
        previous_processed_action(env),
        noise_cfg=noise_cfg,
    )
    delay_state = env.observation_delay_state
    if delay_state.max_delay_steps > 0:
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        observation = delay_state.apply(observation, env.observation_delay_steps, active)
    return observation


def current_actor_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        return torch.zeros((env.num_envs, ACTOR_OBSERVATION_DIM), device=env.device)
    return env.observation_history_state.current


def actor_observation_history(env: Any) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        return torch.zeros(
            (env.num_envs, HISTORY_LENGTH, ACTOR_OBSERVATION_DIM), device=env.device
        )
    result = env.observation_history_state.history
    if result is None:
        raise RuntimeError("actor history observation is disabled for this environment")
    if result.shape[1:] != (HISTORY_LENGTH, ACTOR_OBSERVATION_DIM):
        raise RuntimeError("actor history schema must be [N,61,96]")
    return result


def teacher_extrinsics(env: Any, expected_dim: int | None = None) -> torch.Tensor:
    """Manager adapter for independently randomized, pre-normalized extrinsics."""

    value = getattr(env, "normalized_teacher_extrinsics", None)
    if value is None:
        if expected_dim is not None:
            return torch.zeros((env.num_envs, expected_dim), device=env.device)
        raise RuntimeError(
            "reset randomization must install normalized_teacher_extrinsics from independent variables"
        )
    if value.ndim != 2 or value.shape[0] != env.num_envs:
        raise ValueError("normalized_teacher_extrinsics must have shape [N,E]")
    return value


def critic_privileged_state(env: Any, expected_dim: int | None = None) -> torch.Tensor:
    """Manager adapter for teacher extrinsics plus simulator-only diagnostics."""

    if not all(
        hasattr(env, name)
        for name in (
            "analytic_force_state",
            "path_tangent_w",
            "path_lateral_w",
            "path_normal_w",
            "rickshaw_state",
            "stability_state",
        )
    ):
        if expected_dim is not None:
            return torch.zeros((env.num_envs, expected_dim), device=env.device)
        raise RuntimeError("critic privileged state was requested before MDP startup")
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]

    def slope_components(velocity_w: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.sum(velocity_w * env.path_tangent_w, dim=-1),
                torch.sum(velocity_w * env.path_lateral_w, dim=-1),
                torch.sum(velocity_w * env.path_normal_w, dim=-1),
            ),
            dim=-1,
        )

    analytic = env.analytic_force_state
    return torch.cat(
        (
            teacher_extrinsics(env),
            env.slope[:, None],
            slope_components(robot.data.root_lin_vel_w),
            slope_components(cart.data.root_lin_vel_w),
            env.rickshaw_state.pitch[:, None],
            env.rickshaw_state.wheel_normal_force,
            env.rickshaw_state.d6_wrench_w.reshape(env.num_envs, -1),
            env.rickshaw_state.d6_residual[:, None],
            analytic.a_s[:, None],
            torch.stack((analytic.t_s, analytic.t_n), dim=-1),
            env.stability_state.zmp_margin[:, None],
        ),
        dim=-1,
    )


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "HISTORY_LENGTH",
    "INDEPENDENT_EXTRINSIC_NAMES",
    "ObservationHistoryState",
    "ObservationDelayState",
    "ObservationNoiseCfg",
    "actor_observation",
    "actor_observation_history",
    "assemble_actor_observation",
    "assemble_teacher_extrinsics",
    "base_angular_velocity",
    "critic_privileged_state",
    "current_actor_observation",
    "joint_velocity",
    "normalize_to_minus_one_one",
    "previous_processed_action",
    "projected_gravity",
    "teacher_extrinsics",
    "wrap_to_pi",
]
