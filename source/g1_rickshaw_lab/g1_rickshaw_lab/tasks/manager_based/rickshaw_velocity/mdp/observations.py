"""Fixed-schema actor observations and the exclusive 61-frame history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
)
from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS

from .actions import ACTION_DIM


TEACHER_STATIC_DOMAIN_DIM = TEACHER_STATIC_DIM - 1
SLOPE_LOWER = min(SLOPE_GRADIENTS)
SLOPE_UPPER = max(SLOPE_GRADIENTS)

BASE_ANGULAR_VELOCITY_SLICE = slice(0, 3)
PROJECTED_GRAVITY_SLICE = slice(3, 6)
TASK_SIGNAL_SLICE = slice(6, 9)
JOINT_POSITION_SLICE = slice(9, 38)
JOINT_VELOCITY_SLICE = slice(38, 67)
PREVIOUS_ACTION_SLICE = slice(67, ACTOR_OBSERVATION_DIM)

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
        f"d6.{side}.{kind}.{axis}"
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

    position_error = joint_position - q_ref
    task_scale = torch.tensor(
        TASK_SIGNAL_SCALE,
        device=base_angular_velocity_b.device,
        dtype=base_angular_velocity_b.dtype,
    )
    task = torch.stack((v_ref, lateral_error, wrap_to_pi(heading_error)), dim=-1) * task_scale
    observation = torch.cat(
        (
            base_angular_velocity_b * BASE_ANGULAR_VELOCITY_SCALE,
            projected_gravity_b * PROJECTED_GRAVITY_SCALE,
            task,
            position_error * JOINT_POSITION_SCALE,
            joint_velocity_value * JOINT_VELOCITY_SCALE,
            previous_processed_action * PREVIOUS_ACTION_SCALE,
        ),
        dim=-1,
    )
    if observation.shape[-1] != ACTOR_OBSERVATION_DIM:
        raise RuntimeError(f"actor observation is {observation.shape[-1]}-D, expected {ACTOR_OBSERVATION_DIM}-D")
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
        if history_length != HISTORY_LENGTH or observation_dim <= 0:
            raise ValueError(f"history length is fixed at {HISTORY_LENGTH} and feature dimension must be positive")
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
        if observation.shape != (ids.numel(), self.current.shape[-1]):
            raise ValueError("initial history observation has the wrong shape")
        if self.history is not None:
            self.history[ids] = observation[:, None, :].expand(-1, HISTORY_LENGTH, -1)
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
        next_history[initialize_mask] = initial[:, None, :].expand(-1, HISTORY_LENGTH, -1)
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
) -> torch.Tensor:
    """Isaac Lab observation-manager adapter for the complete 96-D vector."""

    asset = _resolve_asset(env, asset_cfg)
    ids = _policy_joint_ids(env, asset_cfg)
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
    )
    if env.cfg.observation_noise_enabled:
        observation += torch.empty_like(observation).uniform_(-1.0, 1.0) * env.actor_observation_noise_scale
    return observation


def _is_observation_shape_probe(env: Any) -> bool:
    """Isaac Lab calls terms once while constructing, before binding, its manager."""

    return not hasattr(env, "observation_manager")


def _shape_placeholder(env: Any, *feature_shape: int) -> torch.Tensor:
    return torch.empty((), device=env.device).expand(env.num_envs, *feature_shape)


def current_actor_observation(env: Any) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        if _is_observation_shape_probe(env):
            return _shape_placeholder(env, ACTOR_OBSERVATION_DIM)
        raise RuntimeError("actor observation requested before MDP startup")
    return env.observation_history_state.current


def actor_observation_history(env: Any) -> torch.Tensor:
    if not hasattr(env, "observation_history_state"):
        if _is_observation_shape_probe(env):
            return _shape_placeholder(env, HISTORY_LENGTH, ACTOR_OBSERVATION_DIM)
        raise RuntimeError("actor history requested before MDP startup")
    result = env.observation_history_state.history
    if result is None:
        raise RuntimeError("actor history observation is disabled for this environment")
    if result.shape[1:] != (HISTORY_LENGTH, ACTOR_OBSERVATION_DIM):
        raise RuntimeError(f"actor history schema must be [N,{HISTORY_LENGTH},{ACTOR_OBSERVATION_DIM}]")
    return result


def teacher_static(env: Any, expected_dim: int | None = None) -> torch.Tensor:
    """Return normalized fixed-domain features and terrain slope."""

    if expected_dim is not None and expected_dim != TEACHER_STATIC_DIM:
        raise ValueError("teacher static dimension differs from observation config")
    domain = getattr(env, "normalized_teacher_static_domain", None)
    if domain is None:
        if _is_observation_shape_probe(env):
            return _shape_placeholder(env, TEACHER_STATIC_DIM)
        raise RuntimeError("teacher static privilege requested before domain initialization")
    if domain.shape != (env.num_envs, TEACHER_STATIC_DOMAIN_DIM):
        raise ValueError(f"normalized teacher static domain must have shape [N,{TEACHER_STATIC_DOMAIN_DIM}]")
    slope = torch.clamp(
        2.0 * (env.slope[:, None] - SLOPE_LOWER) / (SLOPE_UPPER - SLOPE_LOWER) - 1.0,
        -1.0,
        1.0,
    )
    result = torch.cat((domain, slope), dim=-1)
    if result.shape != (env.num_envs, TEACHER_STATIC_DIM):
        raise RuntimeError(f"teacher static privilege must have shape [N,{TEACHER_STATIC_DIM}]")
    return result


def _slope_components(env: Any, vector_w: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.sum(vector_w * env.path_tangent_w, dim=-1),
            torch.sum(vector_w * env.path_lateral_w, dim=-1),
            torch.sum(vector_w * env.path_normal_w, dim=-1),
        ),
        dim=-1,
    )


def dynamic_privileged_observation(env: Any) -> torch.Tensor:
    """Assemble the raw 21-D dynamic teacher/critic state in the slope frame."""

    required = (
        "path_tangent_w",
        "path_lateral_w",
        "path_normal_w",
        "rickshaw_state",
    )
    if not all(hasattr(env, name) for name in required):
        raise RuntimeError("dynamic privilege requested before MDP startup")
    robot = env.scene["robot"]
    cart = env.scene["rickshaw"]
    cart_velocity_w = cart.data.root_lin_vel_w
    basis = torch.stack((env.path_tangent_w, env.path_lateral_w, env.path_normal_w), dim=1)
    wrench_w = env.rickshaw_state.d6_truth_wrench_w
    force_sln = torch.einsum("nsw,ncw->nsc", wrench_w[..., :3], basis)
    torque_sln = torch.einsum("nsw,ncw->nsc", wrench_w[..., 3:], basis)
    d6_wrench_sln = torch.cat((force_sln, torque_sln), dim=-1).reshape(env.num_envs, -1)
    result = torch.cat(
        (
            _slope_components(env, robot.data.root_lin_vel_w),
            _slope_components(env, cart_velocity_w),
            env.rickshaw_state.pitch[:, None],
            env.rickshaw_state.wheel_normal_force,
            d6_wrench_sln,
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, TEACHER_DYNAMIC_DIM):
        raise RuntimeError(f"dynamic privilege must have shape [N,{TEACHER_DYNAMIC_DIM}]")
    return result


def teacher_dynamic_history(env: Any, expected_dim: int | None = None) -> torch.Tensor:
    """Return the causal 61-frame history of raw dynamic privilege."""

    if expected_dim is not None and expected_dim != TEACHER_DYNAMIC_DIM:
        raise ValueError("teacher dynamic dimension differs from observation config")
    state = getattr(env, "teacher_dynamic_history_state", None)
    if state is None:
        if _is_observation_shape_probe(env):
            return _shape_placeholder(env, HISTORY_LENGTH, TEACHER_DYNAMIC_DIM)
        raise RuntimeError("teacher dynamic history requested before MDP startup")
    if state.history is None or state.history.shape != (
        env.num_envs,
        HISTORY_LENGTH,
        TEACHER_DYNAMIC_DIM,
    ):
        raise RuntimeError(f"teacher dynamic history must have shape [N,{HISTORY_LENGTH},{TEACHER_DYNAMIC_DIM}]")
    return state.history


def critic_privileged_state(env: Any, expected_dim: int | None = None) -> torch.Tensor:
    """Return static10 + current dynamic21 + residual/ZMP/acceleration3."""

    if expected_dim is not None and expected_dim != CRITIC_PRIVILEGED_DIM:
        raise ValueError("critic privilege dimension differs from observation config")
    state = getattr(env, "teacher_dynamic_history_state", None)
    if state is None:
        if _is_observation_shape_probe(env):
            return _shape_placeholder(env, CRITIC_PRIVILEGED_DIM)
        raise RuntimeError("critic privilege requested before MDP startup")
    result = torch.cat(
        (
            teacher_static(env),
            state.current,
            env.rickshaw_state.d6_residual[:, None],
            env.stability_state.zmp_margin[:, None],
            env.analytic_force_state.a_s[:, None],
        ),
        dim=-1,
    )
    if result.shape != (env.num_envs, CRITIC_PRIVILEGED_DIM):
        raise RuntimeError(f"critic privilege must have shape [N,{CRITIC_PRIVILEGED_DIM}]")
    return result


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "CRITIC_PRIVILEGED_DIM",
    "HISTORY_LENGTH",
    "TEACHER_DYNAMIC_DIM",
    "TEACHER_DYNAMIC_FEATURE_NAMES",
    "TEACHER_STATIC_DIM",
    "TEACHER_STATIC_DOMAIN_DIM",
    "TEACHER_STATIC_FEATURE_NAMES",
    "ACTOR_OBSERVATION_NOISE_SCALE",
    "ObservationHistoryState",
    "actor_observation",
    "actor_observation_history",
    "assemble_actor_observation",
    "base_angular_velocity",
    "critic_privileged_state",
    "current_actor_observation",
    "dynamic_privileged_observation",
    "joint_velocity",
    "previous_processed_action",
    "projected_gravity",
    "normalize_features",
    "teacher_dynamic_history",
    "teacher_static",
    "wrap_to_pi",
]
