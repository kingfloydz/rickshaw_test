"""Action processing for the 29-DoF G1 policy.

The filter state in this module deliberately has no simulator dependency. This
keeps the deployment-side action contract and the simulator action term on the
same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTION_SCALE,
    BUTTERWORTH_A1,
    BUTTERWORTH_B0,
    BUTTERWORTH_B1,
)

ACTION_GROUP_DIMS = {
    "lower": 12,
    "waist": 3,
    "shoulder": 6,
    "elbow": 2,
    "wrist": 6,
}
ARM_ACTION_START_INDEX = ACTION_GROUP_DIMS["lower"] + ACTION_GROUP_DIMS["waist"]


def _check_last_dim(value: torch.Tensor, expected: int, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != expected:
        raise ValueError(f"{name} must have last dimension {expected}, got {tuple(value.shape)}")


def action_scale_vector(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return the fixed 29-D scale vector in checkpoint joint-group order."""

    result = torch.tensor(ACTION_SCALE, device=device, dtype=dtype)
    _check_last_dim(result, ACTION_DIM, "action scale")
    return result


def canonicalize_action_scale(
    scale: torch.Tensor | float,
    action_dim: int,
    num_envs: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return one action-scale row after validating a batched representation."""

    value = torch.as_tensor(scale, device=device)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1).expand(action_dim)
    if value.shape[-1] != action_dim:
        raise ValueError(f"action scale must end in dimension {action_dim}, got {tuple(value.shape)}")

    rows = value.reshape(-1, action_dim)
    if rows.shape[0] not in (1, num_envs):
        raise ValueError(f"action scale has {rows.shape[0]} rows; expected one or {num_envs}")
    reference = rows[0]
    if rows.shape[0] > 1 and not torch.allclose(rows, reference.unsqueeze(0).expand_as(rows), rtol=0.0, atol=0.0):
        raise ValueError("action scale differs between environments")
    return reference


def butterworth_filter_step(
    normalized_action: torch.Tensor,
    scale: torch.Tensor | float,
    q_ref: torch.Tensor,
    x_prev: torch.Tensor,
    y_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one normalized-action mapping and Butterworth filter step.

    Returns ``(x, y)`` where ``x`` is the unfiltered position target and ``y``
    is the target that must be sent to the articulation.
    """

    if normalized_action.shape != q_ref.shape:
        raise ValueError(
            "normalized_action and q_ref must have identical shapes, got "
            f"{tuple(normalized_action.shape)} and {tuple(q_ref.shape)}"
        )
    if x_prev.shape != q_ref.shape or y_prev.shape != q_ref.shape:
        raise ValueError("filter histories must have the same shape as q_ref")
    if not torch.is_tensor(scale):
        scale = torch.as_tensor(scale, device=normalized_action.device, dtype=normalized_action.dtype)
    else:
        scale = scale.to(device=normalized_action.device, dtype=normalized_action.dtype)
    if scale.ndim > 0 and scale.shape[-1] not in (1, normalized_action.shape[-1]):
        raise ValueError("action scale is not broadcastable over the action dimension")

    x = normalized_action * scale + q_ref
    y = BUTTERWORTH_B0 * x + BUTTERWORTH_B1 * x_prev - BUTTERWORTH_A1 * y_prev
    return x, y


@dataclass
class ButterworthActionState:
    """Mutable per-environment state for filtered joint-position actions."""

    q_ref: torch.Tensor
    raw_action: torch.Tensor
    prev_raw_action: torch.Tensor
    x_prev: torch.Tensor
    y_prev: torch.Tensor
    target: torch.Tensor
    prev_target: torch.Tensor
    prev_prev_target: torch.Tensor

    @classmethod
    def create(cls, q_ref: torch.Tensor) -> ButterworthActionState:
        if q_ref.ndim != 2:
            raise ValueError(f"q_ref must have shape [N, D], got {tuple(q_ref.shape)}")
        reference = q_ref.clone()
        zeros = torch.zeros_like(reference)
        return cls(
            q_ref=reference,
            raw_action=zeros.clone(),
            prev_raw_action=zeros.clone(),
            x_prev=reference.clone(),
            y_prev=reference.clone(),
            target=reference.clone(),
            prev_target=reference.clone(),
            prev_prev_target=reference.clone(),
        )

    @property
    def num_envs(self) -> int:
        return self.q_ref.shape[0]

    @property
    def action_dim(self) -> int:
        return self.q_ref.shape[1]

    def reset(self, q_ref: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set the episode-fixed reference and initialize every history to it."""

        if env_ids is None:
            if q_ref.shape != self.q_ref.shape:
                raise ValueError("full reset q_ref has the wrong shape")
            ids: slice | torch.Tensor = slice(None)
        else:
            env_ids = env_ids.to(device=self.q_ref.device, dtype=torch.long)
            if q_ref.shape == self.q_ref.shape:
                q_ref = q_ref[env_ids]
            if q_ref.shape != (env_ids.numel(), self.action_dim):
                raise ValueError("partial reset q_ref has the wrong shape")
            ids = env_ids

        self.q_ref[ids] = q_ref
        self.raw_action[ids] = 0.0
        self.prev_raw_action[ids] = 0.0
        self.x_prev[ids] = q_ref
        self.y_prev[ids] = q_ref
        self.target[ids] = q_ref
        self.prev_target[ids] = q_ref
        self.prev_prev_target[ids] = q_ref

    def process(
        self,
        normalized_action: torch.Tensor,
        scale: torch.Tensor | float,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process actions and advance the target rate/jerk histories."""

        if env_ids is None:
            ids: slice | torch.Tensor = slice(None)
            expected_shape = self.q_ref.shape
        else:
            env_ids = env_ids.to(device=self.q_ref.device, dtype=torch.long)
            ids = env_ids
            expected_shape = (env_ids.numel(), self.action_dim)
        if normalized_action.shape != expected_shape:
            raise ValueError(
                f"normalized_action must have shape {expected_shape}, got {tuple(normalized_action.shape)}"
            )

        x, y = butterworth_filter_step(
            normalized_action,
            scale,
            self.q_ref[ids],
            self.x_prev[ids],
            self.y_prev[ids],
        )
        self.prev_raw_action[ids] = self.raw_action[ids]
        self.raw_action[ids] = normalized_action
        self.prev_prev_target[ids] = self.prev_target[ids]
        self.prev_target[ids] = self.target[ids]
        self.target[ids] = y
        self.x_prev[ids] = x
        self.y_prev[ids] = y
        return y


def butterworth_dc_gain() -> float:
    """Return the filter's DC gain (one within coefficient rounding)."""

    return (BUTTERWORTH_B0 + BUTTERWORTH_B1) / (1.0 + BUTTERWORTH_A1)


def butterworth_gain(frequency_hz: float, sample_rate_hz: float = 50.0) -> float:
    """Return the discrete filter magnitude at ``frequency_hz``."""

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    omega = torch.tensor(2.0 * torch.pi * frequency_hz / sample_rate_hz, dtype=torch.float64)
    z_inv = torch.exp(-1j * omega)
    response = (BUTTERWORTH_B0 + BUTTERWORTH_B1 * z_inv) / (1.0 + BUTTERWORTH_A1 * z_inv)
    return float(torch.abs(response))


__all__ = [
    "ACTION_DIM",
    "ACTION_GROUP_DIMS",
    "ARM_ACTION_START_INDEX",
    "BUTTERWORTH_A1",
    "BUTTERWORTH_B0",
    "BUTTERWORTH_B1",
    "ButterworthActionState",
    "action_scale_vector",
    "canonicalize_action_scale",
    "butterworth_dc_gain",
    "butterworth_filter_step",
    "butterworth_gain",
]
