"""Action processing for the 29-DoF G1 policy.

The filter state in this module deliberately has no Isaac Lab dependency.  This
keeps the deployment-side action contract and the simulator action term on the
same implementation.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Any

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
ACTION_GROUP_SCALES = {
    "lower": ACTION_SCALE[0],
    "waist": ACTION_SCALE[12],
    "shoulder": ACTION_SCALE[15],
    "elbow": ACTION_SCALE[18],
    "wrist": ACTION_SCALE[19],
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
    """Return one action-scale row after validating Isaac Lab's batched representation."""

    value = torch.as_tensor(scale, device=device)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1).expand(action_dim)
    if value.shape[-1] != action_dim:
        raise ValueError(
            f"action scale must end in dimension {action_dim}, got {tuple(value.shape)}"
        )

    rows = value.reshape(-1, action_dim)
    if rows.shape[0] not in (1, num_envs):
        raise ValueError(
            f"action scale has {rows.shape[0]} rows; expected one or {num_envs}"
        )
    reference = rows[0]
    if rows.shape[0] > 1 and not torch.allclose(
        rows, reference.unsqueeze(0).expand_as(rows), rtol=0.0, atol=0.0
    ):
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


def gain_compensated_static_target(
    q_reset: torch.Tensor,
    nominal_target: torch.Tensor,
    motor_strength: torch.Tensor,
    joint_model_error: torch.Tensor,
) -> torch.Tensor:
    """Convert a nominal static PD target to the episode's effective gains."""

    if q_reset.ndim != 2 or nominal_target.shape != q_reset.shape:
        raise ValueError("static target inputs must have identical [N,D] shapes")
    if motor_strength.ndim != 1 or motor_strength.shape[0] != q_reset.shape[0]:
        raise ValueError("motor_strength must have shape [N]")
    if joint_model_error.shape != q_reset.shape:
        raise ValueError("joint_model_error must have shape [N,D]")
    gain = motor_strength[:, None] * (1.0 + joint_model_error)
    if torch.any(~torch.isfinite(gain)) or torch.any(gain <= 0.0):
        raise ValueError("effective static actuator gains must be finite and positive")
    return q_reset + (nominal_target - q_reset) / gain


@dataclass
class ButterworthActionState:
    """Mutable per-environment state for filtered joint-position actions."""

    q_ref: torch.Tensor
    raw_action: torch.Tensor
    x_prev: torch.Tensor
    y_prev: torch.Tensor
    target: torch.Tensor
    prev_target: torch.Tensor
    prev_prev_target: torch.Tensor

    @classmethod
    def create(cls, q_ref: torch.Tensor) -> "ButterworthActionState":
        if q_ref.ndim != 2:
            raise ValueError(f"q_ref must have shape [N, D], got {tuple(q_ref.shape)}")
        reference = q_ref.clone()
        zeros = torch.zeros_like(reference)
        return cls(
            q_ref=reference,
            raw_action=zeros.clone(),
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
        self.raw_action[ids] = normalized_action
        self.prev_prev_target[ids] = self.prev_target[ids]
        self.prev_target[ids] = self.target[ids]
        self.target[ids] = y
        self.x_prev[ids] = x
        self.y_prev[ids] = y
        return y


@dataclass
class ControlDelayState:
    """Integer policy-step delay applied before the deployment action filter."""

    buffer: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        action_dim: int,
        max_delay_steps: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ControlDelayState":
        if max_delay_steps < 0:
            raise ValueError("max_delay_steps cannot be negative")
        return cls(
            buffer=torch.zeros(
                (num_envs, max_delay_steps + 1, action_dim),
                device=device,
                dtype=dtype,
            )
        )

    @property
    def max_delay_steps(self) -> int:
        return self.buffer.shape[1] - 1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.buffer[ids] = 0.0

    def apply(self, action: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
        if action.shape != (self.buffer.shape[0], self.buffer.shape[2]):
            raise ValueError("action shape differs from the control-delay buffer")
        if delay_steps.shape != (self.buffer.shape[0],) or delay_steps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("delay_steps must be an integer tensor with shape [N]")
        if torch.any((delay_steps < 0) | (delay_steps > self.max_delay_steps)):
            raise ValueError("control delay exceeds the allocated buffer")
        self.buffer[:, :-1] = self.buffer[:, 1:].clone()
        self.buffer[:, -1] = action
        read_index = self.max_delay_steps - delay_steps
        env_index = torch.arange(action.shape[0], device=action.device)
        return self.buffer[env_index, read_index]


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


def _resolve_nested_attr(obj: Any, path: str) -> Any:
    result = obj
    for item in path.split("."):
        result = getattr(result, item)
    return result


try:  # Isaac Lab is intentionally optional for CPU-only validation.
    from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the lightweight test environment.
    JointPositionAction = object  # type: ignore[assignment,misc]
    JointPositionActionCfg = object  # type: ignore[assignment,misc]
    ISAACLAB_AVAILABLE = False

    def configclass(cls: type) -> type:
        return cls


if ISAACLAB_AVAILABLE:

    class FilteredJointPositionAction(JointPositionAction):
        """Isaac Lab action term using the episode's closed-chain IK reference."""

        cfg: "FilteredJointPositionActionCfg"

        def __init__(self, cfg: "FilteredJointPositionActionCfg", env: Any):
            super().__init__(cfg, env)
            self._env = env
            self._reference_indices = cfg.reference_indices
            self._new_policy_action = False
            if cfg.physics_hook_owner:
                if hasattr(env, "_rickshaw_physics_hook_action_term"):
                    raise RuntimeError(
                        "exactly one FilteredJointPositionAction may own rickshaw physics hooks"
                    )
                env._rickshaw_physics_hook_action_term = self
            q_ref = self._read_q_ref(require=False)
            self._filter_state = ButterworthActionState.create(q_ref)

        def _read_q_ref(self, *, require: bool = True) -> torch.Tensor:
            try:
                q_ref = _resolve_nested_attr(self._env, self.cfg.reference_attribute)
            except AttributeError:
                if require:
                    raise RuntimeError(
                        "closed-chain q_ref was not installed before ActionTerm reset"
                    ) from None
                return torch.zeros(
                    (self._env.num_envs, self.action_dim),
                    device=self._env.device,
                    dtype=self._raw_actions.dtype,
                )
            if not torch.is_tensor(q_ref) or q_ref.ndim != 2:
                raise ValueError(f"{self.cfg.reference_attribute} must be a [N, D] tensor")
            if self._reference_indices is not None:
                q_ref = q_ref[:, self._reference_indices]
            elif q_ref.shape[-1] == self.action_dim:
                pass
            elif q_ref.shape[-1] == self._asset.num_joints:
                q_ref = q_ref[:, self._joint_ids]
            else:
                raise ValueError(
                    "q_ref is neither term-local nor articulation-sized; set reference_indices "
                    "to the persisted checkpoint indices"
                )
            if q_ref.shape[-1] != self.action_dim:
                raise ValueError("resolved q_ref dimension differs from this ActionTerm")
            return q_ref

        def process_actions(self, actions: torch.Tensor) -> None:
            self._raw_actions[:] = actions
            effective_actions = actions
            max_delay = int(self._env.max_control_delay_steps)
            if max_delay > 0:
                state = getattr(self, "_control_delay_state", None)
                if state is None or state.max_delay_steps != max_delay:
                    state = ControlDelayState.zeros(
                        self._env.num_envs,
                        self.action_dim,
                        max_delay,
                        device=self._env.device,
                        dtype=actions.dtype,
                    )
                    self._control_delay_state = state
                effective_actions = state.apply(
                    effective_actions, self._env.control_delay_steps
                )
            target = self._filter_state.process(effective_actions, self._scale)
            self._processed_actions[:] = target
            self._new_policy_action = True

        def _sync_global_action_state(self) -> None:
            terms = self._env.action_manager._terms.values()
            target = torch.cat([term.processed_actions for term in terms], dim=-1)
            if target.shape[-1] != ACTION_DIM:
                raise RuntimeError(
                    f"ActionManager processed target is not {ACTION_DIM}-D"
                )
            state = self._env.action_state
            state.prev_prev_target[:] = state.prev_target
            state.prev_target[:] = state.target
            state.target[:] = target
            state.raw_action[:] = torch.cat([term.raw_actions for term in terms], dim=-1)

        def apply_actions(self) -> None:
            if self.cfg.physics_hook_owner:
                self._env._g1_rickshaw_pre_physics_step()
            # The reset state already includes the full static handle preload.
            # Every substep therefore uses the normal policy controller target.
            self._processed_actions[:] = self._filter_state.target
            super().apply_actions()
            if not self.cfg.physics_hook_owner:
                return
            if self._new_policy_action:
                self._sync_global_action_state()
                self._new_policy_action = False

        def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
            if env_ids is None or isinstance(env_ids, slice):
                ids = None
            else:
                ids = env_ids
            q_ref = self._read_q_ref()
            self._filter_state.reset(q_ref if ids is None else q_ref[ids], ids)
            if hasattr(self, "_control_delay_state"):
                self._control_delay_state.reset(ids)
            if ids is None:
                self._raw_actions.zero_()
                self._processed_actions[:] = q_ref
            else:
                self._raw_actions[ids] = 0.0
                self._processed_actions[ids] = q_ref[ids]

        @property
        def filter_state(self) -> ButterworthActionState:
            return self._filter_state


    @configclass
    class FilteredJointPositionActionCfg(JointPositionActionCfg):
        """Configuration for :class:`FilteredJointPositionAction`.

        ``reference_indices`` is required when the shared reference tensor is in
        the global 29-joint checkpoint order and this term controls one group.
        """

        class_type: type = FilteredJointPositionAction
        reference_attribute: str = "action_state.q_ref"
        reference_indices: tuple[int, ...] | None = None
        physics_hook_owner: bool = False

else:

    class FilteredJointPositionAction:  # pragma: no cover - import compatibility only.
        def __init__(self, *_: Any, **__: Any):
            raise RuntimeError("FilteredJointPositionAction requires Isaac Lab")


    @dataclass
    class FilteredJointPositionActionCfg:
        """Importable stand-in used by non-Isaac unit tests."""

        joint_names: tuple[str, ...] = MISSING
        scale: float | tuple[float, ...] = MISSING
        asset_name: str = "robot"
        preserve_order: bool = True
        reference_attribute: str = "action_state.q_ref"
        reference_indices: tuple[int, ...] | None = None
        physics_hook_owner: bool = False
        class_type: type = FilteredJointPositionAction


__all__ = [
    "ACTION_DIM",
    "ACTION_GROUP_DIMS",
    "ACTION_GROUP_SCALES",
    "ARM_ACTION_START_INDEX",
    "BUTTERWORTH_A1",
    "BUTTERWORTH_B0",
    "BUTTERWORTH_B1",
    "ButterworthActionState",
    "ControlDelayState",
    "FilteredJointPositionAction",
    "FilteredJointPositionActionCfg",
    "action_scale_vector",
    "canonicalize_action_scale",
    "butterworth_dc_gain",
    "butterworth_filter_step",
    "gain_compensated_static_target",
    "butterworth_gain",
]
