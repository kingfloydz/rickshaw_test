"""Isaac Lab action-term adapter for the pure action filter."""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from .actions import ACTION_DIM, ButterworthActionState


def _resolve_nested_attr(obj: Any, path: str) -> Any:
    result = obj
    for item in path.split("."):
        result = getattr(result, item)
    return result


class FilteredJointPositionAction(JointPositionAction):
    """Joint-position action using the episode's closed-chain reference."""

    cfg: FilteredJointPositionActionCfg

    def __init__(self, cfg: FilteredJointPositionActionCfg, env: Any):
        super().__init__(cfg, env)
        self._env = env
        self._reference_indices = cfg.reference_indices
        self._new_policy_action = False
        if cfg.physics_hook_owner:
            if hasattr(env, "_rickshaw_physics_hook_action_term"):
                raise RuntimeError("exactly one action term may own rickshaw physics hooks")
            env._rickshaw_physics_hook_action_term = self
        self._filter_state = ButterworthActionState.create(self._read_q_ref(require=False))

    def _read_q_ref(self, *, require: bool = True) -> torch.Tensor:
        try:
            q_ref = _resolve_nested_attr(self._env, self.cfg.reference_attribute)
        except AttributeError:
            if require:
                raise RuntimeError("closed-chain q_ref was not installed before ActionTerm reset") from None
            return torch.zeros(
                (self._env.num_envs, self.action_dim),
                device=self._env.device,
                dtype=self._raw_actions.dtype,
            )
        if not torch.is_tensor(q_ref) or q_ref.ndim != 2:
            raise ValueError(f"{self.cfg.reference_attribute} must be a [N, D] tensor")
        if self._reference_indices is not None:
            q_ref = q_ref[:, self._reference_indices]
        elif q_ref.shape[-1] == self._asset.num_joints:
            q_ref = q_ref[:, self._joint_ids]
        if q_ref.shape[-1] != self.action_dim:
            raise ValueError("resolved q_ref dimension differs from this ActionTerm")
        return q_ref

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._filter_state.process(actions, self._scale)
        self._new_policy_action = True

    def _sync_global_action_state(self) -> None:
        terms = self._env.action_manager._terms.values()
        target = torch.cat([term.processed_actions for term in terms], dim=-1)
        if target.shape[-1] != ACTION_DIM:
            raise RuntimeError(f"ActionManager processed target is not {ACTION_DIM}-D")
        state = self._env.action_state
        state.prev_prev_target[:] = state.prev_target
        state.prev_target[:] = state.target
        state.target[:] = target
        state.raw_action[:] = torch.cat([term.raw_actions for term in terms], dim=-1)

    def apply_actions(self) -> None:
        if self.cfg.physics_hook_owner:
            self._env._g1_rickshaw_pre_physics_step()
        self._processed_actions[:] = self._filter_state.target
        super().apply_actions()
        if self.cfg.physics_hook_owner and self._new_policy_action:
            self._sync_global_action_state()
            self._new_policy_action = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = None if env_ids is None or isinstance(env_ids, slice) else env_ids
        q_ref = self._read_q_ref()
        self._filter_state.reset(q_ref if ids is None else q_ref[ids], ids)
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
    """Configuration for :class:`FilteredJointPositionAction`."""

    class_type: type = FilteredJointPositionAction
    reference_attribute: str = "action_state.q_ref"
    reference_indices: tuple[int, ...] | None = None
    physics_hook_owner: bool = False


__all__ = ["FilteredJointPositionAction", "FilteredJointPositionActionCfg"]
