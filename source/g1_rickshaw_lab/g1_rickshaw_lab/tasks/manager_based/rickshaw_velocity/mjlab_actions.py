"""Mjlab action term with a per-environment static-equilibrium reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .mdp.actions import ButterworthActionState, action_scale_vector


@dataclass(kw_only=True)
class StaticReferenceJointPositionActionCfg(BaseActionCfg):
    """Joint-position action centered on the reset solver's current ``q_ref``."""

    def build(self, env):
        return StaticReferenceJointPositionAction(self, env)


class StaticReferenceJointPositionAction(BaseAction):
    cfg: StaticReferenceJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        q_ref = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self.state = ButterworthActionState.create(q_ref)
        self._scale = action_scale_vector(device=self.device).unsqueeze(0)
        if self.action_dim != self._scale.shape[-1]:
            raise ValueError("static-reference action must control the fixed 29-joint policy order")
        env.action_state = self.state

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.state.target

    def set_reference(self, q_ref: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self.state.reset(q_ref, env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions = self.state.process(actions, self._scale)

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            self.state.target - encoder_bias,
            joint_ids=self._target_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)
        self.state.reset(self.state.q_ref[ids], ids)
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = self.state.target[ids]


__all__ = [
    "StaticReferenceJointPositionAction",
    "StaticReferenceJointPositionActionCfg",
]
