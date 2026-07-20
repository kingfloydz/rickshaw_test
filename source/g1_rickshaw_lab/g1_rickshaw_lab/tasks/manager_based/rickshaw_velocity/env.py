"""Task environment lifecycle extensions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaacsim.core.simulation_manager import SimulationManager

from .mdp.events import bootstrap_reset_observation
from .runtime import RickshawRuntime


class G1RickshawRLEnv(ManagerBasedRLEnv):
    """Manager environment whose explicit reset returns a real policy frame."""

    rickshaw_runtime: RickshawRuntime

    @property
    def command_state(self):
        return self.rickshaw_runtime.command

    @property
    def path_state(self):
        return self.rickshaw_runtime.path

    @property
    def rickshaw_state(self):
        return self.rickshaw_runtime.cart

    @property
    def stability_state(self):
        return self.rickshaw_runtime.stability

    @property
    def action_state(self):
        return self.rickshaw_runtime.action

    @property
    def analytic_force_state(self):
        return self.rickshaw_runtime.analytic_force

    @property
    def cart_interaction_wrench_state(self):
        return self.rickshaw_runtime.cart_interaction_wrench

    @property
    def observation_history_state(self):
        return self.rickshaw_runtime.observation_history

    @property
    def teacher_dynamic_history_state(self):
        return self.rickshaw_runtime.teacher_dynamic_history

    @property
    def termination_state(self):
        return self.rickshaw_runtime.termination

    @property
    def termination_cause_state(self):
        return self.rickshaw_runtime.termination_causes

    def read_d6_reaction_residual(self):
        return self.d6_reaction_adapter.read()

    def _g1_rickshaw_pre_physics_step(self) -> None:
        from .mdp.dynamics import accumulate_cart_interaction_wrench, apply_rolling_resistance

        rolling_force_w = apply_rolling_resistance(self, self.rickshaw_runtime.rolling_resistance_cfg)
        accumulate_cart_interaction_wrench(self, rolling_force_w)

    def write_closed_chain_reset_state(self, env_ids: torch.Tensor) -> None:
        from .mdp.events import write_closed_chain_reset_state

        write_closed_chain_reset_state(self, env_ids)

    def _bootstrap_reset_observations(self, env_ids: torch.Tensor) -> None:
        bootstrap_reset_observation(self, env_ids, self.cfg.policy_update)

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset selected environments and initialize their policy history.

        Isaac Lab's base reset computes observations without running interval
        events.  This task needs one policy-rate command/reference tick after
        kinematics are forwarded, matching the frame produced for environments
        reset inside :meth:`step`.
        """

        del options
        if env_ids is None:
            reset_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            reset_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        self.recorder_manager.record_pre_reset(reset_ids)
        if seed is not None:
            self.seed(seed)

        self._reset_idx(reset_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        self.recorder_manager.record_post_reset(reset_ids)
        self._bootstrap_reset_observations(reset_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        return self.obs_buf, self.extras


__all__ = ["G1RickshawRLEnv"]
