"""Task environment lifecycle extensions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.events import bootstrap_reset_observation


class G1RickshawRLEnv(ManagerBasedRLEnv):
    """Manager environment whose explicit reset returns a real policy frame."""

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
