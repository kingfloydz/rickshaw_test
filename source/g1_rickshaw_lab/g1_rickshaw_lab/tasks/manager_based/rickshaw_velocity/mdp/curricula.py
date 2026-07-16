"""Single-stage training distribution and task-specific terrain curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from g1_rickshaw_lab.slope_contract import MAX_TRAINING_DOWNHILL_LEVEL
from .dynamics import update_slope_frame


class CurriculumStage(IntEnum):
    TRAINING = 1


@dataclass(kw_only=True)
class CurriculumScheduleCfg:
    """Iteration synchronization for the single training distribution."""

    rollout_steps_per_iteration: int = 48

    def validate(self) -> None:
        if self.rollout_steps_per_iteration <= 0:
            raise ValueError("rollout_steps_per_iteration must be positive")


def curriculum_stage_for_iteration(
    iteration: int,
) -> CurriculumStage:
    """Resolve the sole training stage while validating the iteration."""

    if iteration < 0:
        raise ValueError("curriculum iteration cannot be negative")
    return CurriculumStage.TRAINING


@dataclass
class CurriculumRuntimeState:
    """Explicit iteration state for one training distribution."""

    cfg: CurriculumScheduleCfg
    num_envs: int
    device: torch.device
    iteration: int = 0
    stage: CurriculumStage = CurriculumStage.TRAINING

    @classmethod
    def create(
        cls,
        terrain_strata: torch.Tensor,
        terrain_direction: torch.Tensor,
        cfg: CurriculumScheduleCfg,
    ) -> "CurriculumRuntimeState":
        cfg.validate()
        if terrain_strata.shape != terrain_direction.shape:
            raise ValueError("terrain strata and directions must have identical shapes")
        return cls(cfg=cfg, num_envs=terrain_strata.numel(), device=terrain_strata.device)

    def set_iteration(self, iteration: int) -> CurriculumStage:
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError("curriculum iteration must be an integer")
        if iteration < 0:
            raise ValueError("curriculum iteration cannot be negative")
        self.iteration = iteration
        self.stage = curriculum_stage_for_iteration(iteration)
        return self.stage

    def stage_per_environment(self) -> torch.Tensor:
        return torch.full(
            (self.num_envs,), int(self.stage), dtype=torch.long, device=self.device
        )

    def distribution(self) -> dict[str, int]:
        stage = self.stage_per_environment()
        return {
            item.name: int(torch.sum(stage == int(item)).item())
            for item in CurriculumStage
        }


def speed_tracking_score(v_ref: torch.Tensor, actual_speed: torch.Tensor) -> torch.Tensor:
    """Per-step curriculum score from section 11.5."""

    if v_ref.shape != actual_speed.shape:
        raise ValueError("v_ref and actual_speed shapes differ")
    return torch.exp(-torch.square((v_ref - actual_speed) / 0.25))


def terrain_level_delta(
    mean_score: torch.Tensor,
    timed_out: torch.Tensor,
    safety_triggered: torch.Tensor,
    terminated_early: torch.Tensor,
) -> torch.Tensor:
    """Return -1/0/+1 according to the exact timeout and score gates."""

    if not (
        mean_score.shape
        == timed_out.shape
        == safety_triggered.shape
        == terminated_early.shape
    ):
        raise ValueError("all curriculum decision tensors must have identical shapes")
    move_up = timed_out & (mean_score >= 0.8) & ~safety_triggered
    move_down = terminated_early | (mean_score < 0.5)
    # An early/safety failure always wins over a nominal timeout flag.
    move_up &= ~move_down
    return move_up.to(torch.long) - move_down.to(torch.long)


@dataclass
class TerrainCurriculumState:
    score_sum: torch.Tensor
    sample_count: torch.Tensor
    safety_triggered: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TerrainCurriculumState":
        return cls(
            score_sum=torch.zeros(num_envs, device=device, dtype=dtype),
            sample_count=torch.zeros(num_envs, device=device, dtype=torch.long),
            safety_triggered=torch.zeros(num_envs, device=device, dtype=torch.bool),
        )

    def record(
        self,
        v_ref: torch.Tensor,
        actual_speed: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
        safety_mask: torch.Tensor | None = None,
    ) -> None:
        score = speed_tracking_score(v_ref, actual_speed)
        if active_mask is None:
            active_mask = torch.ones_like(self.safety_triggered)
        if active_mask.shape != self.safety_triggered.shape:
            raise ValueError("active_mask must have shape [N]")
        self.score_sum[active_mask] += score[active_mask]
        self.sample_count[active_mask] += 1
        if safety_mask is not None:
            self.safety_triggered |= safety_mask

    def mean_score(self, env_ids: torch.Tensor) -> torch.Tensor:
        count = self.sample_count[env_ids]
        return torch.where(
            count > 0,
            self.score_sum[env_ids] / torch.clamp(count, min=1).to(self.score_sum.dtype),
            torch.zeros_like(self.score_sum[env_ids]),
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.score_sum[env_ids] = 0.0
        self.sample_count[env_ids] = 0
        self.safety_triggered[env_ids] = False


def record_curriculum_tracking(
    env: Any, actual_speed: torch.Tensor | None = None
) -> None:
    """Record policy-step tracking and safety statistics."""

    if actual_speed is None:
        robot = env.scene["robot"]
        actual_speed = torch.sum(
            robot.data.root_lin_vel_w * env.path_tangent_w, dim=-1
        )
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    safety = torch.any(env.termination_state.counters > 0, dim=-1)
    env.curriculum_state.record(
        env.command_state.v_ref,
        actual_speed,
        active_mask=active,
        safety_mask=safety,
    )


def terrain_level_curriculum(env: Any, env_ids: torch.Tensor) -> torch.Tensor:
    """Reset-time ManagerTerm that updates levels before closed-chain reset."""

    state: TerrainCurriculumState = env.curriculum_state
    mean_score = state.mean_score(env_ids)
    manager = env.termination_manager
    if hasattr(manager, "time_outs"):
        timed_out = manager.time_outs[env_ids]
    elif hasattr(env, "time_out_buf"):
        timed_out = env.time_out_buf[env_ids]
    else:
        raise AttributeError("timeout state is required by the terrain curriculum")
    terminated = manager.terminated[env_ids]
    early = terminated & ~timed_out
    delta = terrain_level_delta(
        mean_score,
        timed_out,
        state.safety_triggered[env_ids],
        early,
    )
    move_up = delta > 0
    move_down = delta < 0
    levels = env.scene.terrain.terrain_levels[env_ids]
    downhill_at_limit = (env.slope[env_ids] < 0.0) & (
        levels >= MAX_TRAINING_DOWNHILL_LEVEL
    )
    move_up &= ~downhill_at_limit
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    # The signed slope basis must be current before cart/robot roots are written.
    update_slope_frame(env, env_ids)
    state.reset(env_ids)
    return torch.mean(env.scene.terrain.terrain_levels.float())


__all__ = [
    "CurriculumRuntimeState",
    "CurriculumScheduleCfg",
    "CurriculumStage",
    "TerrainCurriculumState",
    "curriculum_stage_for_iteration",
    "record_curriculum_tracking",
    "speed_tracking_score",
    "terrain_level_curriculum",
    "terrain_level_delta",
]
