"""Regression tests for terrain progression and domain-randomization phases."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from g1_rickshaw_lab.slope_contract import (
    SLOPE_COUNT,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import curricula
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.curricula import (
    TerrainCurriculumState,
    balanced_slope_assignment,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    DomainRandomizationStage,
    DomainRandomizationScheduleCfg,
)


def test_domain_randomization_scale_progresses_every_200_iterations() -> None:
    cfg = DomainRandomizationScheduleCfg()

    assert cfg.resolve(199) == (DomainRandomizationStage.NOMINAL, 0.0)
    assert cfg.resolve(200) == (DomainRandomizationStage.NARROW, 1.0 / 30.0)
    assert cfg.resolve(3400) == (DomainRandomizationStage.NARROW, 17.0 / 30.0)
    assert cfg.resolve(3600) == (DomainRandomizationStage.FULL, 0.6)
    assert cfg.resolve(8000) == (DomainRandomizationStage.FULL, 0.6)


def test_first_stage_balances_every_configured_slope() -> None:
    slots, levels, terrain_types = balanced_slope_assignment(2 * SLOPE_COUNT + 7, device="cpu")

    assert slots[:SLOPE_COUNT].tolist() == list(range(SLOPE_COUNT))
    assert levels[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_LEVELS)
    assert terrain_types[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_TYPES)
    counts = torch.bincount(slots, minlength=SLOPE_COUNT)
    assert int(torch.max(counts) - torch.min(counts)) <= 1


def test_terrain_curriculum_advances_eligible_environments(monkeypatch) -> None:
    state = TerrainCurriculumState.zeros(2)
    state.score_sum[:] = 0.9
    state.sample_count[:] = 1

    class Terrain:
        def __init__(self) -> None:
            self.terrain_levels = torch.zeros(2, dtype=torch.long)

        def update_env_origins(
            self,
            env_ids: torch.Tensor,
            move_up: torch.Tensor,
            move_down: torch.Tensor,
        ) -> None:
            self.terrain_levels[env_ids] += move_up.to(torch.long)
            self.terrain_levels[env_ids] -= move_down.to(torch.long)

    env = SimpleNamespace(
        curriculum_state=state,
        termination_manager=SimpleNamespace(
            time_outs=torch.ones(2, dtype=torch.bool),
            terminated=torch.zeros(2, dtype=torch.bool),
        ),
        scene=SimpleNamespace(terrain=Terrain()),
        slope=torch.zeros(2),
    )
    monkeypatch.setattr(curricula, "update_slope_frame", lambda _env, _ids: None)

    curricula.terrain_level_curriculum(env, torch.arange(2))

    assert env.scene.terrain.terrain_levels.tolist() == [1, 1]
