"""Regression tests for the two-stage physical curriculum."""

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
    CurriculumRuntimeState,
    CurriculumScheduleCfg,
    CurriculumStage,
    TerrainCurriculumState,
    balanced_slope_assignment,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    _set_cart_gravity_disabled,
)


def test_stage_transition_is_adopted_only_by_reset_environments() -> None:
    cfg = CurriculumScheduleCfg(static_hand_load_iterations=3)
    state = CurriculumRuntimeState.create(
        torch.arange(4), torch.zeros(4, dtype=torch.long), cfg
    )

    assert state.stage == CurriculumStage.STATIC_HAND_LOAD
    assert torch.all(state.stage_per_environment() == int(CurriculumStage.STATIC_HAND_LOAD))

    assert state.set_iteration(3) == CurriculumStage.TRAINING
    assert torch.all(state.stage_per_environment() == int(CurriculumStage.STATIC_HAND_LOAD))
    changed = state.activate(torch.tensor([1, 3]))

    assert changed.tolist() == [1, 3]
    assert state.stage_per_environment().tolist() == [0, 1, 0, 1]


def test_first_stage_balances_every_configured_slope() -> None:
    slots, levels, terrain_types = balanced_slope_assignment(2 * SLOPE_COUNT + 7, device="cpu")

    assert slots[:SLOPE_COUNT].tolist() == list(range(SLOPE_COUNT))
    assert levels[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_LEVELS)
    assert terrain_types[:SLOPE_COUNT].tolist() == list(SLOPE_TERRAIN_TYPES)
    counts = torch.bincount(slots, minlength=SLOPE_COUNT)
    assert int(torch.max(counts) - torch.min(counts)) <= 1


def test_zero_length_static_stage_starts_with_real_cart() -> None:
    cfg = CurriculumScheduleCfg(static_hand_load_iterations=0)
    state = CurriculumRuntimeState.create(
        torch.arange(3), torch.zeros(3, dtype=torch.long), cfg
    )

    assert state.stage == CurriculumStage.TRAINING
    assert torch.all(state.stage_per_environment() == int(CurriculumStage.TRAINING))


def test_static_stage_keeps_slope_while_training_stage_can_advance(monkeypatch) -> None:
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
        curriculum_stage_per_env=torch.tensor(
            [int(CurriculumStage.STATIC_HAND_LOAD), int(CurriculumStage.TRAINING)]
        ),
        termination_manager=SimpleNamespace(
            time_outs=torch.ones(2, dtype=torch.bool),
            terminated=torch.zeros(2, dtype=torch.bool),
        ),
        scene=SimpleNamespace(terrain=Terrain()),
        slope=torch.zeros(2),
    )
    monkeypatch.setattr(curricula, "update_slope_frame", lambda _env, _ids: None)

    curricula.terrain_level_curriculum(env, torch.arange(2))

    assert env.scene.terrain.terrain_levels.tolist() == [0, 1]


def test_parked_cart_gravity_is_changed_only_for_selected_environments() -> None:
    class View:
        def __init__(self) -> None:
            self.flags = torch.zeros((3, 4), dtype=torch.uint8)

        def get_disable_gravities(self) -> torch.Tensor:
            return self.flags

        def set_disable_gravities(
            self, flags: torch.Tensor, indices: torch.Tensor
        ) -> None:
            self.flags = flags.clone()
            self.indices = indices.clone()

    view = View()
    env = SimpleNamespace(
        scene={"rickshaw": SimpleNamespace(root_physx_view=view)}
    )

    _set_cart_gravity_disabled(env, torch.tensor([0, 2]), True)
    assert view.flags.tolist() == [
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ]
    assert view.indices.tolist() == [0, 2]

    _set_cart_gravity_disabled(env, torch.tensor([2]), False)
    assert torch.all(view.flags[2] == 0)
