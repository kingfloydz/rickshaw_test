from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _rollout_audit import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_NUM_ENVS,
    ROLLOUT_MANIFEST_SCHEMA_VERSION,
    SIGNED_SLOPES,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    slope_environment_assignment,
)
from train_context import _normalize_shard, seed_s1_training  # noqa: E402


def _canonical_rollout_tensors(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "current": torch.zeros(batch_size, 96),
        "history": torch.zeros(batch_size, 61, 96),
        "teacher_action_mean": torch.zeros(batch_size, ACTION_DIM),
        "teacher_action_std": torch.ones(batch_size, ACTION_DIM),
        "z_star": torch.zeros(batch_size, 16),
        "curriculum_stage": torch.ones(batch_size, 1, dtype=torch.long),
        "collection_segment": torch.zeros(batch_size, 1, dtype=torch.long),
        "environment_id": torch.arange(batch_size).unsqueeze(-1),
        "episode_id": torch.arange(batch_size).unsqueeze(-1),
        "slope": torch.zeros(batch_size, 1),
        "terrain_level": torch.zeros(batch_size, 1, dtype=torch.long),
        "terrain_type": torch.zeros(batch_size, 1, dtype=torch.long),
    }


def _write_rollout_shard(path: Path, tensors: dict[str, torch.Tensor], *, root: str = "rollout") -> None:
    torch.save(
        {
            "schema_version": ROLLOUT_MANIFEST_SCHEMA_VERSION,
            root: tensors,
        },
        path,
    )


def test_formal_rollout_assignment_covers_all_19_slopes() -> None:
    assignment = slope_environment_assignment(DEFAULT_NUM_ENVS)
    counts = torch.bincount(
        assignment["slope_index"], minlength=len(SIGNED_SLOPES)
    )
    assert counts.tolist() == [216] * 11 + [215] * 8
    torch.testing.assert_close(
        torch.unique(assignment["slope"], sorted=True),
        torch.tensor(SIGNED_SLOPES),
    )
    assert len(SLOPE_TERRAIN_LEVELS) == len(SIGNED_SLOPES)
    assert len(SLOPE_TERRAIN_TYPES) == len(SIGNED_SLOPES)


def test_formal_rollout_shard_accepts_only_canonical_tensor_shapes(tmp_path: Path) -> None:
    shard = tmp_path / "rollout.pt"
    _write_rollout_shard(shard, _canonical_rollout_tensors())

    normalized = _normalize_shard(shard)

    assert normalized["current"].shape == (2, 96)
    assert normalized["teacher_action_std"].shape == (2, ACTION_DIM)


@pytest.mark.parametrize("latent_dim", (8, 16, 24, 32))
def test_rollout_shard_uses_the_teacher_checkpoint_latent_width(
    tmp_path: Path, latent_dim: int
) -> None:
    shard = tmp_path / "rollout.pt"
    tensors = _canonical_rollout_tensors()
    tensors["z_star"] = torch.zeros(2, latent_dim)
    _write_rollout_shard(shard, tensors)

    assert _normalize_shard(shard, latent_dim)["z_star"].shape == (2, latent_dim)


def test_formal_rollout_shard_rejects_legacy_root(tmp_path: Path) -> None:
    shard = tmp_path / "rollout.pt"
    _write_rollout_shard(shard, _canonical_rollout_tensors(), root="data")

    with pytest.raises(ValueError, match="canonical rollout mapping"):
        _normalize_shard(shard)


def test_formal_rollout_shard_rejects_aliases_and_legacy_shapes(tmp_path: Path) -> None:
    alias_shard = tmp_path / "alias.pt"
    alias_tensors = _canonical_rollout_tensors()
    alias_tensors["policy"] = alias_tensors.pop("current")
    _write_rollout_shard(alias_shard, alias_tensors)

    with pytest.raises(KeyError, match="'current'"):
        _normalize_shard(alias_shard)

    shape_shard = tmp_path / "shape.pt"
    shape_tensors = _canonical_rollout_tensors()
    shape_tensors["z_star"] = torch.zeros(2)
    _write_rollout_shard(shape_shard, shape_tensors)

    with pytest.raises(ValueError, match="z_star must have shape"):
        _normalize_shard(shape_shard)


def test_s1_seed_explicitly_selects_fast_algorithms() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)

        seed_s1_training(42)

        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
