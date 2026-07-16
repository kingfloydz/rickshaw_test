from __future__ import annotations

from pathlib import Path
import sys

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _rollout_audit import (  # noqa: E402
    FORMAL_NUM_ENVS,
    SIGNED_SLOPES,
    SLOPE_TERRAIN_LEVELS,
    SLOPE_TERRAIN_TYPES,
    formal_slope_environment_assignment,
)


def test_formal_rollout_assignment_covers_all_19_slopes() -> None:
    assignment = formal_slope_environment_assignment(FORMAL_NUM_ENVS)
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
