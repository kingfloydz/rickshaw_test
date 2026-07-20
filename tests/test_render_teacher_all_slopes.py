"""CPU-only checks for the all-slope teacher renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _isaaclab_wrappers import FOLLOW_CAMERA_EYE, FOLLOW_CAMERA_LOOKAT  # noqa: E402
from render_teacher_all_slopes import (  # noqa: E402
    DEFAULT_FRAMES_PER_SLOPE,
    slope_index_for_frame,
)


def test_default_keeps_exactly_one_thousand_frames_per_slope() -> None:
    assert DEFAULT_FRAMES_PER_SLOPE == 1000
    assert slope_index_for_frame(0, 1000) == 0
    assert slope_index_for_frame(999, 1000) == 0
    assert slope_index_for_frame(1000, 1000) == 1
    assert slope_index_for_frame(18_999, 1000) == 18


@pytest.mark.parametrize("frame, frames_per_slope", [(-1, 1000), (0, 0)])
def test_slope_frame_mapping_rejects_invalid_values(frame: int, frames_per_slope: int) -> None:
    with pytest.raises(ValueError):
        slope_index_for_frame(frame, frames_per_slope)


def test_follow_camera_matches_the_reset_renderer_side_view() -> None:
    assert FOLLOW_CAMERA_EYE == (0.0, 4.2, 1.4)
    assert FOLLOW_CAMERA_LOOKAT == (0.0, 0.0, 0.85)
