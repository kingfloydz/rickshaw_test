"""Pure policy ABI constants shared by training, simulation, and deployment."""

from __future__ import annotations

from typing import Final


ACTOR_OBSERVATION_DIM: Final[int] = 96
HISTORY_LENGTH: Final[int] = 61
DEFAULT_CONTEXT_DIM: Final[int] = 16
SUPPORTED_CONTEXT_DIMS: Final[tuple[int, ...]] = (8, 16, 24, 32)
TEACHER_DYNAMIC_DIM: Final[int] = 21
TEACHER_STATIC_DIM: Final[int] = 40
CRITIC_PRIVILEGED_DIM: Final[int] = 64

ACTION_SCALE: Final[tuple[float, ...]] = (
    (0.40,) * 12
    + (0.20,) * 3
    + (0.25, 0.25, 0.25, 0.30, 0.15, 0.15, 0.15) * 2
)
ACTION_DIM: Final[int] = len(ACTION_SCALE)

BUTTERWORTH_B0: Final[float] = 0.20430082
BUTTERWORTH_B1: Final[float] = 0.20430082
BUTTERWORTH_A1: Final[float] = -0.59139835


def validate_context_dim(value: int) -> int:
    if value not in SUPPORTED_CONTEXT_DIMS:
        raise ValueError(
            f"context dimension must be one of {SUPPORTED_CONTEXT_DIMS}, got {value!r}"
        )
    return value


__all__ = [
    "ACTION_DIM",
    "ACTION_SCALE",
    "ACTOR_OBSERVATION_DIM",
    "BUTTERWORTH_A1",
    "BUTTERWORTH_B0",
    "BUTTERWORTH_B1",
    "DEFAULT_CONTEXT_DIM",
    "CRITIC_PRIVILEGED_DIM",
    "HISTORY_LENGTH",
    "SUPPORTED_CONTEXT_DIMS",
    "TEACHER_DYNAMIC_DIM",
    "TEACHER_STATIC_DIM",
    "validate_context_dim",
]
