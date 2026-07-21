"""Pure policy ABI constants shared by training, simulation, and deployment."""

from __future__ import annotations

from typing import Final

from .g1_motor_defaults import G1_ACTION_SCALE

ACTOR_OBSERVATION_DIM: Final[int] = 98
HISTORY_LENGTH: Final[int] = 61
SUPPORTED_HISTORY_LENGTHS: Final[tuple[int, ...]] = (61, 91)
DEFAULT_CONTEXT_DIM: Final[int] = 16
SUPPORTED_CONTEXT_DIMS: Final[tuple[int, ...]] = (
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    24,
    32,
)
TEACHER_DYNAMIC_DIM: Final[int] = 21
TEACHER_STATIC_DIM: Final[int] = 10
CRITIC_PRIVILEGED_DIM: Final[int] = 34

ACTION_SCALE: Final[tuple[float, ...]] = G1_ACTION_SCALE
ACTION_DIM: Final[int] = len(ACTION_SCALE)

BUTTERWORTH_B0: Final[float] = 0.20430082
BUTTERWORTH_B1: Final[float] = 0.20430082
BUTTERWORTH_A1: Final[float] = -0.59139835


def validate_context_dim(value: int) -> int:
    if type(value) is not int or value not in SUPPORTED_CONTEXT_DIMS:
        raise ValueError(f"context dimension must be one of {SUPPORTED_CONTEXT_DIMS}, got {value!r}")
    return value


def validate_history_length(value: int) -> int:
    if type(value) is not int or value not in SUPPORTED_HISTORY_LENGTHS:
        raise ValueError(f"history length must be one of {SUPPORTED_HISTORY_LENGTHS}, got {value!r}")
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
    "SUPPORTED_HISTORY_LENGTHS",
    "SUPPORTED_CONTEXT_DIMS",
    "TEACHER_DYNAMIC_DIM",
    "TEACHER_STATIC_DIM",
    "validate_context_dim",
    "validate_history_length",
]
