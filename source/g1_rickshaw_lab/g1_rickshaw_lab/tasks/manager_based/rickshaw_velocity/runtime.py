"""Typed runtime state owned by :class:`G1RickshawRLEnv`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RickshawRuntime:
    """All policy-rate mutable state with one explicit lifecycle owner."""

    command: Any
    path: Any
    cart: Any
    stability: Any
    action: Any
    analytic_force: Any
    cart_interaction_wrench: Any
    observation_history: Any
    teacher_dynamic_history: Any
    termination: Any
    termination_causes: Any
    rolling_resistance_cfg: Any


__all__ = ["RickshawRuntime"]
