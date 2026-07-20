"""Validated immutable artifacts consumed by the simulator configuration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .configuration import (
    FeasibilityEnvelope,
    ResetPoseLibrary,
    load_feasibility_envelope,
    load_reset_pose_library,
)


@dataclass(frozen=True, slots=True)
class TaskArtifacts:
    feasibility: FeasibilityEnvelope
    reset_poses: ResetPoseLibrary


@lru_cache(maxsize=8)
def load_task_artifacts(feasibility_path: str, reset_pose_path: str) -> TaskArtifacts:
    """Load and validate one artifact pair once per process."""

    feasibility = Path(feasibility_path).resolve()
    reset_poses = Path(reset_pose_path).resolve()
    return TaskArtifacts(
        feasibility=load_feasibility_envelope(feasibility),
        reset_poses=load_reset_pose_library(reset_poses),
    )


__all__ = ["TaskArtifacts", "load_task_artifacts"]
