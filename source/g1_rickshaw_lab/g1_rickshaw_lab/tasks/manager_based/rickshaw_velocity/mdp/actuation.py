"""Actuator-limit utilities shared by safety gates and acceptance tooling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


ACTUATOR_TORQUE_RATIO_SOURCE = (
    "robot.data.applied_torque / current actuator.effort_limit"
)


def _joint_id_list(robot: Any, joint_ids: Any) -> list[int]:
    if isinstance(joint_ids, slice):
        values = list(range(int(robot.num_joints)))[joint_ids]
    elif torch.is_tensor(joint_ids):
        if joint_ids.ndim != 1:
            raise ValueError("joint_ids tensor must be one-dimensional")
        values = [int(value) for value in joint_ids.detach().cpu().tolist()]
    elif isinstance(joint_ids, Sequence) and not isinstance(joint_ids, (str, bytes)):
        if any(isinstance(value, bool) for value in joint_ids):
            raise TypeError("joint_ids must contain integer indices, not booleans")
        values = [int(value) for value in joint_ids]
    else:
        raise TypeError("joint_ids must be a slice, tensor, or integer sequence")
    if not values:
        raise ValueError("joint_ids must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("joint_ids must be unique")
    if any(value < 0 or value >= int(robot.num_joints) for value in values):
        raise IndexError("joint_ids contains an index outside the articulation")
    return values


def actuator_effort_limits(robot: Any, joint_ids: Any) -> torch.Tensor:
    """Return current physical actuator limits for requested articulation joints.

    ``ArticulationData.joint_effort_limits`` is the PhysX solver limit.  For an
    explicit actuator such as ``DCMotor`` it is deliberately much larger than
    the motor's clipped output, so it is not a hardware torque denominator.
    This helper resolves each joint through its owning actuator and fails on
    missing, duplicate, non-finite, or non-positive limits.
    """

    requested = _joint_id_list(robot, joint_ids)
    requested_positions = {joint_id: index for index, joint_id in enumerate(requested)}
    result: torch.Tensor | None = None
    owners = [0] * len(requested)

    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, dict) or not actuators:
        raise RuntimeError("articulation does not expose a non-empty actuator mapping")
    for name, actuator in actuators.items():
        actuator_ids = _joint_id_list(robot, actuator.joint_indices)
        effort_limit = getattr(actuator, "effort_limit", None)
        if not torch.is_tensor(effort_limit) or effort_limit.ndim != 2:
            raise RuntimeError(
                f"actuator {name!r} effort_limit must be a [num_envs, num_joints] tensor"
            )
        if effort_limit.shape[1] != len(actuator_ids):
            raise RuntimeError(
                f"actuator {name!r} effort_limit width does not match its joint indices"
            )
        if result is None:
            result = torch.empty(
                (effort_limit.shape[0], len(requested)),
                device=effort_limit.device,
                dtype=effort_limit.dtype,
            )
        elif (
            effort_limit.shape[0] != result.shape[0]
            or effort_limit.device != result.device
            or effort_limit.dtype != result.dtype
        ):
            raise RuntimeError("all actuator effort-limit tensors must share batch/device/dtype")

        for local_index, joint_id in enumerate(actuator_ids):
            requested_index = requested_positions.get(joint_id)
            if requested_index is None:
                continue
            owners[requested_index] += 1
            if owners[requested_index] > 1:
                raise RuntimeError(f"joint {joint_id} is owned by multiple actuators")
            result[:, requested_index] = effort_limit[:, local_index]

    missing = [requested[index] for index, count in enumerate(owners) if count == 0]
    if missing:
        raise RuntimeError(f"requested joints have no actuator effort limit: {missing}")
    assert result is not None
    if torch.any(~torch.isfinite(result)) or torch.any(result <= 0.0):
        raise RuntimeError("actuator effort limits must be finite and positive")
    return result


__all__ = ["ACTUATOR_TORQUE_RATIO_SOURCE", "actuator_effort_limits"]
