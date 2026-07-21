"""Small task-specific mjlab MDP terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def fat2_prior(
    env: ManagerBasedRlEnv,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Weak torso-pitch prior initialized by the MuJoCo static solve."""

    robot: Entity = env.scene[asset_cfg.name]
    quat = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    w, x, y, z = quat.unbind(-1)
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    target = env.fat2_reference_angle
    return torch.exp(-torch.square(pitch - target) / (sigma * sigma))


def cart_forward_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    cart: Entity = env.scene["rickshaw"]
    return cart.data.root_link_lin_vel_b[:, 0:1]


__all__ = ["cart_forward_velocity", "fat2_prior"]
