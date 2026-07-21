from __future__ import annotations

import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


def test_speed_kernels_use_the_configured_scales() -> None:
    v_ref = torch.tensor([1.0])
    v_robot = torch.tensor([0.5])
    lateral_speed = torch.tensor([0.0])

    torch.testing.assert_close(
        rewards.track_speed_exp_value(v_ref, v_robot, lateral_speed),
        torch.exp(torch.tensor([-1.0])),
    )
    torch.testing.assert_close(
        rewards.track_speed_exp_value(v_ref, v_robot, torch.tensor([0.5])),
        torch.exp(torch.tensor([-2.0])),
    )


def test_gait_reward_requires_motion_and_matching_contact_phase() -> None:
    episode_time = torch.tensor([0.30, 0.30, 0.30])
    contact = torch.tensor([[True, False], [False, True], [True, False]])
    v_ref = torch.tensor([0.5, 0.5, 0.0])

    value = rewards.feet_gait_value(episode_time, contact, v_ref)

    torch.testing.assert_close(value, torch.tensor([2.0, 0.0, 0.0]))


def test_swing_height_penalty_uses_only_airborne_feet() -> None:
    penalty = rewards.feet_swing_height_value(
        torch.tensor([[0.0, 0.02]]),
        torch.tensor([[True, False]]),
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0025]))
