from __future__ import annotations

import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


def test_speed_kernels_use_the_configured_scales() -> None:
    v_ref = torch.tensor([1.0])
    v_robot = torch.tensor([0.5])
    v_lateral = torch.tensor([0.0])

    torch.testing.assert_close(
        rewards.track_speed_exp_value(v_ref, v_robot, v_lateral),
        torch.exp(torch.tensor([-1.0])),
    )
    torch.testing.assert_close(
        rewards.track_speed_exp_value(v_ref, v_robot, torch.tensor([0.5])),
        torch.exp(torch.tensor([-2.0])),
    )


def test_gait_reward_matches_alternating_contact_phase() -> None:
    episode_time = torch.tensor([0.2, 0.6, 0.2])
    is_contact = torch.tensor([[True, False], [False, True], [True, False]])
    v_ref = torch.tensor([0.2, 0.2, 0.04])

    value = rewards.feet_gait_value(episode_time, is_contact, v_ref)

    torch.testing.assert_close(value, torch.tensor([2.0, 2.0, 0.0]))


def test_foot_clearance_uses_height_error_and_swing_speed() -> None:
    foot_height = torch.tensor([[0.07, 0.00], [0.07, 0.00]])
    foot_speed = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    v_ref = torch.tensor([0.2, 0.04])

    value = rewards.foot_clearance_reward_value(foot_height, foot_speed, v_ref)
    expected = torch.exp(
        -torch.tensor([0.07**2])
        * torch.tanh(torch.tensor([2.0]))
        / rewards.FOOT_CLEARANCE_STD_M
    )

    torch.testing.assert_close(value, torch.tensor([expected.item(), 0.0]))
