from __future__ import annotations

from types import SimpleNamespace

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
    torch.testing.assert_close(
        rewards.track_speed_exp_value(
            v_ref, v_robot, torch.tensor([0.5]), lateral_penalty_scale=0.1
        ),
        torch.exp(torch.tensor([-1.1])),
    )


def test_lateral_speed_penalty_recovers_with_command_curriculum() -> None:
    command_cfg = SimpleNamespace(maximum=0.1, limit_maximum=1.0)
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            policy_update=SimpleNamespace(command_sampling=command_cfg)
        ),
        command_state=SimpleNamespace(v_ref=torch.tensor([1.0])),
        policy_robot_speed_s=torch.tensor([0.5]),
        policy_robot_speed_l=torch.tensor([0.5]),
    )

    initial = rewards.track_speed_exp(env)
    command_cfg.maximum = 1.0
    restored = rewards.track_speed_exp(env)

    torch.testing.assert_close(initial, torch.exp(torch.tensor([-1.1])))
    torch.testing.assert_close(restored, torch.exp(torch.tensor([-2.0])))


def test_requested_reward_weights_are_bound() -> None:
    assert rewards.REWARD_WEIGHTS["feet_gait"] == 0.25
    assert rewards.REWARD_WEIGHTS["feet_swing_height"] == -20.0
    assert rewards.REWARD_WEIGHTS["processed_action_rate_l2"] == -0.03


def test_gait_reward_matches_alternating_contact_phase() -> None:
    episode_time = torch.tensor([0.2, 0.8, 0.2])
    is_contact = torch.tensor([[True, False], [False, True], [True, False]])
    v_ref = torch.tensor([0.2, 0.2, 0.04])

    value = rewards.feet_gait_value(episode_time, is_contact, v_ref)

    assert rewards.GAIT_PERIOD_S == 1.2
    torch.testing.assert_close(value, torch.tensor([2.0, 2.0, 0.0]))


def test_feet_swing_height_penalizes_only_non_contact_feet() -> None:
    foot_height = torch.tensor([[0.07, 0.02], [0.10, 0.04]])
    is_contact = torch.tensor([[False, True], [False, False]])

    value = rewards.feet_swing_height_value(foot_height, is_contact)

    torch.testing.assert_close(
        value,
        torch.tensor([0.0, (0.10 - 0.07) ** 2 + (0.04 - 0.07) ** 2]),
    )


def test_feet_slide_matches_unitree_contact_velocity_penalty() -> None:
    velocity = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float64)
    is_contact = torch.tensor([[True, False]])

    torch.testing.assert_close(
        rewards.feet_slide_value(velocity, is_contact),
        torch.tensor([14.0], dtype=torch.float64),
    )


def test_processed_action_rate_matches_unitree_action_difference() -> None:
    action = torch.tensor([[1.0, -1.0, 0.5]])
    previous = torch.tensor([[0.0, -0.5, 0.5]])

    torch.testing.assert_close(
        rewards.processed_action_rate_l2_value(action, previous),
        torch.tensor([1.25]),
    )
