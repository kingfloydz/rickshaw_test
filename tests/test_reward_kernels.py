from __future__ import annotations

import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards


def test_speed_kernels_use_the_configured_scales() -> None:
    v_ref = torch.tensor([1.0])
    v_robot = torch.tensor([0.5])

    torch.testing.assert_close(
        rewards.track_speed_exp_value(v_ref, v_robot), torch.exp(torch.tensor([-1.0]))
    )
    torch.testing.assert_close(
        rewards.track_speed_precise_exp_value(v_ref, v_robot),
        torch.exp(torch.tensor([-4.0])),
    )
    torch.testing.assert_close(
        rewards.speed_error_pseudo_huber_value(v_ref, v_robot),
        torch.sqrt(torch.tensor([2.0])) - 1.0,
    )


def test_landing_reward_requires_actual_motion_and_tracking() -> None:
    first_contact = torch.tensor([[True, False], [True, False], [True, True]])
    last_air_time = torch.full((3, 2), rewards.FEET_LANDING_TARGET_AIR_TIME_S)
    v_ref = torch.tensor([0.5, 0.5, 0.5])
    v_robot = torch.tensor([0.4, 0.05, 0.4])

    value = rewards.feet_landing_value(first_contact, last_air_time, v_ref, v_robot)

    torch.testing.assert_close(value, torch.tensor([1.0, 0.0, 0.0]))


def test_overlong_swing_is_penalized_on_landing_and_while_airborne() -> None:
    first_contact = torch.tensor([[True, False]])
    last_air_time = torch.tensor([[0.70, 0.0]])
    landing = rewards.feet_landing_value(
        first_contact,
        last_air_time,
        torch.tensor([0.5]),
        torch.tensor([0.4]),
    )
    airborne = rewards.feet_air_time_excess_l2_value(torch.tensor([[0.70, 0.10]]))

    assert landing.item() < 0.0
    torch.testing.assert_close(airborne, torch.tensor([1.0]))
