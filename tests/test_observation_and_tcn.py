"""Acceptance tests for the fixed 96-D observation and sole causal history."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from g1_rickshaw_lab.rl.actor_critic import G1RickshawStudentActor
from g1_rickshaw_lab.rl.teacher_model import G1RickshawTeacherActor
from g1_rickshaw_lab.rl.context_encoder import (
    DILATIONS,
    HISTORY_LENGTH as TCN_HISTORY_LENGTH,
    KERNEL_SIZE,
    ContextEncoder,
    temporal_receptive_field,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import rewards
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.observations import (
    ACTOR_OBSERVATION_DIM,
    BASE_ANGULAR_VELOCITY_SLICE,
    HISTORY_LENGTH,
    JOINT_POSITION_SLICE,
    JOINT_VELOCITY_SLICE,
    PREVIOUS_ACTION_SLICE,
    PROJECTED_GRAVITY_SLICE,
    TASK_SIGNAL_SLICE,
    ObservationHistoryState,
    assemble_actor_observation,
)


def test_actor_observation_is_exactly_96d_in_fixed_scaled_order() -> None:
    dtype = torch.float64
    angular_velocity = torch.tensor([[4.0, -2.0, 1.0]], dtype=dtype)
    gravity = torch.tensor([[0.1, 0.2, -0.9]], dtype=dtype)
    v_ref = torch.tensor([0.7], dtype=dtype)
    lateral_error = torch.tensor([-0.2], dtype=dtype)
    heading_error = torch.tensor([3.0 * torch.pi], dtype=dtype)
    q_ref = torch.linspace(-0.2, 0.2, 29, dtype=dtype).unsqueeze(0)
    position_error = torch.linspace(-0.1, 0.1, 29, dtype=dtype).unsqueeze(0)
    joint_position = q_ref + position_error
    joint_velocity = torch.linspace(-2.0, 2.0, 29, dtype=dtype).unsqueeze(0)
    previous_processed_action = torch.linspace(-0.5, 0.5, 29, dtype=dtype).unsqueeze(0)

    observation = assemble_actor_observation(
        angular_velocity,
        gravity,
        v_ref,
        lateral_error,
        heading_error,
        joint_position,
        q_ref,
        joint_velocity,
        previous_processed_action,
    )

    assert observation.shape == (1, ACTOR_OBSERVATION_DIM)
    assert ACTOR_OBSERVATION_DIM == 96
    torch.testing.assert_close(
        observation[:, BASE_ANGULAR_VELOCITY_SLICE], angular_velocity * 0.25
    )
    torch.testing.assert_close(observation[:, PROJECTED_GRAVITY_SLICE], gravity)
    torch.testing.assert_close(
        observation[:, TASK_SIGNAL_SLICE],
        torch.tensor([[1.4, -0.4, torch.pi]], dtype=dtype),
    )
    torch.testing.assert_close(observation[:, JOINT_POSITION_SLICE], position_error)
    torch.testing.assert_close(observation[:, JOINT_VELOCITY_SLICE], joint_velocity * 0.05)
    torch.testing.assert_close(
        observation[:, PREVIOUS_ACTION_SLICE], previous_processed_action
    )


def test_observation_and_reward_interfaces_do_not_read_v_sample() -> None:
    assert "v_sample" not in inspect.signature(assemble_actor_observation).parameters

    command = SimpleNamespace(
        v_ref=torch.tensor([0.7], dtype=torch.float64),
        v_sample=torch.tensor([100.0], dtype=torch.float64),
    )
    env = SimpleNamespace(
        command_state=command,
        policy_robot_speed_s=torch.tensor([0.5], dtype=torch.float64),
    )
    before = rewards.track_speed_exp(env)
    command.v_sample[:] = -100.0
    after_sample_change = rewards.track_speed_exp(env)
    torch.testing.assert_close(before, after_sample_change)
    command.v_ref[:] = 0.5
    after_reference_change = rewards.track_speed_exp(env)
    assert after_reference_change.item() > before.item()


def test_history_is_61x96_and_explicitly_excludes_current() -> None:
    state = ObservationHistoryState.zeros(1, dtype=torch.float64)
    first = torch.full((1, 96), 10.0, dtype=torch.float64)
    second = torch.full((1, 96), 20.0, dtype=torch.float64)
    third = torch.full((1, 96), 30.0, dtype=torch.float64)

    state.advance(first)
    assert state.history.shape == (1, HISTORY_LENGTH, ACTOR_OBSERVATION_DIM)
    assert (HISTORY_LENGTH, ACTOR_OBSERVATION_DIM) == (61, 96)
    torch.testing.assert_close(state.history, first[:, None, :].expand(-1, 61, -1))
    torch.testing.assert_close(state.current, first)

    state.advance(second)
    torch.testing.assert_close(state.history[:, -1], first)
    torch.testing.assert_close(state.current, second)
    assert not torch.any(state.history == 20.0)

    state.advance(third)
    torch.testing.assert_close(state.history[:, -2], first)
    torch.testing.assert_close(state.history[:, -1], second)
    torch.testing.assert_close(state.current, third)
    assert not torch.any(state.history == 30.0)

    frozen_history = state.history.clone()
    frozen_current = state.current.clone()
    state.advance(torch.full_like(third, 40.0), valid_mask=torch.tensor([False]))
    torch.testing.assert_close(state.history, frozen_history)
    torch.testing.assert_close(state.current, frozen_current)


def test_history_state_can_track_current_without_allocating_temporal_storage() -> None:
    state = ObservationHistoryState.zeros(2, history_enabled=False)
    observation = torch.randn(2, ACTOR_OBSERVATION_DIM)

    state.advance(observation)

    assert state.history is None
    torch.testing.assert_close(state.current, observation)
    assert torch.all(state.initialized)


def test_tcn_schema_receptive_field_and_single_history_path() -> None:
    assert TCN_HISTORY_LENGTH == HISTORY_LENGTH == 61
    assert KERNEL_SIZE == 5
    assert DILATIONS == (1, 2, 4, 8)
    assert temporal_receptive_field() == 61

    encoder = ContextEncoder().eval()
    blocks = list(encoder.blocks)
    assert tuple(block.conv.dilation[0] for block in blocks) == DILATIONS
    for block in blocks:
        dilated_convolutions = [
            module
            for module in block.modules()
            if isinstance(module, nn.Conv1d) and module.kernel_size == (KERNEL_SIZE,)
        ]
        assert len(dilated_convolutions) == 1
        assert dilated_convolutions[0].stride == (1,)

    assert encoder(torch.zeros(2, 61, 96)).shape == (2, 16)
    with pytest.raises(ValueError, match=r"\[N, 61, 96\]"):
        encoder(torch.zeros(2, 60, 96))
    with pytest.raises(ValueError, match=r"\[N, 61, 96\]"):
        encoder(torch.zeros(2, 61, 95))

    student = G1RickshawStudentActor()
    history_encoders = [
        module for module in student.modules() if isinstance(module, ContextEncoder)
    ]
    recurrent_modules = [
        module for module in student.modules() if isinstance(module, (nn.RNNBase, nn.RNNCellBase))
    ]
    assert len(history_encoders) == 1
    assert recurrent_modules == []


def test_fixed_teacher_and_student_context_interfaces() -> None:
    teacher = G1RickshawTeacherActor().eval()
    student = G1RickshawStudentActor().eval()
    current = torch.zeros(2, 96)
    observation_history = torch.zeros(2, 61, 96)
    dynamic_history = torch.zeros(2, 61, 21)
    static_privilege = torch.zeros(2, 40)
    with torch.no_grad():
        teacher_distribution, teacher_context = teacher.forward_with_context(
            current,
            observation_history,
            dynamic_history,
            static_privilege,
        )
        student_distribution, student_context = student.forward_with_context(
            current, observation_history
        )
    assert teacher.encoder.latent_dim == 16
    assert student.context_encoder.latent_dim == 16
    assert teacher_context.shape == (2, 16)
    assert student_context.shape == (2, 16)
    assert teacher_distribution.mean.shape == (2, 29)
    assert student_distribution.mean.shape == (2, 29)
    assert teacher.actor.network[0].in_features == 96 + 16
    assert student.actor.network[0].in_features == 96 + 16
    assert not hasattr(teacher, "context_projection")
    assert not hasattr(student, "context_projection")
    assert not any("aux" in name for name, _ in teacher.named_modules())
    assert not any("aux" in name for name, _ in student.named_modules())


@pytest.mark.parametrize("latent_dim", (8, 16, 24, 32))
def test_teacher_and_student_use_the_selected_latent_width(latent_dim: int) -> None:
    teacher = G1RickshawTeacherActor(latent_dim).eval()
    student = G1RickshawStudentActor(latent_dim).eval()
    current = torch.zeros(2, 96)
    history = torch.zeros(2, 61, 96)
    with torch.no_grad():
        teacher_distribution, teacher_context = teacher.forward_with_context(
            current,
            history,
            torch.zeros(2, 61, 21),
            torch.zeros(2, 40),
        )
        student_distribution, student_context = student.forward_with_context(
            current, history
        )

    assert teacher_context.shape == student_context.shape == (2, latent_dim)
    assert teacher.actor.network[0].in_features == 96 + latent_dim
    assert student.actor.network[0].in_features == 96 + latent_dim
    assert set(teacher.actor.state_dict()) == set(student.actor.state_dict())
    assert teacher_distribution.mean.shape == student_distribution.mean.shape == (2, 29)


def test_tcn_oldest_frame_is_used_but_outside_and_future_frames_are_causal() -> None:
    encoder = ContextEncoder().eval()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.fill_(0.01)

    # At output index 61 the exact 61-frame receptive field is input 1..61.
    probe = torch.full((1, 62, 96), 0.01)
    outside_perturbed = probe.clone()
    outside_perturbed[:, 0] += 10.0
    with torch.no_grad():
        base_feature = encoder.blocks(encoder.input(probe.transpose(1, 2)))[:, :, -1]
        perturbed_feature = encoder.blocks(
            encoder.input(outside_perturbed.transpose(1, 2))
        )[:, :, -1]
        base_context = encoder.context(base_feature)
        perturbed_context = encoder.context(perturbed_feature)
    torch.testing.assert_close(base_context, perturbed_context, rtol=0.0, atol=0.0)

    oldest_perturbed = probe.clone()
    oldest_perturbed[:, 1] += 0.1
    with torch.no_grad():
        oldest_feature = encoder.blocks(
            encoder.input(oldest_perturbed.transpose(1, 2))
        )[:, :, -1]
        oldest_context = encoder.context(oldest_feature)
    assert not torch.allclose(base_context, oldest_context)

    prefix = torch.randn(1, 61, 96)
    future = torch.randn(1, 7, 96)
    with torch.no_grad():
        prefix_outputs = encoder.blocks(encoder.input(prefix.transpose(1, 2)))
        extended_outputs = encoder.blocks(
            encoder.input(torch.cat((prefix, future), dim=1).transpose(1, 2))
        )
    torch.testing.assert_close(
        prefix_outputs, extended_outputs[:, :, :61], rtol=0.0, atol=0.0
    )
