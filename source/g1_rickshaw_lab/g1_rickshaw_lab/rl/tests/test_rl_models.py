"""Pure-PyTorch acceptance tests for the rickshaw policy models."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from g1_rickshaw_lab.rl import (
    ContextEncoder,
    G1RickshawCritic,
    G1RickshawStudentActor,
    G1RickshawTeacherActor,
    GaussianActor,
    StudentDistillationLoss,
    gaussian_kl,
    masked_mse,
    temporal_receptive_field,
)


class TestRickshawRLModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_context_input_and_receptive_field_are_fixed(self) -> None:
        encoder = ContextEncoder()
        self.assertEqual(encoder.history_length, 61)
        self.assertEqual(encoder.observation_dim, 96)
        self.assertEqual(encoder.receptive_field, temporal_receptive_field())
        self.assertEqual(encoder.receptive_field, 61)
        self.assertEqual(encoder(torch.zeros(2, 61, 96))[0].shape, (2, 16))

        for invalid in (torch.zeros(2, 60, 96), torch.zeros(2, 61, 95)):
            with self.assertRaisesRegex(ValueError, r"\[N, 61, 96\]"):
                encoder(invalid)

    def test_tcn_is_causal_and_uses_exactly_61_frames(self) -> None:
        encoder = ContextEncoder().eval()
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.fill_(0.01)

        history = torch.full((1, 61, 96), 0.01, requires_grad=True)
        encoder.encode(history).sum().backward()
        self.assertIsNotNone(history.grad)
        self.assertGreater(history.grad[:, 0].abs().sum().item(), 0.0)

        # A final output over 62 probe frames sees indices 1..61, never index 0.
        probe = torch.full((1, 62, 96), 0.01, requires_grad=True)
        feature = encoder.blocks(encoder.input(probe.transpose(1, 2)))[:, :, -1]
        encoder.context(feature).sum().backward()
        self.assertIsNotNone(probe.grad)
        self.assertEqual(probe.grad[:, 0].count_nonzero().item(), 0)
        self.assertGreater(probe.grad[:, 1].abs().sum().item(), 0.0)

        # Appending future frames cannot alter any already-computed prefix output.
        prefix = torch.randn(1, 61, 96)
        future = torch.randn(1, 5, 96)
        with torch.no_grad():
            prefix_output = encoder.blocks(encoder.input(prefix.transpose(1, 2)))
            extended = torch.cat((prefix, future), dim=1)
            extended_output = encoder.blocks(encoder.input(extended.transpose(1, 2)))
        torch.testing.assert_close(prefix_output, extended_output[:, :, :61])

    def test_actor_architecture_and_groupwise_initial_std(self) -> None:
        actor = GaussianActor()
        linear_shapes = [
            (layer.in_features, layer.out_features)
            for layer in actor.network
            if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(
            linear_shapes,
            [(112, 512), (512, 256), (256, 128), (128, 29)],
        )
        torch.testing.assert_close(actor.std[:12], torch.full((12,), 0.4))
        torch.testing.assert_close(actor.std[12:], torch.full((17,), 0.25))

        distribution = actor(torch.randn(3, 96), torch.randn(3, 16))
        self.assertEqual(distribution.mean.shape, (3, 29))
        self.assertEqual(distribution.stddev.shape, (3, 29))

        student = G1RickshawStudentActor()
        student(torch.randn(2, 96), torch.randn(2, 61, 96)).mean.sum().backward()
        self.assertIsNone(student.context_encoder.phase.weight.grad)
        self.assertIsNone(student.context_encoder.frequency.weight.grad)
        self.assertIsNone(student.context_encoder.contact.weight.grad)
        self.assertIsNone(student.context_encoder.cart_lag.weight.grad)

    def test_named_teacher_student_and_critic_interfaces(self) -> None:
        batch_size, extrinsics_dim, privileged_dim = 3, 11, 23
        current = torch.randn(batch_size, 96)
        history = torch.randn(batch_size, 61, 96)
        extrinsics = torch.randn(batch_size, extrinsics_dim)

        teacher = G1RickshawTeacherActor(extrinsics_dim)
        teacher_distribution, z_star = teacher.forward_with_context(
            current, extrinsics
        )
        student = G1RickshawStudentActor()
        student_distribution, z_hat, auxiliary = student.forward_with_context(
            current, history
        )
        critic = G1RickshawCritic(privileged_dim)

        self.assertEqual(teacher_distribution.mean.shape, (batch_size, 29))
        self.assertEqual(student_distribution.mean.shape, (batch_size, 29))
        self.assertEqual(z_star.shape, (batch_size, 16))
        self.assertEqual(z_hat.shape, (batch_size, 16))
        self.assertEqual(auxiliary["phase"].shape, (batch_size, 2))
        value = critic(current, z_hat, torch.randn(batch_size, privileged_dim))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_distillation_backward_and_empty_masks(self) -> None:
        batch_size, extrinsics_dim = 4, 9
        current = torch.randn(batch_size, 96)
        history = torch.randn(batch_size, 61, 96)
        teacher = G1RickshawTeacherActor(extrinsics_dim)
        student = G1RickshawStudentActor()
        teacher_distribution, z_star = teacher.forward_with_context(
            current, torch.randn(batch_size, extrinsics_dim)
        )
        student_distribution, z_hat, auxiliary = student.forward_with_context(
            current, history
        )

        criterion = StudentDistillationLoss()
        loss, metrics = criterion(
            teacher_distribution,
            student_distribution,
            z_hat,
            z_star,
            auxiliary,
            torch.nn.functional.normalize(torch.randn(batch_size, 2), dim=-1),
            torch.rand(batch_size, 1),
            torch.randint(0, 2, (batch_size, 2), dtype=torch.bool),
            torch.randn(batch_size, 1),
            torch.tensor([[1], [0], [1], [1]], dtype=torch.bool),
            torch.zeros(batch_size, 1, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["cart_lag_loss"].item(), 0.0)
        loss.backward()
        self.assertIsNotNone(student.context_encoder.input.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

        identical_kl = gaussian_kl(teacher_distribution, teacher_distribution)
        self.assertLess(identical_kl.abs().max().item(), 1.0e-6)
        empty = masked_mse(
            torch.ones(batch_size, 1, requires_grad=True),
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size, dtype=torch.bool),
        )
        self.assertEqual(empty.item(), 0.0)
        empty.backward()


if __name__ == "__main__":
    unittest.main()
