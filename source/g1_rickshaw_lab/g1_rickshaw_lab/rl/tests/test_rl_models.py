"""Pure-PyTorch tests for the teacher, student, actor, and critic."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from g1_rickshaw_lab.rl import (
    DYNAMIC_PRIVILEGE_DIM,
    STATIC_PRIVILEGE_DIM,
    ContextEncoder,
    G1RickshawStudentActor,
    G1RickshawTeacherActor,
    GaussianActor,
    PrivilegedCritic,
    StudentDistillationLoss,
    gaussian_kl,
    temporal_receptive_field,
)


class TestRickshawRLModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_context_schema_and_receptive_field(self) -> None:
        encoder = ContextEncoder()
        self.assertEqual(encoder.receptive_field, temporal_receptive_field())
        self.assertEqual(encoder.receptive_field, 61)
        self.assertEqual(encoder(torch.zeros(2, 61, 96)).shape, (2, 16))

        history = torch.full((1, 61, 96), 0.01, requires_grad=True)
        encoder(history).sum().backward()
        self.assertGreater(history.grad[:, 0].abs().sum().item(), 0.0)

        probe = torch.full((1, 62, 96), 0.01, requires_grad=True)
        feature = encoder.blocks(encoder.input(probe.transpose(1, 2)))[:, :, -1]
        encoder.context(feature).sum().backward()
        self.assertEqual(probe.grad[:, 0].count_nonzero().item(), 0)
        self.assertGreater(probe.grad[:, 1].abs().sum().item(), 0.0)

    def test_actor_and_critic_architectures(self) -> None:
        actor = GaussianActor()
        actor_shapes = [
            (layer.in_features, layer.out_features)
            for layer in actor.network
            if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(
            actor_shapes,
            [(112, 512), (512, 256), (256, 128), (128, 29)],
        )
        torch.testing.assert_close(actor.std[:12], torch.full((12,), 0.4))
        torch.testing.assert_close(actor.std[12:], torch.full((17,), 0.25))

        critic = PrivilegedCritic()
        critic_shapes = [
            (layer.in_features, layer.out_features)
            for layer in critic.network
            if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(critic_shapes, [(160, 256), (256, 128), (128, 1)])
        self.assertEqual(
            critic(torch.randn(3, 96), torch.randn(3, 64)).shape,
            (3, 1),
        )

    def test_teacher_and_student_share_the_actor_latent_abi(self) -> None:
        batch = 3
        current = torch.randn(batch, 96)
        history = torch.randn(batch, 61, 96)
        dynamic = torch.randn(batch, 61, DYNAMIC_PRIVILEGE_DIM)
        static = torch.randn(batch, STATIC_PRIVILEGE_DIM)

        teacher = G1RickshawTeacherActor()
        teacher_distribution, z_star = teacher.forward_with_context(
            current, history, dynamic, static
        )
        student = G1RickshawStudentActor()
        student_distribution, z_hat = student.forward_with_context(current, history)

        self.assertEqual(teacher_distribution.mean.shape, (batch, 29))
        self.assertEqual(student_distribution.mean.shape, (batch, 29))
        self.assertEqual(z_star.shape, (batch, 16))
        self.assertEqual(z_hat.shape, (batch, 16))
        self.assertEqual(
            set(teacher.actor.state_dict()), set(student.actor.state_dict())
        )

        for latent_dim in (8, 16, 24, 32):
            with self.subTest(latent_dim=latent_dim):
                teacher = G1RickshawTeacherActor(latent_dim)
                student = G1RickshawStudentActor(latent_dim)
                teacher_distribution, z_star = teacher.forward_with_context(
                    current, history, dynamic, static
                )
                student_distribution, z_hat = student.forward_with_context(
                    current, history
                )
                self.assertEqual(z_star.shape, (batch, latent_dim))
                self.assertEqual(z_hat.shape, (batch, latent_dim))
                self.assertEqual(
                    teacher.actor.network[0].in_features,
                    96 + latent_dim,
                )
                self.assertEqual(
                    student.actor.network[0].in_features,
                    96 + latent_dim,
                )
                self.assertEqual(
                    teacher_distribution.mean.shape,
                    student_distribution.mean.shape,
                )

    def test_minimal_distillation_is_student_only(self) -> None:
        batch = 4
        current = torch.randn(batch, 96)
        history = torch.randn(batch, 61, 96)
        teacher = G1RickshawTeacherActor()
        student = G1RickshawStudentActor()
        teacher_distribution, z_star = teacher.forward_with_context(
            current,
            history,
            torch.randn(batch, 61, DYNAMIC_PRIVILEGE_DIM),
            torch.randn(batch, STATIC_PRIVILEGE_DIM),
        )
        student_distribution, z_hat = student.forward_with_context(current, history)

        loss, metrics = StudentDistillationLoss()(
            teacher_distribution, student_distribution, z_hat, z_star
        )
        self.assertEqual(set(metrics), {"loss", "action_kl", "latent_smooth_l1"})
        loss.backward()
        self.assertIsNotNone(student.context_encoder.input.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertLess(
            gaussian_kl(teacher_distribution, teacher_distribution).abs().max().item(),
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
