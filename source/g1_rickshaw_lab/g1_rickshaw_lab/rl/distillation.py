"""Minimal teacher-to-student distillation objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Independent, Normal


def _normal_parameters(
    distribution: Distribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(distribution, Independent) or not isinstance(
        distribution.base_dist, Normal
    ):
        raise TypeError("expected Independent(Normal(...), 1)")
    return distribution.base_dist.loc, distribution.base_dist.scale


def gaussian_kl(
    teacher_distribution: Distribution,
    student_distribution: Distribution,
) -> torch.Tensor:
    """Return KL(teacher || student), summed over the 29 actions."""

    teacher_mean, teacher_std = _normal_parameters(teacher_distribution)
    student_mean, student_std = _normal_parameters(student_distribution)
    teacher_mean = teacher_mean.detach()
    teacher_std = teacher_std.detach()
    elementwise = (
        torch.log(student_std / teacher_std)
        + (
            teacher_std.square()
            + (teacher_mean - student_mean).square()
        )
        / (2.0 * student_std.square())
        - 0.5
    )
    return elementwise.sum(dim=-1)


class StudentDistillationLoss(nn.Module):
    def forward(
        self,
        teacher_distribution: Distribution,
        student_distribution: Distribution,
        z_hat: torch.Tensor,
        z_star: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_hat.shape != z_star.shape:
            raise ValueError("teacher and student latent shapes differ")
        action_kl = gaussian_kl(
            teacher_distribution, student_distribution
        ).mean()
        latent = F.smooth_l1_loss(z_hat, z_star.detach())
        total = action_kl + 0.1 * latent
        return total, {
            "loss": total,
            "action_kl": action_kl,
            "latent_smooth_l1": latent,
        }


__all__ = [
    "StudentDistillationLoss",
    "gaussian_kl",
]
