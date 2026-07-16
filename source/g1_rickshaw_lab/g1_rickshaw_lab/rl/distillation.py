"""S1 on-policy teacher-to-student distillation losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Independent, Normal


@dataclass(frozen=True)
class DistillationWeights:
    """Loss weights fixed by implementation guide section 10.2."""

    action_kl: float = 1.0
    latent: float = 0.05
    phase: float = 0.10
    frequency: float = 0.05
    contact: float = 0.05
    cart_lag: float = 0.05

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0.0:
                raise ValueError(f"{name} weight must be non-negative, got {value}")


def _normal_parameters(
    distribution: Distribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(distribution, Independent):
        raise TypeError(
            "expected torch.distributions.Independent(Normal(...), 1), "
            f"got {type(distribution).__name__}"
        )
    if distribution.reinterpreted_batch_ndims != 1 or not isinstance(
        distribution.base_dist, Normal
    ):
        raise TypeError("distribution must be a one-event diagonal Normal")
    return distribution.base_dist.loc, distribution.base_dist.scale


def gaussian_kl(
    teacher_distribution: Distribution,
    student_distribution: Distribution,
    *,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """Return ``KL(teacher || student)`` summed over actions, per sample."""

    teacher_mean, teacher_std = _normal_parameters(teacher_distribution)
    student_mean, student_std = _normal_parameters(student_distribution)
    if teacher_mean.shape != student_mean.shape:
        raise ValueError(
            "teacher and student action shapes must match, got "
            f"{tuple(teacher_mean.shape)} and {tuple(student_mean.shape)}"
        )
    if teacher_mean.ndim != 2:
        raise ValueError(
            f"action distributions must have batch shape [N, A], got {teacher_mean.shape}"
        )
    if teacher_mean.device != student_mean.device:
        raise ValueError("teacher and student distributions must share a device")
    if detach_teacher:
        teacher_mean = teacher_mean.detach()
        teacher_std = teacher_std.detach()

    teacher_variance = teacher_std.square()
    student_variance = student_std.square()
    elementwise = (
        torch.log(student_std / teacher_std)
        + (teacher_variance + (teacher_mean - student_mean).square())
        / (2.0 * student_variance)
        - 0.5
    )
    return elementwise.sum(dim=-1)


def _expanded_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 0:
        raise ValueError("mask must include the batch dimension")
    if mask.shape[0] != reference.shape[0]:
        raise ValueError(
            f"mask batch {mask.shape[0]} does not match tensor batch {reference.shape[0]}"
        )
    while mask.ndim > reference.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim > reference.ndim:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot cover {tuple(reference.shape)}"
        )
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(reference)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape cannot broadcast to {tuple(reference.shape)}"
        ) from error
    weights = mask.to(device=reference.device, dtype=reference.dtype)
    if torch.any(weights < 0):
        raise ValueError("mask weights must be non-negative")
    return weights


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = _expanded_mask(mask, values)
    # clamp_min yields a differentiable zero when a mini-batch has no valid label.
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over valid elements, or differentiable zero if empty."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    if prediction.ndim < 1:
        raise ValueError("prediction and target must include a batch dimension")
    return _masked_mean((prediction - target).square(), mask)


def masked_phase_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Circular phase loss for ``[sin(phase), cos(phase)]`` labels."""

    if prediction.ndim != 2 or prediction.shape[-1] != 2:
        raise ValueError(
            f"phase prediction must have shape [N, 2], got {tuple(prediction.shape)}"
        )
    if target.shape != prediction.shape:
        raise ValueError(
            f"phase target must have shape {tuple(prediction.shape)}, got {tuple(target.shape)}"
        )
    prediction_unit = F.normalize(prediction, dim=-1, eps=eps)
    target_unit = F.normalize(target, dim=-1, eps=eps)
    circular_error = 1.0 - (prediction_unit * target_unit).sum(dim=-1)
    return _masked_mean(circular_error, mask)


class StudentDistillationLoss(nn.Module):
    """Guide-defined S1 objective with action KL as its primary term."""

    REQUIRED_AUXILIARY_KEYS = frozenset(("phase", "frequency", "contact", "cart_lag"))

    def __init__(self, weights: DistillationWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or DistillationWeights()

    def forward(
        self,
        teacher_distribution: Distribution,
        student_distribution: Distribution,
        z_hat: torch.Tensor,
        z_star: torch.Tensor,
        auxiliary: Mapping[str, torch.Tensor],
        phase_target: torch.Tensor,
        frequency_target: torch.Tensor,
        contact_target: torch.Tensor,
        cart_lag_target: torch.Tensor,
        gait_mask: torch.Tensor,
        lag_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = self.REQUIRED_AUXILIARY_KEYS.difference(auxiliary)
        if missing:
            raise KeyError(f"missing auxiliary predictions: {sorted(missing)}")
        if z_hat.shape != z_star.shape or z_hat.ndim != 2:
            raise ValueError(
                f"z_hat and z_star must share shape [N, Z], got {tuple(z_hat.shape)} "
                f"and {tuple(z_star.shape)}"
            )

        teacher_mean, _ = _normal_parameters(teacher_distribution)
        student_mean, _ = _normal_parameters(student_distribution)
        batch_size = teacher_mean.shape[0]
        named_batches = {
            "student_distribution": student_mean.shape[0],
            "z_hat": z_hat.shape[0],
            "phase": auxiliary["phase"].shape[0],
            "frequency": auxiliary["frequency"].shape[0],
            "contact": auxiliary["contact"].shape[0],
            "cart_lag": auxiliary["cart_lag"].shape[0],
        }
        mismatched = {
            name: size for name, size in named_batches.items() if size != batch_size
        }
        if mismatched:
            raise ValueError(
                f"all distillation branches must use batch {batch_size}, got {mismatched}"
            )

        action_kl = gaussian_kl(teacher_distribution, student_distribution).mean()
        latent = F.mse_loss(z_hat, z_star.detach())
        phase = masked_phase_loss(auxiliary["phase"], phase_target, gait_mask)
        frequency = masked_mse(
            auxiliary["frequency"], frequency_target, gait_mask
        )
        if auxiliary["contact"].shape != contact_target.shape:
            raise ValueError(
                "contact prediction and target shapes must match, got "
                f"{tuple(auxiliary['contact'].shape)} and {tuple(contact_target.shape)}"
            )
        contact = F.binary_cross_entropy_with_logits(
            auxiliary["contact"],
            contact_target.to(
                device=auxiliary["contact"].device,
                dtype=auxiliary["contact"].dtype,
            ),
        )
        cart_lag = masked_mse(auxiliary["cart_lag"], cart_lag_target, lag_mask)

        weighted = {
            "action_kl": self.weights.action_kl * action_kl,
            "latent_mse": self.weights.latent * latent,
            "phase": self.weights.phase * phase,
            "frequency": self.weights.frequency * frequency,
            "contact": self.weights.contact * contact,
            "cart_lag": self.weights.cart_lag * cart_lag,
        }
        total = sum(weighted.values())
        metrics = {
            "loss": total,
            "action_kl": action_kl,
            "latent_mse": latent,
            "phase_loss": phase,
            "frequency_loss": frequency,
            "contact_loss": contact,
            "cart_lag_loss": cart_lag,
            **{f"weighted_{name}": value for name, value in weighted.items()},
        }
        return total, metrics


def freeze_teacher(module: nn.Module) -> nn.Module:
    """Put a teacher in evaluation mode and disable all of its gradients."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def initialize_student_actor(
    student_actor: nn.Module, teacher_actor: nn.Module
) -> None:
    """Strictly initialize the student actor, including Gaussian log-std."""

    student_actor.load_state_dict(teacher_actor.state_dict(), strict=True)


def distillation_loss(
    teacher_distribution: Distribution,
    student_distribution: Distribution,
    z_hat: torch.Tensor,
    z_star: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    phase_target: torch.Tensor,
    frequency_target: torch.Tensor,
    contact_target: torch.Tensor,
    cart_lag_target: torch.Tensor,
    gait_mask: torch.Tensor,
    lag_mask: torch.Tensor,
    weights: DistillationWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Functional entry point equivalent to :class:`StudentDistillationLoss`."""

    return StudentDistillationLoss(weights)(
        teacher_distribution,
        student_distribution,
        z_hat,
        z_star,
        auxiliary,
        phase_target,
        frequency_target,
        contact_target,
        cart_lag_target,
        gait_mask,
        lag_mask,
    )


__all__ = [
    "DistillationWeights",
    "StudentDistillationLoss",
    "distillation_loss",
    "freeze_teacher",
    "gaussian_kl",
    "initialize_student_actor",
    "masked_mse",
    "masked_phase_loss",
]
