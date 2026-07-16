"""Privileged teacher components used only during S0/S1 training."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Independent

from .actor_critic import (
    ACTION_DIM,
    CURRENT_OBSERVATION_DIM,
    DEFAULT_LATENT_DIM,
    GaussianActor,
    PrivilegedCritic,
)


class TeacherEncoder(nn.Module):
    """Map normalized independent extrinsics to the teacher context latent."""

    def __init__(self, extrinsics_dim: int, latent_dim: int = DEFAULT_LATENT_DIM) -> None:
        super().__init__()
        if not isinstance(extrinsics_dim, int) or extrinsics_dim < 1:
            raise ValueError(
                f"extrinsics_dim must be a positive integer, got {extrinsics_dim!r}"
            )
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError(
                f"latent_dim must be a positive integer, got {latent_dim!r}"
            )
        self.extrinsics_dim = extrinsics_dim
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(extrinsics_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.ndim != 2 or extrinsics.shape[1] != self.extrinsics_dim:
            raise ValueError(
                f"extrinsics must have shape [N, {self.extrinsics_dim}], "
                f"got {tuple(extrinsics.shape)}"
            )
        if not extrinsics.is_floating_point():
            raise TypeError(f"extrinsics must be floating point, got {extrinsics.dtype}")
        return self.network(extrinsics)


def normalize_extrinsics(
    extrinsics: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    clip: bool = True,
) -> torch.Tensor:
    """Map continuous training ranges to ``[-1, 1]``.

    Fixed parameters must be removed before calling this function; a zero-width
    range is rejected to make that requirement explicit.
    """

    if extrinsics.ndim < 1:
        raise ValueError("extrinsics must have at least one dimension")
    expected = (extrinsics.shape[-1],)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError(
            f"lower and upper must both have shape {expected}, got "
            f"{tuple(lower.shape)} and {tuple(upper.shape)}"
        )
    if not extrinsics.is_floating_point():
        raise TypeError(f"extrinsics must be floating point, got {extrinsics.dtype}")
    if lower.device != extrinsics.device or upper.device != extrinsics.device:
        raise ValueError("extrinsics and bounds must be on the same device")
    if lower.dtype != extrinsics.dtype or upper.dtype != extrinsics.dtype:
        raise ValueError("extrinsics and bounds must have the same dtype")
    if torch.any(upper <= lower):
        raise ValueError("every extrinsic upper bound must be greater than its lower bound")

    normalized = 2.0 * (extrinsics - lower) / (upper - lower) - 1.0
    return normalized.clamp(-1.0, 1.0) if clip else normalized


class TeacherModel(nn.Module):
    """Teacher encoder plus actor, with an optional independent S0 critic."""

    def __init__(
        self,
        extrinsics_dim: int,
        privileged_dim: int | None = None,
        current_dim: int = CURRENT_OBSERVATION_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.encoder = TeacherEncoder(extrinsics_dim, latent_dim)
        self.actor = GaussianActor(
            current_dim=current_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        self.critic = (
            PrivilegedCritic(
                privileged_dim=privileged_dim,
                current_dim=current_dim,
                latent_dim=latent_dim,
            )
            if privileged_dim is not None
            else None
        )

    def encode(self, extrinsics: torch.Tensor) -> torch.Tensor:
        return self.encoder(extrinsics)

    def forward(
        self, current: torch.Tensor, extrinsics: torch.Tensor
    ) -> Independent:
        distribution, _ = self.forward_with_context(current, extrinsics)
        return distribution

    def forward_with_context(
        self, current: torch.Tensor, extrinsics: torch.Tensor
    ) -> tuple[Independent, torch.Tensor]:
        context = self.encode(extrinsics)
        return self.actor(current, context), context

    def act(
        self,
        current: torch.Tensor,
        extrinsics: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.actor.act(
            current, self.encode(extrinsics), deterministic=deterministic
        )

    def value(
        self,
        current: torch.Tensor,
        extrinsics: torch.Tensor,
        privileged: torch.Tensor,
    ) -> torch.Tensor:
        if self.critic is None:
            raise RuntimeError("TeacherModel was created without privileged_dim/critic")
        return self.critic(current, self.encode(extrinsics), privileged)


class G1RickshawTeacherActor(TeacherModel):
    """Named teacher actor composed only of extrinsics encoder and actor."""

    def __init__(
        self,
        extrinsics_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ) -> None:
        super().__init__(
            extrinsics_dim=extrinsics_dim,
            privileged_dim=None,
            latent_dim=latent_dim,
        )


TeacherExtrinsicsEncoder = TeacherEncoder


__all__ = [
    "G1RickshawTeacherActor",
    "TeacherEncoder",
    "TeacherExtrinsicsEncoder",
    "TeacherModel",
    "normalize_extrinsics",
]
