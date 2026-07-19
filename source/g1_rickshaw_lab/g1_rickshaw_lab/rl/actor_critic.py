"""Gaussian actor and raw-privilege critic for G1 rickshaw control."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.distributions import Independent, Normal

from g1_rickshaw_lab.policy_schema import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_DIM,
    DEFAULT_CONTEXT_DIM,
    validate_context_dim,
)

from .context_encoder import ContextEncoder


CURRENT_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM
LOWER_BODY_ACTION_DIM = 12
CRITIC_PRIVILEGE_DIM = CRITIC_PRIVILEGED_DIM
ACTOR_HIDDEN_DIMS = (512, 256, 128)
CRITIC_HIDDEN_DIMS = (256, 128)
MIN_ACTION_STD = 0.05
LOWER_BODY_MAX_ACTION_STD = 0.8
UPPER_BODY_MAX_ACTION_STD = 0.5


def _build_mlp(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int
) -> nn.Sequential:
    dimensions = (input_dim, *hidden_dims, output_dim)
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(source, target))
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def _matrix(tensor: torch.Tensor, width: int, name: str) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")


class GaussianActor(nn.Module):
    """Map the current observation and selected context to 29 actions."""

    current_dim = CURRENT_OBSERVATION_DIM
    action_dim = ACTION_DIM

    def __init__(self, latent_dim: int = DEFAULT_CONTEXT_DIM) -> None:
        super().__init__()
        self.latent_dim = validate_context_dim(latent_dim)
        self.network = _build_mlp(
            CURRENT_OBSERVATION_DIM + self.latent_dim,
            ACTOR_HIDDEN_DIMS,
            ACTION_DIM,
        )
        initial_std = torch.full((ACTION_DIM,), 0.25)
        initial_std[:LOWER_BODY_ACTION_DIM] = 0.4
        self.log_std = nn.Parameter(initial_std.log())

    def _bounded_log_std(self) -> torch.Tensor:
        maximum = torch.full_like(self.log_std, UPPER_BODY_MAX_ACTION_STD)
        maximum[:LOWER_BODY_ACTION_DIM] = LOWER_BODY_MAX_ACTION_STD
        minimum = self.log_std.new_full(self.log_std.shape, math.log(MIN_ACTION_STD))
        return torch.minimum(torch.maximum(self.log_std, minimum), torch.log(maximum))

    def distribution(
        self, current: torch.Tensor, context: torch.Tensor
    ) -> Independent:
        _matrix(current, CURRENT_OBSERVATION_DIM, "current")
        _matrix(context, self.latent_dim, "context")
        if current.shape[0] != context.shape[0]:
            raise ValueError("current and context batch dimensions differ")
        mean = self.network(torch.cat((current, context), dim=-1))
        std = self._bounded_log_std().exp().to(dtype=mean.dtype).expand_as(mean)
        return Independent(Normal(mean, std, validate_args=False), 1)

    def forward(self, current: torch.Tensor, context: torch.Tensor) -> Independent:
        return self.distribution(current, context)

    def act(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        distribution = self.distribution(current, context)
        return distribution.mean if deterministic else distribution.sample()

    @property
    def std(self) -> torch.Tensor:
        return self._bounded_log_std().exp()


class PrivilegedCritic(nn.Module):
    """Independent value network using current observation and raw privilege."""

    current_dim = CURRENT_OBSERVATION_DIM
    privileged_dim = CRITIC_PRIVILEGE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.network = _build_mlp(
            CURRENT_OBSERVATION_DIM + CRITIC_PRIVILEGE_DIM,
            CRITIC_HIDDEN_DIMS,
            1,
        )

    def forward(
        self, current: torch.Tensor, privileged: torch.Tensor
    ) -> torch.Tensor:
        _matrix(current, CURRENT_OBSERVATION_DIM, "current")
        _matrix(privileged, self.privileged_dim, "privileged")
        if current.shape[0] != privileged.shape[0]:
            raise ValueError("current and privileged batch dimensions differ")
        return self.network(torch.cat((current, privileged), dim=-1))


class G1RickshawStudentActor(nn.Module):
    """Deployable student composed only of observation TCN and actor."""

    def __init__(self, latent_dim: int = DEFAULT_CONTEXT_DIM) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(latent_dim)
        self.actor = GaussianActor(latent_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(history)

    def forward_with_context(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> tuple[Independent, torch.Tensor]:
        context = self.encode(history)
        return self.actor(current, context), context

    def forward(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> Independent:
        return self.actor(current, self.encode(history))

    def act(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.actor.act(
            current, self.encode(history), deterministic=deterministic
        )


__all__ = [
    "ACTION_DIM",
    "ACTOR_HIDDEN_DIMS",
    "CRITIC_PRIVILEGE_DIM",
    "CRITIC_HIDDEN_DIMS",
    "CURRENT_OBSERVATION_DIM",
    "LOWER_BODY_ACTION_DIM",
    "MIN_ACTION_STD",
    "LOWER_BODY_MAX_ACTION_STD",
    "UPPER_BODY_MAX_ACTION_STD",
    "G1RickshawStudentActor",
    "GaussianActor",
    "PrivilegedCritic",
]
