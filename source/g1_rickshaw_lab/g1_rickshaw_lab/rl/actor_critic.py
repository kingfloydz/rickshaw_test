"""Asymmetric Gaussian actor and privileged critic for G1 rickshaw control."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Independent, Normal

from .context_encoder import ContextEncoder


CURRENT_OBSERVATION_DIM = 96
DEFAULT_LATENT_DIM = 16
ACTION_DIM = 29
LOWER_BODY_ACTION_DIM = 12
HIDDEN_DIMS = (512, 256, 128)

DiagonalGaussian = Independent


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def build_context_projection(latent_dim: int) -> nn.Module:
    """Preserve the exact teacher actor ABI with a genuine latent bottleneck."""

    latent_dim = _positive_int("latent_dim", latent_dim)
    if latent_dim == DEFAULT_LATENT_DIM:
        return nn.Identity()
    return nn.Sequential(nn.ELU(), nn.Linear(latent_dim, DEFAULT_LATENT_DIM))


def _build_mlp(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int
) -> nn.Sequential:
    dimensions = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(in_features, out_features))
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def _check_matrix(name: str, tensor: torch.Tensor, width: int) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [N, {width}], got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")


def _check_same_batch(**tensors: torch.Tensor) -> None:
    batch_sizes = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(batch_sizes.values())) != 1:
        formatted = ", ".join(f"{name}={size}" for name, size in batch_sizes.items())
        raise ValueError(f"batch dimensions must match, got {formatted}")


class GaussianActor(nn.Module):
    """Policy mapping only ``current + context`` to a 29-D Gaussian action."""

    def __init__(
        self,
        current_dim: int = CURRENT_OBSERVATION_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: Sequence[int] = HIDDEN_DIMS,
        lower_body_action_dim: int = LOWER_BODY_ACTION_DIM,
        lower_body_initial_std: float = 0.4,
        upper_body_initial_std: float = 0.25,
    ) -> None:
        super().__init__()
        self.current_dim = _positive_int("current_dim", current_dim)
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.action_dim = _positive_int("action_dim", action_dim)
        if self.current_dim != CURRENT_OBSERVATION_DIM:
            raise ValueError(
                f"actor current_dim is fixed to {CURRENT_OBSERVATION_DIM}, "
                f"got {self.current_dim}"
            )
        if self.action_dim != ACTION_DIM:
            raise ValueError(
                f"actor action_dim is fixed to {ACTION_DIM}, got {self.action_dim}"
            )
        if tuple(hidden_dims) != HIDDEN_DIMS:
            raise ValueError(
                f"actor hidden_dims are fixed to {HIDDEN_DIMS}, got {tuple(hidden_dims)}"
            )
        if not 0 <= lower_body_action_dim <= action_dim:
            raise ValueError(
                "lower_body_action_dim must be between zero and action_dim, "
                f"got {lower_body_action_dim} and {action_dim}"
            )
        if lower_body_action_dim != LOWER_BODY_ACTION_DIM:
            raise ValueError(
                f"lower_body_action_dim is fixed to {LOWER_BODY_ACTION_DIM}, "
                f"got {lower_body_action_dim}"
            )
        if lower_body_initial_std <= 0.0 or upper_body_initial_std <= 0.0:
            raise ValueError("initial Gaussian standard deviations must be positive")

        self.lower_body_action_dim = lower_body_action_dim
        self.network = _build_mlp(
            self.current_dim + self.latent_dim, HIDDEN_DIMS, self.action_dim
        )
        initial_std = torch.full((self.action_dim,), upper_body_initial_std)
        initial_std[:lower_body_action_dim] = lower_body_initial_std
        self.log_std = nn.Parameter(initial_std.log())

    def _inputs(self, current: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        _check_matrix("current", current, self.current_dim)
        _check_matrix("context", context, self.latent_dim)
        _check_same_batch(current=current, context=context)
        if current.device != context.device:
            raise ValueError(
                f"current and context must share a device, got {current.device} and "
                f"{context.device}"
            )
        if current.dtype != context.dtype:
            raise ValueError(
                f"current and context must share a dtype, got {current.dtype} and "
                f"{context.dtype}"
            )
        return torch.cat((current, context), dim=-1)

    def mean(self, current: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.network(self._inputs(current, context))

    def distribution(
        self, current: torch.Tensor, context: torch.Tensor
    ) -> Independent:
        mean = self.mean(current, context)
        std = self.log_std.exp().to(dtype=mean.dtype).expand_as(mean)
        return Independent(Normal(mean, std, validate_args=False), 1)

    def forward(self, current: torch.Tensor, context: torch.Tensor) -> Independent:
        return self.distribution(current, context)

    def act(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        distribution = self.distribution(current, context)
        return distribution.mean if deterministic else distribution.sample()

    def evaluate_actions(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(current, context)
        _check_matrix("actions", actions, self.action_dim)
        _check_same_batch(current=current, context=context, actions=actions)
        return distribution.log_prob(actions), distribution.entropy()

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()


class PrivilegedCritic(nn.Module):
    """Value network with a trunk that is never shared with the actor."""

    def __init__(
        self,
        privileged_dim: int,
        current_dim: int = CURRENT_OBSERVATION_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.current_dim = _positive_int("current_dim", current_dim)
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.privileged_dim = _positive_int("privileged_dim", privileged_dim)
        if self.current_dim != CURRENT_OBSERVATION_DIM:
            raise ValueError(
                f"critic current_dim is fixed to {CURRENT_OBSERVATION_DIM}, "
                f"got {self.current_dim}"
            )
        if tuple(hidden_dims) != HIDDEN_DIMS:
            raise ValueError(
                f"critic hidden_dims are fixed to {HIDDEN_DIMS}, got {tuple(hidden_dims)}"
            )
        self.network = _build_mlp(
            self.current_dim + self.latent_dim + self.privileged_dim,
            HIDDEN_DIMS,
            1,
        )

    def forward(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        privileged: torch.Tensor,
    ) -> torch.Tensor:
        _check_matrix("current", current, self.current_dim)
        _check_matrix("context", context, self.latent_dim)
        _check_matrix("privileged", privileged, self.privileged_dim)
        _check_same_batch(current=current, context=context, privileged=privileged)
        devices = {current.device, context.device, privileged.device}
        dtypes = {current.dtype, context.dtype, privileged.dtype}
        if len(devices) != 1:
            raise ValueError(f"critic inputs must share a device, got {devices}")
        if len(dtypes) != 1:
            raise ValueError(f"critic inputs must share a dtype, got {dtypes}")
        return self.network(torch.cat((current, context, privileged), dim=-1))


class ActorCritic(nn.Module):
    """Small composition used by PPO without actor/critic parameter sharing."""

    def __init__(
        self,
        privileged_dim: int,
        current_dim: int = CURRENT_OBSERVATION_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.actor = GaussianActor(
            current_dim=current_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        self.critic = PrivilegedCritic(
            privileged_dim=privileged_dim,
            current_dim=current_dim,
            latent_dim=latent_dim,
        )

        actor_parameter_ids = {id(parameter) for parameter in self.actor.parameters()}
        critic_parameter_ids = {id(parameter) for parameter in self.critic.parameters()}
        if actor_parameter_ids & critic_parameter_ids:
            raise RuntimeError("actor and critic must not share parameters")

    def forward(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        privileged: torch.Tensor,
    ) -> tuple[Independent, torch.Tensor]:
        return self.actor(current, context), self.critic(current, context, privileged)

    def act(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.actor.act(current, context, deterministic=deterministic)

    def evaluate(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        privileged: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(current, context, privileged)


class G1RickshawStudentActor(nn.Module):
    """Deployment policy composed of the sole history encoder and actor."""

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(latent_dim=latent_dim)
        self.context_projection = build_context_projection(latent_dim)
        # The actor input remains 16-D so every ablation preserves exact S0
        # actor initialization; only the history bottleneck is varied.
        self.actor = GaussianActor(latent_dim=DEFAULT_LATENT_DIM)

    def encode(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.context_encoder(history)

    def forward_with_context(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> tuple[Independent, torch.Tensor, dict[str, torch.Tensor]]:
        context, auxiliary = self.encode(history)
        actor_context = self.context_projection(context)
        return self.actor(current, actor_context), actor_context, auxiliary

    def forward(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> Independent:
        context = self.context_projection(self.context_encoder.encode(history))
        return self.actor(current, context)

    def act(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        context = self.context_projection(self.context_encoder.encode(history))
        return self.actor.act(current, context, deterministic=deterministic)


class G1RickshawCritic(PrivilegedCritic):
    """Named compatibility entry point for the independent privileged critic."""


# Concise aliases for callers that compose the teacher/student encoders themselves.
Actor = GaussianActor
Critic = PrivilegedCritic


__all__ = [
    "ACTION_DIM",
    "CURRENT_OBSERVATION_DIM",
    "DEFAULT_LATENT_DIM",
    "HIDDEN_DIMS",
    "LOWER_BODY_ACTION_DIM",
    "ActorCritic",
    "Actor",
    "Critic",
    "build_context_projection",
    "DiagonalGaussian",
    "G1RickshawCritic",
    "G1RickshawStudentActor",
    "GaussianActor",
    "PrivilegedCritic",
]
