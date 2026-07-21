"""Causal observation-history encoder used by the deployable student."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from g1_rickshaw_lab.policy_schema import (
    ACTOR_OBSERVATION_DIM,
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    validate_history_length,
    validate_context_dim,
)

OBSERVATION_DIM: Final[int] = ACTOR_OBSERVATION_DIM
FEATURE_DIM: Final[int] = 64
KERNEL_SIZE: Final[int] = 5
DILATIONS: Final[tuple[int, ...]] = (1, 2, 4, 8)
HISTORY_KERNEL_SIZES: Final[dict[int, int]] = {61: 5, 91: 7}


def temporal_receptive_field(
    kernel_size: int = KERNEL_SIZE, dilations: tuple[int, ...] = DILATIONS
) -> int:
    if kernel_size < 1 or not dilations or any(value < 1 for value in dilations):
        raise ValueError("kernel size and dilations must be positive")
    return 1 + (kernel_size - 1) * sum(dilations)


class CausalBlock(nn.Module):
    """One residual causal convolution and one channel-mixing projection."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv(F.pad(value, (self.left_pad, 0)))
        return F.elu(value + self.mix(F.elu(residual)))


def validate_history(
    history: torch.Tensor,
    *,
    feature_dim: int,
    name: str,
    history_length: int = HISTORY_LENGTH,
) -> None:
    history_length = validate_history_length(history_length)
    if history.ndim != 3 or history.shape[1:] != (history_length, feature_dim):
        raise ValueError(
            f"{name} must have shape [N, {history_length}, {feature_dim}], "
            f"got {tuple(history.shape)}"
        )
    if not history.is_floating_point():
        raise TypeError(f"{name} must be floating point")


class ContextEncoder(nn.Module):
    """Encode the preceding 61 actor observations into the selected latent."""

    observation_dim: Final[int] = OBSERVATION_DIM
    feature_dim: Final[int] = FEATURE_DIM
    receptive_field: Final[int] = temporal_receptive_field()

    def __init__(
        self,
        latent_dim: int = DEFAULT_CONTEXT_DIM,
        history_length: int = HISTORY_LENGTH,
    ) -> None:
        super().__init__()
        self.latent_dim = validate_context_dim(latent_dim)
        self.history_length = validate_history_length(history_length)
        self.kernel_size = HISTORY_KERNEL_SIZES[self.history_length]
        self.receptive_field = temporal_receptive_field(
            self.kernel_size,
            DILATIONS,
        )
        self.input = nn.Conv1d(OBSERVATION_DIM, FEATURE_DIM, kernel_size=1)
        self.blocks = nn.Sequential(
            *(
                CausalBlock(FEATURE_DIM, dilation, self.kernel_size)
                for dilation in DILATIONS
            )
        )
        self.context = nn.Linear(FEATURE_DIM, self.latent_dim)

    def extract_feature(self, history: torch.Tensor) -> torch.Tensor:
        validate_history(
            history,
            feature_dim=OBSERVATION_DIM,
            name="history",
            history_length=self.history_length,
        )
        encoded = self.blocks(self.input(history.transpose(1, 2)))
        return encoded[:, :, -1]

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.context(self.extract_feature(history))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.encode(history)


__all__ = [
    "DEFAULT_CONTEXT_DIM",
    "DILATIONS",
    "FEATURE_DIM",
    "HISTORY_KERNEL_SIZES",
    "HISTORY_LENGTH",
    "KERNEL_SIZE",
    "OBSERVATION_DIM",
    "CausalBlock",
    "ContextEncoder",
    "temporal_receptive_field",
    "validate_history",
]
