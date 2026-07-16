"""Causal history encoder for the G1 rickshaw policy.

The encoder intentionally has one temporal input only.  Its final feature sees
exactly the 61 observations preceding the current policy observation.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import nn


HISTORY_LENGTH: Final[int] = 61
OBSERVATION_DIM: Final[int] = 96
FEATURE_DIM: Final[int] = 128
CONTEXT_DIM: Final[int] = 16
KERNEL_SIZE: Final[int] = 5
DILATIONS: Final[tuple[int, ...]] = (1, 2, 4, 8)


def temporal_receptive_field(
    kernel_size: int = KERNEL_SIZE, dilations: tuple[int, ...] = DILATIONS
) -> int:
    """Return the receptive field of one convolution per causal block."""

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if not dilations or any(dilation < 1 for dilation in dilations):
        raise ValueError(f"dilations must contain positive integers, got {dilations}")
    return 1 + (kernel_size - 1) * sum(dilations)


class CausalBlock(nn.Module):
    """One residual causal dilated convolution followed by a 1x1 mix."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if dilation < 1:
            raise ValueError(f"dilation must be positive, got {dilation}")

        self.channels = channels
        self.dilation = dilation
        self.left_pad = (KERNEL_SIZE - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=KERNEL_SIZE,
            dilation=dilation,
            stride=1,
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if x.ndim != 3:
                raise ValueError(
                    f"CausalBlock expects [N, C, T], got shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.channels:
                raise ValueError(
                    f"CausalBlock expects {self.channels} channels, got {x.shape[1]}"
                )
            if not x.is_floating_point():
                raise TypeError(f"CausalBlock expects floating-point input, got {x.dtype}")

        y = self.conv(F.pad(x, (self.left_pad, 0)))
        y = self.mix(F.elu(y))
        return F.elu(x + y)


class ContextEncoder(nn.Module):
    """Encode ``history[N, 61, 96]`` into a context latent and auxiliaries.

    ``latent_dim`` defaults to the deployed 16-D representation.  It remains a
    constructor argument solely for the required 8/16/24 latent ablation; the
    temporal and observation dimensions are fixed by the deployment contract.
    """

    history_length: Final[int] = HISTORY_LENGTH
    observation_dim: Final[int] = OBSERVATION_DIM
    feature_dim: Final[int] = FEATURE_DIM
    receptive_field: Final[int] = temporal_receptive_field()

    def __init__(self, latent_dim: int = CONTEXT_DIM) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if self.receptive_field != self.history_length:
            raise RuntimeError(
                "TCN architecture no longer matches the 61-frame history contract"
            )

        self.latent_dim = latent_dim
        self.input = nn.Conv1d(OBSERVATION_DIM, FEATURE_DIM, kernel_size=1)
        self.blocks = nn.Sequential(
            *(CausalBlock(FEATURE_DIM, dilation) for dilation in DILATIONS)
        )
        self.context = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        self.phase = nn.Linear(FEATURE_DIM, 2)
        self.frequency = nn.Linear(FEATURE_DIM, 1)
        self.contact = nn.Linear(FEATURE_DIM, 2)
        self.cart_lag = nn.Linear(FEATURE_DIM, 1)

    @staticmethod
    def _validate_history(history: torch.Tensor) -> None:
        if history.ndim != 3:
            raise ValueError(
                "ContextEncoder expects history with shape [N, 61, 96], "
                f"got {tuple(history.shape)}"
            )
        if history.shape[1:] != (HISTORY_LENGTH, OBSERVATION_DIM):
            raise ValueError(
                "ContextEncoder expects history with shape [N, 61, 96], "
                f"got {tuple(history.shape)}"
            )
        if not history.is_floating_point():
            raise TypeError(
                f"ContextEncoder expects floating-point history, got {history.dtype}"
            )

    def extract_feature(self, history: torch.Tensor) -> torch.Tensor:
        """Return the 128-D feature at the final causal time step."""

        self._validate_history(history)
        x = self.input(history.transpose(1, 2))
        x = self.blocks(x)
        feature = x[:, :, -1]
        if feature.shape != (history.shape[0], FEATURE_DIM):
            raise RuntimeError(f"unexpected TCN feature shape {tuple(feature.shape)}")
        return feature

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Return only ``z_hat`` for deployment wrappers."""

        return self.context(self.extract_feature(history))

    def forward(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature = self.extract_feature(history)
        context = self.context(feature)
        auxiliary = {
            "phase": self.phase(feature),
            "frequency": self.frequency(feature),
            "contact": self.contact(feature),
            "cart_lag": self.cart_lag(feature),
        }
        return context, auxiliary


__all__ = [
    "CONTEXT_DIM",
    "DILATIONS",
    "FEATURE_DIM",
    "HISTORY_LENGTH",
    "KERNEL_SIZE",
    "OBSERVATION_DIM",
    "CausalBlock",
    "ContextEncoder",
    "temporal_receptive_field",
]
