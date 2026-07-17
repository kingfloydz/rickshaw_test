"""Temporal privileged encoder used only by the S0 teacher."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent

from g1_rickshaw_lab.policy_schema import (
    DEFAULT_CONTEXT_DIM,
    TEACHER_DYNAMIC_DIM,
    TEACHER_STATIC_DIM,
    validate_context_dim,
)

from .actor_critic import GaussianActor
from .context_encoder import (
    DILATIONS,
    FEATURE_DIM,
    OBSERVATION_DIM,
    CausalBlock,
    validate_history,
)


DYNAMIC_PRIVILEGE_DIM = TEACHER_DYNAMIC_DIM
STATIC_PRIVILEGE_DIM = TEACHER_STATIC_DIM
STATIC_FEATURE_DIM = 32


class TeacherEncoder(nn.Module):
    """Fuse observation/physical histories with episode-static physics."""

    def __init__(self, latent_dim: int = DEFAULT_CONTEXT_DIM) -> None:
        super().__init__()
        self.latent_dim = validate_context_dim(latent_dim)
        self.observation_input = nn.Conv1d(
            OBSERVATION_DIM, FEATURE_DIM, kernel_size=1
        )
        self.privilege_input = nn.Conv1d(
            DYNAMIC_PRIVILEGE_DIM, FEATURE_DIM, kernel_size=1
        )
        self.blocks = nn.Sequential(
            *(CausalBlock(FEATURE_DIM, dilation) for dilation in DILATIONS)
        )
        self.static = nn.Sequential(
            nn.Linear(STATIC_PRIVILEGE_DIM, STATIC_FEATURE_DIM),
            nn.ELU(),
        )
        self.context = nn.Linear(FEATURE_DIM + STATIC_FEATURE_DIM, self.latent_dim)

    def forward(
        self,
        observation_history: torch.Tensor,
        dynamic_privilege_history: torch.Tensor,
        static_privilege: torch.Tensor,
    ) -> torch.Tensor:
        validate_history(
            observation_history,
            feature_dim=OBSERVATION_DIM,
            name="observation_history",
        )
        validate_history(
            dynamic_privilege_history,
            feature_dim=DYNAMIC_PRIVILEGE_DIM,
            name="dynamic_privilege_history",
        )
        if static_privilege.ndim != 2 or static_privilege.shape[1] != STATIC_PRIVILEGE_DIM:
            raise ValueError(
                f"static_privilege must have shape [N, {STATIC_PRIVILEGE_DIM}]"
            )
        batch = observation_history.shape[0]
        if dynamic_privilege_history.shape[0] != batch or static_privilege.shape[0] != batch:
            raise ValueError("teacher encoder batch dimensions differ")

        observation = self.observation_input(observation_history.transpose(1, 2))
        privilege = self.privilege_input(
            dynamic_privilege_history.transpose(1, 2)
        )
        temporal = self.blocks(F.elu(observation + privilege))[:, :, -1]
        static = self.static(static_privilege)
        return self.context(torch.cat((temporal, static), dim=-1))


class G1RickshawTeacherActor(nn.Module):
    """S0 policy with temporal privilege compressed into the shared actor ABI."""

    def __init__(self, latent_dim: int = DEFAULT_CONTEXT_DIM) -> None:
        super().__init__()
        self.encoder = TeacherEncoder(latent_dim)
        self.actor = GaussianActor(latent_dim)

    def encode(
        self,
        observation_history: torch.Tensor,
        dynamic_privilege_history: torch.Tensor,
        static_privilege: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            observation_history,
            dynamic_privilege_history,
            static_privilege,
        )

    def forward_with_context(
        self,
        current: torch.Tensor,
        observation_history: torch.Tensor,
        dynamic_privilege_history: torch.Tensor,
        static_privilege: torch.Tensor,
    ) -> tuple[Independent, torch.Tensor]:
        context = self.encode(
            observation_history,
            dynamic_privilege_history,
            static_privilege,
        )
        return self.actor(current, context), context

    def forward(
        self,
        current: torch.Tensor,
        observation_history: torch.Tensor,
        dynamic_privilege_history: torch.Tensor,
        static_privilege: torch.Tensor,
    ) -> Independent:
        return self.forward_with_context(
            current,
            observation_history,
            dynamic_privilege_history,
            static_privilege,
        )[0]


__all__ = [
    "DYNAMIC_PRIVILEGE_DIM",
    "STATIC_PRIVILEGE_DIM",
    "G1RickshawTeacherActor",
    "TeacherEncoder",
]
