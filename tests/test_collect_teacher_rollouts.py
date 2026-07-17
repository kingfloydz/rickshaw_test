from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from collect_teacher_rollouts import _step_teacher_policy  # noqa: E402


class _FakeTeacher:
    def __init__(self) -> None:
        self.policy = self

    def encode(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return observation["policy"] * 0.5

    def distribution(
        self, current: torch.Tensor, latent: torch.Tensor
    ) -> torch.distributions.Independent:
        mean = current + latent
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, torch.ones_like(mean)), 1
        )


class _LazyStateEnvironment:
    def __init__(self) -> None:
        self.lazy_state: torch.Tensor | None = None
        self.reset_calls = 0

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.reset_calls += 1
        if self.lazy_state is not None:
            self.lazy_state.zero_()
        return {"policy": torch.full((2, 3), float(self.reset_calls))}, {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.lazy_state is None:
            self.lazy_state = torch.zeros_like(actions)
        self.lazy_state.copy_(actions)
        observation = {"policy": self.lazy_state + 1.0}
        reward = torch.sum(self.lazy_state, dim=-1)
        dones = torch.zeros(self.lazy_state.shape[0], dtype=torch.bool)
        return observation, reward, dones, {"lazy_state": self.lazy_state}


def test_teacher_collection_survives_reset_between_segments_without_inference_tensors() -> None:
    actor = _FakeTeacher()
    env = _LazyStateEnvironment()

    for _ in range(2):
        observation, _ = env.reset()
        for _ in range(2):
            (
                latent,
                mean,
                std,
                observation,
                reward,
                dones,
                extras,
            ) = _step_teacher_policy(actor, observation, env)
            tensors = (
                latent,
                mean,
                std,
                observation["policy"],
                reward,
                dones,
                extras["lazy_state"],
            )
            assert all(not torch.is_inference(tensor) for tensor in tensors)

    assert env.reset_calls == 2
    assert env.lazy_state is not None
    scale = torch.tensor(1.0, requires_grad=True)
    tracked_value = torch.sum(env.lazy_state * scale)
    assert tracked_value.requires_grad
