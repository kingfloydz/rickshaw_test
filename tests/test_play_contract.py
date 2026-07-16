"""CPU-only checks for the checkpoint-owned play/export command surface."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from play_student import validate_operational_play_arguments  # noqa: E402


def test_play_allows_only_operational_arguments() -> None:
    validate_operational_play_arguments(
        [
            "--headless",
            "--device",
            "cuda:1",
            "--num_envs=13",
            "--video",
            "--video_length",
            "200",
        ]
    )


@pytest.mark.parametrize(
    "argument",
    (
        "env.rolling_resistance.enabled=false",
        "env.rewards.fat2_prior_exp.weight=0.0",
        "agent.actor.latent_dim=24",
        "--agent",
        "rsl_rl_cfg_entry_point",
        "--disable_fabric",
    ),
)
def test_play_rejects_policy_environment_and_physics_overrides(argument: str) -> None:
    with pytest.raises(ValueError, match="rejects policy or environment override"):
        validate_operational_play_arguments(argument.split())
