from __future__ import annotations

from pathlib import Path
import sys

import pytest

from g1_rickshaw_lab.training_contract import (
    TRAINING_CONFIGURATION_SCHEMA_VERSION,
    finalize_training_configuration,
    training_configuration_sha256,
    validate_training_configuration,
)


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _training_configuration import (  # noqa: E402
    validate_training_configuration as validate_launcher_training_configuration,
)


def _configuration() -> dict:
    return {
        "schema_version": TRAINING_CONFIGURATION_SCHEMA_VERSION,
        "stage": "s0_teacher",
        "formal": False,
        "task": "Isaac-G1-Rickshaw-Directional-Slope-v0",
        "num_envs": 32,
        "seed": 42,
        "max_iterations": 10,
        "argv": [],
        "hydra_overrides": [],
        "guide_parameters": {},
        "resolved_parameters": {},
        "actor_initialized_from_teacher": None,
        "stage_coverage": None,
        "ablation_values": {
            "fat2_weight": "0.1",
            "rollout_steps": "48",
            "latent_dim": "16",
        },
        "inputs_sha256": {"feasibility": "0" * 64},
    }


def test_training_configuration_has_one_canonical_validator_and_digest() -> None:
    signed = finalize_training_configuration(_configuration())
    signed["content_sha256"] = signed["content_sha256"].upper()

    normalized = validate_training_configuration(signed, require_formal=False)

    assert normalized["ablation_values"] == {
        "fat2_weight": 0.1,
        "rollout_steps": 48,
        "latent_dim": 16,
    }
    assert normalized["content_sha256"] == training_configuration_sha256(normalized)
    assert validate_launcher_training_configuration(signed) == normalized


def test_training_configuration_digest_rejects_mutation() -> None:
    signed = finalize_training_configuration(_configuration())
    signed["stage"] = "s2_student_ppo"

    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        validate_launcher_training_configuration(signed)


def test_training_configuration_finalize_rejects_non_json_values() -> None:
    with pytest.raises(TypeError):
        finalize_training_configuration(
            {**_configuration(), "resolved_parameters": {"path": Path("x")}}
        )
    with pytest.raises(ValueError):
        finalize_training_configuration(
            {**_configuration(), "resolved_parameters": {"value": float("nan")}}
        )
