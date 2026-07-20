from __future__ import annotations

from pathlib import Path

import pytest
from g1_rickshaw_lab.training_contract import (
    DEFAULT_TRAINING_PARAMETERS,
    GUIDE_TRAINING_PARAMETERS,
    SUPPORTED_FAT2_WEIGHTS,
    TRAINING_CONFIGURATION_SCHEMA_VERSION,
    finalize_training_configuration,
    validate_guide_training_configuration,
    validate_training_configuration,
)


def _configuration() -> dict:
    return {
        "schema_version": TRAINING_CONFIGURATION_SCHEMA_VERSION,
        "stage": "s0_teacher",
        "task": "Isaac-G1-Rickshaw-Directional-Slope-v0",
        "num_envs": 32,
        "seed": 42,
        "max_iterations": 10,
        "guide_parameters": {},
        "resolved_parameters": {},
        "actor_initialized_from_teacher": None,
        "stage_coverage": None,
        "training_parameters": dict(DEFAULT_TRAINING_PARAMETERS),
    }


def test_training_configuration_has_one_canonical_validator() -> None:
    configuration = finalize_training_configuration(_configuration())
    normalized = validate_training_configuration(configuration)

    assert normalized["training_parameters"] == DEFAULT_TRAINING_PARAMETERS


def test_s0_configuration_binds_startup_randomization() -> None:
    configuration = _configuration()
    configuration["guide_parameters"] = dict(GUIDE_TRAINING_PARAMETERS["s0_teacher"])

    validated = validate_guide_training_configuration(
        configuration, expected_stage="s0_teacher"
    )

    assert validated["guide_parameters"] == {
        "domain_randomization": "startup_fixed",
        "terrain_slopes": "startup_balanced_fixed",
        "observation_noise": "unitree_g1_uniform",
    }
    configuration["guide_parameters"] = {}
    with pytest.raises(ValueError, match="guide parameters differ"):
        validate_guide_training_configuration(
            configuration, expected_stage="s0_teacher"
        )


def test_training_configuration_rejects_unknown_or_non_mainline_fields() -> None:
    configuration = finalize_training_configuration(_configuration())
    configuration["legacy_digest"] = "removed"
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_training_configuration(configuration)

    configuration = _configuration()
    configuration["training_parameters"]["latent_dim"] = 7
    with pytest.raises(ValueError, match="context dimension"):
        validate_training_configuration(configuration)


@pytest.mark.parametrize("latent_dim", (8, 16, 24, 32))
@pytest.mark.parametrize("rollout_steps", (24, 48, 64))
@pytest.mark.parametrize("fat2_weight", SUPPORTED_FAT2_WEIGHTS)
def test_training_configuration_accepts_supported_variants(
    fat2_weight: float, latent_dim: int, rollout_steps: int
) -> None:
    configuration = _configuration()
    configuration["training_parameters"].update(
        fat2_weight=fat2_weight,
        latent_dim=latent_dim,
        rollout_steps=rollout_steps,
    )
    normalized = validate_training_configuration(configuration)
    assert normalized["training_parameters"]["fat2_weight"] == fat2_weight
    assert normalized["training_parameters"]["latent_dim"] == latent_dim
    assert normalized["training_parameters"]["rollout_steps"] == rollout_steps


@pytest.mark.parametrize("field", ("latent_dim", "rollout_steps"))
def test_training_configuration_rejects_non_integer_variants(field: str) -> None:
    configuration = _configuration()
    configuration["training_parameters"][field] = 24.5
    with pytest.raises(ValueError, match="must be an integer"):
        validate_training_configuration(configuration)


@pytest.mark.parametrize("fat2_weight", (-0.1, 0.3, True, "0.1"))
def test_training_configuration_rejects_unsupported_fat2_weight(
    fat2_weight: object,
) -> None:
    configuration = _configuration()
    configuration["training_parameters"]["fat2_weight"] = fat2_weight
    with pytest.raises(ValueError, match="fat2_weight"):
        validate_training_configuration(configuration)


def test_training_configuration_finalize_rejects_non_json_values() -> None:
    with pytest.raises(TypeError):
        finalize_training_configuration(
            {**_configuration(), "resolved_parameters": {"path": Path("x")}}
        )
    with pytest.raises(ValueError):
        finalize_training_configuration(
            {**_configuration(), "resolved_parameters": {"value": float("nan")}}
        )
