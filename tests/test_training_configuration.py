from __future__ import annotations

from pathlib import Path
import sys

import pytest

from g1_rickshaw_lab.training_contract import (
    MAINLINE_PARAMETERS,
    TRAINING_CONFIGURATION_SCHEMA_VERSION,
    finalize_training_configuration,
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
        "task": "Isaac-G1-Rickshaw-Directional-Slope-v0",
        "num_envs": 32,
        "seed": 42,
        "max_iterations": 10,
        "guide_parameters": {},
        "resolved_parameters": {},
        "actor_initialized_from_teacher": None,
        "stage_coverage": None,
        "mainline_parameters": dict(MAINLINE_PARAMETERS),
    }


def test_training_configuration_has_one_canonical_validator() -> None:
    configuration = finalize_training_configuration(_configuration())
    normalized = validate_training_configuration(configuration)

    assert normalized["mainline_parameters"] == MAINLINE_PARAMETERS
    assert validate_launcher_training_configuration(configuration) == normalized


def test_training_configuration_rejects_unknown_or_non_mainline_fields() -> None:
    configuration = finalize_training_configuration(_configuration())
    configuration["legacy_digest"] = "removed"
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_launcher_training_configuration(configuration)

    configuration = _configuration()
    configuration["mainline_parameters"]["latent_dim"] = 24
    with pytest.raises(ValueError, match="exactly"):
        validate_launcher_training_configuration(configuration)


def test_training_configuration_finalize_rejects_non_json_values() -> None:
    with pytest.raises(TypeError):
        finalize_training_configuration(
            {**_configuration(), "resolved_parameters": {"path": Path("x")}}
        )
    with pytest.raises(ValueError):
        finalize_training_configuration(
            {**_configuration(), "resolved_parameters": {"value": float("nan")}}
        )
