"""Configuration artifact contracts retained by the Mjlab runtime."""

from __future__ import annotations

import pytest

from g1_rickshaw_lab.configuration import (
    G1_JOINT_ORDER,
    REQUIRED_CALIBRATION_FIELDS,
    REQUIRED_FEASIBILITY_RANGES,
    SLOPE_GRADIENTS,
    ConfigurationContractError,
    FeasibilityEnvelope,
)


def _range_for(name: str) -> dict[str, float]:
    if name.startswith("payload.com."):
        return {"min": -0.05, "max": 0.05}
    if name == "payload.mass":
        return {"min": 0.0, "max": 10.0}
    if name == "terrain.friction":
        return {"min": 0.6, "max": 1.2}
    if name in {"command.acceleration_limit", "command.jerk_limit"}:
        return {"min": 0.1, "max": 1.0}
    return {"min": 0.01, "max": 10.0}


def _calibration_for(name: str):
    vectors = {
        "safety.hitch_height_bounds": [0.65, 0.85],
        "safety.rickshaw_pitch_bounds": [-0.2, 0.5],
        "fat.com_radius_bounds": [0.5, 0.85],
    }
    if name in vectors:
        return vectors[name]
    if name == "fat.wrench_consistency_window_steps":
        return 5
    return 0.7 if name == "fat.com_radius" else 1.0


def _valid_mapping() -> dict:
    return {
        "schema_version": 1,
        "slopes": list(SLOPE_GRADIENTS),
        "joint_order": list(G1_JOINT_ORDER),
        "ranges": {name: _range_for(name) for name in REQUIRED_FEASIBILITY_RANGES},
        "calibration": {
            name: _calibration_for(name) for name in REQUIRED_CALIBRATION_FIELDS
        },
    }


def test_feasibility_envelope_requires_exact_schema_fields() -> None:
    envelope = FeasibilityEnvelope.from_mapping(_valid_mapping())
    assert tuple(envelope.joint_order) == G1_JOINT_ORDER
    assert tuple(envelope.slopes) == SLOPE_GRADIENTS
    assert set(envelope.ranges) == set(REQUIRED_FEASIBILITY_RANGES)
    assert set(envelope.calibration) == set(REQUIRED_CALIBRATION_FIELDS)

    missing = _valid_mapping()
    del missing["ranges"]["torso.mass_delta"]
    with pytest.raises(ConfigurationContractError, match="missing"):
        FeasibilityEnvelope.from_mapping(missing)

    unknown = _valid_mapping()
    unknown["calibration"]["unvalidated.default"] = 1.0
    with pytest.raises(ConfigurationContractError, match="unknown"):
        FeasibilityEnvelope.from_mapping(unknown)


def test_feasibility_envelope_rejects_joint_order_drift() -> None:
    mapping = _valid_mapping()
    mapping["joint_order"][0], mapping["joint_order"][1] = (
        mapping["joint_order"][1],
        mapping["joint_order"][0],
    )
    with pytest.raises(ConfigurationContractError, match="fixed checkpoint order"):
        FeasibilityEnvelope.from_mapping(mapping)

def test_legacy_actuator_fields_are_rejected() -> None:
    mapping = _valid_mapping()
    mapping["calibration"]["control.linear_stiffness_nominal"] = 5.0
    with pytest.raises(ConfigurationContractError, match="unknown"):
        FeasibilityEnvelope.from_mapping(mapping)
