"""Configuration artifact contracts for generated calibration files."""

from __future__ import annotations

import pytest

from g1_rickshaw_lab.configuration import (
    G1_JOINT_ORDER,
    REQUIRED_CALIBRATION_FIELDS,
    REQUIRED_FEASIBILITY_RANGES,
    SLOPE_GRADIENTS,
    ConfigurationContractError,
    FeasibilityEnvelope,
    ResetPoseLibrary,
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
        "d6.robot_body_paths": [
            "/World/envs/env_0/Robot/left_grasp",
            "/World/envs/env_0/Robot/right_grasp",
        ],
        "d6.hitch_body_paths": [
            "/World/envs/env_0/Rickshaw/left_tow_hitch_link",
            "/World/envs/env_0/Rickshaw/right_tow_hitch_link",
        ],
        "d6.rotation_free_axes": [False, True, False],
        "d6.rotation_driven_axes": [True, False, True],
        "dex.q_open": [0.0, 0.0, 0.0, 0.0],
        "dex.q_grasp": [0.3, 0.3, 0.3, 0.3],
        "dex.left_grasp_center_frame": [0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0],
        "dex.right_grasp_center_frame": [0.0, -0.05, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    if name in vectors:
        return vectors[name]
    if name == "d6.reaction_is_joint_on_robot":
        return True
    if name == "fat.wrench_consistency_window_steps":
        return 5
    if name == "fat.com_radius":
        return 0.7
    return 1.0


def _valid_feasibility_mapping() -> dict:
    return {
        "schema_version": 1,
        "slopes": list(SLOPE_GRADIENTS),
        "joint_order": list(G1_JOINT_ORDER),
        "ranges": {name: _range_for(name) for name in REQUIRED_FEASIBILITY_RANGES},
        "calibration": {
            name: _calibration_for(name) for name in REQUIRED_CALIBRATION_FIELDS
        },
    }


def _valid_reset_pose_mapping() -> dict:
    torque_basis = {
        "tau_unloaded": [0.0] * 29,
        "tau_per_tangent_force": [0.0] * 29,
        "tau_per_normal_force": [0.0] * 29,
        "tau_per_tangent_difference": [0.0] * 29,
        "handle_wrenches_sln": [[0.0] * 6, [0.0] * 6],
        "wheel_contact_forces_sln": [
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 100.0],
        ],
    }
    return {
        "schema_version": 4,
        "joint_order": list(G1_JOINT_ORDER),
        "poses": [
            {
                "gradient": gradient,
                "root_pitch": 0.5 * gradient,
                "root_height": 0.74 + gradient,
                "q_reset": [gradient + 0.02 * index for index in range(29)],
                "q_ref_unloaded": [2.0 * gradient + 0.015 * index for index in range(29)],
                **torque_basis,
                "q_ref": [3.0 * gradient + 0.01 * index for index in range(29)],
            }
            for gradient in SLOPE_GRADIENTS
        ],
    }


def test_feasibility_envelope_requires_exact_schema_fields() -> None:
    mapping = _valid_feasibility_mapping()
    envelope = FeasibilityEnvelope.from_mapping(mapping)

    assert tuple(envelope.joint_order) == G1_JOINT_ORDER
    assert tuple(envelope.slopes) == SLOPE_GRADIENTS
    assert set(envelope.ranges) == set(REQUIRED_FEASIBILITY_RANGES)
    assert set(envelope.calibration) == set(REQUIRED_CALIBRATION_FIELDS)

    missing = _valid_feasibility_mapping()
    del missing["ranges"]["motor.strength"]
    with pytest.raises(ConfigurationContractError, match="missing"):
        FeasibilityEnvelope.from_mapping(missing)

    unknown = _valid_feasibility_mapping()
    unknown["calibration"]["unvalidated.default"] = 1.0
    with pytest.raises(ConfigurationContractError, match="unknown"):
        FeasibilityEnvelope.from_mapping(unknown)


def test_feasibility_envelope_rejects_joint_order_and_d6_axis_drift() -> None:
    mapping = _valid_feasibility_mapping()
    mapping["joint_order"][0], mapping["joint_order"][1] = (
        mapping["joint_order"][1],
        mapping["joint_order"][0],
    )
    with pytest.raises(ConfigurationContractError, match="fixed checkpoint order"):
        FeasibilityEnvelope.from_mapping(mapping)

    invalid_axis = _valid_feasibility_mapping()
    invalid_axis["calibration"]["d6.rotation_free_axes"] = [True, False, False]
    invalid_axis["calibration"]["d6.rotation_driven_axes"] = [True, False, False]
    with pytest.raises(ConfigurationContractError, match="cannot be both"):
        FeasibilityEnvelope.from_mapping(invalid_axis)


def test_d6_nominal_calibration_is_explicit_and_not_a_runtime_range() -> None:
    mapping = _valid_feasibility_mapping()
    mapping["calibration"]["d6.linear_stiffness_nominal"] = 5.0
    envelope = FeasibilityEnvelope.from_mapping(mapping)
    assert envelope.calibration["d6.linear_stiffness_nominal"] == 5.0
    assert not any(name.startswith("d6.") for name in envelope.ranges)


def test_reset_pose_library_requires_all_19_slopes_and_fixed_order() -> None:
    mapping = _valid_reset_pose_mapping()
    library = ResetPoseLibrary.from_mapping(mapping)

    assert tuple(pose.gradient for pose in library.poses) == SLOPE_GRADIENTS
    assert library.interpolate_q_ref(0.035) == pytest.approx(
        [3.0 * 0.035 + 0.01 * index for index in range(29)]
    )
    assert library.interpolate_q_reset(0.035) == pytest.approx(
        [0.035 + 0.02 * index for index in range(29)]
    )
    assert library.interpolate_q_ref_unloaded(0.035) == pytest.approx(
        [2.0 * 0.035 + 0.015 * index for index in range(29)]
    )
    assert library.interpolate_root_pitch(0.035) == pytest.approx(0.0175)
    assert library.interpolate_root_height(0.035) == pytest.approx(0.775)

    for gradient in (SLOPE_GRADIENTS[0], 0.04, SLOPE_GRADIENTS[-1]):
        pose = library.pose_for_gradient(gradient)
        assert library.interpolate_q_ref(gradient) == pose.q_ref
        assert library.interpolate_q_reset(gradient) == pose.q_reset
        assert library.interpolate_q_ref_unloaded(gradient) == pose.q_ref_unloaded
        assert library.interpolate_root_pitch(gradient) == pose.root_pitch
        assert library.interpolate_root_height(gradient) == pose.root_height

    duplicate = dict(mapping)
    duplicate["poses"] = list(mapping["poses"])
    duplicate["poses"][0] = {**duplicate["poses"][0], "gradient": 0.0}
    with pytest.raises(ConfigurationContractError, match="exactly one pose"):
        ResetPoseLibrary.from_mapping(duplicate)


@pytest.mark.parametrize("field", ("root_pitch", "root_height"))
def test_reset_pose_library_requires_explicit_root_pose(field: str) -> None:
    mapping = _valid_reset_pose_mapping()
    del mapping["poses"][0][field]

    with pytest.raises(ConfigurationContractError, match=field):
        ResetPoseLibrary.from_mapping(mapping)


@pytest.mark.parametrize(
    "method_name",
    (
        "interpolate_q_ref",
        "interpolate_q_reset",
        "interpolate_q_ref_unloaded",
        "interpolate_root_pitch",
        "interpolate_root_height",
    ),
)
@pytest.mark.parametrize(
    "gradient",
    (-0.081, 0.101, float("nan"), float("inf"), float("-inf")),
)
def test_reset_pose_interpolation_rejects_invalid_gradients(
    method_name: str, gradient: float
) -> None:
    library = ResetPoseLibrary.from_mapping(_valid_reset_pose_mapping())

    with pytest.raises(ConfigurationContractError):
        getattr(library, method_name)(gradient)


def test_reset_pose_library_rejects_missing_static_endpoint() -> None:
    mapping = {
        "schema_version": 4,
        "joint_order": list(G1_JOINT_ORDER),
        "poses": [
            {
                "gradient": gradient,
                "q_reset": [0.0] * 29,
                "tau_unloaded": [0.0] * 29,
                "tau_per_tangent_force": [0.0] * 29,
                "tau_per_normal_force": [0.0] * 29,
                "tau_per_tangent_difference": [0.0] * 29,
                "handle_wrenches_sln": [[0.0] * 6, [0.0] * 6],
                "wheel_contact_forces_sln": [
                    [0.0, 0.0, 100.0],
                    [0.0, 0.0, 100.0],
                ],
                "q_ref": [0.0] * 29,
            }
            for gradient in SLOPE_GRADIENTS
        ],
    }
    with pytest.raises(ConfigurationContractError, match="q_ref_unloaded"):
        ResetPoseLibrary.from_mapping(mapping)
