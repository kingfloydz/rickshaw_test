from __future__ import annotations

import math
import xml.etree.ElementTree as ET


from g1_rickshaw_lab.assets.g1_dex1 import (
    G1_DEX1_URDF_PATH,
    get_g1_spec,
    validate_g1_urdf,
)
from g1_rickshaw_lab.assets.rickshaw import get_rickshaw_spec, validate_rickshaw_urdf
from g1_rickshaw_lab.rickshaw_spec import RICKSHAW_URDF_SPEC
from g1_rickshaw_lab.static_equilibrium import fat2_reference_angle_scalar
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (
    build_assembled_spec,
    validate_assembled_model,
)


def test_fixed_grippers_leave_exactly_29_robot_dofs() -> None:
    root = ET.parse(G1_DEX1_URDF_PATH).getroot()
    gripper_joints = [
        joint
        for joint in root.findall("joint")
        if "dex1_finger_joint" in joint.attrib["name"]
    ]
    assert len(gripper_joints) == 4
    assert all(joint.attrib["type"] == "fixed" for joint in gripper_joints)
    assert validate_g1_urdf() == ()
    model = get_g1_spec().compile()
    assert model.njnt == 30  # free base + 29 G1 joints


def test_rickshaw_has_0_6m_wheels_aligned_with_lowered_body() -> None:
    assert validate_rickshaw_urdf() == ()
    spec = RICKSHAW_URDF_SPEC
    assert spec.wheel_radius == 0.3
    assert math.isclose(spec.body_vertical_offset, -(0.374999 - 0.3), abs_tol=1.0e-12)
    assert math.isclose(
        spec.base_com_x_before_shift - spec.base_com_x, 0.02, abs_tol=1.0e-12
    )
    model = get_rickshaw_spec().compile()
    for name in ("left_wheel_link", "right_wheel_link"):
        geom_id = int(model.body_geomadr[model.body(name).id])
        assert math.isclose(model.geom_size[geom_id, 0] * 2.0, 0.6, abs_tol=1.0e-12)


def test_g1_uses_official_builtin_position_actuator_defaults() -> None:
    from mjlab.actuator import BuiltinPositionActuatorCfg

    from g1_rickshaw_lab.assets.g1_dex1 import get_g1_robot_cfg
    from g1_rickshaw_lab.g1_motor_defaults import (
        ARMATURE_4010,
        ARMATURE_5020,
        ARMATURE_7520_14,
        ARMATURE_7520_22,
        DAMPING_4010,
        DAMPING_5020,
        DAMPING_7520_14,
        DAMPING_7520_22,
        DAMPING_RATIO,
        NATURAL_FREQUENCY,
        STIFFNESS_4010,
        STIFFNESS_5020,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
    )

    actuators = get_g1_robot_cfg().articulation.actuators
    assert len(actuators) == 6
    assert all(isinstance(actuator, BuiltinPositionActuatorCfg) for actuator in actuators)
    assert math.isclose(NATURAL_FREQUENCY, 20.0 * 3.1415926535)
    assert DAMPING_RATIO == 2.0
    expected = (
        (STIFFNESS_5020, DAMPING_5020, 25.0, ARMATURE_5020),
        (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
        (STIFFNESS_7520_22, DAMPING_7520_22, 139.0, ARMATURE_7520_22),
        (STIFFNESS_4010, DAMPING_4010, 5.0, ARMATURE_4010),
        (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
        (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    )
    for actuator, values in zip(actuators, expected, strict=True):
        actual = (
            actuator.stiffness,
            actuator.damping,
            actuator.effort_limit,
            actuator.armature,
        )
        assert all(math.isclose(value, target) for value, target in zip(actual, values, strict=True))


def test_assembled_model_uses_two_connections_and_only_tow_rod_collision() -> None:
    model = build_assembled_spec().compile()
    assert validate_assembled_model(model) == ()
    assert model.neq == 2


def test_fat2_prior_uses_tangent_and_normal_hand_force() -> None:
    first = fat2_reference_angle_scalar(
        handle_s=0.4,
        handle_n=0.8,
        hand_force_s=-120.0,
        hand_force_n=40.0,
        robot_mass=34.1299349,
        com_radius=0.715,
        theta_max=0.8,
    )
    without_normal = fat2_reference_angle_scalar(
        handle_s=0.4,
        handle_n=0.8,
        hand_force_s=-120.0,
        hand_force_n=0.0,
        robot_mass=34.1299349,
        com_radius=0.715,
        theta_max=0.8,
    )
    assert first != without_normal
