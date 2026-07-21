"""Official Unitree G1 motor and position-control defaults."""

from __future__ import annotations

from types import MappingProxyType

# Mirrors unitreerobotics/unitree_rl_mjlab src/assets/robots/unitree_g1/g1_constants.py.
NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _reflected_inertia(
    rotor_inertias: tuple[float, ...], gears: tuple[float, ...]
) -> float:
    return sum(
        inertia * gear * gear
        for inertia, gear in zip(rotor_inertias, gears, strict=True)
    )


ARMATURE_5020 = _reflected_inertia(
    (0.139e-4, 0.017e-4, 0.169e-4),
    (1.0, 1.0 + 46.0 / 18.0, 1.0 + 56.0 / 16.0),
)
ARMATURE_7520_14 = _reflected_inertia(
    (0.489e-4, 0.098e-4, 0.533e-4),
    (1.0, 4.5, 1.0 + 48.0 / 22.0),
)
ARMATURE_7520_22 = _reflected_inertia(
    (0.489e-4, 0.109e-4, 0.738e-4),
    (1.0, 4.5, 5.0),
)
ARMATURE_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1.0, 5.0, 5.0))


def _pd(armature: float) -> tuple[float, float]:
    return (
        armature * NATURAL_FREQUENCY**2,
        2.0 * DAMPING_RATIO * armature * NATURAL_FREQUENCY,
    )


STIFFNESS_5020, DAMPING_5020 = _pd(ARMATURE_5020)
STIFFNESS_7520_14, DAMPING_7520_14 = _pd(ARMATURE_7520_14)
STIFFNESS_7520_22, DAMPING_7520_22 = _pd(ARMATURE_7520_22)
STIFFNESS_4010, DAMPING_4010 = _pd(ARMATURE_4010)

_parameters = {
    "left_hip_pitch_joint": (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
    "left_hip_roll_joint": (STIFFNESS_7520_22, DAMPING_7520_22, 139.0, ARMATURE_7520_22),
    "left_hip_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
    "left_knee_joint": (STIFFNESS_7520_22, DAMPING_7520_22, 139.0, ARMATURE_7520_22),
    "left_ankle_pitch_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    "left_ankle_roll_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    "right_hip_pitch_joint": (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
    "right_hip_roll_joint": (STIFFNESS_7520_22, DAMPING_7520_22, 139.0, ARMATURE_7520_22),
    "right_hip_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
    "right_knee_joint": (STIFFNESS_7520_22, DAMPING_7520_22, 139.0, ARMATURE_7520_22),
    "right_ankle_pitch_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    "right_ankle_roll_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    "waist_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14, 88.0, ARMATURE_7520_14),
    "waist_roll_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
    "waist_pitch_joint": (2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 50.0, 2.0 * ARMATURE_5020),
}
for side in ("left", "right"):
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
    ):
        _parameters[f"{side}_{joint}_joint"] = (
            STIFFNESS_5020,
            DAMPING_5020,
            25.0,
            ARMATURE_5020,
        )
    for joint in ("wrist_pitch", "wrist_yaw"):
        _parameters[f"{side}_{joint}_joint"] = (
            STIFFNESS_4010,
            DAMPING_4010,
            5.0,
            ARMATURE_4010,
        )

G1_MOTOR_PARAMETERS_BY_JOINT = MappingProxyType(_parameters)
G1_MOTOR_JOINT_ORDER = tuple(_parameters)
G1_JOINT_STIFFNESS = tuple(value[0] for value in _parameters.values())
G1_JOINT_DAMPING = tuple(value[1] for value in _parameters.values())
G1_JOINT_EFFORT_LIMITS = tuple(value[2] for value in _parameters.values())
G1_JOINT_ARMATURE = tuple(value[3] for value in _parameters.values())
G1_ACTION_SCALE = tuple(
    0.25 * effort_limit / stiffness
    for stiffness, _, effort_limit, _ in _parameters.values()
)


__all__ = [
    "ARMATURE_4010",
    "ARMATURE_5020",
    "ARMATURE_7520_14",
    "ARMATURE_7520_22",
    "DAMPING_4010",
    "DAMPING_5020",
    "DAMPING_7520_14",
    "DAMPING_7520_22",
    "G1_ACTION_SCALE",
    "G1_JOINT_ARMATURE",
    "G1_JOINT_DAMPING",
    "G1_JOINT_EFFORT_LIMITS",
    "G1_JOINT_STIFFNESS",
    "G1_MOTOR_JOINT_ORDER",
    "G1_MOTOR_PARAMETERS_BY_JOINT",
    "STIFFNESS_4010",
    "STIFFNESS_5020",
    "STIFFNESS_7520_14",
    "STIFFNESS_7520_22",
]
