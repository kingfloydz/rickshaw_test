"""Versioned runtime configuration contracts for the G1 rickshaw task.

The feasibility and reset-pose files are generated artifacts, not places for
fallback values.  Loading either file therefore performs complete validation
before returning an object that can be used by training or deployment code.

Canonical ``feasibility_envelope.yaml`` layout::

    schema_version: 2
    slopes: [-0.08, ..., 0.0, ..., 0.10]
    joint_order: [29 exact G1 joint names]
    ranges:
      payload.mass: {min: 0.0, max: 10.0}
      # all names in REQUIRED_FEASIBILITY_RANGES are required
    calibration:
      rickshaw.pitch_inertia_about_axle: 1.0
      # all names in REQUIRED_CALIBRATION_FIELDS are required

Nested mappings are accepted in ``ranges`` and ``calibration`` and are
flattened with dots.  An interval may be written as ``{min: x, max: y}`` or as
``[x, y]``.  The canonical serializer always emits the mapping form.

Canonical reset-pose layout::

    schema_version: 4
    joint_order: [29 exact G1 joint names]
    poses:
      - gradient: -0.06
        q_reset: [29 finite reset-state joint positions]
        q_ref_unloaded: [29 finite gravity-only PD targets]
        tau_unloaded: [29 finite gravity-only joint torques in Nm]
        tau_per_tangent_force: [29 finite joint-torque coefficients in Nm/N]
        tau_per_normal_force: [29 finite joint-torque coefficients in Nm/N]
        tau_per_tangent_difference: [29 coefficients for F_s,left - F_s,right]
        handle_wrenches_sln: [[left Fs, Fl, Fn, Ms, Ml, Mn], [right ...]]
        wheel_contact_forces_sln: [[left Fs, Fl, Fn], [right ...]]
        q_ref: [29 finite static-preload PD targets]
      # exactly one entry for every required gradient
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .slope_contract import SLOPE_GRADIENTS


FEASIBILITY_SCHEMA_VERSION = 1
RESET_POSE_SCHEMA_VERSION = 4
RESET_POSE_GRADIENTS = SLOPE_GRADIENTS
SLOPE_MATCH_TOLERANCE = 1.0e-9
RESET_TORQUE_LIMIT_FRACTION = 0.86

# This is the source-URDF order after applying the guide's one-time grouping
# rule: lower_names + waist_names + arm_names.  Runtime regex ordering is never
# used as a policy/deployment contract.
G1_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
FIXED_G1_JOINT_ORDER = G1_JOINT_ORDER

# Nominal reset PD gains and continuous hardware limits in policy-joint order.
RESET_LOWER_STIFFNESS = (300.0, 300.0, 300.0, 300.0, 200.0, 200.0) * 2
RESET_WAIST_STIFFNESS = (5000.0,) * 3
RESET_ARM_STIFFNESS = (3000.0,) * 14
LOWER_HARDWARE_EFFORT_LIMITS = (88.0, 88.0, 88.0, 139.0, 50.0, 50.0) * 2
WAIST_HARDWARE_EFFORT_LIMITS = (88.0, 50.0, 50.0)
ARM_HARDWARE_EFFORT_LIMITS = (25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4) * 2

# Marginal bounds produced by the feasibility scan for runtime domain parameters.
REQUIRED_FEASIBILITY_RANGES = (
    "payload.mass",
    "payload.com.x",
    "payload.com.y",
    "payload.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
    "motor.strength",
    "joint.model_error",
    "control.delay",
    "observation.delay",
    "command.acceleration_limit",
    "command.jerk_limit",
)

# Values that the guide explicitly leaves MISSING until asset inspection,
# hardware specification, calibration, or feasibility validation has run.
# Vector lengths are validated below; scalar fields must be finite.
REQUIRED_CALIBRATION_FIELDS = (
    "rickshaw.pitch_inertia_about_axle",
    "rickshaw_pose.hitch_height_target",
    "rickshaw_pose.hitch_height_tolerance",
    "rickshaw_pose.hitch_vertical_speed_tolerance",
    "rolling_resistance.c_rr_nominal",
    "terrain.friction_nominal",
    "d6.linear_stiffness_nominal",
    "d6.linear_damping_nominal",
    "d6.angular_stiffness_nominal",
    "d6.angular_damping_nominal",
    "d6.max_force_nominal",
    "d6.max_torque_nominal",
    "d6.linear_limit_nominal",
    "d6.angular_limit_nominal",
    "fat.robot_mass",
    "fat.com_radius",
    "fat.com_radius_bounds",
    "fat.wrench_consistency_relative_tolerance",
    "fat.wrench_consistency_absolute_floor_n",
    "fat.wrench_consistency_window_steps",
    "support.foot_half_length",
    "support.foot_half_width",
    "support.foot_center_offset_x",
    "safety.theta_max",
    "safety.illegal_contact_force_threshold",
    "safety.robot_cart_contact_force_threshold",
    "safety.cart_ground_contact_force_threshold",
    "safety.minimum_wheel_normal_force",
    "safety.min_ground_reaction",
    "safety.d6_residual_limit",
    "safety.d6_impulse_limit",
    "safety.hitch_height_bounds",
    "safety.rickshaw_pitch_bounds",
    "safety.corridor_half_width",
    "safety.heading_error_limit",
    "safety.overspeed_margin",
    "safety.arm_torque_limit",
    "d6.robot_body_paths",
    "d6.hitch_body_paths",
    "d6.rotation_free_axes",
    "d6.rotation_driven_axes",
    "d6.reaction_is_joint_on_robot",
    "reset.hand_position_tolerance",
    "control.leg_stiffness",
    "control.leg_damping",
    "control.foot_stiffness",
    "control.foot_damping",
    "control.waist_stiffness",
    "control.waist_damping",
    "control.arm_stiffness",
    "control.arm_damping",
    "dex.actuator_stiffness",
    "dex.actuator_damping",
    "dex.effort_limit",
    "dex.velocity_limit",
    "dex.q_open",
    "dex.q_grasp",
    "dex.grasp_velocity",
    "dex.grasp_timeout",
    "dex.left_grasp_center_frame",
    "dex.right_grasp_center_frame",
)

_CALIBRATION_VECTOR_LENGTHS = {
    "fat.com_radius_bounds": 2,
    "safety.hitch_height_bounds": 2,
    "safety.rickshaw_pitch_bounds": 2,
    "dex.q_open": 4,
    "dex.q_grasp": 4,
    # xyz + unit quaternion in wxyz order.
    "dex.left_grasp_center_frame": 7,
    "dex.right_grasp_center_frame": 7,
}
_CALIBRATION_STRING_VECTOR_LENGTHS = {
    "d6.robot_body_paths": 2,
    "d6.hitch_body_paths": 2,
}
_CALIBRATION_BOOL_VECTOR_LENGTHS = {
    "d6.rotation_free_axes": 3,
    "d6.rotation_driven_axes": 3,
}
_CALIBRATION_STRICTLY_POSITIVE = frozenset(
    {
        "rickshaw.pitch_inertia_about_axle",
        "rickshaw_pose.hitch_height_target",
        "rickshaw_pose.hitch_height_tolerance",
        "rickshaw_pose.hitch_vertical_speed_tolerance",
        "rolling_resistance.c_rr_nominal",
        "terrain.friction_nominal",
        "d6.linear_stiffness_nominal",
        "d6.linear_damping_nominal",
        "d6.angular_stiffness_nominal",
        "d6.angular_damping_nominal",
        "d6.max_force_nominal",
        "d6.max_torque_nominal",
        "d6.linear_limit_nominal",
        "d6.angular_limit_nominal",
        "fat.robot_mass",
        "fat.com_radius",
        "fat.wrench_consistency_absolute_floor_n",
        "support.foot_half_length",
        "support.foot_half_width",
        "support.foot_center_offset_x",
        "safety.theta_max",
        "safety.illegal_contact_force_threshold",
        "safety.robot_cart_contact_force_threshold",
        "safety.cart_ground_contact_force_threshold",
        "safety.minimum_wheel_normal_force",
        "safety.min_ground_reaction",
        "safety.d6_residual_limit",
        "safety.d6_impulse_limit",
        "safety.corridor_half_width",
        "safety.heading_error_limit",
        "safety.overspeed_margin",
        "safety.arm_torque_limit",
        "reset.hand_position_tolerance",
        "control.leg_stiffness",
        "control.leg_damping",
        "control.foot_stiffness",
        "control.foot_damping",
        "control.waist_stiffness",
        "control.waist_damping",
        "control.arm_stiffness",
        "control.arm_damping",
        "dex.actuator_stiffness",
        "dex.actuator_damping",
        "dex.effort_limit",
        "dex.velocity_limit",
        "dex.grasp_velocity",
        "dex.grasp_timeout",
    }
)

_NONNEGATIVE_RANGE_NAMES = frozenset(
    name
    for name in REQUIRED_FEASIBILITY_RANGES
    if name
    not in {
        "payload.com.x",
        "payload.com.y",
        "payload.com.z",
        "joint.model_error",
    }
)

_NOMINAL_CALIBRATION_BY_RANGE = {
    "rolling_resistance.c_rr": "rolling_resistance.c_rr_nominal",
    "terrain.friction": "terrain.friction_nominal",
}


class ConfigurationContractError(ValueError):
    """Raised when a generated configuration violates the runtime contract."""


class ConfigurationDependencyError(RuntimeError):
    """Raised when an optional parser dependency is unavailable."""


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationContractError(f"{path} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationContractError(f"{path} must be finite, got {value!r}")
    return result


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationContractError(f"{path} must be a mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ConfigurationContractError(f"{path} keys must be non-empty strings")
    return value


def _expect_exact_keys(mapping: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unknown={extra}")
        raise ConfigurationContractError(f"invalid {path} fields: " + ", ".join(details))


def validate_joint_order(joint_order: Iterable[str], *, path: str = "joint_order") -> tuple[str, ...]:
    if isinstance(joint_order, (str, bytes)):
        raise ConfigurationContractError(f"{path} must be a sequence of joint names")
    try:
        names = tuple(joint_order)
    except TypeError as exc:
        raise ConfigurationContractError(f"{path} must be iterable") from exc
    if not all(isinstance(name, str) and name for name in names):
        raise ConfigurationContractError(f"{path} must contain non-empty strings")
    if len(names) != 29:
        raise ConfigurationContractError(f"{path} must contain exactly 29 joints, got {len(names)}")
    if len(set(names)) != len(names):
        raise ConfigurationContractError(f"{path} contains duplicate joint names")
    if names != G1_JOINT_ORDER:
        mismatch = next(
            (
                index,
                expected,
                actual,
            )
            for index, (expected, actual) in enumerate(zip(G1_JOINT_ORDER, names, strict=True))
            if expected != actual
        )
        index, expected, actual = mismatch
        raise ConfigurationContractError(
            f"{path}[{index}] is {actual!r}; fixed checkpoint order requires {expected!r}"
        )
    return names


def _validate_required_slopes(values: Any, path: str = "slopes") -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ConfigurationContractError(f"{path} must be a sequence")
    slopes = tuple(_finite_float(value, f"{path}[{index}]") for index, value in enumerate(values))
    if len(slopes) != len(SLOPE_GRADIENTS):
        raise ConfigurationContractError(
            f"{path} must contain exactly {len(SLOPE_GRADIENTS)} gradients, got {len(slopes)}"
        )
    for expected in SLOPE_GRADIENTS:
        matches = [
            index
            for index, actual in enumerate(slopes)
            if math.isclose(actual, expected, rel_tol=0.0, abs_tol=SLOPE_MATCH_TOLERANCE)
        ]
        if len(matches) != 1:
            raise ConfigurationContractError(
                f"{path} requires exactly one occurrence of gradient {expected}; found {len(matches)}"
            )
    return SLOPE_GRADIENTS


@dataclass(frozen=True, slots=True)
class NumericRange:
    """A finite closed interval used by feasibility and training configs."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum, "range.min")
        maximum = _finite_float(self.maximum, "range.max")
        if minimum > maximum:
            raise ConfigurationContractError(f"range min {minimum} exceeds max {maximum}")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_value(cls, value: Any, *, path: str = "range") -> "NumericRange":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            interval = _expect_mapping(value, path)
            _expect_exact_keys(interval, {"min", "max"}, path)
            return cls(interval["min"], interval["max"])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ConfigurationContractError(f"{path} interval sequence must have length 2")
            return cls(value[0], value[1])
        return cls(value, value)

    def contains(self, value: float | "NumericRange", *, tolerance: float = 0.0) -> bool:
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and non-negative")
        candidate = value if isinstance(value, NumericRange) else NumericRange.from_value(value)
        return (
            candidate.minimum >= self.minimum - tolerance
            and candidate.maximum <= self.maximum + tolerance
        )

    def assert_contains(
        self,
        value: float | "NumericRange",
        *,
        name: str = "value",
        tolerance: float = 0.0,
    ) -> None:
        candidate = value if isinstance(value, NumericRange) else NumericRange.from_value(value, path=name)
        if not self.contains(candidate, tolerance=tolerance):
            raise ConfigurationContractError(
                f"{name}=[{candidate.minimum}, {candidate.maximum}] is outside feasibility "
                f"envelope [{self.minimum}, {self.maximum}]"
            )

    def to_mapping(self) -> dict[str, float]:
        return {"min": self.minimum, "max": self.maximum}


def _looks_like_interval(value: Any) -> bool:
    if isinstance(value, NumericRange):
        return True
    if isinstance(value, Mapping):
        return set(value) == {"min", "max"}
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2


def _flatten_interval_mapping(
    value: Mapping[str, Any], *, path: str = "", allow_scalars: bool = False
) -> dict[str, NumericRange]:
    result: dict[str, NumericRange] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key or key.startswith(".") or key.endswith("."):
            raise ConfigurationContractError("range keys must be non-empty dotted identifiers")
        name = f"{path}.{key}" if path else key
        if _looks_like_interval(child):
            result[name] = NumericRange.from_value(child, path=f"ranges.{name}")
        elif not isinstance(child, Mapping):
            if not allow_scalars:
                raise ConfigurationContractError(
                    f"ranges.{name} must be an explicit {{min, max}} or [min, max] interval"
                )
            result[name] = NumericRange.from_value(child, path=f"ranges.{name}")
        else:
            result.update(
                _flatten_interval_mapping(child, path=name, allow_scalars=allow_scalars)
            )
    return result


def _flatten_values(value: Mapping[str, Any], *, path: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key or key.startswith(".") or key.endswith("."):
            raise ConfigurationContractError("calibration keys must be non-empty dotted identifiers")
        name = f"{path}.{key}" if path else key
        if isinstance(child, Mapping):
            result.update(_flatten_values(child, path=name))
        else:
            result[name] = child
    return result


def _validate_calibration(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    flattened = _flatten_values(calibration)
    _expect_exact_keys(flattened, set(REQUIRED_CALIBRATION_FIELDS), "calibration")
    validated: dict[str, Any] = {}
    for name in REQUIRED_CALIBRATION_FIELDS:
        value = flattened[name]
        string_vector_length = _CALIBRATION_STRING_VECTOR_LENGTHS.get(name)
        if string_vector_length is not None:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ConfigurationContractError(
                    f"calibration.{name} must be a length-{string_vector_length} string sequence"
                )
            if len(value) != string_vector_length or not all(
                isinstance(component, str) and component for component in value
            ):
                raise ConfigurationContractError(
                    f"calibration.{name} must contain {string_vector_length} non-empty strings"
                )
            validated[name] = tuple(value)
            continue
        bool_vector_length = _CALIBRATION_BOOL_VECTOR_LENGTHS.get(name)
        if bool_vector_length is not None:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ConfigurationContractError(
                    f"calibration.{name} must be a length-{bool_vector_length} bool sequence"
                )
            if len(value) != bool_vector_length or not all(
                isinstance(component, bool) for component in value
            ):
                raise ConfigurationContractError(
                    f"calibration.{name} must contain {bool_vector_length} explicit booleans"
                )
            validated[name] = tuple(value)
            continue
        if name == "d6.reaction_is_joint_on_robot":
            if not isinstance(value, bool):
                raise ConfigurationContractError(f"calibration.{name} must be an explicit boolean")
            validated[name] = value
            continue
        if name == "fat.wrench_consistency_window_steps":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationContractError(f"calibration.{name} must be a positive integer")
            validated[name] = value
            continue
        vector_length = _CALIBRATION_VECTOR_LENGTHS.get(name)
        if vector_length is None:
            scalar = _finite_float(value, f"calibration.{name}")
            if name in _CALIBRATION_STRICTLY_POSITIVE and scalar <= 0.0:
                raise ConfigurationContractError(f"calibration.{name} must be positive")
            validated[name] = scalar
            continue
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ConfigurationContractError(f"calibration.{name} must be a length-{vector_length} sequence")
        if len(value) != vector_length:
            raise ConfigurationContractError(
                f"calibration.{name} must have length {vector_length}, got {len(value)}"
            )
        vector = tuple(
            _finite_float(component, f"calibration.{name}[{index}]")
            for index, component in enumerate(value)
        )
        if name in {
            "fat.com_radius_bounds",
            "safety.hitch_height_bounds",
            "safety.rickshaw_pitch_bounds",
        }:
            if vector[0] >= vector[1]:
                raise ConfigurationContractError(
                    f"calibration.{name} lower bound must be less than its upper bound"
                )
        if name.endswith("grasp_center_frame"):
            quaternion_norm = math.sqrt(sum(component * component for component in vector[3:]))
            if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
                raise ConfigurationContractError(
                    f"calibration.{name} quaternion must be normalized, norm={quaternion_norm}"
                )
        validated[name] = vector
    for free, driven in zip(
        validated["d6.rotation_free_axes"],
        validated["d6.rotation_driven_axes"],
        strict=True,
    ):
        if free and driven:
            raise ConfigurationContractError(
                "a D6 rotation axis cannot be both physically free and driven"
            )
    theta_max = validated["safety.theta_max"]
    if theta_max >= math.pi / 2.0:
        raise ConfigurationContractError("calibration.safety.theta_max must lie in (0, pi/2)")
    wrench_tolerance = validated["fat.wrench_consistency_relative_tolerance"]
    if not 0.0 <= wrench_tolerance <= 1.0:
        raise ConfigurationContractError(
            "calibration.fat.wrench_consistency_relative_tolerance must lie in [0,1]"
        )
    radius_min, radius_max = validated["fat.com_radius_bounds"]
    if radius_min <= 0.0 or not radius_min <= validated["fat.com_radius"] <= radius_max:
        raise ConfigurationContractError(
            "calibration.fat.com_radius must lie within positive fat.com_radius_bounds"
        )
    return MappingProxyType(validated)


@dataclass(frozen=True, slots=True)
class FeasibilityEnvelope:
    """Validated feasibility scan output and runtime range authority."""

    ranges: Mapping[str, NumericRange]
    calibration: Mapping[str, Any]
    slopes: tuple[float, ...] = SLOPE_GRADIENTS
    joint_order: tuple[str, ...] = G1_JOINT_ORDER
    schema_version: int = FEASIBILITY_SCHEMA_VERSION
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != FEASIBILITY_SCHEMA_VERSION
        ):
            raise ConfigurationContractError(
                f"unsupported feasibility schema_version={self.schema_version!r}; "
                f"expected {FEASIBILITY_SCHEMA_VERSION}"
            )
        slopes = _validate_required_slopes(self.slopes)
        joint_order = validate_joint_order(self.joint_order)
        parsed_ranges = {
            name: NumericRange.from_value(value, path=f"ranges.{name}")
            for name, value in self.ranges.items()
        }
        _expect_exact_keys(parsed_ranges, set(REQUIRED_FEASIBILITY_RANGES), "ranges")
        for name in _NONNEGATIVE_RANGE_NAMES:
            if parsed_ranges[name].minimum < 0.0:
                raise ConfigurationContractError(f"ranges.{name}.min must be non-negative")
        # The limiter is undefined at zero and all force/drive magnitudes must
        # be strictly usable, not merely non-negative placeholders.
        for name in (
            "terrain.friction",
            "wheel.left_damping",
            "wheel.right_damping",
            "motor.strength",
            "command.acceleration_limit",
            "command.jerk_limit",
        ):
            if parsed_ranges[name].minimum <= 0.0:
                raise ConfigurationContractError(f"ranges.{name}.min must be positive")
        calibration = _validate_calibration(self.calibration)
        for range_name, calibration_name in _NOMINAL_CALIBRATION_BY_RANGE.items():
            nominal = float(calibration[calibration_name])
            interval = parsed_ranges[range_name]
            if not interval.minimum <= nominal <= interval.maximum:
                raise ConfigurationContractError(
                    f"calibration.{calibration_name}={nominal} lies outside ranges.{range_name}"
                )
        object.__setattr__(self, "slopes", slopes)
        object.__setattr__(self, "joint_order", joint_order)
        object.__setattr__(self, "ranges", MappingProxyType(parsed_ranges))
        object.__setattr__(self, "calibration", calibration)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, source_path: str | Path | None = None
    ) -> "FeasibilityEnvelope":
        data = _expect_mapping(mapping, "feasibility envelope")
        _expect_exact_keys(
            data,
            {"schema_version", "slopes", "joint_order", "ranges", "calibration"},
            "feasibility envelope",
        )
        ranges = _flatten_interval_mapping(
            _expect_mapping(data["ranges"], "ranges"), allow_scalars=False
        )
        calibration = _expect_mapping(data["calibration"], "calibration")
        return cls(
            schema_version=data["schema_version"],
            slopes=data["slopes"],
            joint_order=data["joint_order"],
            ranges=ranges,
            calibration=calibration,
            source_path=None if source_path is None else Path(source_path),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slopes": list(self.slopes),
            "joint_order": list(self.joint_order),
            "ranges": {name: self.ranges[name].to_mapping() for name in REQUIRED_FEASIBILITY_RANGES},
            "calibration": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.calibration.items()
            },
        }

    def assert_contains(
        self,
        name: str,
        value: float | Sequence[float] | Mapping[str, float] | NumericRange,
        *,
        tolerance: float = 1.0e-12,
    ) -> None:
        try:
            allowed = self.ranges[name]
        except KeyError as exc:
            raise ConfigurationContractError(f"unknown feasibility range {name!r}") from exc
        allowed.assert_contains(
            NumericRange.from_value(value, path=f"training_ranges.{name}"),
            name=f"training_ranges.{name}",
            tolerance=tolerance,
        )

    def assert_sampling_ranges(
        self,
        sampling_ranges: Mapping[str, Any],
        *,
        require_all: bool = True,
        tolerance: float = 1.0e-12,
    ) -> None:
        parsed = _flatten_interval_mapping(
            _expect_mapping(sampling_ranges, "training ranges"), allow_scalars=True
        )
        unknown = sorted(set(parsed) - set(self.ranges))
        missing = sorted(set(self.ranges) - set(parsed)) if require_all else []
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ConfigurationContractError("invalid training range fields: " + ", ".join(details))
        for name, candidate in parsed.items():
            self.ranges[name].assert_contains(
                candidate, name=f"training_ranges.{name}", tolerance=tolerance
            )


def _require_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigurationDependencyError(
            "PyYAML is required to load G1 rickshaw configuration files; install it with "
            "`python -m pip install PyYAML`."
        ) from exc
    return yaml


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    yaml = _require_yaml()

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ConfigurationContractError(f"duplicate YAML key {key!r} in {path}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationContractError(f"invalid YAML in {file_path}: {exc}") from exc
    if value is None:
        raise ConfigurationContractError(f"configuration file is empty: {file_path}")
    return _expect_mapping(value, str(file_path))


def load_feasibility_envelope(path: str | Path) -> FeasibilityEnvelope:
    """Load and fully validate a feasibility envelope YAML file."""

    file_path = Path(path)
    return FeasibilityEnvelope.from_mapping(_load_yaml_mapping(file_path), source_path=file_path)


def assert_sampling_ranges_within_envelope(
    sampling_ranges: Mapping[str, Any],
    envelope: FeasibilityEnvelope | str | Path,
    *,
    require_all: bool = True,
    tolerance: float = 1.0e-12,
) -> FeasibilityEnvelope:
    """Assert that every configured training interval lies in scan output."""

    loaded = load_feasibility_envelope(envelope) if isinstance(envelope, (str, Path)) else envelope
    if not isinstance(loaded, FeasibilityEnvelope):
        raise TypeError("envelope must be a FeasibilityEnvelope or YAML path")
    loaded.assert_sampling_ranges(
        sampling_ranges, require_all=require_all, tolerance=tolerance
    )
    return loaded


@dataclass(frozen=True, slots=True)
class ResetPose:
    gradient: float
    q_ref: tuple[float, ...]
    root_pitch: float
    root_height: float
    q_reset: tuple[float, ...]
    q_ref_unloaded: tuple[float, ...]
    tau_unloaded: tuple[float, ...]
    tau_per_tangent_force: tuple[float, ...]
    tau_per_normal_force: tuple[float, ...]
    tau_per_tangent_difference: tuple[float, ...]
    handle_wrenches_sln: tuple[tuple[float, ...], tuple[float, ...]]
    wheel_contact_forces_sln: tuple[tuple[float, ...], tuple[float, ...]]

    def __post_init__(self) -> None:
        gradient = _finite_float(self.gradient, "pose.gradient")
        if not any(
            math.isclose(gradient, expected, rel_tol=0.0, abs_tol=SLOPE_MATCH_TOLERANCE)
            for expected in RESET_POSE_GRADIENTS
        ):
            raise ConfigurationContractError(f"pose.gradient={gradient} is not a required task slope")
        if isinstance(self.q_ref, (str, bytes)) or not isinstance(self.q_ref, Sequence):
            raise ConfigurationContractError("pose.q_ref must be a sequence")
        if len(self.q_ref) != 29:
            raise ConfigurationContractError(f"pose.q_ref must have length 29, got {len(self.q_ref)}")
        q_ref = tuple(
            _finite_float(value, f"pose.q_ref[{index}]")
            for index, value in enumerate(self.q_ref)
        )
        if self.q_reset is None:
            raise ConfigurationContractError(
                "pose.q_reset is required by the static-reset contract"
            )
        raw_q_reset = self.q_reset
        if isinstance(raw_q_reset, (str, bytes)) or not isinstance(raw_q_reset, Sequence):
            raise ConfigurationContractError("pose.q_reset must be a sequence")
        if len(raw_q_reset) != 29:
            raise ConfigurationContractError(
                f"pose.q_reset must have length 29, got {len(raw_q_reset)}"
            )
        q_reset = tuple(
            _finite_float(value, f"pose.q_reset[{index}]")
            for index, value in enumerate(raw_q_reset)
        )
        raw_q_ref_unloaded = self.q_ref_unloaded
        if raw_q_ref_unloaded is None:
            raise ConfigurationContractError(
                "pose.q_ref_unloaded is required by the static-reset contract"
            )
        if isinstance(raw_q_ref_unloaded, (str, bytes)) or not isinstance(
            raw_q_ref_unloaded, Sequence
        ):
            raise ConfigurationContractError("pose.q_ref_unloaded must be a sequence")
        if len(raw_q_ref_unloaded) != 29:
            raise ConfigurationContractError(
                "pose.q_ref_unloaded must have length 29, got "
                f"{len(raw_q_ref_unloaded)}"
            )
        q_ref_unloaded = tuple(
            _finite_float(value, f"pose.q_ref_unloaded[{index}]")
            for index, value in enumerate(raw_q_ref_unloaded)
        )
        torque_basis: dict[str, tuple[float, ...]] = {}
        for name in (
            "tau_unloaded",
            "tau_per_tangent_force",
            "tau_per_normal_force",
            "tau_per_tangent_difference",
        ):
            raw_value = getattr(self, name)
            if raw_value is None:
                raise ConfigurationContractError(
                    f"pose.{name} is required by the payload-aware static-reset contract"
                )
            if isinstance(raw_value, (str, bytes)) or not isinstance(
                raw_value, Sequence
            ):
                raise ConfigurationContractError(f"pose.{name} must be a sequence")
            if len(raw_value) != 29:
                raise ConfigurationContractError(
                    f"pose.{name} must have length 29, got {len(raw_value)}"
                )
            torque_basis[name] = tuple(
                _finite_float(value, f"pose.{name}[{index}]")
                for index, value in enumerate(raw_value)
            )
        contact_solution: dict[str, tuple[tuple[float, ...], ...]] = {}
        for name, width in (
            ("handle_wrenches_sln", 6),
            ("wheel_contact_forces_sln", 3),
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, (str, bytes)) or not isinstance(
                raw_value, Sequence
            ) or len(raw_value) != 2:
                raise ConfigurationContractError(f"pose.{name} must have shape [2,{width}]")
            rows: list[tuple[float, ...]] = []
            for row_index, row in enumerate(raw_value):
                if isinstance(row, (str, bytes)) or not isinstance(
                    row, Sequence
                ) or len(row) != width:
                    raise ConfigurationContractError(
                        f"pose.{name} must have shape [2,{width}]"
                    )
                rows.append(
                    tuple(
                        _finite_float(value, f"pose.{name}[{row_index}][{index}]")
                        for index, value in enumerate(row)
                    )
                )
            contact_solution[name] = tuple(rows)
        if any(row[2] <= 0.0 for row in contact_solution["wheel_contact_forces_sln"]):
            raise ConfigurationContractError(
                "pose.wheel_contact_forces_sln requires positive normal forces"
            )
        root_pitch = _finite_float(self.root_pitch, "pose.root_pitch")
        if abs(root_pitch) > 0.5:
            raise ConfigurationContractError("pose.root_pitch must lie in [-0.5, 0.5] rad")
        root_height = _finite_float(self.root_height, "pose.root_height")
        if not 0.6 <= root_height <= 0.9:
            raise ConfigurationContractError("pose.root_height must lie in [0.6, 0.9] m")
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "q_ref", q_ref)
        object.__setattr__(self, "q_reset", q_reset)
        object.__setattr__(self, "q_ref_unloaded", q_ref_unloaded)
        for name, value in torque_basis.items():
            object.__setattr__(self, name, value)
        for name, value in contact_solution.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "root_pitch", root_pitch)
        object.__setattr__(self, "root_height", root_height)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "gradient": self.gradient,
            "root_pitch": self.root_pitch,
            "root_height": self.root_height,
            "q_reset": list(self.q_reset),
            "q_ref_unloaded": list(self.q_ref_unloaded),
            "tau_unloaded": list(self.tau_unloaded),
            "tau_per_tangent_force": list(self.tau_per_tangent_force),
            "tau_per_normal_force": list(self.tau_per_normal_force),
            "tau_per_tangent_difference": list(self.tau_per_tangent_difference),
            "handle_wrenches_sln": [list(row) for row in self.handle_wrenches_sln],
            "wheel_contact_forces_sln": [
                list(row) for row in self.wheel_contact_forces_sln
            ],
            "q_ref": list(self.q_ref),
        }


@dataclass(frozen=True, slots=True)
class ResetPoseLibrary:
    poses: tuple[ResetPose, ...]
    joint_order: tuple[str, ...] = G1_JOINT_ORDER
    schema_version: int = RESET_POSE_SCHEMA_VERSION
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RESET_POSE_SCHEMA_VERSION
        ):
            raise ConfigurationContractError(
                f"unsupported reset-pose schema_version={self.schema_version!r}; "
                f"expected {RESET_POSE_SCHEMA_VERSION}"
            )
        joint_order = validate_joint_order(self.joint_order)
        try:
            poses = tuple(
                pose if isinstance(pose, ResetPose) else ResetPose(**pose)
                for pose in self.poses
            )
        except TypeError as exc:
            raise ConfigurationContractError("each pose must contain gradient and q_ref") from exc
        if len(poses) != len(RESET_POSE_GRADIENTS):
            raise ConfigurationContractError(
                f"reset pose library must contain exactly {len(RESET_POSE_GRADIENTS)} poses, "
                f"got {len(poses)}"
            )
        ordered: list[ResetPose] = []
        for expected in RESET_POSE_GRADIENTS:
            matches = [
                index
                for index, pose in enumerate(poses)
                if math.isclose(pose.gradient, expected, rel_tol=0.0, abs_tol=SLOPE_MATCH_TOLERANCE)
            ]
            if len(matches) != 1:
                raise ConfigurationContractError(
                    f"reset pose library requires exactly one pose for gradient {expected}; "
                    f"found {len(matches)}"
                )
            ordered.append(poses[matches[0]])
        object.__setattr__(self, "poses", tuple(ordered))
        object.__setattr__(self, "joint_order", joint_order)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, source_path: str | Path | None = None
    ) -> "ResetPoseLibrary":
        data = _expect_mapping(mapping, "reset pose library")
        _expect_exact_keys(data, {"schema_version", "joint_order", "poses"}, "reset pose library")
        pose_values = data["poses"]
        poses: list[ResetPose] = []
        if isinstance(pose_values, Mapping):
            raise ConfigurationContractError(
                "schema v4 reset poses must use the canonical list form with "
                "explicit static endpoints and torque bases"
            )
        elif isinstance(pose_values, Sequence) and not isinstance(pose_values, (str, bytes)):
            for index, value in enumerate(pose_values):
                pose = _expect_mapping(value, f"poses[{index}]")
                required = {
                    "gradient",
                    "root_pitch",
                    "root_height",
                    "q_reset",
                    "q_ref_unloaded",
                    "tau_unloaded",
                    "tau_per_tangent_force",
                    "tau_per_normal_force",
                    "tau_per_tangent_difference",
                    "handle_wrenches_sln",
                    "wheel_contact_forces_sln",
                    "q_ref",
                }
                missing = required - set(pose)
                unknown = set(pose) - required
                if missing or unknown:
                    details = []
                    if missing:
                        details.append(f"missing={sorted(missing)}")
                    if unknown:
                        details.append(f"unknown={sorted(unknown)}")
                    raise ConfigurationContractError(
                        f"poses[{index}] fields differ: " + ", ".join(details)
                    )
                poses.append(
                    ResetPose(
                        gradient=pose["gradient"],
                        q_ref=pose["q_ref"],
                        root_pitch=pose["root_pitch"],
                        root_height=pose["root_height"],
                        q_reset=pose["q_reset"],
                        q_ref_unloaded=pose["q_ref_unloaded"],
                        tau_unloaded=pose["tau_unloaded"],
                        tau_per_tangent_force=pose["tau_per_tangent_force"],
                        tau_per_normal_force=pose["tau_per_normal_force"],
                        tau_per_tangent_difference=pose["tau_per_tangent_difference"],
                        handle_wrenches_sln=pose["handle_wrenches_sln"],
                        wheel_contact_forces_sln=pose["wheel_contact_forces_sln"],
                    )
                )
        else:
            raise ConfigurationContractError("poses must be a sequence or gradient-to-q_ref mapping")
        return cls(
            schema_version=data["schema_version"],
            joint_order=data["joint_order"],
            poses=tuple(poses),
            source_path=None if source_path is None else Path(source_path),
        )

    def pose_for_gradient(self, gradient: float, *, tolerance: float = SLOPE_MATCH_TOLERANCE) -> ResetPose:
        value = _finite_float(gradient, "gradient")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        matches = [
            pose
            for pose in self.poses
            if math.isclose(pose.gradient, value, rel_tol=0.0, abs_tol=tolerance)
        ]
        if len(matches) != 1:
            raise ConfigurationContractError(
                f"gradient {value} is outside the reset-pose library coverage {RESET_POSE_GRADIENTS}"
            )
        return matches[0]

    def _interpolation_bracket(self, gradient: float) -> tuple[ResetPose, ResetPose, float]:
        value = _finite_float(gradient, "gradient")
        if value < RESET_POSE_GRADIENTS[0] or value > RESET_POSE_GRADIENTS[-1]:
            raise ConfigurationContractError(
                f"gradient {value} is outside [{RESET_POSE_GRADIENTS[0]}, {RESET_POSE_GRADIENTS[-1]}]"
            )
        for pose in self.poses:
            if math.isclose(pose.gradient, value, rel_tol=0.0, abs_tol=SLOPE_MATCH_TOLERANCE):
                return pose, pose, 0.0
        upper_index = next(index for index, pose in enumerate(self.poses) if pose.gradient > value)
        lower = self.poses[upper_index - 1]
        upper = self.poses[upper_index]
        fraction = (value - lower.gradient) / (upper.gradient - lower.gradient)
        return lower, upper, fraction

    def _interpolate_vector(self, gradient: float, field: str) -> tuple[float, ...]:
        lower, upper, fraction = self._interpolation_bracket(gradient)
        lower_values = getattr(lower, field)
        if lower is upper:
            return lower_values
        upper_values = getattr(upper, field)
        return tuple(
            lower_value + fraction * (upper_value - lower_value)
            for lower_value, upper_value in zip(lower_values, upper_values, strict=True)
        )

    def _interpolate_scalar(self, gradient: float, field: str) -> float:
        lower, upper, fraction = self._interpolation_bracket(gradient)
        lower_value = getattr(lower, field)
        if lower is upper:
            return lower_value
        upper_value = getattr(upper, field)
        return lower_value + fraction * (upper_value - lower_value)

    def interpolate_q_ref(self, gradient: float) -> tuple[float, ...]:
        """Linearly interpolate only inside the validated deployment coverage."""

        return self._interpolate_vector(gradient, "q_ref")

    def interpolate_q_reset(self, gradient: float) -> tuple[float, ...]:
        """Interpolate the physical reset state inside the validated slope range."""

        return self._interpolate_vector(gradient, "q_reset")

    def interpolate_q_ref_unloaded(self, gradient: float) -> tuple[float, ...]:
        """Interpolate the zero-handle-load actuator reference."""

        return self._interpolate_vector(gradient, "q_ref_unloaded")

    def interpolate_root_pitch(self, gradient: float) -> float:
        """Interpolate the reset root pitch inside the validated slope range."""

        return self._interpolate_scalar(gradient, "root_pitch")

    def interpolate_root_height(self, gradient: float) -> float:
        """Interpolate the statically preloaded root height inside the slope range."""

        return self._interpolate_scalar(gradient, "root_height")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "joint_order": list(self.joint_order),
            "poses": [pose.to_mapping() for pose in self.poses],
        }


def load_reset_pose_library(path: str | Path) -> ResetPoseLibrary:
    """Load every configured slope IK reference and verify policy joint order."""

    file_path = Path(path)
    return ResetPoseLibrary.from_mapping(_load_yaml_mapping(file_path), source_path=file_path)


def static_preload_hardware_ratios(
    library: ResetPoseLibrary,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return per-slope lower-body and arm reset PD preload ratios."""

    lower_ratios: list[float] = []
    arm_ratios: list[float] = []
    for gradient in RESET_POSE_GRADIENTS:
        pose = library.pose_for_gradient(gradient)
        offsets = tuple(
            abs(reference - reset)
            for reference, reset in zip(pose.q_ref, pose.q_reset, strict=True)
        )
        lower_ratios.append(
            max(
                offset * stiffness / effort
                for offset, stiffness, effort in zip(
                    offsets[:12],
                    RESET_LOWER_STIFFNESS,
                    LOWER_HARDWARE_EFFORT_LIMITS,
                    strict=True,
                )
            )
        )
        arm_ratios.append(
            max(
                offset * stiffness / effort
                for offset, stiffness, effort in zip(
                    offsets[15:],
                    RESET_ARM_STIFFNESS,
                    ARM_HARDWARE_EFFORT_LIMITS,
                    strict=True,
                )
            )
        )
    return tuple(lower_ratios), tuple(arm_ratios)


def static_waist_preload_hardware_ratios(
    library: ResetPoseLibrary,
) -> tuple[float, ...]:
    """Return the per-slope waist reset PD preload hardware ratios."""

    ratios: list[float] = []
    for gradient in RESET_POSE_GRADIENTS:
        pose = library.pose_for_gradient(gradient)
        offsets = tuple(
            abs(reference - reset)
            for reference, reset in zip(
                pose.q_ref[12:15], pose.q_reset[12:15], strict=True
            )
        )
        ratios.append(
            max(
                offset * stiffness / effort
                for offset, stiffness, effort in zip(
                    offsets,
                    RESET_WAIST_STIFFNESS,
                    WAIST_HARDWARE_EFFORT_LIMITS,
                    strict=True,
                )
            )
        )
    return tuple(ratios)


__all__ = [
    "ConfigurationContractError",
    "ConfigurationDependencyError",
    "ARM_HARDWARE_EFFORT_LIMITS",
    "FEASIBILITY_SCHEMA_VERSION",
    "FIXED_G1_JOINT_ORDER",
    "FeasibilityEnvelope",
    "G1_JOINT_ORDER",
    "LOWER_HARDWARE_EFFORT_LIMITS",
    "NumericRange",
    "REQUIRED_CALIBRATION_FIELDS",
    "REQUIRED_FEASIBILITY_RANGES",
    "RESET_POSE_SCHEMA_VERSION",
    "RESET_TORQUE_LIMIT_FRACTION",
    "RESET_ARM_STIFFNESS",
    "RESET_LOWER_STIFFNESS",
    "RESET_WAIST_STIFFNESS",
    "ResetPose",
    "ResetPoseLibrary",
    "SLOPE_GRADIENTS",
    "RESET_POSE_GRADIENTS",
    "assert_sampling_ranges_within_envelope",
    "load_feasibility_envelope",
    "load_reset_pose_library",
    "static_preload_hardware_ratios",
    "static_waist_preload_hardware_ratios",
    "WAIST_HARDWARE_EFFORT_LIMITS",
    "validate_joint_order",
]
