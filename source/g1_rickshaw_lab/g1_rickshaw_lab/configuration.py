"""Versioned runtime configuration contracts for the G1 rickshaw task.

The feasibility file is a generated artifact, not a place for fallback
values. Loading it therefore performs complete validation
before returning an object that can be used by training or deployment code.

Canonical ``feasibility_envelope.yaml`` layout::

    schema_version: 2
    slopes: [-0.08, ..., 0.0, ..., 0.10]
    joint_order: [29 exact G1 joint names]
    ranges:
      payload.mass: {min: -3.0, max: 3.0}
      # all names in REQUIRED_FEASIBILITY_RANGES are required
    calibration:
      rickshaw.pitch_inertia_about_axle: 1.0
      # all names in REQUIRED_CALIBRATION_FIELDS are required

Nested mappings are accepted in ``ranges`` and ``calibration`` and are
flattened with dots.  An interval may be written as ``{min: x, max: y}`` or as
``[x, y]``.  The canonical serializer always emits the mapping form.

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
SLOPE_MATCH_TOLERANCE = 1.0e-9

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

# Continuous hardware limits in policy-joint order.
LOWER_HARDWARE_EFFORT_LIMITS = (88.0, 139.0, 88.0, 139.0, 50.0, 50.0) * 2
WAIST_HARDWARE_EFFORT_LIMITS = (88.0, 50.0, 50.0)
ARM_HARDWARE_EFFORT_LIMITS = (25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0) * 2

# Marginal bounds produced by the feasibility scan for runtime domain parameters.
REQUIRED_FEASIBILITY_RANGES = (
    "torso.mass_delta",
    "payload.mass",
    "payload.com.x",
    "payload.com.y",
    "payload.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
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
    "safety.hitch_height_bounds",
    "safety.rickshaw_pitch_bounds",
    "safety.corridor_half_width",
    "safety.heading_error_limit",
    "safety.overspeed_margin",
    "safety.arm_torque_limit",
)

_CALIBRATION_VECTOR_LENGTHS = {
    "fat.com_radius_bounds": 2,
    "safety.hitch_height_bounds": 2,
    "safety.rickshaw_pitch_bounds": 2,
}
_CALIBRATION_STRICTLY_POSITIVE = frozenset(
    {
        "rickshaw.pitch_inertia_about_axle",
        "rickshaw_pose.hitch_height_target",
        "rickshaw_pose.hitch_height_tolerance",
        "rickshaw_pose.hitch_vertical_speed_tolerance",
        "rolling_resistance.c_rr_nominal",
        "terrain.friction_nominal",
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
        "safety.corridor_half_width",
        "safety.heading_error_limit",
        "safety.overspeed_margin",
        "safety.arm_torque_limit",
    }
)

_NONNEGATIVE_RANGE_NAMES = frozenset(
    name
    for name in REQUIRED_FEASIBILITY_RANGES
    if name
    not in {
        "torso.mass_delta",
        "payload.mass",
        "payload.com.x",
        "payload.com.y",
        "payload.com.z",
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
        raise ConfigurationContractError(f"{path}[{index}] is {actual!r}; fixed checkpoint order requires {expected!r}")
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
        return candidate.minimum >= self.minimum - tolerance and candidate.maximum <= self.maximum + tolerance

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
            result.update(_flatten_interval_mapping(child, path=name, allow_scalars=allow_scalars))
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
            raise ConfigurationContractError(f"calibration.{name} must have length {vector_length}, got {len(value)}")
        vector = tuple(
            _finite_float(component, f"calibration.{name}[{index}]") for index, component in enumerate(value)
        )
        if name in {
            "fat.com_radius_bounds",
            "safety.hitch_height_bounds",
            "safety.rickshaw_pitch_bounds",
        }:
            if vector[0] >= vector[1]:
                raise ConfigurationContractError(f"calibration.{name} lower bound must be less than its upper bound")
        validated[name] = vector
    theta_max = validated["safety.theta_max"]
    if theta_max >= math.pi / 2.0:
        raise ConfigurationContractError("calibration.safety.theta_max must lie in (0, pi/2)")
    wrench_tolerance = validated["fat.wrench_consistency_relative_tolerance"]
    if not 0.0 <= wrench_tolerance <= 1.0:
        raise ConfigurationContractError("calibration.fat.wrench_consistency_relative_tolerance must lie in [0,1]")
    radius_min, radius_max = validated["fat.com_radius_bounds"]
    if radius_min <= 0.0 or not radius_min <= validated["fat.com_radius"] <= radius_max:
        raise ConfigurationContractError("calibration.fat.com_radius must lie within positive fat.com_radius_bounds")
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
                f"unsupported feasibility schema_version={self.schema_version!r}; expected {FEASIBILITY_SCHEMA_VERSION}"
            )
        slopes = _validate_required_slopes(self.slopes)
        joint_order = validate_joint_order(self.joint_order)
        parsed_ranges = {
            name: NumericRange.from_value(value, path=f"ranges.{name}") for name, value in self.ranges.items()
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
        ranges = _flatten_interval_mapping(_expect_mapping(data["ranges"], "ranges"), allow_scalars=False)
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
                name: list(value) if isinstance(value, tuple) else value for name, value in self.calibration.items()
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
        parsed = _flatten_interval_mapping(_expect_mapping(sampling_ranges, "training ranges"), allow_scalars=True)
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
            self.ranges[name].assert_contains(candidate, name=f"training_ranges.{name}", tolerance=tolerance)


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

    UniqueKeySafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
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
    loaded.assert_sampling_ranges(sampling_ranges, require_all=require_all, tolerance=tolerance)
    return loaded


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
    "SLOPE_GRADIENTS",
    "assert_sampling_ranges_within_envelope",
    "load_feasibility_envelope",
    "WAIST_HARDWARE_EFFORT_LIMITS",
    "validate_joint_order",
]
