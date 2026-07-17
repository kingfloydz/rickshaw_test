"""Structured reports for optional offline physics diagnostics."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .configuration import RESET_TORQUE_LIMIT_FRACTION, SLOPE_GRADIENTS

VALIDATION_REPORT_SCHEMA_VERSION = 3
FEASIBILITY_MINIMUM_PASS_FRACTION = 0.99
VALIDATION_TOOLS = ("validate_feasibility", "validate_dynamics")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSET_DEPENDENCY_SUFFIXES = frozenset({".urdf", ".usd", ".stl", ".yaml"})
VALIDATION_SIGNED_SLOPES = SLOPE_GRADIENTS
MAX_WRENCH_RELATIVE_TOLERANCE = 0.35
WRENCH_ABSOLUTE_FLOOR_N = 12.0
GUIDE_SCAN_RANGE_ORDER = (
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
)
FEASIBILITY_MEASUREMENT_SOURCES = MappingProxyType(
    {
        "wheel_normal_force": (
            "PhysX wheel ContactSensor net force projected onto the slope normal"
        ),
        "foot_friction_cone": (
            "PhysX foot ContactSensor net force resolved in the slope frame"
        ),
        "zmp_margin": (
            "whole-robot CoM dynamics, cart momentum-balance hand force, and measured foot support polygon"
        ),
        "arm_leg_torque_ratio": (
            "robot.data.applied_torque / current actuator.effort_limit"
        ),
        "waist_torque_ratio": (
            "robot.data.applied_torque / current actuator.effort_limit"
        ),
        "d6_force_torque_ratio": (
            "retained-hitch incoming-joint constraint proxy / configured D6 limit"
        ),
        "joint_limit_margin": "q_ref / PhysX hard joint position limits",
    }
)
RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT = MappingProxyType(
    {
        "numerator": "robot.data.applied_torque",
        "denominator": "current actuator.effort_limit",
        "scope": "per-joint arm and lower-body hardware torque ratio",
    }
)
SAFETY_AUTHORITY_SCHEMA_VERSION = 1
SAFETY_THRESHOLD_FIELDS = (
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
)
SAFETY_AUTHORITY_REQUIRED_SOURCES = (
    "implementation_guide",
    "reset_pose_library",
    "reset_alignment",
)
_SAFETY_VECTOR_LENGTHS = {
    "safety.hitch_height_bounds": 2,
    "safety.rickshaw_pitch_bounds": 2,
}


class ValidationReportError(RuntimeError):
    """Raised when a diagnostic report is structurally invalid."""


@dataclass(frozen=True, slots=True)
class FeasibilityScanPoint:
    """One deterministic parameter assignment evaluated on every task slope."""

    name: str
    values: Mapping[str, float]
    required_slope: float | None = None


@dataclass(frozen=True, slots=True)
class ConservativeLimit:
    """Raw fully feasible candidate and its guide-mandated safety-scaled value."""

    maximum_feasible: float
    derived_limit: float


@dataclass(frozen=True, slots=True)
class SafetyAuthoritySource:
    """One provenance input referenced by a safety authority."""

    path: Path


@dataclass(frozen=True, slots=True)
class SafetyThresholdAuthority:
    """Independent source for hardware safety thresholds."""

    authority_id: str
    method: str
    thresholds: Mapping[str, float | tuple[float, ...]]
    threshold_sources: Mapping[str, tuple[str, ...]]
    threshold_rationales: Mapping[str, str]
    sources: Mapping[str, SafetyAuthoritySource]
    source_path: Path
    schema_version: int = SAFETY_AUTHORITY_SCHEMA_VERSION

    def evidence_record(self) -> dict[str, Any]:
        """Return the exact record embedded in feasibility reports."""

        return {
            "path": str(self.source_path),
            "authority_id": self.authority_id,
            "method": self.method,
            "source_files": {
                name: {"path": str(source.path)}
                for name, source in self.sources.items()
            },
            "thresholds": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.thresholds.items()
            },
            "threshold_provenance": {
                name: {
                    "sources": list(self.threshold_sources[name]),
                    "rationale": self.threshold_rationales[name],
                }
                for name in SAFETY_THRESHOLD_FIELDS
            },
        }


def _load_unique_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    """Load one YAML mapping while rejecting duplicate keys at every level."""

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to load safety authority files") from exc

    file_path = Path(path).resolve()

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key {key!r} in {file_path}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {file_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{file_path} must contain a YAML mapping")
    return value


def _expect_exact_authority_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _authority_threshold_value(
    name: str, value: Any
) -> float | tuple[float, ...]:
    vector_length = _SAFETY_VECTOR_LENGTHS.get(name)
    if vector_length is not None:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"thresholds.{name}.value must be a length-{vector_length} sequence")
        if len(value) != vector_length:
            raise ValueError(f"thresholds.{name}.value must have length {vector_length}")
        if any(
            isinstance(component, bool) or not isinstance(component, (int, float))
            for component in value
        ):
            raise ValueError(f"thresholds.{name}.value must contain explicit numbers")
        parsed = tuple(float(component) for component in value)
        if not all(math.isfinite(component) for component in parsed):
            raise ValueError(f"thresholds.{name}.value must contain finite numbers")
        if parsed[0] >= parsed[1]:
            raise ValueError(f"thresholds.{name}.value must be strictly ordered")
        return parsed

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"thresholds.{name}.value must be an explicit number")
    parsed_scalar = float(value)
    if not math.isfinite(parsed_scalar) or parsed_scalar <= 0.0:
        raise ValueError(f"thresholds.{name}.value must be finite and positive")
    if name == "safety.theta_max" and parsed_scalar >= math.pi / 2.0:
        raise ValueError("thresholds.safety.theta_max.value must lie in (0, pi/2)")
    if name == "safety.arm_torque_limit" and parsed_scalar > 1.0:
        raise ValueError("thresholds.safety.arm_torque_limit.value must not exceed 1")
    return parsed_scalar


def load_safety_threshold_authority(
    path: str | Path,
    *,
    forbidden_source_paths: Sequence[str | Path] = (),
) -> SafetyThresholdAuthority:
    """Load and verify an independent safety-threshold authority.

    Every threshold names its provenance records. The candidate/generated
    envelope is forbidden so it cannot authorize its own acceptance thresholds.
    """

    authority_path = Path(path).resolve()
    mapping = _load_unique_yaml_mapping(authority_path)
    _expect_exact_authority_keys(
        mapping,
        {"schema_version", "authority_id", "provenance", "source_files", "thresholds"},
        "safety authority",
    )
    if (
        isinstance(mapping["schema_version"], bool)
        or not isinstance(mapping["schema_version"], int)
        or mapping["schema_version"] != SAFETY_AUTHORITY_SCHEMA_VERSION
    ):
        raise ValueError(
            "safety authority schema_version must be "
            f"{SAFETY_AUTHORITY_SCHEMA_VERSION}"
        )
    authority_id = mapping["authority_id"]
    if not isinstance(authority_id, str) or not authority_id.strip():
        raise ValueError("safety authority authority_id must be a non-empty string")

    provenance = mapping["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("safety authority provenance must be a mapping")
    _expect_exact_authority_keys(provenance, {"method"}, "safety authority provenance")
    method = provenance["method"]
    if not isinstance(method, str) or not method.strip():
        raise ValueError("safety authority provenance.method must be a non-empty string")

    source_files = mapping["source_files"]
    if not isinstance(source_files, Mapping):
        raise ValueError("safety authority source_files must be a mapping")
    _expect_exact_authority_keys(
        source_files,
        set(SAFETY_AUTHORITY_REQUIRED_SOURCES),
        "safety authority source_files",
    )
    forbidden = {Path(item).resolve() for item in forbidden_source_paths}
    forbidden.add(authority_path)
    sources: dict[str, SafetyAuthoritySource] = {}
    for name in SAFETY_AUTHORITY_REQUIRED_SOURCES:
        record = source_files[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"source_files.{name} must be a mapping")
        _expect_exact_authority_keys(record, {"path"}, f"source_files.{name}")
        raw_path = record["path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"source_files.{name}.path must be a non-empty string")
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = authority_path.parent / source_path
        source_path = source_path.resolve()
        if source_path in forbidden:
            raise ValueError(
                f"source_files.{name} points at the authority or feasibility envelope; "
                "an independent source is required"
            )
        if not source_path.is_file():
            raise ValueError(f"safety authority provenance source {name!r} is missing")
        sources[name] = SafetyAuthoritySource(source_path)

    threshold_records = mapping["thresholds"]
    if not isinstance(threshold_records, Mapping):
        raise ValueError("safety authority thresholds must be a mapping")
    _expect_exact_authority_keys(
        threshold_records, set(SAFETY_THRESHOLD_FIELDS), "safety authority thresholds"
    )
    thresholds: dict[str, float | tuple[float, ...]] = {}
    threshold_sources: dict[str, tuple[str, ...]] = {}
    rationales: dict[str, str] = {}
    for name in SAFETY_THRESHOLD_FIELDS:
        record = threshold_records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"thresholds.{name} must be a mapping")
        _expect_exact_authority_keys(
            record, {"value", "sources", "rationale"}, f"thresholds.{name}"
        )
        record_sources = record["sources"]
        if isinstance(record_sources, (str, bytes)) or not isinstance(
            record_sources, Sequence
        ):
            raise ValueError(f"thresholds.{name}.sources must be a non-empty string list")
        parsed_sources = tuple(record_sources)
        if (
            not parsed_sources
            or any(not isinstance(source, str) for source in parsed_sources)
            or len(set(parsed_sources)) != len(parsed_sources)
            or any(source not in sources for source in parsed_sources)
        ):
            raise ValueError(
                f"thresholds.{name}.sources must contain unique source_files names"
            )
        rationale = record["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"thresholds.{name}.rationale must be a non-empty string")
        thresholds[name] = _authority_threshold_value(name, record["value"])
        threshold_sources[name] = parsed_sources
        rationales[name] = rationale

    return SafetyThresholdAuthority(
        authority_id=authority_id,
        method=method,
        thresholds=MappingProxyType(thresholds),
        threshold_sources=MappingProxyType(threshold_sources),
        threshold_rationales=MappingProxyType(rationales),
        sources=MappingProxyType(sources),
        source_path=authority_path,
    )


def _flatten_value_mapping(value: Mapping[str, Any], *, path: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("calibration keys must be non-empty strings")
        name = f"{path}.{key}" if path else key
        nested = _flatten_value_mapping(child, path=name) if isinstance(child, Mapping) else {name: child}
        duplicate = set(result).intersection(nested)
        if duplicate:
            raise ValueError(f"calibration contains duplicate flattened keys: {sorted(duplicate)}")
        result.update(nested)
    return result


def assert_safety_thresholds_match(
    calibration: Mapping[str, Any],
    thresholds: Mapping[str, float | Sequence[float]],
    *,
    label: str = "feasibility calibration",
) -> None:
    """Fail unless an envelope carries exactly the independently authorized values."""

    if set(thresholds) != set(SAFETY_THRESHOLD_FIELDS):
        raise ValueError(f"{label} safety authority does not contain the exact threshold set")
    flattened = _flatten_value_mapping(calibration)
    missing = sorted(set(SAFETY_THRESHOLD_FIELDS) - set(flattened))
    if missing:
        raise ValueError(f"{label} is missing authorized safety thresholds: {missing}")
    for name in SAFETY_THRESHOLD_FIELDS:
        expected = _authority_threshold_value(name, thresholds[name])
        actual = _authority_threshold_value(name, flattened[name])
        expected_values = expected if isinstance(expected, tuple) else (expected,)
        actual_values = actual if isinstance(actual, tuple) else (actual,)
        if len(actual_values) != len(expected_values) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(actual_values, expected_values, strict=True)
        ):
            raise ValueError(
                f"{label}.{name}={actual!r} does not match independent authority "
                f"value {expected!r}"
            )


def utc_timestamp() -> str:
    """Return a stable UTC timestamp suitable for JSON artifacts."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically write a deterministic JSON report."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def write_yaml_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically write a deterministic, human-readable YAML mapping."""

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to write a feasibility envelope") from exc
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def build_positive_candidate_grid(
    interval: Sequence[float], *, count: int = 5, name: str = "candidate"
) -> tuple[float, ...]:
    """Expand a positive search interval into an inclusive deterministic grid."""

    if len(interval) != 2:
        raise ValueError(f"{name} interval must contain two values")
    low, high = float(interval[0]), float(interval[1])
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
        raise ValueError(f"{name} interval must be finite, positive, and ordered")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("candidate count must be a positive integer")
    if low == high:
        return (low,)
    if count < 2:
        raise ValueError("a non-singleton candidate interval requires count >= 2")
    step = (high - low) / float(count - 1)
    return tuple(low + index * step for index in range(count))


def select_conservative_limit(
    candidate_passed: Mapping[float, bool], *, safety_factor: float = 0.8
) -> ConservativeLimit:
    """Select the highest contiguous all-pass candidate and apply a safety factor.

    Feasibility is expected to decrease monotonically as acceleration or jerk is
    increased.  A pass after an earlier failure is retained in diagnostics but
    cannot bridge that failed point when authoring the training envelope.
    """

    if not candidate_passed:
        raise ValueError("candidate results must not be empty")
    if not math.isfinite(safety_factor) or not 0.0 < safety_factor < 1.0:
        raise ValueError("safety_factor must lie in (0, 1)")
    normalized: dict[float, bool] = {}
    for candidate, passed in candidate_passed.items():
        value = float(candidate)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("candidate values must be finite and positive")
        if value in normalized:
            raise ValueError(f"duplicate candidate value {value}")
        if not isinstance(passed, bool):
            raise TypeError("candidate results must be explicit booleans")
        normalized[value] = passed
    maximum: float | None = None
    for candidate in sorted(normalized):
        if not normalized[candidate]:
            break
        maximum = candidate
    if maximum is None:
        raise ValueError("the lowest candidate did not pass full physical coverage")
    return ConservativeLimit(
        maximum_feasible=maximum,
        derived_limit=safety_factor * maximum,
    )


def derive_feasibility_envelope_mapping(
    candidate_mapping: Mapping[str, Any],
    *,
    acceleration_limit: float,
    jerk_limit: float,
    safety_thresholds: Mapping[str, float | Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Replace derived limits and, when supplied, bind independent safety values."""

    acceleration = float(acceleration_limit)
    jerk = float(jerk_limit)
    if not math.isfinite(acceleration) or acceleration <= 0.0:
        raise ValueError("acceleration_limit must be finite and positive")
    if not math.isfinite(jerk) or jerk <= 0.0:
        raise ValueError("jerk_limit must be finite and positive")
    result = copy.deepcopy(dict(candidate_mapping))
    ranges = result.get("ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("candidate envelope must contain a ranges mapping")
    mutable_ranges = dict(ranges)
    for name, value in (
        ("command.acceleration_limit", acceleration),
        ("command.jerk_limit", jerk),
    ):
        if name not in mutable_ranges:
            raise ValueError(f"candidate envelope is missing ranges.{name}")
        mutable_ranges[name] = {"min": value, "max": value}
    result["ranges"] = mutable_ranges
    if safety_thresholds is not None:
        calibration = result.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError("candidate envelope must contain a calibration mapping")
        assert_safety_thresholds_match(
            calibration,
            safety_thresholds,
            label="candidate feasibility calibration",
        )
        mutable_calibration = dict(calibration)
        for name in SAFETY_THRESHOLD_FIELDS:
            if name not in mutable_calibration:
                raise ValueError(
                    "canonical candidate calibration must use dotted safety keys before "
                    f"authoring; missing {name!r}"
                )
            value = _authority_threshold_value(name, safety_thresholds[name])
            mutable_calibration[name] = list(value) if isinstance(value, tuple) else value
        result["calibration"] = mutable_calibration
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationReportError(f"{label} must be a JSON object")
    return value


def _validate_json_finite(value: Any, label: str) -> None:
    """Reject non-finite numbers while preserving arbitrary diagnostic structure."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValidationReportError(f"{label} keys must be strings")
            _validate_json_finite(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_finite(child, f"{label}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationReportError(f"{label} must be finite")
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValidationReportError(f"{label} contains a non-JSON value")


def validation_input_assets(repository_root: str | Path = REPOSITORY_ROOT) -> dict[str, Path]:
    """Return the transitive local asset files referenced by diagnostic reports."""

    root = Path(repository_root).resolve()
    assets_root = root / "assets"
    required_roots = (assets_root / "g1_dex1", assets_root / "rickshaw")
    missing = [path for path in required_roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing asset directories: {missing}")
    result: dict[str, Path] = {}
    for asset_root in required_roots:
        for path in sorted(asset_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in ASSET_DEPENDENCY_SUFFIXES:
                continue
            result[path.relative_to(assets_root).as_posix()] = path
    required_files = (
        "g1_dex1/g1_29dof_mode_15_with_dex1_1.urdf",
        "g1_dex1/g1_29dof_mode_15_with_dex1_1.usd",
        "rickshaw/rickshaw.urdf",
        "rickshaw/rickshaw.usd",
    )
    missing_files = [name for name in required_files if name not in result]
    if missing_files:
        raise FileNotFoundError(f"missing root asset files: {missing_files}")
    return result


def validation_runtime_sources(
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Path]:
    """Return task/validator sources whose changes invalidate physical reports."""

    root = Path(repository_root).resolve()
    package = root / "source" / "g1_rickshaw_lab" / "g1_rickshaw_lab"
    required_files = (
        root / "G1_Rickshaw_IsaacLab_Implementation_Guide.md",
        root / "scripts" / "validate_feasibility.py",
        root / "scripts" / "validate_dynamics.py",
        root / "scripts" / "solve_reset_poses.py",
        root / "scripts" / "_isaaclab_wrappers.py",
        package / "validation.py",
        package / "configuration.py",
        package / "rickshaw_spec.py",
    )
    candidates = list(required_files)
    candidates.extend(sorted((package / "assets").glob("*.py")))
    candidates.extend(
        sorted(
            (package / "tasks" / "manager_based" / "rickshaw_velocity").rglob("*.py")
        )
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation runtime sources: {missing}")
    result = {
        path.relative_to(root).as_posix(): path
        for path in candidates
        if path.is_file()
    }
    return result


def build_feasibility_scan_plan(
    ranges: Mapping[str, Sequence[float]],
) -> tuple[FeasibilityScanPoint, ...]:
    """Build nominal, marginal-endpoint, and coupled feasibility samples."""

    missing = [name for name in GUIDE_SCAN_RANGE_ORDER if name not in ranges]
    if missing:
        raise ValueError(f"scan ranges are missing: {missing}")
    bounds: dict[str, tuple[float, float]] = {}
    for name in GUIDE_SCAN_RANGE_ORDER:
        interval = ranges[name]
        if len(interval) != 2:
            raise ValueError(f"scan range {name!r} must contain two values")
        low, high = float(interval[0]), float(interval[1])
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError(f"scan range {name!r} is not finite and ordered")
        bounds[name] = (low, high)
    nominal = {name: 0.5 * (low + high) for name, (low, high) in bounds.items()}
    plan: list[FeasibilityScanPoint] = [FeasibilityScanPoint("nominal", nominal)]
    for name in GUIDE_SCAN_RANGE_ORDER:
        low, high = bounds[name]
        for endpoint_name, value in (("minimum", low), ("maximum", high)):
            point = dict(nominal)
            point[name] = value
            plan.append(FeasibilityScanPoint(f"{name}:{endpoint_name}", point))

    heavy_high_rr = dict(nominal)
    heavy_high_rr["payload.mass"] = bounds["payload.mass"][1]
    heavy_high_rr["payload.com.x"] = bounds["payload.com.x"][1]
    heavy_high_rr["rolling_resistance.c_rr"] = bounds["rolling_resistance.c_rr"][1]
    plan.append(FeasibilityScanPoint("cross:heavy_high_rr", heavy_high_rr))

    low_friction_downhill = dict(nominal)
    low_friction_downhill["terrain.friction"] = bounds["terrain.friction"][0]
    plan.append(
        FeasibilityScanPoint(
            "cross:low_friction_downhill",
            low_friction_downhill,
            required_slope=-0.06,
        )
    )

    return tuple(plan)


def build_report(
    *,
    tool: str,
    task: str,
    passed: bool,
    feasibility_path: str | Path,
    reset_pose_path: str | Path,
    assets: Mapping[str, str | Path],
    additional_inputs: Mapping[str, str | Path] | None = None,
    runtime_sources: Mapping[str, str | Path] | None = None,
    metrics: Mapping[str, Any],
    failures: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a diagnostic report with explicit input paths."""

    if tool not in VALIDATION_TOOLS:
        raise ValueError(f"unsupported validation tool {tool!r}")
    failure_list = [str(item) for item in failures]
    if passed and failure_list:
        raise ValueError("a passed report cannot contain failures")
    return {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "tool": tool,
        "status": "passed" if passed else "failed",
        "task": str(task),
        "created_utc": utc_timestamp(),
        "inputs": {
            "feasibility_path": str(Path(feasibility_path).resolve()),
            "reset_pose_path": str(Path(reset_pose_path).resolve()),
            "assets": {
                name: str(Path(input_path).resolve())
                for name, input_path in sorted(assets.items())
            },
            "runtime_sources": {
                name: str(Path(input_path).resolve())
                for name, input_path in sorted(
                    (validation_runtime_sources() if runtime_sources is None else runtime_sources).items()
                )
            },
            "additional_inputs": {
                name: str(Path(input_path).resolve())
                for name, input_path in sorted((additional_inputs or {}).items())
            },
        },
        "metrics": dict(metrics),
        "failures": failure_list,
        "metadata": dict(metadata or {}),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationReportError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationReportError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValidationReportError(f"{label} must be a finite number")
    return result


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationReportError(f"{label} must be a JSON array")
    return value


def _close(actual: Any, expected: float, label: str, *, tolerance: float = 1.0e-9) -> None:
    value = _finite_number(actual, label)
    if not math.isclose(value, expected, rel_tol=1.0e-7, abs_tol=tolerance):
        raise ValidationReportError(f"{label}={value} does not match expected {expected}")


def _load_json_evidence(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationReportError(f"{label} is not readable JSON: {exc}") from exc
    return _require_mapping(value, label)


def validate_safety_authority_source_evidence(
    authority: SafetyThresholdAuthority,
    *,
    task: str,
    feasibility_path: str | Path,
    reset_pose_path: str | Path,
) -> None:
    """Recompute the project-specific claims made by authority source records."""

    reset_source = authority.sources["reset_pose_library"]
    if reset_source.path != Path(reset_pose_path).resolve():
        raise ValidationReportError(
            "safety authority reset-pose source is not the report reset-pose library"
        )

    reset_report = _load_json_evidence(
        authority.sources["reset_alignment"].path,
        "safety authority reset_alignment source",
    )
    reset_inputs = _require_mapping(
        reset_report.get("inputs"), "reset_alignment.inputs"
    )
    if (
        reset_inputs.get("feasibility_path") != str(Path(feasibility_path).resolve())
        or reset_inputs.get("reset_pose_path") != str(Path(reset_pose_path).resolve())
    ):
        raise ValidationReportError(
            "safety authority reset alignment references different feasibility/reset inputs"
        )
    reset_slopes = tuple(
        _finite_number(value, "reset_alignment.slopes")
        for value in _require_sequence(
            reset_report.get("slopes"), "reset_alignment.slopes"
        )
    )
    if (
        reset_report.get("schema_version") != 2
        or reset_report.get("tool") != "validate_reset_alignment"
        or reset_report.get("status") != "passed"
        or reset_report.get("task") != task
        or not isinstance(reset_report.get("steps"), int)
        or isinstance(reset_report.get("steps"), bool)
        or reset_report["steps"] < 1000
        or reset_slopes != VALIDATION_SIGNED_SLOPES
        or reset_report.get("continuous_standing") is not True
        or reset_report.get("physics_mode") != "fixed"
    ):
        raise ValidationReportError(
            "safety authority reset alignment must cover this task, nominal 1000-step "
            f"continuous standing, and all {len(VALIDATION_SIGNED_SLOPES)} slopes"
        )
    summary = _require_mapping(
        reset_report.get("summary"), "reset_alignment.summary"
    )
    torque_contract = _require_mapping(
        reset_report.get("torque_measurement_contract"),
        "reset_alignment.torque_measurement_contract",
    )
    if dict(torque_contract) != dict(RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT):
        raise ValidationReportError(
            "safety authority reset alignment does not use physical actuator torque limits"
        )
    reset_thresholds = _require_mapping(
        reset_report.get("safety_thresholds"), "reset_alignment.safety_thresholds"
    )
    for report_name, authority_name in (
        ("arm_torque_ratio", "safety.arm_torque_limit"),
        ("d6_residual_m_or_rad", "safety.d6_residual_limit"),
        ("d6_impulse_n_s", "safety.d6_impulse_limit"),
    ):
        _close(
            reset_thresholds.get(report_name),
            authority.thresholds[authority_name],
            f"reset_alignment.safety_thresholds.{report_name}",
        )
    for report_name in (
        "static_lower_preload_ratio",
        "static_waist_preload_ratio",
        "static_arm_preload_ratio",
    ):
        _close(
            reset_thresholds.get(report_name),
            RESET_TORQUE_LIMIT_FRACTION,
            f"reset_alignment.safety_thresholds.{report_name}",
        )
    checks = _require_mapping(
        summary.get("checks"), "reset_alignment.summary.checks"
    )
    if not checks or any(value is not True for value in checks.values()):
        raise ValidationReportError("safety authority reset alignment contains failed checks")
    observed_limits = (
        (
            "rollout_d6_residual_max_m_or_rad",
            "safety.d6_residual_limit",
        ),
        ("rollout_d6_impulse_max_n_s", "safety.d6_impulse_limit"),
        (
            "static_preload_arm_hardware_ratio_max",
            "safety.arm_torque_limit",
        ),
        (
            "continuous_standing_abs_torso_pitch_max_rad",
            "safety.theta_max",
        ),
    )
    for metric_name, threshold_name in observed_limits:
        observed = _finite_number(
            summary.get(metric_name), f"reset_alignment.summary.{metric_name}"
        )
        threshold = float(authority.thresholds[threshold_name])
        if observed < 0.0 or observed > threshold:
            raise ValidationReportError(
                f"reset alignment {metric_name}={observed} exceeds authority "
                f"{threshold_name}={threshold}"
            )


def load_report(path: str | Path, *, expected_tool: str | None = None) -> Mapping[str, Any]:
    """Load and structurally validate one report."""

    report_path = Path(path)
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationReportError(f"missing validation report: {report_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationReportError(f"invalid validation report {report_path}: {exc}") from exc
    report = _require_mapping(value, str(report_path))
    if report.get("schema_version") != VALIDATION_REPORT_SCHEMA_VERSION:
        raise ValidationReportError(
            f"unsupported validation schema in {report_path}: {report.get('schema_version')!r}"
        )
    tool = report.get("tool")
    if tool not in VALIDATION_TOOLS:
        raise ValidationReportError(f"invalid validation tool in {report_path}: {tool!r}")
    if expected_tool is not None and tool != expected_tool:
        raise ValidationReportError(
            f"{report_path} was produced by {tool!r}, expected {expected_tool!r}"
        )
    if report.get("status") not in {"passed", "failed"}:
        raise ValidationReportError(f"invalid validation status in {report_path}")
    inputs = _require_mapping(report.get("inputs"), f"{report_path}.inputs")
    for name in ("feasibility_path", "reset_pose_path"):
        if not isinstance(inputs.get(name), str) or not inputs[name]:
            raise ValidationReportError(f"inputs.{name} must be a non-empty path")
    assets = _require_mapping(inputs.get("assets"), "inputs.assets")
    if not assets:
        raise ValidationReportError("inputs.assets cannot be empty")
    if not all(isinstance(name, str) and isinstance(value, str) for name, value in assets.items()):
        raise ValidationReportError("inputs.assets must map names to paths")
    runtime_sources = _require_mapping(inputs.get("runtime_sources"), "inputs.runtime_sources")
    if not runtime_sources:
        raise ValidationReportError("inputs.runtime_sources cannot be empty")
    if not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in runtime_sources.items()
    ):
        raise ValidationReportError("inputs.runtime_sources must map names to paths")
    additional_inputs = _require_mapping(
        inputs.get("additional_inputs", {}), "inputs.additional_inputs"
    )
    if not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in additional_inputs.items()
    ):
        raise ValidationReportError("inputs.additional_inputs must map names to paths")
    metrics = _require_mapping(report.get("metrics"), f"{report_path}.metrics")
    metadata = _require_mapping(report.get("metadata"), f"{report_path}.metadata")
    _validate_json_finite(metrics, f"{report_path}.metrics")
    _validate_json_finite(metadata, f"{report_path}.metadata")
    failures = report.get("failures")
    if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        raise ValidationReportError(f"{report_path}.failures must be a string list")
    if report["status"] == "passed" and failures:
        raise ValidationReportError(f"passed report {report_path} contains failures")
    return report


@dataclass(frozen=True, slots=True)
class CoastDownResult:
    measured_force_n: float
    expected_force_n: float
    relative_error: float
    deceleration_delta_mps2: float
    passed: bool


def evaluate_coast_down(
    *,
    mass_kg: float,
    mean_normal_force_n: float,
    c_rr: float,
    acceleration_without_rr_mps2: float,
    acceleration_with_rr_mps2: float,
    relative_tolerance: float,
    minimum_deceleration_delta_mps2: float = 1.0e-3,
) -> CoastDownResult:
    """Evaluate the guide's physical rolling-resistance coast-down contract."""

    values = (
        mass_kg,
        mean_normal_force_n,
        c_rr,
        acceleration_without_rr_mps2,
        acceleration_with_rr_mps2,
        relative_tolerance,
        minimum_deceleration_delta_mps2,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coast-down inputs must be finite")
    if mass_kg <= 0.0 or mean_normal_force_n <= 0.0 or c_rr <= 0.0:
        raise ValueError("mass, normal force, and c_rr must be positive")
    if relative_tolerance < 0.0 or minimum_deceleration_delta_mps2 <= 0.0:
        raise ValueError("coast-down tolerances are invalid")
    measured_force = mass_kg * (
        acceleration_without_rr_mps2 - acceleration_with_rr_mps2
    )
    expected_force = c_rr * mean_normal_force_n
    relative_error = abs(measured_force - expected_force) / expected_force
    deceleration_delta = acceleration_without_rr_mps2 - acceleration_with_rr_mps2
    passed = (
        deceleration_delta >= minimum_deceleration_delta_mps2
        and relative_error <= relative_tolerance
    )
    return CoastDownResult(
        measured_force_n=measured_force,
        expected_force_n=expected_force,
        relative_error=relative_error,
        deceleration_delta_mps2=deceleration_delta,
        passed=passed,
    )


@dataclass(frozen=True, slots=True)
class WrenchComparison:
    analytic_mean: float
    measured_mean: float
    relative_error: float
    same_sign: bool
    passed: bool


def compare_wrench_component(
    analytic_samples: Sequence[float],
    measured_samples: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_floor: float = 1.0,
) -> WrenchComparison:
    """Compare one window-averaged analytic/measured wrench component."""

    if len(analytic_samples) == 0 or len(analytic_samples) != len(measured_samples):
        raise ValueError("wrench windows must be non-empty and have equal length")
    analytic = [float(value) for value in analytic_samples]
    measured = [float(value) for value in measured_samples]
    if not all(math.isfinite(value) for value in (*analytic, *measured)):
        raise ValueError("wrench samples must be finite")
    if relative_tolerance < 0.0 or absolute_floor <= 0.0:
        raise ValueError("wrench comparison tolerances are invalid")
    analytic_mean = sum(analytic) / len(analytic)
    measured_mean = sum(measured) / len(measured)
    denominator = max(abs(analytic_mean), absolute_floor)
    relative_error = abs(measured_mean - analytic_mean) / denominator
    same_sign = (
        abs(analytic_mean) < absolute_floor
        or abs(measured_mean) < absolute_floor
        or math.copysign(1.0, analytic_mean) == math.copysign(1.0, measured_mean)
    )
    return WrenchComparison(
        analytic_mean=analytic_mean,
        measured_mean=measured_mean,
        relative_error=relative_error,
        same_sign=same_sign,
        passed=same_sign and relative_error <= relative_tolerance,
    )


def evaluate_feasibility_sample(
    metrics: Mapping[str, float | bool],
    *,
    minimum_wheel_normal_force: float,
    minimum_zmp_margin: float = 0.02,
    maximum_torque_ratio: float = 1.0,
    maximum_d6_ratio: float = 0.7,
    minimum_joint_limit_margin: float = 0.02,
) -> tuple[bool, list[str]]:
    """Evaluate one scan sample against every guide section 11.4 constraint."""

    required = {
        "left_wheel_normal_force",
        "right_wheel_normal_force",
        "friction_cone_margin",
        "zmp_margin",
        "arm_torque_ratio",
        "leg_torque_ratio",
        "waist_torque_ratio",
        "d6_force_ratio",
        "d6_torque_ratio",
        "joint_limit_margin",
        "finite",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"feasibility sample is missing metrics: {missing}")
    failures: list[str] = []
    numeric_names = required - {"finite"}
    for name in numeric_names:
        value = float(metrics[name])
        if not math.isfinite(value):
            failures.append(f"{name} is not finite")
    if not bool(metrics["finite"]):
        failures.append("runtime state contains NaN/Inf")
    if float(metrics["left_wheel_normal_force"]) < minimum_wheel_normal_force:
        failures.append("left wheel normal force below lift margin")
    if float(metrics["right_wheel_normal_force"]) < minimum_wheel_normal_force:
        failures.append("right wheel normal force below lift margin")
    if float(metrics["friction_cone_margin"]) < 0.0:
        failures.append("foot friction cone infeasible")
    if float(metrics["zmp_margin"]) < minimum_zmp_margin:
        failures.append("ZMP margin below 0.02 m")
    if (
        max(
            float(metrics["arm_torque_ratio"]),
            float(metrics["leg_torque_ratio"]),
            float(metrics["waist_torque_ratio"]),
        )
        > maximum_torque_ratio
    ):
        failures.append("arm/leg/waist torque exceeds the actuator hardware limit")
    if max(float(metrics["d6_force_ratio"]), float(metrics["d6_torque_ratio"])) > maximum_d6_ratio:
        failures.append("D6 wrench exceeds 0.7 configured limit")
    if float(metrics["joint_limit_margin"]) < minimum_joint_limit_margin:
        failures.append("q_ref joint-limit margin below threshold")
    return not failures, failures


__all__ = [
    "CoastDownResult",
    "ASSET_DEPENDENCY_SUFFIXES",
    "ConservativeLimit",
    "FeasibilityScanPoint",
    "FEASIBILITY_MEASUREMENT_SOURCES",
    "FEASIBILITY_MINIMUM_PASS_FRACTION",
    "GUIDE_SCAN_RANGE_ORDER",
    "RESET_ALIGNMENT_TORQUE_MEASUREMENT_CONTRACT",
    "SAFETY_AUTHORITY_REQUIRED_SOURCES",
    "SAFETY_AUTHORITY_SCHEMA_VERSION",
    "SAFETY_THRESHOLD_FIELDS",
    "SafetyAuthoritySource",
    "SafetyThresholdAuthority",
    "VALIDATION_SIGNED_SLOPES",
    "VALIDATION_TOOLS",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "ValidationReportError",
    "WrenchComparison",
    "assert_safety_thresholds_match",
    "build_report",
    "build_feasibility_scan_plan",
    "build_positive_candidate_grid",
    "compare_wrench_component",
    "evaluate_coast_down",
    "evaluate_feasibility_sample",
    "derive_feasibility_envelope_mapping",
    "load_safety_threshold_authority",
    "load_report",
    "utc_timestamp",
    "validate_safety_authority_source_evidence",
    "validation_input_assets",
    "validation_runtime_sources",
    "write_json_atomic",
    "select_conservative_limit",
    "write_yaml_atomic",
]
