"""Content-addressed reports for optional offline physics diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._hashing import sha256_file as _sha256_file
from .configuration import SLOPE_GRADIENTS
from .rickshaw_spec import RICKSHAW_TOTAL_MASS

VALIDATION_REPORT_SCHEMA_VERSION = 2
FEASIBILITY_MINIMUM_PASS_FRACTION = 0.99
REQUIRED_GATE_TOOLS = ("validate_feasibility", "validate_dynamics")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSET_DEPENDENCY_SUFFIXES = frozenset({".urdf", ".usd", ".stl", ".yaml"})
VALIDATION_SIGNED_SLOPES = SLOPE_GRADIENTS
FEASIBILITY_FORCE_DIRECTIONS = (-1.0, 1.0)
DYNAMICS_CONDITION_SLOPES = {
    "flat_static": 0.0,
    "flat_constant_speed": 0.0,
    "uphill_acceleration": 0.06,
    "downhill_braking": -0.06,
}
MAX_COAST_RELATIVE_TOLERANCE = 0.20
MAX_WRENCH_RELATIVE_TOLERANCE = 0.35
WRENCH_ABSOLUTE_FLOOR_N = 12.0
MIN_DYNAMICS_WINDOW_SAMPLES = 25
MIN_DYNAMICS_SETTLING_STEPS = 25
GUIDE_SCAN_RANGE_ORDER = (
    "payload.mass",
    "payload.com.x",
    "payload.com.y",
    "payload.com.z",
    "rolling_resistance.c_rr",
    "terrain.friction",
    "wheel.left_damping",
    "wheel.right_damping",
    "d6.linear_stiffness",
    "d6.linear_damping",
    "d6.angular_stiffness",
    "d6.angular_damping",
    "d6.max_force",
    "d6.max_torque",
    "d6.linear_limit",
    "d6.angular_limit",
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


class ValidationGateError(RuntimeError):
    """Raised when a validation artifact cannot authorize training."""


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
    """One immutable provenance input referenced by a safety authority."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SafetyThresholdAuthority:
    """Independent, content-addressed source for hardware safety thresholds."""

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
            "sha256": sha256_file(self.source_path),
            "authority_id": self.authority_id,
            "method": self.method,
            "source_files": {
                name: {"path": str(source.path), "sha256": source.sha256}
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


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return _sha256_file(file_path)


def reset_dynamics_feasibility_sha256(path: str | Path) -> str:
    """Hash only immutable feasibility inputs that affect reset dynamics.

    Command acceleration/jerk are authored by the scan itself, while safety
    thresholds are owned by the independent authority.  Excluding both avoids
    a provenance cycle when the completed scan atomically updates its envelope.
    """

    mapping = _load_unique_yaml_mapping(path)
    ranges = mapping.get("ranges")
    calibration = mapping.get("calibration")
    if not isinstance(ranges, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError("feasibility envelope must contain ranges and calibration mappings")
    flattened_calibration = _flatten_value_mapping(calibration)
    projection = {
        "schema_version": mapping.get("schema_version"),
        "slopes": mapping.get("slopes"),
        "joint_order": mapping.get("joint_order"),
        "physical_ranges": {
            name: value
            for name, value in ranges.items()
            if name not in {"command.acceleration_limit", "command.jerk_limit"}
        },
        "reset_dynamics_calibration": {
            name: value
            for name, value in flattened_calibration.items()
            if not name.startswith("safety.")
        },
    }
    encoded = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    Every threshold names its provenance records, and every provenance record
    is hash checked on load.  The candidate/generated envelope is supplied as a
    forbidden source so it cannot authorize its own acceptance thresholds.
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
        _expect_exact_authority_keys(record, {"path", "sha256"}, f"source_files.{name}")
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
        expected_digest = record["sha256"]
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"source_files.{name}.sha256 must be a SHA-256 digest")
        try:
            int(expected_digest, 16)
        except ValueError as exc:
            raise ValueError(f"source_files.{name}.sha256 must be a SHA-256 digest") from exc
        expected_digest = expected_digest.lower()
        actual_digest = sha256_file(source_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"safety authority provenance source {name!r} is stale: "
                f"{actual_digest} != {expected_digest}"
            )
        sources[name] = SafetyAuthoritySource(source_path, expected_digest)

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


def synchronize_runtime_randomization_events(
    env_cfg: Any, runtime_randomization: Any
) -> None:
    """Atomically bind startup/reset event terms to one randomization object."""

    events = getattr(env_cfg, "events", None)
    if events is None:
        raise ValueError("environment configuration has no events object")
    terms: list[tuple[Any, dict[str, Any]]] = []
    for name in ("initialize_curriculum", "sample_physics"):
        term = getattr(events, name, None)
        if term is None:
            raise ValueError(f"environment events are missing {name!r}")
        params = getattr(term, "params", None)
        if not isinstance(params, Mapping):
            raise ValueError(f"environment event {name!r} params must be a mapping")
        rebound = dict(params)
        rebound["cfg"] = runtime_randomization
        terms.append((term, rebound))

    env_cfg.runtime_randomization = runtime_randomization
    for term, params in terms:
        term.params = params


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
        raise ValidationGateError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationGateError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValidationGateError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def asset_hashes(paths: Mapping[str, str | Path]) -> dict[str, str]:
    """Hash named source assets/configs for a report."""

    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def validation_input_assets(repository_root: str | Path = REPOSITORY_ROOT) -> dict[str, Path]:
    """Return the transitive local asset files bound into every gate report."""

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

    soft_d6_heavy = dict(nominal)
    soft_d6_heavy["payload.mass"] = bounds["payload.mass"][1]
    for name in (
        "d6.linear_stiffness",
        "d6.linear_damping",
        "d6.angular_stiffness",
        "d6.angular_damping",
    ):
        soft_d6_heavy[name] = bounds[name][0]
    plan.append(FeasibilityScanPoint("cross:soft_d6_heavy", soft_d6_heavy))
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
    """Construct a gate report whose inputs are content addressed."""

    if tool not in REQUIRED_GATE_TOOLS:
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
            "feasibility_sha256": sha256_file(feasibility_path),
            "reset_pose_path": str(Path(reset_pose_path).resolve()),
            "reset_pose_sha256": sha256_file(reset_pose_path),
            "assets": asset_hashes(assets),
            "runtime_sources": asset_hashes(
                validation_runtime_sources() if runtime_sources is None else runtime_sources
            ),
            "additional_inputs": {
                name: {
                    "path": str(Path(input_path).resolve()),
                    "sha256": sha256_file(input_path),
                }
                for name, input_path in sorted((additional_inputs or {}).items())
            },
        },
        "metrics": dict(metrics),
        "failures": failure_list,
        "metadata": dict(metadata or {}),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationGateError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationGateError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValidationGateError(f"{label} must be a finite number")
    return result


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationGateError(f"{label} must be a JSON array")
    return value


def _close(actual: Any, expected: float, label: str, *, tolerance: float = 1.0e-9) -> None:
    value = _finite_number(actual, label)
    if not math.isclose(value, expected, rel_tol=1.0e-7, abs_tol=tolerance):
        raise ValidationGateError(f"{label}={value} does not match expected {expected}")


def _required_scan_point_names() -> tuple[str, ...]:
    names = ["nominal"]
    for name in GUIDE_SCAN_RANGE_ORDER:
        names.extend((f"{name}:minimum", f"{name}:maximum"))
    names.extend(
        (
            "cross:heavy_high_rr",
            "cross:low_friction_downhill",
            "cross:soft_d6_heavy",
        )
    )
    return tuple(names)


def _parameters_match(
    actual: Mapping[str, float], expected: Mapping[str, float]
) -> bool:
    """Return whether two complete scan assignments are numerically identical."""

    return set(actual) == set(expected) and all(
        math.isclose(actual[name], expected[name], rel_tol=1.0e-7, abs_tol=1.0e-9)
        for name in actual
    )


def _load_json_evidence(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationGateError(f"{label} is not readable JSON: {exc}") from exc
    return _require_mapping(value, label)


def validate_safety_authority_source_evidence(
    authority: SafetyThresholdAuthority,
    *,
    task: str,
    assets_sha256: Mapping[str, str],
    feasibility_path: str | Path,
    reset_pose_path: str | Path,
    reset_pose_sha256: str,
) -> None:
    """Recompute the project-specific claims made by authority source records."""

    reset_source = authority.sources["reset_pose_library"]
    if (
        reset_source.path != Path(reset_pose_path).resolve()
        or reset_source.sha256 != reset_pose_sha256
    ):
        raise ValidationGateError(
            "safety authority reset-pose source is not the report reset-pose library"
        )

    reset_report = _load_json_evidence(
        authority.sources["reset_alignment"].path,
        "safety authority reset_alignment source",
    )
    reset_inputs = _require_mapping(
        reset_report.get("inputs"), "reset_alignment.inputs"
    )
    if set(reset_inputs) != {
        "feasibility_path",
        "feasibility_reset_dynamics_sha256",
        "reset_pose_path",
        "reset_pose_sha256",
        "assets",
        "runtime_sources",
    }:
        raise ValidationGateError(
            "safety authority reset alignment has an incomplete input binding"
        )
    if (
        reset_inputs.get("feasibility_path") != str(Path(feasibility_path).resolve())
        or reset_inputs.get("feasibility_reset_dynamics_sha256")
        != reset_dynamics_feasibility_sha256(feasibility_path)
        or reset_inputs.get("reset_pose_path") != str(Path(reset_pose_path).resolve())
        or reset_inputs.get("reset_pose_sha256") != reset_pose_sha256
    ):
        raise ValidationGateError(
            "safety authority reset alignment is not bound to the current feasibility/reset inputs"
        )
    reset_assets = _require_mapping(reset_inputs.get("assets"), "reset_alignment.inputs.assets")
    if dict(reset_assets) != dict(assets_sha256):
        raise ValidationGateError(
            "safety authority reset alignment is not bound to the current assets"
        )
    reset_runtime_sources = _require_mapping(
        reset_inputs.get("runtime_sources"), "reset_alignment.inputs.runtime_sources"
    )
    current_runtime_sources = asset_hashes(validation_runtime_sources(REPOSITORY_ROOT))
    if dict(reset_runtime_sources) != current_runtime_sources:
        raise ValidationGateError(
            "safety authority reset alignment is stale for the current runtime sources"
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
        or reset_report.get("sample_physics_ranges") is not False
    ):
        raise ValidationGateError(
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
        raise ValidationGateError(
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
        "static_arm_preload_ratio",
    ):
        value = _finite_number(
            reset_thresholds.get(report_name),
            f"reset_alignment.safety_thresholds.{report_name}",
        )
        if not 0.0 < value <= 1.0:
            raise ValidationGateError(
                f"reset_alignment.safety_thresholds.{report_name} must lie in (0,1]"
            )
    checks = _require_mapping(
        summary.get("checks"), "reset_alignment.summary.checks"
    )
    if not checks or any(value is not True for value in checks.values()):
        raise ValidationGateError("safety authority reset alignment contains failed checks")
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
            raise ValidationGateError(
                f"reset alignment {metric_name}={observed} exceeds authority "
                f"{threshold_name}={threshold}"
            )


def _validate_scan_point_parameters(
    point_parameters: Mapping[str, Mapping[str, float]],
    point_names: tuple[str, ...],
    *,
    stage: str,
) -> None:
    """Prove that named feasibility points contain the guide-defined assignments."""

    if set(point_parameters) != set(point_names):
        raise ValidationGateError(f"{stage} parameter evidence does not cover every scan point")

    nominal = point_parameters["nominal"]
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for name in GUIDE_SCAN_RANGE_ORDER:
        minimum = point_parameters[f"{name}:minimum"]
        maximum = point_parameters[f"{name}:maximum"]
        for label, assignment in (("minimum", minimum), ("maximum", maximum)):
            expected = dict(nominal)
            expected[name] = assignment[name]
            if not _parameters_match(assignment, expected):
                raise ValidationGateError(
                    f"{stage} scan point {name}:{label} changes parameters other than {name}"
                )
        lower[name] = minimum[name]
        upper[name] = maximum[name]
        if lower[name] > nominal[name] or nominal[name] > upper[name]:
            raise ValidationGateError(
                f"{stage} scan endpoints for {name} do not bracket the nominal value"
            )

    expected_crosses: dict[str, dict[str, float]] = {}

    heavy_high_rr = dict(nominal)
    for name in ("payload.mass", "payload.com.x", "rolling_resistance.c_rr"):
        heavy_high_rr[name] = upper[name]
    expected_crosses["cross:heavy_high_rr"] = heavy_high_rr

    low_friction_downhill = dict(nominal)
    low_friction_downhill["terrain.friction"] = lower["terrain.friction"]
    expected_crosses["cross:low_friction_downhill"] = low_friction_downhill

    soft_d6_heavy = dict(nominal)
    soft_d6_heavy["payload.mass"] = upper["payload.mass"]
    for name in (
        "d6.linear_stiffness",
        "d6.linear_damping",
        "d6.angular_stiffness",
        "d6.angular_damping",
    ):
        soft_d6_heavy[name] = lower[name]
    expected_crosses["cross:soft_d6_heavy"] = soft_d6_heavy

    for point, expected in expected_crosses.items():
        if not _parameters_match(point_parameters[point], expected):
            raise ValidationGateError(
                f"{stage} scan point {point} does not match the guide-defined cross assignment"
            )


def _validate_search_evidence(
    search: Mapping[str, Any],
    *,
    stage: str,
    candidate_key: str,
    point_names: tuple[str, ...],
    target_acceleration: float | None,
    minimum_wheel_normal_force: float,
) -> ConservativeLimit:
    candidates = tuple(
        _finite_number(value, f"metrics.{stage}_search.{candidate_key}")
        for value in _require_sequence(
            search.get(candidate_key), f"metrics.{stage}_search.{candidate_key}"
        )
    )
    if len(candidates) < 2 or any(value <= 0.0 for value in candidates):
        raise ValidationGateError(f"{stage} search must contain at least two positive candidates")
    if candidates != tuple(sorted(set(candidates))):
        raise ValidationGateError(f"{stage} candidates must be unique and strictly increasing")

    expected_keys = {format(value, ".12g") for value in candidates}
    passed_mapping = _require_mapping(
        search.get("candidate_full_coverage_passed"),
        f"metrics.{stage}_search.candidate_full_coverage_passed",
    )
    if set(passed_mapping) != expected_keys or not all(
        isinstance(value, bool) for value in passed_mapping.values()
    ):
        raise ValidationGateError(f"{stage} candidate pass map is incomplete or malformed")
    candidate_passed = {
        candidate: bool(passed_mapping[format(candidate, ".12g")])
        for candidate in candidates
    }
    try:
        selected = select_conservative_limit(candidate_passed, safety_factor=0.8)
    except (TypeError, ValueError) as exc:
        raise ValidationGateError(f"{stage} search has no valid contiguous feasible prefix: {exc}") from exc

    suffix = "mps2" if stage == "acceleration" else "mps3"
    _close(
        search.get(f"maximum_fully_feasible_{suffix}"),
        selected.maximum_feasible,
        f"metrics.{stage}_search.maximum_fully_feasible_{suffix}",
    )
    _close(
        search.get(f"derived_limit_{suffix}"),
        selected.derived_limit,
        f"metrics.{stage}_search.derived_limit_{suffix}",
    )
    _close(search.get("safety_factor"), 0.8, f"metrics.{stage}_search.safety_factor")

    coverage = _require_mapping(search.get("coverage"), f"metrics.{stage}_search.coverage")
    if set(coverage) != expected_keys:
        raise ValidationGateError(f"{stage} search coverage keys differ from candidates")
    expected_per_candidate = (
        len(point_names) * len(VALIDATION_SIGNED_SLOPES) * len(FEASIBILITY_FORCE_DIRECTIONS)
    )
    slope_labels = {f"{slope:+.2f}" for slope in VALIDATION_SIGNED_SLOPES}
    for candidate in candidates:
        key = format(candidate, ".12g")
        detail = _require_mapping(coverage[key], f"metrics.{stage}_search.coverage.{key}")
        if detail.get("complete") is not True:
            raise ValidationGateError(f"{stage} candidate {key} coverage is incomplete")
        for field in ("expected_trials", "observed_trials"):
            if detail.get(field) != expected_per_candidate:
                raise ValidationGateError(
                    f"{stage} candidate {key} {field} must equal {expected_per_candidate}"
                )
        if detail.get("all_physical_endpoints_passed") is not candidate_passed[candidate]:
            raise ValidationGateError(f"{stage} candidate {key} coverage/pass result disagrees")
        slope_coverage = _require_mapping(
            detail.get("slope_coverage"), f"metrics.{stage}_search.coverage.{key}.slope_coverage"
        )
        if set(slope_coverage) != slope_labels:
            raise ValidationGateError(
                f"{stage} candidate {key} does not cover all "
                f"{len(VALIDATION_SIGNED_SLOPES)} slopes"
            )
        expected_per_slope = len(point_names) * len(FEASIBILITY_FORCE_DIRECTIONS)
        for label, counts in slope_coverage.items():
            counts = _require_mapping(counts, f"{stage}.coverage.{key}.{label}")
            if counts.get("observed") != expected_per_slope:
                raise ValidationGateError(f"{stage} candidate {key} slope {label} quota is incomplete")

    trials = _require_sequence(search.get("trials"), f"metrics.{stage}_search.trials")
    expected_trials = {
        (candidate, point, slope, direction)
        for candidate in candidates
        for point in point_names
        for slope in VALIDATION_SIGNED_SLOPES
        for direction in FEASIBILITY_FORCE_DIRECTIONS
    }
    observed_trials: set[tuple[float, str, float, float]] = set()
    passed_count_by_candidate = {candidate: 0 for candidate in candidates}
    point_parameters: dict[str, dict[str, float]] = {}
    for index, raw_row in enumerate(trials):
        row = _require_mapping(raw_row, f"metrics.{stage}_search.trials[{index}]")
        if row.get("stage") != stage or not isinstance(row.get("passed"), bool):
            raise ValidationGateError(f"{stage} trial {index} has an invalid stage/pass flag")
        key = (
            _finite_number(row.get("candidate"), f"{stage}.trials[{index}].candidate"),
            str(row.get("point")),
            _finite_number(row.get("slope"), f"{stage}.trials[{index}].slope"),
            _finite_number(row.get("force_direction"), f"{stage}.trials[{index}].force_direction"),
        )
        if key not in expected_trials or key in observed_trials:
            raise ValidationGateError(f"{stage} trial coverage contains an unknown or duplicate row: {key}")
        observed_trials.add(key)
        candidate = key[0]
        passed_count_by_candidate[candidate] += int(bool(row["passed"]))

        raw_parameters = _require_mapping(
            row.get("parameters"), f"{stage}.trials[{index}].parameters"
        )
        if set(raw_parameters) != set(GUIDE_SCAN_RANGE_ORDER):
            raise ValidationGateError(
                f"{stage} trial {index} does not contain the complete physical parameter vector"
            )
        parameters = {
            name: _finite_number(
                raw_parameters[name], f"{stage}.trials[{index}].parameters.{name}"
            )
            for name in GUIDE_SCAN_RANGE_ORDER
        }
        point = key[1]
        previous_parameters = point_parameters.setdefault(point, parameters)
        if not _parameters_match(parameters, previous_parameters):
            raise ValidationGateError(
                f"{stage} scan point {point} uses inconsistent parameters across trials"
            )

        evidence = _require_mapping(row.get("dynamic_evidence"), f"{stage}.trials[{index}].dynamic_evidence")
        mass = _finite_number(evidence.get("cart_mass_kg"), f"{stage}.trials[{index}].cart_mass_kg")
        if mass <= 0.0 or evidence.get("force_body") != "base_link":
            raise ValidationGateError(f"{stage} trial {index} lacks a valid physical cart force target")
        if evidence.get("force_api") != (
            "Articulation.permanent_wrench_composer.set_forces_and_torques"
        ):
            raise ValidationGateError(f"{stage} trial {index} did not use the physical force API")
        for group in ("arm", "leg", "waist"):
            limits = _require_mapping(
                evidence.get(f"{group}_actuator_effort_limit_nm"),
                f"{stage}.trials[{index}].{group}_actuator_effort_limit_nm",
            )
            minimum_limit = _finite_number(
                limits.get("minimum"),
                f"{stage}.trials[{index}].{group}_actuator_effort_limit_nm.minimum",
            )
            maximum_limit = _finite_number(
                limits.get("maximum"),
                f"{stage}.trials[{index}].{group}_actuator_effort_limit_nm.maximum",
            )
            if minimum_limit <= 0.0 or maximum_limit < minimum_limit:
                raise ValidationGateError(
                    f"{stage} trial {index} has invalid {group} actuator effort limits"
                )
        target = _finite_number(
            evidence.get("target_equivalent_acceleration_mps2"),
            f"{stage}.trials[{index}].target_acceleration",
        )
        expected_target = candidate if stage == "acceleration" else target_acceleration
        assert expected_target is not None
        if not math.isclose(target, expected_target, rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise ValidationGateError(f"{stage} trial {index} used the wrong acceleration load")
        measured = _require_mapping(
            evidence.get("measured_cart_acceleration_mps2"),
            f"{stage}.trials[{index}].measured_cart_acceleration_mps2",
        )
        if row["passed"] and (
            not isinstance(measured.get("sample_count"), int) or measured["sample_count"] <= 0
        ):
            raise ValidationGateError(f"{stage} passed trial {index} has no measured acceleration")
        row_metrics = _require_mapping(row.get("metrics"), f"{stage}.trials[{index}].metrics")
        try:
            physically_passed, _ = evaluate_feasibility_sample(
                row_metrics,
                minimum_wheel_normal_force=minimum_wheel_normal_force,
            )
        except (TypeError, ValueError) as exc:
            raise ValidationGateError(
                f"{stage} trial {index} metrics are malformed: {exc}"
            ) from exc
        if row["passed"]:
            if row_metrics.get("terminated") is not False:
                raise ValidationGateError(f"{stage} passed trial {index} terminated")
            if not physically_passed:
                raise ValidationGateError(f"{stage} trial {index} claims pass with infeasible metrics")
            if stage == "jerk" and _finite_number(
                row_metrics.get("d6_impulse_ratio"), f"jerk.trials[{index}].d6_impulse_ratio"
            ) > 1.0:
                raise ValidationGateError(f"jerk passed trial {index} exceeded the D6 impulse threshold")
    if observed_trials != expected_trials:
        raise ValidationGateError(
            f"{stage} physical trial grid is incomplete: {len(observed_trials)}/{len(expected_trials)}"
        )
    _validate_scan_point_parameters(point_parameters, point_names, stage=stage)
    passed_by_candidate = {
        candidate: (
            passed_count_by_candidate[candidate] / expected_per_candidate
            >= FEASIBILITY_MINIMUM_PASS_FRACTION
        )
        for candidate in candidates
    }
    if passed_by_candidate != candidate_passed:
        raise ValidationGateError(f"{stage} per-trial results disagree with candidate pass map")
    return selected


def _validate_feasibility_evidence(report: Mapping[str, Any]) -> None:
    metrics = _require_mapping(report.get("metrics"), "metrics")
    if metrics.get("coverage") != "full":
        raise ValidationGateError("passed feasibility report must have full physical coverage")
    point_names = _required_scan_point_names()
    if tuple(_require_sequence(metrics.get("physical_scan_points"), "metrics.physical_scan_points")) != point_names:
        raise ValidationGateError("feasibility report does not contain the exact 36-point physical plan")
    if metrics.get("physical_scan_point_count") != len(point_names):
        raise ValidationGateError("feasibility physical scan point count is incorrect")
    slopes = tuple(float(value) for value in _require_sequence(metrics.get("slopes"), "metrics.slopes"))
    directions = tuple(
        float(value)
        for value in _require_sequence(metrics.get("force_directions"), "metrics.force_directions")
    )
    if slopes != VALIDATION_SIGNED_SLOPES or directions != FEASIBILITY_FORCE_DIRECTIONS:
        raise ValidationGateError(
            f"feasibility report must cover exact {len(VALIDATION_SIGNED_SLOPES)} slopes "
            "and both force directions"
        )
    force_definition = _require_mapping(metrics.get("force_definition"), "metrics.force_definition")
    if (
        force_definition.get("body") != "base_link"
        or force_definition.get("frame") != "world"
        or force_definition.get("direction") != "signed path tangent"
    ):
        raise ValidationGateError("feasibility report did not apply the required physical cart force")

    requirements = _require_mapping(metrics.get("requirements"), "metrics.requirements")
    measurement_sources = _require_mapping(
        metrics.get("measurement_sources"), "metrics.measurement_sources"
    )
    if dict(measurement_sources) != dict(FEASIBILITY_MEASUREMENT_SOURCES):
        raise ValidationGateError(
            "feasibility report does not bind the required physical measurement sources"
        )
    minimum_wheel_normal_force = _finite_number(
        requirements.get("minimum_wheel_normal_force_n"),
        "metrics.requirements.minimum_wheel_normal_force_n",
    )
    if minimum_wheel_normal_force <= 0.0:
        raise ValidationGateError("minimum wheel normal force must be positive")
    for name, expected in (
        ("minimum_zmp_margin_m", 0.02),
        ("maximum_arm_leg_torque_ratio", 0.7),
        ("maximum_waist_torque_ratio", 0.7),
        ("maximum_d6_force_torque_ratio", 0.7),
        ("jerk_maximum_d6_impulse_ratio", 1.0),
        ("minimum_joint_limit_margin_rad", 0.02),
    ):
        _close(requirements.get(name), expected, f"metrics.requirements.{name}")
    d6_impulse_limit = _finite_number(
        requirements.get("d6_impulse_limit"), "metrics.requirements.d6_impulse_limit"
    )
    if d6_impulse_limit <= 0.0:
        raise ValidationGateError("D6 impulse limit must be positive")

    inputs = _require_mapping(report.get("inputs"), "inputs")
    additional = _require_mapping(inputs.get("additional_inputs"), "inputs.additional_inputs")
    candidate_input = _require_mapping(
        additional.get("command_candidate_config"),
        "inputs.additional_inputs.command_candidate_config",
    )
    candidate_record = _require_mapping(
        metrics.get("command_candidate_config"), "metrics.command_candidate_config"
    )
    if (
        candidate_record.get("path") != candidate_input.get("path")
        or candidate_record.get("sha256") != candidate_input.get("sha256")
    ):
        raise ValidationGateError("feasibility candidate configuration binding is inconsistent")

    authority_input = _require_mapping(
        additional.get("safety_threshold_authority"),
        "inputs.additional_inputs.safety_threshold_authority",
    )
    authority_path = authority_input.get("path")
    if not isinstance(authority_path, str):
        raise ValidationGateError("safety threshold authority input has no path")
    try:
        authority = load_safety_threshold_authority(
            authority_path,
            forbidden_source_paths=(str(inputs.get("feasibility_path")),),
        )
        current_authority_sha256 = sha256_file(authority.source_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValidationGateError(f"safety threshold authority is invalid or stale: {exc}") from exc
    if current_authority_sha256 != authority_input.get("sha256"):
        raise ValidationGateError("safety threshold authority input is stale")
    authority_record = _require_mapping(
        metrics.get("safety_threshold_authority"),
        "metrics.safety_threshold_authority",
    )
    if dict(authority_record) != authority.evidence_record():
        raise ValidationGateError(
            "feasibility safety threshold authority binding is inconsistent"
        )
    report_assets = _require_mapping(inputs.get("assets"), "inputs.assets")
    validate_safety_authority_source_evidence(
        authority,
        task=str(report.get("task")),
        assets_sha256={str(name): str(digest) for name, digest in report_assets.items()},
        feasibility_path=str(inputs.get("feasibility_path")),
        reset_pose_path=str(inputs.get("reset_pose_path")),
        reset_pose_sha256=str(inputs.get("reset_pose_sha256")),
    )
    _close(
        minimum_wheel_normal_force,
        float(authority.thresholds["safety.minimum_wheel_normal_force"]),
        "metrics.requirements.minimum_wheel_normal_force_n",
    )
    _close(
        d6_impulse_limit,
        float(authority.thresholds["safety.d6_impulse_limit"]),
        "metrics.requirements.d6_impulse_limit",
    )
    try:
        envelope_mapping = _load_unique_yaml_mapping(str(inputs.get("feasibility_path")))
        envelope_calibration = envelope_mapping.get("calibration")
        if not isinstance(envelope_calibration, Mapping):
            raise ValueError("generated feasibility envelope has no calibration mapping")
        assert_safety_thresholds_match(
            envelope_calibration,
            authority.thresholds,
            label="generated feasibility calibration",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValidationGateError(
            f"generated feasibility envelope is not bound to the safety authority: {exc}"
        ) from exc

    acceleration = _validate_search_evidence(
        _require_mapping(metrics.get("acceleration_search"), "metrics.acceleration_search"),
        stage="acceleration",
        candidate_key="candidates_mps2",
        point_names=point_names,
        target_acceleration=None,
        minimum_wheel_normal_force=minimum_wheel_normal_force,
    )
    jerk_search = _require_mapping(metrics.get("jerk_search"), "metrics.jerk_search")
    _close(
        jerk_search.get("acceleration_held_at_derived_limit_mps2"),
        acceleration.derived_limit,
        "metrics.jerk_search.acceleration_held_at_derived_limit_mps2",
    )
    _close(
        jerk_search.get("d6_impulse_threshold_ratio"),
        1.0,
        "metrics.jerk_search.d6_impulse_threshold_ratio",
    )
    _validate_search_evidence(
        jerk_search,
        stage="jerk",
        candidate_key="candidates_mps3",
        point_names=point_names,
        target_acceleration=acceleration.derived_limit,
        minimum_wheel_normal_force=minimum_wheel_normal_force,
    )
    generated = _require_mapping(metrics.get("generated_envelope"), "metrics.generated_envelope")
    if (
        generated.get("path") != inputs.get("feasibility_path")
        or generated.get("sha256") != inputs.get("feasibility_sha256")
        or generated.get("physical_ranges_preserved") is not True
    ):
        raise ValidationGateError("generated feasibility envelope is not bound to the passed report")


def _validate_wrench_record(
    value: Any, *, label: str, relative_tolerance: float, absolute_floor: float
) -> None:
    record = _require_mapping(value, label)
    analytic = _finite_number(record.get("analytic_mean"), f"{label}.analytic_mean")
    measured = _finite_number(record.get("measured_mean"), f"{label}.measured_mean")
    relative_error = abs(measured - analytic) / max(abs(analytic), absolute_floor)
    _close(record.get("relative_error"), relative_error, f"{label}.relative_error")
    same_sign = (
        abs(analytic) < absolute_floor
        or abs(measured) < absolute_floor
        or math.copysign(1.0, analytic) == math.copysign(1.0, measured)
    )
    if record.get("same_sign") is not same_sign or not same_sign:
        raise ValidationGateError(f"{label} has inconsistent measured/analytic sign")
    if record.get("passed") is not True or relative_error > relative_tolerance:
        raise ValidationGateError(f"{label} exceeds the measured/analytic error limit")


def _validate_dynamics_evidence(report: Mapping[str, Any]) -> None:
    metrics = _require_mapping(report.get("metrics"), "metrics")
    metadata = _require_mapping(report.get("metadata"), "metadata")
    coast_tolerance = _finite_number(
        metadata.get("coast_relative_tolerance"), "metadata.coast_relative_tolerance"
    )
    wrench_tolerance = _finite_number(
        metadata.get("wrench_relative_tolerance"), "metadata.wrench_relative_tolerance"
    )
    absolute_floor = _finite_number(
        metadata.get("wrench_absolute_floor_n"), "metadata.wrench_absolute_floor_n"
    )
    if not 0.0 <= coast_tolerance <= MAX_COAST_RELATIVE_TOLERANCE:
        raise ValidationGateError("dynamics coast tolerance is wider than the normative limit")
    if (
        not 0.0 <= wrench_tolerance <= MAX_WRENCH_RELATIVE_TOLERANCE
        or absolute_floor != WRENCH_ABSOLUTE_FLOOR_N
    ):
        raise ValidationGateError("dynamics wrench tolerance/floor differs from the normative limit")
    settling_steps = metadata.get("settling_steps")
    measurement_steps = metadata.get("measurement_steps")
    window_start = metadata.get("window_start")
    if (
        not isinstance(settling_steps, int)
        or settling_steps < MIN_DYNAMICS_SETTLING_STEPS
        or not isinstance(measurement_steps, int)
        or not isinstance(window_start, int)
        or not 0 <= window_start < measurement_steps
    ):
        raise ValidationGateError("dynamics settling/measurement window is too short or malformed")
    sample_count = measurement_steps - window_start
    if sample_count < MIN_DYNAMICS_WINDOW_SAMPLES:
        raise ValidationGateError("dynamics comparison window has too few samples")

    coast = _require_mapping(metrics.get("coast_down"), "metrics.coast_down")
    if coast.get("sample_count") != sample_count or coast.get("passed") is not True:
        raise ValidationGateError("coast-down evidence is incomplete or did not pass")
    mass = _finite_number(coast.get("mass_kg"), "metrics.coast_down.mass_kg")
    normal_force = _finite_number(
        coast.get("mean_normal_force_n"), "metrics.coast_down.mean_normal_force_n"
    )
    c_rr = _finite_number(coast.get("c_rr"), "metrics.coast_down.c_rr")
    without = _finite_number(
        coast.get("acceleration_without_rr_mps2"),
        "metrics.coast_down.acceleration_without_rr_mps2",
    )
    with_rr = _finite_number(
        coast.get("acceleration_with_rr_mps2"),
        "metrics.coast_down.acceleration_with_rr_mps2",
    )
    if mass <= 0.0 or normal_force <= 0.0 or c_rr <= 0.0:
        raise ValidationGateError("coast-down physical inputs must be positive")
    measured_force = mass * (without - with_rr)
    expected_force = c_rr * normal_force
    relative_error = abs(measured_force - expected_force) / expected_force
    deceleration_delta = without - with_rr
    _close(coast.get("measured_force_n"), measured_force, "metrics.coast_down.measured_force_n")
    _close(coast.get("expected_force_n"), expected_force, "metrics.coast_down.expected_force_n")
    _close(coast.get("relative_error"), relative_error, "metrics.coast_down.relative_error")
    _close(
        coast.get("deceleration_delta_mps2"),
        deceleration_delta,
        "metrics.coast_down.deceleration_delta_mps2",
    )
    if relative_error > coast_tolerance or deceleration_delta < 1.0e-3:
        raise ValidationGateError("coast-down values do not satisfy the recorded tolerance")
    masses = tuple(
        _finite_number(value, "metrics.coast_down.masses_kg")
        for value in _require_sequence(coast.get("masses_kg"), "metrics.coast_down.masses_kg")
    )
    if len(masses) != 2 or any(abs(value - RICKSHAW_TOTAL_MASS) > 0.05 for value in masses):
        raise ValidationGateError("coast-down carts do not have the calibrated PhysX mass")

    forces = _require_mapping(
        metadata.get("controlled_pelvis_force_n"), "metadata.controlled_pelvis_force_n"
    )
    if set(forces) != set(DYNAMICS_CONDITION_SLOPES):
        raise ValidationGateError("dynamics controlled-force conditions are incomplete")
    if (
        _finite_number(forces["flat_static"], "flat_static force") != 0.0
        or _finite_number(forces["flat_constant_speed"], "flat_constant_speed force") != 0.0
        or _finite_number(forces["uphill_acceleration"], "uphill force") <= 0.0
        or _finite_number(forces["downhill_braking"], "downhill force") >= 0.0
    ):
        raise ValidationGateError("dynamics controlled-force signs do not represent the four guide cases")
    if metadata.get("coast_wheel_force_location") != "wheel centers":
        raise ValidationGateError("coast rolling resistance was not applied at wheel centers")
    if metadata.get("coast_normal_force_source") != "level_vehicle_weight":
        raise ValidationGateError("coast-down normal-force source is not the level cart weight")
    if metadata.get("coast_rail_free_axes") != ["transX"]:
        raise ValidationGateError("coast-down rail does not isolate world-X force response")
    if metadata.get("policy_safety_terminations_disabled") is not True:
        raise ValidationGateError(
            "dynamics validation did not isolate prescribed-force tests from policy safety"
        )
    if metadata.get("measured_wrench_source") != "whole_cart_momentum_balance":
        raise ValidationGateError(
            "dynamics interaction wrench was not measured from whole-cart momentum balance"
        )
    if (
        metadata.get("ground_contact_force_source")
        != "two_wheel_contact_sensor_net_forces"
    ):
        raise ValidationGateError(
            "dynamics momentum balance did not isolate wheel-ground contact force"
        )
    if (
        metadata.get("incoming_joint_wrench_role")
        != "constraint_residual_impulse_proxy_only"
    ):
        raise ValidationGateError(
            "incoming joint wrench was not isolated from the physical force gate"
        )

    conditions = _require_mapping(
        metrics.get("d6_analytic_conditions"), "metrics.d6_analytic_conditions"
    )
    if set(conditions) != set(DYNAMICS_CONDITION_SLOPES):
        raise ValidationGateError("dynamics report does not contain the exact four guide conditions")
    for name, expected_slope in DYNAMICS_CONDITION_SLOPES.items():
        condition = _require_mapping(conditions[name], f"metrics.d6_analytic_conditions.{name}")
        _close(condition.get("slope"), expected_slope, f"conditions.{name}.slope")
        if condition.get("analytic_valid_entire_window") is not True or condition.get("terminated") is not False:
            raise ValidationGateError(f"dynamics condition {name} was invalid or terminated")
        _validate_wrench_record(
            condition.get("tangential"),
            label=f"conditions.{name}.tangential",
            relative_tolerance=wrench_tolerance,
            absolute_floor=absolute_floor,
        )
        _validate_wrench_record(
            condition.get("normal"),
            label=f"conditions.{name}.normal",
            relative_tolerance=wrench_tolerance,
            absolute_floor=absolute_floor,
        )


def _validate_passed_report_evidence(report: Mapping[str, Any]) -> None:
    if report["tool"] == "validate_feasibility":
        _validate_feasibility_evidence(report)
    elif report["tool"] == "validate_dynamics":
        _validate_dynamics_evidence(report)


def load_report(path: str | Path, *, expected_tool: str | None = None) -> Mapping[str, Any]:
    """Load and structurally validate one report."""

    report_path = Path(path)
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationGateError(f"missing validation report: {report_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationGateError(f"invalid validation report {report_path}: {exc}") from exc
    report = _require_mapping(value, str(report_path))
    if report.get("schema_version") != VALIDATION_REPORT_SCHEMA_VERSION:
        raise ValidationGateError(
            f"unsupported validation schema in {report_path}: {report.get('schema_version')!r}"
        )
    tool = report.get("tool")
    if tool not in REQUIRED_GATE_TOOLS:
        raise ValidationGateError(f"invalid validation tool in {report_path}: {tool!r}")
    if expected_tool is not None and tool != expected_tool:
        raise ValidationGateError(
            f"{report_path} was produced by {tool!r}, expected {expected_tool!r}"
        )
    if report.get("status") not in {"passed", "failed"}:
        raise ValidationGateError(f"invalid validation status in {report_path}")
    inputs = _require_mapping(report.get("inputs"), f"{report_path}.inputs")
    _require_sha256(inputs.get("feasibility_sha256"), "inputs.feasibility_sha256")
    _require_sha256(inputs.get("reset_pose_sha256"), "inputs.reset_pose_sha256")
    assets = _require_mapping(inputs.get("assets"), "inputs.assets")
    if not assets:
        raise ValidationGateError("inputs.assets cannot be empty")
    for name, digest in assets.items():
        _require_sha256(digest, f"inputs.assets.{name}")
    runtime_sources = _require_mapping(inputs.get("runtime_sources"), "inputs.runtime_sources")
    if not runtime_sources:
        raise ValidationGateError("inputs.runtime_sources cannot be empty")
    for name, digest in runtime_sources.items():
        _require_sha256(digest, f"inputs.runtime_sources.{name}")
    additional_inputs = _require_mapping(
        inputs.get("additional_inputs", {}), "inputs.additional_inputs"
    )
    for name, entry in additional_inputs.items():
        item = _require_mapping(entry, f"inputs.additional_inputs.{name}")
        if set(item) != {"path", "sha256"} or not isinstance(item["path"], str):
            raise ValidationGateError(
                f"inputs.additional_inputs.{name} must contain path and sha256"
            )
        _require_sha256(item["sha256"], f"inputs.additional_inputs.{name}.sha256")
    _require_mapping(report.get("metrics"), f"{report_path}.metrics")
    failures = report.get("failures")
    if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        raise ValidationGateError(f"{report_path}.failures must be a string list")
    if report["status"] == "passed" and failures:
        raise ValidationGateError(f"passed report {report_path} contains failures")
    if report["status"] == "passed":
        try:
            _validate_passed_report_evidence(report)
        except ValidationGateError as exc:
            raise ValidationGateError(f"passed report {report_path} lacks required evidence: {exc}") from exc
    return report


def validate_training_gate(
    validation_dir: str | Path,
    *,
    feasibility_path: str | Path,
    reset_pose_path: str | Path,
    assets: Mapping[str, str | Path],
    task: str | None = None,
    runtime_sources: Mapping[str, str | Path] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Require fresh, mutually consistent feasibility and dynamics reports."""

    directory = Path(validation_dir)
    expected_feasibility = sha256_file(feasibility_path)
    expected_reset = sha256_file(reset_pose_path)
    expected_assets = asset_hashes(assets)
    expected_runtime_sources = asset_hashes(
        validation_runtime_sources() if runtime_sources is None else runtime_sources
    )
    reports: dict[str, Mapping[str, Any]] = {}
    for tool in REQUIRED_GATE_TOOLS:
        path = directory / f"{tool.removeprefix('validate_')}_report.json"
        report = load_report(path, expected_tool=tool)
        if report["status"] != "passed":
            failures = "; ".join(report["failures"][:5]) or "unspecified failure"
            raise ValidationGateError(f"{tool} did not pass: {failures}")
        if task is not None and report.get("task") != task:
            raise ValidationGateError(
                f"{tool} task mismatch: {report.get('task')!r} != {task!r}"
            )
        inputs = report["inputs"]
        if inputs["feasibility_sha256"] != expected_feasibility:
            raise ValidationGateError(f"{tool} report is stale for the feasibility envelope")
        if inputs["reset_pose_sha256"] != expected_reset:
            raise ValidationGateError(f"{tool} report is stale for the reset-pose library")
        if dict(inputs["assets"]) != expected_assets:
            raise ValidationGateError(f"{tool} report is stale for the source assets")
        if dict(inputs["runtime_sources"]) != expected_runtime_sources:
            raise ValidationGateError(f"{tool} report is stale for the task/validator sources")
        for name, entry in inputs.get("additional_inputs", {}).items():
            try:
                current_digest = sha256_file(entry["path"])
            except OSError as exc:
                raise ValidationGateError(
                    f"{tool} report additional input {name!r} is unavailable"
                ) from exc
            if current_digest != entry["sha256"]:
                raise ValidationGateError(
                    f"{tool} report is stale for additional input {name!r}"
                )
        reports[tool] = report
    return reports


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
    "REQUIRED_GATE_TOOLS",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "ValidationGateError",
    "WrenchComparison",
    "asset_hashes",
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
    "reset_dynamics_feasibility_sha256",
    "sha256_file",
    "utc_timestamp",
    "validate_training_gate",
    "validate_safety_authority_source_evidence",
    "validation_input_assets",
    "validation_runtime_sources",
    "write_json_atomic",
    "select_conservative_limit",
    "synchronize_runtime_randomization_events",
    "write_yaml_atomic",
]
