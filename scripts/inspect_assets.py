#!/usr/bin/env python3
"""Validate source assets and their fully composed USD physics ABI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_rickshaw_lab.assets.g1_dex1 import (  # noqa: E402
    COMBINED_DOF_COUNT,
    G1_DEX1_URDF_PATH,
    G1_DEX1_USD_PATH,
    G1_TOTAL_MASS,
    partition_joint_names,
    validate_g1_urdf_inertials,
)
from g1_rickshaw_lab.assets.rickshaw import (  # noqa: E402
    HITCH_JOINT_NAMES,
    RICKSHAW_TOTAL_MASS,
    RICKSHAW_URDF_PATH,
    RICKSHAW_URDF_SPEC,
    RICKSHAW_USD_PATH,
    WHEEL_JOINT_NAMES,
    validate_rickshaw_urdf,
)
from g1_rickshaw_lab.configuration import G1_JOINT_ORDER  # noqa: E402


MASS_TOLERANCE_KG = 2.0e-5
GEOMETRY_TOLERANCE_M = 2.0e-5
INERTIA_ABSOLUTE_TOLERANCE = 2.0e-5
INERTIA_RELATIVE_TOLERANCE = 2.0e-4


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _urdf_nonfixed_joint_names(path: Path) -> tuple[str, ...]:
    root = ET.parse(path).getroot()
    return tuple(
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
    )


def _vector(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(component) for component in value]


def _attr_vector(schema: Any, attribute_name: str) -> list[float]:
    attribute = getattr(schema, f"Get{attribute_name}Attr")()
    return _vector(attribute.Get())


def _walk_prim_specs(prim_spec: Any):
    yield prim_spec
    for child in prim_spec.nameChildren:
        yield from _walk_prim_specs(child)


def _unresolved_local_references(stage: Any) -> list[str]:
    """Return internal reference arcs whose target is absent from its layer."""

    issues: list[str] = []
    for layer in stage.GetUsedLayers():
        for root_prim in layer.rootPrims:
            for prim_spec in _walk_prim_specs(root_prim):
                references = []
                for list_name in (
                    "prependedItems",
                    "explicitItems",
                    "appendedItems",
                ):
                    references.extend(getattr(prim_spec.referenceList, list_name))
                for reference in references:
                    if reference.assetPath or not reference.primPath.pathString:
                        continue
                    if layer.GetPrimAtPath(reference.primPath) is None:
                        issues.append(
                            f"{Path(layer.realPath).name}:{prim_spec.path} -> "
                            f"{reference.primPath}"
                        )
    return sorted(set(issues))


def _inspect_usd(path: Path) -> dict[str, object]:
    """Read physics schemas from the composed stage, including references."""

    try:
        from pxr import Usd, UsdPhysics
    except ModuleNotFoundError:
        return {"available": False, "reason": "pxr is not importable before Kit starts"}

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"USD stage could not be opened: {path}")
    unresolved_local_references = _unresolved_local_references(stage)

    bodies: dict[str, dict[str, object]] = {}
    joints: dict[str, dict[str, object]] = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            mass = api.GetMassAttr().Get()
            if mass is None:
                raise RuntimeError(f"MassAPI has no authored mass: {prim.GetPath()}")
            bodies[prim.GetName()] = {
                "path": prim.GetPath().pathString,
                "mass_kg": float(mass),
                "center_of_mass_m": _attr_vector(api, "CenterOfMass"),
                "diagonal_inertia_kg_m2": _attr_vector(api, "DiagonalInertia"),
            }

        joint_type: str | None = None
        joint_schema: Any | None = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "revolute"
            joint_schema = UsdPhysics.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "prismatic"
            joint_schema = UsdPhysics.PrismaticJoint(prim)
        elif prim.IsA(UsdPhysics.FixedJoint):
            joint_type = "fixed"
            joint_schema = UsdPhysics.FixedJoint(prim)
        elif prim.IsA(UsdPhysics.Joint):
            joint_type = "generic"
            joint_schema = UsdPhysics.Joint(prim)
        if joint_schema is not None:
            info: dict[str, object] = {
                "path": prim.GetPath().pathString,
                "type": joint_type,
                "body0": [target.pathString for target in joint_schema.GetBody0Rel().GetTargets()],
                "body1": [target.pathString for target in joint_schema.GetBody1Rel().GetTargets()],
                "local_pos0_m": _attr_vector(joint_schema, "LocalPos0"),
                "local_pos1_m": _attr_vector(joint_schema, "LocalPos1"),
            }
            if joint_type in {"revolute", "prismatic"}:
                info["axis"] = str(joint_schema.GetAxisAttr().Get())
            joints[prim.GetName()] = info

    return {
        "available": True,
        "unresolved_local_references": unresolved_local_references,
        "body_count": len(bodies),
        "total_mass_kg": sum(float(body["mass_kg"]) for body in bodies.values()),
        "bodies": bodies,
        "joint_count_by_type": {
            kind: sum(joint["type"] == kind for joint in joints.values())
            for kind in ("revolute", "prismatic", "fixed", "generic")
        },
        "joints": joints,
    }


def _parse_xyz(element: ET.Element | None, default: tuple[float, float, float]) -> list[float]:
    if element is None:
        return list(default)
    return [float(value) for value in element.attrib.get("xyz", " ".join(map(str, default))).split()]


def _symmetric_eigenvalues(
    ixx: float,
    ixy: float,
    ixz: float,
    iyy: float,
    iyz: float,
    izz: float,
) -> list[float]:
    # numpy is already part of Isaac Sim, and eigvalsh avoids assuming that a
    # URDF inertia tensor is diagonal in the link frame.
    import numpy as np

    tensor = np.asarray(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
    )
    return sorted(float(value) for value in np.linalg.eigvalsh(tensor))


def _urdf_body_physics(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for link in ET.parse(path).getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            raise RuntimeError(f"{path}: {link.attrib['name']} has no inertial block")
        mass_element = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass_element is None or inertia is None:
            raise RuntimeError(f"{path}: {link.attrib['name']} has incomplete inertial data")
        values = {key: float(inertia.attrib[key]) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        result[link.attrib["name"]] = {
            "mass_kg": float(mass_element.attrib["value"]),
            "center_of_mass_m": _parse_xyz(inertial.find("origin"), (0.0, 0.0, 0.0)),
            "principal_inertia_kg_m2": _symmetric_eigenvalues(**values),
        }
    return result


def _close_sequence(
    actual: list[float],
    expected: list[float],
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(a, e, rel_tol=relative_tolerance, abs_tol=absolute_tolerance)
        for a, e in zip(actual, expected)
    )


def _validate_body_physics(
    label: str,
    usd_info: Mapping[str, object],
    urdf_path: Path,
    expected_total_mass: float,
) -> None:
    expected = _urdf_body_physics(urdf_path)
    bodies = usd_info.get("bodies")
    if not isinstance(bodies, Mapping):
        raise RuntimeError(f"{label} USD contains no composed MassAPI body table")
    if set(bodies) != set(expected):
        missing = sorted(set(expected) - set(bodies))
        extra = sorted(set(bodies) - set(expected))
        raise RuntimeError(f"{label} USD body mismatch: missing={missing}, extra={extra}")

    issues: list[str] = []
    for name, source in expected.items():
        composed = bodies[name]
        if not isinstance(composed, Mapping):
            issues.append(f"{name}: malformed composed body record")
            continue
        mass = float(composed["mass_kg"])
        if not math.isclose(mass, float(source["mass_kg"]), rel_tol=0.0, abs_tol=MASS_TOLERANCE_KG):
            issues.append(f"{name}: mass {mass} != URDF {source['mass_kg']}")
        com = [float(value) for value in composed["center_of_mass_m"]]
        if not _close_sequence(
            com,
            list(source["center_of_mass_m"]),
            absolute_tolerance=GEOMETRY_TOLERANCE_M,
        ):
            issues.append(f"{name}: local CoM {com} != URDF {source['center_of_mass_m']}")
        inertia = sorted(float(value) for value in composed["diagonal_inertia_kg_m2"])
        if any(value <= 0.0 or not math.isfinite(value) for value in inertia):
            issues.append(f"{name}: invalid principal inertia {inertia}")
        elif not _close_sequence(
            inertia,
            list(source["principal_inertia_kg_m2"]),
            absolute_tolerance=INERTIA_ABSOLUTE_TOLERANCE,
            relative_tolerance=INERTIA_RELATIVE_TOLERANCE,
        ):
            issues.append(
                f"{name}: principal inertia {inertia} != URDF "
                f"{source['principal_inertia_kg_m2']}"
            )

    total_mass = float(usd_info.get("total_mass_kg", float("nan")))
    if not math.isclose(total_mass, expected_total_mass, rel_tol=0.0, abs_tol=MASS_TOLERANCE_KG):
        issues.append(f"total mass {total_mass} != expected {expected_total_mass}")
    if issues:
        raise RuntimeError(f"{label} composed USD physics mismatch: " + "; ".join(issues))


def _urdf_joint_types(path: Path) -> dict[str, str]:
    conversion = {"continuous": "revolute", "revolute": "revolute", "prismatic": "prismatic", "fixed": "fixed"}
    return {
        joint.attrib["name"]: conversion[joint.attrib["type"]]
        for joint in ET.parse(path).getroot().findall("joint")
        if joint.attrib.get("type") in conversion
    }


def _validate_joint_types(label: str, usd_info: Mapping[str, object], urdf_path: Path) -> None:
    expected = _urdf_joint_types(urdf_path)
    joints = usd_info.get("joints")
    if not isinstance(joints, Mapping):
        raise RuntimeError(f"{label} USD contains no composed joint table")
    actual = {
        name: joint.get("type")
        for name, joint in joints.items()
        if isinstance(joint, Mapping)
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        raise RuntimeError(
            f"{label} USD joint ABI mismatch: missing={missing}, extra={extra}, wrong_type={wrong}"
        )


def _validate_rickshaw_joint_frames(usd_info: Mapping[str, object]) -> None:
    joints = usd_info["joints"]
    assert isinstance(joints, Mapping)
    spec = RICKSHAW_URDF_SPEC
    expected_positions = {
        WHEEL_JOINT_NAMES[0]: [0.0, spec.wheel_track / 2.0, spec.wheel_radius],
        WHEEL_JOINT_NAMES[1]: [0.0, -spec.wheel_track / 2.0, spec.wheel_radius],
        HITCH_JOINT_NAMES[0]: [spec.hitch_x, spec.hitch_half_width, spec.hitch_z],
        HITCH_JOINT_NAMES[1]: [spec.hitch_x, -spec.hitch_half_width, spec.hitch_z],
    }
    issues: list[str] = []
    for name, expected in expected_positions.items():
        joint = joints.get(name)
        if not isinstance(joint, Mapping):
            issues.append(f"missing joint {name}")
            continue
        local_pos0 = [float(value) for value in joint.get("local_pos0_m", [])]
        local_pos1 = [float(value) for value in joint.get("local_pos1_m", [])]
        if not _close_sequence(local_pos0, expected, absolute_tolerance=GEOMETRY_TOLERANCE_M):
            issues.append(f"{name} localPos0 {local_pos0} != {expected}")
        if not _close_sequence(local_pos1, [0.0, 0.0, 0.0], absolute_tolerance=GEOMETRY_TOLERANCE_M):
            issues.append(f"{name} localPos1 {local_pos1} != [0, 0, 0]")
    for name in WHEEL_JOINT_NAMES:
        joint = joints.get(name)
        if isinstance(joint, Mapping) and joint.get("axis") != "Y":
            issues.append(f"{name} axis {joint.get('axis')} != Y")
    if issues:
        raise RuntimeError("rickshaw composed USD frame mismatch: " + "; ".join(issues))


def _validate_composed_usd(
    g1_info: Mapping[str, object],
    rickshaw_info: Mapping[str, object],
) -> None:
    for label, info in (("G1+Dex1", g1_info), ("rickshaw", rickshaw_info)):
        if not info.get("available"):
            raise RuntimeError(f"{label} composed USD inspection unavailable: {info.get('reason')}")
        unresolved = info.get("unresolved_local_references")
        if not isinstance(unresolved, list) or unresolved:
            raise RuntimeError(
                f"{label} USD has unresolved local references: {unresolved}"
            )
    _validate_body_physics("G1+Dex1", g1_info, G1_DEX1_URDF_PATH, G1_TOTAL_MASS)
    _validate_body_physics("rickshaw", rickshaw_info, RICKSHAW_URDF_PATH, RICKSHAW_TOTAL_MASS)
    _validate_joint_types("G1+Dex1", g1_info, G1_DEX1_URDF_PATH)
    _validate_joint_types("rickshaw", rickshaw_info, RICKSHAW_URDF_PATH)
    _validate_rickshaw_joint_frames(rickshaw_info)


def inspect_assets(*, require_usd_stage: bool = False) -> dict[str, object]:
    missing = [
        str(path)
        for path in (G1_DEX1_URDF_PATH, G1_DEX1_USD_PATH, RICKSHAW_URDF_PATH, RICKSHAW_USD_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing required asset files: " + ", ".join(missing))

    g1_joint_names = _urdf_nonfixed_joint_names(G1_DEX1_URDF_PATH)
    partition = partition_joint_names(g1_joint_names)
    if len(g1_joint_names) != COMBINED_DOF_COUNT:
        raise RuntimeError(f"G1+Dex URDF has {len(g1_joint_names)} non-fixed joints")
    if partition.action_names != G1_JOINT_ORDER:
        raise RuntimeError("G1 policy joint order differs from the fixed checkpoint order")
    g1_inertial_issues = validate_g1_urdf_inertials(G1_DEX1_URDF_PATH)
    if g1_inertial_issues:
        raise RuntimeError("G1 URDF inertial violations: " + "; ".join(g1_inertial_issues))

    rickshaw_issues = validate_rickshaw_urdf(RICKSHAW_URDF_PATH)
    if rickshaw_issues:
        raise RuntimeError("rickshaw URDF violations: " + "; ".join(rickshaw_issues))

    usd = {
        "g1_dex1": _inspect_usd(G1_DEX1_USD_PATH),
        "rickshaw": _inspect_usd(RICKSHAW_USD_PATH),
    }
    if require_usd_stage:
        _validate_composed_usd(usd["g1_dex1"], usd["rickshaw"])

    return {
        "schema_version": 1,
        "tool": "inspect_assets",
        "status": "passed" if require_usd_stage else "static-only",
        "created_utc": _utc_timestamp(),
        "g1_dex1": {
            "urdf": str(G1_DEX1_URDF_PATH),
            "usd": str(G1_DEX1_USD_PATH),
            "nonfixed_joint_count": len(g1_joint_names),
            "policy_joint_count": len(partition.action_names),
            "dex_joint_count": len(partition.dex_names),
            "policy_joint_order": list(partition.action_names),
        },
        "rickshaw": {
            "urdf": str(RICKSHAW_URDF_PATH),
            "usd": str(RICKSHAW_USD_PATH),
        },
        "usd_stage": usd,
    }


def _emit_report(report: Mapping[str, object], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, output)
    print(payload, flush=True)


def _launch_and_emit(app_argv: list[str], output: Path | None) -> None:
    isaaclab_path = Path(
        os.environ.get("ISAACLAB_PATH", REPOSITORY_ROOT.parent / "IsaacLab")
    ).resolve()
    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
        package_path = isaaclab_path / "source" / package_name
        if package_path.is_dir() and str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(app_argv)
    # Inspection is non-interactive. Keeping this default avoids opening a GUI
    # for the exact guide command while still accepting all AppLauncher flags.
    app_args.headless = True
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    try:
        # SimulationApp.close() terminates some Isaac Sim Python launch modes,
        # so persist the result while Kit is still alive.
        _emit_report(inspect_assets(require_usd_stage=True), output)
    finally:
        simulation_app.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Only validate source files; this mode cannot produce a passed USD ABI result.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args, app_argv = parser.parse_known_args()
    if args.static_only:
        _emit_report(inspect_assets(require_usd_stage=False), args.output)
    else:
        _launch_and_emit(app_argv, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
