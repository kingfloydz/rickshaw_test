"""Compile the assembled MuJoCo model and validate its geometry contracts."""

from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh

from g1_rickshaw_lab.assets import (
    RICKSHAW_URDF_SPEC,
    validate_g1_urdf,
    validate_rickshaw_urdf,
)
from g1_rickshaw_lab.assets.rickshaw import TOW_ROD_COLLISION_GEOM_NAMES
from g1_rickshaw_lab.project_paths import ASSET_ROOT
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (
    build_assembled_spec,
    validate_assembled_model,
)


def hitch_mesh_evidence() -> dict[str, object]:
    mesh = trimesh.load_mesh(ASSET_ROOT / "rickshaw" / "body.stl", process=False)
    mesh.vertices *= 1.0e-4
    mesh.vertices[:, 0] -= 0.414504
    mesh.vertices[:, 1] -= 1.94034
    points = np.asarray(RICKSHAW_URDF_SPEC.body_stl_hitch_points)
    closest, distances, triangles = trimesh.proximity.closest_point_naive(mesh, points)
    if not np.all((distances > 0.004) & (distances < 0.012)):
        raise RuntimeError(
            f"hitch points do not pass through the two side rods: {distances.tolist()}"
        )
    if not np.isclose(distances[0], distances[1], atol=1.0e-6):
        raise RuntimeError("left/right hitch points are not symmetric on body.stl")
    return {
        "body_stl_points_m": points.tolist(),
        "closest_surface_points_m": closest.tolist(),
        "distance_to_surface_m": distances.tolist(),
        "triangle_ids": triangles.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mjlab_asset_validation.json")
    args = parser.parse_args()
    issues = [*validate_g1_urdf(), *validate_rickshaw_urdf()]
    model = build_assembled_spec().compile()
    issues.extend(validate_assembled_model(model))
    report = {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "model": {
            "nq": model.nq,
            "nv": model.nv,
            "njnt": model.njnt,
            "ngeom": model.ngeom,
            "neq": model.neq,
        },
        "collision_filter": {
            "g1_rickshaw_pairs": "tow_rods_only",
            "tow_rod_geoms": list(TOW_ROD_COLLISION_GEOM_NAMES),
            "gripper_rickshaw_collision": False,
        },
        "hitches": hitch_mesh_evidence(),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
