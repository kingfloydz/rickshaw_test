#!/usr/bin/env python3
"""Convert the pinned G1+Dex and rickshaw URDF assets to USD.

Run this script through Isaac Lab's Python so the URDF importer extensions are
available. Fixed joints are deliberately retained for both assets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_PATH = Path(os.environ.get("ISAACLAB_PATH", ROOT.parent / "IsaacLab")).resolve()
for package_name in ("isaaclab", "isaaclab_assets"):
    package_path = ISAACLAB_PATH / "source" / package_name
    if package_path.is_dir() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="Regenerate USD files even when they already exist.")
parser.add_argument(
    "--asset",
    choices=("all", "g1_dex", "rickshaw"),
    default="all",
    help="Convert only the selected asset family.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Sdf  # noqa: E402


EMPTY_VISUAL_LINKS = {
    "g1_dex": (
        "d435_link",
        "imu_in_pelvis",
        "imu_in_torso",
        "mid360_link",
    ),
    "rickshaw": (
        "left_tow_hitch_link",
        "right_tow_hitch_link",
    ),
}


def _remove_missing_visual_references(
    layer_path: Path,
    root_prim_name: str,
    link_names: tuple[str, ...],
) -> None:
    """Remove importer-created visual arcs only when their target prim is absent."""

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        raise RuntimeError(f"could not open generated USD layer: {layer_path}")
    changed = False
    for link_name in link_names:
        prim_path = Sdf.Path(f"/{root_prim_name}/{link_name}/visuals")
        prim_spec = layer.GetPrimAtPath(prim_path)
        if prim_spec is None:
            raise RuntimeError(f"generated USD is missing visual container {prim_path}")
        references = []
        for list_name in ("prependedItems", "explicitItems", "appendedItems"):
            references.extend(getattr(prim_spec.referenceList, list_name))
        if not references:
            continue
        expected = Sdf.Reference("", Sdf.Path(f"/visuals/{link_name}"))
        if references != [expected]:
            raise RuntimeError(
                f"refusing to alter unexpected visual references at {prim_path}: "
                f"{references}"
            )
        if layer.GetPrimAtPath(expected.primPath) is not None:
            continue
        prim_spec.referenceList.ClearEdits()
        changed = True
    if changed and not layer.Save():
        raise RuntimeError(f"could not save repaired USD layer: {layer_path}")


def convert_g1_dex() -> Path:
    source = ROOT / "assets" / "g1_dex1" / "g1_29dof_mode_15_with_dex1_1.urdf"
    output = ROOT / "assets" / "g1_dex1" / "g1_29dof_mode_15_with_dex1_1.usd"
    if output.is_file() and not args.force:
        result = output
    else:
        cfg = UrdfConverterCfg(
            asset_path=str(source),
            usd_dir=str(output.parent),
            usd_file_name=output.name,
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=False,
            collider_type="convex_hull",
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=100.0, damping=1.0
                ),
            ),
        )
        result = Path(UrdfConverter(cfg).usd_path)
    _remove_missing_visual_references(
        output.parent
        / "configuration"
        / "g1_29dof_mode_15_with_dex1_1_base.usd",
        "g1_29dof_mode_15",
        EMPTY_VISUAL_LINKS["g1_dex"],
    )
    return result


def convert_rickshaw() -> Path:
    source = ROOT / "assets" / "rickshaw" / "rickshaw.urdf"
    output = ROOT / "assets" / "rickshaw" / "rickshaw.usd"
    if output.is_file() and not args.force:
        result = output
    else:
        cfg = UrdfConverterCfg(
            asset_path=str(source),
            usd_dir=str(output.parent),
            usd_file_name=output.name,
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=False,
            collider_type="convex_decomposition",
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=None
                ),
            ),
        )
        result = Path(UrdfConverter(cfg).usd_path)
    _remove_missing_visual_references(
        output.parent / "configuration" / "rickshaw_base.usd",
        "rickshaw",
        EMPTY_VISUAL_LINKS["rickshaw"],
    )
    return result


def main() -> None:
    converters = {
        "g1_dex": (convert_g1_dex,),
        "rickshaw": (convert_rickshaw,),
        "all": (convert_g1_dex, convert_rickshaw),
    }
    for convert in converters[args.asset]:
        path = convert()
        print(f"generated: {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
