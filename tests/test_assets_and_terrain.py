"""CPU-only acceptance tests for assets, terrain frames, and reset geometry."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from g1_rickshaw_lab.assets.g1_dex1 import (
    G1_DEX1_URDF_PATH,
    partition_joint_names,
    validate_g1_urdf_inertials,
)
from g1_rickshaw_lab.assets.rickshaw import (
    HITCH_HALF_WIDTH,
    HITCH_X,
    HITCH_Z,
    RICKSHAW_URDF_PATH,
    validate_rickshaw_urdf,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.terrain_cfg import (
    ALL_SIGNED_TERRAIN_GRADIENTS,
    DirectionalPlaneSlopeCfg,
    RickshawPoseTargetCfg,
    cart_root_height_from_pitch,
    directional_plane_slope_geometry,
    hitch_height_from_pitch,
    signed_gradient_from_terrain,
    slope_frame_from_gradient,
    target_pitch_from_hitch_height,
)


EXPECTED_ACTUATED_JOINT_ORDER = (
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
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)

VISUALLESS_REFERENCE_LINKS = {
    G1_DEX1_URDF_PATH: (
        "d435_link",
        "imu_in_pelvis",
        "imu_in_torso",
        "mid360_link",
    ),
    RICKSHAW_URDF_PATH: (
        "left_tow_hitch_link",
        "right_tow_hitch_link",
    ),
}


def test_combined_asset_has_fixed_33_dof_order_and_29_dof_policy_partition() -> None:
    root = ET.parse(G1_DEX1_URDF_PATH).getroot()
    actuated_names = tuple(
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    )

    assert actuated_names == EXPECTED_ACTUATED_JOINT_ORDER
    partition = partition_joint_names(actuated_names)
    assert tuple(
        map(
            len,
            (
                partition.lower_ids,
                partition.waist_ids,
                partition.arm_ids,
                partition.dex_ids,
            ),
        )
    ) == (12, 3, 14, 4)
    assert len(partition.action_ids) == 29
    assert len(partition.all_ids) == 33
    assert len(set(partition.action_ids)) == 29
    assert len(set(partition.all_ids)) == 33
    assert partition.action_names == EXPECTED_ACTUATED_JOINT_ORDER[:29]
    assert partition.all_names == EXPECTED_ACTUATED_JOINT_ORDER


def test_combined_asset_has_explicit_inertials_and_calibrated_total_mass() -> None:
    assert validate_g1_urdf_inertials() == ()


def test_rickshaw_source_urdf_matches_guide_mechanical_contract() -> None:
    assert validate_rickshaw_urdf() == ()


def test_rickshaw_body_uses_dark_red_visual_material() -> None:
    root = ET.parse(RICKSHAW_URDF_PATH).getroot()
    material = root.find("material[@name='rickshaw_body_dark_red']")
    assert material is not None
    color = material.find("color")
    assert color is not None
    assert tuple(float(value) for value in color.attrib["rgba"].split()) == (
        0.18,
        0.004,
        0.008,
        1.0,
    )
    visual_material = root.find("link[@name='base_link']/visual/material")
    assert visual_material is not None
    assert visual_material.attrib["name"] == "rickshaw_body_dark_red"


def test_importer_visual_reference_exceptions_have_no_source_geometry() -> None:
    for urdf_path, link_names in VISUALLESS_REFERENCE_LINKS.items():
        root = ET.parse(urdf_path).getroot()
        for link_name in link_names:
            link = root.find(f"link[@name='{link_name}']")
            assert link is not None
            assert link.find("visual") is None
            assert link.find("collision") is None
            assert link.find("inertial") is not None


def test_asset_inspector_detects_only_missing_local_reference_targets() -> None:
    from scripts.inspect_assets import _unresolved_local_references

    class FakePath:
        def __init__(self, value: str) -> None:
            self.pathString = value

        def __str__(self) -> str:
            return self.pathString

    missing_path = FakePath("/visuals/missing")
    present_path = FakePath("/visuals/present")
    references = [
        SimpleNamespace(assetPath="", primPath=missing_path),
        SimpleNamespace(assetPath="", primPath=present_path),
    ]
    reference_list = SimpleNamespace(
        prependedItems=references,
        explicitItems=[],
        appendedItems=[],
    )
    prim = SimpleNamespace(
        path="/Robot/link/visuals",
        nameChildren=[],
        referenceList=reference_list,
    )
    layer = SimpleNamespace(
        realPath="/tmp/base.usd",
        rootPrims=[prim],
        GetPrimAtPath=lambda path: object() if path is present_path else None,
    )
    stage = SimpleNamespace(GetUsedLayers=lambda: [layer])

    assert _unresolved_local_references(stage) == [
        f"{Path(layer.realPath).name}:{prim.path} -> {missing_path.pathString}"
    ]


def test_asset_inspector_static_mode_cannot_claim_usd_abi_pass() -> None:
    from scripts.inspect_assets import inspect_assets

    report = inspect_assets(require_usd_stage=False)
    assert report["status"] == "static-only"
    assert report["g1_dex1"]["nonfixed_joint_count"] == 33
    assert report["g1_dex1"]["policy_joint_count"] == 29
    dependencies = report["inputs"]["asset_dependencies_sha256"]
    assert "g1_dex1/configuration/g1_29dof_mode_15_with_dex1_1_physics.usd" in dependencies
    assert "rickshaw/configuration/rickshaw_physics.usd" in dependencies


def test_all_21_terrain_gradients_have_exact_frames_and_origins() -> None:
    levels = np.arange(10, dtype=np.int64)
    grid_gradients = np.concatenate(
        (
            np.asarray([signed_gradient_from_terrain(0, 0)]),
            signed_gradient_from_terrain(levels, np.full(10, 9)),
            signed_gradient_from_terrain(levels, np.full(10, 18)),
        )
    )
    np.testing.assert_allclose(
        grid_gradients,
        np.asarray(ALL_SIGNED_TERRAIN_GRADIENTS),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert len(np.unique(grid_gradients)) == 21

    signed_levels = [(0, 0), *((1, level) for level in range(10)), *((-1, level) for level in range(10))]
    for direction, level in signed_levels:
        difficulty = (level + 0.25) / 10.0
        cfg = DirectionalPlaneSlopeCfg(
            size=(26.0, 6.0),
            direction=direction,
            spawn_x=4.0,
        )
        geometry = directional_plane_slope_geometry(difficulty, cfg)
        expected_gradient = direction * (level + 1) * 0.01

        assert math.isclose(geometry.gradient, expected_gradient, rel_tol=0.0, abs_tol=1.0e-15)
        np.testing.assert_array_equal(geometry.origin, np.asarray([4.0, 3.0, 0.0]))
        expected_top_height = expected_gradient * (geometry.vertices[:4, 0] - 4.0)
        np.testing.assert_allclose(
            geometry.vertices[:4, 2], expected_top_height, rtol=0.0, atol=1.0e-15
        )

        frame = slope_frame_from_gradient(expected_gradient)
        normalization = math.sqrt(1.0 + expected_gradient**2)
        expected_tangent = np.asarray([1.0, 0.0, expected_gradient]) / normalization
        expected_normal = np.asarray([-expected_gradient, 0.0, 1.0]) / normalization
        np.testing.assert_allclose(frame.tangent, expected_tangent, rtol=0.0, atol=1.0e-15)
        np.testing.assert_array_equal(frame.lateral, np.asarray([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(frame.normal, expected_normal, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            frame.rotation_matrix.T @ frame.rotation_matrix,
            np.eye(3),
            rtol=0.0,
            atol=2.0e-15,
        )
        assert math.isclose(np.linalg.det(frame.rotation_matrix), 1.0, abs_tol=2.0e-15)
        np.testing.assert_allclose(
            np.cross(frame.tangent, frame.lateral), frame.normal, rtol=0.0, atol=1.0e-15
        )


def test_hitch_height_pitch_and_cart_root_height_round_trip() -> None:
    cfg = RickshawPoseTargetCfg(
        hitch_height_target=0.77,
        hitch_height_tolerance=1.0e-6,
        hitch_vertical_speed_tolerance=1.0e-3,
    )
    alpha = target_pitch_from_hitch_height(cfg)
    root_height = cart_root_height_from_pitch(alpha, cfg)

    assert math.isclose(alpha, 0.3180382172908412, rel_tol=0.0, abs_tol=1.0e-14)
    assert (cfg.hitch_x, cfg.hitch_half_width, cfg.hitch_z) == (
        HITCH_X,
        HITCH_HALF_WIDTH,
        HITCH_Z,
    )
    assert root_height > 0.0
    assert abs(hitch_height_from_pitch(alpha, cfg) - cfg.hitch_height_target) < 1.0e-12

    # Reconstruct H using the reset root offset plus the rotated hitch frame.
    reconstructed_height = (
        root_height + cfg.hitch_x * math.sin(alpha) + cfg.hitch_z * math.cos(alpha)
    )
    assert abs(reconstructed_height - cfg.hitch_height_target) < 1.0e-12
