"""Validate the canonical asset evidence required before training."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from _isaaclab_wrappers import SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_rickshaw_lab.validation import (  # noqa: E402
    asset_hashes,
    validation_input_assets,
)


def validate_training_reset_inputs(
    validation_dir: str | Path,
    *,
    feasibility_path: str | Path,
    reset_pose_path: str | Path,
) -> None:
    """Validate assets only; reset reports are not a training gate."""

    validation_dir = Path(validation_dir).resolve()
    del feasibility_path, reset_pose_path
    assets = asset_hashes(validation_input_assets())
    asset_report_path = validation_dir / "asset_inspection.json"
    if not asset_report_path.is_file():
        raise FileNotFoundError(
            f"training asset inspection report does not exist: {asset_report_path}"
        )

    asset_report = json.loads(asset_report_path.read_text(encoding="utf-8"))
    inspected = asset_report.get("inputs", {}).get("asset_dependencies_sha256")
    if asset_report.get("status") != "passed" or inspected != assets:
        raise RuntimeError("training asset inspection is failed or stale")


__all__ = ["validate_training_reset_inputs"]
