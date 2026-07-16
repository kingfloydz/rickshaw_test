"""Validate the canonical asset evidence required before training."""

from __future__ import annotations

import json
from pathlib import Path

from _isaaclab_wrappers import add_project_source_to_path

add_project_source_to_path()

from g1_rickshaw_lab.validation import (  # noqa: E402
    asset_hashes,
    validation_input_assets,
)


def validate_training_assets(validation_dir: str | Path) -> None:
    """Validate the canonical asset report required before training."""

    validation_dir = Path(validation_dir).resolve()
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


__all__ = ["validate_training_assets"]
