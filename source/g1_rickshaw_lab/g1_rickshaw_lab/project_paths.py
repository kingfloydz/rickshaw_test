"""Single source of truth for project data paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "G1_RICKSHAW_PROJECT_ROOT"
_EDITABLE_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get(PROJECT_ROOT_ENV, _EDITABLE_PROJECT_ROOT)).resolve()
ASSET_ROOT = PROJECT_ROOT / "assets"
CONFIG_ROOT = PROJECT_ROOT / "config"


__all__ = ["ASSET_ROOT", "CONFIG_ROOT", "PROJECT_ROOT", "PROJECT_ROOT_ENV"]
