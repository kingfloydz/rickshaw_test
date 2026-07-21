"""Project data paths used by MuJoCo asset loaders."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = PROJECT_ROOT / "assets"
CONFIG_ROOT = PROJECT_ROOT / "config"

__all__ = ["ASSET_ROOT", "CONFIG_ROOT", "PROJECT_ROOT"]
