"""Installation script for the g1_rickshaw_lab Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_DATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")

setup(
    name="g1_rickshaw_lab",
    version=EXTENSION_DATA["package"]["version"],
    description=EXTENSION_DATA["package"]["description"],
    url=EXTENSION_DATA["package"]["repository"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "PyYAML",
        "rsl-rl-lib==5.0.1",
        "tensordict>=0.6",
    ],
    extras_require={
        "test": ["pytest", "toml"],
    },
    zip_safe=False,
)
