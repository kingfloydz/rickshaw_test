"""Installation script for the G1 rickshaw mjlab task."""

from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="g1_rickshaw_lab",
    version="0.2.0",
    description="MuJoCo/mjlab manager-based G1 rickshaw task",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "mujoco>=3.3.6",
        "mjlab==1.2.0",
        "mujoco-warp==3.5.0",
        "rsl-rl-lib==5.0.1",
        "scipy",
    ],
    extras_require={
        "test": ["pytest", "trimesh"],
    },
    zip_safe=False,
)
