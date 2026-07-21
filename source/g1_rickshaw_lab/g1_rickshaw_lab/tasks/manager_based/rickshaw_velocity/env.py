"""Environment class export for downstream launchers."""

from __future__ import annotations


def get_environment_class():
    from mjlab.envs import ManagerBasedRlEnv

    return ManagerBasedRlEnv


try:
    G1RickshawRLEnv = get_environment_class()
except ModuleNotFoundError:  # Lightweight configuration/unit-test environment.
    G1RickshawRLEnv = None

__all__ = ["G1RickshawRLEnv", "get_environment_class"]
