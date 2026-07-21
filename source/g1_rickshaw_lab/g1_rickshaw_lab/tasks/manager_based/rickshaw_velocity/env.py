"""The task uses mjlab's standard ManagerBasedRlEnv directly."""

from __future__ import annotations


def get_environment_class():
    from mjlab.envs import ManagerBasedRlEnv

    return ManagerBasedRlEnv


G1RickshawRLEnv = get_environment_class

__all__ = ["G1RickshawRLEnv", "get_environment_class"]
