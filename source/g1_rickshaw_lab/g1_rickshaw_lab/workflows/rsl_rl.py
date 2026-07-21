"""Thin entry points around mjlab's maintained RSL-RL workflows."""

from __future__ import annotations


def train_main() -> None:
    import g1_rickshaw_lab.tasks  # noqa: F401
    from mjlab.scripts.train import main

    main()


def play_main() -> None:
    import g1_rickshaw_lab.tasks  # noqa: F401
    from mjlab.scripts.play import main

    main()


__all__ = ["play_main", "train_main"]
