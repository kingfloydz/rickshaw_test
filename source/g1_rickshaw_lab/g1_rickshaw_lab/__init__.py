"""G1 rickshaw velocity-tracking extension.

The policy modules remain importable in a plain PyTorch process. Gym task
registration is activated automatically inside an Isaac Lab environment.
"""

from importlib.util import find_spec


if (
    find_spec("gymnasium") is not None
    and find_spec("isaaclab") is not None
    and find_spec("pxr") is not None
):
    from . import tasks

    __all__ = ["tasks"]
else:
    __all__ = []
