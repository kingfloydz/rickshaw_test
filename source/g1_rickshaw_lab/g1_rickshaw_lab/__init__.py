"""G1 rickshaw MuJoCo/mjlab task package."""

from importlib.util import find_spec

if find_spec("mjlab") is not None:
    from . import tasks

    __all__ = ["tasks"]
else:
    __all__ = []
