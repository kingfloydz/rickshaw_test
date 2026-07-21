"""Play or evaluate the registered task through mjlab's native CLI."""

import g1_rickshaw_lab.tasks  # noqa: F401
from mjlab.scripts.play import main


if __name__ == "__main__":
    main()
