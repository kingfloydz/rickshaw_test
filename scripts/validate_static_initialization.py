#!/usr/bin/env python3
"""Solve and report all 19 MuJoCo static initialization states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source" / "g1_rickshaw_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS  # noqa: E402
from g1_rickshaw_lab.configuration import (  # noqa: E402
    ARM_HARDWARE_EFFORT_LIMITS,
    LOWER_HARDWARE_EFFORT_LIMITS,
    WAIST_HARDWARE_EFFORT_LIMITS,
)
from g1_rickshaw_lab.static_equilibrium import (  # noqa: E402
    solve_mujoco_static_equilibrium,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.closed_chain import (  # noqa: E402
    build_assembled_spec,
)


def main() -> int:
    model = build_assembled_spec().compile()
    solutions = []
    qpos_seed = None
    for gradient in SLOPE_GRADIENTS:
        solution = solve_mujoco_static_equilibrium(
            model, gradient, qpos_seed=qpos_seed
        )
        solutions.append(solution)
        qpos_seed = solution.qpos
    effort_limit = np.array(
        LOWER_HARDWARE_EFFORT_LIMITS
        + WAIST_HARDWARE_EFFORT_LIMITS
        + ARM_HARDWARE_EFFORT_LIMITS
    )
    torque_ratios = [
        np.abs(solution.joint_actuator_torque) / effort_limit
        for solution in solutions
    ]
    report = {
        "status": "passed",
        "count": len(solutions),
        "solutions": [
            {
                "gradient": solution.gradient,
                "equality_position_error_m": solution.equality_position_error,
                "support_height_error_m": solution.support_height_error,
                "acceleration_error": solution.acceleration_error,
                "actuator_torque_error_nm": solution.actuator_torque_error,
                "fat2_reference_angle_rad": solution.fat2_reference_angle,
                "actuator_torque_ratio_max": float(np.max(torque_ratio)),
            }
            for solution, torque_ratio in zip(solutions, torque_ratios, strict=True)
        ],
        "actuator_torque_ratio_max": float(np.max(torque_ratios)),
    }
    if len(solutions) != len(SLOPE_GRADIENTS):
        raise RuntimeError("static solver did not return all 19 configured slopes")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
