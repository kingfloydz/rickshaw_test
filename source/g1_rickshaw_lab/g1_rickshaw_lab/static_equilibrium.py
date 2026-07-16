"""Fixed-contact statics shared by reset generation and runtime loading.

The hand wrench convention is robot-on-cart.  Wheel contact forces are
ground-on-cart.  Components use the slope frame ``(s, l, n)`` where ``s`` is
the path tangent, ``l`` is lateral (left), and ``n`` is the terrain normal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class FixedContactStaticSolution:
    """Scalar fixed-contact solution for two hitches and two passive wheels."""

    handle_wrenches_sln: tuple[tuple[float, ...], tuple[float, ...]]
    wheel_contact_forces_sln: tuple[tuple[float, ...], tuple[float, ...]]
    cart_force_residual_sln: tuple[float, float, float]
    cart_moment_residual_sln: tuple[float, float, float]


def fixed_contact_static_components(
    *,
    gravity_tangent: Any,
    gravity_normal: Any,
    com_s: Any,
    com_l: Any,
    com_n: Any,
    handle_s: Any,
    handle_n: Any,
    hitch_half_width: float,
    wheel_track: float,
    pitch_torque_on_robot: Any = 0.0,
) -> tuple[tuple[tuple[Any, ...], tuple[Any, ...]], tuple[tuple[Any, ...], tuple[Any, ...]]]:
    """Allocate the closed-chain static wrench using passive-wheel mechanics.

    The wheel bearings are passive, so a zero-speed equilibrium cannot rely on
    wheel tangent force: the two hitches carry the cart's full downslope load.
    Equal normal hand loading is selected from the redundant lateral load
    family; the wheels carry the lateral-CoM roll moment.  A lateral CoM offset
    gives unequal hand tangent forces to cancel yaw.  The corresponding joint
    torque is affine in ``F_s,left - F_s,right`` and is stored as a separate
    reset-library basis.

    Inputs may be floats, NumPy arrays, or Torch tensors.  The function uses
    only elementwise arithmetic so the exact same equations can be evaluated
    offline and for a batch of randomized runtime environments.
    """

    zero = gravity_tangent * 0.0
    hand_tangent_total = gravity_tangent
    gravity_pitch_moment = com_s * gravity_normal - com_n * gravity_tangent
    hand_normal_total = (
        handle_n * hand_tangent_total
        + gravity_pitch_moment
        - pitch_torque_on_robot
    ) / handle_s

    hand_tangent_difference = com_l * gravity_tangent / hitch_half_width
    hand_normal_difference = zero
    wheel_normal_total = gravity_normal - hand_normal_total
    wheel_normal_difference = 2.0 * com_l * gravity_normal / wheel_track

    # Preserve the batch type when callers pass the physically required scalar
    # zero for the crossbar's free pitch axis.
    hand_torque_on_cart = zero - pitch_torque_on_robot
    left_hand = (
        0.5 * (hand_tangent_total + hand_tangent_difference),
        zero,
        0.5 * (hand_normal_total + hand_normal_difference),
        zero,
        0.5 * hand_torque_on_cart,
        zero,
    )
    right_hand = (
        0.5 * (hand_tangent_total - hand_tangent_difference),
        zero,
        0.5 * (hand_normal_total - hand_normal_difference),
        zero,
        0.5 * hand_torque_on_cart,
        zero,
    )
    left_wheel = (
        zero,
        zero,
        0.5 * (wheel_normal_total + wheel_normal_difference),
    )
    right_wheel = (
        zero,
        zero,
        0.5 * (wheel_normal_total - wheel_normal_difference),
    )
    return (left_hand, right_hand), (left_wheel, right_wheel)


def solve_fixed_contact_statics(
    *,
    mass: float,
    gradient: float,
    com_from_axle_sln: tuple[float, float, float],
    handle_from_axle_sn: tuple[float, float],
    hitch_half_width: float,
    wheel_track: float,
    pitch_torque_on_robot: float = 0.0,
    gravity: float = 9.81,
) -> FixedContactStaticSolution:
    """Return and independently verify one scalar fixed-contact equilibrium."""

    scalars = (
        mass,
        gradient,
        *com_from_axle_sln,
        *handle_from_axle_sn,
        hitch_half_width,
        wheel_track,
        pitch_torque_on_robot,
        gravity,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("fixed-contact statics inputs must be finite")
    if mass <= 0.0 or gravity <= 0.0:
        raise ValueError("mass and gravity must be positive")
    if handle_from_axle_sn[0] <= 0.0:
        raise ValueError("handle tangent offset from the axle must be positive")
    if hitch_half_width <= 0.0 or wheel_track <= 0.0:
        raise ValueError("hitch half-width and wheel track must be positive")

    gamma = math.atan(gradient)
    gravity_tangent = mass * gravity * math.sin(gamma)
    gravity_normal = mass * gravity * math.cos(gamma)
    hand_wrenches, wheel_forces = fixed_contact_static_components(
        gravity_tangent=gravity_tangent,
        gravity_normal=gravity_normal,
        com_s=com_from_axle_sln[0],
        com_l=com_from_axle_sln[1],
        com_n=com_from_axle_sln[2],
        handle_s=handle_from_axle_sn[0],
        handle_n=handle_from_axle_sn[1],
        hitch_half_width=hitch_half_width,
        wheel_track=wheel_track,
        pitch_torque_on_robot=pitch_torque_on_robot,
    )
    hand_wrenches = tuple(tuple(float(value) for value in row) for row in hand_wrenches)
    wheel_forces = tuple(tuple(float(value) for value in row) for row in wheel_forces)

    gravity_force = (-gravity_tangent, 0.0, -gravity_normal)
    force_residual = tuple(
        gravity_force[axis]
        + sum(wrench[axis] for wrench in hand_wrenches)
        + sum(force[axis] for force in wheel_forces)
        for axis in range(3)
    )
    com_s, com_l, com_n = com_from_axle_sln
    handle_s, handle_n = handle_from_axle_sn
    gravity_moment = (
        -com_l * gravity_normal,
        com_s * gravity_normal - com_n * gravity_tangent,
        com_l * gravity_tangent,
    )
    moment_residual = [float(value) for value in gravity_moment]
    for lateral, wrench in zip((hitch_half_width, -hitch_half_width), hand_wrenches):
        force_s, force_l, force_n, torque_s, torque_l, torque_n = wrench
        moment_residual[0] += lateral * force_n - handle_n * force_l + torque_s
        moment_residual[1] += handle_n * force_s - handle_s * force_n + torque_l
        moment_residual[2] += handle_s * force_l - lateral * force_s + torque_n
    for lateral, force in zip((0.5 * wheel_track, -0.5 * wheel_track), wheel_forces):
        force_s, _force_l, force_n = force
        moment_residual[0] += lateral * force_n
        moment_residual[2] += -lateral * force_s

    return FixedContactStaticSolution(
        handle_wrenches_sln=hand_wrenches,  # type: ignore[arg-type]
        wheel_contact_forces_sln=wheel_forces,  # type: ignore[arg-type]
        cart_force_residual_sln=tuple(float(value) for value in force_residual),
        cart_moment_residual_sln=tuple(moment_residual),
    )


__all__ = [
    "FixedContactStaticSolution",
    "fixed_contact_static_components",
    "solve_fixed_contact_statics",
]
