"""Command, cart, FAT2, and ZMP dynamics for the rickshaw task.

All numerical kernels are plain Torch functions.  The small environment
adapters at the end use the state names defined by the implementation guide and
can be called from a ManagerBasedRLEnv pre-physics hook.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, replace
from typing import Any

import torch

from g1_rickshaw_lab.slope_contract import TERRAIN_COLUMNS_PER_TYPE


GRAVITY = 9.81


@dataclass
class SpeedReferenceCfg:
    acceleration_limit: float = MISSING
    jerk_limit: float = MISSING
    response_time: float = 0.5
    velocity_tolerance: float = 1.0e-3

    def validate(self) -> None:
        if self.acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be positive")
        if self.jerk_limit <= 0.0:
            raise ValueError("jerk_limit must be positive")
        if self.response_time <= 0.0:
            raise ValueError("response_time must be positive")
        if self.velocity_tolerance < 0.0:
            raise ValueError("velocity_tolerance must be non-negative")


@dataclass
class SpeedReferenceState:
    v_ref: torch.Tensor
    a_ref: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SpeedReferenceState":
        value = torch.zeros(num_envs, device=device, dtype=dtype)
        return cls(v_ref=value.clone(), a_ref=value.clone())

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.v_ref[ids] = 0.0
        self.a_ref[ids] = 0.0


def update_speed_reference(
    state: SpeedReferenceState,
    v_sample: torch.Tensor,
    dt: float,
    cfg: SpeedReferenceCfg,
) -> SpeedReferenceState:
    """Advance the jerk- and acceleration-limited reference in place.

    The stopping-velocity look-ahead and the target snap condition are exactly
    the command contract from section 5.5 of the guide.
    """

    cfg.validate()
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if state.v_ref.shape != state.a_ref.shape or v_sample.shape != state.v_ref.shape:
        raise ValueError("v_sample, v_ref, and a_ref must have identical shapes")

    v_stop = state.v_ref + state.a_ref * torch.abs(state.a_ref) / (2.0 * cfg.jerk_limit)
    a_des = torch.clamp(
        (v_sample - v_stop) / cfg.response_time,
        -cfg.acceleration_limit,
        cfg.acceleration_limit,
    )
    da = torch.clamp(
        a_des - state.a_ref,
        -cfg.jerk_limit * dt,
        cfg.jerk_limit * dt,
    )
    a_next = torch.clamp(
        state.a_ref + da,
        -cfg.acceleration_limit,
        cfg.acceleration_limit,
    )
    v_next = state.v_ref + a_next * dt
    settled = (torch.abs(v_sample - v_next) <= cfg.velocity_tolerance) & (
        torch.abs(a_next) <= cfg.jerk_limit * dt
    )
    state.v_ref[:] = torch.where(settled, v_sample, v_next)
    state.a_ref[:] = torch.where(settled, torch.zeros_like(a_next), a_next)
    return state


@dataclass(frozen=True)
class SlopeFrame:
    slope: torch.Tensor
    gamma: torch.Tensor
    tangent_w: torch.Tensor
    lateral_w: torch.Tensor
    normal_w: torch.Tensor
    quaternion_wxyz: torch.Tensor


def compute_slope_frame(
    terrain_level: torch.Tensor,
    terrain_column: torch.Tensor,
    *,
    columns_per_type: int = TERRAIN_COLUMNS_PER_TYPE,
) -> SlopeFrame:
    """Construct the configured slope frame from terrain row and column."""

    if terrain_level.shape != terrain_column.shape:
        raise ValueError("terrain_level and terrain_column must have identical shapes")
    if columns_per_type <= 0:
        raise ValueError("columns_per_type must be positive")
    dtype = torch.float32 if not terrain_level.dtype.is_floating_point else terrain_level.dtype
    level = terrain_level.to(dtype=dtype)
    column = terrain_column.to(device=level.device)
    magnitude = 0.01 + 0.01 * level
    sign = torch.where(
        column < columns_per_type,
        torch.zeros_like(magnitude),
        torch.where(
            column < 2 * columns_per_type,
            torch.ones_like(magnitude),
            -torch.ones_like(magnitude),
        ),
    )
    slope = sign * magnitude
    gamma = torch.atan(slope)
    zeros = torch.zeros_like(gamma)
    ones = torch.ones_like(gamma)
    tangent = torch.stack((torch.cos(gamma), zeros, torch.sin(gamma)), dim=-1)
    lateral = torch.stack((zeros, ones, zeros), dim=-1)
    normal = torch.stack((-torch.sin(gamma), zeros, torch.cos(gamma)), dim=-1)

    # The frame matrix has [e_s, e_y, e_n] as columns.  In Isaac Lab's wxyz
    # convention this is a rotation of -gamma about world Y.
    quaternion = torch.stack(
        (torch.cos(0.5 * gamma), zeros, -torch.sin(0.5 * gamma), zeros), dim=-1
    )
    return SlopeFrame(slope, gamma, tangent, lateral, normal, quaternion)


def low_pass(
    previous: torch.Tensor,
    sample: torch.Tensor,
    *,
    cutoff_hz: float,
    dt: float,
) -> torch.Tensor:
    """One-pole low pass with an exact continuous-time pole mapping."""

    if previous.shape != sample.shape:
        raise ValueError("low-pass state and sample must have identical shapes")
    if cutoff_hz <= 0.0 or dt <= 0.0:
        raise ValueError("cutoff_hz and dt must be positive")
    gain = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    return previous + gain * (sample - previous)


@dataclass
class RollingResistanceCfg:
    c_rr: tuple[float, float] = MISSING
    enabled: bool = True
    velocity_epsilon: float = 0.05
    normal_force_filter_hz: float = 20.0

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("rolling-resistance enabled must be boolean")
        if len(self.c_rr) != 2 or self.c_rr[0] < 0.0 or self.c_rr[1] < self.c_rr[0]:
            raise ValueError("c_rr must be a non-negative ordered (low, high) pair")
        if self.velocity_epsilon <= 0.0:
            raise ValueError("velocity_epsilon must be positive")
        if self.normal_force_filter_hz <= 0.0:
            raise ValueError("normal_force_filter_hz must be positive")


def configure_rolling_resistance(env_cfg: Any, *, enabled: bool) -> RollingResistanceCfg:
    """Replace the rolling-resistance config at both runtime bindings."""

    if type(enabled) is not bool:
        raise ValueError("rolling-resistance enabled must be boolean")
    current = getattr(env_cfg, "rolling_resistance", None)
    if not isinstance(current, RollingResistanceCfg):
        raise TypeError("env_cfg.rolling_resistance must be a RollingResistanceCfg")
    events = getattr(env_cfg, "events", None)
    initialize_mdp = getattr(events, "initialize_mdp", None)
    params = getattr(initialize_mdp, "params", None)
    if not isinstance(params, dict) or "rolling_resistance_cfg" not in params:
        raise RuntimeError(
            "env_cfg.events.initialize_mdp.params must contain rolling_resistance_cfg"
        )

    configured = replace(current, enabled=enabled)
    env_cfg.rolling_resistance = configured
    params["rolling_resistance_cfg"] = configured
    return configured


def rolling_resistance_wrench(
    wheel_velocity_w: torch.Tensor,
    wheel_contact_force_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
    c_rr: torch.Tensor,
    previous_normal_force: torch.Tensor,
    *,
    velocity_epsilon: float = 0.05,
    normal_force_filter_hz: float = 20.0,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical wheel-center rolling-resistance forces.

    Returns ``(force_w, filtered_normal_force, tangential_velocity)``.  No axle
    torque is returned because adding one would count rolling resistance twice.
    """

    if wheel_velocity_w.ndim != 3 or wheel_velocity_w.shape[-1] != 3:
        raise ValueError("wheel_velocity_w must have shape [N, W, 3]")
    if wheel_contact_force_w.shape != wheel_velocity_w.shape:
        raise ValueError("wheel contact forces must match wheel velocity shape")
    if path_tangent_w.shape != (wheel_velocity_w.shape[0], 3):
        raise ValueError("path_tangent_w must have shape [N, 3]")
    if path_normal_w.shape != path_tangent_w.shape:
        raise ValueError("path normal and tangent shapes differ")
    if previous_normal_force.shape != wheel_velocity_w.shape[:2]:
        raise ValueError("normal-force filter state must have shape [N, W]")
    if velocity_epsilon <= 0.0:
        raise ValueError("velocity_epsilon must be positive")

    tangential_velocity = torch.sum(
        wheel_velocity_w * path_tangent_w[:, None, :], dim=-1
    )
    raw_normal_force = torch.clamp(
        torch.sum(wheel_contact_force_w * path_normal_w[:, None, :], dim=-1),
        min=0.0,
    )
    normal_force = low_pass(
        previous_normal_force,
        raw_normal_force,
        cutoff_hz=normal_force_filter_hz,
        dt=dt,
    )
    coefficient = torch.as_tensor(
        c_rr, device=wheel_velocity_w.device, dtype=wheel_velocity_w.dtype
    )
    if coefficient.ndim == 0:
        coefficient = coefficient.expand(wheel_velocity_w.shape[0])
    if coefficient.shape != (wheel_velocity_w.shape[0],):
        raise ValueError("c_rr must be scalar or have shape [N]")
    direction = torch.tanh(tangential_velocity / velocity_epsilon)
    magnitude = -coefficient[:, None] * normal_force * direction
    force_w = magnitude[..., None] * path_tangent_w[:, None, :]
    return force_w, normal_force, tangential_velocity


@dataclass
class SecondOrderLowPassState:
    """Two cascaded filter stages plus histories for finite differences."""

    stage_1: torch.Tensor
    stage_2: torch.Tensor
    previous: torch.Tensor
    previous_previous: torch.Tensor

    @classmethod
    def initialized(cls, value: torch.Tensor) -> "SecondOrderLowPassState":
        return cls(value.clone(), value.clone(), value.clone(), value.clone())

    def reset(self, value: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        target = value
        if env_ids is not None and value.shape == self.stage_1.shape:
            target = value[env_ids]
        self.stage_1[ids] = target
        self.stage_2[ids] = target
        self.previous[ids] = target
        self.previous_previous[ids] = target


def _filtered_signal_step(
    value: torch.Tensor,
    state: SecondOrderLowPassState,
    *,
    dt: float,
    cutoff_hz: float,
) -> torch.Tensor:
    stage_1 = low_pass(state.stage_1, value, cutoff_hz=cutoff_hz, dt=dt)
    stage_2 = low_pass(state.stage_2, stage_1, cutoff_hz=cutoff_hz, dt=dt)
    state.stage_1[:] = stage_1
    state.stage_2[:] = stage_2
    return stage_2


def filtered_first_derivative(
    value: torch.Tensor,
    state: SecondOrderLowPassState,
    dt: float,
    *,
    cutoff_hz: float = 20.0,
) -> torch.Tensor:
    filtered = _filtered_signal_step(value, state, dt=dt, cutoff_hz=cutoff_hz)
    derivative = (filtered - state.previous) / dt
    state.previous_previous[:] = state.previous
    state.previous[:] = filtered
    return derivative


def filtered_second_derivative(
    value: torch.Tensor,
    state: SecondOrderLowPassState,
    dt: float,
    *,
    cutoff_hz: float = 20.0,
) -> torch.Tensor:
    filtered = _filtered_signal_step(value, state, dt=dt, cutoff_hz=cutoff_hz)
    derivative = (filtered - 2.0 * state.previous + state.previous_previous) / (dt * dt)
    state.previous_previous[:] = state.previous
    state.previous[:] = filtered
    return derivative


@dataclass(frozen=True)
class RickshawMassProperties:
    """Per-environment cart quantities in the cart frame about the wheel axle."""

    m_cart: torch.Tensor
    com_x_from_axle: torch.Tensor
    com_z_from_axle: torch.Tensor
    pitch_inertia_about_axle: torch.Tensor
    m_eff: torch.Tensor
    b_eff: torch.Tensor
    handle_x_from_axle: torch.Tensor
    handle_z_from_axle: torch.Tensor


def articulation_center_of_mass(
    body_com_pos_w: torch.Tensor,
    body_com_lin_vel_w: torch.Tensor,
    body_masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mass-weighted whole-articulation CoM position and velocity.

    Isaac Lab's ``root_com_*`` fields describe only the root rigid body.  ZMP
    and FAT require the system CoM across every retained articulation body.
    """

    if body_com_pos_w.ndim != 3 or body_com_pos_w.shape[-1] != 3:
        raise ValueError("body CoM positions must have shape [N,B,3]")
    if body_com_lin_vel_w.shape != body_com_pos_w.shape:
        raise ValueError("body CoM linear velocities must match positions")
    if body_masses.shape != body_com_pos_w.shape[:2]:
        raise ValueError("body masses must have shape [N,B]")
    if torch.any(~torch.isfinite(body_masses)) or torch.any(body_masses <= 0.0):
        raise ValueError("every retained articulation body must have finite positive mass")
    total_mass = torch.sum(body_masses, dim=-1)
    weights = body_masses / total_mass[:, None]
    position = torch.sum(body_com_pos_w * weights[..., None], dim=1)
    velocity = torch.sum(body_com_lin_vel_w * weights[..., None], dim=1)
    return position, velocity, total_mass


def robot_system_mass_kinematics(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read current whole-G1 CoM kinematics using startup-validated masses."""

    robot = env.scene["robot"]
    masses = getattr(env, "robot_body_masses", None)
    if masses is None:
        raise RuntimeError("startup must install validated env.robot_body_masses")
    return articulation_center_of_mass(
        robot.data.body_com_pos_w,
        robot.data.body_com_lin_vel_w,
        masses.to(device=robot.device, dtype=robot.data.body_com_pos_w.dtype),
    )


def cart_system_mass_kinematics(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read current whole-rickshaw CoM kinematics using sampled body masses."""

    cart = env.scene["rickshaw"]
    masses = getattr(env, "rickshaw_body_masses", None)
    if masses is None:
        raise RuntimeError("payload randomization must publish current cart body masses")
    return articulation_center_of_mass(
        cart.data.body_com_pos_w,
        cart.data.body_com_lin_vel_w,
        masses.to(device=cart.device, dtype=cart.data.body_com_pos_w.dtype),
    )


def cart_ground_contact_force_w(env: Any) -> torch.Tensor:
    """Return the external terrain force on the cart from its two wheels.

    The all-body cart contact sensor also contains the intended gripper/crossbar
    contact. Subtracting that force from cart momentum would remove part of the
    robot-cart interaction that this estimator is intended to measure.
    """

    force_w = env.scene["wheel_contacts"].data.net_forces_w[:, env.wheel_sensor_ids]
    if force_w.ndim != 3 or force_w.shape[1:] != (2, 3):
        raise ValueError("wheel ground-contact force must have shape [N,2,3]")
    return torch.sum(force_w, dim=1)


def accumulate_cart_interaction_wrench(
    env: Any, rolling_force_w: torch.Tensor
) -> None:
    """Accumulate one pre-physics ground-contact/rolling-force sample."""

    state = getattr(env, "cart_interaction_wrench_state", None)
    if state is None:
        raise RuntimeError("cart interaction wrench state is not initialized")
    if rolling_force_w.ndim != 3 or rolling_force_w.shape[1:] != (2, 3):
        raise ValueError("rolling resistance force must have shape [N,2,3]")
    contact_force_w = cart_ground_contact_force_w(env)
    state.accumulate(contact_force_w, torch.sum(rolling_force_w, dim=1))


def update_cart_interaction_wrench(
    env: Any,
    cart_kinematics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Recover the measured cart-on-robot force from whole-cart momentum balance."""

    if not hasattr(env, "read_d6_reaction_residual"):
        raise RuntimeError("D6 residual/impulse adapter was not installed at startup")
    # The incoming-joint reader remains a conservative residual/impulse proxy;
    # its wrench is not a valid external D6 force for excluded closed-loop joints.
    env.read_d6_reaction_residual()
    state = getattr(env, "cart_interaction_wrench_state", None)
    if state is None:
        raise RuntimeError("cart interaction wrench state is not initialized")
    if cart_kinematics is None:
        cart_kinematics = cart_system_mass_kinematics(env)
    _, com_velocity_w, total_mass = cart_kinematics
    final_contact_force_w = cart_ground_contact_force_w(env)
    gravity_w = torch.tensor(
        env.cfg.sim.gravity,
        device=env.device,
        dtype=com_velocity_w.dtype,
    )
    force_on_cart_w, valid = state.finish(
        com_velocity_w,
        total_mass,
        gravity_w,
        final_contact_force_w,
        env.step_dt,
    )
    hand_force_w = -force_on_cart_w
    env.rickshaw_state.hand_force_w[:] = hand_force_w
    env.rickshaw_state.hand_torque_w.zero_()
    env.rickshaw_state.d6_wrench_w.zero_()
    env.rickshaw_state.d6_wrench_w[..., :3] = 0.5 * hand_force_w[:, None, :]
    env.cart_interaction_wrench_valid = valid
    return hand_force_w


def parallel_axis_inertia(
    inertia_at_com: torch.Tensor, mass: torch.Tensor, displacement: torch.Tensor
) -> torch.Tensor:
    """Shift a 3-D inertia tensor from its CoM by ``displacement``."""

    if inertia_at_com.shape[-2:] != (3, 3) or displacement.shape[-1] != 3:
        raise ValueError("inertia must end in [3,3] and displacement in [3]")
    eye = torch.eye(3, device=inertia_at_com.device, dtype=inertia_at_com.dtype)
    squared_distance = torch.sum(displacement * displacement, dim=-1)
    outer = displacement[..., :, None] * displacement[..., None, :]
    return inertia_at_com + mass[..., None, None] * (
        squared_distance[..., None, None] * eye - outer
    )


def combine_mass_properties(
    base_mass: torch.Tensor,
    base_com: torch.Tensor,
    base_inertia_at_com: torch.Tensor,
    payload_mass: torch.Tensor,
    payload_com: torch.Tensor,
    payload_inertia_at_com: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine base and payload mass/CoM/inertia using parallel-axis shifts."""

    total_mass = base_mass + payload_mass
    if torch.any(total_mass <= 0.0):
        raise ValueError("combined mass must be positive")
    total_com = (
        base_mass[..., None] * base_com + payload_mass[..., None] * payload_com
    ) / total_mass[..., None]
    base_shift = base_com - total_com
    payload_shift = payload_com - total_com
    total_inertia = parallel_axis_inertia(
        base_inertia_at_com, base_mass, base_shift
    ) + parallel_axis_inertia(payload_inertia_at_com, payload_mass, payload_shift)
    return total_mass, total_com, total_inertia


def effective_cart_mass(
    cart_mass: torch.Tensor, wheel_spin_inertia: torch.Tensor, wheel_radius: torch.Tensor
) -> torch.Tensor:
    if torch.any(wheel_radius <= 0.0):
        raise ValueError("wheel radii must be positive")
    return cart_mass + torch.sum(wheel_spin_inertia / torch.square(wheel_radius), dim=-1)


def effective_wheel_damping(
    wheel_damping: torch.Tensor, wheel_radius: torch.Tensor
) -> torch.Tensor:
    if torch.any(wheel_radius <= 0.0):
        raise ValueError("wheel radii must be positive")
    return torch.sum(wheel_damping / torch.square(wheel_radius), dim=-1)


@dataclass
class AnalyticForceCfg:
    minimum_wheel_normal_force: float = MISSING
    velocity_epsilon: float = 0.05
    derivative_filter_hz: float = 20.0
    minimum_handle_x: float = 0.5


@dataclass
class AnalyticHandleForceState:
    velocity_filter: SecondOrderLowPassState
    pitch_filter: SecondOrderLowPassState
    a_s: torch.Tensor
    alpha_ddot: torch.Tensor
    t_s: torch.Tensor
    t_n: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def initialized(
        cls, tangential_velocity: torch.Tensor, pitch: torch.Tensor
    ) -> "AnalyticHandleForceState":
        zeros = torch.zeros_like(tangential_velocity)
        return cls(
            velocity_filter=SecondOrderLowPassState.initialized(tangential_velocity),
            pitch_filter=SecondOrderLowPassState.initialized(pitch),
            a_s=zeros.clone(),
            alpha_ddot=zeros.clone(),
            t_s=zeros.clone(),
            t_n=zeros.clone(),
            valid=torch.zeros_like(tangential_velocity, dtype=torch.bool),
        )

    def reset(
        self,
        tangential_velocity: torch.Tensor,
        pitch: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.velocity_filter.reset(tangential_velocity, env_ids)
        self.pitch_filter.reset(pitch, env_ids)
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.a_s[ids] = 0.0
        self.alpha_ddot[ids] = 0.0
        self.t_s[ids] = 0.0
        self.t_n[ids] = 0.0
        self.valid[ids] = False


@dataclass
class WrenchConsistencyState:
    analytic_buffer: torch.Tensor
    measured_buffer: torch.Tensor
    source_valid_buffer: torch.Tensor
    cursor: torch.Tensor
    count: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        window_steps: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "WrenchConsistencyState":
        if num_envs <= 0 or window_steps <= 0:
            raise ValueError("wrench consistency dimensions must be positive")
        return cls(
            analytic_buffer=torch.zeros(
                (num_envs, window_steps, 2), device=device, dtype=dtype
            ),
            measured_buffer=torch.zeros(
                (num_envs, window_steps, 2), device=device, dtype=dtype
            ),
            source_valid_buffer=torch.zeros(
                (num_envs, window_steps), device=device, dtype=torch.bool
            ),
            cursor=torch.zeros(num_envs, device=device, dtype=torch.long),
            count=torch.zeros(num_envs, device=device, dtype=torch.long),
        )

    @property
    def window_steps(self) -> int:
        return int(self.analytic_buffer.shape[1])

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.analytic_buffer[ids] = 0.0
        self.measured_buffer[ids] = 0.0
        self.source_valid_buffer[ids] = False
        self.cursor[ids] = 0
        self.count[ids] = 0


@dataclass
class FAT2ComRadiusState:
    """Windowed sagittal CoM radius with a calibrated reset fallback."""

    sample_buffer: torch.Tensor
    running_sum: torch.Tensor
    cursor: torch.Tensor
    count: torch.Tensor
    filtered_radius: torch.Tensor
    reference_radius: torch.Tensor

    @classmethod
    def initialized(
        cls,
        num_envs: int,
        window_steps: int,
        reference_radius: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FAT2ComRadiusState":
        if num_envs <= 0 or window_steps <= 0 or reference_radius <= 0.0:
            raise ValueError("FAT2 CoM radius state dimensions and reference must be positive")
        reference = torch.full(
            (num_envs,), reference_radius, device=device, dtype=dtype
        )
        return cls(
            sample_buffer=torch.zeros(
                (num_envs, window_steps), device=device, dtype=dtype
            ),
            running_sum=torch.zeros(num_envs, device=device, dtype=dtype),
            cursor=torch.zeros(num_envs, device=device, dtype=torch.long),
            count=torch.zeros(num_envs, device=device, dtype=torch.long),
            filtered_radius=reference.clone(),
            reference_radius=reference,
        )

    @property
    def window_steps(self) -> int:
        return int(self.sample_buffer.shape[1])

    def update(
        self,
        sample: torch.Tensor,
        valid: torch.Tensor,
        *,
        minimum: float,
        maximum: float,
    ) -> torch.Tensor:
        expected = self.filtered_radius.shape
        if sample.shape != expected or valid.shape != expected or valid.dtype != torch.bool:
            raise ValueError("FAT2 CoM radius sample and validity must have shape [N]")
        if minimum <= 0.0 or minimum >= maximum:
            raise ValueError("FAT2 CoM radius bounds must be positive and ordered")
        valid = valid & torch.isfinite(sample)
        clipped = torch.clamp(sample, min=minimum, max=maximum)
        env_ids = torch.arange(expected[0], device=self.cursor.device)
        write_ids = env_ids[valid]
        outgoing = self.sample_buffer[write_ids, self.cursor[write_ids]]
        self.running_sum[write_ids] += clipped[write_ids] - outgoing
        self.sample_buffer[write_ids, self.cursor[write_ids]] = clipped[write_ids]
        self.cursor[:] = torch.where(
            valid, (self.cursor + 1) % self.window_steps, self.cursor
        )
        self.count[:] = torch.where(
            valid,
            torch.clamp(self.count + 1, max=self.window_steps),
            self.count,
        )
        denominator = torch.clamp(self.count, min=1).to(sample.dtype)
        window_mean = self.running_sum / denominator
        self.filtered_radius[:] = torch.where(
            valid, window_mean, self.filtered_radius
        )
        return self.filtered_radius

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.sample_buffer[ids] = 0.0
        self.running_sum[ids] = 0.0
        self.cursor[ids] = 0
        self.count[ids] = 0
        self.filtered_radius[ids] = self.reference_radius[ids]


@dataclass
class CartInteractionWrenchState:
    """Policy-rate cart interaction force reconstructed from momentum balance."""

    previous_com_velocity_w: torch.Tensor
    contact_force_sum_w: torch.Tensor
    first_contact_force_w: torch.Tensor
    rolling_force_sum_w: torch.Tensor
    sample_count: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def initialized(cls, com_velocity_w: torch.Tensor) -> "CartInteractionWrenchState":
        if com_velocity_w.ndim != 2 or com_velocity_w.shape[-1] != 3:
            raise ValueError("cart CoM velocity must have shape [N,3]")
        zeros = torch.zeros_like(com_velocity_w)
        return cls(
            previous_com_velocity_w=com_velocity_w.clone(),
            contact_force_sum_w=zeros.clone(),
            first_contact_force_w=zeros.clone(),
            rolling_force_sum_w=zeros.clone(),
            sample_count=torch.zeros(
                com_velocity_w.shape[0],
                device=com_velocity_w.device,
                dtype=torch.long,
            ),
            valid=torch.zeros(
                com_velocity_w.shape[0],
                device=com_velocity_w.device,
                dtype=torch.bool,
            ),
        )

    def reset(
        self, com_velocity_w: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            source = com_velocity_w
        elif com_velocity_w.shape[0] == self.previous_com_velocity_w.shape[0]:
            source = com_velocity_w[env_ids]
        elif com_velocity_w.shape == (env_ids.numel(), 3):
            source = com_velocity_w
        else:
            raise ValueError("reset cart CoM velocity has the wrong batch shape")
        self.previous_com_velocity_w[ids] = source
        self.contact_force_sum_w[ids] = 0.0
        self.first_contact_force_w[ids] = 0.0
        self.rolling_force_sum_w[ids] = 0.0
        self.sample_count[ids] = 0
        self.valid[ids] = False

    def accumulate(
        self, contact_force_w: torch.Tensor, rolling_force_w: torch.Tensor
    ) -> None:
        expected = self.previous_com_velocity_w.shape
        if contact_force_w.shape != expected or rolling_force_w.shape != expected:
            raise ValueError("cart force samples must have shape [N,3]")
        first = self.sample_count == 0
        self.first_contact_force_w[first] = contact_force_w[first]
        self.contact_force_sum_w += contact_force_w
        self.rolling_force_sum_w += rolling_force_w
        self.sample_count += 1

    def finish(
        self,
        com_velocity_w: torch.Tensor,
        total_mass: torch.Tensor,
        gravity_w: torch.Tensor,
        final_contact_force_w: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dt <= 0.0:
            raise ValueError("cart interaction update dt must be positive")
        expected = self.previous_com_velocity_w.shape
        if com_velocity_w.shape != expected or final_contact_force_w.shape != expected:
            raise ValueError("cart interaction kinematics/contacts must have shape [N,3]")
        if total_mass.shape != expected[:1] or gravity_w.shape != (3,):
            raise ValueError("cart interaction mass/gravity shapes are invalid")
        count = self.sample_count.to(com_velocity_w.dtype)
        ready = self.sample_count > 0
        denominator = torch.clamp(count, min=1.0)[:, None]
        # Pre-physics hooks observe the previous interval's final contact on
        # their first call. Replace that sample with this interval's final one.
        contact_average = (
            self.contact_force_sum_w
            - self.first_contact_force_w
            + final_contact_force_w
        ) / denominator
        rolling_average = self.rolling_force_sum_w / denominator
        acceleration_w = (
            com_velocity_w - self.previous_com_velocity_w
        ) / dt
        force_on_cart_w = (
            total_mass[:, None] * acceleration_w
            - total_mass[:, None] * gravity_w[None, :]
            - contact_average
            - rolling_average
        )
        finite = torch.all(torch.isfinite(force_on_cart_w), dim=-1)
        valid = ready & finite & torch.isfinite(total_mass) & (total_mass > 0.0)
        force_on_cart_w = torch.where(
            valid[:, None], force_on_cart_w, torch.zeros_like(force_on_cart_w)
        )
        self.previous_com_velocity_w[:] = com_velocity_w
        self.contact_force_sum_w.zero_()
        self.first_contact_force_w.zero_()
        self.rolling_force_sum_w.zero_()
        self.sample_count.zero_()
        self.valid[:] = valid
        return force_on_cart_w, valid


def update_wrench_consistency_state(
    state: WrenchConsistencyState,
    analytic_force_sn: torch.Tensor,
    measured_force_sn: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_floor_n: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update transient-robust force windows and apply the FAT2 validity gate.

    The comparison is an impulse-bias test rather than a pointwise force test.
    Its normalization uses the mean analytic force magnitude, so a large
    oscillatory wrench cannot create a singular relative error when its signed
    window mean is close to zero.
    """

    expected = (state.analytic_buffer.shape[0], 2)
    if analytic_force_sn.shape != expected or measured_force_sn.shape != expected:
        raise ValueError(f"wrench consistency forces must have shape {expected}")
    if source_valid.shape != expected[:1] or source_valid.dtype != torch.bool:
        raise ValueError("wrench consistency source_valid must be bool [N]")
    if not 0.0 <= relative_tolerance <= 1.0 or absolute_floor_n <= 0.0:
        raise ValueError("wrench consistency tolerances are invalid")
    env_ids = torch.arange(expected[0], device=state.cursor.device)
    state.analytic_buffer[env_ids, state.cursor] = analytic_force_sn
    state.measured_buffer[env_ids, state.cursor] = measured_force_sn
    state.source_valid_buffer[env_ids, state.cursor] = source_valid
    state.cursor[:] = (state.cursor + 1) % state.window_steps
    state.count[:] = torch.clamp(state.count + 1, max=state.window_steps)

    denominator_count = torch.clamp(state.count, min=1).to(analytic_force_sn.dtype)
    analytic_mean = torch.sum(state.analytic_buffer, dim=1) / denominator_count[:, None]
    measured_mean = torch.sum(state.measured_buffer, dim=1) / denominator_count[:, None]
    analytic_abs_mean = (
        torch.sum(torch.abs(state.analytic_buffer), dim=1) / denominator_count[:, None]
    )
    floor = torch.as_tensor(
        absolute_floor_n, device=analytic_force_sn.device, dtype=analytic_force_sn.dtype
    )
    normalization_force = torch.maximum(analytic_abs_mean, floor)
    relative_error = torch.abs(measured_mean - analytic_mean) / normalization_force
    # Sign is only identifiable when both net forces exceed the calibrated
    # uncertainty band.  Relative error still rejects a material opposite bias.
    sign_resolved = (
        (torch.abs(analytic_mean) > relative_tolerance * normalization_force)
        & (torch.abs(measured_mean) > relative_tolerance * normalization_force)
    )
    same_sign = (~sign_resolved) | (
        torch.sign(analytic_mean) == torch.sign(measured_mean)
    )
    full_window = state.count >= state.window_steps
    source_window_valid = torch.all(state.source_valid_buffer, dim=-1)
    consistent = (
        full_window
        & source_window_valid
        & torch.all(same_sign & (relative_error <= relative_tolerance), dim=-1)
    )
    return consistent, relative_error, analytic_mean


def analytic_handle_force(
    v_s: torch.Tensor,
    a_s: torch.Tensor,
    alpha_ddot: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    c_rr: torch.Tensor,
    wheel_normal_force: torch.Tensor,
    properties: RickshawMassProperties,
    *,
    velocity_epsilon: float = 0.05,
    minimum_handle_x: float = 0.5,
    handle_from_axle_sn: torch.Tensor | None = None,
    com_from_axle_sn: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the complete cart tangent force and axle moment balance.

    The stored CoM and handle coordinates are cart-frame vectors from the axle.
    They are rotated by the current front-lift pitch before evaluating moments
    in the slope frame.
    """

    if velocity_epsilon <= 0.0:
        raise ValueError("velocity_epsilon must be positive")
    n_w = torch.sum(wheel_normal_force, dim=-1)
    rr_magnitude = c_rr * n_w * torch.tanh(v_s / velocity_epsilon)
    t_s = (
        properties.m_eff * a_s
        + properties.m_cart * GRAVITY * torch.sin(gamma)
        + rr_magnitude
        + properties.b_eff * v_s
    )
    if (handle_from_axle_sn is None) != (com_from_axle_sn is None):
        raise ValueError("actual handle and CoM geometry must be supplied together")
    if handle_from_axle_sn is None:
        cosine = torch.cos(alpha)
        sine = torch.sin(alpha)
        handle_x = cosine * properties.handle_x_from_axle - sine * properties.handle_z_from_axle
        handle_z = sine * properties.handle_x_from_axle + cosine * properties.handle_z_from_axle
        com_x = cosine * properties.com_x_from_axle - sine * properties.com_z_from_axle
        com_z = sine * properties.com_x_from_axle + cosine * properties.com_z_from_axle
    else:
        expected_shape = (v_s.shape[0], 2)
        if handle_from_axle_sn.shape != expected_shape or com_from_axle_sn.shape != expected_shape:
            raise ValueError(f"actual cart geometry must have shape {expected_shape}")
        handle_x, handle_z = handle_from_axle_sn.unbind(dim=-1)
        com_x, com_z = com_from_axle_sn.unbind(dim=-1)
    valid = handle_x > minimum_handle_x
    denominator = torch.where(valid, handle_x, torch.ones_like(v_s))
    t_n = (
        properties.pitch_inertia_about_axle * alpha_ddot
        + handle_z * t_s
        + properties.m_cart
        * GRAVITY
        * (
            com_x * torch.cos(gamma)
            - com_z * torch.sin(gamma)
        )
    ) / denominator
    t_s = torch.where(valid, t_s, torch.zeros_like(t_s))
    t_n = torch.where(valid, t_n, torch.zeros_like(t_n))
    return t_s, t_n, valid


def update_analytic_handle_force_state(
    state: AnalyticHandleForceState,
    v_s: torch.Tensor,
    pitch: torch.Tensor,
    gamma: torch.Tensor,
    c_rr: torch.Tensor,
    wheel_normal_force: torch.Tensor,
    properties: RickshawMassProperties,
    dt: float,
    cfg: AnalyticForceCfg,
    *,
    handle_from_axle_sn: torch.Tensor | None = None,
    com_from_axle_sn: torch.Tensor | None = None,
) -> AnalyticHandleForceState:
    """Filter/differentiate cart motion and update the analytic FAT2 reference."""

    a_s = filtered_first_derivative(
        v_s, state.velocity_filter, dt, cutoff_hz=cfg.derivative_filter_hz
    )
    alpha_ddot = filtered_second_derivative(
        pitch, state.pitch_filter, dt, cutoff_hz=cfg.derivative_filter_hz
    )
    t_s, t_n, geometry_valid = analytic_handle_force(
        v_s,
        a_s,
        alpha_ddot,
        pitch,
        gamma,
        c_rr,
        wheel_normal_force,
        properties,
        velocity_epsilon=cfg.velocity_epsilon,
        minimum_handle_x=cfg.minimum_handle_x,
        handle_from_axle_sn=handle_from_axle_sn,
        com_from_axle_sn=com_from_axle_sn,
    )
    wheel_valid = torch.all(wheel_normal_force >= cfg.minimum_wheel_normal_force, dim=-1)
    state.a_s[:] = a_s
    state.alpha_ddot[:] = alpha_ddot
    state.t_s[:] = t_s
    state.t_n[:] = t_n
    state.valid[:] = geometry_valid & wheel_valid
    return state


def actual_rickshaw_geometry_in_slope_frame(
    env: Any,
    *,
    cart_com_w: torch.Tensor | None = None,
    hitch_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actual ``(s,n)`` hitch and whole-cart CoM vectors from the axle."""

    cart = env.scene["rickshaw"]
    if cart_com_w is None:
        cart_com_w, _, _ = cart_system_mass_kinematics(env)
    axle_w = torch.mean(cart.data.body_pos_w[:, env.wheel_body_ids], dim=1)
    if hitch_w is None:
        hitch_w = torch.mean(cart.data.body_pos_w[:, env.hitch_body_ids], dim=1)
    handle_vector = hitch_w - axle_w
    com_vector = cart_com_w - axle_w
    handle_sn = torch.stack(
        (
            torch.sum(handle_vector * env.path_tangent_w, dim=-1),
            torch.sum(handle_vector * env.path_normal_w, dim=-1),
        ),
        dim=-1,
    )
    com_sn = torch.stack(
        (
            torch.sum(com_vector * env.path_tangent_w, dim=-1),
            torch.sum(com_vector * env.path_normal_w, dim=-1),
        ),
        dim=-1,
    )
    return handle_sn, com_sn


def quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by wxyz quaternions without an Isaac Lab dependency."""

    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector dimensions must end in 4/3")
    q_vec = quaternion[..., 1:]
    uv = torch.linalg.cross(q_vec, vector, dim=-1)
    uuv = torch.linalg.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def rickshaw_pitch_from_quaternion(
    quaternion_wxyz: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Return front-lift pitch ``alpha`` relative to the signed slope frame."""

    local_x = torch.zeros_like(path_tangent_w)
    local_x[..., 0] = 1.0
    forward_w = quat_apply_wxyz(quaternion_wxyz, local_x)
    forward_s = torch.sum(forward_w * path_tangent_w, dim=-1)
    forward_n = torch.sum(forward_w * path_normal_w, dim=-1)
    return torch.atan2(forward_n, forward_s)


def torso_tilt_from_slope_normal(
    torso_quaternion_wxyz: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Return the unsigned 3D angle between torso +Z and the slope normal."""

    if torso_quaternion_wxyz.shape[:-1] != path_normal_w.shape[:-1]:
        raise ValueError("torso quaternion and path normal batch shapes differ")
    if torso_quaternion_wxyz.shape[-1] != 4 or path_normal_w.shape[-1] != 3:
        raise ValueError("torso quaternion/path normal dimensions must end in 4/3")
    local_z = torch.zeros_like(path_normal_w)
    local_z[..., 2] = 1.0
    up_w = quat_apply_wxyz(torso_quaternion_wxyz, local_z)
    up_norm = torch.linalg.vector_norm(up_w, dim=-1, keepdim=True)
    normal_norm = torch.linalg.vector_norm(path_normal_w, dim=-1, keepdim=True)
    if torch.any(up_norm <= 1.0e-8) or torch.any(normal_norm <= 1.0e-8):
        raise ValueError("torso up and path normal vectors must be nonzero")
    up_w = up_w / up_norm
    normal_w = path_normal_w / normal_norm
    sine = torch.linalg.vector_norm(torch.linalg.cross(up_w, normal_w, dim=-1), dim=-1)
    cosine = torch.sum(up_w * normal_w, dim=-1)
    return torch.atan2(sine, cosine)


def torso_pitch_from_world_vertical(
    torso_quaternion_wxyz: torch.Tensor,
    path_tangent_w: torch.Tensor,
) -> torch.Tensor:
    """Return torso tilt from world vertical, positive along the path."""

    local_z = torch.zeros_like(path_tangent_w)
    local_z[..., 2] = 1.0
    up_w = quat_apply_wxyz(torso_quaternion_wxyz, local_z)
    world_up = torch.zeros_like(path_tangent_w)
    world_up[..., 2] = 1.0
    horizontal_forward = path_tangent_w - torch.sum(
        path_tangent_w * world_up, dim=-1, keepdim=True
    ) * world_up
    horizontal_norm = torch.linalg.vector_norm(horizontal_forward, dim=-1, keepdim=True)
    if torch.any(horizontal_norm <= 1.0e-6):
        raise ValueError("path tangent must have a nonzero horizontal projection")
    horizontal_forward = horizontal_forward / horizontal_norm
    return torch.atan2(
        torch.sum(up_w * horizontal_forward, dim=-1),
        torch.sum(up_w * world_up, dim=-1),
    )


@dataclass
class FAT2Cfg:
    robot_mass: float = MISSING
    com_radius: float = MISSING
    com_radius_bounds: tuple[float, float] = MISSING
    theta_max: float = MISSING
    wrench_consistency_relative_tolerance: float = MISSING
    wrench_consistency_absolute_floor_n: float = MISSING
    wrench_consistency_window_steps: int = MISSING

    def validate(self) -> None:
        if self.robot_mass <= 0.0 or self.com_radius <= 0.0:
            raise ValueError("FAT2 robot mass and CoM radius must be calibrated")
        if len(self.com_radius_bounds) != 2:
            raise ValueError("FAT2 CoM radius bounds must contain two values")
        radius_min, radius_max = self.com_radius_bounds
        if radius_min <= 0.0 or radius_min >= radius_max:
            raise ValueError("FAT2 CoM radius bounds must be positive and ordered")
        if not radius_min <= self.com_radius <= radius_max:
            raise ValueError("FAT2 calibrated CoM radius must lie within its bounds")
        if not 0.0 < self.theta_max < math.pi / 2.0:
            raise ValueError("FAT2 theta_max must lie in (0, pi/2)")
        if not 0.0 <= self.wrench_consistency_relative_tolerance <= 1.0:
            raise ValueError("FAT2 wrench relative tolerance must lie in [0,1]")
        if self.wrench_consistency_absolute_floor_n <= 0.0:
            raise ValueError("FAT2 wrench absolute floor must be positive")
        if (
            isinstance(self.wrench_consistency_window_steps, bool)
            or not isinstance(self.wrench_consistency_window_steps, int)
            or self.wrench_consistency_window_steps <= 0
        ):
            raise ValueError("FAT2 wrench consistency window must be a positive integer")


def fat2_reference_angle(
    handle_s: torch.Tensor,
    handle_n: torch.Tensor,
    hand_force_s: torch.Tensor,
    hand_force_n: torch.Tensor,
    robot_mass: torch.Tensor | float,
    com_radius: torch.Tensor | float,
    theta_max: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the full-wrench FAT2 weak torso prior."""

    mass = torch.as_tensor(robot_mass, device=handle_s.device, dtype=handle_s.dtype)
    radius = torch.as_tensor(com_radius, device=handle_s.device, dtype=handle_s.dtype)
    maximum = torch.as_tensor(theta_max, device=handle_s.device, dtype=handle_s.dtype)
    if torch.any(mass <= 0.0) or torch.any(radius <= 0.0):
        raise ValueError("robot_mass and com_radius must be positive")
    if torch.any((maximum <= 0.0) | (maximum >= math.pi / 2.0)):
        raise ValueError("theta_max must lie in (0, pi/2)")
    hand_moment = handle_s * hand_force_n - handle_n * hand_force_s
    ratio = hand_moment / (mass * GRAVITY * radius)
    limit = torch.sin(maximum)
    return torch.asin(torch.clamp(ratio, min=-limit, max=limit))


def sagittal_com_radius(
    robot_com_w: torch.Tensor,
    support_center_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Return support-to-CoM distance in the path tangent/normal plane."""

    if robot_com_w.ndim != 2 or robot_com_w.shape[-1] != 3:
        raise ValueError("robot CoM must have shape [N,3]")
    if any(
        value.shape != robot_com_w.shape
        for value in (support_center_w, path_tangent_w, path_normal_w)
    ):
        raise ValueError("FAT2 sagittal geometry tensors must have identical shapes")
    offset = robot_com_w - support_center_w
    offset_s = torch.sum(offset * path_tangent_w, dim=-1)
    offset_n = torch.sum(offset * path_normal_w, dim=-1)
    return torch.sqrt(torch.square(offset_s) + torch.square(offset_n))


def adapt_d6_reaction_wrench(
    reaction_wrench: torch.Tensor, *, reaction_is_joint_on_body: bool
) -> torch.Tensor:
    """Apply the D6 sign convention once at the simulator API boundary."""

    if reaction_wrench.shape[-1] != 6:
        raise ValueError("reaction_wrench must end in six force/torque components")
    return reaction_wrench if reaction_is_joint_on_body else -reaction_wrench


def project_hand_wrench_to_slope(
    force_w: torch.Tensor,
    torque_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_normal_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a summed world-frame hand wrench to ``(F_s,F_n,tau_y)``."""

    return (
        torch.sum(force_w * path_tangent_w, dim=-1),
        torch.sum(force_w * path_normal_w, dim=-1),
        torch.sum(torque_w * path_lateral_w, dim=-1),
    )


@dataclass
class ZMPCfg:
    min_ground_reaction: float = MISSING


@dataclass(kw_only=True)
class SupportPolygonCfg:
    foot_half_length: float = MISSING
    foot_half_width: float = MISSING
    foot_center_offset_x: float = MISSING

    def validate(self) -> None:
        if self.foot_half_length <= 0.0 or self.foot_half_width <= 0.0:
            raise ValueError("calibrated foot half dimensions must be positive")
        if not math.isfinite(self.foot_center_offset_x):
            raise ValueError("calibrated foot center offset must be finite")


@dataclass
class ZMPKinematicState:
    tangential_velocity_filter: SecondOrderLowPassState
    normal_velocity_filter: SecondOrderLowPassState
    acceleration_s: torch.Tensor
    acceleration_n: torch.Tensor

    @classmethod
    def initialized(
        cls, velocity_s: torch.Tensor, velocity_n: torch.Tensor
    ) -> "ZMPKinematicState":
        zeros = torch.zeros_like(velocity_s)
        return cls(
            tangential_velocity_filter=SecondOrderLowPassState.initialized(velocity_s),
            normal_velocity_filter=SecondOrderLowPassState.initialized(velocity_n),
            acceleration_s=zeros.clone(),
            acceleration_n=zeros.clone(),
        )

    def reset(
        self,
        velocity_s: torch.Tensor,
        velocity_n: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.tangential_velocity_filter.reset(velocity_s, env_ids)
        self.normal_velocity_filter.reset(velocity_n, env_ids)
        ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
        self.acceleration_s[ids] = 0.0
        self.acceleration_n[ids] = 0.0


def slope_zmp(
    com_s: torch.Tensor,
    com_n: torch.Tensor,
    com_acceleration_s: torch.Tensor,
    com_acceleration_n: torch.Tensor,
    handle_s: torch.Tensor,
    handle_n: torch.Tensor,
    hand_force_s: torch.Tensor,
    hand_force_n: torch.Tensor,
    hand_torque_y: torch.Tensor,
    robot_mass: torch.Tensor | float,
    gamma: torch.Tensor,
    *,
    min_ground_reaction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute slope-frame ZMP including the complete hand wrench."""

    mass = torch.as_tensor(robot_mass, device=com_s.device, dtype=com_s.dtype)
    r_s = mass * (com_acceleration_s + GRAVITY * torch.sin(gamma)) - hand_force_s
    r_n = mass * (com_acceleration_n + GRAVITY * torch.cos(gamma)) - hand_force_n
    valid = r_n > min_ground_reaction
    denominator = torch.where(valid, r_n, torch.ones_like(r_n))
    hand_moment_about_com = (
        (handle_s - com_s) * hand_force_n - (handle_n - com_n) * hand_force_s
    )
    zmp_s = com_s + (
        -com_n * r_s - hand_moment_about_com - hand_torque_y
    ) / denominator
    zmp_s = torch.where(valid, zmp_s, torch.zeros_like(zmp_s))
    return zmp_s, r_s, r_n, valid


def _cross_2d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


def convex_support_margin(
    support_points: torch.Tensor,
    query_point: torch.Tensor,
    point_mask: torch.Tensor | None = None,
    *,
    tolerance: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance from points to batched convex support polygons.

    ``support_points`` may be unordered.  All directed point pairs whose other
    valid points lie to their left form candidate hull half-spaces.  The minimum
    inward distance is positive inside and negative outside the convex hull.
    """

    if support_points.ndim != 3 or support_points.shape[-1] != 2:
        raise ValueError("support_points must have shape [N, K, 2]")
    if query_point.shape != (support_points.shape[0], 2):
        raise ValueError("query_point must have shape [N, 2]")
    num_envs, num_points, _ = support_points.shape
    if point_mask is None:
        point_mask = torch.ones(
            (num_envs, num_points), device=support_points.device, dtype=torch.bool
        )
    if point_mask.shape != (num_envs, num_points):
        raise ValueError("point_mask must have shape [N, K]")

    starts = support_points[:, :, None, :]  # [N, i, 1, 2]
    edges = support_points[:, None, :, :] - starts  # [N, i, j, 2]
    lengths = torch.linalg.vector_norm(edges, dim=-1)
    vectors_to_points = support_points[:, None, None, :, :] - starts[:, :, :, None, :]
    side = _cross_2d(edges[:, :, :, None, :], vectors_to_points)
    other_valid = point_mask[:, None, None, :]
    all_left = torch.all((side >= -tolerance) | ~other_valid, dim=-1)
    endpoints_valid = point_mask[:, :, None] & point_mask[:, None, :]
    candidate = endpoints_valid & (lengths > tolerance) & all_left

    vector_to_query = query_point[:, None, None, :] - starts
    distances = _cross_2d(edges, vector_to_query) / torch.clamp(lengths, min=tolerance)
    infinity = torch.full_like(distances, torch.inf)
    margin = torch.min(torch.where(candidate, distances, infinity), dim=-1).values
    margin = torch.min(margin, dim=-1).values

    # At least three non-collinear valid points are required for a polygon.
    has_three_points = torch.sum(point_mask, dim=-1) >= 3
    area_witness = torch.amax(torch.abs(side), dim=(-1, -2, -3)) > tolerance
    valid = has_three_points & area_witness & torch.isfinite(margin)
    margin = torch.where(valid, margin, torch.zeros_like(margin))
    return margin, valid


def foot_support_polygon(
    foot_position_w: torch.Tensor,
    foot_quaternion_wxyz: torch.Tensor,
    foot_contact: torch.Tensor,
    path_origin_w: torch.Tensor,
    path_tangent_w: torch.Tensor,
    path_lateral_w: torch.Tensor,
    *,
    foot_half_length: float,
    foot_half_width: float,
    foot_center_offset_x: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return actual-pose foot corners, contact mask, and support center."""

    if foot_position_w.ndim != 3 or foot_position_w.shape[1:] != (2, 3):
        raise ValueError("foot_position_w must have shape [N,2,3]")
    if foot_quaternion_wxyz.shape != (foot_position_w.shape[0], 2, 4):
        raise ValueError("foot quaternion must have shape [N,2,4]")
    if foot_contact.shape != (foot_position_w.shape[0], 2):
        raise ValueError("foot_contact must have shape [N,2]")
    if foot_half_length <= 0.0 or foot_half_width <= 0.0:
        raise ValueError("foot half dimensions must be positive")
    if not math.isfinite(foot_center_offset_x):
        raise ValueError("foot center offset must be finite")
    local_corners = torch.tensor(
        (
            (foot_center_offset_x + foot_half_length, foot_half_width, 0.0),
            (foot_center_offset_x - foot_half_length, foot_half_width, 0.0),
            (foot_center_offset_x - foot_half_length, -foot_half_width, 0.0),
            (foot_center_offset_x + foot_half_length, -foot_half_width, 0.0),
        ),
        device=foot_position_w.device,
        dtype=foot_position_w.dtype,
    )
    local_corners = local_corners.view(1, 1, 4, 3).expand(
        foot_position_w.shape[0], 2, -1, -1
    )
    world_corners = foot_position_w[:, :, None, :] + quat_apply_wxyz(
        foot_quaternion_wxyz[:, :, None, :].expand(-1, -1, 4, -1), local_corners
    )
    relative = world_corners - path_origin_w[:, None, None, :]
    corner_s = torch.sum(relative * path_tangent_w[:, None, None, :], dim=-1)
    corner_y = torch.sum(relative * path_lateral_w[:, None, None, :], dim=-1)
    points = torch.stack((corner_s, corner_y), dim=-1).reshape(-1, 8, 2)
    point_mask = foot_contact[:, :, None].expand(-1, -1, 4).reshape(-1, 8)
    contact_count = torch.sum(foot_contact, dim=-1, keepdim=True)
    local_center = torch.zeros_like(foot_position_w)
    local_center[..., 0] = foot_center_offset_x
    foot_center_w = foot_position_w + quat_apply_wxyz(foot_quaternion_wxyz, local_center)
    support_center = torch.sum(
        foot_center_w * foot_contact[..., None].to(foot_position_w.dtype), dim=1
    ) / torch.clamp(contact_count, min=1).to(foot_position_w.dtype)
    support_center = torch.where(
        (contact_count > 0), support_center, torch.zeros_like(support_center)
    )
    return points, point_mask, support_center


def update_analytic_rickshaw_force(
    env: Any,
    cfg: AnalyticForceCfg,
    *,
    cart_speed_s: torch.Tensor | None = None,
    pitch: torch.Tensor | None = None,
    geometry_sn: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Policy-step manager adapter for filtered analytic ``T_s/T_n``."""

    if not hasattr(env, "rickshaw_mass_properties"):
        raise RuntimeError("payload reset must install env.rickshaw_mass_properties")
    cart = env.scene["rickshaw"]
    if cart_speed_s is None:
        cart_speed_s = torch.sum(
            cart.data.root_lin_vel_w * env.path_tangent_w, dim=-1
        )
    if pitch is None:
        pitch = rickshaw_pitch_from_quaternion(
            cart.data.root_quat_w, env.path_tangent_w, env.path_normal_w
        )
    if geometry_sn is None:
        geometry_sn = actual_rickshaw_geometry_in_slope_frame(env)
    handle_sn, com_sn = geometry_sn
    update_analytic_handle_force_state(
        env.analytic_force_state,
        cart_speed_s,
        pitch,
        env.gamma,
        env.c_rr,
        env.rickshaw_state.wheel_normal_force,
        env.rickshaw_mass_properties,
        env.step_dt,
        cfg,
        handle_from_axle_sn=handle_sn,
        com_from_axle_sn=com_sn,
    )


def update_support_polygon(env: Any, cfg: SupportPolygonCfg) -> None:
    """Read actual foot poses/contact state and build the current convex support."""

    cfg.validate()
    robot = env.scene["robot"]
    contact_sensor = env.scene["robot_contacts"]
    if not hasattr(contact_sensor.data, "current_contact_time"):
        raise AttributeError("support polygon requires contact sensor current_contact_time")
    foot_contact = contact_sensor.data.current_contact_time[:, env.foot_sensor_ids] > 0.0
    points, mask, center = foot_support_polygon(
        robot.data.body_pos_w[:, env.foot_body_ids],
        robot.data.body_quat_w[:, env.foot_body_ids],
        foot_contact,
        env.scene.terrain.env_origins,
        env.path_tangent_w,
        env.path_lateral_w,
        foot_half_length=cfg.foot_half_length,
        foot_half_width=cfg.foot_half_width,
        foot_center_offset_x=cfg.foot_center_offset_x,
    )
    env.stability_state.support_points_sy[:] = points
    env.stability_state.support_point_mask[:] = mask
    env.stability_state.support_center_w[:] = center


def update_fat2_reference(
    env: Any,
    cfg: FAT2Cfg,
    *,
    robot_kinematics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    hitch_w: torch.Tensor | None = None,
) -> None:
    """Update the weak FAT2 reference from analytic cart force balance."""

    cfg.validate()
    cart = env.scene["rickshaw"]
    if hitch_w is None:
        hitch_w = torch.mean(cart.data.body_pos_w[:, env.hitch_body_ids], dim=1)
    support_center = env.stability_state.support_center_w
    handle_from_support = hitch_w - support_center
    handle_s = torch.sum(handle_from_support * env.path_tangent_w, dim=-1)
    handle_n = torch.sum(handle_from_support * env.path_normal_w, dim=-1)
    analytic = env.analytic_force_state
    consistency_state = getattr(env, "fat2_wrench_consistency_state", None)
    if consistency_state is None:
        consistency_state = WrenchConsistencyState.zeros(
            env.num_envs,
            cfg.wrench_consistency_window_steps,
            device=env.device,
            dtype=analytic.t_s.dtype,
        )
        env.fat2_wrench_consistency_state = consistency_state
    elif consistency_state.window_steps != cfg.wrench_consistency_window_steps:
        raise RuntimeError("FAT2 wrench consistency window changed after initialization")
    measured_force_on_cart_w = -env.rickshaw_state.hand_force_w
    measured_force_sn = torch.stack(
        (
            torch.sum(measured_force_on_cart_w * env.path_tangent_w, dim=-1),
            torch.sum(measured_force_on_cart_w * env.path_normal_w, dim=-1),
        ),
        dim=-1,
    )
    wrench_consistent, wrench_relative_error, filtered_analytic_force_sn = (
        update_wrench_consistency_state(
            consistency_state,
            torch.stack((analytic.t_s, analytic.t_n), dim=-1),
            measured_force_sn,
            analytic.valid & env.cart_interaction_wrench_valid,
            relative_tolerance=cfg.wrench_consistency_relative_tolerance,
            absolute_floor_n=cfg.wrench_consistency_absolute_floor_n,
        )
    )
    env.stability_state.fat_wrench_consistent[:] = wrench_consistent
    env.stability_state.fat_wrench_relative_error[:] = wrench_relative_error
    if robot_kinematics is None:
        robot_kinematics = robot_system_mass_kinematics(env)
    robot_com_w, _, robot_mass = robot_kinematics
    current_com_radius = sagittal_com_radius(
        robot_com_w,
        support_center,
        env.path_tangent_w,
        env.path_normal_w,
    )
    radius_state = getattr(env, "fat2_com_radius_state", None)
    if radius_state is None:
        radius_state = FAT2ComRadiusState.initialized(
            env.num_envs,
            cfg.wrench_consistency_window_steps,
            cfg.com_radius,
            device=env.device,
            dtype=current_com_radius.dtype,
        )
        env.fat2_com_radius_state = radius_state
        env.fat_com_radius = radius_state.filtered_radius
        env.fat_com_radius_raw = current_com_radius.clone()
    elif radius_state.window_steps != cfg.wrench_consistency_window_steps:
        raise RuntimeError("FAT2 CoM radius window changed after initialization")
    env.fat_com_radius_raw[:] = current_com_radius
    radius_min, radius_max = cfg.com_radius_bounds
    radius_state.update(
        current_com_radius,
        torch.any(env.stability_state.support_point_mask, dim=-1),
        minimum=radius_min,
        maximum=radius_max,
    )
    # T_s/T_n are robot-on-cart; FAT2 needs the cart-on-robot force.  Use the
    # same low-frequency window as the gate so the weak torso prior does not
    # chase gait-impact force peaks.
    hand_force_s = -filtered_analytic_force_sn[:, 0]
    hand_force_n = -filtered_analytic_force_sn[:, 1]
    theta = fat2_reference_angle(
        handle_s,
        handle_n,
        hand_force_s,
        hand_force_n,
        robot_mass,
        env.fat_com_radius,
        cfg.theta_max,
    )
    env.stability_state.theta_fat[:] = theta
    env.stability_state.fat_valid[:] = (
        analytic.valid
        & wrench_consistent
        & torch.any(env.stability_state.support_point_mask, dim=-1)
    )
    robot = env.scene["robot"]
    env.stability_state.torso_pitch[:] = torso_pitch_from_world_vertical(
        robot.data.body_quat_w[:, env.torso_body_id],
        env.path_tangent_w,
    )


def update_zmp_stability(
    env: Any,
    cfg: ZMPCfg,
    *,
    robot_kinematics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    hitch_w: torch.Tensor | None = None,
) -> None:
    """Update measured-handle-force ZMP and its signed support-polygon margin."""

    if cfg.min_ground_reaction <= 0.0:
        raise ValueError("min_ground_reaction must come from feasibility validation")
    if robot_kinematics is None:
        robot_kinematics = robot_system_mass_kinematics(env)
    robot_com_w, robot_com_velocity_w, robot_mass = robot_kinematics
    origin = env.scene.terrain.env_origins
    com_relative = robot_com_w - origin
    com_s = torch.sum(com_relative * env.path_tangent_w, dim=-1)
    com_n = torch.sum(com_relative * env.path_normal_w, dim=-1)
    velocity_s = torch.sum(robot_com_velocity_w * env.path_tangent_w, dim=-1)
    velocity_n = torch.sum(robot_com_velocity_w * env.path_normal_w, dim=-1)
    if not hasattr(env, "zmp_kinematic_state"):
        env.zmp_kinematic_state = ZMPKinematicState.initialized(velocity_s, velocity_n)
    state = env.zmp_kinematic_state
    acceleration_s = filtered_first_derivative(
        velocity_s, state.tangential_velocity_filter, env.step_dt, cutoff_hz=20.0
    )
    acceleration_n = filtered_first_derivative(
        velocity_n, state.normal_velocity_filter, env.step_dt, cutoff_hz=20.0
    )
    state.acceleration_s[:] = acceleration_s
    state.acceleration_n[:] = acceleration_n

    cart = env.scene["rickshaw"]
    if hitch_w is None:
        hitch_w = torch.mean(cart.data.body_pos_w[:, env.hitch_body_ids], dim=1)
    hand_force_w = env.rickshaw_state.hand_force_w
    # The crossbar pitch axis is physically free in both D6 joints. Equal
    # half-force bookkeeping at the two symmetric hitch points contributes no
    # sagittal moment about their midpoint.
    hand_torque_w = torch.zeros_like(hand_force_w)
    hand_relative = hitch_w - origin
    handle_s = torch.sum(hand_relative * env.path_tangent_w, dim=-1)
    handle_n = torch.sum(hand_relative * env.path_normal_w, dim=-1)
    force_s, force_n, torque_y = project_hand_wrench_to_slope(
        hand_force_w,
        hand_torque_w,
        env.path_tangent_w,
        env.path_normal_w,
        env.path_lateral_w,
    )
    zmp_s, _, reaction_n, dynamics_valid = slope_zmp(
        com_s,
        com_n,
        acceleration_s,
        acceleration_n,
        handle_s,
        handle_n,
        force_s,
        force_n,
        torque_y,
        robot_mass,
        env.gamma,
        min_ground_reaction=cfg.min_ground_reaction,
    )
    support_relative = env.stability_state.support_center_w - origin
    support_y = torch.sum(support_relative * env.path_lateral_w, dim=-1)
    margin, polygon_valid = convex_support_margin(
        env.stability_state.support_points_sy,
        torch.stack((zmp_s, support_y), dim=-1),
        env.stability_state.support_point_mask,
    )
    valid = dynamics_valid & polygon_valid & env.cart_interaction_wrench_valid
    env.stability_state.zmp_s[:] = zmp_s
    env.stability_state.ground_reaction_normal[:] = reaction_n
    env.stability_state.zmp_margin[:] = torch.where(valid, margin, torch.zeros_like(margin))
    env.stability_state.zmp_valid[:] = valid


def apply_rolling_resistance(env: Any, cfg: RollingResistanceCfg) -> torch.Tensor:
    """Isaac Lab pre-physics adapter that applies wheel-center resistance."""

    cart = env.scene["rickshaw"]
    wheel_velocity = cart.data.body_lin_vel_w[:, env.wheel_body_ids]
    contact_force = env.scene["wheel_contacts"].data.net_forces_w[:, env.wheel_sensor_ids]
    effective_c_rr = env.c_rr if cfg.enabled else torch.zeros_like(env.c_rr)
    force_w, normal_force, _ = rolling_resistance_wrench(
        wheel_velocity,
        contact_force,
        env.path_tangent_w,
        env.path_normal_w,
        effective_c_rr,
        env.rickshaw_state.wheel_normal_force,
        velocity_epsilon=cfg.velocity_epsilon,
        normal_force_filter_hz=cfg.normal_force_filter_hz,
        dt=env.physics_dt,
    )
    env.rickshaw_state.wheel_normal_force[:] = normal_force
    cart.permanent_wrench_composer.set_forces_and_torques(
        force_w,
        torch.zeros_like(force_w),
        body_ids=env.wheel_body_ids,
        is_global=True,
    )
    return force_w


def update_slope_frame(env: Any, env_ids: torch.Tensor | None = None) -> SlopeFrame:
    """Recompute the path frame after a terrain-level change."""

    levels = env.scene.terrain.terrain_levels
    columns = env.scene.terrain.terrain_types
    ids: slice | torch.Tensor = slice(None) if env_ids is None else env_ids
    frame = compute_slope_frame(levels[ids], columns[ids])

    if not hasattr(env, "path_tangent_w"):
        shape = (env.num_envs, 3)
        env.path_tangent_w = torch.zeros(shape, device=env.device)
        env.path_lateral_w = torch.zeros(shape, device=env.device)
        env.path_normal_w = torch.zeros(shape, device=env.device)
        env.gamma = torch.zeros(env.num_envs, device=env.device)
        env.slope = torch.zeros(env.num_envs, device=env.device)
        env.slope_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
    env.path_tangent_w[ids] = frame.tangent_w
    env.path_lateral_w[ids] = frame.lateral_w
    env.path_normal_w[ids] = frame.normal_w
    env.gamma[ids] = frame.gamma
    env.slope[ids] = frame.slope
    env.slope_quat_w[ids] = frame.quaternion_wxyz
    return frame


__all__ = [
    "AnalyticForceCfg",
    "AnalyticHandleForceState",
    "CartInteractionWrenchState",
    "FAT2Cfg",
    "FAT2ComRadiusState",
    "GRAVITY",
    "RickshawMassProperties",
    "RollingResistanceCfg",
    "SecondOrderLowPassState",
    "SlopeFrame",
    "SpeedReferenceCfg",
    "SpeedReferenceState",
    "SupportPolygonCfg",
    "WrenchConsistencyState",
    "ZMPCfg",
    "ZMPKinematicState",
    "adapt_d6_reaction_wrench",
    "actual_rickshaw_geometry_in_slope_frame",
    "accumulate_cart_interaction_wrench",
    "analytic_handle_force",
    "apply_rolling_resistance",
    "articulation_center_of_mass",
    "combine_mass_properties",
    "configure_rolling_resistance",
    "cart_system_mass_kinematics",
    "cart_ground_contact_force_w",
    "compute_slope_frame",
    "convex_support_margin",
    "effective_cart_mass",
    "effective_wheel_damping",
    "fat2_reference_angle",
    "filtered_first_derivative",
    "filtered_second_derivative",
    "foot_support_polygon",
    "low_pass",
    "parallel_axis_inertia",
    "project_hand_wrench_to_slope",
    "quat_apply_wxyz",
    "rickshaw_pitch_from_quaternion",
    "robot_system_mass_kinematics",
    "rolling_resistance_wrench",
    "sagittal_com_radius",
    "slope_zmp",
    "torso_tilt_from_slope_normal",
    "torso_pitch_from_world_vertical",
    "update_analytic_handle_force_state",
    "update_analytic_rickshaw_force",
    "update_cart_interaction_wrench",
    "update_fat2_reference",
    "update_slope_frame",
    "update_speed_reference",
    "update_support_polygon",
    "update_wrench_consistency_state",
    "update_zmp_stability",
]
