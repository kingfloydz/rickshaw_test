"""CPU-only contracts for epoch-fixed physics and privileged observations."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import (
    events as events_module,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    DOMAIN_PARAMETER_NAMES,
    DomainRandomizationCfg,
    effective_cart_mass_com_bounds,
    sample_domain_parameters,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.observations import (
    CRITIC_PRIVILEGED_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    TEACHER_DYNAMIC_FEATURE_NAMES,
    TEACHER_STATIC_DIM,
    TEACHER_STATIC_DOMAIN_DIM,
    TEACHER_STATIC_FEATURE_NAMES,
    ObservationHistoryState,
    critic_privileged_state,
    dynamic_privileged_observation,
    normalize_features,
    teacher_dynamic_history,
    teacher_static,
)


def _domain_cfg(*, enabled: bool = True) -> DomainRandomizationCfg:
    ranges = {
        "torso.mass_delta": (-1.0, 3.0),
        "payload.mass": (-3.0, 3.0),
        "payload.com.x": (-0.5, 0.7),
        "payload.com.y": (-0.3, 0.3),
        "payload.com.z": (0.2, 1.2),
        "rolling_resistance.c_rr": (0.01, 0.03),
        "terrain.friction": (0.6, 1.1),
        "wheel.left_damping": (0.1, 0.3),
        "wheel.right_damping": (0.2, 0.6),
    }
    nominal = {name: 0.5 * (low + high) for name, (low, high) in ranges.items()}
    return DomainRandomizationCfg(
        enabled=enabled,
        ranges=ranges,
        nominal=nominal,
        calibration={},
    )


def test_privileged_feature_schemas_are_explicit_and_fixed() -> None:
    assert set(_domain_cfg().ranges) == set(DOMAIN_PARAMETER_NAMES)
    assert len(TEACHER_STATIC_FEATURE_NAMES) == TEACHER_STATIC_DIM == 10
    assert len(TEACHER_DYNAMIC_FEATURE_NAMES) == TEACHER_DYNAMIC_DIM == 21
    assert CRITIC_PRIVILEGED_DIM == TEACHER_STATIC_DIM + TEACHER_DYNAMIC_DIM + 3 == 34
    assert TEACHER_STATIC_FEATURE_NAMES == (
        "robot.torso_mass",
        "cart.total_mass",
        "cart.com.x",
        "cart.com.y",
        "cart.com.z",
        "rolling_resistance.c_rr",
        "terrain.friction",
        "wheel.left_damping",
        "wheel.right_damping",
        "terrain.slope",
    )


def test_shape_probe_is_explicit_and_missing_runtime_state_fails() -> None:
    probe = SimpleNamespace(num_envs=3, device="cpu")
    assert teacher_static(probe).shape == (3, TEACHER_STATIC_DIM)
    assert teacher_dynamic_history(probe).shape == (
        3,
        HISTORY_LENGTH,
        TEACHER_DYNAMIC_DIM,
    )
    assert critic_privileged_state(probe).shape == (3, CRITIC_PRIVILEGED_DIM)

    probe.observation_manager = SimpleNamespace()
    with pytest.raises(RuntimeError, match="before domain initialization"):
        teacher_static(probe)
    with pytest.raises(RuntimeError, match="before MDP startup"):
        teacher_dynamic_history(probe)
    with pytest.raises(RuntimeError, match="before MDP startup"):
        critic_privileged_state(probe)


def test_teacher_static_contains_normalized_effective_physics_in_declared_order() -> (
    None
):
    cfg = _domain_cfg()
    num_envs = 3
    cart_lower, cart_upper = effective_cart_mass_com_bounds(cfg.ranges)
    cart_lower_t = torch.tensor(cart_lower)
    cart_upper_t = torch.tensor(cart_upper)
    effective_cart = torch.stack(
        (cart_lower_t, 0.5 * (cart_lower_t + cart_upper_t), cart_upper_t)
    )
    sampled = {
        name: torch.tensor((low, 0.5 * (low + high), high))
        for name, (low, high) in cfg.ranges.items()
    }
    nominal_torso_mass = 5.0
    effective_torso_mass = nominal_torso_mass + sampled["torso.mass_delta"]
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        effective_cart_mass_com=effective_cart,
        effective_torso_mass=effective_torso_mass,
        torso_body_id=0,
        _default_robot_masses_cpu=torch.full((num_envs, 1), nominal_torso_mass),
        slope=torch.tensor(
            (
                min(SLOPE_GRADIENTS),
                0.5 * (min(SLOPE_GRADIENTS) + max(SLOPE_GRADIENTS)),
                max(SLOPE_GRADIENTS),
            )
        ),
    )

    events_module._update_teacher_static_domain(env, cfg, sampled)

    expected_raw = torch.cat(
        (
            effective_torso_mass[:, None],
            effective_cart,
            sampled["rolling_resistance.c_rr"][:, None],
            sampled["terrain.friction"][:, None],
            torch.stack(
                (sampled["wheel.left_damping"], sampled["wheel.right_damping"]),
                dim=-1,
            ),
        ),
        dim=-1,
    )
    torch.testing.assert_close(env.teacher_static_domain_raw, expected_raw)
    lower, upper = env.teacher_static_domain_bounds
    torch.testing.assert_close(
        env.normalized_teacher_static_domain,
        normalize_features(expected_raw, lower, upper),
    )
    result = teacher_static(env)
    assert result.shape == (num_envs, TEACHER_STATIC_DIM)
    torch.testing.assert_close(result[:, :-1], env.normalized_teacher_static_domain)
    torch.testing.assert_close(result[:, -1], torch.tensor((-1.0, 0.0, 1.0)))


def _dynamic_env() -> SimpleNamespace:
    # Deliberately permute the world basis: [x,y,z] -> [s,l,n] = [y,z,x].
    tangent = torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1)
    lateral = torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1)
    normal = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
    wrench = torch.tensor(
        (
            ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)),
            (
                (101.0, 102.0, 103.0, 104.0, 105.0, 106.0),
                (107.0, 108.0, 109.0, 110.0, 111.0, 112.0),
            ),
        )
    )
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        path_tangent_w=tangent,
        path_lateral_w=lateral,
        path_normal_w=normal,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_lin_vel_w=torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
                )
            ),
            "rickshaw": SimpleNamespace(
                data=SimpleNamespace(
                    root_lin_vel_w=torch.tensor(((7.0, 8.0, 9.0), (10.0, 11.0, 12.0)))
                )
            ),
        },
        rickshaw_state=SimpleNamespace(
            pitch=torch.tensor((0.1, 0.2)),
            wheel_normal_force=torch.tensor(((20.0, 21.0), (22.0, 23.0))),
            d6_truth_wrench_w=wrench,
            d6_wrench_w=-wrench,
            d6_residual=torch.tensor((0.01, 0.02)),
        ),
    )


def test_dynamic_privilege_uses_sln_frame_and_force_then_torque_per_hand() -> None:
    env = _dynamic_env()
    result = dynamic_privileged_observation(env)
    expected = torch.tensor(
        (
            (
                2.0,
                3.0,
                1.0,
                8.0,
                9.0,
                7.0,
                0.1,
                20.0,
                21.0,
                2.0,
                3.0,
                1.0,
                5.0,
                6.0,
                4.0,
                8.0,
                9.0,
                7.0,
                11.0,
                12.0,
                10.0,
            ),
            (
                5.0,
                6.0,
                4.0,
                11.0,
                12.0,
                10.0,
                0.2,
                22.0,
                23.0,
                102.0,
                103.0,
                101.0,
                105.0,
                106.0,
                104.0,
                108.0,
                109.0,
                107.0,
                111.0,
                112.0,
                110.0,
            ),
        )
    )
    assert result.shape == (2, TEACHER_DYNAMIC_DIM)
    torch.testing.assert_close(result, expected)


def test_dynamic_privilege_requires_d6_truth() -> None:
    env = _dynamic_env()
    del env.rickshaw_state.d6_truth_wrench_w

    with pytest.raises(AttributeError, match="d6_truth_wrench_w"):
        dynamic_privileged_observation(env)


def test_dynamic_history_partial_reset_isolated_and_bootstraps_real_frames() -> None:
    state = ObservationHistoryState.zeros(4, observation_dim=TEACHER_DYNAMIC_DIM)
    first = torch.arange(4 * TEACHER_DYNAMIC_DIM, dtype=torch.float32).reshape(4, -1)
    second = first + 1000.0
    state.initialize(first)
    state.advance(second)
    retained_ids = torch.tensor((0, 2))
    reset_ids = torch.tensor((1, 3))
    retained_current = state.current[retained_ids].clone()
    retained_history = state.history[retained_ids].clone()

    state.reset(reset_ids)

    assert state.initialized.tolist() == [True, False, True, False]
    torch.testing.assert_close(state.current[retained_ids], retained_current)
    torch.testing.assert_close(state.history[retained_ids], retained_history)
    assert torch.count_nonzero(state.current[reset_ids]) == 0
    assert torch.count_nonzero(state.history[reset_ids]) == 0

    reset_frame = second[reset_ids] + 500.0
    state.initialize(reset_frame, reset_ids)
    env = SimpleNamespace(num_envs=4, teacher_dynamic_history_state=state)
    history = teacher_dynamic_history(env)
    assert history.shape == (4, HISTORY_LENGTH, TEACHER_DYNAMIC_DIM)
    torch.testing.assert_close(
        history[reset_ids],
        reset_frame[:, None, :].expand(-1, HISTORY_LENGTH, -1),
    )
    torch.testing.assert_close(history[retained_ids], retained_history)


def test_critic_is_exact_static_dynamic_and_three_diagnostics() -> None:
    num_envs = 2
    dynamic = torch.arange(num_envs * TEACHER_DYNAMIC_DIM, dtype=torch.float32).reshape(
        num_envs, -1
    )
    history_state = ObservationHistoryState.zeros(
        num_envs,
        observation_dim=TEACHER_DYNAMIC_DIM,
        history_enabled=False,
    )
    history_state.initialize(dynamic)
    static_domain = torch.linspace(
        -1.0, 1.0, num_envs * TEACHER_STATIC_DOMAIN_DIM
    ).reshape(num_envs, -1)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        slope=torch.tensor((min(SLOPE_GRADIENTS), max(SLOPE_GRADIENTS))),
        normalized_teacher_static_domain=static_domain,
        teacher_dynamic_history_state=history_state,
        rickshaw_state=SimpleNamespace(d6_residual=torch.tensor((0.1, 0.2))),
        stability_state=SimpleNamespace(zmp_margin=torch.tensor((0.3, 0.4))),
        analytic_force_state=SimpleNamespace(a_s=torch.tensor((0.5, 0.6))),
    )
    expected = torch.cat(
        (
            teacher_static(env),
            dynamic,
            env.rickshaw_state.d6_residual[:, None],
            env.stability_state.zmp_margin[:, None],
            env.analytic_force_state.a_s[:, None],
        ),
        dim=-1,
    )

    result = critic_privileged_state(env, expected_dim=CRITIC_PRIVILEGED_DIM)

    assert result.shape == (num_envs, CRITIC_PRIVILEGED_DIM)
    torch.testing.assert_close(result, expected)


def _sample_startup(cfg: DomainRandomizationCfg, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return sample_domain_parameters(cfg, 8, device="cpu", generator=generator)


def test_startup_sampling_is_deterministic_and_nominal_mode_ignores_seed() -> None:
    cfg = _domain_cfg()
    first_values = _sample_startup(cfg, 1234)
    repeated_values = _sample_startup(cfg, 1234)
    next_values = _sample_startup(cfg, 1235)

    for name in first_values:
        torch.testing.assert_close(
            first_values[name], repeated_values[name], rtol=0.0, atol=0.0
        )
    assert any(
        not torch.equal(first_values[name], next_values[name]) for name in first_values
    )

    fixed = replace(cfg, enabled=False)
    fixed_a = _sample_startup(fixed, 1)
    fixed_b = _sample_startup(fixed, 999)
    for name, value in fixed_a.items():
        torch.testing.assert_close(value, fixed_b[name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(value, torch.full_like(value, fixed.nominal[name]))
