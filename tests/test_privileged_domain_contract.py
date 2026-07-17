"""CPU-only contracts for epoch-fixed physics and privileged observations."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.slope_contract import SLOPE_GRADIENTS
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import events as events_module
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actions import ACTION_DIM
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.curricula import (
    CurriculumScheduleCfg,
    CurriculumStage,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    DOMAIN_PARAMETER_NAMES,
    DomainRandomizationCfg,
    domain_epoch_seed,
    effective_cart_mass_com_bounds,
    initialize_domain_randomization,
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
        "payload.mass": (0.0, 10.0),
        "payload.com.x": (-0.5, 0.7),
        "payload.com.y": (-0.3, 0.3),
        "payload.com.z": (0.2, 1.2),
        "rolling_resistance.c_rr": (0.01, 0.03),
        "terrain.friction": (0.5, 1.5),
        "wheel.left_damping": (0.1, 0.3),
        "wheel.right_damping": (0.2, 0.6),
        "motor.strength": (0.8, 1.2),
        "joint.model_error": (-0.25, 0.25),
        "control.delay": (0.0, 0.04),
        "observation.delay": (0.0, 0.06),
    }
    nominal = {name: 0.5 * (low + high) for name, (low, high) in ranges.items()}
    return DomainRandomizationCfg(
        enabled=enabled,
        ranges=ranges,
        nominal=nominal,
        calibration={},
        curriculum=CurriculumScheduleCfg(),
        refresh_interval_iterations=200,
    )


def test_privileged_feature_schemas_are_explicit_and_fixed() -> None:
    assert set(_domain_cfg().ranges) == set(DOMAIN_PARAMETER_NAMES)
    assert len(TEACHER_STATIC_FEATURE_NAMES) == TEACHER_STATIC_DIM == 40
    assert len(TEACHER_DYNAMIC_FEATURE_NAMES) == TEACHER_DYNAMIC_DIM == 21
    assert CRITIC_PRIVILEGED_DIM == TEACHER_STATIC_DIM + TEACHER_DYNAMIC_DIM + 3 == 64
    assert TEACHER_STATIC_FEATURE_NAMES[:8] == (
        "cart.total_mass",
        "cart.com.x",
        "cart.com.y",
        "cart.com.z",
        "rolling_resistance.c_rr",
        "terrain.friction",
        "wheel.left_damping",
        "wheel.right_damping",
    )
    assert TEACHER_STATIC_FEATURE_NAMES[8:37] == tuple(
        f"actuator.effective_gain.{index}" for index in range(ACTION_DIM)
    )
    assert TEACHER_STATIC_FEATURE_NAMES[-3:] == (
        "control.delay",
        "observation.delay",
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


def test_teacher_static_contains_normalized_effective_physics_in_declared_order() -> None:
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
        if name != "joint.model_error"
    }
    joint_error = torch.stack(
        (
            torch.full((ACTION_DIM,), -0.25),
            torch.linspace(-0.25, 0.25, ACTION_DIM),
            torch.full((ACTION_DIM,), 0.25),
        )
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        effective_cart_mass_com=effective_cart,
        slope=torch.tensor(
            (
                min(SLOPE_GRADIENTS),
                0.5 * (min(SLOPE_GRADIENTS) + max(SLOPE_GRADIENTS)),
                max(SLOPE_GRADIENTS),
            )
        ),
    )

    events_module._update_teacher_static_domain(env, cfg, sampled, joint_error)

    effective_gain = sampled["motor.strength"][:, None] * (1.0 + joint_error)
    expected_raw = torch.cat(
        (
            effective_cart,
            sampled["rolling_resistance.c_rr"][:, None],
            sampled["terrain.friction"][:, None],
            torch.stack(
                (sampled["wheel.left_damping"], sampled["wheel.right_damping"]),
                dim=-1,
            ),
            effective_gain,
            sampled["control.delay"][:, None],
            sampled["observation.delay"][:, None],
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
        curriculum_stage_per_env=torch.tensor(
            (int(CurriculumStage.STATIC_HAND_LOAD), int(CurriculumStage.TRAINING))
        ),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_lin_vel_w=torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))))
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
            d6_wrench_w=wrench,
            d6_residual=torch.tensor((0.01, 0.02)),
        ),
    )


def test_dynamic_privilege_uses_sln_frame_and_force_then_torque_per_hand() -> None:
    env = _dynamic_env()
    result = dynamic_privileged_observation(env)
    expected = torch.tensor(
        (
            (
                2.0, 3.0, 1.0,
                0.0, 0.0, 0.0,
                0.1,
                20.0, 21.0,
                2.0, 3.0, 1.0, 5.0, 6.0, 4.0,
                8.0, 9.0, 7.0, 11.0, 12.0, 10.0,
            ),
            (
                5.0, 6.0, 4.0,
                11.0, 12.0, 10.0,
                0.2,
                22.0, 23.0,
                102.0, 103.0, 101.0, 105.0, 106.0, 104.0,
                108.0, 109.0, 107.0, 111.0, 112.0, 110.0,
            ),
        )
    )
    assert result.shape == (2, TEACHER_DYNAMIC_DIM)
    torch.testing.assert_close(result, expected)


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


def _sample_epoch(cfg: DomainRandomizationCfg, seed: int, epoch: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(domain_epoch_seed(seed, epoch))
    return sample_domain_parameters(cfg, 8, device="cpu", generator=generator)


def test_epoch_sampling_is_deterministic_and_nominal_mode_ignores_seed() -> None:
    cfg = _domain_cfg()
    first_values, first_error = _sample_epoch(cfg, 1234, 7)
    repeated_values, repeated_error = _sample_epoch(cfg, 1234, 7)
    next_values, next_error = _sample_epoch(cfg, 1234, 8)

    for name in first_values:
        torch.testing.assert_close(first_values[name], repeated_values[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_error, repeated_error, rtol=0.0, atol=0.0)
    assert any(not torch.equal(first_values[name], next_values[name]) for name in first_values)
    assert not torch.equal(first_error, next_error)

    fixed = replace(cfg, enabled=False)
    fixed_a, fixed_error_a = _sample_epoch(fixed, 1, 0)
    fixed_b, fixed_error_b = _sample_epoch(fixed, 999, 99)
    for name, value in fixed_a.items():
        torch.testing.assert_close(value, fixed_b[name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(value, torch.full_like(value, fixed.nominal[name]))
    torch.testing.assert_close(fixed_error_a, fixed_error_b, rtol=0.0, atol=0.0)


def test_domain_iteration_repeated_inside_epoch_is_a_strict_noop(monkeypatch) -> None:
    cfg = _domain_cfg()
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        step_dt=0.02,
        slope=torch.zeros(4),
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_types=torch.arange(4))),
        cfg=SimpleNamespace(seed=42),
        extras={},
        applied_epochs=[],
    )
    monkeypatch.setattr(events_module, "install_balanced_slope_assignment", lambda _env: None)

    def fake_apply(target, _cfg, epoch: int) -> None:
        target.applied_epochs.append(epoch)
        target.domain_randomization_epoch = epoch
        target.domain_randomization_initialized = True
        target.normalized_teacher_static_domain = torch.full(
            (target.num_envs, TEACHER_STATIC_DOMAIN_DIM), float(epoch)
        )

    monkeypatch.setattr(events_module, "_apply_domain_epoch", fake_apply)
    initialize_domain_randomization(env, None, cfg)
    epoch_zero = env.normalized_teacher_static_domain.clone()

    assert env.set_domain_randomization_iteration(199) is False
    assert env.applied_epochs == [0]
    torch.testing.assert_close(env.normalized_teacher_static_domain, epoch_zero)

    assert env.set_domain_randomization_iteration(200) is True
    assert env.applied_epochs == [0, 1]
    epoch_one = env.normalized_teacher_static_domain.clone()
    assert env.set_domain_randomization_iteration(200) is False
    assert env.applied_epochs == [0, 1]
    torch.testing.assert_close(env.normalized_teacher_static_domain, epoch_one)
