"""Pure regression tests for curriculum, domain physics, and safety contracts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp import events as events_module
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actuation import (
    actuator_effort_limits,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.curricula import (
    CurriculumRuntimeState,
    CurriculumScheduleCfg,
    CurriculumStage,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    CommandState,
    DOMAIN_PARAMETER_NAMES,
    DomainRandomizationCfg,
    SpeedCommandSamplingCfg,
    _write_actuator_parameters,
    advance_speed_command_resampling,
    domain_epoch_seed,
    initialize_domain_randomization,
    sample_domain_parameters,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.terminations import (
    IMMEDIATE_CAUSES,
    PERSISTENT_CAUSES,
    PersistentSafetyCfg,
    PersistentTerminationState,
    TERMINATION_CAUSES,
    TerminationCauseState,
    finite_tensor_violation,
    persistent_condition_matrix,
)


def test_startup_event_signature_accepts_event_manager_env_ids() -> None:
    parameters = tuple(inspect.signature(initialize_domain_randomization).parameters)
    assert parameters[:3] == ("env", "env_ids", "cfg")


def _domain_cfg(*, enabled: bool = True) -> DomainRandomizationCfg:
    ranges = {
        "payload.mass": (0.0, 1.0),
        "payload.com.x": (0.3, 0.9),
        "payload.com.y": (-0.15, 0.15),
        "payload.com.z": (0.45, 0.95),
        "rolling_resistance.c_rr": (0.01, 0.03),
        "terrain.friction": (0.6, 1.2),
        "wheel.left_damping": (0.015, 0.025),
        "wheel.right_damping": (0.015, 0.025),
        "motor.strength": (0.9, 1.1),
        "control.delay": (0.0, 0.04),
        "observation.delay": (0.0, 0.04),
        "joint.model_error": (-0.05, 0.05),
    }
    nominal = {
        "payload.mass": 0.0,
        "payload.com.x": 0.6,
        "payload.com.y": 0.0,
        "payload.com.z": 0.7,
        "rolling_resistance.c_rr": 0.02,
        "terrain.friction": 1.0,
        "wheel.left_damping": 0.02,
        "wheel.right_damping": 0.02,
        "motor.strength": 1.0,
        "control.delay": 0.0,
        "observation.delay": 0.0,
        "joint.model_error": 0.0,
    }
    return DomainRandomizationCfg(
        enabled=enabled,
        ranges=ranges,
        nominal=nominal,
        calibration={},
        curriculum=CurriculumScheduleCfg(),
    )


def test_domain_schema_is_11_scalars_plus_per_joint_error() -> None:
    cfg = _domain_cfg()
    cfg.validate()
    assert set(cfg.ranges) == set(DOMAIN_PARAMETER_NAMES)
    assert not any(name.startswith("d6.") for name in cfg.ranges)
    generator = torch.Generator().manual_seed(domain_epoch_seed(7, 3))
    values, joint_error = sample_domain_parameters(cfg, 8, generator=generator)
    assert len(values) == 11
    assert joint_error.shape == (8, 29)
    assert torch.any(joint_error[:, 0] != joint_error[:, 1])


def test_disabled_domain_uses_nominal_values_and_never_changes_by_epoch() -> None:
    cfg = _domain_cfg(enabled=False)
    first, first_error = sample_domain_parameters(cfg, 3)
    second, second_error = sample_domain_parameters(cfg, 3)
    for name in first:
        torch.testing.assert_close(first[name], second[name])
        assert torch.all(first[name] == cfg.nominal[name])
    assert torch.all(first_error == 0.0)
    torch.testing.assert_close(first_error, second_error)


def test_domain_epoch_seed_supports_resume_without_sampling_prior_epochs() -> None:
    cfg = _domain_cfg()
    direct = torch.Generator().manual_seed(domain_epoch_seed(123, 4))
    resumed = torch.Generator().manual_seed(domain_epoch_seed(123, 4))
    direct_values, direct_error = sample_domain_parameters(cfg, 5, generator=direct)
    resumed_values, resumed_error = sample_domain_parameters(cfg, 5, generator=resumed)
    for name in direct_values:
        torch.testing.assert_close(direct_values[name], resumed_values[name])
    torch.testing.assert_close(direct_error, resumed_error)


def test_domain_iteration_refreshes_once_per_epoch_and_supports_direct_resume(
    monkeypatch,
) -> None:
    cfg = _domain_cfg()
    calls: list[int] = []

    monkeypatch.setattr(events_module, "install_balanced_slope_assignment", lambda _env: None)

    def apply_epoch(env, _cfg, epoch: int) -> None:
        calls.append(epoch)
        env.domain_randomization_epoch = epoch

    monkeypatch.setattr(events_module, "_apply_domain_epoch", apply_epoch)
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        step_dt=0.02,
        slope=torch.zeros(4),
        cfg=SimpleNamespace(seed=17),
        scene=SimpleNamespace(
            terrain=SimpleNamespace(terrain_types=torch.arange(4, dtype=torch.long))
        ),
    )
    initialize_domain_randomization(env, None, cfg)

    assert calls == [0]
    assert env.set_domain_randomization_iteration(199) is False
    assert calls == [0]
    assert env.set_domain_randomization_iteration(200) is True
    assert calls == [0, 1]
    assert env.set_domain_randomization_iteration(200) is False
    assert calls == [0, 1]
    assert env.set_domain_randomization_iteration(600) is True
    assert calls == [0, 1, 3]
    with pytest.raises(ValueError, match="backwards"):
        env.set_domain_randomization_iteration(400)


def test_curriculum_switches_only_reset_environments_to_training() -> None:
    strata = torch.arange(30)
    state = CurriculumRuntimeState.create(
        strata, torch.zeros_like(strata), CurriculumScheduleCfg()
    )
    assert state.set_iteration(1999) == CurriculumStage.STATIC_HAND_LOAD
    assert state.set_iteration(2000) == CurriculumStage.TRAINING
    assert torch.all(state.stage_per_environment() == int(CurriculumStage.STATIC_HAND_LOAD))
    reset_ids = torch.tensor([0, 3, 7])
    state.activate(reset_ids)
    assert torch.all(state.stage_per_environment()[reset_ids] == int(CurriculumStage.TRAINING))


def test_nominal_actuator_domain_preserves_configured_gains_and_binds_limits() -> None:
    class FakeRobot:
        def __init__(self) -> None:
            self.num_joints = 2
            self.data = SimpleNamespace(
                joint_stiffness=torch.tensor([[0.0, 20.0], [0.0, 20.0]]),
                joint_damping=torch.tensor([[0.0, 2.0], [0.0, 2.0]]),
                joint_effort_limits=torch.tensor([[1.0e9, 40.0], [1.0e9, 40.0]]),
            )
            self.actuators = {
                "explicit": SimpleNamespace(
                    joint_indices=torch.tensor([0]),
                    stiffness=torch.full((2, 1), 100.0),
                    damping=torch.full((2, 1), 10.0),
                    effort_limit=torch.full((2, 1), 50.0),
                    _saturation_effort=torch.full((2, 1), 80.0),
                ),
                "implicit": SimpleNamespace(
                    joint_indices=torch.tensor([1]),
                    stiffness=torch.full((2, 1), 20.0),
                    damping=torch.full((2, 1), 2.0),
                    effort_limit=torch.full((2, 1), 40.0),
                ),
            }

        def _write(self, name, value, joint_ids, env_ids) -> None:
            getattr(self.data, name)[env_ids[:, None], joint_ids] = value

        def write_joint_stiffness_to_sim(self, value, *, joint_ids, env_ids) -> None:
            self._write("joint_stiffness", value, joint_ids, env_ids)

        def write_joint_damping_to_sim(self, value, *, joint_ids, env_ids) -> None:
            self._write("joint_damping", value, joint_ids, env_ids)

        def write_joint_effort_limit_to_sim(self, value, *, joint_ids, env_ids) -> None:
            self._write("joint_effort_limits", value, joint_ids, env_ids)

    robot = FakeRobot()
    env = SimpleNamespace(
        scene={"robot": robot},
        policy_joint_ids=[0, 1],
        num_envs=2,
        device="cpu",
    )
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    _write_actuator_parameters(
        env,
        env_ids,
        torch.ones(2),
        torch.zeros((2, 2)),
    )

    assert torch.all(robot.data.joint_stiffness[:, 0] == 0.0)
    assert torch.all(robot.data.joint_stiffness[:, 1] == 20.0)
    assert torch.all(robot.actuators["explicit"].stiffness[:, 0] == 100.0)
    assert torch.all(robot.actuators["explicit"]._saturation_effort[:, 0] == 80.0)
    assert torch.allclose(
        robot.data.joint_effort_limits,
        torch.tensor([[50.0, 40.0], [50.0, 40.0]]),
    )
    assert torch.allclose(
        actuator_effort_limits(robot, [0, 1]),
        robot.data.joint_effort_limits,
    )


def test_actuator_effort_limits_never_uses_permissive_physx_limit() -> None:
    robot = SimpleNamespace(
        num_joints=2,
        data=SimpleNamespace(joint_effort_limits=torch.full((2, 2), 1.0e9)),
        actuators={
            "motor": SimpleNamespace(
                joint_indices=slice(0, 2),
                effort_limit=torch.tensor([[50.0, 25.0], [45.0, 22.5]]),
            )
        },
    )
    limits = actuator_effort_limits(robot, torch.tensor([1, 0]))
    assert torch.equal(limits, torch.tensor([[25.0, 50.0], [22.5, 45.0]]))
    assert torch.all(limits < robot.data.joint_effort_limits[:, [1, 0]])


def test_speed_command_resamples_on_the_ten_second_timer() -> None:
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        command_state=CommandState.zeros(2),
    )
    env.command_state.resampling_elapsed_s[:] = torch.tensor([9.98, 9.96])
    cfg = SpeedCommandSamplingCfg(standing_fraction=0.0)
    assert advance_speed_command_resampling(env, cfg).tolist() == [0]
    assert torch.allclose(
        env.command_state.resampling_elapsed_s, torch.tensor([0.0, 9.98])
    )
    assert advance_speed_command_resampling(env, cfg).tolist() == [1]


def test_arm_hardware_limit_is_a_strict_ten_step_persistent_gate() -> None:
    zeros = torch.zeros(2)
    cfg = PersistentSafetyCfg(
        torso_tilt_max=0.5,
        hitch_height_bounds=(0.65, 0.85),
        rickshaw_pitch_bounds=(0.15, 0.45),
        lateral_corridor=0.3,
        heading_envelope=0.3,
        overspeed_margin=0.25,
        arm_torque_limit=0.9,
    )
    violations = persistent_condition_matrix(
        torch.full((2,), 0.7),
        zeros,
        torch.full((2,), 0.75),
        torch.full((2,), 0.3),
        zeros,
        zeros,
        zeros,
        zeros,
        torch.tensor([[0.9], [0.9001]]),
        torch.full((2,), 0.02),
        torch.ones(2, dtype=torch.bool),
        cfg,
    )
    arm_index = PERSISTENT_CAUSES.index("arm_torque")
    assert violations[:, arm_index].tolist() == [False, True]
    state = PersistentTerminationState.zeros(2)
    for _ in range(9):
        assert not torch.any(state.update(violations))
    assert state.update(violations).tolist() == [False, True]


def test_termination_histogram_and_non_finite_detection() -> None:
    state = TerminationCauseState.zeros(2)
    state.begin_policy_step()
    causes = torch.zeros(2, len(IMMEDIATE_CAUSES), dtype=torch.bool)
    causes[0, 0] = True
    causes[1, 2] = True
    state.record(IMMEDIATE_CAUSES, causes)
    histogram = state.histogram()
    assert set(histogram) == set(TERMINATION_CAUSES)
    assert histogram["non_finite"] == 1
    assert histogram[IMMEDIATE_CAUSES[2]] == 1

    value = torch.zeros(2, 3)
    value[1, 0] = torch.nan
    assert finite_tensor_violation(value).tolist() == [False, True]
