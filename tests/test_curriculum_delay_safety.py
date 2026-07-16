"""Pure regression tests for curriculum, latency, and safety contracts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.actions import (
    ControlDelayState,
)
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
    RuntimeRandomizationCfg,
    SpeedCommandSamplingCfg,
    _write_actuator_parameters,
    advance_speed_command_resampling,
    initialize_curriculum_runtime,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.observations import (
    ACTOR_OBSERVATION_DIM,
    ObservationDelayState,
    ObservationNoiseCfg,
    assemble_actor_observation,
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
    parameters = tuple(inspect.signature(initialize_curriculum_runtime).parameters)
    assert parameters[:3] == ("env", "env_ids", "cfg")


def _runtime_randomization_cfg(
    *, sample_ranges: bool, payload_range: tuple[float, float], joint_range: tuple[float, float]
) -> RuntimeRandomizationCfg:
    return RuntimeRandomizationCfg(
        ranges={
            "payload.mass": payload_range,
            "joint.model_error": joint_range,
        },
        calibration={},
        nominal_values={
            "payload.mass": payload_range[0],
            "joint.model_error": joint_range[0],
        },
        curriculum=CurriculumScheduleCfg(),
        sample_ranges=sample_ranges,
        teacher_extrinsic_names=("payload.mass",),
    )


def test_replicated_physics_accepts_singleton_scan_ranges() -> None:
    _runtime_randomization_cfg(
        sample_ranges=True,
        payload_range=(0.5, 0.5),
        joint_range=(0.1, 0.1),
    ).validate()


@pytest.mark.parametrize("varying", ("teacher", "joint"))
def test_replicated_physics_rejects_non_singleton_scan_ranges(varying: str) -> None:
    cfg = _runtime_randomization_cfg(
        sample_ranges=True,
        payload_range=(0.5, 0.6) if varying == "teacher" else (0.5, 0.5),
        joint_range=(0.1, 0.2) if varying == "joint" else (0.1, 0.1),
    )
    with pytest.raises(ValueError, match="singleton scan ranges"):
        cfg.validate()


def test_curriculum_is_single_training_stage_for_all_iterations() -> None:
    strata = torch.arange(30)
    state = CurriculumRuntimeState.create(
        strata, torch.zeros_like(strata), CurriculumScheduleCfg()
    )
    for iteration in (0, 999, 1000, 6000):
        assert state.set_iteration(iteration) == CurriculumStage.TRAINING
        assert torch.all(
            state.stage_per_environment() == int(CurriculumStage.TRAINING)
        )


def test_control_and_observation_delay_are_exact_integer_steps() -> None:
    delays = torch.tensor([0, 1, 2], dtype=torch.long)
    control = ControlDelayState.zeros(3, 1, 2)
    assert control.apply(torch.ones(3, 1), delays).squeeze(-1).tolist() == [1.0, 0.0, 0.0]
    assert control.apply(torch.full((3, 1), 2.0), delays).squeeze(-1).tolist() == [2.0, 1.0, 0.0]
    assert control.apply(torch.full((3, 1), 3.0), delays).squeeze(-1).tolist() == [3.0, 2.0, 1.0]

    observation = ObservationDelayState.zeros(3, 2)
    first = torch.ones(3, ACTOR_OBSERVATION_DIM)
    second = torch.full_like(first, 2.0)
    third = torch.full_like(first, 3.0)
    assert torch.all(observation.apply(first, delays) == 1.0)
    delayed_second = observation.apply(second, delays)[:, 0]
    delayed_third = observation.apply(third, delays)[:, 0]
    assert delayed_second.tolist() == [2.0, 1.0, 1.0]
    assert delayed_third.tolist() == [3.0, 2.0, 1.0]


def test_observation_corruption_can_be_disabled_per_environment() -> None:
    batch = 2
    zeros3 = torch.zeros(batch, 3)
    zeros29 = torch.zeros(batch, 29)
    scalar = torch.zeros(batch)
    noise = ObservationNoiseCfg().scaled(torch.tensor([0.0, 1.0]))
    generator = torch.Generator().manual_seed(7)
    result = assemble_actor_observation(
        zeros3,
        zeros3,
        scalar,
        scalar,
        scalar,
        zeros29,
        zeros29,
        zeros29,
        zeros29,
        noise_cfg=noise,
        generator=generator,
    )
    assert torch.all(result[0] == 0.0)
    assert torch.any(result[1] != 0.0)


def test_motor_randomization_updates_explicit_model_without_enabling_solver_pd() -> None:
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
        device="cpu",
    )
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    strength = torch.tensor([0.9, 1.1])
    _write_actuator_parameters(
        env, env_ids, strength, torch.zeros(2, 2)
    )

    assert torch.all(robot.data.joint_stiffness[:, 0] == 0.0)
    assert torch.allclose(robot.data.joint_stiffness[:, 1], torch.tensor([18.0, 22.0]))
    assert torch.allclose(
        robot.actuators["explicit"].stiffness[:, 0], torch.tensor([90.0, 110.0])
    )
    assert torch.allclose(
        robot.actuators["explicit"]._saturation_effort[:, 0],
        torch.tensor([72.0, 88.0]),
    )
    assert torch.allclose(
        robot.data.joint_effort_limits,
        torch.tensor([[45.0, 36.0], [55.0, 44.0]]),
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
