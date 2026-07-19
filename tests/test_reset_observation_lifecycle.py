"""Pure-Torch regression tests for reset-time policy observations."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.dynamics import SpeedReferenceCfg
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.events import (
    CommandState,
    SpeedCommandSamplingCfg,
    advance_policy_interval,
    bootstrap_reset_observation,
)
from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity.mdp.observations import (
    ACTOR_OBSERVATION_DIM,
    HISTORY_LENGTH,
    TEACHER_DYNAMIC_DIM,
    ObservationHistoryState,
    ObservationDelayState,
)


def _policy_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        command_sampling=SpeedCommandSamplingCfg(
            minimum=0.0,
            maximum=1.0,
            standing_fraction=0.0,
            resampling_time_s=10.0,
        ),
        speed_reference=SpeedReferenceCfg(
            acceleration_limit=0.8,
            jerk_limit=2.5,
            response_time=0.5,
            velocity_tolerance=1.0e-3,
        ),
    )


def _fake_env() -> SimpleNamespace:
    num_envs = 2
    joint_count = 29
    history = ObservationHistoryState.zeros(num_envs)
    dynamic_history = ObservationHistoryState.zeros(
        num_envs, observation_dim=TEACHER_DYNAMIC_DIM
    )
    retained_frame = torch.full((1, ACTOR_OBSERVATION_DIM), 7.0)
    history.initialize(retained_frame, torch.tensor([0]))
    dynamic_history.initialize(
        torch.full((1, TEACHER_DYNAMIC_DIM), 5.0), torch.tensor([0])
    )

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_ang_vel_b=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            root_lin_vel_w=torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            projected_gravity_b=torch.tensor(
                [[0.0, 0.0, -1.0], [0.1, 0.0, -0.995]]
            ),
            joint_pos=torch.stack(
                (torch.zeros(joint_count), torch.linspace(-0.2, 0.2, joint_count))
            ),
            joint_vel=torch.stack(
                (torch.zeros(joint_count), torch.linspace(-1.0, 1.0, joint_count))
            ),
        )
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.02,
        scene={
            "robot": robot,
            "rickshaw": SimpleNamespace(
                data=SimpleNamespace(root_lin_vel_w=torch.zeros((num_envs, 3)))
            ),
        },
        path_tangent_w=torch.tensor([[1.0, 0.0, 0.0]] * num_envs),
        path_lateral_w=torch.tensor([[0.0, 1.0, 0.0]] * num_envs),
        path_normal_w=torch.tensor([[0.0, 0.0, 1.0]] * num_envs),
        rickshaw_state=SimpleNamespace(
            pitch=torch.tensor([0.1, 0.2]),
            wheel_normal_force=torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            d6_truth_wrench_w=torch.zeros((num_envs, 2, 6)),
        ),
        policy_joint_ids=torch.arange(joint_count),
        command_state=CommandState(
            v_sample=torch.tensor([0.4, 1.0]),
            v_ref=torch.tensor([0.2, 0.0]),
            a_ref=torch.tensor([0.1, 0.0]),
            resampling_elapsed_s=torch.tensor([2.0, 0.0]),
        ),
        path_state=SimpleNamespace(
            lateral_error=torch.tensor([0.3, -0.1]),
            heading_error=torch.tensor([0.2, -0.2]),
        ),
        action_state=SimpleNamespace(
            q_ref=torch.zeros((num_envs, joint_count)),
            target=torch.stack(
                (torch.zeros(joint_count), torch.linspace(0.1, 0.3, joint_count))
            ),
        ),
        observation_history_state=history,
        teacher_dynamic_history_state=dynamic_history,
        observation_delay_state=ObservationDelayState.zeros(num_envs, 2),
        observation_delay_steps=torch.tensor([2, 2], dtype=torch.long),
        all_env_ids=torch.arange(num_envs),
        all_env_mask=torch.ones(num_envs, dtype=torch.bool),
    )


def test_explicit_reset_bootstrap_matches_automatic_reset_frame() -> None:
    explicit_env = _fake_env()
    automatic_env = _fake_env()
    cfg = _policy_cfg()
    reset_ids = torch.tensor([1])

    retained_command = torch.stack(
        (
            explicit_env.command_state.v_sample,
            explicit_env.command_state.v_ref,
            explicit_env.command_state.a_ref,
            explicit_env.command_state.resampling_elapsed_s,
        ),
        dim=-1,
    )[0].clone()
    retained_history = explicit_env.observation_history_state.history[0].clone()
    retained_dynamic_history = explicit_env.teacher_dynamic_history_state.history[0].clone()
    bootstrap_reset_observation(explicit_env, reset_ids, cfg)
    advance_policy_interval(automatic_env, None, cfg)

    torch.testing.assert_close(
        explicit_env.observation_history_state.current[reset_ids],
        automatic_env.observation_history_state.current[reset_ids],
    )
    torch.testing.assert_close(
        explicit_env.observation_history_state.history[reset_ids],
        automatic_env.observation_history_state.history[reset_ids],
    )
    torch.testing.assert_close(
        explicit_env.teacher_dynamic_history_state.current[reset_ids],
        automatic_env.teacher_dynamic_history_state.current[reset_ids],
    )
    torch.testing.assert_close(
        explicit_env.teacher_dynamic_history_state.history[reset_ids],
        automatic_env.teacher_dynamic_history_state.history[reset_ids],
    )
    reset_frame = explicit_env.observation_history_state.current[reset_ids]
    torch.testing.assert_close(
        explicit_env.observation_history_state.history[reset_ids],
        reset_frame[:, None, :].expand(-1, HISTORY_LENGTH, -1),
    )
    assert torch.any(reset_frame != 0.0)

    current_command = torch.stack(
        (
            explicit_env.command_state.v_sample,
            explicit_env.command_state.v_ref,
            explicit_env.command_state.a_ref,
            explicit_env.command_state.resampling_elapsed_s,
        ),
        dim=-1,
    )[0]
    torch.testing.assert_close(current_command, retained_command)
    torch.testing.assert_close(
        explicit_env.observation_history_state.history[0], retained_history
    )
    torch.testing.assert_close(
        explicit_env.teacher_dynamic_history_state.history[0], retained_dynamic_history
    )
    dynamic_reset_frame = explicit_env.teacher_dynamic_history_state.current[reset_ids]
    torch.testing.assert_close(
        explicit_env.teacher_dynamic_history_state.history[reset_ids],
        dynamic_reset_frame[:, None, :].expand(-1, HISTORY_LENGTH, -1),
    )
