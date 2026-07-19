from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
import torch

import g1_rickshaw_lab.training_contract as contract


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.act_observations: list[Any] = []
        self.update_calls = 0

    def act(self, observation: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        self.act_observations.append(observation)
        return observation

    def update(self, *args: Any, **kwargs: Any) -> int:
        del args, kwargs
        self.update_calls += 1
        return self.update_calls

    def save(self) -> dict[str, Any]:
        return {}


class _FakeEnvironment:
    def __init__(
        self,
        *,
        current_epoch: int,
        enabled: bool = True,
        initial_global_resets: int = 1,
    ) -> None:
        self.current_epoch = current_epoch
        self.enabled = enabled
        self.global_reset_calls = initial_global_resets
        self.episode_reset_calls = 0
        self.domain_iterations: list[int] = []
        self.applied_epochs: list[int] = []
        self.reset_inference_modes: list[bool] = []

    @property
    def unwrapped(self) -> _FakeEnvironment:
        return self

    def set_domain_randomization_iteration(self, iteration: int) -> bool:
        self.domain_iterations.append(iteration)
        if not self.enabled:
            return False
        epoch = iteration // contract.TRAINING_ARTIFACT_INTERVAL
        if epoch == self.current_epoch:
            return False
        self.current_epoch = epoch
        self.applied_epochs.append(epoch)
        return True

    def reset(self) -> tuple[str, dict[str, Any]]:
        self.reset_inference_modes.append(torch.is_inference_mode_enabled())
        self.global_reset_calls += 1
        return f"reset-observation-{self.global_reset_calls}", {}

    def auto_reset_episode(self) -> None:
        self.episode_reset_calls += 1


def _install_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage: str,
    rollout_steps: int,
    curriculum_start: int = 0,
) -> type:
    class FakeOnPolicyRunner:
        def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.env = env
            self.cfg = {"num_steps_per_env": rollout_steps}
            self.alg = _FakeAlgorithm()
            self.current_learning_iteration = 0

        def learn(self, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

        def load(self, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

        def export_policy_to_jit(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def export_policy_to_onnx(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    runners_module = ModuleType("rsl_rl.runners")
    runners_module.OnPolicyRunner = FakeOnPolicyRunner
    package_module = ModuleType("rsl_rl")
    package_module.runners = runners_module
    monkeypatch.setitem(sys.modules, "rsl_rl", package_module)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners_module)
    monkeypatch.setenv("G1_RICKSHAW_RUNNER_HOOK", "1")
    monkeypatch.setenv("G1_RICKSHAW_CHECKPOINT_STAGE", stage)
    monkeypatch.setenv("G1_RICKSHAW_CHECKPOINT_LINEAGE", "{}")
    monkeypatch.delenv("G1_RICKSHAW_TRAINING_CONFIGURATION", raising=False)
    if stage == "s2_student_ppo":
        monkeypatch.setenv(
            "G1_RICKSHAW_CURRICULUM_START_ITERATION", str(curriculum_start)
        )
    else:
        monkeypatch.delenv(
            "G1_RICKSHAW_CURRICULUM_START_ITERATION", raising=False
        )
    monkeypatch.setattr(contract, "require_pinned_rsl_rl", lambda: None)
    monkeypatch.setattr(contract, "collect_runtime_metadata", lambda: object())
    contract.install_runner_hooks_from_environment()
    return FakeOnPolicyRunner


def test_startup_epoch_zero_does_not_duplicate_the_initial_global_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_type = _install_fake_runner(
        monkeypatch, stage="s0_teacher", rollout_steps=48
    )
    env = _FakeEnvironment(current_epoch=0, initial_global_resets=1)

    runner = runner_type(env)

    assert env.domain_iterations == [0]
    assert env.applied_epochs == []
    assert env.global_reset_calls == 1
    assert runner._g1_pending_reset_observation is None


def test_new_startup_epoch_zero_triggers_exactly_one_global_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_type = _install_fake_runner(
        monkeypatch, stage="s0_teacher", rollout_steps=48
    )
    env = _FakeEnvironment(current_epoch=-1, initial_global_resets=0)

    runner = runner_type(env)

    assert env.domain_iterations == [0]
    assert env.applied_epochs == [0]
    assert env.global_reset_calls == 1
    assert runner._g1_pending_reset_observation == "reset-observation-1"
    runner.alg.act("pre-reset-observation")
    assert runner.alg.act_observations == ["reset-observation-1"]


@pytest.mark.parametrize("rollout_steps", (24, 48, 64))
def test_domain_refresh_occurs_once_at_the_200_baseline_iteration_boundary(
    monkeypatch: pytest.MonkeyPatch,
    rollout_steps: int,
) -> None:
    runner_type = _install_fake_runner(
        monkeypatch, stage="s0_teacher", rollout_steps=rollout_steps
    )
    env = _FakeEnvironment(current_epoch=0)
    runner = runner_type(env)
    boundary_updates = (
        contract.TRAINING_ARTIFACT_INTERVAL * contract.BASELINE_ROLLOUT_STEPS
    ) // rollout_steps

    for _ in range(boundary_updates - 1):
        runner.alg.update()
    assert env.domain_iterations == [0]
    assert env.global_reset_calls == 1

    runner.alg.update()

    assert runner._g1_curriculum_iteration == contract.TRAINING_ARTIFACT_INTERVAL
    assert env.domain_iterations == [0]
    assert env.applied_epochs == []
    assert env.global_reset_calls == 1
    assert runner._g1_pending_domain_refresh_iteration == (
        contract.TRAINING_ARTIFACT_INTERVAL
    )

    with torch.inference_mode():
        runner.alg.act("stale-rollout-observation")
    runner.alg.act("next-live-observation")
    assert env.domain_iterations == [0, contract.TRAINING_ARTIFACT_INTERVAL]
    assert env.applied_epochs == [1]
    assert env.global_reset_calls == 2
    assert env.reset_inference_modes == [True]
    assert runner.alg.act_observations == [
        "reset-observation-2",
        "next-live-observation",
    ]
    assert runner._g1_pending_domain_refresh_iteration is None
    assert runner._g1_pending_reset_observation is None


@pytest.mark.parametrize("rollout_steps", (24, 48, 64))
def test_first_progressive_randomization_refresh_has_equal_policy_step_budget(
    monkeypatch: pytest.MonkeyPatch,
    rollout_steps: int,
) -> None:
    runner_type = _install_fake_runner(
        monkeypatch, stage="s0_teacher", rollout_steps=rollout_steps
    )
    env = _FakeEnvironment(current_epoch=0)
    runner = runner_type(env)
    boundary_iteration = contract.TRAINING_ARTIFACT_INTERVAL
    boundary_updates = boundary_iteration * contract.BASELINE_ROLLOUT_STEPS // rollout_steps

    for _ in range(boundary_updates - 1):
        runner.alg.update()
    assert runner._g1_curriculum_iteration < boundary_iteration

    runner.alg.update()

    assert runner._g1_curriculum_iteration == boundary_iteration
    assert runner._g1_stage_policy_steps == boundary_iteration * contract.BASELINE_ROLLOUT_STEPS


def test_episode_resets_do_not_request_domain_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_type = _install_fake_runner(
        monkeypatch, stage="s0_teacher", rollout_steps=48
    )
    env = _FakeEnvironment(current_epoch=0)
    runner = runner_type(env)

    for _ in range(8):
        env.auto_reset_episode()
        runner.alg.act("post-episode-reset-observation")
    for _ in range(contract.TRAINING_ARTIFACT_INTERVAL - 1):
        runner.alg.update()

    assert env.episode_reset_calls == 8
    assert env.domain_iterations == [0]
    assert env.applied_epochs == []
    assert env.global_reset_calls == 1


def test_disabled_domain_does_not_reset_at_the_refresh_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_type = _install_fake_runner(
        monkeypatch, stage="s0_teacher", rollout_steps=48
    )
    env = _FakeEnvironment(current_epoch=0, enabled=False)
    runner = runner_type(env)

    for _ in range(contract.TRAINING_ARTIFACT_INTERVAL):
        runner.alg.update()

    assert env.domain_iterations == [0]
    with torch.inference_mode():
        runner.alg.act("live-observation")

    assert env.domain_iterations == [0, contract.TRAINING_ARTIFACT_INTERVAL]
    assert env.applied_epochs == []
    assert env.global_reset_calls == 1
    assert runner.alg.act_observations == ["live-observation"]
    assert runner._g1_pending_reset_observation is None


def test_s2_bootstrap_starts_from_the_teacher_curriculum_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curriculum_iteration = 600
    runner_type = _install_fake_runner(
        monkeypatch,
        stage="s2_student_ppo",
        rollout_steps=48,
        curriculum_start=curriculum_iteration,
    )
    env = _FakeEnvironment(current_epoch=curriculum_iteration // 200)
    runner = runner_type(env)
    checkpoint = {
        contract.CHECKPOINT_STAGE_KEY: "s2_bootstrap",
        contract.CHECKPOINT_CURRICULUM_ITERATION_KEY: curriculum_iteration,
        contract.TRAINING_CONFIGURATION_KEY: {},
    }
    _stub_checkpoint_validation(monkeypatch, checkpoint)

    runner.load("bootstrap.pt")

    assert runner._g1_curriculum_start_iteration == curriculum_iteration
    assert runner._g1_curriculum_iteration == curriculum_iteration
    assert runner._g1_training_iterations == 0
    assert runner._g1_stage_policy_steps == 0
    assert env.global_reset_calls == 1


@pytest.mark.parametrize("rollout_steps", (24, 48, 64))
def test_s2_resume_reconstructs_the_stage_start_from_completed_samples(
    monkeypatch: pytest.MonkeyPatch, rollout_steps: int
) -> None:
    stage_start = 600
    completed_updates = 12
    completed_baseline_iterations = (
        completed_updates * rollout_steps // contract.BASELINE_ROLLOUT_STEPS
    )
    curriculum_iteration = stage_start + completed_baseline_iterations
    runner_type = _install_fake_runner(
        monkeypatch,
        stage="s2_student_ppo",
        rollout_steps=rollout_steps,
        curriculum_start=curriculum_iteration,
    )
    env = _FakeEnvironment(current_epoch=curriculum_iteration // 200)
    runner = runner_type(env)
    checkpoint = {
        contract.CHECKPOINT_STAGE_KEY: "s2_student_ppo",
        contract.CHECKPOINT_CURRICULUM_ITERATION_KEY: curriculum_iteration,
        contract.TRAINING_CONFIGURATION_KEY: {},
        "iter": completed_updates - 1,
    }
    _stub_checkpoint_validation(monkeypatch, checkpoint)

    runner.load("resume.pt")

    assert runner._g1_curriculum_start_iteration == stage_start
    assert runner._g1_curriculum_iteration == curriculum_iteration
    assert runner._g1_training_iterations == completed_updates
    assert runner._g1_stage_policy_steps == completed_updates * rollout_steps
    assert runner.current_learning_iteration == completed_updates

    runner.alg.update()
    assert runner._g1_curriculum_iteration == stage_start + (
        (completed_updates + 1) * rollout_steps // contract.BASELINE_ROLLOUT_STEPS
    )


@pytest.mark.parametrize(
    ("stage", "curriculum_start", "completed_updates", "curriculum_iteration"),
    (
        ("s0_teacher", 0, 300, 400),
        ("s2_student_ppo", 600, 150, 800),
    ),
)
def test_resume_refreshes_domain_on_first_inference_mode_action(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    curriculum_start: int,
    completed_updates: int,
    curriculum_iteration: int,
) -> None:
    rollout_steps = 64
    runner_type = _install_fake_runner(
        monkeypatch,
        stage=stage,
        rollout_steps=rollout_steps,
        curriculum_start=curriculum_iteration,
    )
    initial_iteration = 0 if stage == "s0_teacher" else curriculum_iteration
    env = _FakeEnvironment(
        current_epoch=initial_iteration // contract.TRAINING_ARTIFACT_INTERVAL
    )
    runner = runner_type(env)
    env.current_epoch = 0
    checkpoint = {
        contract.CHECKPOINT_STAGE_KEY: stage,
        contract.CHECKPOINT_CURRICULUM_ITERATION_KEY: curriculum_iteration,
        contract.TRAINING_CONFIGURATION_KEY: {},
        "iter": completed_updates - 1,
    }
    _stub_checkpoint_validation(monkeypatch, checkpoint)

    runner.load("resume.pt")

    assert runner._g1_curriculum_start_iteration == curriculum_start
    assert env.domain_iterations == [initial_iteration]
    assert env.global_reset_calls == 1
    with torch.inference_mode():
        runner.alg.act("stale-observation")

    assert env.domain_iterations == [initial_iteration, curriculum_iteration]
    assert env.global_reset_calls == 2
    assert env.reset_inference_modes == [True]
    assert runner.alg.act_observations == ["reset-observation-2"]


def _stub_checkpoint_validation(
    monkeypatch: pytest.MonkeyPatch, checkpoint: dict[str, Any]
) -> None:
    monkeypatch.setattr(contract, "_torch_load", lambda _path: checkpoint)
    monkeypatch.setattr(contract, "validate_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        contract, "validate_training_configuration", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        contract, "validate_guide_training_configuration", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        contract, "validate_student_checkpoint_architecture", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        contract, "validate_teacher_checkpoint_architecture", lambda *args, **kwargs: None
    )
