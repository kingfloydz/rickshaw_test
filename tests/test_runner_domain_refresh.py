"""Runner regressions for fixed domains and stability-reward switching."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import g1_rickshaw_lab.training_contract as contract
from g1_rickshaw_lab.rl.runner import RunnerContext, create_rickshaw_runner_type


class _FakeAlgorithm:
    def __init__(self, latent_dim: int = 16) -> None:
        self.actor = SimpleNamespace(latent_dim=latent_dim)
        self.update_calls = 0

    def update(self) -> int:
        self.update_calls += 1
        return self.update_calls

    def save(self) -> dict[str, Any]:
        return {}


class _RewardManager:
    def __init__(self) -> None:
        self.terms = {
            "fat2_prior_exp": SimpleNamespace(weight=0.2),
            "zmp_margin_barrier": SimpleNamespace(weight=-1.0),
        }

    def get_term_cfg(self, name: str) -> SimpleNamespace:
        return self.terms[name]


class _FakeEnvironment:
    def __init__(self) -> None:
        self.reward_manager = _RewardManager()
        self.global_reset_calls = 1

    @property
    def unwrapped(self) -> "_FakeEnvironment":
        return self


def _install_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stability_reward_curriculum: bool,
    rollout_steps: int = 48,
) -> type:
    class FakeOnPolicyRunner:
        def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.env = env
            self.cfg = {
                "num_steps_per_env": rollout_steps,
                "save_interval": contract.training_artifact_interval(rollout_steps),
            }
            self.alg = _FakeAlgorithm()
            self.logger = SimpleNamespace(lenbuffer=[], save_model=lambda *args: None)
            self.current_learning_iteration = 0

        def learn(
            self, *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

        def save(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def load(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {}

        def export_policy_to_jit(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def export_policy_to_onnx(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    del monkeypatch
    configuration = {
        "training_parameters": {
            "fat2_weight": 0.0,
            "rollout_steps": rollout_steps,
            "latent_dim": 16,
            "history_length": 61,
            "stability_reward_curriculum": stability_reward_curriculum,
        }
    }
    context = RunnerContext(
        stage="s0_teacher",
        training_configuration=configuration,
        metadata=object(),
    )
    return create_rickshaw_runner_type(context, base_runner_type=FakeOnPolicyRunner)


def test_runner_never_resamples_or_resets_fixed_startup_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_type = _install_fake_runner(monkeypatch, stability_reward_curriculum=False)
    env = _FakeEnvironment()
    runner = runner_type(env)

    for _ in range(600):
        runner.alg.update()

    assert env.global_reset_calls == 1
    assert runner._g1_curriculum_iteration == 600


def test_stability_rewards_enable_once_logged_mean_length_exceeds_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_type = _install_fake_runner(monkeypatch, stability_reward_curriculum=True)
    env = _FakeEnvironment()
    runner = runner_type(env)

    assert env.reward_manager.get_term_cfg("fat2_prior_exp").weight == 0.0
    assert env.reward_manager.get_term_cfg("zmp_margin_barrier").weight == 0.0

    runner.logger.lenbuffer[:] = [500.0]
    runner.alg.update()
    assert runner._g1_stability_rewards_active is False

    runner.logger.lenbuffer[:] = [501.0]
    runner.alg.update()
    assert runner._g1_stability_rewards_active is True
    assert env.reward_manager.get_term_cfg("fat2_prior_exp").weight == 0.2
    assert env.reward_manager.get_term_cfg("zmp_margin_barrier").weight == -1.0
    assert env.global_reset_calls == 1
