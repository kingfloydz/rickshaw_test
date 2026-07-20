"""Project-owned RSL-RL runner integration.

The runner is created explicitly by the project launcher.  It keeps checkpoint
provenance, iteration accounting, and the optional stability-reward schedule in
one place without modifying RSL-RL classes globally.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from g1_rickshaw_lab.provenance import (
    CheckpointMetadata,
    atomic_torch_save,
    attach_checkpoint_metadata,
    validate_checkpoint,
)
from g1_rickshaw_lab.reward_profile import reward_weight_overrides_from_configuration
from g1_rickshaw_lab.training_contract import (
    BASELINE_ROLLOUT_STEPS,
    CHECKPOINT_CURRICULUM_ITERATION_KEY,
    CHECKPOINT_LINEAGE_KEY,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STABILITY_REWARDS_ACTIVE_KEY,
    CHECKPOINT_STAGE_KEY,
    TRAINING_CONFIGURATION_KEY,
    checkpoint_stage,
    collect_runtime_metadata,
    s0_remaining_learning_iterations,
    s2_remaining_learning_iterations,
    training_artifact_interval,
    validate_guide_training_configuration,
    validate_student_checkpoint_architecture,
    validate_teacher_checkpoint_architecture,
    validate_training_configuration,
    write_deployment_manifest,
)


@dataclass(frozen=True, slots=True)
class RunnerContext:
    """Immutable run metadata passed directly from a workflow to the runner."""

    stage: str
    lineage: Mapping[str, Any] = field(default_factory=dict)
    training_configuration: Mapping[str, Any] | None = None
    curriculum_start_iteration: int = 0
    metadata: CheckpointMetadata | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"s0_teacher", "s2_student_ppo"}:
            raise ValueError(f"unsupported PPO runner stage: {self.stage!r}")
        if not isinstance(self.lineage, Mapping):
            raise TypeError("runner lineage must be a mapping")
        if isinstance(self.curriculum_start_iteration, bool) or self.curriculum_start_iteration < 0:
            raise ValueError("curriculum_start_iteration must be a non-negative integer")

    @classmethod
    def training(
        cls,
        *,
        stage: str,
        training_configuration: Mapping[str, Any],
        lineage: Mapping[str, Any] | None = None,
        curriculum_start_iteration: int = 0,
        metadata: CheckpointMetadata | None = None,
    ) -> RunnerContext:
        configuration = validate_guide_training_configuration(
            training_configuration,
            expected_stage=stage,
        )
        return cls(
            stage=stage,
            lineage={} if lineage is None else dict(lineage),
            training_configuration=configuration,
            curriculum_start_iteration=curriculum_start_iteration,
            metadata=metadata,
        )

    @classmethod
    def playback(
        cls,
        *,
        stage: str = "s2_student_ppo",
        curriculum_start_iteration: int,
        metadata: CheckpointMetadata | None = None,
    ) -> RunnerContext:
        return cls(
            stage=stage,
            curriculum_start_iteration=curriculum_start_iteration,
            metadata=metadata,
        )


def _torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint root must be a mapping")
    return checkpoint


def _unwrap_env(env: Any) -> Any:
    visited: set[int] = set()
    while id(env) not in visited:
        visited.add(id(env))
        next_env = getattr(env, "unwrapped", None)
        if next_env is None or next_env is env:
            next_env = getattr(env, "env", None)
        if next_env is None or next_env is env:
            break
        env = next_env
    return env


def create_rickshaw_runner_type(
    context: RunnerContext,
    *,
    base_runner_type: type | None = None,
) -> type:
    """Create the concrete runner type for one explicitly configured run."""

    if base_runner_type is None:
        from rsl_rl.runners import OnPolicyRunner

        base_runner_type = OnPolicyRunner

    metadata = context.metadata or collect_runtime_metadata()
    stage = context.stage
    lineage = dict(context.lineage)
    training_configuration = None if context.training_configuration is None else dict(context.training_configuration)

    class RickshawOnPolicyRunner(base_runner_type):
        """RSL-RL runner with the G1 rickshaw checkpoint and curriculum ABI."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._g1_stability_reward_curriculum = bool(
                training_configuration and training_configuration["training_parameters"]["stability_reward_curriculum"]
            )
            if self._g1_stability_reward_curriculum:
                env = _unwrap_env(self.env)
                self._g1_stability_reward_weights = {
                    name: float(env.reward_manager.get_term_cfg(name).weight)
                    for name in ("fat2_prior_exp", "zmp_margin_barrier")
                }
                self._set_stability_rewards(False)
            else:
                self._g1_stability_rewards_active = True

            if training_configuration is not None:
                parameters = training_configuration["training_parameters"]
                configured_steps = int(parameters["rollout_steps"])
                if int(self.cfg["num_steps_per_env"]) != configured_steps:
                    raise RuntimeError("training rollout length differs from the published configuration")
                if int(self.alg.actor.latent_dim) != int(parameters["latent_dim"]):
                    raise RuntimeError("actor latent width differs from the published configuration")
                if int(getattr(self.alg.actor, "history_length", 61)) != int(parameters.get("history_length", 61)):
                    raise RuntimeError("actor history length differs from the published configuration")
                if int(self.cfg["save_interval"]) != training_artifact_interval(configured_steps):
                    raise RuntimeError("checkpoint interval differs from the fixed transition cadence")

            self._g1_training_iterations = 0
            self._g1_curriculum_start_iteration = int(context.curriculum_start_iteration)
            self._g1_stage_policy_steps = 0
            self._g1_curriculum_iteration = self._g1_curriculum_start_iteration
            self._install_algorithm_update_callback()

        def _set_stability_rewards(self, active: bool) -> None:
            env = _unwrap_env(self.env)
            for name, default_weight in self._g1_stability_reward_weights.items():
                env.reward_manager.get_term_cfg(name).weight = default_weight if active else 0.0
            self._g1_stability_rewards_active = active

        def _install_algorithm_update_callback(self) -> None:
            algorithm_update = self.alg.update

            def update_with_progress(*args: Any, **kwargs: Any):
                result = algorithm_update(*args, **kwargs)
                self._g1_training_iterations += 1
                self._g1_stage_policy_steps += int(self.cfg["num_steps_per_env"])
                self._g1_curriculum_iteration = self._g1_curriculum_start_iteration + (
                    self._g1_stage_policy_steps // BASELINE_ROLLOUT_STEPS
                )
                if (
                    self._g1_stability_reward_curriculum
                    and not self._g1_stability_rewards_active
                    and self.logger.lenbuffer
                    and sum(self.logger.lenbuffer) / len(self.logger.lenbuffer) > 500.0
                ):
                    self._set_stability_rewards(True)
                    print("[INFO] FAT2 and ZMP rewards enabled at mean episode length > 500")
                return result

            self.alg.update = update_with_progress

        def learn(self, *args: Any, **kwargs: Any):
            if training_configuration is None:
                return super().learn(*args, **kwargs)
            call_args = list(args)
            call_kwargs = dict(kwargs)
            if call_args:
                requested = call_args[0]
            elif "num_learning_iterations" in call_kwargs:
                requested = call_kwargs["num_learning_iterations"]
            else:
                raise RuntimeError("RSL learn call is missing num_learning_iterations")
            if isinstance(requested, bool) or not isinstance(requested, int):
                raise RuntimeError("RSL num_learning_iterations must be an integer")
            remaining = (
                s0_remaining_learning_iterations(
                    requested_iterations=requested,
                    completed_iterations=int(self._g1_training_iterations),
                )
                if stage == "s0_teacher"
                else s2_remaining_learning_iterations(
                    requested_iterations=requested,
                    completed_iterations=int(self._g1_training_iterations),
                )
            )
            if remaining == 0:
                print(f"[INFO] {stage} reached the iteration target")
                return None
            if call_args:
                call_args[0] = remaining
            else:
                call_kwargs["num_learning_iterations"] = remaining
            return super().learn(*call_args, **call_kwargs)

        def save(self, path: str, infos: dict | None = None) -> None:
            if training_configuration is None:
                raise RuntimeError("training checkpoint save requires a training configuration")
            saved = self.alg.save()
            if not isinstance(saved, Mapping):
                raise RuntimeError("RSL algorithm save payload must be a mapping")
            checkpoint = dict(saved)
            checkpoint["iter"] = self.current_learning_iteration
            checkpoint["infos"] = infos
            checkpoint["schema_version"] = CHECKPOINT_SCHEMA_VERSION
            checkpoint[CHECKPOINT_STAGE_KEY] = stage
            checkpoint[CHECKPOINT_CURRICULUM_ITERATION_KEY] = int(self._g1_curriculum_iteration)
            checkpoint[CHECKPOINT_STABILITY_REWARDS_ACTIVE_KEY] = bool(self._g1_stability_rewards_active)
            checkpoint[TRAINING_CONFIGURATION_KEY] = dict(training_configuration)
            checkpoint[CHECKPOINT_LINEAGE_KEY] = lineage
            attach_checkpoint_metadata(checkpoint, metadata, replace=True)
            atomic_torch_save(checkpoint, path)
            self.logger.save_model(path, self.current_learning_iteration)

        def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
            checkpoint = _torch_load(path)
            validate_checkpoint(checkpoint, expected=metadata, validate_torch_runtime=True)
            loaded_stage = checkpoint_stage(checkpoint)
            loaded_configuration = (
                validate_training_configuration(
                    checkpoint.get(TRAINING_CONFIGURATION_KEY),
                    expected_stage=loaded_stage,
                )
                if loaded_stage == "s2_bootstrap"
                else validate_guide_training_configuration(
                    checkpoint.get(TRAINING_CONFIGURATION_KEY),
                    expected_stage=loaded_stage,
                )
            )
            if training_configuration is not None:
                if loaded_configuration["training_parameters"] != training_configuration["training_parameters"]:
                    raise RuntimeError("loaded checkpoint training parameters differ from the active run")
                if reward_weight_overrides_from_configuration(
                    loaded_configuration
                ) != reward_weight_overrides_from_configuration(training_configuration):
                    raise RuntimeError("loaded checkpoint reward weights differ from the active run")
            if loaded_stage in {"s2_bootstrap", "s2_student_ppo"}:
                validate_student_checkpoint_architecture(checkpoint, loaded_configuration)
            elif loaded_stage == "s0_teacher":
                validate_teacher_checkpoint_architecture(checkpoint, loaded_configuration)

            allowed = {"s0_teacher"} if stage == "s0_teacher" else {"s2_bootstrap", "s2_student_ppo"}
            if loaded_stage not in allowed:
                raise RuntimeError(f"runner stage {stage!r} cannot load checkpoint stage {loaded_stage!r}")
            if loaded_stage == "s2_bootstrap":
                load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
                strict = True
            result = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
            self._g1_rickshaw_checkpoint_path = os.fspath(path)
            curriculum_iteration = checkpoint.get(CHECKPOINT_CURRICULUM_ITERATION_KEY)
            if isinstance(curriculum_iteration, bool) or not isinstance(curriculum_iteration, int):
                raise RuntimeError("loaded checkpoint is missing an audited curriculum iteration")
            self._g1_curriculum_iteration = curriculum_iteration
            if loaded_stage in {"s0_teacher", "s2_student_ppo"}:
                saved_iteration = checkpoint.get("iter")
                if isinstance(saved_iteration, bool) or not isinstance(saved_iteration, int) or saved_iteration < 0:
                    raise RuntimeError("loaded PPO checkpoint has no valid iteration")
                self._g1_training_iterations = saved_iteration + 1
                self._g1_stage_policy_steps = self._g1_training_iterations * int(self.cfg["num_steps_per_env"])
                completed = self._g1_stage_policy_steps // BASELINE_ROLLOUT_STEPS
                self._g1_curriculum_start_iteration = curriculum_iteration - completed
                if self._g1_curriculum_start_iteration < 0:
                    raise RuntimeError("loaded checkpoint curriculum precedes its completed sample budget")
                self.current_learning_iteration = self._g1_training_iterations
            if self._g1_stability_reward_curriculum:
                self._set_stability_rewards(bool(checkpoint.get(CHECKPOINT_STABILITY_REWARDS_ACTIVE_KEY, False)))
            return result

        def export_policy_to_jit(self, path: str, filename: str = "policy.pt"):
            result = super().export_policy_to_jit(path, filename)
            checkpoint_path = getattr(self, "_g1_rickshaw_checkpoint_path", None)
            if checkpoint_path is None:
                raise RuntimeError("deployment export requires a provenance-validated loaded checkpoint")
            controller_factory = getattr(self.alg.get_policy(), "as_deployment_controller", None)
            if not callable(controller_factory):
                raise RuntimeError("student policy does not expose the deployment controller contract")
            controller = controller_factory().to("cpu").eval()
            torch.jit.script(controller).save(os.fspath(Path(path) / "deployment_controller.pt"))
            return result

        def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False):
            result = super().export_policy_to_onnx(path, filename, verbose)
            checkpoint_path = getattr(self, "_g1_rickshaw_checkpoint_path", None)
            if checkpoint_path is None:
                raise RuntimeError("deployment export requires a provenance-validated loaded checkpoint")
            write_deployment_manifest(path, checkpoint_path)
            return result

    RickshawOnPolicyRunner.__name__ = "RickshawOnPolicyRunner"
    RickshawOnPolicyRunner.__qualname__ = "RickshawOnPolicyRunner"
    return RickshawOnPolicyRunner


__all__ = ["RunnerContext", "create_rickshaw_runner_type"]
