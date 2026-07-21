"""RSL-RL 5.0.1 adapters for the fixed rickshaw policy architecture."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.distributions import Independent, Normal

from g1_rickshaw_lab.policy_schema import (
    ACTION_SCALE,
    ACTOR_OBSERVATION_DIM,
    BUTTERWORTH_A1,
    BUTTERWORTH_B0,
    BUTTERWORTH_B1,
    DEFAULT_CONTEXT_DIM,
    HISTORY_LENGTH,
    validate_history_length,
    validate_context_dim,
)

from .actor_critic import (
    ACTION_DIM,
    CRITIC_PRIVILEGE_DIM,
    GaussianActor,
    PrivilegedCritic,
)
from .context_encoder import ContextEncoder
from .teacher_model import TeacherEncoder


def _require_rsl_rl() -> None:
    try:
        import rsl_rl  # noqa: F401
        import tensordict  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("RSL-RL adapters require rsl-rl-lib==5.0.1 and tensordict") from exc


class _RslModelContract(nn.Module):
    """Non-recurrent model methods required by RSL-RL."""

    is_recurrent = False
    obs_normalization = False

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs) -> None:
        del obs


class RslRickshawActorModel(_RslModelContract):
    """Teacher or student actor selected from the configured observation set."""

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        stochastic: Any = None,
        init_noise_std: Any = None,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        latent_dim: int = DEFAULT_CONTEXT_DIM,
        history_length: int = HISTORY_LENGTH,
    ) -> None:
        super().__init__()
        del stochastic, init_noise_std, noise_std_type, state_dependent_std
        _require_rsl_rl()
        if output_dim != ACTION_DIM:
            raise ValueError(f"rickshaw action dimension is fixed to {ACTION_DIM}, got {output_dim}")
        if tuple(hidden_dims) != (512, 256, 128) or activation.lower() != "elu":
            raise ValueError("rickshaw actor architecture is fixed to [512,256,128] with ELU")
        if obs_normalization:
            raise ValueError("runtime empirical observation normalization is forbidden")
        if distribution_cfg is None:
            raise ValueError("the PPO actor requires a Gaussian distribution configuration")
        self.latent_dim = validate_context_dim(latent_dim)
        self.history_length = validate_history_length(history_length)

        self.obs_groups = list(obs_groups[obs_set])
        if (
            "policy" not in self.obs_groups
            or obs["policy"].shape[-1] != ACTOR_OBSERVATION_DIM
        ):
            raise ValueError(
                f"actor observation set must contain policy[N,{ACTOR_OBSERVATION_DIM}]"
            )
        groups = set(self.obs_groups)
        teacher_groups = {
            "policy",
            "history",
            "teacher_dynamic_history",
            "teacher_static",
        }
        if groups == teacher_groups:
            self.encoder = TeacherEncoder(self.latent_dim, self.history_length)
            self.stage = "teacher"
        else:
            if groups != {"policy", "history"}:
                raise ValueError(
                    "actor groups must match the fixed teacher or student interface"
                )
            if tuple(obs["history"].shape[1:]) != (
                self.history_length,
                ACTOR_OBSERVATION_DIM,
            ):
                raise ValueError(
                    "history must have shape "
                    f"[N,{self.history_length},{ACTOR_OBSERVATION_DIM}]"
                )
            self.encoder = ContextEncoder(self.latent_dim, self.history_length)
            self.stage = "student"
        self.policy = GaussianActor(self.latent_dim)
        self._distribution: Independent | None = None

    def encode(self, obs) -> torch.Tensor:
        if self.stage == "teacher":
            return self.encoder(
                obs["history"],
                obs["teacher_dynamic_history"],
                obs["teacher_static"],
            )
        return self.encoder(obs["history"])

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        self._distribution = self.policy.distribution(obs["policy"], self.encode(obs))
        return self._distribution.sample() if stochastic_output else self._distribution.mean

    def _checked_distribution(self) -> Independent:
        if self._distribution is None:
            raise RuntimeError("actor distribution is unavailable before forward()")
        return self._distribution

    @property
    def output_mean(self) -> torch.Tensor:
        return self._checked_distribution().mean

    @property
    def output_std(self) -> torch.Tensor:
        return self._checked_distribution().base_dist.scale

    @property
    def output_entropy(self) -> torch.Tensor:
        return self._checked_distribution().entropy()

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.output_mean, self.output_std

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._checked_distribution().log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, torch.Tensor],
        new_params: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        divergence = torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        )
        return divergence.sum(dim=-1)

    def as_jit(self) -> nn.Module:
        if self.stage != "student":
            raise RuntimeError("only the student actor is deployable")
        return _StudentExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        if self.stage != "student":
            raise RuntimeError("only the student actor is deployable")
        return _StudentOnnxExport(self, verbose)

    def as_deployment_controller(self) -> nn.Module:
        """Return the stateless policy plus the exact deployment action contract."""

        if self.stage != "student":
            raise RuntimeError("only the student actor is deployable")
        return _DeploymentController(_StudentExport(self))


class RslRickshawCriticModel(_RslModelContract):
    """Independent value trunk using only current and raw privileged state."""

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        stochastic: Any = None,
        init_noise_std: Any = None,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        del stochastic, init_noise_std, noise_std_type, state_dependent_std
        _require_rsl_rl()
        if output_dim != 1 or distribution_cfg is not None:
            raise ValueError("critic must be deterministic with scalar output")
        if tuple(hidden_dims) != (256, 128) or activation.lower() != "elu":
            raise ValueError("rickshaw critic architecture is fixed to [256,128] with ELU")
        if obs_normalization:
            raise ValueError("runtime empirical observation normalization is forbidden")
        self.obs_groups = list(obs_groups[obs_set])
        if set(self.obs_groups) != {"policy", "critic"}:
            raise ValueError("critic observation set requires only policy and critic")
        if obs["critic"].shape[-1] != CRITIC_PRIVILEGE_DIM:
            raise ValueError(
                f"critic privilege must have width {CRITIC_PRIVILEGE_DIM}"
            )
        self.value = PrivilegedCritic()

    def forward(self, obs, masks: torch.Tensor | None = None, hidden_state: Any = None) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        return self.value(obs["policy"], obs["critic"])


class _RelativeLearningRateAdam(torch.optim.Adam):
    """Keep context/base LR ratios when RSL-RL's adaptive schedule updates LR."""

    def step(self, closure=None):
        original = [group["lr"] for group in self.param_groups]
        try:
            for group, learning_rate in zip(self.param_groups, original, strict=True):
                group["lr"] = learning_rate * group["lr_multiplier"]
            return super().step(closure)
        finally:
            for group, learning_rate in zip(self.param_groups, original, strict=True):
                group["lr"] = learning_rate


class RickshawPPO:
    """Factory facade returning RSL-RL PPO with a separate encoder LR."""

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        _require_rsl_rl()
        from rsl_rl.algorithms import PPO
        from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
        from rsl_rl.storage import RolloutStorage
        from rsl_rl.utils import resolve_callable, resolve_obs_groups

        algorithm_cfg = cfg["algorithm"]
        alg_class = resolve_callable(algorithm_cfg.pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        algorithm_cfg = resolve_rnd_config(algorithm_cfg, obs, cfg["obs_groups"], env)
        algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)
        algorithm_cfg.pop("share_cnn_encoders", None)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        if not isinstance(actor, RslRickshawActorModel) or not isinstance(critic, RslRickshawCriticModel):
            raise TypeError("RickshawPPO requires the rickshaw actor and critic adapters")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm = alg_class(
            actor,
            critic,
            storage,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **algorithm_cfg,
        )
        if not isinstance(algorithm, PPO):
            raise TypeError("configured algorithm must derive from rsl_rl.algorithms.PPO")
        return algorithm

    def __new__(
        cls,
        actor,
        critic,
        storage,
        context_learning_rate: float | None = None,
        learning_rate: float = 3.0e-4,
        optimizer: str = "adam",
        **kwargs,
    ):
        from rsl_rl.algorithms import PPO

        if optimizer.lower() != "adam":
            raise ValueError("the fixed rickshaw optimizer is Adam")
        if not isinstance(actor, RslRickshawActorModel) or not isinstance(critic, RslRickshawCriticModel):
            raise TypeError("RickshawPPO requires rickshaw model adapters")
        algorithm = PPO(
            actor,
            critic,
            storage,
            learning_rate=learning_rate,
            optimizer=optimizer,
            **kwargs,
        )
        context_lr = learning_rate if context_learning_rate is None else context_learning_rate
        if context_lr <= 0.0 or learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        context_parameters = list(actor.encoder.parameters())
        context_ids = {id(parameter) for parameter in context_parameters}
        actor_head = [parameter for parameter in actor.parameters() if id(parameter) not in context_ids]
        groups = [
            {
                "params": context_parameters,
                "lr": learning_rate,
                "lr_multiplier": context_lr / learning_rate,
            },
            {"params": actor_head, "lr": learning_rate, "lr_multiplier": 1.0},
            {"params": list(critic.parameters()), "lr": learning_rate, "lr_multiplier": 1.0},
        ]
        algorithm.optimizer = _RelativeLearningRateAdam(groups, lr=learning_rate)
        return algorithm


class _StudentExport(nn.Module):
    def __init__(self, model: RslRickshawActorModel) -> None:
        super().__init__()
        self.context_encoder = _DeploymentContextEncoder(model.encoder)
        self.policy = copy.deepcopy(model.policy.network)

    def forward(self, current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        context = self.context_encoder(history)
        return self.policy(torch.cat((current, context), dim=-1)).clamp(-1.0, 1.0)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _StudentOnnxExport(_StudentExport):
    def __init__(self, model: RslRickshawActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self):
        return (
            torch.zeros(1, ACTOR_OBSERVATION_DIM),
            torch.zeros(1, self.context_encoder.history_length, ACTOR_OBSERVATION_DIM),
        )

    @property
    def input_names(self) -> list[str]:
        return ["current", "history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _DeploymentController(nn.Module):
    """Stateless policy and 4 Hz action-filter step for deployment runtimes."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer(
            "action_scale", torch.tensor(ACTION_SCALE, dtype=torch.float32)
        )
        self.b0 = BUTTERWORTH_B0
        self.b1 = BUTTERWORTH_B1
        self.a1 = BUTTERWORTH_A1

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        q_ref: torch.Tensor,
        x_prev: torch.Tensor,
        y_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_action = torch.clamp(self.policy(current, history), -1.0, 1.0)
        scale = self.action_scale.to(dtype=normalized_action.dtype)
        x_next = normalized_action * scale + q_ref
        y_next = self.b0 * x_next + self.b1 * x_prev - self.a1 * y_prev
        return normalized_action, y_next, x_next, y_next

    @torch.jit.export
    def reset(self) -> None:
        pass


class _DeploymentContextEncoder(nn.Module):
    """Scriptable copy of the student TCN."""

    def __init__(self, encoder: ContextEncoder) -> None:
        super().__init__()
        self.history_length = encoder.history_length
        self.input = copy.deepcopy(encoder.input)
        self.blocks = copy.deepcopy(encoder.blocks)
        self.context = copy.deepcopy(encoder.context)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.input(history.transpose(1, 2)))[:, :, -1]
        return self.context(features)


__all__ = ["RickshawPPO", "RslRickshawActorModel", "RslRickshawCriticModel"]
