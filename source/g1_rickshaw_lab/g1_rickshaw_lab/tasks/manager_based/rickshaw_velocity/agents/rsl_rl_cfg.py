"""RSL-RL PPO defaults fixed by the implementation guide."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from g1_rickshaw_lab.training_contract import GUIDE_MAX_ITERATIONS, TRAINING_ARTIFACT_INTERVAL


def _gaussian_cfg() -> RslRlMLPModelCfg.GaussianDistributionCfg:
    # Per-joint lower/upper body stds are installed by the custom policy checkpoint loader.
    return RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4, std_type="log")


@configclass
class G1RickshawPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "g1_rickshaw_lab.rl.rsl_rl_models:RickshawPPO"
    context_learning_rate: float | None = None


@configclass
class G1RickshawModelCfg(RslRlMLPModelCfg):
    """Custom actor/critic model configuration for the required latent sweep."""

    latent_dim: int = 16


@configclass
class G1RickshawTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """S0 privileged-teacher PPO configuration."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 48
    max_iterations = GUIDE_MAX_ITERATIONS["s0_teacher"]
    save_interval = TRAINING_ARTIFACT_INTERVAL
    experiment_name = "g1_rickshaw_teacher"
    run_name = "s0"
    empirical_normalization = False
    clip_actions = 1.0
    obs_groups = {
        "actor": ["policy", "teacher_extrinsics"],
        "critic": ["policy", "teacher_extrinsics", "critic"],
    }
    actor = G1RickshawModelCfg(
        class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawActorModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=_gaussian_cfg(),
    )
    critic = G1RickshawModelCfg(
        class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawCriticModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    algorithm = G1RickshawPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.97,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RickshawStudentPPORunnerCfg(G1RickshawTeacherPPORunnerCfg):
    """S2 student PPO fine-tuning configuration."""

    max_iterations = GUIDE_MAX_ITERATIONS["s2_student_ppo"]
    experiment_name = "g1_rickshaw_student"
    run_name = "s2"
    obs_groups = {
        "actor": ["policy", "history"],
        "critic": ["policy", "history", "critic"],
    }
    actor = G1RickshawModelCfg(
        class_name="g1_rickshaw_lab.rl.rsl_rl_models:RslRickshawActorModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=_gaussian_cfg(),
    )

    def __post_init__(self) -> None:
        self.algorithm.context_learning_rate = 1.0e-4
