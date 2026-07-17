"""IsaacLab Hydra round-trip regression for the manager-based task configs."""

from __future__ import annotations

from dataclasses import is_dataclass
from importlib.util import find_spec

import pytest


ISAACLAB_RUNTIME_AVAILABLE = find_spec("isaaclab") is not None and find_spec("isaacsim") is not None


if ISAACLAB_RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    _APP_LAUNCHER = AppLauncher(headless=True)
    _SIMULATION_APP = _APP_LAUNCHER.app


from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import mdp


HYDRA_EMBEDDED_CFG_TYPES = (
    mdp.CurriculumScheduleCfg,
    mdp.SpeedReferenceCfg,
    mdp.RollingResistanceCfg,
    mdp.AnalyticForceCfg,
    mdp.FAT2Cfg,
    mdp.ZMPCfg,
    mdp.SupportPolygonCfg,
    mdp.RickshawPoseTargetCfg,
    mdp.SpeedCommandSamplingCfg,
    mdp.DomainRandomizationCfg,
    mdp.HandleConstraintCfg,
    mdp.ResetValidationCfg,
    mdp.TaskEntityNamesCfg,
    mdp.PolicyStateUpdateCfg,
    mdp.ImmediateSafetyCfg,
    mdp.PersistentSafetyCfg,
)


if ISAACLAB_RUNTIME_AVAILABLE:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
    from isaaclab.utils import replace_strings_with_slices
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    from omegaconf import OmegaConf

    import g1_rickshaw_lab  # noqa: F401  # registers the two Gym tasks after Kit starts
    from g1_rickshaw_lab.tasks.manager_based.rickshaw_velocity import PLAY_TASK_ID, TRAIN_TASK_ID
else:
    PLAY_TASK_ID = "Isaac-G1-Rickshaw-Directional-Slope-Play-v0"
    TRAIN_TASK_ID = "Isaac-G1-Rickshaw-Directional-Slope-v0"


def _hydra_round_trip(task_id: str, overrides: list[str]):
    env_cfg, _ = register_task_to_hydra(task_id, None)
    try:
        with initialize(config_path=None, version_base="1.3"):
            hydra_cfg = compose(config_name=task_id, overrides=overrides)
        native_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)
        native_cfg = replace_strings_with_slices(native_cfg)
        env_cfg.from_dict(native_cfg["env"])
        return replace_strings_with_env_cfg_spaces(env_cfg)
    finally:
        GlobalHydra.instance().clear()


def _assert_manager_cfg_bindings(env_cfg) -> None:
    initialize_params = env_cfg.events.initialize_mdp.params

    assert env_cfg.scene.closed_chain.spawn.handle_constraint is env_cfg.handle_constraint
    assert initialize_params["handle_constraint_cfg"] is env_cfg.handle_constraint
    assert initialize_params["rolling_resistance_cfg"] is env_cfg.rolling_resistance
    assert initialize_params["entity_names_cfg"] is env_cfg.task_entity_names
    assert initialize_params["rickshaw_pose_cfg"] is env_cfg.rickshaw_pose

    assert env_cfg.events.initialize_domain.params["cfg"] is env_cfg.domain_randomization
    assert env_cfg.events.policy_interval.params["cfg"] is env_cfg.policy_update
    assert env_cfg.terminations.refresh_policy_state.params["cfg"] is env_cfg.policy_update


def test_hydra_embedded_cfg_dataclasses_are_mutable() -> None:
    for cfg_type in HYDRA_EMBEDDED_CFG_TYPES:
        assert is_dataclass(cfg_type)
        assert not cfg_type.__dataclass_params__.frozen, cfg_type.__name__


def test_privileged_observation_dimensions_are_fixed() -> None:
    assert mdp.TEACHER_STATIC_DIM == 40
    assert mdp.TEACHER_DYNAMIC_DIM == 21
    assert mdp.CRITIC_PRIVILEGED_DIM == 64


@pytest.mark.skipif(not ISAACLAB_RUNTIME_AVAILABLE, reason="IsaacLab runtime is not installed")
@pytest.mark.parametrize("task_id", (TRAIN_TASK_ID, PLAY_TASK_ID))
def test_manager_env_cfg_survives_full_hydra_round_trip(task_id: str) -> None:
    env_cfg = _hydra_round_trip(
        task_id,
        [
            "env.scene.num_envs=8",
            "env.handle_constraint.max_force=1234.5",
            "env.rolling_resistance.enabled=false",
            "env.policy_update.speed_reference.response_time=0.4",
        ],
    )

    assert env_cfg.scene.num_envs == 8
    assert env_cfg.scene.replicate_physics is True
    assert env_cfg.scene.clone_in_fabric is False
    assert env_cfg.domain_randomization.enabled is (task_id == TRAIN_TASK_ID)
    assert set(env_cfg.domain_randomization.ranges) == set(mdp.DOMAIN_PARAMETER_NAMES)
    assert set(env_cfg.domain_randomization.nominal) == set(mdp.DOMAIN_PARAMETER_NAMES)
    assert len(env_cfg.reset_pose_library.poses) == 19
    assert "reset_pose_library" not in env_cfg.to_dict()
    assert hasattr(env_cfg.observations, "teacher_dynamic_history")
    assert hasattr(env_cfg.observations, "teacher_static")
    assert not hasattr(env_cfg.observations, "teacher_extrinsics")
    assert env_cfg.handle_constraint.max_force == pytest.approx(1234.5)
    assert env_cfg.scene.closed_chain.spawn.handle_constraint.max_force == pytest.approx(1234.5)
    assert env_cfg.rolling_resistance.enabled is False
    assert env_cfg.events.initialize_mdp.params["rolling_resistance_cfg"].enabled is False
    assert env_cfg.events.policy_interval.params["cfg"].speed_reference.response_time == pytest.approx(0.4)
    assert env_cfg.terminations.refresh_policy_state.params["cfg"].speed_reference.response_time == pytest.approx(
        0.4
    )
    _assert_manager_cfg_bindings(env_cfg)
