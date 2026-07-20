"""First-party Isaac Lab/RSL-RL train and play launchers."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from g1_rickshaw_lab.rl.runner import RunnerContext, create_rickshaw_runner_type


@dataclass(frozen=True, slots=True)
class PlayOptions:
    """Project-specific presentation options kept outside Hydra policy state."""

    video_dir: Path | None = None
    export_policy: bool = True
    export_only: bool = False
    follow_robot_camera: bool = False
    slope_frames: int | None = None

    def __post_init__(self) -> None:
        if self.export_only and not self.export_policy:
            raise ValueError("export_only requires export_policy")
        if self.slope_frames is not None and self.slope_frames <= 0:
            raise ValueError("slope_frames must be positive")


def _add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--experiment_name", default=None)
    group.add_argument("--run_name", default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", default=None)
    group.add_argument("--checkpoint", default=None)
    group.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
    group.add_argument("--log_project_name", default=None)


def _update_agent_cfg(agent_cfg: Any, args: argparse.Namespace) -> Any:
    for argument, attribute in (
        ("resume", "resume"),
        ("load_run", "load_run"),
        ("checkpoint", "load_checkpoint"),
        ("experiment_name", "experiment_name"),
        ("run_name", "run_name"),
        ("logger", "logger"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(agent_cfg, attribute, value)
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if agent_cfg.logger == "wandb" and args.log_project_name:
        agent_cfg.wandb_project = args.log_project_name
    if agent_cfg.logger == "neptune" and args.log_project_name:
        agent_cfg.neptune_project = args.log_project_name
    return agent_cfg


def _parser(mode: Literal["train", "play"]):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=f"G1 rickshaw RSL-RL {mode} launcher")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=None)
    if mode == "train":
        parser.add_argument("--video_interval", type=int, default=2000)
        parser.add_argument("--max_iterations", type=int, default=None)
        parser.add_argument("--distributed", action="store_true", default=False)
    else:
        parser.add_argument("--real-time", action="store_true", default=False)
    _add_runner_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def run_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    runner_context: RunnerContext,
    play_options: PlayOptions | None = None,
) -> None:
    """Launch one project-owned RSL-RL workflow without rewriting upstream code."""

    from isaaclab.app import AppLauncher

    parser = _parser(mode)
    args, hydra_args = parser.parse_known_args(argv)
    if args.video:
        args.enable_cameras = True
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *hydra_args]
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        if mode == "train":
            _run_train(args, runner_context)
        else:
            _run_play(args, runner_context, play_options or PlayOptions(), simulation_app)
    finally:
        simulation_app.close()
        sys.argv = previous_argv


def _run_train(args: argparse.Namespace, runner_context: RunnerContext) -> None:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import g1_rickshaw_lab.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def main(env_cfg, agent_cfg) -> None:
        agent_cfg = _update_agent_cfg(agent_cfg, args)
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        env_cfg.seed = agent_cfg.seed
        if args.device is not None:
            env_cfg.sim.device = args.device
        if args.distributed:
            if "cpu" in env_cfg.sim.device:
                raise ValueError("distributed training requires a CUDA device")
            env_cfg.sim.device = f"cuda:{args.local_rank}"
            agent_cfg.device = f"cuda:{args.local_rank}"
            env_cfg.seed += args.local_rank
            agent_cfg.seed = env_cfg.seed

        log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
        log_root = log_root.resolve()
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            run_name += f"_{agent_cfg.run_name}"
        log_dir = log_root / run_name
        env_cfg.log_dir = os.fspath(log_dir)
        resume_path = None
        if agent_cfg.resume:
            resume_path = get_checkpoint_path(
                os.fspath(log_root),
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.fspath(log_dir / "videos/train"),
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_type = create_rickshaw_runner_type(runner_context)
        runner = runner_type(env, agent_cfg.to_dict(), log_dir=os.fspath(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            runner.load(resume_path)
        dump_yaml(os.fspath(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(os.fspath(log_dir / "params/agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        env.close()

    main()


def _run_play(
    args: argparse.Namespace,
    runner_context: RunnerContext,
    options: PlayOptions,
    simulation_app: Any,
) -> None:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import g1_rickshaw_lab.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def main(env_cfg, agent_cfg) -> None:
        agent_cfg = _update_agent_cfg(agent_cfg, args)
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = agent_cfg.seed
        if args.device is not None:
            env_cfg.sim.device = args.device
        if options.follow_robot_camera:
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.eye = (0.0, 4.2, 1.4)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.85)

        log_root = (Path("logs/rsl_rl") / agent_cfg.experiment_name).resolve()
        resume_path = (
            retrieve_file_path(args.checkpoint)
            if args.checkpoint
            else get_checkpoint_path(os.fspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
        log_dir = Path(resume_path).parent
        env_cfg.log_dir = os.fspath(log_dir)
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            video_dir = options.video_dir or log_dir / "videos/play"
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.fspath(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_type = create_rickshaw_runner_type(runner_context)
        runner = runner_type(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if options.export_policy:
            export_dir = log_dir / "exported"
            runner.export_policy_to_jit(path=os.fspath(export_dir), filename="policy.pt")
            runner.export_policy_to_onnx(path=os.fspath(export_dir), filename="policy.onnx")
        if options.export_only:
            env.close()
            return

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            if args.video:
                timestep += 1
                if options.slope_frames is not None:
                    index = min((timestep + 1) // options.slope_frames, env.unwrapped.num_envs - 1)
                    env.unwrapped.viewport_camera_controller.set_view_env_index(index)
                    robot = env.unwrapped.scene["robot"].data.root_pos_w[index]
                    cart = env.unwrapped.scene["rickshaw"].data.root_pos_w[index]
                    target = 0.5 * (robot + cart)
                    target[2] = torch.maximum(target[2], target.new_tensor(0.85))
                    eye = target + target.new_tensor((0.0, 4.2, 1.4))
                    env.unwrapped.sim.set_camera_view(
                        tuple(float(value) for value in eye.detach().cpu()),
                        tuple(float(value) for value in target.detach().cpu()),
                    )
                if timestep == args.video_length:
                    break
            sleep_time = dt - (time.time() - start_time)
            if args.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        env.close()

    main()


__all__ = ["PlayOptions", "run_rsl_rl"]
