"""Project-owned Mjlab/RSL-RL train and play launchers."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from g1_rickshaw_lab.rl.runner import RunnerContext, create_rickshaw_runner_type


@dataclass(frozen=True, slots=True)
class PlayOptions:
    video_dir: Path | None = None
    export_policy: bool = True
    export_only: bool = False
    follow_robot_camera: bool = False
    slope_frames: int | None = None
    video_name_prefix: str = "rl-video"
    video_segment_callback: Callable[[Path, int], None] | None = None

    def __post_init__(self) -> None:
        if self.export_only and not self.export_policy:
            raise ValueError("export_only requires export_policy")
        if self.slope_frames is not None and self.slope_frames <= 0:
            raise ValueError("slope_frames must be positive")


class _JsonlSummaryWriter:
    """Small scalar writer used when TensorBoard cannot import its optional TF shim."""

    def __init__(self, log_dir: str | os.PathLike[str]) -> None:
        self.path = Path(log_dir) / "scalars.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int) -> None:
        value = scalar_value.item() if hasattr(scalar_value, "item") else scalar_value
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"tag": tag, "step": int(global_step), "value": float(value)}) + "\n")


def _tensorboard_available() -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter

        del SummaryWriter
    except Exception:
        return False
    return True


def _install_jsonl_logger(runner: Any) -> None:
    logger = runner.logger

    def init_logging_writer() -> None:
        logger.logger_type = "jsonl"
        logger.writer = _JsonlSummaryWriter(logger.log_dir)
        logger._store_code_state()

    logger.init_logging_writer = init_logging_writer


def _parser(mode: Literal["train", "play"]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"G1 rickshaw Mjlab RSL-RL {mode} launcher")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "jsonl"), default=None)
    parser.add_argument("--log_project_name", default=None)
    if mode == "train":
        parser.add_argument("--video_interval", type=int, default=2000)
        parser.add_argument("--max_iterations", type=int, default=None)
    else:
        parser.add_argument("--real-time", action="store_true", default=False)
    return parser


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if value.lower() in {"none", "null"}:
        return None
    return value


def _set_path(root: Any, path: str, value: str) -> None:
    parts = path.split(".")
    target = root
    for part in parts[:-1]:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    leaf = parts[-1]
    current = target[leaf] if isinstance(target, dict) else getattr(target, leaf)
    converted = _coerce(value, current)
    if isinstance(target, dict):
        target[leaf] = converted
    else:
        setattr(target, leaf, converted)


def _apply_overrides(env_cfg: Any, agent_cfg: Any, overrides: list[str]) -> None:
    for token in overrides:
        if "=" not in token:
            raise ValueError(f"unsupported Mjlab override: {token}")
        key, value = token.lstrip("+").split("=", 1)
        if key == "env.history_length":
            history_length = int(value)
            env_cfg.history_length = history_length
            runtime = env_cfg.events["initialize_task"].params["cfg"]
            runtime = replace(runtime, history_length=history_length)
            env_cfg.events["initialize_task"].params["cfg"] = runtime
            env_cfg.events["initialize_domain"].params["cfg"] = runtime
            env_cfg.events["policy_state"].params["cfg"] = runtime
            env_cfg.policy_update = runtime
            env_cfg.observations["history"].terms["history"].params["history_length"] = history_length
            env_cfg.observations["teacher_dynamic_history"].terms["history"].params[
                "history_length"
            ] = history_length
        elif key.startswith("env."):
            _set_path(env_cfg, key.removeprefix("env."), value)
        elif key.startswith("agent."):
            _set_path(agent_cfg, key.removeprefix("agent."), value)
        else:
            raise ValueError(f"override must start with env. or agent.: {token}")


def run_rsl_rl(
    mode: Literal["train", "play"],
    argv: list[str],
    *,
    runner_context: RunnerContext,
    play_options: PlayOptions | None = None,
) -> None:
    parser = _parser(mode)
    args, overrides = parser.parse_known_args(argv)
    if mode == "train":
        _run_train(args, overrides, runner_context)
    else:
        _run_play(args, overrides, runner_context, play_options or PlayOptions())


def _load_configs(args: argparse.Namespace, overrides: list[str], *, play: bool):
    import g1_rickshaw_lab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(args.task, play=play)
    agent_cfg = load_rl_cfg(args.task)
    _apply_overrides(env_cfg, agent_cfg, overrides)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if args.log_project_name is not None:
        agent_cfg.wandb_project = args.log_project_name
    if args.resume:
        agent_cfg.resume = True
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    return env_cfg, agent_cfg


def _run_train(args: argparse.Namespace, overrides: list[str], context: RunnerContext) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml, get_checkpoint_path
    from mjlab.utils.wrappers import VideoRecorder

    env_cfg, agent_cfg = _load_configs(args, overrides, play=False)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_name).resolve()
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root.resolve(), agent_cfg.load_run, agent_cfg.load_checkpoint)
    env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "train",
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_type = create_rickshaw_runner_type(context, base_runner_type=MjlabOnPolicyRunner)
    runner = runner_type(wrapped, asdict(agent_cfg), str(log_dir), device)
    if agent_cfg.logger == "jsonl" or (agent_cfg.logger == "tensorboard" and not _tensorboard_available()):
        if agent_cfg.logger == "tensorboard":
            print("[WARN] TensorBoard is unavailable; writing scalar metrics to scalars.jsonl")
        _install_jsonl_logger(runner)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        runner.load(str(resume_path))
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    wrapped.close()


def _run_play(
    args: argparse.Namespace,
    overrides: list[str],
    context: RunnerContext,
    options: PlayOptions,
) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.os import get_checkpoint_path
    from mjlab.utils.wrappers import VideoRecorder

    env_cfg, agent_cfg = _load_configs(args, overrides, play=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log_root = (Path("logs/rsl_rl") / agent_cfg.experiment_name).resolve()
    if args.checkpoint and Path(args.checkpoint).is_file():
        resume_path = Path(args.checkpoint).resolve()
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = resume_path.parent
    env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array" if args.video else None)
    video_dir = options.video_dir or log_dir / "videos" / "play"
    video_recorder = None
    if args.video:
        video_recorder = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=(lambda step: step == 0)
            if options.slope_frames is None
            else (lambda step: step % options.slope_frames == 0),
            video_length=args.video_length
            if options.slope_frames is None
            else options.slope_frames,
            name_prefix=options.video_name_prefix,
            disable_logger=True,
        )
        env = video_recorder
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_type = create_rickshaw_runner_type(context, base_runner_type=MjlabOnPolicyRunner)
    runner = runner_type(wrapped, asdict(agent_cfg), None, device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
    if options.export_policy:
        export_dir = log_dir / "exported"
        runner.export_policy_to_jit(str(export_dir), "policy.pt")
        runner.export_policy_to_onnx(str(export_dir), "policy.onnx")
    if options.export_only:
        wrapped.close()
        return
    if args.video:
        obs = wrapped.get_observations()
        for step in range(args.video_length):
            start = time.time()
            with torch.inference_mode():
                action = policy(obs)
                obs, _, dones, _ = wrapped.step(action)
                policy.reset(dones)
            if options.slope_frames and options.video_segment_callback and (step + 1) % options.slope_frames == 0:
                if video_recorder is None or video_recorder.is_recording:
                    raise RuntimeError("video segment callback ran before the MP4 was finalized")
                segment = (step + 1) // options.slope_frames - 1
                raw = video_recorder.video_folder / (
                    f"{options.video_name_prefix}-step-{segment * options.slope_frames}.mp4"
                )
                options.video_segment_callback(raw, segment)
            if args.real_time:
                time.sleep(max(0.0, wrapped.unwrapped.step_dt - (time.time() - start)))
    else:
        from mjlab.viewer import ViserPlayViewer

        ViserPlayViewer(wrapped.unwrapped, policy).run()
    wrapped.close()


__all__ = ["PlayOptions", "run_rsl_rl"]
