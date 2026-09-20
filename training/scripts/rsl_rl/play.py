# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_width", type=int, default=None, help="Recorded video width in pixels.")
parser.add_argument("--video_height", type=int, default=None, help="Recorded video height in pixels.")
parser.add_argument("--video_folder", type=str, default=None, help="Directory in which to save recorded videos.")
parser.add_argument(
    "--video_warmup_frames",
    type=int,
    default=8,
    help="Number of render frames to generate before video recording starts.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Camera position relative to the configured viewer origin.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Camera target relative to the configured viewer origin.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Exit after this many policy steps, including headless runs.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be positive")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.multi_gpu = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if (args_cli.video_width is None) != (args_cli.video_height is None):
        raise ValueError("--video_width and --video_height must be specified together.")
    if args_cli.video_width is not None:
        if args_cli.video_width <= 0 or args_cli.video_height <= 0:
            raise ValueError("Video dimensions must be positive.")
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
    if args_cli.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    if args_cli.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        if args_cli.video_warmup_frames < 0:
            raise ValueError("--video_warmup_frames must be non-negative.")
        video_folder = args_cli.video_folder or os.path.join(log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": os.path.abspath(video_folder),
            "step_trigger": lambda step: step == args_cli.video_warmup_frames,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    run_name = os.path.basename(log_dir)
    if version.parse(installed_version) >= version.parse("4.0.0"):
        ppo_runner.export_policy_to_jit(
            path=export_model_dir, filename=f"{agent_cfg.experiment_name}_{run_name}.pt"
        )
        ppo_runner.export_policy_to_onnx(
            path=export_model_dir, filename=f"{agent_cfg.experiment_name}_{run_name}.onnx"
        )
    else:
        policy_nn = (
            ppo_runner.alg.policy
            if version.parse(installed_version) >= version.parse("2.3.0")
            else ppo_runner.alg.actor_critic
        )
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        elif hasattr(ppo_runner, "obs_normalizer"):
            normalizer = ppo_runner.obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename=f"{agent_cfg.experiment_name}_{run_name}.pt",
        )
        export_policy_as_onnx(
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename=f"{agent_cfg.experiment_name}_{run_name}.onnx",
        )

    if args_cli.headless and not args_cli.video and args_cli.max_steps is None:
        print("[INFO] Headless mode and no video recording. Exiting after model export.")
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if installed_version.startswith("2.3."):
        obs, _ = obs
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            if args_cli.max_steps is not None and not torch.isfinite(actions).all():
                raise RuntimeError("Non-finite policy action during bounded playback")
            obs, rewards, dones, _ = env.step(actions)
            if args_cli.max_steps is not None and not torch.isfinite(rewards).all():
                raise RuntimeError("Non-finite reward during bounded playback")
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            if timestep <= args_cli.video_warmup_frames:
                env.unwrapped.render(recompute=True)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_warmup_frames + args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            print(f"[INFO] Bounded playback passed: {timestep} steps, finite actions and rewards.")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
