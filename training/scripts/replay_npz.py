"""This script demonstrates how to replay K1 robot motions from npz files.

.. code-block:: bash

    # Usage - Direct file path
    python replay_npz.py --motion <path_to_motion.npz>
    
    # Usage - From wandb registry
    python replay_npz.py --registry_name <wandb_registry_name>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--robot", choices=("booster_k1", "mos"), default="booster_k1")
parser.add_argument("--motion", type=str, default=None, help="Path to the motion npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument("--once", action="store_true", help="Replay one complete clip and exit.")
parser.add_argument("--video_path", type=str, help="Record the reference to an MP4 file; implies --once.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video_path:
    args_cli.enable_cameras = True
    args_cli.once = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionLoader

ROBOT_CFG = BOOSTER_K1_CFG
ROOT_NAME = "Trunk"
if args_cli.robot == "mos":
    from booster_train.assets.robots.mos import MOS_CFG as ROBOT_CFG
    ROOT_NAME = "body"


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Determine motion file path
    if args_cli.motion:
        # Use direct file path
        motion_file = args_cli.motion
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
    elif args_cli.registry_name:
        # Download from wandb registry
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is required when using --registry_name. Install it with: pip install wandb")
        
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Either --motion or --registry_name must be provided.")

    # Load npz file to get body names and determine body_indexes
    # For K1, we typically use Trunk as anchor body (index 0)
    # body_indexes should be a list of indices corresponding to the bodies we want to use
    # For replay, we only need the anchor body (Trunk), which is typically at index 0
    motion = MotionLoader(
        motion_file,
        [ROOT_NAME],
        robot.joint_names,
        tail_len=0,
        device=str(sim.device),
    )
    frame = 0
    writer = None
    annotator = None
    if args_cli.video_path:
        import imageio.v2 as imageio
        import omni.replicator.core as rep

        path = pathlib.Path(args_cli.video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        product = rep.create.render_product("/OmniverseKit_Persp", (960, 720))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        writer = imageio.get_writer(str(path), fps=float(np.asarray(motion.fps).item()))
    print(f"[INFO] Replaying {motion_file}: {motion.time_step_total} frames; joints mapped by name.")

    # Simulation loop
    while simulation_app.is_running():
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[frame, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[frame, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[frame, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[frame, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[frame:frame+1], motion.joint_vel[frame:frame+1])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        target = pos_lookat.copy()
        target[2] = .5
        sim.set_camera_view(target + np.array([1.6, 1.6, .7]), target)
        if writer is not None:
            for _ in range(12 if frame == 0 else 2):
                sim.render()
            pixels = annotator.get_data()
            if pixels.size == 0 or pixels[:, :, :3].std() < 1:
                raise RuntimeError(f"Empty/blank replay frame {frame}.")
            writer.append_data(pixels[:, :, :3])
            if frame in {0, motion.time_step_total // 2, motion.time_step_total - 1}:
                imageio.imwrite(str(path.with_name(f"{path.stem}_{frame:04d}.png")), pixels[:, :, :3])
        frame += 1
        if frame == motion.time_step_total:
            if args_cli.once:
                if writer is not None:
                    writer.close()
                print("[INFO] Reference replay complete.")
                return
            frame = 0
    if writer is not None:
        writer.close()
    if args_cli.once:
        raise RuntimeError("Simulator closed before reference replay finished.")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        SimulationContext.clear_instance()
        simulation_app.close()
