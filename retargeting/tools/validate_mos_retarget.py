"""Retarget the three local MOS locomotion samples and inspect their kinematics."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco as mj
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.robot_motion_viewer import draw_skeleton
from general_motion_retargeting.utils.mos import foot_mesh_heights, retarget_mos_clip
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from scripts.smplx_to_robot import build_visual_skeleton
from wbmr.paths import MOTIONS, body_model_dir

INPUTS = {
    "walk": "ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz",
    "jog": "CMU/02/02_03_stageii.npz",
    "run": "ACCAD/Male2Running_c3d/C3_-_run_stageii.npz",
}


def validate(name, source, output, models):
    source_data, body, smpl_output, height = load_smplx_file(source, models)
    frames, fps = get_smplx_data_offline_fast(source_data, body, smpl_output)
    del source_data, body, smpl_output
    retargeter = GMR("smplx", "mos", height, verbose=False)
    qpos, raw_heights, shift = retarget_mos_clip(retargeter, frames, align_ground=True)
    assert qpos.shape == (len(frames), 27)
    assert np.isfinite(qpos).all() and np.isfinite(fps) and fps > 0
    np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-8)
    assert abs((raw_heights + shift).min()) < 1e-8
    model, data = retargeter.model, mj.MjData(retargeter.model)
    angle_errors, foot_positions, head_errors = [], [], []
    for frame, q in zip(frames, qpos):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        targets = retargeter.target_builder(frame)
        head_errors.append(np.linalg.norm(
            targets["head"][0] + [0, 0, shift] - data.xpos[model.body("head").id]
        ))
        angle_errors.append([
            (R.from_quat(targets[joint][1], scalar_first=True).inv()
             * R.from_quat(data.xquat[model.body(body).id], scalar_first=True)).magnitude()
            for joint, body in retargeter.target_builder.BODY_MAP.items()
        ])
        foot_positions.append([data.xpos[model.body(foot).id].copy() for foot in ("Lfoot", "Rfoot")])
        np.testing.assert_allclose(
            foot_mesh_heights(model, data),
            raw_heights[len(foot_positions) - 1] + shift, atol=1e-7,
        )
    errors = np.rad2deg(angle_errors)
    foot_positions = np.asarray(foot_positions)
    heights = raw_heights + shift
    near_floor = (heights[:-1] < 0.02) & (heights[1:] < 0.02)
    foot_speeds = np.linalg.norm(np.diff(foot_positions[:, :, :2], axis=0), axis=2) * fps
    joint_step = np.abs(np.diff(qpos[:, 7:], axis=0))
    report = {
        "source": str(source), "fps": float(fps), "frames": len(frames),
        "root_scale": float(retargeter.target_builder.root_scale),
        "constant_ground_shift_m": shift,
        "raw_foot_min_m": float(raw_heights.min()),
        "aligned_foot_min_m": float(heights.min()),
        "lowest_foot_height_percentiles_m": np.percentile(heights.min(axis=1), [0, 50, 95, 100]).tolist(),
        "near_floor_horizontal_speed_p95_m_s": (
            float(np.percentile(foot_speeds[near_floor], 95)) if near_floor.any() else None
        ),
        "max_joint_step_rad": float(joint_step.max()),
        "max_joint_step_frame": int(np.unravel_index(joint_step.argmax(), joint_step.shape)[0] + 1),
        "joint_names": [model.joint(i).name for i in range(1, model.njnt)],
        "joint_min_rad": qpos[:, 7:].min(axis=0).tolist(),
        "joint_max_rad": qpos[:, 7:].max(axis=0).tolist(),
        "orientation_error_mean_deg": dict(zip(retargeter.target_builder.BODY_MAP, errors.mean(axis=0).tolist())),
        "orientation_error_max_deg": dict(zip(retargeter.target_builder.BODY_MAP, errors.max(axis=0).tolist())),
        "head_target_to_link_origin_error_m": {
            "mean": float(np.mean(head_errors)), "max": float(np.max(head_errors)),
        },
        "limitations": [
            "Kinematic reference only; continuous URDF joints have no supplied mechanical limits.",
            "Rigid torso landmarks follow the pelvis; independent human spine bending is not reproduced.",
            "The head target marks a joint origin, not the visible head mesh center.",
            "Shoulders have only two axes; human arm twist cannot be reproduced exactly.",
            "One constant mesh-floor correction does not enforce contacts or prevent sliding.",
            "Loop boundaries are not made cyclic. Collision proxies are not validated.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    motion = {
        "fps": float(fps), "root_pos": qpos[:, :3],
        "root_rot": qpos[:, [4, 5, 6, 3]], "dof_pos": qpos[:, 7:],
        "local_body_pos": None, "link_body_list": None,
    }
    with (output / f"{name}.pkl").open("wb") as stream:
        pickle.dump(motion, stream)
    (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    indices = np.linspace(0, len(qpos) - 1, 6).astype(int)
    montage = Image.new("RGB", (640 * 3, 480 * 4))
    skeleton_montage = Image.new("RGB", montage.size)
    retargeter.set_ground_offset(-shift)
    opt = mj.MjvOption()
    opt.geomgroup[3] = 0
    with mj.Renderer(model, height=480, width=640) as renderer:
        for row, view in enumerate((0, 90)):
            for col, index in enumerate(indices):
                q = qpos[index]
                data.qpos[:] = q
                mj.mj_forward(model, data)
                cam = mj.MjvCamera()
                cam.lookat[:] = q[:3] + [0, 0, -0.05]
                cam.distance, cam.elevation = 1.65, -12
                forward = data.xmat[model.body("body").id].reshape(3, 3)[:, 0]
                cam.azimuth = np.rad2deg(np.arctan2(forward[1], forward[0])) + view
                renderer.update_scene(data, camera=cam, scene_option=opt)
                pixels = renderer.render()
                assert pixels.std() > 2, "Blank render"
                image = Image.fromarray(pixels)
                ImageDraw.Draw(image).text((12, 12), f"{name} frame {index} / view {view}", fill="white")
                montage.paste(image, (col % 3 * 640, (row * 2 + col // 3) * 480))
                retargeter.update_targets(frames[index])
                visual = build_visual_skeleton(frames[index], retargeter, False)
                draw_skeleton(visual, SimpleNamespace(user_scn=renderer.scene))
                image = Image.fromarray(renderer.render())
                ImageDraw.Draw(image).text(
                    (12, 12), f"{name} frame {index} / targets / view {view}", fill="white"
                )
                skeleton_montage.paste(image, (col % 3 * 640, (row * 2 + col // 3) * 480))
    montage.save(output / f"{name}_poses.jpg")
    skeleton_montage.save(output / f"{name}_skeleton.jpg")
    print(json.dumps({key: report[key] for key in (
        "frames", "fps", "constant_ground_shift_m", "max_joint_step_rad",
        "lowest_foot_height_percentiles_m", "near_floor_horizontal_speed_p95_m_s",
        "head_target_to_link_origin_error_m",
    )}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True, help="Your authorized ACCAD/CMU inputs.")
    parser.add_argument("--body_model_dir", help="Licensed model directory; defaults to SMPLX_MODEL_DIR.")
    parser.add_argument("--output", type=Path, default=MOTIONS / "gmr/mos_locomotion_calibrated")
    parser.add_argument("--clips", choices=list(INPUTS), nargs="+", default=list(INPUTS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    for name in args.clips:
        print(f"Validating {name}", flush=True)
        validate(name, args.data_root / INPUTS[name], args.output, body_model_dir(args.body_model_dir))


if __name__ == "__main__":
    main()
