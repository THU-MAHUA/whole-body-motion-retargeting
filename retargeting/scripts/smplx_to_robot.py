import argparse
import pathlib
import os
import time
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.mos import MosTargets
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print
from wbmr.paths import MOTIONS, body_model_dir


def build_visual_skeleton(frame, retargeter, offset_to_ground):
    """Scale all SMPL-X joints using the same regional factors as GMR."""
    if retargeter.target_builder is not None:
        # Display calibrated anatomical targets, not the obsolete K1 proportions.
        visual = {
            name: [pos.copy(), quat.copy()]
            for name, (pos, quat) in retargeter.scaled_human_data.items()
        }
        for side in ("left", "right"):
            visual[f"{side}_foot"] = visual.pop(f"{side}_ankle")
        if isinstance(retargeter.target_builder, MosTargets):
            # The MOS head-joint origin is off-center; center only the display
            # landmark in the pelvis frame, leaving IK targets and rotations intact.
            pelvis_rotation = (
                R.from_quat(visual["pelvis"][1], scalar_first=True)
                * retargeter.target_builder.rotations["pelvis"].inv()
            )
            offset = pelvis_rotation.inv().apply(visual["head"][0] - visual["pelvis"][0])
            offset[1] = 0.0
            visual["head"][0] = visual["pelvis"][0] + pelvis_rotation.apply(offset)
        pelvis, head = visual["pelvis"][0], visual["head"][0]
        for name, fraction in (("spine1", 0.25), ("spine2", 0.5), ("spine3", 0.7), ("neck", 0.9)):
            visual[name] = [(1 - fraction) * pelvis + fraction * head, visual["pelvis"][1]]
        return visual
    root = np.asarray(frame["pelvis"][0])
    root_scale = float(retargeter.human_scale_table["pelvis"])
    scaled_root = root * root_scale
    visual = {}
    for name, (pos, quat) in frame.items():
        if name == "head":
            scale = float(retargeter.human_scale_table["head"])
        elif "shoulder" in name or "collar" in name:
            scale = float(retargeter.human_scale_table["left_shoulder"])
        elif "elbow" in name or "wrist" in name or name.endswith("hand"):
            scale = float(retargeter.human_scale_table["left_elbow"])
        else:
            scale = root_scale
        visual[name] = [(np.asarray(pos) - root) * scale + scaled_root, quat]

    if offset_to_ground:
        lowest = min(visual[name][0][2] for name in ("left_foot", "right_foot"))
        for pos, _ in visual.values():
            pos[2] += 0.1 - lowest
    return visual

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--robot",
        choices=["booster_k1", "mos"],
        default="booster_k1",
    )
    parser.add_argument("--body_model_dir", help="Licensed SMPL-X model directory; defaults to SMPLX_MODEL_DIR.")
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--offset_to_ground",
        default=False,
        action="store_true",
        help="Translate human foot targets so the lowest foot is 0.1 m above the ground.",
    )
    parser.add_argument(
        "--show_skeleton",
        action="store_true",
        help="Overlay a colored SMPL-X stick skeleton instead of axis frames.",
    )
    parser.add_argument(
        "--align_ground", action="store_true",
        help="MOS only: precompute the clip and apply one constant foot-mesh floor correction.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Export without opening the motion viewer; requires --save_path.",
    )
    parser.add_argument(
        "--flat_feet", action="store_true",
        help="MOS only: precompute flat, stationary support soles with contact validation.",
    )
    parser.add_argument(
        "--contact_schedule",
        help="Optional JSON N-by-2 boolean contact_mask [left, right]; requires --flat_feet.",
    )
    parser.add_argument(
        "--rigid_torso", action="store_true",
        help="MOS flat-foot experiment: soften pelvis lean/twist and upper-body tracking.",
    )

    args = parser.parse_args()
    if args.align_ground and (args.robot != "mos" or args.offset_to_ground):
        parser.error("--align_ground requires --robot mos and cannot be combined with --offset_to_ground")
    if args.headless and (not args.save_path or args.loop or args.record_video):
        parser.error("--headless requires --save_path and cannot use --loop or --record_video")
    if args.flat_feet and (args.robot != "mos" or args.align_ground or args.offset_to_ground):
        parser.error("--flat_feet requires MOS and replaces --align_ground/--offset_to_ground")
    if args.contact_schedule and not args.flat_feet:
        parser.error("--contact_schedule requires --flat_feet")
    if args.rigid_torso and not args.flat_feet:
        parser.error("--rigid_torso requires --robot mos --flat_feet")
    if args.rigid_torso and args.save_path and pathlib.Path(args.save_path).exists():
        parser.error("Rigid-torso experiments require a new --save_path; existing motions are preserved")


    SMPLX_FOLDER = body_model_dir(args.body_model_dir)
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    precomputed = None
    flat_result = None
    if args.flat_feet:
        from general_motion_retargeting.utils.mos_contacts import retarget_mos_flat_feet, FlatFootError
        contacts = None
        if args.contact_schedule:
            with open(args.contact_schedule) as stream:
                schedule = json.load(stream)
            contacts = np.asarray(schedule["contact_mask"])
        report_path = (
            pathlib.Path(args.save_path).with_suffix(".contacts.json") if args.save_path
            else MOTIONS / "gmr" / ("mos_rigid_torso" if args.rigid_torso else "mos_flat_feet")
            / (pathlib.Path(args.smplx_file).stem + ".contacts.json")
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        print("Solving MOS flat-foot contacts (offline)...")
        try:
            flat_result = retarget_mos_flat_feet(
                retarget, smplx_data_frames, aligned_fps, contacts, rigid_torso=args.rigid_torso,
            )
        except FlatFootError as exc:
            exc.report["source"] = str(pathlib.Path(args.smplx_file).resolve())
            report_path.write_text(json.dumps(exc.report, indent=2) + "\n")
            raise SystemExit(f"{exc}. No motion exported. Diagnostic report: {report_path}") from exc
        flat_result.report["source"] = str(pathlib.Path(args.smplx_file).resolve())
        report_path.write_text(json.dumps(flat_result.report, indent=2) + "\n")
        precomputed = flat_result.qpos
        print(f"Contact checks passed. Report: {report_path}")
    elif args.align_ground:
        from general_motion_retargeting.utils.mos import retarget_mos_clip
        print("Solving MOS clip and measuring foot meshes...")
        precomputed, _, ground_shift = retarget_mos_clip(
            retarget, smplx_data_frames, align_ground=True
        )
        retarget.set_ground_offset(-ground_shift)
        print(f"Constant floor correction: {ground_shift:+.6f} m")
    
    robot_motion_viewer = None if args.headless else RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = -1 if args.robot == "mos" else 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        if precomputed is None:
            qpos = retarget.retarget(smplx_data, offset_to_ground=args.offset_to_ground)
        else:
            qpos = precomputed[i]
            if flat_result is not None:
                retarget.scaled_human_data = flat_result.targets[i]
            else:
                retarget.update_targets(smplx_data)

        visual_data = (
            build_visual_skeleton(smplx_data, retarget, args.offset_to_ground)
            if args.show_skeleton
            else retarget.scaled_human_data
        )

        # visualize
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=visual_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                human_skeleton=args.show_skeleton,
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
