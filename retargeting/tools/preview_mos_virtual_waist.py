"""Isolated kinematic virtual-waist comparison; never a real-MOS training export."""

import argparse
import hashlib
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from wbmr.paths import ROOT, MOTIONS, ROBOTS, body_model_dir

DEFAULT_DIR = MOTIONS / "gmr/mos_virtual_waist"
BASELINE = MOTIONS / "gmr/mos_rigid_torso/walk_v2.pkl"
MODEL = ROBOTS / "mos/retargeting/MOS9.2.xml"
UPPER_JOINTS = ("b_n", "n_h", "b_Rs", "Rs_Ra", "Ra_Rh", "b_Ls", "Ls_La", "La_Lh")


def model_xml(source=MODEL):
    """Move the intact torso and its upper chains onto a new ball-joint body."""
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "MOS_VIRTUAL_WAIST_PREVIEW_ONLY")
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", ".")).resolve()))
    pelvis = root.find("./worldbody/body[@name='body']")
    torso = ET.Element("body", name="virtual_torso")
    ET.SubElement(torso, "joint", name="virtual_waist", type="ball", limited="false")
    for child in list(pelvis):
        if child.tag in ("inertial", "geom") or (
            child.tag == "body" and child.get("name") in ("neck", "Rshoulder", "Lshoulder")
        ):
            pelvis.remove(child)
            torso.append(child)
    # Compiler-only placeholder: the invented pelvis has no physical mass model.
    ET.SubElement(pelvis, "inertial", pos="0 0 0", mass="0.001",
                  diaginertia="0.000001 0.000001 0.000001")
    pelvis.append(torso)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def map_baseline(original, virtual, qpos):
    """Map by joint name; the new ball quaternion is not a scalar joint angle."""
    result = np.tile(virtual.qpos0, (len(qpos), 1))
    result[:, :7] = qpos[:, :7]
    for j in range(1, original.njnt):
        joint = original.joint(j)
        result[:, virtual.joint(joint.name).qposadr[0]] = qpos[:, joint.qposadr[0]]
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(directory, baseline, source, models):
    import mujoco as mj
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.mos import MosTargets
    from general_motion_retargeting.utils.mos_contacts import Sole
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

    if directory.exists():
        raise FileExistsError(f"Use a new experiment directory; preserving {directory}")
    contact_path = baseline.with_suffix(".contacts.json")
    contact_report = json.loads(contact_path.read_text())
    if contact_report["status"] != "passed":
        raise ValueError("Baseline must have passed contact validation")
    source = Path(source).expanduser().resolve()
    protected = [MODEL, baseline, contact_path, source]
    hashes = {str(p): sha256(p) for p in protected}
    with baseline.open("rb") as stream:
        motion = pickle.load(stream)  # Trusted local GMR export only.
    baseline_q = np.column_stack((
        motion["root_pos"], motion["root_rot"][:, [3, 0, 1, 2]], motion["dof_pos"],
    ))
    fps = float(motion["fps"])
    if baseline_q.shape[1] != 27 or not np.isfinite(baseline_q).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid real-MOS baseline")
    np.testing.assert_allclose(np.linalg.norm(baseline_q[:, 3:7], axis=1), 1, atol=1e-8)
    smpl, body, output, height = load_smplx_file(source, models)
    frames, source_fps = get_smplx_data_offline_fast(smpl, body, output, tgt_fps=30)
    if len(frames) != len(baseline_q) or not np.isclose(fps, source_fps):
        raise ValueError("Human clip and baseline must have matching frames and FPS")
    gmr = GMR("smplx", "mos", actual_human_height=height, verbose=False)
    original = gmr.model
    xml = model_xml()
    model = mj.MjModel.from_xml_string(xml)
    data, original_data = mj.MjData(model), mj.MjData(original)
    rigid = map_baseline(original, model, baseline_q)
    candidate = rigid.copy()
    waist_adr = model.joint("virtual_waist").qposadr[0]
    upper_adr = np.asarray([model.joint(name).qposadr[0] for name in UPPER_JOINTS])
    chest = R.from_quat([f["spine3"][1] for f in frames], scalar_first=True) * MosTargets.HUMAN_TO_ROBOT.inv()
    pelvis = R.from_quat(baseline_q[:, 3:7], scalar_first=True)
    waist = pelvis.inv() * chest
    candidate[:, waist_adr:waist_adr + 4] = waist.as_quat(scalar_first=True)
    if np.rad2deg(waist.magnitude()).max() > 45:
        raise ValueError("Virtual waist exceeds the 45-degree preview sanity gate; no clamping applied")
    tracking = (("head", "head", 3), ("right_shoulder", "Rarm", 5),
                ("right_elbow", "Rhand", 5), ("left_shoulder", "Larm", 5),
                ("left_elbow", "Lhand", 5))
    body_ids = [model.body(name).id for _, name, _ in tracking]
    errors = {"rigid": [], "virtual": []}
    max_foot_difference = 0.0
    stance_height, stance_tilt, lowest = [], [], []
    soles = [Sole.from_model(model, prefix) for prefix in ("L", "R")]
    contacts = np.asarray(contact_report["contact_mask"], dtype=bool)
    if contacts.shape != (len(frames), 2) or not contacts.any():
        raise ValueError("Invalid baseline contact schedule")
    previous = rigid[0, upper_adr].copy()
    for i, frame in enumerate(frames):
        target = gmr.target_builder(frame)
        desired = [R.from_quat(target[name][1], scalar_first=True) for name, _, _ in tracking]

        def residual(upper):
            data.qpos[:] = candidate[i]
            data.qpos[upper_adr] = upper
            mj.mj_forward(model, data)
            terms = [
                np.sqrt(weight) * (goal.inv() * R.from_quat(data.xquat[bid], scalar_first=True)).as_rotvec()
                for goal, bid, (_, _, weight) in zip(desired, body_ids, tracking)
            ]
            return np.concatenate(terms + [
                np.sqrt(.5) * (upper - rigid[i, upper_adr]), upper - previous,
            ])

        solution = least_squares(residual, previous, max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if not solution.success or not np.isfinite(solution.x).all():
            raise RuntimeError(f"Upper-body IK failed at frame {i}: {solution.message}")
        candidate[i, upper_adr] = solution.x
        previous = solution.x.copy()
        for label, q in (("rigid", rigid[i]), ("virtual", candidate[i])):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            actual_chest = R.from_quat(data.xquat[model.body("virtual_torso").id], scalar_first=True)
            errors[label].append(float(np.rad2deg((chest[i].inv() * actual_chest).magnitude())))
        original_data.qpos[:] = baseline_q[i]
        mj.mj_forward(original, original_data)
        for side, sole in enumerate(soles):
            name = model.body(sole.body_id).name
            oid = original.body(name).id
            max_foot_difference = max(max_foot_difference,
                                     float(np.linalg.norm(data.xpos[sole.body_id] - original_data.xpos[oid])))
            np.testing.assert_allclose(data.xmat[sole.body_id], original_data.xmat[oid], atol=1e-12)
            lowest.append(float(sole.points(data, vertices=True)[:, 2].min()))
            if contacts[i, side]:
                stance_height.append(float(np.abs(sole.points(data)[:, 2]).max()))
                _, rotation = sole.pose(data.xpos[sole.body_id], R.from_quat(data.xquat[sole.body_id], scalar_first=True))
                stance_tilt.append(float(np.rad2deg(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1)))))
    np.testing.assert_array_equal(candidate[:, :7], rigid[:, :7])
    for joint in range(1, original.njnt):
        name = original.joint(joint).name
        if name not in UPPER_JOINTS:
            adr = model.joint(name).qposadr[0]
            np.testing.assert_array_equal(candidate[:, adr], rigid[:, adr])
    if max_foot_difference > 1e-10 or max(stance_height) > .001 or max(stance_tilt) > .5 or min(lowest) < -.001:
        raise RuntimeError("Virtual-waist comparison did not preserve valid foot contacts")
    if not np.isfinite(candidate).all():
        raise RuntimeError("Non-finite virtual motion")
    np.testing.assert_allclose(np.linalg.norm(candidate[:, waist_adr:waist_adr + 4], axis=1), 1, atol=1e-8)
    for path in protected:
        if sha256(path) != hashes[str(path)]:
            raise RuntimeError(f"Protected input changed: {path}")
    report = {
        "status": "kinematic_comparison_only", "training_compatible": False,
        "source": str(source), "baseline": str(baseline), "frames": len(frames), "fps": fps,
        "waist_type": "ball, 3 rotational DOFs, WXYZ quaternion",
        "waist_pivot": "original MOS body origin", "chest_source": "SMPL-X spine3 global orientation",
        "root_and_all_leg_joint_values_identical": True,
        "max_foot_position_difference_m": max_foot_difference,
        "max_stance_corner_height_m": max(stance_height),
        "max_stance_tilt_deg": max(stance_tilt), "min_sole_mesh_height_m": min(lowest),
        "max_waist_rotation_deg": float(np.rad2deg(waist.magnitude()).max()),
        "waist_peak_abs_xyz_euler_deg": np.abs(waist.as_euler("xyz", degrees=True)).max(axis=0).tolist(),
        "chest_tracking_rms_deg": {key: float(np.sqrt(np.mean(np.square(val)))) for key, val in errors.items()},
        "max_upper_joint_step_rad": float(np.abs(np.diff(candidate[:, upper_adr], axis=0)).max()),
        "preserved_sha256": hashes,
        "limitations": [
            "Invented waist: cannot execute this motion on real MOS or train its existing asset from it.",
            "Pelvis and legs are frozen to the rigid-torso candidate to isolate torso freedom, not reoptimized.",
            "Chest follows the human exactly by construction; this is not evidence of improved balance.",
            "Torso mesh is intact and pivots above the hip chains; this is not a CAD design for a waist.",
            "Invented pelvis inertia is only a compiler placeholder; no physics is simulated.",
            "Original knee transition and nonperiodic clip boundary remain unchanged.",
            "Upper-body self-collision and physical joint limits are not validated.",
        ],
    }
    directory.mkdir(parents=True)
    saved_xml = ET.fromstring(xml)
    saved_xml.find("compiler").set("meshdir", os.path.relpath(MODEL.parent / "meshes", directory))
    ET.ElementTree(saved_xml).write(directory / "model.xml", encoding="unicode")
    np.savez_compressed(directory / "comparison.npz", virtual_qpos=candidate, rigid_qpos=rigid,
                        fps=fps, contact_mask=contacts, format="mos_virtual_waist_preview_v1",
                        training_compatible=False)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def load(directory):
    import mujoco as mj
    with np.load(directory / "comparison.npz", allow_pickle=False) as archive:
        if str(archive["format"]) != "mos_virtual_waist_preview_v1":
            raise ValueError("Not a virtual-waist preview")
        poses = {name: archive[f"{name}_qpos"].copy() for name in ("rigid", "virtual")}
        fps = float(archive["fps"])
    model = mj.MjModel.from_xml_path(str(directory / "model.xml"))
    for q in poses.values():
        if q.shape[1] != model.nq or not np.isfinite(q).all():
            raise ValueError("Invalid motion/model pairing")
    return model, poses, fps


def camera(cam, root_pos, heading, side=False):
    cam.lookat[:] = [root_pos[0], root_pos[1], .55]
    cam.distance, cam.elevation = 1.8, -12
    cam.azimuth = heading + (90 if side else 0)


def render(directory):
    import imageio.v2 as imageio
    import mujoco as mj
    from PIL import Image, ImageDraw
    model, poses, fps = load(directory)
    data = mj.MjData(model)
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    heading = float(np.rad2deg(np.unwrap(R.from_quat(poses["rigid"][:, 3:7], scalar_first=True).as_euler("xyz")[:, 2])).mean())
    samples = []
    with mj.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(
        str(directory / "comparison.mp4"), fps=fps, codec="libx264", quality=8,
    ) as writer:
        for i in range(len(poses["rigid"])):
            image = Image.new("RGB", (1280, 960))
            for row in range(2):
                for col, name in enumerate(("rigid", "virtual")):
                    data.qpos[:] = poses[name][i]
                    mj.mj_forward(model, data)
                    cam = mj.MjvCamera()
                    camera(cam, data.qpos[:3], heading, side=row == 1)
                    renderer.update_scene(data, camera=cam, scene_option=option)
                    pixels = renderer.render()
                    if pixels.std() < 2:
                        raise RuntimeError("Blank comparison render")
                    panel = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, 640, 36), fill=(20, 20, 20))
                    label = "REAL MOS: rigid torso" if name == "rigid" else "VIRTUAL WAIST: not hardware-executable"
                    draw.text((12, 12), f"{label} | frame {i}", fill="white")
                    image.paste(panel, (col * 640, row * 480))
            writer.append_data(np.asarray(image))
            if i == 100:
                image.save(directory / "comparison.jpg")
            if i in (30, 100):
                samples.append(np.asarray(image).astype(float))
    if len(samples) == 2 and np.mean(np.abs(samples[1] - samples[0])) < 1:
        raise RuntimeError("Comparison motion did not render")
    print(f"Rendered {directory / 'comparison.mp4'}")


def play(directory, mode, cycles, view):
    import mujoco as mj
    import mujoco.viewer
    model, poses, fps = load(directory)
    data = mj.MjData(model)
    state = {"mode": mode, "side": view == "side"}

    def key_callback(key):
        if key == 86:
            state["mode"] = "rigid" if state["mode"] == "virtual" else "virtual"
            print(f"Showing {state['mode']}", flush=True)
        elif key == 70:
            state["side"] = not state["side"]

    heading = float(np.rad2deg(np.unwrap(R.from_quat(poses["rigid"][:, 3:7], scalar_first=True).as_euler("xyz")[:, 2])).mean())
    print("PREVIEW ONLY. V: toggle virtual/rigid; F: front/side. Close window or Ctrl+C to stop.", flush=True)
    count = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.opt.geomgroup[3] = 0
        while viewer.is_running() and (cycles == 0 or count < cycles * len(poses["rigid"])):
            start = time.monotonic()
            data.qpos[:] = poses[state["mode"]][count % len(poses["rigid"])]
            mj.mj_forward(model, data)
            camera(viewer.cam, data.qpos[:3], heading, state["side"])
            viewer.sync()
            time.sleep(max(0, 1 / fps - (time.monotonic() - start)))
            count += 1
    # Let the passive viewer's render thread release GL resources before exit.
    time.sleep(.5)
    print(f"Played {count} frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--build", action="store_true", help="Create a new experiment, then exit unless --render")
    parser.add_argument("--smplx_file", type=Path, help="Authorized human source, required for --build.")
    parser.add_argument("--body_model_dir", help="Licensed model directory; defaults to SMPLX_MODEL_DIR.")
    parser.add_argument("--render", action="store_true", help="Write front/side comparison video, then exit")
    parser.add_argument("--mode", choices=("rigid", "virtual"), default="virtual")
    parser.add_argument("--view", choices=("front", "side"), default="front")
    parser.add_argument("--cycles", type=int, default=0, help="0 loops until closed")
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles cannot be negative")
    if args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.build:
        if args.smplx_file is None:
            parser.error("--build requires --smplx_file; human recordings are not bundled")
        build(args.directory.resolve(), args.baseline.resolve(), args.smplx_file,
              body_model_dir(args.body_model_dir))
    if args.render:
        render(args.directory.resolve())
    elif not args.build:
        try:
            play(args.directory.resolve(), args.mode, args.cycles, args.view)
        except KeyboardInterrupt:
            time.sleep(.5)


if __name__ == "__main__":
    main()
