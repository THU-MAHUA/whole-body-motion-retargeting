"""Check an exported MOS flat-foot motion and render front/side transitions."""

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.mos_contacts import Sole, intervals


def validate(path):
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    report = json.loads(path.with_suffix(".contacts.json").read_text())
    assert report["status"] == "passed"
    qpos = np.column_stack((
        motion["root_pos"], motion["root_rot"][:, [3, 0, 1, 2]], motion["dof_pos"],
    ))
    contacts = np.asarray(report["contact_mask"])
    assert qpos.shape == (report["frames"], 27)
    assert contacts.shape == (len(qpos), 2) and contacts.dtype == np.bool_
    assert contacts.any()
    assert np.isfinite(qpos).all() and np.isfinite(motion["fps"]) and motion["fps"] > 0
    np.testing.assert_allclose(motion["fps"], report["fps"])
    np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-8)
    model = GMR("smplx", "mos", verbose=False).model
    data = mj.MjData(model)
    soles = [Sole.from_model(model, prefix) for prefix in ("L", "R")]
    positions, heights, tilts, mesh_min = [], [], [], []
    for q in qpos:
        data.qpos[:] = q
        mj.mj_forward(model, data)
        frame_positions, frame_heights, frame_tilts = [], [], []
        for sole in soles:
            center, rotation = sole.pose(
                data.xpos[sole.body_id],
                R.from_quat(data.xquat[sole.body_id], scalar_first=True),
            )
            frame_positions.append(center)
            frame_heights.append(np.abs(sole.points(data)[:, 2]).max())
            frame_tilts.append(np.rad2deg(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1))))
            mesh_min.append(sole.points(data, vertices=True)[:, 2].min())
        positions.append(frame_positions)
        heights.append(frame_heights)
        tilts.append(frame_tilts)
    positions = np.asarray(positions)
    maximum_height = float(np.asarray(heights)[contacts].max())
    maximum_tilt = float(np.asarray(tilts)[contacts].max())
    maximum_drift = 0.0
    for s in range(2):
        for start, end in intervals(contacts[:, s]):
            maximum_drift = max(
                maximum_drift,
                float(np.linalg.norm(positions[start:end, s] - positions[start, s], axis=1).max()),
            )
    assert maximum_height < 0.001 and maximum_tilt < 0.5 and maximum_drift < 0.001
    assert min(mesh_min) >= -0.001
    step = np.abs(np.diff(qpos[:, 7:], axis=0))
    index, joint = np.unravel_index(step.argmax(), step.shape)
    index += 1
    result = {
        "frames": len(qpos), "fps": float(motion["fps"]),
        "max_stance_corner_height_m": maximum_height,
        "max_stance_tilt_deg": maximum_tilt,
        "max_stance_drift_from_contact_start_m": maximum_drift,
        "min_mesh_height_m": float(min(mesh_min)),
        "max_joint_step_rad": float(step.max()),
        "max_joint_step_frame": int(index),
        "max_joint_step_name": model.joint(int(joint) + 1).name,
        "status": "passed",
    }
    transitions = np.flatnonzero(np.any(contacts[1:] != contacts[:-1], axis=1)) + 1
    transition = int(transitions[len(transitions) // 2]) if len(transitions) else len(qpos) // 2
    indices = np.clip([0, index - 1, index, transition - 1, transition, len(qpos) - 1], 0, len(qpos) - 1)
    montage = Image.new("RGB", (640 * 3, 480 * 4))
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    with mj.Renderer(model, height=480, width=640) as renderer:
        for row, view in enumerate((0, 90)):
            for col, frame in enumerate(indices):
                data.qpos[:] = qpos[frame]
                mj.mj_forward(model, data)
                camera = mj.MjvCamera()
                camera.lookat[:] = qpos[frame, :3] + [0, 0, -0.05]
                camera.distance, camera.elevation = 1.65, -12
                forward = data.xmat[model.body("body").id].reshape(3, 3)[:, 0]
                camera.azimuth = np.rad2deg(np.arctan2(forward[1], forward[0])) + view
                renderer.update_scene(data, camera=camera, scene_option=option)
                pixels = renderer.render()
                assert pixels.std() > 2, "Blank render"
                image = Image.fromarray(pixels)
                support = "/".join(side for s, side in enumerate(("L", "R")) if contacts[frame, s]) or "flight"
                ImageDraw.Draw(image).text(
                    (12, 12), f"frame {frame} / view {view} / stance {support}", fill="white",
                )
                montage.paste(image, (col % 3 * 640, (row * 2 + col // 3) * 480))
    montage.save(path.with_suffix(".poses.jpg"))
    path.with_suffix(".validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Trusted local MOS flat-foot pickle")
    validate(parser.parse_args().motion)
