"""SMPL-X anatomical targets calibrated to the MOS URDF zero pose."""

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R


class MosTargets:
    # SMPL-X rest axes: left=X, up=Y, forward=Z. MOS: forward=X, left=Y, up=Z.
    HUMAN_TO_ROBOT = R.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    BODY_MAP = {
        "pelvis": "body",
        "head": "head",
        "left_shoulder": "Larm",
        "right_shoulder": "Rarm",
        "left_elbow": "Lhand",
        "right_elbow": "Rhand",
        "left_hip": "Lleg1",
        "right_hip": "Rleg1",
        "left_knee": "Lleg2",
        "right_knee": "Rleg2",
        "left_ankle": "Lfoot",
        "right_ankle": "Rfoot",
    }

    def __init__(self, model, root_scale):
        self.root_scale = root_scale
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        self.positions = {
            model.body(i).name: data.xpos[i].copy() for i in range(1, model.nbody)
        }
        self.rotations = {
            name: R.from_quat(data.xquat[model.body(body).id], scalar_first=True)
            for name, body in self.BODY_MAP.items()
        }

    def __call__(self, human):
        delta = {
            name: R.from_quat(quat, scalar_first=True) * self.HUMAN_TO_ROBOT.inv()
            for name, (_, quat) in human.items()
        }
        neutral = self.positions
        root = np.asarray(human["pelvis"][0]) * self.root_scale
        positions = {"pelvis": root}
        # MOS has no spine joint: torso-mounted landmarks follow the pelvis.
        positions["head"] = root + delta["pelvis"].apply(
            neutral["head"] - neutral["body"]
        )
        for side, prefix in (("left", "L"), ("right", "R")):
            hip, knee, ankle = (f"{side}_{part}" for part in ("hip", "knee", "ankle"))
            shoulder, elbow = f"{side}_shoulder", f"{side}_elbow"
            # Hip axes are separated in MOS, so rotate the thigh-origin offset too.
            positions[hip] = (
                root
                + delta["pelvis"].apply(neutral[prefix + "hip"] - neutral["body"])
                + delta[hip].apply(neutral[prefix + "leg1"] - neutral[prefix + "hip"])
            )
            positions[knee] = positions[hip] + delta[hip].apply(
                neutral[prefix + "leg2"] - neutral[prefix + "leg1"]
            )
            positions[ankle] = (
                positions[knee]
                + delta[knee].apply(neutral[prefix + "ankle"] - neutral[prefix + "leg2"])
                + delta[ankle].apply(neutral[prefix + "foot"] - neutral[prefix + "ankle"])
            )
            positions[shoulder] = root + delta["pelvis"].apply(
                neutral[prefix + "arm"] - neutral["body"]
            )
            positions[elbow] = positions[shoulder] + delta[shoulder].apply(
                neutral[prefix + "hand"] - neutral[prefix + "arm"]
            )
        return {
            name: [
                positions[name],
                (delta[name] * self.rotations[name]).as_quat(scalar_first=True),
            ]
            for name in self.BODY_MAP
        }


def foot_mesh_heights(model, data):
    """Lowest rendered vertex of each foot; these are not collision proxies."""
    heights = []
    for name in ("Lfoot_visual", "Rfoot_visual"):
        gid = model.geom(name).id
        mid = model.geom_dataid[gid]
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        heights.append(
            np.min(vertices @ data.geom_xmat[gid].reshape(3, 3)[2])
            + data.geom_xpos[gid, 2]
        )
    return np.asarray(heights)


def retarget_mos_clip(retargeter, frames, align_ground=False):
    """Solve a finite clip, optionally applying one constant floor correction."""
    if not isinstance(retargeter.target_builder, MosTargets):
        raise ValueError("MOS clip conversion requires the MOS target builder")
    qpos = np.asarray([retargeter.retarget(frame) for frame in frames])
    if qpos.ndim != 2 or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError("MOS retargeting returned an empty or non-finite motion")
    data = mj.MjData(retargeter.model)
    heights = []
    for q in qpos:
        data.qpos[:] = q
        mj.mj_forward(retargeter.model, data)
        heights.append(foot_mesh_heights(retargeter.model, data))
    heights = np.asarray(heights)
    shift = -float(heights.min()) if align_ground else 0.0
    qpos[:, 2] += shift
    return qpos, heights, shift
