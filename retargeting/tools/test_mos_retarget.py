"""Regression tests for anatomical MOS targets, independent of motion archives."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.mos import MosTargets, retarget_mos_clip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def neutral_human(scale=0.5):
    quat = MosTargets.HUMAN_TO_ROBOT.as_quat(scalar_first=True)
    return {
        name: [np.array([0.0, 0.0, 0.55 / scale]), quat.copy()]
        for name in (*MosTargets.BODY_MAP, "spine3")
    }


class MosRetargetTest(unittest.TestCase):
    def setUp(self):
        self.gmr = GMR("smplx", "mos", verbose=False)

    def test_neutral_frames_and_ik(self):
        human = neutral_human()
        targets = self.gmr.target_builder(human)
        for name, body in MosTargets.BODY_MAP.items():
            np.testing.assert_allclose(
                targets[name][0],
                self.gmr.target_builder.positions[body] + [0, 0, 0.55],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                R.from_quat(targets[name][1], scalar_first=True).as_matrix(),
                self.gmr.target_builder.rotations[name].as_matrix(),
                atol=1e-12,
            )
        q = self.gmr.retarget(human)
        np.testing.assert_allclose(q[7:], 0, atol=1e-8)
        np.testing.assert_allclose(q[:3], [0, 0, 0.55], atol=1e-8)

    def test_left_knee_bends_without_right_leg_motion(self):
        human = neutral_human()
        self.gmr.retarget(human)
        quat = (
            R.from_rotvec([0, 0.3, 0]) * MosTargets.HUMAN_TO_ROBOT
        ).as_quat(scalar_first=True)
        human["left_knee"][1] = quat
        human["left_ankle"][1] = quat
        for _ in range(20):
            q = self.gmr.retarget(human)
        jid = self.gmr.model.joint("Ll1_Ll2").qposadr[0]
        self.assertAlmostEqual(q[jid], -0.3, delta=0.005)
        right = self.gmr.model.joint("Rl1_Rl2").qposadr[0]
        self.assertAlmostEqual(q[right], 0, delta=0.005)

    def test_global_yaw_equivariance(self):
        human = neutral_human()
        original = self.gmr.target_builder(human)
        yaw = R.from_euler("z", 0.7)
        shifted = {
            n: [yaw.apply(p), (yaw * R.from_quat(q, scalar_first=True)).as_quat(scalar_first=True)]
            for n, (p, q) in human.items()
        }
        targets = self.gmr.target_builder(shifted)
        for name, (pos, quat) in original.items():
            np.testing.assert_allclose(targets[name][0], yaw.apply(pos), atol=1e-12)
            np.testing.assert_allclose(
                R.from_quat(targets[name][1], scalar_first=True).as_matrix(),
                (yaw * R.from_quat(quat, scalar_first=True)).as_matrix(),
                atol=1e-12,
            )

    def test_human_spine_lean_does_not_move_rigid_torso_targets(self):
        human = neutral_human()
        original = self.gmr.target_builder(human)
        human["spine3"][1] = (
            R.from_euler("xyz", [0.2, 0.5, -0.1]) * MosTargets.HUMAN_TO_ROBOT
        ).as_quat(scalar_first=True)
        targets = self.gmr.target_builder(human)
        for name in original:
            for actual, expected in zip(targets[name], original[name]):
                np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(self.gmr.retarget(human)[7:], 0, atol=1e-8)

    def test_torso_landmarks_follow_pelvis_tilt(self):
        human = neutral_human()
        tilt = R.from_euler("xyz", [0.2, -0.3, 0.4])
        human["pelvis"][1] = (
            tilt * MosTargets.HUMAN_TO_ROBOT
        ).as_quat(scalar_first=True)
        targets = self.gmr.target_builder(human)
        neutral = self.gmr.target_builder.positions
        for name, body in (("head", "head"), ("left_shoulder", "Larm"),
                           ("right_shoulder", "Rarm")):
            np.testing.assert_allclose(
                targets[name][0] - targets["pelvis"][0],
                tilt.apply(neutral[body] - neutral["body"]),
                atol=1e-12,
            )

    def test_head_orientation_and_overlay_are_preserved(self):
        from scripts.smplx_to_robot import build_visual_skeleton

        human = neutral_human()
        turn = R.from_euler("zy", [0.3, 0.2])
        human["head"][1] = (
            turn * MosTargets.HUMAN_TO_ROBOT
        ).as_quat(scalar_first=True)
        self.gmr.set_ground_offset(-0.07)
        self.gmr.update_targets(human)
        visual = build_visual_skeleton(human, self.gmr, False)
        expected = turn * self.gmr.target_builder.rotations["head"]
        np.testing.assert_allclose(
            R.from_quat(visual["head"][1], scalar_first=True).as_matrix(),
            expected.as_matrix(), atol=1e-12,
        )
        for name in ("pelvis", "left_shoulder", "right_shoulder"):
            np.testing.assert_allclose(visual[name][0], self.gmr.scaled_human_data[name][0])
        np.testing.assert_allclose(
            visual["head"][0] - self.gmr.scaled_human_data["head"][0], [0, 0.033, 0],
            atol=1e-12,
        )
        np.testing.assert_allclose(visual["pelvis"][0], [0, 0, 0.62])
        for name, fraction in (("spine1", 0.25), ("spine2", 0.5),
                               ("spine3", 0.7), ("neck", 0.9)):
            np.testing.assert_allclose(
                visual[name][0],
                (1 - fraction) * visual["pelvis"][0] + fraction * visual["head"][0],
            )

    def test_centered_overlay_is_display_only_under_pelvis_rotation(self):
        from scripts.smplx_to_robot import build_visual_skeleton

        reference = GMR("smplx", "mos", verbose=False)
        for angles in ([0, 0, 0], [0.12, -0.2, 1.1], [-0.15, 0.08, -2.0]):
            human = neutral_human()
            rotation = R.from_euler("xyz", angles)
            human["pelvis"][1] = (
                rotation * MosTargets.HUMAN_TO_ROBOT
            ).as_quat(scalar_first=True)
            human["head"][1] = (
                rotation * R.from_euler("zy", [-0.25, 0.15]) * MosTargets.HUMAN_TO_ROBOT
            ).as_quat(scalar_first=True)
            source = copy.deepcopy(human)
            qpos = self.gmr.retarget(human).copy()
            np.testing.assert_array_equal(qpos, reference.retarget(human))
            targets = copy.deepcopy(self.gmr.scaled_human_data)
            visual = build_visual_skeleton(human, self.gmr, False)
            local_before = rotation.inv().apply(targets["head"][0] - targets["pelvis"][0])
            local_after = rotation.inv().apply(visual["head"][0] - visual["pelvis"][0])
            np.testing.assert_allclose(local_after[[0, 2]], local_before[[0, 2]], atol=1e-12)
            self.assertAlmostEqual(local_after[1], 0.0, places=12)
            for name in ("spine1", "spine2", "spine3", "neck"):
                offset = rotation.inv().apply(visual[name][0] - visual["pelvis"][0])
                self.assertAlmostEqual(offset[1], 0.0, places=12)
            for name in targets:
                for actual, expected in zip(self.gmr.scaled_human_data[name], targets[name]):
                    np.testing.assert_array_equal(actual, expected)
                displayed = name.replace("_ankle", "_foot")
                np.testing.assert_array_equal(visual[displayed][1], targets[name][1])
                if name != "head":
                    np.testing.assert_array_equal(visual[displayed][0], targets[name][0])
            for name in human:
                for actual, expected in zip(human[name], source[name]):
                    np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(self.gmr.configuration.q, qpos)

    def test_ground_flag_is_finite(self):
        q = self.gmr.retarget(neutral_human(), offset_to_ground=True)
        self.assertTrue(np.isfinite(q).all())

    def test_constant_alignment_preserves_motion(self):
        frames = [neutral_human() for _ in range(3)]
        for frame, dz in zip(frames, (0, 0.1, 0.04)):
            frame["pelvis"][0][2] += dz
        original, heights, _ = retarget_mos_clip(self.gmr, frames)
        other = GMR("smplx", "mos", verbose=False)
        aligned, raw_heights, shift = retarget_mos_clip(other, frames, align_ground=True)
        np.testing.assert_allclose(raw_heights, heights, atol=1e-10)
        np.testing.assert_allclose(aligned[:, 3:], original[:, 3:], atol=1e-10)
        np.testing.assert_allclose(aligned[:, :2], original[:, :2], atol=1e-10)
        np.testing.assert_allclose(aligned[:, 2] - original[:, 2], shift, atol=1e-10)
        np.testing.assert_allclose(np.diff(aligned[:, 2]), np.diff(original[:, 2]), atol=1e-10)
        self.assertAlmostEqual(float((raw_heights + shift).min()), 0)

    def test_first_frame_initialization(self):
        human = neutral_human()
        arm = (R.from_rotvec([1.2, 0, 0]) * MosTargets.HUMAN_TO_ROBOT).as_quat(scalar_first=True)
        human["left_shoulder"][1] = arm
        human["left_elbow"][1] = arm
        first = self.gmr.retarget(human)
        for _ in range(20):
            settled = self.gmr.retarget(human)
        np.testing.assert_allclose(first, settled, atol=0.01)

    def test_height_scaling_is_inverse_for_mos_only(self):
        small = GMR("smplx", "mos", actual_human_height=1.5, verbose=False)
        tall = GMR("smplx", "mos", actual_human_height=2.0, verbose=False)
        self.assertGreater(small.target_builder.root_scale, tall.target_builder.root_scale)
        k1 = GMR("smplx", "booster_k1", actual_human_height=1.8, verbose=False)
        self.assertIsNone(k1.target_builder)
        import json
        from general_motion_retargeting.params import IK_CONFIG_DICT
        with open(IK_CONFIG_DICT["smplx"]["booster_k1"]) as stream:
            config = json.load(stream)
        for name, scale in config["human_scale_table"].items():
            self.assertAlmostEqual(
                k1.human_scale_table[name], scale * 1.8 / config["human_height_assumption"]
            )
        human = {
            name: [np.array([0.1, 0.2, 1.0]), np.array([1.0, 0, 0, 0])]
            for name in k1.human_scale_table
        }
        expected = k1.scale_human_data(copy.deepcopy(human), k1.human_root_name, k1.human_scale_table)
        expected = k1.offset_human_data(expected, k1.pos_offsets1, k1.rot_offsets1)
        expected = k1.apply_ground_offset(expected)
        k1.update_targets(human)
        for name in expected:
            for actual, reference in zip(k1.scaled_human_data[name], expected[name]):
                np.testing.assert_array_equal(actual, reference)


if __name__ == "__main__":
    unittest.main()
