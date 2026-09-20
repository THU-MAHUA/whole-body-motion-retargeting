"""Regression checks for opt-in rigid-torso targets and constrained solves."""

import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.mos import MosTargets
from general_motion_retargeting.utils.mos_contacts import retarget_mos_flat_feet
from general_motion_retargeting.utils.mos_torso import adapt_rigid_torso
from test_mos_retarget import neutral_human


class RigidTorsoTest(unittest.TestCase):
    def setUp(self):
        self.gmr = GMR("smplx", "mos", verbose=False)

    def targets(self, n=80):
        targets = []
        for i in range(n):
            frame = neutral_human()
            frame["pelvis"][1] = (
                R.from_euler("xyz", [.2 * np.sin(i * .4), .15, 2.8 + i * .01 + .1 * np.sin(i * .4)])
                * MosTargets.HUMAN_TO_ROBOT
            ).as_quat(scalar_first=True)
            targets.append(self.gmr.target_builder(frame))
        return targets

    def test_preserves_inputs_feet_legs_and_translation(self):
        targets = self.targets()
        before = copy.deepcopy(targets)
        adapted, report = adapt_rigid_torso(targets, 30)
        for i in range(len(targets)):
            for name in targets[i]:
                for a, b in zip(targets[i][name], before[i][name]):
                    np.testing.assert_array_equal(a, b)
                if name.endswith(("_hip", "_knee", "_ankle")):
                    for a, b in zip(targets[i][name], adapted[i][name]):
                        np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(targets[i]["pelvis"][0], adapted[i]["pelvis"][0])
        self.assertLess(report["target_roll_pitch_peak_deg"][0], 8)
        self.assertLess(report["target_roll_pitch_peak_deg"][1], 10)

    def test_wrap_and_turn_heading_and_upper_geometry(self):
        targets = self.targets()
        adapted, _ = adapt_rigid_torso(targets, 30)
        before = R.from_quat([t["pelvis"][1] for t in targets], scalar_first=True)
        after = R.from_quat([t["pelvis"][1] for t in adapted], scalar_first=True)
        y0 = np.unwrap(before.as_euler("xyz")[:, 2])
        y1 = np.unwrap(after.as_euler("xyz")[:, 2])
        np.testing.assert_allclose(y0[[0, -1]], y1[[0, -1]], atol=1e-12)
        self.assertLess(np.abs(np.diff(y1)).max(), .05)
        for i in range(len(targets)):
            for name in ("head", "left_shoulder", "right_shoulder"):
                a = before[i].inv().apply(targets[i][name][0] - targets[i]["pelvis"][0])
                b = after[i].inv().apply(adapted[i][name][0] - adapted[i]["pelvis"][0])
                np.testing.assert_allclose(a, b, atol=1e-12)

    def test_global_heading_equivariance(self):
        targets = self.targets()
        yaw = R.from_euler("z", 1.2)
        turned = [{name: [yaw.apply(p), (yaw * R.from_quat(q, scalar_first=True)).as_quat(scalar_first=True)]
                   for name, (p, q) in frame.items()} for frame in targets]
        a, _ = adapt_rigid_torso(targets, 30)
        b, _ = adapt_rigid_torso(turned, 30)
        for first, second in zip(a, b):
            for name in first:
                np.testing.assert_allclose(yaw.apply(first[name][0]), second[name][0], atol=1e-12)
                expected = yaw * R.from_quat(first[name][1], scalar_first=True)
                actual = R.from_quat(second[name][1], scalar_first=True)
                np.testing.assert_allclose(expected.as_matrix(), actual.as_matrix(), atol=1e-12)

    def test_alternating_twist_reduced_without_removing_lean(self):
        targets = self.targets(n=240)
        adapted, _ = adapt_rigid_torso(targets, 30)
        before = R.from_quat([t["pelvis"][1] for t in targets], scalar_first=True)
        after = R.from_quat([t["pelvis"][1] for t in adapted], scalar_first=True)
        a = np.unwrap(before.as_euler("xyz")[:, 2])
        b = np.unwrap(after.as_euler("xyz")[:, 2])
        # Compare curvature away from the finite clip's endpoints.
        self.assertLess(np.std(np.diff(b[30:-30], n=2)),
                        .2 * np.std(np.diff(a[30:-30], n=2)))
        pitch = after.as_euler("xyz")[:, 1]
        self.assertTrue(np.all(pitch > 0))
        self.assertTrue(np.all(pitch < .15))

    def test_constrained_solve_and_default_unchanged(self):
        frames = [neutral_human() for _ in range(3)]
        contact = np.ones((3, 2), dtype=bool)
        a = retarget_mos_flat_feet(self.gmr, frames, 30, contact)
        b = retarget_mos_flat_feet(GMR("smplx", "mos", verbose=False), frames, 30, contact, rigid_torso=False)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        c = retarget_mos_flat_feet(GMR("smplx", "mos", verbose=False), frames, 30, contact, rigid_torso=True)
        self.assertEqual(c.report["status"], "passed")
        self.assertLess(c.report["max_stance_tilt_deg"], .5)
        self.assertLess(c.report["max_stance_drift_m"], .001)
        self.assertEqual(c.qpos.shape, (3, 27))
        self.assertIn("torso_adaptation", c.report)


if __name__ == "__main__":
    unittest.main()
