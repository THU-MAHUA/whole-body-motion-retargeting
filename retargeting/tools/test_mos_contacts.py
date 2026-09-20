"""MOS sole geometry, stance scheduling, and constrained IK regression tests."""

import unittest
from unittest.mock import patch

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.mos_contacts import (
    FlatFootError, Sole, detect_contacts, intervals, plan_feet, retarget_mos_flat_feet,
)
from test_mos_retarget import neutral_human


class MosContactsTest(unittest.TestCase):
    def setUp(self):
        self.gmr = GMR("smplx", "mos", verbose=False)
        self.soles = [Sole.from_model(self.gmr.model, p) for p in ("L", "R")]

    def test_sole_calibration_and_body_pose_round_trip(self):
        for sole in self.soles:
            yaw = R.from_euler("z", 0.4)
            center = np.array([0.2, -0.1, 0])
            pos, quat = sole.body_target(center, yaw)
            rotation = R.from_quat(quat, scalar_first=True)
            recovered, orientation = sole.pose(pos, rotation)
            np.testing.assert_allclose(recovered, center, atol=1e-12)
            np.testing.assert_allclose(orientation.as_matrix(), yaw.as_matrix(), atol=1e-12)
            corners = rotation.apply(sole.corners) + pos
            np.testing.assert_allclose(corners[:, 2], 0, atol=1e-12)
            vertices = rotation.apply(sole.vertices) + pos
            self.assertGreaterEqual(vertices[:, 2].min(), -1e-7)

    def test_contact_detection_preserves_flight(self):
        frames = [neutral_human() for _ in range(30)]
        for i, frame in enumerate(frames):
            z = 0.3 if 10 <= i < 20 else 0
            for side in ("left", "right"):
                frame[side + "_ankle"][0] = np.array([0, 0, z])
                frame[side + "_foot"] = [np.array([0.1, 0, z]), frame["pelvis"][1].copy()]
        contacts = detect_contacts(frames, 30)
        self.assertTrue(contacts[:8].all())
        self.assertFalse(contacts[12:18].any())
        self.assertTrue(contacts[22:].all())
        np.testing.assert_array_equal(intervals(np.array([True, True, False, True])), [[0, 2], [3, 4]])

    def test_locked_stance_and_swing_transition(self):
        targets = [self.gmr.target_builder(neutral_human()) for _ in range(20)]
        for i, frame in enumerate(targets):
            for pos, _ in frame.values():
                pos[0] += i * 0.002
        contacts = np.ones((20, 2), dtype=bool)
        contacts[6:14, 0] = False
        _, centers, rotations = plan_feet(targets, self.soles, contacts, 30)
        np.testing.assert_allclose(centers[:6, 0], np.tile(centers[0, 0], (6, 1)))
        np.testing.assert_allclose(centers[14:, 0], np.tile(centers[14, 0], (6, 1)))
        self.assertGreater(centers[10, 0, 2], 0.02)
        for i in (0, 5, 14, 19):
            np.testing.assert_allclose(rotations[i][0].apply([0, 0, 1]), [0, 0, 1], atol=1e-12)

    def test_double_support_constraints_and_input_preservation(self):
        frames = [neutral_human() for _ in range(3)]
        original = frames[0]["pelvis"][0].copy()
        result = retarget_mos_flat_feet(self.gmr, frames, 30, np.ones((3, 2), dtype=bool))
        self.assertEqual(result.report["status"], "passed")
        self.assertLess(result.report["max_stance_corner_height_m"], 0.001)
        self.assertLess(result.report["max_stance_tilt_deg"], 0.5)
        self.assertLess(result.report["max_stance_drift_m"], 0.001)
        np.testing.assert_array_equal(frames[0]["pelvis"][0], original)
        data = mj.MjData(self.gmr.model)
        for q in result.qpos:
            data.qpos[:] = q
            mj.mj_forward(self.gmr.model, data)
            for sole in self.soles:
                self.assertLess(np.abs(sole.points(data)[:, 2]).max(), 0.001)

    def test_invalid_schedule_and_robot_rejected(self):
        frames = [neutral_human() for _ in range(3)]
        with self.assertRaises(ValueError):
            retarget_mos_flat_feet(self.gmr, frames, 30, np.zeros((3, 2), dtype=bool))
        with self.assertRaises(ValueError):
            retarget_mos_flat_feet(self.gmr, frames, 30, np.ones((3, 2)))
        with self.assertRaises(ValueError):
            retarget_mos_flat_feet(self.gmr, frames, 0)
        k1 = GMR("smplx", "booster_k1", verbose=False)
        with self.assertRaises(ValueError):
            retarget_mos_flat_feet(k1, frames, 30)

    def test_solver_failure_returns_failed_report(self):
        original = mink.solve_ik

        def fail_contact_solve(*args, **kwargs):
            if "constraints" in kwargs:
                raise RuntimeError("injected infeasible contact")
            return original(*args, **kwargs)

        with patch.object(mink, "solve_ik", side_effect=fail_contact_solve):
            with self.assertRaises(FlatFootError) as caught:
                retarget_mos_flat_feet(
                    self.gmr, [neutral_human() for _ in range(3)],
                    30, np.ones((3, 2), dtype=bool),
                )
        self.assertEqual(caught.exception.report["status"], "failed")
        self.assertEqual(caught.exception.report["failed_frame"], 0)
        self.assertIn("infeasible", caught.exception.report["reason"])


if __name__ == "__main__":
    unittest.main()
