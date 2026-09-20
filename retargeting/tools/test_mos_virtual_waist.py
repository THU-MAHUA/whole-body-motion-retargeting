"""Ensure the virtual comparison cannot change the original leg kinematics."""

import unittest

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from preview_mos_virtual_waist import MODEL, UPPER_JOINTS, map_baseline, model_xml


class VirtualWaistTest(unittest.TestCase):
    def setUp(self):
        self.original = mj.MjModel.from_xml_path(str(MODEL))
        self.virtual = mj.MjModel.from_xml_string(model_xml())

    def test_dimensions_and_neutral_mesh_pose(self):
        self.assertEqual(self.virtual.nq, self.original.nq + 4)
        self.assertEqual(self.virtual.nv, self.original.nv + 3)
        first, second = mj.MjData(self.original), mj.MjData(self.virtual)
        mj.mj_forward(self.original, first)
        mj.mj_forward(self.virtual, second)
        for i in range(self.original.ngeom):
            name = self.original.geom(i).name
            if name:
                j = self.virtual.geom(name).id
                np.testing.assert_allclose(first.geom_xpos[i], second.geom_xpos[j], atol=1e-12)
                np.testing.assert_allclose(first.geom_xmat[i], second.geom_xmat[j], atol=1e-12)

    def test_waist_and_upper_motion_do_not_change_feet(self):
        rng = np.random.default_rng(19)
        q = np.tile(self.original.qpos0, (8, 1))
        q[:, :3] = rng.normal(size=(8, 3))
        q[:, 3:7] = R.random(8, random_state=rng).as_quat(scalar_first=True)
        q[:, 7:] = rng.normal(scale=.2, size=(8, self.original.nq - 7))
        mapped = map_baseline(self.original, self.virtual, q)
        adr = self.virtual.joint("virtual_waist").qposadr[0]
        mapped[:, adr:adr + 4] = R.from_euler("xyz", [.1, .2, -.15]).as_quat(scalar_first=True)
        for name in UPPER_JOINTS:
            mapped[:, self.virtual.joint(name).qposadr[0]] += .3
        first, second = mj.MjData(self.original), mj.MjData(self.virtual)
        for a, b in zip(q, mapped):
            first.qpos[:], second.qpos[:] = a, b
            mj.mj_forward(self.original, first)
            mj.mj_forward(self.virtual, second)
            for name in ("body", "Lfoot", "Rfoot", "Lleg1", "Rleg1"):
                i, j = self.original.body(name).id, self.virtual.body(name).id
                np.testing.assert_allclose(first.xpos[i], second.xpos[j], atol=1e-12)
                np.testing.assert_allclose(first.xmat[i], second.xmat[j], atol=1e-12)
            self.assertGreater(np.linalg.norm(
                first.xmat[self.original.body("head").id] - second.xmat[self.virtual.body("head").id]
            ), .01)


if __name__ == "__main__":
    unittest.main()
