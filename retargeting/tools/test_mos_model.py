"""Compare MOS MJCF kinematics and visual placement with its source URDF."""

import unittest
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from urdf_to_mjcf_mos import OUT, URDF


def transform(origin):
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    result = np.eye(4)
    # Independently compose the URDF fixed-axis rotation.
    result[:3, :3] = (
        Rotation.from_rotvec([0, 0, yaw]).as_matrix()
        @ Rotation.from_rotvec([0, pitch, 0]).as_matrix()
        @ Rotation.from_rotvec([roll, 0, 0]).as_matrix()
    )
    result[:3, 3] = xyz
    return result


class MosModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = ET.parse(URDF).getroot()
        cls.model = mj.MjModel.from_xml_path(str(OUT))
        cls.joints = cls.urdf.findall("joint")

    def forward_urdf(self, angles):
        frames = {"body": np.eye(4)}
        pending = self.joints.copy()
        while pending:
            count = len(pending)
            for joint in pending.copy():
                parent = joint.find("parent").get("link")
                if parent not in frames:
                    continue
                child = joint.find("child").get("link")
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    axis / np.linalg.norm(axis) * angles[joint.get("name")]
                ).as_matrix()
                frames[child] = frames[parent] @ transform(joint.find("origin").attrib) @ motion
                pending.remove(joint)
            self.assertLess(len(pending), count, "URDF contains a disconnected/cyclic tree")
        return frames

    def test_link_frames_and_axes(self):
        rng = np.random.default_rng(42)
        max_position_error = 0.0
        for sample in range(11):
            data = mj.MjData(self.model)
            angles = {}
            for joint in self.joints:
                name = joint.get("name")
                angles[name] = 0.0 if sample == 0 else rng.uniform(-0.8, 0.8)
                data.qpos[self.model.joint(name).qposadr[0]] = angles[name]
            mj.mj_forward(self.model, data)
            frames = self.forward_urdf(angles)
            for name, expected in frames.items():
                bid = self.model.body(name).id
                np.testing.assert_allclose(data.xpos[bid], expected[:3, 3], atol=1e-10)
                np.testing.assert_allclose(data.xmat[bid].reshape(3, 3), expected[:3, :3], atol=1e-10)
                max_position_error = max(max_position_error, np.linalg.norm(data.xpos[bid] - expected[:3, 3]))
            for joint in self.joints:
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                expected = frames[joint.find("child").get("link")][:3, :3] @ axis
                np.testing.assert_allclose(data.xaxis[self.model.joint(joint.get("name")).id], expected, atol=1e-10)
        print(f"21 link frames, 11 poses: max position error {max_position_error:.3g} m")

    def test_visual_placement(self):
        data = mj.MjData(self.model)
        mj.mj_forward(self.model, data)
        frames = self.forward_urdf({j.get("name"): 0 for j in self.joints})
        for link in self.urdf.findall("link"):
            name = link.get("name")
            visual = link.find("visual")
            mesh = trimesh.load_mesh(OUT.parent / "meshes" / f"{name}.stl")
            scale = np.fromstring(visual.find("geometry/mesh").get("scale", "1 1 1"), sep=" ")
            expected_pose = frames[name] @ transform(visual.find("origin").attrib)
            points = mesh.vertices * scale
            expected = points @ expected_pose[:3, :3].T + expected_pose[:3, 3]
            gid = self.model.geom(f"{name}_visual").id
            mid = self.model.geom_dataid[gid]
            start = self.model.mesh_vertadr[mid]
            vertices = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mid]]
            actual = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            np.testing.assert_allclose(actual.min(axis=0), expected.min(axis=0), atol=2e-6)
            np.testing.assert_allclose(actual.max(axis=0), expected.max(axis=0), atol=2e-6)
            self.assertEqual(self.model.geom_contype[gid], 0)
            self.assertEqual(self.model.geom_conaffinity[gid], 0)
            expected_color = np.fromstring(visual.find("material/color").get("rgba"), sep=" ")
            np.testing.assert_allclose(self.model.geom_rgba[gid], expected_color, atol=1e-7)

    def test_joint_order(self):
        self.assertEqual(self.model.nq, 27)
        self.assertEqual(self.model.nv, 26)
        self.assertEqual(
            [self.model.joint(i).name for i in range(1, self.model.njnt)],
            [joint.get("name") for joint in self.joints],
        )

    def test_scene_appearance_matches_k1(self):
        mos = ET.parse(OUT).getroot()
        from wbmr.paths import ROBOTS
        k1 = ET.parse(ROBOTS / "k1/retargeting/K1_serial.xml").getroot()
        for path in (
            "asset/texture[@type='skybox']",
            "asset/texture[@name='texplane']",
            "asset/material[@name='matplane']",
        ):
            self.assertEqual(mos.find(path).attrib, k1.find(path).attrib)
        self.assertEqual(
            [light.attrib for light in mos.findall("worldbody/light")],
            [light.attrib for light in k1.findall("worldbody/light")],
        )
        floor = mos.find("worldbody/geom[@name='ground']")
        k1_floor = k1.find("worldbody/geom[@name='ground']")
        for attribute in ("type", "pos", "size", "material"):
            self.assertEqual(floor.get(attribute), k1_floor.get(attribute))
        gid = self.model.geom("ground").id
        self.assertEqual(self.model.geom_condim[gid], 3)
        self.assertEqual(self.model.geom_contype[gid], 1)
        self.assertEqual(self.model.geom_conaffinity[gid], 1)


if __name__ == "__main__":
    unittest.main()
