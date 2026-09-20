import pickle
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_conversion import finish_motion, load_gmr, validate_motion
from prepare_mos import prepare


from wbmr.paths import ROOT as PROJECT
URDF = PROJECT / "robots/mos/training/mos_training.urdf"
JOINT_NAMES = [j.get("name") for j in ET.parse(URDF).findall("./joint")]
BODY_NAMES = [b.get("name") for b in ET.parse(URDF).findall("./link")]


def motion_data():
    n, b = 5, len(BODY_NAMES)
    positions = np.zeros((n, b, 3), dtype=np.float32)
    positions[:, :, 2] = np.arange(n)[:, None] * .02 - .1
    return {
        "fps": [50], "joint_names": np.array(JOINT_NAMES), "body_names": np.array(BODY_NAMES),
        "joint_pos": np.zeros((n, 20), dtype=np.float32),
        "joint_vel": np.ones((n, 20), dtype=np.float32),
        "body_pos_w": positions, "body_quat_w": np.tile([1., 0, 0, 0], (n, b, 1)),
        "body_lin_vel_w": np.ones((n, b, 3)), "body_ang_vel_w": np.ones((n, b, 3)),
    }


def test_twenty_joint_gmr_and_unnamed_mjcf(tmp_path):
    path, model = tmp_path / "motion.pkl", tmp_path / "model.xml"
    data = {"fps": 29.901639344262296, "root_pos": np.zeros((5, 3)),
            "root_rot": np.tile([0., 0, 0, 1], (5, 1)), "dof_pos": np.zeros((5, 20))}
    path.write_bytes(pickle.dumps(data))
    tree = ET.Element("mujoco")
    body = ET.SubElement(ET.SubElement(tree, "worldbody"), "body")
    ET.SubElement(body, "joint", name="root", type="free")
    for name in JOINT_NAMES:
        ET.SubElement(body, "joint", name=name)
    ET.ElementTree(tree).write(model)
    values, fps = load_gmr(path, JOINT_NAMES, model)
    assert values.shape == (5, 27)
    assert fps == data["fps"]
    with pytest.raises(ValueError, match="order"):
        load_gmr(path, list(reversed(JOINT_NAMES)), model)


def test_mos_alignment_preserves_motion_and_reports_velocity(tmp_path):
    data = motion_data()
    before = {key: np.array(value, copy=True) for key, value in data.items()}
    source = tmp_path / "input.pkl"
    source.write_bytes(b"source")
    report = finish_motion(
        data, source, 29.901639344262296, URDF, True, tmp_path / "walk",
        JOINT_NAMES, root_name="body", foot_names=("Lfoot", "Rfoot"), required_bodies=BODY_NAMES,
    )
    delta = data["body_pos_w"] - before["body_pos_w"]
    np.testing.assert_allclose(delta[:, :, :2], 0)
    np.testing.assert_allclose(delta[:, :, 2], report["constant_z_translation_m"], atol=1e-7)
    np.testing.assert_allclose(np.diff(data["body_pos_w"], axis=0),
                               np.diff(before["body_pos_w"], axis=0), atol=1e-7)
    for key in before:
        if key != "body_pos_w":
            np.testing.assert_array_equal(data[key], before[key])
    assert min(report["feet_after"][name]["min_m"] for name in ("Lfoot", "Rfoot")) == pytest.approx(0, abs=1e-6)
    assert report["joint_limits"][JOINT_NAMES[0]]["max_abs_velocity_rad_s"] == 1


def test_missing_body_and_wrong_robot_rejected():
    data = motion_data()
    validate_motion(data, JOINT_NAMES, required_bodies=BODY_NAMES)
    with pytest.raises(ValueError, match="body names"):
        validate_motion(data, JOINT_NAMES, required_bodies=BODY_NAMES + ["missing"])
    with pytest.raises(ValueError, match="joints"):
        validate_motion(data, JOINT_NAMES + ["extra1", "extra2"], required_bodies=BODY_NAMES)


def test_prepare_preserves_source_transforms_and_inertials(tmp_path):
    source = tmp_path / "original.urdf"
    meshes = tmp_path / "original_meshes"
    meshes.mkdir()
    (meshes / "body.stl").write_bytes(b"mesh fixture")
    source.write_text(
        '<robot name="test"><link name="body"><inertial><mass value="1"/></inertial>'
        '<visual><origin xyz="1 2 3" rpy="0 0 1"/><geometry>'
        '<mesh filename="package://test/meshes/body.STL"/></geometry></visual></link>'
        '<joint name="test" type="continuous"><origin xyz="0 0 1" rpy="1 0 0"/>'
        '<axis xyz="0 0 -1"/><parent link="body"/><child link="foot"/></joint></robot>'
    )
    original = source.read_bytes()
    output = prepare(source, meshes, tmp_path / "training/mos.urdf")
    assert source.read_bytes() == original
    before, after = ET.fromstring(original), ET.parse(output).getroot()
    for xpath in ("./joint/origin", "./joint/axis", "./link/inertial", "./link/visual/origin"):
        assert [(e.tag, e.attrib) for e in before.find(xpath).iter()] == [
            (e.tag, e.attrib) for e in after.find(xpath).iter()
        ]
    assert after.find("./joint").get("type") == "revolute"
    assert after.find("./joint/limit").get("velocity") == "20"
    assert (output.parent / "meshes/body.stl").read_bytes() == b"mesh fixture"
    with pytest.raises(ValueError, match="overwrite"):
        prepare(source, meshes, source)
