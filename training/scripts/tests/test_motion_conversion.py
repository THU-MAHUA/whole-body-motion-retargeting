import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_conversion import finish_motion, load_gmr, validate_motion
from booster_assets import BOOSTER_ASSETS_DIR
from booster_assets.motions import K1_JOINT_NAMES


URDF = Path(BOOSTER_ASSETS_DIR) / "robots/k1/training/K1_22dof.urdf"


def motion_data():
    n = 5
    return {
        "fps": [50],
        "joint_pos": np.zeros((n, 22), dtype=np.float32),
        "joint_vel": np.ones((n, 22), dtype=np.float32),
        "body_pos_w": np.tile(np.array([[[0, 0, -.1], [0, .1, -.2], [0, -.1, -.15]]]), (n, 1, 1)),
        "body_quat_w": np.tile([1., 0, 0, 0], (n, 3, 1)),
        "body_lin_vel_w": np.ones((n, 3, 3)),
        "body_ang_vel_w": np.ones((n, 3, 3)),
        "joint_names": np.array(K1_JOINT_NAMES),
        "body_names": np.array(["Trunk", "left_foot_link", "right_foot_link"]),
    }


def test_alignment_is_constant_and_preserves_every_other_array(tmp_path):
    data = motion_data()
    data["body_pos_w"][:, :, 2] += np.arange(5)[:, None] * .02
    before = {key: np.array(value, copy=True) for key, value in data.items()}
    source = tmp_path / "source.pkl"
    source.write_bytes(b"test-source")
    report = finish_motion(data, source, 29.81595, URDF, True, tmp_path / "result.npz", K1_JOINT_NAMES)
    delta = data["body_pos_w"] - before["body_pos_w"]
    np.testing.assert_allclose(delta[:, :, :2], 0)
    np.testing.assert_allclose(delta[:, :, 2], report["constant_z_translation_m"])
    np.testing.assert_allclose(np.diff(data["body_pos_w"], axis=0), np.diff(before["body_pos_w"], axis=0))
    for key in before:
        if key != "body_pos_w":
            np.testing.assert_array_equal(data[key], before[key])
    assert min(report["feet_after"][foot]["min_m"] for foot in ("left_foot_link", "right_foot_link")) == pytest.approx(0, abs=1e-6)
    assert (tmp_path / "result.json").exists()


def test_no_alignment_keeps_positions(tmp_path):
    data = motion_data()
    before = data["body_pos_w"].copy()
    source = tmp_path / "source.csv"
    source.write_text("test")
    report = finish_motion(data, source, 30, URDF, False, tmp_path / "result", K1_JOINT_NAMES)
    np.testing.assert_array_equal(data["body_pos_w"], before)
    assert report["constant_z_translation_m"] == 0
    assert (tmp_path / "result.npz").exists()


def test_gmr_float_fps_and_order(tmp_path):
    path = tmp_path / "motion.pkl"
    data = {"fps": 29.826589595375722, "root_pos": np.zeros((4, 3)),
            "root_rot": np.tile([0., 0, 0, 1], (4, 1)), "dof_pos": np.zeros((4, 22)),
            "joint_names": K1_JOINT_NAMES}
    path.write_bytes(pickle.dumps(data))
    values, fps = load_gmr(path, K1_JOINT_NAMES)
    assert fps == data["fps"]
    np.testing.assert_array_equal(values[:, 3:7], data["root_rot"])
    with pytest.raises(ValueError, match="order"):
        load_gmr(path, list(reversed(K1_JOINT_NAMES)))
    del data["joint_names"]
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="gmr_model"):
        load_gmr(path, K1_JOINT_NAMES)


@pytest.mark.parametrize("fault", ["nan", "quaternion", "frames", "names", "fps"])
def test_invalid_output_rejected(fault):
    data = motion_data()
    if fault == "nan":
        data["joint_vel"][0, 0] = np.nan
    elif fault == "quaternion":
        data["body_quat_w"][0, 0] *= 2
    elif fault == "frames":
        data["body_pos_w"] = data["body_pos_w"][:-1]
    elif fault == "names":
        data["joint_names"][0] = "wrong"
    else:
        data["fps"] = [30]
    with pytest.raises(ValueError):
        validate_motion(data, K1_JOINT_NAMES)
