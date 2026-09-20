"""Validate trusted bundled robot artifacts without launching Isaac Sim."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from wbmr.paths import ROOT

sys.path.insert(0, str(ROOT / "training/scripts"))
from motion_conversion import GMRUnpickler, load_gmr, sole_positions, validate_motion


MODELS = {
    "MOS": ("robots/mos/retargeting/MOS9.2.xml",
            "robots/mos/training/mos_training.urdf", ("body", "Lfoot", "Rfoot")),
    "K1": ("robots/k1/retargeting/K1_serial.xml",
           "robots/k1/training/K1_22dof.urdf", ("Trunk", "left_foot_link", "right_foot_link")),
}
CHECKPOINTS = {
    "k1_dance/model_29999.pt": ("K1/dance_8_clip1.npz", 119, 22, 29999),
    "mos_walk_rigid_v2/model_24000.pt": ("MOS/walk_rigid_v2.npz", 109, 20, 24000),
}


def joint_names(model):
    return [j.attrib["name"] for j in ET.parse(model).findall(".//worldbody//joint")
            if j.get("type", "hinge") != "free"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root=ROOT, checkpoints=False):
    records = []
    forbidden = {"poses", "betas", "trans", "mocap_frame_rate", "mocap_framerate", "gender"}
    for path in sorted((root / "motions").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        if forbidden.intersection(data):
            raise ValueError(f"Human-source keys found: {path}")
        fps = float(np.asarray(data["fps"]).item())
        assert np.isfinite(fps) and fps > 0, path
        if "virtual_qpos" in data:
            assert str(data["format"]) == "mos_virtual_waist_preview_v1", path
            assert not bool(data["training_compatible"]), path
            n = len(data["virtual_qpos"])
            address, quaternion_slices = 0, []
            for joint in ET.parse(path.parent / "model.xml").findall(".//worldbody//joint"):
                kind = joint.get("type", "hinge")
                if kind in ("free", "ball"):
                    start = address + (3 if kind == "free" else 0)
                    quaternion_slices.append(slice(start, start + 4))
                address += {"free": 7, "ball": 4}.get(kind, 1)
            for key in ("virtual_qpos", "rigid_qpos"):
                q = data[key]
                assert q.shape == (n, address) and address == 31 and np.isfinite(q).all(), path
                for part in quaternion_slices:
                    assert np.allclose(np.linalg.norm(q[:, part], axis=-1), 1, atol=1e-4), path
            assert data["contact_mask"].shape == (n, 2), path
            record = {"frames": n, "fps": fps, "training_compatible": False}
        else:
            model, urdf, bodies = MODELS[path.parent.name]
            names = joint_names(root / model)
            validate_motion(data, names, expected_fps=50, required_bodies=bodies)
            tree = ET.parse(root / urdf)
            assert set(data["body_names"]) == {b.get("name") for b in tree.findall("./link")}, path
            record = {"frames": len(data["joint_pos"]), "fps": fps,
                      "joints": len(names), "bodies": len(data["body_names"])}
            report_path = path.with_suffix(".json")
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("align_ground"):
                    soles = sole_positions(data, root / urdf, bodies[1:])
                    minimum = min(x[:, 2].min() for x in soles.values())
                    assert abs(minimum) <= .001, (path, minimum)
                    record["lowest_sole_m"] = float(minimum)
        record["path"] = str(path.relative_to(root))
        records.append(record)
    for path in sorted((root / "motions/gmr").rglob("*.pkl")):
        with path.open("rb") as stream:
            stored = GMRUnpickler(stream).load()
        assert set(stored) <= {"fps", "root_pos", "root_rot", "dof_pos", "local_body_pos",
                               "link_body_list", "joint_names"}, path
        robot = "MOS" if path.relative_to(root / "motions/gmr").parts[0].startswith("mos") else "K1"
        model = root / MODELS[robot][0]
        values, fps = load_gmr(path, joint_names(model), model)
        records.append({"path": str(path.relative_to(root)), "frames": len(values),
                        "fps": fps, "joints": values.shape[1] - 7})
    for path in sorted((root / "motions/gmr").rglob("*.csv")):
        values = np.loadtxt(path, delimiter=",")
        assert values.ndim == 2 and values.shape[1] == 29 and np.isfinite(values).all(), path
        assert np.allclose(np.linalg.norm(values[:, 3:7], axis=-1), 1, atol=1e-4), path
        records.append({"path": str(path.relative_to(root)), "frames": len(values), "joints": 22})
    if checkpoints:
        import torch

        def finite(value):
            if isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all())
            elif isinstance(value, dict):
                for item in value.values():
                    finite(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    finite(item)

        for name, (reference, obs_dim, actions, iteration) in CHECKPOINTS.items():
            path = root / "checkpoints" / name
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            finite(checkpoint)
            actor = checkpoint["actor_state_dict"]
            weights = [v for k, v in actor.items() if k.startswith("mlp.") and k.endswith(".weight")]
            assert weights[0].shape == (512, obs_dim), (path, weights[0].shape)
            assert weights[-1].shape == (actions, 128), (path, weights[-1].shape)
            assert checkpoint["iter"] == iteration, path
            assert (root / "motions" / reference).is_file(), path
            records.append({"path": str(path.relative_to(root)), "iteration": iteration,
                            "actor_observations": obs_dim, "actions": actions,
                            "reference": "motions/" + reference, "finite_tensors": True})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", action="store_true", help="Also CPU-load the trusted bundled PT files.")
    parser.add_argument("--report", type=Path, help="Write validation summary (does not modify artifacts).")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    records = validate(checkpoints=args.checkpoints)
    if args.verify_checksums:
        manifest = json.loads((ROOT / "docs/artifacts.json").read_text())
        for entry in manifest["artifacts"]:
            path = ROOT / entry["path"]
            assert path.stat().st_size == entry["bytes"], path
            assert sha256(path) == entry["sha256"], path
    report = {"status": "passed", "records": records}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed: {len(records)} robot-motion/checkpoint checks.")


if __name__ == "__main__":
    main()
