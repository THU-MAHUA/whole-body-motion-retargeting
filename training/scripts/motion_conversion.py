"""Simulator-independent input validation and mesh-based ground alignment."""

import hashlib
import json
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


ARRAY_DIMS = {
    "joint_pos": (22,),
    "joint_vel": (22,),
    "body_pos_w": (None, 3),
    "body_quat_w": (None, 4),
    "body_lin_vel_w": (None, 3),
    "body_ang_vel_w": (None, 3),
}


class GMRUnpickler(pickle.Unpickler):
    """Read NumPy 2 array pickles in Isaac Sim's NumPy 1 environment."""

    def find_class(self, module, name):
        if np.lib.NumpyVersion(np.__version__) < "2.0.0" and module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_gmr(path, joint_names, model_path=None):
    """Load a trusted local GMR pickle; unnamed arrays require its source MJCF."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        data = GMRUnpickler(stream).load()
    names = data.get("joint_names")
    if names is None:
        if model_path is None:
            raise ValueError("Unnamed GMR joints require --gmr_model pointing to the exporting MJCF.")
        tree = ET.parse(model_path)
        names = [
            joint.attrib["name"]
            for joint in tree.findall(".//worldbody//joint")
            if joint.get("type", "hinge") != "free"
        ]
    if list(names) != list(joint_names):
        raise ValueError(f"GMR joint order differs from expected robot joints: {list(names)}")
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid source FPS: {fps}")
    root, quat, joints = (np.asarray(data[key]) for key in ("root_pos", "root_rot", "dof_pos"))
    n = len(root)
    if n < 3 or root.shape != (n, 3) or quat.shape != (n, 4) or joints.shape != (n, len(joint_names)):
        raise ValueError(f"Expected at least three matching frames of root_pos(3), root_rot(4), dof_pos({len(joint_names)}).")
    motion = np.concatenate((root, quat, joints), axis=1)
    if not np.isfinite(motion).all():
        raise ValueError("Non-finite GMR input.")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-4):
        raise ValueError("GMR XYZW quaternions are not normalized.")
    return motion, fps


def validate_motion(data, joint_names, expected_fps=50,
                    required_bodies=("Trunk", "left_foot_link", "right_foot_link")):
    n = len(data["joint_pos"])
    names = list(data["joint_names"])
    bodies = list(data["body_names"])
    if n < 3 or len(names) != len(joint_names) or set(names) != set(joint_names):
        raise ValueError("Missing/duplicate robot joints or too few frames.")
    if len(set(bodies)) != len(bodies) or not set(required_bodies) <= set(bodies):
        raise ValueError("Missing/duplicate body names.")
    if float(np.asarray(data["fps"]).item()) != expected_fps:
        raise ValueError("Unexpected output FPS.")
    for key, dims in ARRAY_DIMS.items():
        if key.startswith("joint_"):
            dims = (len(joint_names),)
        shape = (n,) + tuple(len(bodies) if dim is None else dim for dim in dims)
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key}: expected finite {shape}, got {data[key].shape}")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-4):
        raise ValueError("Output WXYZ quaternions are not normalized.")


def foot_vertices(urdf_path, foot_names=("left_foot_link", "right_foot_link")):
    urdf_path = Path(urdf_path).resolve()
    tree = ET.parse(urdf_path)
    result = {}
    for name in foot_names:
        vertices = []
        for collision in tree.findall(f"./link[@name='{name}']/collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Unsupported non-mesh foot collision: {name}")
            filename = mesh.attrib["filename"]
            if filename.startswith("package://"):
                raise ValueError(f"Resolve package mesh path first: {filename}")
            obj = trimesh.load(urdf_path.parent / filename, force="mesh", process=False)
            points = np.asarray(obj.vertices) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                points = Rotation.from_euler("xyz", rpy).apply(points) + xyz
            vertices.append(points)
        if not vertices:
            raise ValueError(f"No foot collisions found: {name}")
        result[name] = np.unique(np.concatenate(vertices), axis=0)
    return result


def sole_positions(data, urdf_path, foot_names=("left_foot_link", "right_foot_link")):
    """Return the lowest collision vertex for each foot and frame in world coordinates."""
    result = {}
    for name, vertices in foot_vertices(urdf_path, foot_names).items():
        index = list(data["body_names"]).index(name)
        quat = data["body_quat_w"][:, index][:, [1, 2, 3, 0]]
        matrices = Rotation.from_quat(quat).as_matrix()
        world = np.einsum("nij,vj->nvi", matrices, vertices) + data["body_pos_w"][:, index, None]
        result[name] = world[np.arange(len(world)), world[:, :, 2].argmin(axis=1)]
    return result


def height_stats(soles, fps):
    result = {}
    for name, points in soles.items():
        height = points[:, 2]
        speed = np.linalg.norm(np.gradient(points[:, :2], 1 / fps, axis=0), axis=1)
        near = height < 0.02
        result[name] = {
            "min_m": float(height.min()), "max_m": float(height.max()),
            "median_m": float(np.median(height)),
            "fraction_below_ground": float(np.mean(height < -0.001)),
            "fraction_within_2cm": float(np.mean(near)),
            "near_ground_vertex_xy_speed_mean_m_s": float(speed[near].mean()) if near.any() else None,
        }
    lowest = np.minimum(*(points[:, 2] for points in soles.values()))
    result["both_feet_above_2cm_fraction"] = float(np.mean(lowest > 0.02))
    result["lowest_foot_max_m"] = float(lowest.max())
    return result


def joint_findings(data, urdf_path):
    tree = ET.parse(urdf_path)
    result = {}
    for i, name in enumerate(data["joint_names"]):
        limit = tree.find(f"./joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"Missing URDF joint limit: {name}")
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        q = data["joint_pos"][:, i]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        result[str(name)] = {
            "min_rad": float(q.min()), "max_rad": float(q.max()),
            "lower_rad": lo, "upper_rad": hi,
            "hard_limit_violation_frames": int(np.sum((q < lo - 1e-5) | (q > hi + 1e-5))),
            "within_0_01rad_of_hard_limit_frames": int(np.sum((q < lo + .01) | (q > hi - .01))),
            "outside_0_9_soft_limit_frames": int(np.sum(np.abs(q - mid) > .9 * half)),
            "max_abs_velocity_rad_s": float(np.abs(data["joint_vel"][:, i]).max()),
            "velocity_limit_rad_s": float(limit.get("velocity", "inf")),
            "velocity_limit_exceeded_frames": int(np.sum(
                np.abs(data["joint_vel"][:, i]) > float(limit.get("velocity", "inf"))
            )),
        }
    return result


def finish_motion(data, source, input_fps, urdf_path, align_ground, output, joint_names, model_path=None,
                  root_name="Trunk", foot_names=("left_foot_link", "right_foot_link"),
                  required_bodies=None):
    fps = float(np.asarray(data["fps"]).item())
    required_bodies = required_bodies or (root_name, *foot_names)
    validate_motion(data, joint_names, fps, required_bodies)
    before = sole_positions(data, urdf_path, foot_names)
    original = data["body_pos_w"].copy()
    shift = -min(points[:, 2].min() for points in before.values()) if align_ground else 0.0
    data["body_pos_w"][:, :, 2] += shift
    after = sole_positions(data, urdf_path, foot_names)
    minimum = min(points[:, 2].min() for points in after.values())
    if align_ground and abs(minimum) > .001:
        raise ValueError(f"Ground alignment failed: {minimum}")
    if not np.allclose(np.diff(original, axis=0), np.diff(data["body_pos_w"], axis=0), atol=2e-6):
        raise ValueError("Ground alignment changed relative motion.")
    validate_motion(data, joint_names, fps, required_bodies)
    output = Path(output)
    if output.suffix != ".npz":
        output = Path(str(output) + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(Path(source).resolve()),
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "source_fps": input_fps, "output_fps": fps,
        "frames": len(original), "sample_span_seconds": (len(original) - 1) / fps,
        "training_urdf": str(Path(urdf_path).resolve()),
        "gmr_joint_order_model": str(Path(model_path).resolve()) if model_path else None,
        "align_ground": align_ground, "constant_z_translation_m": float(shift),
        "feet_before": height_stats(before, fps), "feet_after": height_stats(after, fps),
        "joint_limits": joint_findings(data, urdf_path),
        "validation": "passed",
        "notes": [
            "One constant translation; no joint clamping or per-frame grounding.",
            "Near-ground vertex speed is a sliding heuristic, not a contact measurement.",
            "Flight/floating cannot be distinguished by height alone; inspect reference replay.",
            "Mesh vertices describe nominal URDF geometry, not PhysX contact offsets.",
        ],
    }
    np.savez(output, **data)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
