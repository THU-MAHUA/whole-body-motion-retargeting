"""Contact-aware MOS kinematic references; not a dynamics controller."""

from dataclasses import dataclass

import mink
import mujoco as mj
import numpy as np
from mink.limits import Constraint
from scipy.spatial.transform import Rotation as R

from .mos import MosTargets, retarget_mos_clip

SIDES = ("left", "right")


class FlatFootError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


@dataclass
class Sole:
    body_id: int
    center: np.ndarray
    corners: np.ndarray
    vertices: np.ndarray
    neutral_rotation: R

    @classmethod
    def from_model(cls, model, prefix):
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        bid = model.body(prefix + "foot").id
        gid = model.geom(prefix + "foot_visual").id
        mid = model.geom_dataid[gid]
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        low = world[:, 2].min()
        bottom = world[world[:, 2] < low + 1e-4]
        span = np.ptp(bottom, axis=0)
        if len(bottom) < 4 or min(span[:2]) < 0.03 or span[2] > 1e-5:
            raise ValueError(f"{prefix} foot does not have a calibrated horizontal sole")
        lo, hi = bottom.min(axis=0), bottom.max(axis=0)
        corners = np.array([[x, y, low] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        rotation = R.from_quat(data.xquat[bid], scalar_first=True)
        to_local = lambda points: rotation.inv().apply(points - data.xpos[bid])
        return cls(bid, to_local(corners.mean(axis=0)), to_local(corners),
                   to_local(world), rotation)

    def points(self, data, vertices=False):
        local = self.vertices if vertices else self.corners
        return local @ data.xmat[self.body_id].reshape(3, 3).T + data.xpos[self.body_id]

    def pose(self, position, rotation):
        return position + rotation.apply(self.center), rotation * self.neutral_rotation.inv()

    def body_target(self, center, rotation):
        body_rotation = rotation * self.neutral_rotation
        return [center - body_rotation.apply(self.center),
                body_rotation.as_quat(scalar_first=True)]


def intervals(mask):
    """Half-open true intervals, including contacts at either clip boundary."""
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def detect_contacts(frames, fps, height_threshold=0.10, speed_threshold=0.55):
    """Infer support from human ankle/toe speed and height, in source meters."""
    if len(frames) < 3 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Flat-foot mode requires at least three frames and positive FPS")
    mask = np.zeros((len(frames), 2), dtype=bool)
    for s, side in enumerate(SIDES):
        for joint in ("ankle", "foot"):
            positions = np.asarray([frame[f"{side}_{joint}"][0] for frame in frames])
            if not np.isfinite(positions).all():
                raise ValueError("Non-finite source foot positions")
            speed = np.linalg.norm(np.gradient(positions, 1 / fps, axis=0), axis=1)
            height = positions[:, 2] - np.percentile(positions[:, 2], 5)
            mask[:, s] |= (height < height_threshold) & (speed < speed_threshold)
        # Bridge isolated classification gaps, but never force either foot into contact.
        for start, end in intervals(~mask[:, s]):
            if start > 0 and end < len(frames) and end - start == 1:
                mask[start:end, s] = True
        for start, end in intervals(mask[:, s]):
            if end - start < max(2, round(0.07 * fps)):
                mask[start:end, s] = False
    return mask


class SoleFloorLimit(mink.Limit):
    """Linearized floor clearance for sole corners and the lowest mesh vertex."""

    def __init__(self, soles):
        self.soles = soles

    def compute_qp_inequalities(self, configuration, dt):
        model, data = configuration.model, configuration.data
        rows, heights = [], []
        for sole in self.soles:
            mesh = sole.points(data, vertices=True)
            points = np.vstack([sole.points(data), mesh[mesh[:, 2].argmin()]])
            for point in points:
                jac = np.zeros((3, model.nv))
                mj.mj_jac(model, data, jac, None, point, sole.body_id)
                rows.append(-jac[2])
                heights.append(point[2] + 2e-5)
        if not rows:
            return Constraint()
        return Constraint(G=np.asarray(rows), h=np.asarray(heights))


def _blend_rotation(first, second, weight):
    return first * R.from_rotvec((first.inv() * second).as_rotvec() * weight)


def plan_feet(targets, soles, contact, fps):
    """Lock stance poses and blend swing poses toward neighboring contacts."""
    n = len(targets)
    centers = np.empty((n, 2, 3))
    rotations = [[None, None] for _ in targets]
    for i, frame in enumerate(targets):
        for s, side in enumerate(SIDES):
            pos, quat = frame[side + "_ankle"]
            centers[i, s], rotations[i][s] = soles[s].pose(pos, R.from_quat(quat, scalar_first=True))
    shift = -float(np.median(centers[:, :, 2][contact]))
    centers[:, :, 2] += shift
    for frame in targets:
        for pos, _ in frame.values():
            pos[2] += shift
    for s in range(2):
        for start, end in intervals(contact[:, s]):
            center = np.median(centers[start:end, s], axis=0)
            center[2] = 0
            heading = np.asarray([rotations[i][s].apply([1, 0, 0])[:2] for i in range(start, end)])
            heading = heading.mean(axis=0)
            if np.linalg.norm(heading) < 1e-4:
                raise ValueError("Undefined stance heading")
            yaw = R.from_euler("z", np.arctan2(heading[1], heading[0]))
            centers[start:end, s] = center
            for i in range(start, end):
                rotations[i][s] = yaw
        for start, end in intervals(~contact[:, s]):
            before = start - 1 if start else None
            after = end if end < n else None
            if before is not None and after is not None:
                count = after - before
                original = centers[start:end, s].copy()
                peak = max(0.035, float(np.max(original[:, 2])))
                for i in range(start, end):
                    t = (i - before) / count
                    smooth = t * t * (3 - 2 * t)
                    centers[i, s] = (1 - smooth) * centers[before, s] + smooth * centers[after, s]
                    centers[i, s, 2] = peak * np.sin(np.pi * t) ** 2
                    flat = _blend_rotation(rotations[before][s], rotations[after][s], smooth)
                    # Keep human swing rotation away from the landing/lift-off ramps.
                    free = min(1.0, (i - before) / (0.15 * fps), (after - i) / (0.15 * fps))
                    free = free * free * (3 - 2 * free)
                    rotations[i][s] = _blend_rotation(flat, rotations[i][s], free)
            else:
                anchor = before if before is not None else after
                if anchor is not None:
                    span = max(1, round(0.15 * fps))
                    for i in range(start, end):
                        t = min(1.0, abs(i - anchor) / span)
                        smooth = t * t * (3 - 2 * t)
                        centers[i, s] = (1 - smooth) * centers[anchor, s] + smooth * centers[i, s]
                        rotations[i][s] = _blend_rotation(rotations[anchor][s], rotations[i][s], smooth)
            # Raise only the swing foot target enough to clear its tilted mesh.
            for i in range(start, end):
                sole = soles[s]
                body_pos, quat = sole.body_target(centers[i, s], rotations[i][s])
                minimum = (R.from_quat(quat, scalar_first=True).apply(sole.vertices) + body_pos)[:, 2].min()
                centers[i, s, 2] += max(0, -minimum)
    for i, frame in enumerate(targets):
        for s, side in enumerate(SIDES):
            frame[side + "_ankle"] = soles[s].body_target(centers[i, s], rotations[i][s])
    return shift, centers, rotations


@dataclass
class FlatFootResult:
    qpos: np.ndarray
    targets: list
    contacts: np.ndarray
    report: dict


def retarget_mos_flat_feet(retargeter, frames, fps, contacts=None, *, rigid_torso=False):
    """Solve full-sole stance constraints and reject unmet contact tolerances."""
    if not isinstance(retargeter.target_builder, MosTargets):
        raise ValueError("--flat_feet is supported only for MOS")
    if len(frames) < 3 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Flat-foot mode requires at least three frames and positive FPS")
    inferred = contacts is None
    if inferred:
        contacts = detect_contacts(frames, fps)
    contacts = np.asarray(contacts)
    if contacts.shape != (len(frames), 2) or contacts.dtype != np.bool_:
        raise ValueError("Contact schedule must be an N-by-2 boolean array [left, right]")
    if not contacts.any():
        raise ValueError("No support intervals found; review the source or contact schedule")
    model = retargeter.model
    soles = [Sole.from_model(model, prefix) for prefix in ("L", "R")]
    targets = [retargeter.target_builder(frame) for frame in frames]
    shift, centers, rotations = plan_feet(targets, soles, contacts, fps)
    torso_report = None
    if rigid_torso:
        from .mos_torso import adapt_rigid_torso
        targets, torso_report = adapt_rigid_torso(targets, fps)
    seed, _, _ = retarget_mos_clip(retargeter, frames)
    seed[:, 2] += shift
    report = {
        "mode": "mos_flat_feet", "fps": float(fps), "frames": len(frames),
        "contact_source": "inferred ankle/toe height and speed" if inferred else "explicit schedule",
        "contact_mask": contacts.tolist(),
        "intervals": {side: intervals(contacts[:, s]).tolist() for s, side in enumerate(SIDES)},
        "interval_end_is_exclusive": True,
        "source_height_threshold_m": 0.10 if inferred else None,
        "source_speed_threshold_m_s": 0.55 if inferred else None,
        "reference_vertical_shift_m": shift,
        "tolerances": {"sole_height_m": 0.001, "sole_tilt_deg": 0.5, "stance_drift_m": 0.001},
        "limitations": [
            "Kinematic reference, not dynamically balanced or hardware validated.",
            "Contacts are inferred unless an explicit schedule is supplied; inspect the intervals.",
            "Full-sole support replaces heel/toe rocking in the designated stance intervals.",
            "Swing trajectories are reshaped; loop boundaries are not made periodic.",
            "All MOS joints are continuous in the URDF; mechanical limits are unavailable.",
        ],
        "status": "failed",
    }
    tasks = {}
    if torso_report is not None:
        report["torso_adaptation"] = torso_report
        torso_report["ik_pelvis_position_cost"] = 20
        torso_report["ik_pelvis_orientation_cost"] = 60
        torso_report["ik_upper_body_cost_scale"] = 0.25
    for name, original in retargeter.human_body_to_task1.items():
        position, orientation = original.cost[:3].copy(), original.cost[3:].copy()
        if name == "pelvis":
            position[:], orientation[:] = 20, 60 if rigid_torso else 10
        elif name.endswith(("_hip", "_knee")):
            orientation[:] = 3
        elif rigid_torso and (name == "head" or name.endswith(("_shoulder", "_elbow"))):
            position *= 0.25
            orientation *= 0.25
        tasks[name] = mink.FrameTask(original.frame_name, "body", position, orientation, lm_damping=0.01)
    posture = mink.PostureTask(model, cost=0.5)
    smoothness = mink.PostureTask(model, cost=1.0)
    floor_limit = SoleFloorLimit(soles)
    limits = [mink.ConfigurationLimit(model), floor_limit]
    config = mink.Configuration(model)
    qpos, metrics = [], []
    for i, frame in enumerate(targets):
        config.update(seed[i] if i == 0 else qpos[-1])
        posture.set_target(seed[i])
        smoothness.set_target(config.q.copy())
        for name, task in tasks.items():
            pos, quat = frame[name]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))
        constraints = [tasks[side + "_ankle"] for s, side in enumerate(SIDES) if contacts[i, s]]
        # Stance pose equalities already fix the sole; redundant linearized corner
        # constraints can conflict with their finite-rotation log-map correction.
        floor_limit.soles = [sole for s, sole in enumerate(soles) if not contacts[i, s]]
        soft = [task for task in tasks.values() if task not in constraints] + [posture, smoothness]
        try:
            for iteration in range(100):
                velocity = mink.solve_ik(
                    config, soft, 1 / fps, solver=retargeter.solver,
                    damping=0.1, limits=limits, constraints=constraints,
                )
                config.integrate_inplace(velocity, 1 / fps)
                error = max((np.linalg.norm(task.compute_error(config)) for task in constraints), default=0)
                lowest = min(sole.points(config.data, vertices=True)[:, 2].min() for sole in soles)
                if iteration >= 4 and error < 1e-5 and lowest >= -5e-5 and np.linalg.norm(velocity / fps) < 1e-3:
                    break
        except Exception as exc:
            report.update(failed_frame=i, reason=str(exc))
            raise FlatFootError(f"MOS flat-foot IK failed at frame {i}: {exc}", report) from exc
        q = config.q.copy()
        if not np.isfinite(q).all():
            report.update(failed_frame=i, reason="Non-finite pose")
            raise FlatFootError(f"Non-finite flat-foot pose at frame {i}", report)
        row = []
        for s, sole in enumerate(soles):
            data = config.data
            body_rot = R.from_quat(data.xquat[sole.body_id], scalar_first=True)
            center, rotation = sole.pose(data.xpos[sole.body_id], body_rot)
            tilt = np.rad2deg(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1)))
            height = float(np.abs(sole.points(data)[:, 2]).max())
            drift = float(np.linalg.norm(center - centers[i, s]))
            yaw_error = np.rad2deg((rotations[i][s].inv() * rotation).magnitude())
            minimum = float(sole.points(data, vertices=True)[:, 2].min())
            row.append([height, tilt, drift, yaw_error, minimum])
            bad_stance = contacts[i, s] and (height > 0.001 or tilt > 0.5 or drift > 0.001 or yaw_error > 0.5)
            if bad_stance or minimum < -0.001:
                report.update(failed_frame=i, failed_side=SIDES[s], reason="Contact tolerance not met",
                              failed_metrics=dict(zip(("height_m", "tilt_deg", "drift_m", "orientation_error_deg", "mesh_min_m"), row[-1])))
                raise FlatFootError(f"MOS foot contact failed at frame {i} ({SIDES[s]})", report)
        root_error = float(np.linalg.norm(q[:3] - seed[i, :3]))
        root_angle = np.rad2deg((R.from_quat(seed[i, 3:7], scalar_first=True).inv()
                                * R.from_quat(q[3:7], scalar_first=True)).magnitude())
        if root_error > 0.15 or root_angle > 20:
            report.update(failed_frame=i, reason="Excessive pelvis correction",
                          pelvis_correction_m=root_error, pelvis_correction_deg=float(root_angle))
            raise FlatFootError(f"MOS pelvis correction too large at frame {i}", report)
        qpos.append(q)
        metrics.append(row)
    qpos, metrics = np.asarray(qpos), np.asarray(metrics)
    planted = metrics[contacts]
    drift_speed = []
    for s, sole in enumerate(soles):
        positions = []
        for q in qpos:
            config.update(q)
            positions.append(sole.pose(config.data.xpos[sole.body_id],
                             R.from_quat(config.data.xquat[sole.body_id], scalar_first=True))[0])
        same_contact = contacts[1:, s] & contacts[:-1, s]
        drift_speed.extend((np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps)[same_contact])
    report.update(
        status="passed",
        max_stance_corner_height_m=float(planted[:, 0].max()),
        max_stance_tilt_deg=float(planted[:, 1].max()),
        max_stance_drift_m=float(planted[:, 2].max()),
        max_stance_orientation_error_deg=float(planted[:, 3].max()),
        max_stance_speed_m_s=float(max(drift_speed, default=0)),
        min_foot_mesh_height_m=float(metrics[:, :, 4].min()),
        max_pelvis_translation_change_m=float(np.linalg.norm(qpos[:, :3] - seed[:, :3], axis=1).max()),
        max_joint_step_rad=float(np.abs(np.diff(qpos[:, 7:], axis=0)).max()),
        frames_without_support=int((~contacts.any(axis=1)).sum()),
    )
    if rigid_torso:
        actual = R.from_quat(qpos[:, 3:7], scalar_first=True)
        desired = R.from_quat([frame["pelvis"][1] for frame in targets], scalar_first=True)
        report["torso_adaptation"]["actual_roll_pitch_peak_deg"] = np.rad2deg(
            np.abs(actual.as_euler("xyz")[:, :2]).max(axis=0)
        ).tolist()
        report["torso_adaptation"]["max_actual_target_error_deg"] = float(
            np.rad2deg((desired.inv() * actual).magnitude()).max()
        )
    return FlatFootResult(qpos, targets, contacts, report)
