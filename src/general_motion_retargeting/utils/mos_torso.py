"""Opt-in rigid-torso target adaptation, before contact-constrained MOS IK."""

import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


def adapt_rigid_torso(targets, fps):
    """Preserve foot/leg targets and root translation; soften upper-body motion.

    These are preview heuristics, not a balance or actuator-feasibility solution.
    Heading is unwrapped before filtering so crossing +/-pi cannot reverse it.
    """
    if len(targets) < 3 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Torso adaptation requires three frames and positive FPS")
    result = copy.deepcopy(targets)
    original = R.from_quat([frame["pelvis"][1] for frame in targets], scalar_first=True)
    if not np.isfinite(original.as_quat()).all():
        raise ValueError("Non-finite pelvis rotations")
    angles = original.as_euler("xyz")
    if np.max(np.abs(angles[:, 1])) > np.deg2rad(60):
        raise ValueError("Rigid-torso preview is intended for upright locomotion")
    yaw = np.unwrap(angles[:, 2])
    # Retain slow heading changes and 10% of the alternating pelvis twist.
    trend = gaussian_filter1d(yaw, sigma=max(0.55 * fps, 0.5), mode="nearest")
    heading = trend + 0.10 * (yaw - trend)
    # Keep the source's endpoint heading, including deliberate turns.
    heading += np.linspace(yaw[0] - heading[0], yaw[-1] - heading[-1], len(yaw))
    lean = gaussian_filter1d(angles[:, :2], sigma=max(0.10 * fps, 0.5),
                             axis=0, mode="nearest") * 0.6
    caps = np.deg2rad([8.0, 10.0])
    lean = caps * np.tanh(lean / caps)
    adapted = R.from_euler("xyz", np.column_stack((lean, heading)))
    upper = ("head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
    for i, frame in enumerate(result):
        root = frame["pelvis"][0]
        correction = adapted[i] * original[i].inv()
        frame["pelvis"][1] = adapted[i].as_quat(scalar_first=True)
        for name in upper:
            pos, quat = frame[name]
            frame[name] = [
                root + correction.apply(pos - root),
                (correction * R.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True),
            ]
    report = {
        "mode": "rigid_torso_preview",
        "lean_scale": 0.6, "lean_filter_sigma_s": 0.10,
        "soft_roll_cap_deg": 8.0, "soft_pitch_cap_deg": 10.0,
        "yaw_trend_filter_sigma_s": 0.55, "yaw_residual_scale": 0.10,
        "endpoint_heading_preserved": True,
        "root_translation_and_leg_foot_targets_unchanged": True,
        "source_roll_pitch_peak_deg": np.rad2deg(np.abs(angles[:, :2]).max(axis=0)).tolist(),
        "target_roll_pitch_peak_deg": np.rad2deg(np.abs(lean).max(axis=0)).tolist(),
        "max_target_orientation_change_deg": float(np.rad2deg((original.inv() * adapted).magnitude()).max()),
    }
    return result, report
