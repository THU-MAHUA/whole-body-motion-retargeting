# MOS Rigid-Torso Preview

Current reference: `walk_v2.pkl`. This document records the original kinematic
experiment; the later rigid-v2 RL checkpoint is now included in this standalone
project. The original JSON preview-only status is retained as historical
evidence, not a statement that no later training occurred. `walk.pkl` is the
first, superseded experiment, not the original flat-foot reference.

Original, unchanged: `../mos_flat_feet/walk.pkl`.
Both use 228 frames at 29.901639344262296 FPS and the same contact schedule.
Robot URDF, MJCF, joint axes, and original motion were not modified.

## Changes

The opt-in `--rigid_torso` flag requires `--robot mos --flat_feet`.
It smooths pelvis roll/pitch with a 0.10-second Gaussian sigma, scales lean
by 0.6, and applies soft target caps of 8/10 degrees. It smooths unwrapped yaw
with a 0.55-second sigma and retains 10% of the residual, preserving target
endpoint heading. These are preview heuristics, not motor or balance limits.

Upper-body targets rotate with the torso. Their IK costs are multiplied by
0.25; pelvis orientation cost is 60 instead of 10. Root-position, leg, and
planned foot targets are unchanged; solved poses can differ. Stance feet
remain equality constraints. No virtual waist or per-frame grounding is added.
Default MOS and K1 behavior are unchanged.

## Measurements

| Metric | Original | Candidate |
| --- | ---: | ---: |
| Peak torso roll, degrees | 6.669 | 3.009 |
| Peak torso pitch, degrees | 12.914 | 5.999 |
| Yaw residual RMS, degrees | 1.568 | 0.612 |
| Peak root angular speed, rad/s | 1.569 | 0.371 |
| Peak joint step, rad/frame | 0.494 | 0.508 |
| Peak right ankle-roll angle, degrees | 12.178 | 12.533 |
| Peak left ankle-roll angle, degrees | 12.629 | 12.959 |

Yaw residual is measured against a 0.35-second Gaussian trend for both clips.
The largest joint step remains the left knee `Ll1_Ll2`, at frame 190
(zero-based). It is slightly worse, not fixed. Peak finite-difference joint
speed is about 15.2 rad/s. Address this before accepting a training reference.
Flat stance soles do not require zero ankle joint angles.

Independent FK checks passed: stance height/drift under 1 mm, tilt under
0.5 degrees, and no sole penetration beyond 1 mm. Actual errors are much
smaller; see `walk_v2.validation.json`. These checks do not establish dynamic
balance, actuator feasibility, or hardware safety. Loop endpoints are not
periodic.

## Preview

From the repository root with the GMR environment:

```bash
python retargeting/scripts/vis_robot_motion.py \
  --robot mos \
  --robot_motion_path motions/gmr/mos_rigid_torso/walk_v2.pkl
```

Stop playback with Ctrl+C. To recompute and view the adapted target skeleton:

```bash
python retargeting/scripts/smplx_to_robot.py \
  --smplx_file /absolute/path/to/authorized_human_data/ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz \
  --robot mos --flat_feet --rigid_torso \
  --contact_schedule motions/gmr/mos_flat_feet/walk.contacts.json \
  --show_skeleton --rate_limit --loop
```

The overlay shows MOS-adapted targets, not the untouched human skeleton.
For another export, omit `--loop`, add `--headless`, and use a new
`--save_path`; experimental exports refuse an existing output file.

`walk_v2.compare.mp4` shows original/candidate columns and front/side rows.
`walk_v2.comparison.json` contains measurements and preservation hashes.
All 26 MOS tests passed. A live 912-frame preview completed with front/side
views, with and without the adapted target skeleton.

Human-source paths above are placeholders for separately authorized recordings, not bundled files. Set `SMPLX_MODEL_DIR` to your external body-model directory before human-source commands. For current installation and training commands, use the root README.
