# MOS Model Status

Source: `robots/mos/source/urdf/MOS9.2_urdf_0308-4.urdf`.
Original URDF and STL files are unchanged. Visual meshes in `meshes/`
are simplified copies, not exact high-resolution replicas.

The preview scene matches K1's checkerboard floor, skybox, and two lights.
Only scene appearance is shared; MOS floor height, contact parameters, robot
geometry, and retargeting remain unchanged. The converter preserves this setup.

## Assembly Correction

The original custom converter copied URDF fixed-axis roll/pitch/yaw into
MuJoCo's default Euler convention. This produced incorrect link transforms
(including 180-degree rotations and roughly 0.48 m ankle-position errors).
The converter now emits explicit WXYZ quaternions and preserves URDF visual
origins, mesh scales, colors, and inertial-frame rotations.

From the repository root, in the GMR environment:

```bash
python retargeting/tools/urdf_to_mjcf_mos.py
python retargeting/tools/test_mos_model.py -v
python retargeting/tools/preview_mos.py
```

The preview holds the URDF zero-joint pose, translated vertically to place
the lowest foot vertex on the floor. It does not use IK or step dynamics.
Tests compare all 21 link frames and joint axes at zero and ten random poses,
check visual mesh bounds against URDF placement, and verify joint ordering.
Front and side reference renders are in `validation/neutral_*.png`.

## MOS Retargeting Calibration

`smplx_to_mos.json` now selects a MOS-only anatomical target builder in
`general_motion_retargeting/utils/mos.py`. Rotation offsets come from the
corrected model's zero-pose FK, not the K1 configuration. MOS upper arms
track human shoulders, forearms track elbows, thighs track hips, shins track
knees, and feet track ankles rather than toe joints.

Target bone vectors use MOS dimensions and human segment rotations. Root
translation uses the configuration scale with inverse human-height scaling.
The first solve is initialized at the target root and allowed to converge.
The IK stopping criterion uses the actual weighted MOS objective, ignoring
zero-cost target components. K1 configuration and target behavior are unchanged.
The calibrated skeleton overlay shows targets with centered MOS torso display
landmarks, not a full human mesh.

Head and shoulder position targets follow the pelvis-driven rigid torso, not
the human upper-spine rotation. MOS has no independent spine joint. Head and
arm orientation targets still follow their human segments. The head IK target
represents the head joint origin, not the visible mesh center; small residual
position differences can remain because neck rotation moves that origin and
head position is not an IK objective. This prevents unsupported human spine
lean from displacing the torso overlay without forcing the robot to bend.

With `--show_skeleton`, the displayed head point is centered laterally in the
target pelvis frame, removing the URDF head-joint origin's 33 mm sideways
offset. Its forward position and height are preserved. The interpolated spine
and neck markers follow this centered point. This is a display-only landmark,
not the exact mesh center or joint origin; head orientation, IK targets,
robot poses, saved motions, URDF, and joint axes are unchanged. Human head
turns and residual target-versus-solved-pelvis differences remain visible.

From the repository root, preview walking with the same corrected visual model:

```bash
python retargeting/scripts/smplx_to_robot.py \
  --smplx_file /absolute/path/to/authorized_human_data/ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz \
  --robot mos --align_ground --rate_limit --show_skeleton --loop
```

For jogging, use
`/absolute/path/to/authorized_human_data/CMU/02/02_03_stageii.npz`.
For running, use
`/absolute/path/to/authorized_human_data/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz`.

`--align_ground` precomputes the complete MOS clip and uses the lowest
rendered foot-mesh vertex across every frame to apply ONE constant vertical
translation. It does not clamp joints, change rotations, remove vertical
displacement differences, or enforce contacts. It cannot be combined with
`--offset_to_ground`, which instead adjusts foot targets each frame and can
suppress running flight. No ground correction is applied unless requested.

For a finite export without a viewer, replace `--loop` with
`--headless --save_path motions/gmr/mos_locomotion_calibrated/walk.pkl`.
All MOS frames, including the first, are exported. Pickles retain the GMR
format: floating-point FPS, XYZ root positions, XYZW root quaternions, and
20 joint angles in model order.

## Opt-In Flat-Foot Walking

For flat, stationary support soles, use `--flat_feet` instead of either
grounding flag:

```bash
python retargeting/scripts/smplx_to_robot.py \
  --smplx_file /absolute/path/to/authorized_human_data/ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz \
  --robot mos --flat_feet --rate_limit --show_skeleton --loop
```

This MOS-only mode precomputes the clip before opening the viewer. It
calibrates the sole from the actual foot mesh, infers support intervals from
human ankle/toe height and speed, and imposes fixed sole poses during support.
World-frame sole roll and pitch are zero during support; ankle joint angles
are not zeroed. Swing feet may tilt and are given clearance. Swing paths are
reshaped between contacts; the pelvis and leg joints can adjust to meet the
constraints. The torso target correction is preserved, and K1 is unchanged.

For export, replace `--rate_limit --show_skeleton --loop` with:

```bash
--headless --save_path motions/gmr/mos_flat_feet/walk.pkl
```

With a save path, contact diagnostics are written beside the pickle as
`walk.contacts.json`. Preview-only diagnostics go to
`motions/gmr/mos_flat_feet/<source-stem>.contacts.json`.
The report includes the inferred contact mask, half-open intervals, floor
shift, contact residuals, pelvis correction, and largest joint step.
Unmet stance/floor tolerances or excessive pelvis corrections abort before
export; solver/contact failures write a failed report. Existing output files
are not deleted on failure, so check the exit status and report status.

Review contact timing visually. To override inference, pass
`--contact_schedule path/to/contacts.json` containing
`{"contact_mask": [[true, false], [true, true], ...]}` with exactly one boolean
pair per resampled frame, ordered `[left, right]`. Both false allows flight;
an entirely unsupported clip is rejected. Contact inference is heuristic,
not measured contact or a dynamic-balance guarantee.

```bash
python retargeting/tools/test_mos_contacts.py -v
MUJOCO_GL=egl python retargeting/tools/validate_mos_contacts.py \
  motions/gmr/mos_flat_feet/walk.pkl
```

The independent saved-motion validator writes `walk.validation.json` and
`walk.poses.jpg` beside the pickle. It verifies finite poses, quaternion
normalization, mesh clearance, and sole flatness/stationarity, then renders
front/side poses including the largest joint transition.

Walking validation on 2026-09-17 passed for 228 frames at 29.902 FPS:
stance heights and drift are below 1 mm, sole tilt below 0.5 degrees,
and no foot mesh penetrates more than 1 mm. The largest pelvis translation
adjustment is 4.0 cm. The largest joint step is **0.494 rad** at frame 190
in `Ll1_Ll2`, shortly after left-foot lift-off, compared with 0.264 rad
for the constant-ground-alignment result. This sharp transition remains a
motion-quality concern, not a validated actuator command.

Only walking has been validated in this mode. Full-sole constraints replace
human heel/toe rocking; no dynamic balance, physical contacts, self-collision,
actuator speeds, or mechanical joint limits are validated. Loop boundaries
remain nonperiodic. Do not treat this preview correction as hardware-ready
or training-ready locomotion.

## Validation and Remaining Limitations

```bash
python retargeting/tools/test_mos_model.py -v
python retargeting/tools/test_mos_retarget.py -v
MUJOCO_GL=egl python retargeting/tools/validate_mos_retarget.py \
  --data_root /absolute/path/to/authorized_human_data \
  --body_model_dir /absolute/path/to/body_models
python retargeting/scripts/vis_robot_motion.py --robot mos \
  --robot_motion_path motions/gmr/mos_locomotion_calibrated/walk.pkl
```

The validator regenerates `walk.pkl`, `jog.pkl`, and `run.pkl` under
`motions/gmr/mos_locomotion_calibrated/`, with JSON diagnostics and
front/side pose montages. It checks all frames, quaternion normalization,
finite values, and mesh-floor alignment. Original inputs are unchanged.
It also reports head-target versus head-link-origin errors and saves
`*_skeleton.jpg` front/side overlays. To keep existing exports, use
`--output motions/gmr/mos_locomotion_torso_aligned`.
Old `validation/walk_current_ik.png` is an uncalibrated comparison image.

Local validation on 2026-09-17:

| Clip | Frames | FPS | Constant Floor Shift | Largest Joint Step |
| --- | ---: | ---: | ---: | ---: |
| Walk | 228 | 29.902 | +0.0801 m | 0.264 rad |
| Jog | 43 | 29.827 | +0.0893 m | 0.419 rad |
| Run | 81 | 29.816 | +0.1044 m | 0.705 rad |

The corrected poses are anatomically recognizable in front/side renders,
but these are not contact-accurate locomotion references. The lowest foot
is still roughly 5-6 cm above the floor at the median frame after constant
alignment. Visible floating/sliding and imperfect arm matching remain.
MOS has two shoulder axes and cannot reproduce all human shoulder twist.
No joint-limit or actuator-speed compliance can be established from this URDF.
The running clip also has a sharp source knee change at its first transition
(human knee rotation-vector X goes from approximately 0.798 to 1.487 rad).
No source frames were silently trimmed or smoothed. Loop boundaries are not
made periodic. Revisit contacts, motion quality, and real limits before training
or hardware use; numerical/render validation is not physical validation.

The URDF declares all 20 joints continuous without mechanical limits.
Group-3 collision primitives are legacy approximations, hidden by default,
not validated simulation geometry. This model is for kinematic inspection,
not hardware execution or physical-controller validation.

Human-source paths above are placeholders for separately authorized recordings, not bundled files. Set `SMPLX_MODEL_DIR` to your external body-model directory before human-source commands. For current installation and training commands, use the root README.
