# Virtual Waist: Comparison Only

This is an invented three-axis waist, not the real MOS robot.
No training was launched or changed. The real URDF/MJCF and source motions
are preserved. Do not use this archive as a real-MOS training motion.

## What Is Compared

- Left: the real MOS `mos_rigid_torso/walk_v2.pkl` reference.
- Right: the same pelvis/root and all 12 leg joint angles, with a virtual
  ball joint at the original `body` origin.
- The intact torso mesh, neck, head, and arms are moved under that joint.
  Zero waist rotation reproduces the original mesh transforms.
- The chest follows the source SMPL-X `spine3` orientation, without filtering.
  Head and arm joints are re-solved against the original calibrated human
  orientation targets, with posture and temporal regularization.
- The legs are deliberately NOT re-solved. This isolates upper-body freedom
  and cannot demonstrate reduced ankle compensation or improved balance.

The ball joint uses a four-component WXYZ quaternion and adds three
rotational DOFs. `comparison.npz` contains full 31-coordinate MuJoCo poses,
not the real robot's 20 scalar joint angles. Its format is explicitly marked
`mos_virtual_waist_preview_v1` and `training_compatible=False`.

## Results

228 frames at 29.901639344262296 FPS. Maximum virtual-waist rotation is
18.620 degrees. Peak absolute local XYZ Euler angles are approximately
7.10 / 14.97 / 16.31 degrees.

Foot positions differ from the baseline by zero in this validation. Stance
sole height and tilt pass the existing 1 mm / 0.5 degree tolerances.
Chest tracking is exact by construction, not evidence of dynamic stability.
The existing sharp knee transition and nonperiodic loop reset remain.
Upper-body self-collision and physical joint limits are not validated.

The invented pelvis inertia is a compiler placeholder. Playback runs forward
kinematics only; this model is not suitable for physics or RL training.
The mesh is not split into a manufactured waist assembly, so this is not a
mechanical design either.

## Playback

This invocation also works from a terminal previously configured for Isaac Sim:

```bash
# From the standalone repository root
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$CONDA_PREFIX/bin/python" \
  retargeting/tools/preview_mos_virtual_waist.py
```

- `V`: toggle virtual waist / original rigid torso at the current frame.
- `F`: toggle front / side camera.
- Close the window or Ctrl+C to stop.
- Optional `--mode rigid`, `--view side`, or `--cycles 2`.

`comparison.mp4` has rigid/virtual columns and front/side rows.
`report.json` records measurements, limitations, and hashes of preserved inputs.

## Reproduce

```bash
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$CONDA_PREFIX/bin/python" \
  retargeting/tools/preview_mos_virtual_waist.py \
  --build --render --directory outputs/mos_virtual_waist_new \
  --smplx_file /absolute/path/to/authorized_walk.npz \
  --body_model_dir /absolute/path/to/body_models
```

Build refuses an existing directory. The default baseline is `walk_v2.pkl`.
To regenerate only this comparison video, use `--render`.

All 28 MOS regression tests passed, including zero-waist mesh agreement,
joint-name mapping, and leg/foot invariance under nonzero waist and arm motion.
