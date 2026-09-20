# K1 Locomotion Starter Clips

Generated on September 6, 2026 with the existing `smplx_to_robot.py`,
`booster_k1`, and unchanged K1 IK configuration. No per-frame ground offset,
retiming, smoothing, or postprocessing was applied.

| Output | Source under `/absolute/path/to/authorized_human_data/` | Frames | FPS | Duration |
| --- | --- | ---: | ---: | ---: |
| `walk.pkl` | `ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz` | 227 | 29.9016 | 7.59 s |
| `jog.pkl` | `CMU/02/02_03_stageii.npz` | 42 | 29.8266 | 1.41 s |
| `run.pkl` | `ACCAD/Male2Running_c3d/C3_-_run_stageii.npz` | 80 | 29.8160 | 2.68 s |

`jog.pkl` is a candidate from CMU's run/jog-labeled trial, not a validated
steady-state jogging cycle.

## Replay Saved Motions

```bash
conda activate wbmr-gmr
# From the standalone repository root

python retargeting/scripts/vis_robot_motion.py \
  --robot booster_k1 \
  --robot_motion_path motions/gmr/k1_locomotion/walk.pkl
```

Replace `walk.pkl` with `jog.pkl` or `run.pkl`. Replay loops until interrupted
with Ctrl+C. It repeats the recorded trajectory, not a seamlessly stitched
locomotion cycle.

## Preview With the Human Skeleton

```bash
WALK=/absolute/path/to/authorized_human_data/ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz
JOG=/absolute/path/to/authorized_human_data/CMU/02/02_03_stageii.npz
RUN=/absolute/path/to/authorized_human_data/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz

python retargeting/scripts/smplx_to_robot.py \
  --smplx_file "$WALK" \
  --robot booster_k1 \
  --rate_limit --show_skeleton --loop
```

Replace `"$WALK"` with `"$JOG"` or `"$RUN"`. Stop with Ctrl+C.
`--rate_limit` controls playback pacing, not robot actuator limits.
`--offset_to_ground` is deliberately omitted because its per-frame adjustment
can alter airborne motion.

To re-export, remove `--loop` and add `--save_path` with a new output filename.
The script saves only after its finite processing loop finishes.

## Validation and Limitations

All exports have matching array lengths, finite values, normalized XYZW root
quaternions, positive FPS, and 22 joint angles per frame. Joint-limit violations
are at floating-point roundoff level or zero. The current exporter skips the
first aligned frame; counts above match that existing behavior.

Each `*_validation.json` contains full-motion kinematic measurements.
`previews/` contains a camera-following MP4 and an eight-pose image for each
motion, rendered from the saved output. These previews show the robot only;
the interactive source preview provides the human skeleton overlay.

| Motion | Lowest foot collision-box height | Frames below -1 cm | Frames with both feet above 2 cm |
| --- | ---: | ---: | ---: |
| Walk | -5.81 cm | 107 / 227 | 1 / 227 |
| Jog | -7.53 cm | 37 / 42 | 0 / 42 |
| Run | -5.92 cm | 43 / 80 | 13 / 80 |

Rendered poses show alternating legs and arm swing. Floor penetration remains,
and near-floor foot-center motion indicates possible sliding, particularly in
jog/run. This is a heuristic, not a foot-contact classifier. Running reaches
the shoulder-pitch upper limit. The airborne-height count alone does not prove
natural flight timing, especially with the current ground alignment.

These are raw preview/reference outputs, not accepted contact-corrected motion
or dynamically validated robot controllers. Ground/contact cleanup and further
motion-quality review are needed before downstream training or hardware use.

Human-source paths above are placeholders for separately authorized recordings, not bundled files. Set `SMPLX_MODEL_DIR` to your external body-model directory before human-source commands. For current installation and training commands, use the root README.
