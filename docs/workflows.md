# Workflows

Run from the repository root, in the environment documented in
[installation](installation.md). Included references need no human recordings.

## Preview the Reference (GMR Environment)

```bash
python retargeting/scripts/vis_robot_motion.py --robot mos \
  --robot_motion_path motions/gmr/mos_rigid_torso/walk_v2.pkl
python retargeting/scripts/vis_robot_motion.py --robot booster_k1 \
  --robot_motion_path motions/gmr/k1/dance_8_clip1.pkl
```

The K1 directory also contains grounding/updated comparisons. The policy's
authoritative reference is `motions/K1/dance_8_clip1.npz`, not an arbitrary
PKL variant. See the [catalog](../motions/README.md).

## Generate a New Reference (Optional)

Use your own authorized human recording and model directory:

```bash
python retargeting/scripts/smplx_to_robot.py \
  --smplx_file /absolute/path/to/authorized_walk.npz \
  --body_model_dir /absolute/path/to/body_models \
  --robot mos --flat_feet --rigid_torso --headless \
  --save_path outputs/mos_new_walk.pkl
```

For skeleton preview, omit `--headless`/`--save_path` and add
`--show_skeleton --rate_limit --loop`. Never add `--loop` to a finite export.
`--rate_limit` controls playback pacing, not joint speed. The MOS contact
solver uses inferred support intervals; verify its report and reference
visually. A supplied `--contact_schedule` must match the resampled recording.
For K1 use `--robot booster_k1` and omit the MOS-only flags.

## Convert to Isaac (Training Environment)

The bundled NPZs are already converted. To reproduce MOS conversion without
overwriting them:

```bash
python training/scripts/csv_to_npz.py --robot mos \
  --gmr_file motions/gmr/mos_rigid_torso/walk_v2.pkl \
  --gmr_model robots/mos/retargeting/MOS9.2.xml \
  --output_name outputs/mos_walk_rigid_v2.npz \
  --output_fps 50 --align_ground --headless --device cuda:0
```

K1 conversion:

```bash
python training/scripts/csv_to_npz.py --robot booster_k1 \
  --gmr_file motions/gmr/k1_locomotion/walk.pkl \
  --gmr_model robots/k1/retargeting/K1_serial.xml \
  --output_name outputs/k1_walk.npz --output_fps 50 \
  --align_ground --headless --device cuda:0
```

CSV input remains supported through `--input_file` and `--input_fps`.
Do not pass CSV and GMR input together. `--gmr_model` proves the ordering of
legacy unnamed GMR joints. Conversion retains floating-point input FPS,
converts XYZW root rotations to WXYZ, resamples, and records metadata.
Ground alignment applies one constant height translation to all body positions.

## One-Environment Policy Playback

MOS **must use the rigid-v2 override** with the included rigid-v2 checkpoint.
The original `MOS-Walk-v0` default remains the earlier flat-foot reference.

```bash
python training/scripts/rsl_rl/play.py \
  --task MOS-Walk-v0-Play --num_envs 1 --device cuda:0 \
  --checkpoint checkpoints/mos_walk_rigid_v2/model_24000.pt \
  --camera_eye 3 0 1.4 --camera_lookat 0 0 0.7 \
  env.commands.motion.motion_file="$PWD/motions/MOS/walk_rigid_v2.npz" \
  agent.experiment_name=mos_walk_rigid_v2
```

```bash
python training/scripts/rsl_rl/play.py \
  --task Booster-K1-Dance_8-v0-Play --num_envs 1 --device cuda:0 \
  --checkpoint checkpoints/k1_dance/model_29999.pt \
  --camera_eye 3 0 1.4 --camera_lookat 0 0 0.7
```

Close the window to stop. Add `--headless --max_steps 100` for a finite
non-rendered rollout; without `--max_steps`, headless no-video mode exports
the policy and exits. Playback writes JIT/ONNX exports beside its checkpoint.
Checkpoint tensors are mapped to the selected device, including a K1 policy
originally saved on GPU 1. Only load trusted checkpoints.

## Fresh Training

These commands start long jobs only when you run them. Use `--num_envs 64
--max_iterations 2` for a short pipeline smoke test.

```bash
export CUDA_VISIBLE_DEVICES=0
python training/scripts/rsl_rl/train.py \
  --task MOS-Walk-v0 --headless --device cuda:0 \
  --num_envs 1024 --max_iterations 30000 --logger tensorboard \
  --run_name mos_walk_rigid_v2_1gpu_from_scratch \
  env.commands.motion.motion_file="$PWD/motions/MOS/walk_rigid_v2.npz" \
  agent.experiment_name=mos_walk_rigid_v2
```

```bash
python training/scripts/rsl_rl/train.py \
  --task Booster-K1-Dance_8-v0 --headless --device cuda:0 \
  --num_envs 1024 --max_iterations 30000 --logger tensorboard \
  --run_name k1_dance_1gpu_from_scratch
```

K1 walking/jogging/running use `Booster-K1-Walk-v0`, `Booster-K1-Jog-v0`,
and `Booster-K1-Run-v0` with their included references. Matching playback IDs
end in `-Play`. No locomotion checkpoints for those three K1 tasks are bundled.
Logs and checkpoints go to `logs/rsl_rl/<experiment>/<timestamp>_<run_name>`.

## Resume

For MOS, repeat the fresh-training command with:

```bash
--resume --checkpoint checkpoints/mos_walk_rigid_v2/model_24000.pt
```

Use a new `--run_name`, retain the rigid-v2 motion override, and set
`--max_iterations` to the number of **additional** iterations. The included
checkpoint is a saved iteration, not proof of completing 30,000 iterations.
K1 resumes with `--resume --checkpoint checkpoints/k1_dance/model_29999.pt`.
Resume restores optimizer state. Do not switch reference or robot and assume
the checkpoint remains compatible.

Existing log runs can still be selected with `--resume --load_run <run_name>
--checkpoint 'model_.*.pt'`.

## Record 20 Seconds in UHD

Add these flags to either one-environment playback command:

```bash
--headless --video --video_length 1000 \
--video_width 3840 --video_height 2160 \
--video_folder outputs/mos_front_uhd
```

At the 50 Hz control rate, 1,000 recorded policy steps are approximately
20 seconds. Encoding/container duration may differ by one frame. Cameras are
enabled automatically. UHD recording needs more GPU memory than plain playback.
Monitor training with `tensorboard --logdir logs/rsl_rl`; Ctrl+C stops a
foreground run. No detached full-training queue is installed or started.
