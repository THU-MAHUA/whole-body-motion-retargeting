# Validation

Local packaging checks on **September 20, 2026**. These checks test pipeline
operation, not policy convergence, dynamic stability, or hardware safety.
No full-training job was started; original projects and environments were not
edited. Temporary validation environments inherited the existing installed
runtime dependencies.

## Results

- GMR/MOS geometry, targets, rigid-torso, virtual-waist, ground-alignment and
  conversion tests passed (40 original tests).
- Artifact checks validate 16 GMR PKLs, three CSVs, seven NPZs (including the
  incompatible virtual-waist comparison), and two selected checkpoints.
  Checks cover finite arrays, frame counts, floating-point FPS, joint/body
  coverage, quaternion normalization, 50 FPS training output, nominal sole
  alignment, checkpoint tensor finiteness, and policy dimensions.
- MOS and K1 supplied policies each completed 100 one-environment headless
  steps with finite actions/rewards. K1 loading required explicit device
  mapping because its original tensor storages were saved on GPU 1.
- Each demo task trained for two PPO iterations with 64 environments, saved
  a new checkpoint, and reloaded it for 24 one-environment steps.
- MOS re-conversion reproduced 380 frames at 50 FPS and the +0.004150 m
  constant ground translation. The supplied reference was not overwritten.
- Model/task configuration checks passed, retaining rough-terrain training,
  flat-ground playback, original anchors, 0.02-second control interval, and
  `tail_len=0`.

- An independent local Git clone passed editable installation in both
  disposable environments, all 42 tests, and the 28 artifact/checkpoint checks
  including manifest checksums. Project imports and resources resolved from
  the relocated checkout. Both supplied policies also passed 24-step
  one-environment rollouts there.
- All 74 TensorBoard scalar values from each two-iteration training test
  were finite. MOS re-conversion arrays matched the included NPZ exactly.
- Both requested videos decoded completely with no FFmpeg errors.
- The tracked-file audit found no human-source recordings, old absolute
  workspace paths, credential-pattern matches, or files over 100 MiB.
  Local Markdown links resolved. Hash checks confirmed 339 original source
  files remained unchanged.

See `packaging-validation.json` for the machine-readable summary. Inherited
upstream files retain their original whitespace; no blanket reformatting of
vendored code, model files, or license text was performed.

## Publication Status

Local Git history uses the requested repository-local author identity and
has the requested GitHub remote configured. The initial HTTPS
`git push --dry-run` failed because credentials were unavailable.
On September 20, 2026, the maintainer subsequently completed an authenticated
dry-run and confirmed public redistribution permission for the included assets.
See [NOTICE](../NOTICE.md) for the scope of this confirmation. Publication
status is recorded in `packaging-validation.json`. Public upload succeeded on
September 20, 2026. GitHub's API confirmed the repository is public and all
312 file hashes matched local commit `c8986cc`, including both selected
checkpoints and both requested demo videos.

## Reproduce

GMR environment:

```bash
python -m pytest -q
python tools/validate_artifacts.py --checkpoints --verify-checksums
```

Isaac environment:

```bash
python training/scripts/tests/check_mos_task.py --headless
python training/scripts/rsl_rl/play.py \
  --task MOS-Walk-v0-Play --checkpoint checkpoints/mos_walk_rigid_v2/model_24000.pt \
  --headless --device cuda:0 --num_envs 1 --max_steps 100 \
  env.commands.motion.motion_file="$PWD/motions/MOS/walk_rigid_v2.npz" \
  agent.experiment_name=mos_walk_rigid_v2
python training/scripts/rsl_rl/play.py \
  --task Booster-K1-Dance_8-v0-Play --checkpoint checkpoints/k1_dance/model_29999.pt \
  --headless --device cuda:0 --num_envs 1 --max_steps 100
```

Use the [training commands](workflows.md) with `--max_iterations 2 --num_envs 64`
and separate validation experiment names to repeat the PPO checks.
Untrained two-iteration policies can fail tracking/terminate quickly; the
test establishes finite learning, saving, and reloading, not skill.

## Artifact Selection

K1: exact `model_29999.pt` from
`2026-08-20_20-30-21_l20_gpu1_resume`.

MOS: latest completed checkpoint write, `model_24000.pt`, from
`2026-09-19_14-05-00_mos_walk_rigid_v2_1gpu_from_scratch`.
The binary itself loads and reports iteration 24000. It is not relabeled as
a completed 30,000-iteration policy.

The two demo videos were initially copied byte-for-byte. At the maintainer's
request, each published video was replaced with its first 6 seconds, re-encoded
as H.264 with a streaming-friendly MP4 header. Both retain 3840x2160 resolution
and 50 FPS (300 frames). Full-length recordings in the original projects were
not modified. README thumbnails link directly to the MP4s, bypassing GitHub's
large-file preview page. The K1 video came from the requested run directory,
but its exact recording-checkpoint selection was not independently established.

## Not Established

A clean installation of drivers/Isaac on another physical machine, long-horizon
robustness, deterministic learning, sim-to-real performance, independent review
of asset permission documents, or hardware safety. First-run simulator resource
downloads and version-specific installation remain external prerequisites.
See [NOTICE](../NOTICE.md) before public upload.
