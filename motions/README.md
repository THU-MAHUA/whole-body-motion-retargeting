# Motion Catalog and Provenance

## Recommended for the Included Checkpoints

| Robot | Authoritative training reference | Checkpoint |
| --- | --- | --- |
| MOS | `MOS/walk_rigid_v2.npz` | `../checkpoints/mos_walk_rigid_v2/model_24000.pt` |
| K1 | `K1/dance_8_clip1.npz` | `../checkpoints/k1_dance/model_29999.pt` |

MOS's matching GMR reference is `gmr/mos_rigid_torso/walk_v2.pkl`. It has 20
joint angles; K1 references have 22. GMR uses XYZ root positions and XYZW
quaternions; Isaac archives use WXYZ body quaternions and named joint/body
arrays at 50 FPS. Do not interchange these conventions.

The matching training configurations are retained beside each checkpoint.
Paths inside those historical YAML files were relocated, but numerical
settings were not changed. They are provenance snapshots, not automatically
loaded by the CLI; task configs plus the documented MOS override reproduce
the intended reference selection. The checkpoint binaries are unchanged.

## Historical Comparisons

- `MOS/walk.npz`, `gmr/mos_flat_feet`: previous flat-foot walking baseline.
- `gmr/mos_rigid_torso/walk.pkl`: first rigid-torso experiment, superseded by v2.
- `gmr/mos_locomotion_calibrated`: early MOS calibration.
- `gmr/mos_locomotion_torso_aligned`: subsequent anatomical torso alignment.
- `gmr/mos_virtual_waist`: experimental preview only; invented waist, 31 qpos,
  explicitly incompatible with the real MOS model and training loader.
- `gmr/k1`, `gmr/k1_grounded`, `gmr/k1_updated`: retained dance exports and CSV
  variants. Filenames alone do not prove which PKL produced the training NPZ.
- `gmr/k1_locomotion`, `K1/{walk,jog,run}.npz`: separate locomotion candidates;
  no trained policy for these is bundled.

Reports and comparison videos are historical evidence, not blanket approvals
to run hardware. See [MOS corrections](../docs/mos-walking.md).

## Human-Source Provenance (No Source Files Included)

Walking variants derive from ACCAD
`Female1Walking_c3d/B3_-_walk1_stageii.npz`; jogging from CMU
`02/02_03_stageii.npz`; running from ACCAD
`Male2Running_c3d/C3_-_run_stageii.npz`, supplied as AMASS-style SMPL-X fits.
CMU 02_03 is a run/jog candidate, not a calibrated target-speed command.
The dance source was named `dance_8_clip1.npz`; its underlying recording rights
have not been independently verified.

`external://human/...` report entries are provenance labels, not paths that
must exist after cloning. Generating new references requires external licensed
recordings and body models; preview/training of included robot references does
not. Removing human-source files does not establish new redistribution rights
for derived motions, learned weights, or videos.

The maintainer confirmed public redistribution permission on September 20,
2026; see [NOTICE](../NOTICE.md) for the scope and remaining upstream terms.
