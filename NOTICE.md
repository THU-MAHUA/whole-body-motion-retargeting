# Notices, Attribution, and Release Status

Maintained by **MAAROUF MOHAMED** (`mohamed.maarouf05@gmail.com`).
No new blanket license is granted for original contributions. A public GitHub
repository is not a license waiver and is not automatically unrestricted
open-source software.

## Upstream Code

- [GMR](https://github.com/YanjieZe/GMR): retargeting foundation. Its MIT notice
  is retained in `LICENSES/GMR-MIT.txt`.
- [Booster Train](https://github.com/BoosterRobotics/booster_train):
  Isaac Lab tracking/training foundation. Its upstream notice is retained in
  `LICENSES/BoosterTrain-Apache-2.0.txt`, with the full Apache text separately.
- [Booster Assets](https://github.com/BoosterRobotics/booster_assets):
  K1 assets; retained BSD notice in `LICENSES/BoosterAssets-BSD-3-Clause.txt`.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): retained BSD source
  headers and license. Simulator packages remain external dependencies.
- [Whole-Body Tracking](https://github.com/HybridRobotics/whole_body_tracking):
  tracking foundations; retained MIT license.
- Vendored LAFAN quaternion utilities retain their separate
  **CC BY-NC-ND 4.0** notice at
  `src/general_motion_retargeting/utils/lafan_vendor/license.txt`.
  They are not covered by a new MIT grant. NVIDIA utility headers remain.

Local modifications include MOS model conversion, anatomical/display targets,
contact constraints, torso adaptation, robot/task bindings, motion conversion,
camera/video controls, and packaging portability. Inherited authorship and
licenses remain applicable. See [citations](docs/citations.bib).

## Asset Permissions and Provenance

On September 20, 2026, the maintainer confirmed permission to publicly
distribute the MOS models and the included motion-derived references,
checkpoints, and videos. Publication relies on that confirmation; supporting
permission documents were not independently reviewed during packaging.
This confirmation does not replace upstream terms or grant new rights to users.

1. ACCAD/CMU locomotion came through AMASS-style fitted recordings.
   [AMASS terms](https://amass.is.tue.mpg.de/license.html) restrict redistribution
   and modification of the dataset. Removing raw recordings alone does not
   resolve whether these derived robot trajectories may be published.
   The maintainer's confirmation covers the included generated references
   and their derived weights/media; no source-dataset rights are granted here.
2. The maintainer confirmed permission for the `dance_8_clip1` derived
   reference, checkpoint, and video. The original recording is not bundled.
3. The maintainer confirmed public redistribution authorization for the MOS
   URDF/STL files. The supplied files did not provide a separate asset license.

No AMASS/ACCAD/CMU source datasets or body-model weights are bundled. Their
absence alone does not establish derivative rights. No new blanket license
is added for models, generated artifacts, or original contributions.
