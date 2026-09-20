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

## Asset Permissions: Public Upload Blocked Pending Confirmation

The local package intentionally contains the requested generated references,
checkpoints, and videos, but **has not established redistribution clearance**
for all of them:

1. ACCAD/CMU locomotion came through AMASS-style fitted recordings.
   [AMASS terms](https://amass.is.tue.mpg.de/license.html) restrict redistribution
   and modification of the dataset. Removing raw recordings alone does not
   resolve whether these derived robot trajectories may be published. Obtain
   applicable permission/confirmation for the generated references and their
   derived weights/media. This is an unresolved provenance question, not a
   legal conclusion about every learned model.
2. The original `dance_8_clip1` recording's rights/provenance must be confirmed
   for the dance reference, checkpoint, and video.
3. MOS URDF/STL ownership and public redistribution authorization must be
   confirmed. The supplied files did not provide a separate asset license.

No AMASS/ACCAD/CMU source datasets or body-model weights are bundled. Their
absence is not a representation that all derivative rights have been granted.
Do not publish this artifact-containing snapshot until these issues are
resolved. A code-only release would be a different scope, not silently
substituted for the requested repository.
