# MOS Walking: What Changed

MOS has 20 real joints, including three hip axes per leg, but no independent
waist/spine joint. The problem was not simply a missing hip DOF. Human pelvis
and upper-spine motions cannot both be reproduced exactly by a rigid torso.
The corrections below adapt the reference to that structure.

## Geometry and Targets

1. **Mesh transforms.** The custom URDF converter had interpreted fixed-axis
   URDF roll/pitch/yaw as MuJoCo Euler rotations. It now emits explicit WXYZ
   quaternions and retains visual origins, scales, colors, and inertial
   transforms. Simplified visual STLs preserve the assembly; hidden collision
   primitives are not the intended appearance.
2. **MOS anatomical targets.** Offsets and segment lengths come from MOS
   forward kinematics instead of K1. Feet track ankles, not toes. Head and
   shoulder positional landmarks follow the rigid pelvis-driven torso;
   segment orientation targets still carry human motion.
3. **Centered display landmarks.** The head/neck overlay is centered laterally,
   correcting a 33 mm joint-origin offset in the display only. This does not
   change the URDF, joint axes, IK target, saved pose, or human head turns.

## Contacts and Rigid-Torso Adaptation

The opt-in flat-foot solver infers stance phases and constrains stationary
sole poses during stance. Swing clearance and pelvis/leg adjustments are
allowed. A flat sole in world coordinates does **not** mean zero ankle angle.
Do not transfer unsupported waist motion into the ankles or lock the whole
body upright. Contact inference is heuristic, not measured force data.

`--rigid_torso` requires MOS `--flat_feet`. Roll/pitch targets use a
0.10-second Gaussian sigma, 0.6 amplitude, and soft caps of 8/10 degrees.
Unwrapped yaw uses a 0.55-second sigma and retains 10% of the residual while
preserving endpoint heading. Upper-body tracking weights are multiplied by
0.25, pelvis orientation cost changes from 10 to 60, and root-position cost
stays 20. Legs and planned support poses retain priority. The solved pose can
differ because the objectives compete. No physical waist joint is invented.

Isaac conversion maps 228 frames at 29.901639344262296 FPS to 380 frames at
50 FPS. A single **+0.004150 m** vertical correction aligns the lowest
training-URDF sole point to zero across the clip. Every body is shifted
consistently; relative vertical motion, joint angles, and velocities remain
unchanged. This is not per-frame grounding.

## Measured Reference Changes

From the retained baseline/rigid-v2 comparison:

| Metric | Flat-foot baseline | Rigid-v2 |
| --- | ---: | ---: |
| Peak torso roll | 6.669 degrees | 3.009 degrees |
| Peak torso pitch | 12.914 degrees | 5.999 degrees |
| Yaw residual RMS | 1.568 degrees | 0.612 degrees |
| Peak root angular speed | 1.569 rad/s | 0.371 rad/s |
| Largest joint step | 0.4935 rad | 0.5084 rad |

Kinematic stance checks passed the 1 mm height/drift and 0.5 degree tilt
tolerances. However, the left knee transition at frame 190 became slightly
worse (about 15.2 rad/s finite-difference speed), and peak ankle roll increased
by about 0.35 degrees. No joint clamping concealed these concerns.

Historical JSON reports retain their original preview-only status. They were
written before the subsequent rigid-v2 RL run. Including its checkpoint and
simulation video does not retroactively turn a geometric test into a dynamic
stability or hardware-safety result.

## Remaining Limitations

The clip is short and nonperiodic. Loop/reset transitions are not seamless.
Full-sole stance replaces human heel-to-toe rocking. Self-collision, true
actuator limits, torque margins, balance recovery, and real contacts have not
been established. MOS training travel limits, gains, and actuator parameters
are provisional; the source URDF declares continuous joints.

Earlier calibrated/torso-aligned references retain visible floating/sliding
and imperfect shoulder matching. The two shoulder axes cannot reproduce all
human shoulder twist. The included PPO policy demonstrates simulated tracking,
not speed-commanded locomotion or sim-to-real readiness.

The virtual-waist comparison adds an invented ball joint and placeholder
inertia. Its 31-coordinate poses are **not compatible with real-MOS training**.
Legs were frozen in that comparison, so it cannot show reduced ankle
compensation. It is a kinematic comparison, not a mechanical design.
