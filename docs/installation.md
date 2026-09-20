# Installation

## Two Separate Environments

This is an editable source checkout, not a self-contained simulator or wheel.
Keep the checkout in place after installation: robot assets, references, and
entrypoints are resolved relative to it. Run commands below from its root.
`pip install -e .` exposes `general_motion_retargeting`, `booster_train`,
`booster_assets`, and `wbmr`. Use extras to install workflow dependencies.

Use a fresh terminal when switching between GMR and Isaac. Isaac's `PYTHONPATH`
can otherwise make Python 3.10 import Python 3.11 NumPy binaries. Do not source
the Isaac setup script in the GMR terminal. Only load trusted PKL/PT files:
these formats can execute code.

## Retargeting (Python 3.10)

```bash
conda create -n wbmr-gmr python=3.10 pip -y
conda activate wbmr-gmr
python -m pip install -e '.[retargeting,test]' \
  -c environments/retargeting-constraints.txt
python -c "import general_motion_retargeting, wbmr; print(wbmr.__file__)"
python -m pytest -q
```

The constraints record versions observed in the working environment, not
minimum versions. Package availability on another platform is not guaranteed.
Interactive MuJoCo preview needs a working OpenGL desktop. Offscreen renders
can use `MUJOCO_GL=egl` with a suitable graphics driver. New human retargeting
also needs separately authorized recordings and SMPL-X weights:

```bash
export SMPLX_MODEL_DIR=/absolute/path/to/body_models
```

That directory must contain the `smplx` model subdirectory expected by SMPL-X,
including the gender model required by the recording. Do not commit weights
or human recordings. Included robot-reference preview needs neither.

## Training (Python 3.11)

Install a compatible NVIDIA driver, Isaac Sim **5.1.0**, and Isaac Lab before
installing the training extra. Follow their upstream installation and license
requirements; pip does not install a GPU driver or accept simulator terms for
you.

Tested host: Ubuntu 22.04, NVIDIA Quadro RTX 8000 (48 GB), driver 580.178.04.
Tested Isaac Lab checkout: version file 2.3.2, commit
`f4aa17f87e2e5db5484f0b5974918573e8918ce2`; installed distributions
`isaaclab==0.54.3`, `isaaclab-rl==0.5.0`, `isaaclab_tasks==0.11.14`.
Training uses Torch 2.7.0+cu128 and RSL-RL 5.0.1. These are the tested pairings;
do not upgrade Torch/RSL-RL independently to resolve an unrelated warning.

For an Isaac Sim binary installation with `setup_conda_env.sh`, the setup
sequence is:

```bash
conda create -n wbmr-isaac python=3.11 pip -y
conda activate wbmr-isaac
export ISAACSIM_PATH=/absolute/path/to/isaacsim
source "$ISAACSIM_PATH/setup_conda_env.sh"
git clone https://github.com/isaac-sim/IsaacLab.git /absolute/path/to/IsaacLab
git -C /absolute/path/to/IsaacLab checkout f4aa17f87e2e5db5484f0b5974918573e8918ce2
ln -s "$ISAACSIM_PATH" /absolute/path/to/IsaacLab/_isaac_sim
cd /absolute/path/to/IsaacLab
./isaaclab.sh --install rsl_rl
cd /absolute/path/to/whole-body-motion-retargeting
python -m pip install -e '.[training]' \
  -c environments/training-constraints.txt
```

Use real absolute paths for the examples. If your simulator installation uses
pip instead of the binary distribution, use the upstream pip setup, not the
binary `source` command. The runtime itself remains an external prerequisite.
No files in an old GMR or Booster checkout are required.

The tested binary runtime also needed the environment's C++ runtime before
system libraries:

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
nvidia-smi
python training/scripts/tests/check_mos_task.py --headless
```

Isaac may fetch its built-in textures/materials on first use and creates USD
conversion caches. Those simulator resources are not bundled. Network access
may be needed. 64 environments are used for smoke tests; 1,024 training
environments were used on this 48 GB GPU, not promised on every GPU.

The local validation used disposable editable-install venvs inheriting the
working simulator/GMR dependencies. It did not reinstall the original
environments or prove a fully clean driver/runtime installation.
