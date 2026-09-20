"""Inspect the MOS URDF zero pose without IK or physics stepping."""

import argparse
import time
from pathlib import Path

import mujoco as mj
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--azimuth", type=float, default=25)
    args = parser.parse_args()
    from wbmr.paths import ROBOTS
    model_path = ROBOTS / "mos/retargeting/MOS9.2.xml"
    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    lowest = np.inf
    for name in ("Lfoot_visual", "Rfoot_visual"):
        gid = model.geom(name).id
        mid = model.geom_dataid[gid]
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        lowest = min(lowest, world[:, 2].min())
    data.qpos[2] -= lowest
    mj.mj_forward(model, data)

    def camera(cam):
        cam.lookat[:] = [0, 0, data.qpos[2] - 0.06]
        cam.distance = 1.7
        cam.azimuth = args.azimuth
        cam.elevation = -12

    if args.screenshot:
        from PIL import Image
        cam = mj.MjvCamera()
        camera(cam)
        with mj.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera=cam)
            image = renderer.render()
            if image.std() < 2:
                raise RuntimeError("Rendered image is blank")
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(args.screenshot)
        print(args.screenshot)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            camera(viewer.cam)
            viewer.opt.geomgroup[3] = 0
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)


if __name__ == "__main__":
    main()
