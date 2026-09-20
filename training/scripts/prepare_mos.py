"""Create a separate simulation-only MOS URDF; never alter the source model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from wbmr.paths import ROBOTS


def prepare(source, meshes, output):
    source, meshes, output = Path(source).resolve(), Path(meshes).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Output must not overwrite the original URDF.")
    tree = ET.parse(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_meshes = output.parent / "meshes"
    target_meshes.mkdir(exist_ok=True)
    hashes = {}
    for mesh in tree.findall(".//mesh"):
        name = Path(mesh.attrib["filename"]).stem + ".stl"
        src = meshes / name
        dst = target_meshes / name
        if src != dst:
            shutil.copy2(src, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
        mesh.set("filename", f"meshes/{name}")
    for joint in tree.findall("./joint"):
        if joint.get("type") != "continuous":
            raise ValueError(f"Review unexpected joint type: {joint.attrib}")
        # Wide numerical bounds, not measured mechanical travel.
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.attrib.update(lower=str(-math.pi), upper=str(math.pi), effort="60", velocity="20")
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    report = {
        "source_urdf": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mesh_source": str(meshes), "mesh_sha256": hashes,
        "training_urdf": str(output),
        "status": "PROVISIONAL SIMULATION ONLY; NOT HARDWARE SPECIFICATIONS",
        "joint_limits": "All continuous joints changed to revolute +/-pi, 20 rad/s, 60 Nm URDF ceiling.",
        "actuators": "Actual simulation group torque limits and PD gains are in assets/robots/mos.py.",
        "geometry": "Copied GMR visual meshes; original collision mesh origins retained. PhysX convex hull per link.",
        "self_collision": False,
        "preserved": ["link inertials", "joint origins", "joint axes", "visual/collision origins", "materials"],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROBOTS / "mos/source/urdf/MOS9.2_urdf_0308-4.urdf"))
    parser.add_argument("--meshes", default=str(ROBOTS / "mos/retargeting/meshes"))
    parser.add_argument("--output", default=str(ROBOTS / "mos/training/mos_training.urdf"))
    args = parser.parse_args()
    print(prepare(args.source, args.meshes, args.output))
