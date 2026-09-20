import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


from wbmr.paths import ROBOTS

URDF = ROBOTS / "mos/source/urdf/MOS9.2_urdf_0308-4.urdf"
OUT = ROBOTS / "mos/retargeting/MOS9.2.xml"


def attrs(node, tag):
    child = node.find(tag)
    return {} if child is None else child.attrib


def vec(value, default="0 0 0"):
    return value or default


def rpy_rotation(origin):
    # URDF uses fixed-axis roll, pitch, yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return Rotation.from_euler("xyz", np.fromstring(vec(origin.get("rpy")), sep=" "))


def numbers(values):
    return " ".join(f"{value:.17g}" for value in values)


def pose(origin):
    return {
        "pos": vec(origin.get("xyz")),
        "quat": numbers(rpy_rotation(origin).as_quat(scalar_first=True)),
    }


def main():
    root = ET.parse(URDF).getroot()
    links = {x.attrib["name"]: x for x in root.findall("link")}
    joints = root.findall("joint")
    children = {j.find("child").attrib["link"]: j for j in joints}
    root_link = next(name for name in links if name not in children)

    mj = ET.Element("mujoco", model="MOS9.2")
    ET.SubElement(mj, "compiler", meshdir="meshes", angle="radian", autolimits="true")
    asset = ET.SubElement(mj, "asset")
    # Match K1's preview appearance without changing MOS contact parameters.
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="512")
    ET.SubElement(asset, "texture", name="texplane", type="2d", builtin="checker",
                  rgb1=".2 .3 .4", rgb2=".1 0.15 0.2", width="512", height="512",
                  mark="cross", markrgb=".8 .8 .8")
    ET.SubElement(asset, "material", name="matplane", reflectance="0.3",
                  texture="texplane", texrepeat="1 1", texuniform="true")
    for name, link in links.items():
        visual_mesh = attrs(link, "visual/geometry/mesh")
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl",
                      scale=visual_mesh.get("scale", "1 1 1"))
    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", directional="true", diffuse=".4 .4 .4",
                  specular="0.1 0.1 0.1", pos="0 0 5.0", dir="0 0 -1", castshadow="false")
    ET.SubElement(world, "light", directional="true", diffuse=".6 .6 .6",
                  specular="0.2 0.2 0.2", pos="0 0 4", dir="0 0 -1")
    ET.SubElement(world, "geom", name="ground", type="plane", pos="0 0 0",
                  size="0 0 1", material="matplane", contype="1", conaffinity="1")

    def add_body(parent, link_name):
        link = links[link_name]
        if link_name == root_link:
            body = ET.SubElement(parent, "body", name=link_name)
            ET.SubElement(body, "joint", name="world_joint", type="free", limited="false")
        else:
            joint = children[link_name]
            origin = attrs(joint, "origin")
            body = ET.SubElement(
                parent, "body", name=link_name, **pose(origin),
            )
            ET.SubElement(
                body, "joint", name=joint.attrib["name"], type="hinge",
                axis=vec(attrs(joint, "axis").get("xyz")), limited="false",
            )

        inertial = link.find("inertial")
        if inertial is not None:
            io = attrs(inertial, "origin")
            mass = attrs(inertial, "mass").get("value", "1")
            inertia = attrs(inertial, "inertia")
            tensor = np.array([
                [float(inertia["ixx"]), float(inertia["ixy"]), float(inertia["ixz"])],
                [float(inertia["ixy"]), float(inertia["iyy"]), float(inertia["iyz"])],
                [float(inertia["ixz"]), float(inertia["iyz"]), float(inertia["izz"])],
            ])
            rotation = rpy_rotation(io).as_matrix()
            tensor = rotation @ tensor @ rotation.T
            ET.SubElement(
                body, "inertial", pos=vec(io.get("xyz")),
                mass=mass,
                fullinertia=numbers(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
            )
        visual = link.find("visual")
        ET.SubElement(body, "geom", name=f"{link_name}_visual",
                      type="mesh", mesh=link_name, contype="0",
                      conaffinity="0", group="1",
                      rgba=attrs(visual, "material/color").get("rgba", "0.75 0.78 0.85 1"),
                      **pose(attrs(visual, "origin")))
        # Legacy collision proxies are hidden by default (group 3). They are not
        # validated collision geometry; GMR only uses kinematic body frames.
        if link_name in ("Rfoot", "Lfoot"):
            ET.SubElement(body, "geom", type="box", size="0.11 0.055 0.025",
                          pos="0 -0.035 -0.015", contype="1", conaffinity="1",
                          group="3", rgba="0.75 0.78 0.85 1")
        elif link_name in ("Rleg2", "Lleg2"):
            ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 -0.13 0",
                          size="0.045", contype="1", conaffinity="1", group="3",
                          rgba="0.75 0.78 0.85 1")
        elif link_name in ("Rleg1", "Lleg1", "Rarm", "Larm"):
            ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 -0.11 0",
                          size="0.05", contype="1", conaffinity="1", group="3",
                          rgba="0.75 0.78 0.85 1")
        else:
            ET.SubElement(body, "geom", type="sphere", size="0.06",
                          contype="1", conaffinity="1", group="3",
                          rgba="0.75 0.78 0.85 1")
        for joint in joints:
            if joint.find("parent").attrib["link"] == link_name:
                add_body(body, joint.find("child").attrib["link"])

    add_body(world, root_link)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(mj, space="  ")
    ET.ElementTree(mj).write(OUT, encoding="utf-8", xml_declaration=True)
    print(OUT)


if __name__ == "__main__":
    main()
