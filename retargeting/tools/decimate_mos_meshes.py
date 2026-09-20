from pathlib import Path

import trimesh


from wbmr.paths import ROBOTS

SOURCE = ROBOTS / "mos/source/meshes"
TARGET = ROBOTS / "mos/retargeting/meshes"
TARGET.mkdir(parents=True, exist_ok=True)


def main():
    for source in sorted(SOURCE.glob("*.STL")):
        mesh = trimesh.load_mesh(source, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        target_faces = min(30000, max(1000, len(mesh.faces)))
        if len(mesh.faces) > target_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        mesh.export(TARGET / f"{source.stem}.stl", file_type="stl")
        print(f"{source.name}: {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
