import importlib.util

from wbmr.paths import ROOT


def test_bundled_artifacts():
    spec = importlib.util.spec_from_file_location("artifact_checks", ROOT / "tools/validate_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate()


def test_assets_are_local():
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT

    for path in ROBOT_XML_DICT.values():
        assert path.is_relative_to(ROOT) and path.is_file()
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nv in (26, 28)
