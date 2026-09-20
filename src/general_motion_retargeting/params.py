from pathlib import Path

from wbmr.paths import ROBOTS

IK_CONFIG_ROOT = Path(__file__).parent / "ik_configs"
ASSET_ROOT = ROBOTS
ROBOT_XML_DICT = {
    "booster_k1": ROBOTS / "k1/retargeting/K1_serial.xml",
    "mos": ROBOTS / "mos/retargeting/MOS9.2.xml",
}
IK_CONFIG_DICT = {
    "smplx": {
        "booster_k1": IK_CONFIG_ROOT / "smplx_to_k1.json",
        "mos": IK_CONFIG_ROOT / "smplx_to_mos.json",
    },
}
ROBOT_BASE_DICT = {"booster_k1": "Trunk", "mos": "body"}
VIEWER_CAM_DISTANCE_DICT = {"booster_k1": 2.0, "mos": 2.0}
