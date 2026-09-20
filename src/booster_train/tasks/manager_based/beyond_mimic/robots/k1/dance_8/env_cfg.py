from pathlib import Path

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.beyond_mimic.robots.k1.mj_dance_002.env_cfg import (
    FlatEnvCfg as MJDanceFlatEnvCfg,
)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.mj_dance_002.env_cfg import (
    FlatWoStateEstimationEnvCfg as MJDanceFlatWoStateEstimationEnvCfg,
)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.mj_dance_002.env_cfg import (
    RoughWoStateEstimationEnvCfg as MJDanceRoughWoStateEstimationEnvCfg,
)


from wbmr.paths import MOTIONS

MOTION_FILE = str(MOTIONS / "K1" / "dance_8_clip1.npz")


def _set_dance_8_motion(cfg) -> None:
    cfg.commands.motion.motion_file = MOTION_FILE
    cfg.commands.motion.anchor_body_name = "Trunk"
    cfg.commands.motion.tail_len = 0


@configclass
class FlatEnvCfg(MJDanceFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_dance_8_motion(self)


@configclass
class FlatWoStateEstimationEnvCfg(MJDanceFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_dance_8_motion(self)


@configclass
class RoughWoStateEstimationEnvCfg(MJDanceRoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_dance_8_motion(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None
