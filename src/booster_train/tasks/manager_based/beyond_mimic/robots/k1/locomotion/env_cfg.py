from pathlib import Path

from isaaclab.utils import configclass

from ..mj_dance_002.env_cfg import FlatWoStateEstimationEnvCfg, RoughWoStateEstimationEnvCfg


def _set_motion(cfg, name):
    from wbmr.paths import MOTIONS
    cfg.commands.motion.motion_file = str(MOTIONS / "K1" / f"{name}.npz")
    cfg.commands.motion.anchor_body_name = "Trunk"
    cfg.commands.motion.tail_len = 0


@configclass
class WalkEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_motion(self, "walk")


@configclass
class JogEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_motion(self, "jog")


@configclass
class RunEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_motion(self, "run")


@configclass
class WalkPlayEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_motion(self, "walk")
        self.commands.motion.play = True
        self.events.push_robot = None


@configclass
class JogPlayEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_motion(self, "jog")
        self.commands.motion.play = True
        self.events.push_robot = None


@configclass
class RunPlayEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_motion(self, "run")
        self.commands.motion.play = True
        self.events.push_robot = None
