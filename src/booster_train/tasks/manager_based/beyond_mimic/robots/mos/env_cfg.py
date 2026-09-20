"""MOS bindings for the existing tracking rewards, terrain, and observations."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from booster_train.assets.robots.mos import MOS_ACTION_SCALE, MOS_BODY_NAMES, MOS_CFG, PROJECT_ROOT
from ..k1.mj_dance_002.env_cfg import FlatWoStateEstimationEnvCfg, RoughWoStateEstimationEnvCfg


def _configure_mos(cfg):
    cfg.scene.robot = MOS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = MOS_ACTION_SCALE.copy()
    cfg.commands.motion.motion_file = str(PROJECT_ROOT / "motions/MOS/walk.npz")
    cfg.commands.motion.anchor_body_name = "body"
    cfg.commands.motion.body_names = MOS_BODY_NAMES.copy()
    cfg.commands.motion.tail_len = 0
    cfg.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="body")
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[r"^(?!Lhand$)(?!Rhand$)(?!Lfoot$)(?!Rfoot$).+$"]
    )
    cfg.terminations.ee_body_pos.params["body_names"] = ["Lhand", "Rhand", "Lfoot", "Rfoot"]


@configclass
class WalkEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_mos(self)


@configclass
class WalkPlayEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_mos(self)
        self.commands.motion.play = True
        self.events.push_robot = None

