from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class WalkPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "mos_walk"
