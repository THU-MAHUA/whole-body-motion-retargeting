from isaaclab.utils import configclass

from ....agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class WalkPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "k1_walk"


@configclass
class JogPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "k1_jog"


@configclass
class RunPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "k1_run"
