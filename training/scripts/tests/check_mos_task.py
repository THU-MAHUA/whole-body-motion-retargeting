"""Configuration-only Isaac test; no physics scene, renderer, or PPO run."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    from booster_train.assets.robots.mos import MOS_CFG, MOS_JOINT_NAMES, MOS_BODY_NAMES
    from booster_train.tasks.manager_based.beyond_mimic.robots.mos.env_cfg import WalkEnvCfg, WalkPlayEnvCfg
    from booster_train.tasks.manager_based.beyond_mimic.robots.mos.ppo_cfg import WalkPPORunnerCfg
    from booster_train.tasks.manager_based.beyond_mimic.robots.k1.locomotion.env_cfg import WalkEnvCfg as K1Walk

    training, playback, k1 = WalkEnvCfg(), WalkPlayEnvCfg(), K1Walk()
    assert gym.spec("MOS-Walk-v0")
    assert gym.spec("MOS-Walk-v0-Play")
    assert WalkPPORunnerCfg().experiment_name == "mos_walk"
    assert Path(MOS_CFG.spawn.asset_path).is_file()
    tree = ET.parse(MOS_CFG.spawn.asset_path)
    assert [j.get("name") for j in tree.findall("./joint")] == MOS_JOINT_NAMES
    assert set(b.get("name") for b in tree.findall("./link")) == set(MOS_BODY_NAMES)
    assert len(MOS_JOINT_NAMES) == 20
    controlled = [n for a in MOS_CFG.actuators.values() for n in a.joint_names_expr]
    assert len(controlled) == 20 and set(controlled) == set(MOS_JOINT_NAMES)
    for cfg in (training, playback):
        assert cfg.commands.motion.body_names[0] == cfg.commands.motion.anchor_body_name == "body"
        assert cfg.commands.motion.tail_len == 0
        assert cfg.decimation * cfg.sim.dt == .02
        assert cfg.events.base_com.params["asset_cfg"].body_names == "body"
        assert cfg.terminations.ee_body_pos.params["body_names"] == ["Lhand", "Rhand", "Lfoot", "Rfoot"]
        assert set(cfg.actions.joint_pos.scale) == set(MOS_JOINT_NAMES)
    assert training.scene.terrain.terrain_type == "generator"
    assert playback.scene.terrain.terrain_type == "plane"
    assert playback.events.push_robot is None and playback.commands.motion.play
    assert k1.commands.motion.anchor_body_name == "Trunk"
    assert k1.events.base_com.params["asset_cfg"].body_names == "Trunk"
    assert "left_foot_link" in k1.terminations.ee_body_pos.params["body_names"]
    print("MOS task configuration checks passed; K1 bindings unchanged.")
finally:
    app.close()
