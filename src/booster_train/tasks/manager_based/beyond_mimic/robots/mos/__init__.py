import gymnasium as gym


for suffix, config in (("", "WalkEnvCfg"), ("-Play", "WalkPlayEnvCfg")):
    gym.register(
        id=f"MOS-Walk-v0{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:WalkPPORunnerCfg",
        },
    )
