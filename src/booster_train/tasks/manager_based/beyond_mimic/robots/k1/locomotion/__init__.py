import gymnasium as gym


for motion in ("Walk", "Jog", "Run"):
    for suffix, config in (("", f"{motion}EnvCfg"), ("-Play", f"{motion}PlayEnvCfg")):
        gym.register(
            id=f"Booster-K1-{motion}-v0{suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
                "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{motion}PPORunnerCfg",
            },
        )
