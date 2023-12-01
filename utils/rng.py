from secrets import randbits


def env_seed(cfg):
    if cfg.env.auto_generate_seed:
        return randbits(32)
    else:
        return cfg.env.seed

