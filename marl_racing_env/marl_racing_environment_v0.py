# marl_racing_environment_v0.py

from .env.marl_racing_environment import MARLRacingEnv


def raw_env(**kwargs):
    return MARLRacingEnv(**kwargs)


def parallel_env(**kwargs):
    return raw_env(**kwargs)