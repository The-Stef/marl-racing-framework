from marl_racing_env.env.marl_racing_environment import MARLRacingEnv
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = MARLRacingEnv()
    parallel_api_test(env, num_cycles=1_000_000)