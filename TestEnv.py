# Checks if environment follows Stable Baseline 3 interface
from stable_baselines3.common.env_checker import check_env
from SimpleRacingEnv import SimpleRacingEnv

env = SimpleRacingEnv()
check_env(env, warn=True)

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()   # random action for testing
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()