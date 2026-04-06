from stable_baselines3.common.callbacks import EvalCallback
from SimpleRacingEnv import SimpleRacingEnv
from stable_baselines3 import PPO

env = SimpleRacingEnv()
eval_env = SimpleRacingEnv()

eval_callback = EvalCallback(
    eval_env,
    eval_freq=1000,
    n_eval_episodes=1,
    render=True,
    deterministic=True,
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000   , callback=eval_callback)

model.save("ppo_basic")
env.close()
eval_env.close()