from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from SimpleRacingEnv import SimpleRacingEnv
from stable_baselines3 import PPO

env = Monitor(SimpleRacingEnv())
eval_env = Monitor(SimpleRacingEnv(render_mode="human"))

eval_callback = EvalCallback(
    eval_env,
    eval_freq=5000,
    n_eval_episodes=1,
    render=True,
    deterministic=True,
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, callback=eval_callback)

model.save("ppo_basic")
env.close()
eval_env.close()