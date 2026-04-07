from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from SimpleRacingEnv import SimpleRacingEnv
from stable_baselines3 import PPO

# Alpha - dense | Beta - sparse
env = Monitor(SimpleRacingEnv(alpha=0.0, beta=1.0))
eval_env = Monitor(SimpleRacingEnv(render_mode="human", alpha=0.0, beta=1.0))

eval_callback = EvalCallback(
    eval_env,
    eval_freq=20000,
    n_eval_episodes=1,
    render=True,
    deterministic=True,
)

model = PPO.load(
    "artifacts/models/simplecircle_ppo_sparse9-dense1_100k_run1",
    env,
    verbose=1,
)
model.learn(
    total_timesteps=100000,
    callback=eval_callback,
    reset_num_timesteps=False,
)

model.save("artifacts/models/simplecircle_ppo_sparse_100k_run1")
env.close()
eval_env.close()