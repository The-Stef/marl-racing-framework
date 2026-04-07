from stable_baselines3 import PPO
from SimpleRacingEnv import SimpleRacingEnv
import numpy as np

# Variables for easy value switching
MODEL_DISPLAY_NAME = "SPARSE MODEL"
SPARSE_PATH = "artifacts/models/simplecircle_ppo_sparse_500k_run1"
DENSE_PATH = "artifacts/models/simplecircle_ppo_dense_100k_run1"
PATH = DENSE_PATH
RENDER_MODE = "human"
N_EPISODES = 20
ALPHA = 1.0
BETA = 0.0

env = SimpleRacingEnv(render_mode=RENDER_MODE, alpha=ALPHA, beta=BETA)
model = PPO.load(PATH, env=env)

rewards = []
lengths = []
lap_completions = 0
off_track = 0
timeouts = 0

for episode in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    ep_len = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        ep_len += 1
        done = terminated or truncated

    rewards.append(ep_reward)
    lengths.append(ep_len)

    if env.finished_lap:
        lap_completions += 1
    elif terminated:
        off_track += 1
    elif truncated:
        timeouts += 1

    print(f"Episode {episode+1}: reward={ep_reward:.2f}, length={ep_len}, finished_lap={env.finished_lap}")

env.close()

print("\nEvaluation Summary")
print(f"Model name: {MODEL_DISPLAY_NAME}")
print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Mean length: {np.mean(lengths):.2f}")
print(f"Lap completions: {lap_completions}/{N_EPISODES}")
print(f"Off-track terminations: {off_track}/{N_EPISODES}")
print(f"Timeouts: {timeouts}/{N_EPISODES}")