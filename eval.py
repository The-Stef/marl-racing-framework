from stable_baselines3 import SAC
from env.SimpleRacingEnv import SimpleRacingEnv

MODEL_LOAD_PATH = "artifacts/models/newenv_sac_tiles_100k_action_repeat_2"

def main():
    env = SimpleRacingEnv(render_mode="human")
    model = SAC.load(
        path=MODEL_LOAD_PATH,
        env=env
    )

    obs, info = env.reset()

    for _ in range(5_000):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()