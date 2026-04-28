from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.SimpleRacingEnv import SimpleRacingEnv

SEED = 42
run_name = datetime.now().strftime("sac_ar2_%Y%m%d_%H%M%S")

def main():
    env = Monitor(SimpleRacingEnv())
    eval_env = Monitor(SimpleRacingEnv(render_mode="human"))

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        best_model_save_path=f"artifacts/models/best/{NAME}",
        log_path=f"artifacts/logs/{NAME}",
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
    )

    model.learn(
        total_timesteps=100_000,
        callback=eval_callback,
        reset_num_timesteps=True,
    )

    model.save(f"artifacts/models/{NAME}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()