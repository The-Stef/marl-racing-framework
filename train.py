from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from env.SimpleRacingEnv import SimpleRacingEnv
from datetime import datetime

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
        best_model_save_path=f"artifacts/models/best/{run_name}",
        log_path=f"artifacts/logs/{run_name}",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=f"artifacts/models/checkpoints/{run_name}",
        name_prefix="sac_checkpoint",
    )

    model_logger = configure(
        f"artifacts/logs/{run_name}",
        ["stdout", "json"]
    )

    model = SAC(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
    )

    model.set_logger(model_logger)

    model.learn(
        total_timesteps=100_000,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=True,
    )

    model.save(f"artifacts/models/{run_name}_final")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()