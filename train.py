from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from env.SimpleRacingEnv import SimpleRacingEnv
from datetime import datetime
from pathlib import Path

from callbacks.logger import LoggerCallback
from analysis.plot_training import make_training_plots

SEED = 42
NAME = "the_testt"

def main():
    run_name = datetime.now().strftime(f"{NAME}_%Y%m%d_%H%M%S")

    run_dir = Path("artifacts") / "runs" / run_name
    logs_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    best_dir = results_dir / "best"
    checkpoints_dir = results_dir / "checkpoints"

    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        SimpleRacingEnv(),
        filename=str(logs_dir / "train_monitor.csv")
    )
    eval_env = Monitor(
        SimpleRacingEnv(render_mode="human"),
        filename=str(logs_dir / "eval_monitor.csv")
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        best_model_save_path=str(best_dir),
        log_path=str(logs_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(checkpoints_dir),
        name_prefix="sac_checkpoint",
    )

    logging_callback = LoggerCallback()

    model_logger = configure(
        str(logs_dir),
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
        total_timesteps=20_000,
        callback=[eval_callback, checkpoint_callback, logging_callback],
        reset_num_timesteps=True,
        # progress_bar=True,
    )

    model.save(str(results_dir / "final_model"))
    make_training_plots(run_dir)

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()