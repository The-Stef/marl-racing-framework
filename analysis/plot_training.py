from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Allows plots to be saved without opening windows

import matplotlib.pyplot as plt


def make_training_plots(run_dir: str | Path) -> None:
    """
    Create plots for one training run.

    Expected run structure:
        run_dir/
        ├── logs/
        │   ├── progress.json
        │   ├── train.monitor.csv / train_monitor.csv.monitor.csv / etc.
        │   ├── eval.monitor.csv / eval_monitor.csv.monitor.csv / etc.
        │   └── evaluations.npz
        ├── results/
        └── plots/

    Usage:
        make_training_plots("artifacts/runs/the_testt_20260503_155000")
    """

    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    progress_df = load_progress_json(logs_dir / "progress.json")

    train_monitor_df = find_and_load_monitor_csv(logs_dir, preferred_name="train")
    eval_monitor_df = find_and_load_monitor_csv(logs_dir, preferred_name="eval")

    if progress_df is not None:
        plot_training_reward(progress_df, plots_dir)
        plot_episode_length(progress_df, plots_dir)
        plot_eval_reward(progress_df, plots_dir)
        plot_custom_diagnostics(progress_df, plots_dir)
        plot_actions(progress_df, plots_dir)
        plot_losses(progress_df, plots_dir)
        plot_done_reasons(progress_df, plots_dir)
        plot_trajectory(progress_df, plots_dir)

    if train_monitor_df is not None:
        plot_monitor_rewards(train_monitor_df, plots_dir, name="train")
        plot_monitor_lengths(train_monitor_df, plots_dir, name="train")

    if eval_monitor_df is not None:
        plot_monitor_rewards(eval_monitor_df, plots_dir, name="eval")
        plot_monitor_lengths(eval_monitor_df, plots_dir, name="eval")

    write_summary(
        run_dir=run_dir,
        plots_dir=plots_dir,
        progress_df=progress_df,
        train_monitor_df=train_monitor_df,
        eval_monitor_df=eval_monitor_df,
    )

    print(f"Saved plots to: {plots_dir}")


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------


def load_progress_json(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skipping progress.json: not found at {path}")
        return None

    df = pd.read_json(path, lines=True)
    df = df.reset_index(drop=True)

    if "time/total_timesteps" not in df.columns:
        df["plot_step"] = np.arange(len(df))
    else:
        df["plot_step"] = df["time/total_timesteps"]

    return df


def find_and_load_monitor_csv(logs_dir: Path, preferred_name: str) -> pd.DataFrame | None:
    """
    Finds monitor CSVs even if SB3 added a weird suffix like:
        train_monitor.csv.monitor.csv
        train.monitor.csv
        eval_monitor.csv.monitor.csv
    """

    if not logs_dir.exists():
        return None

    candidates = list(logs_dir.glob("*.csv"))

    preferred = [
        path for path in candidates
        if preferred_name.lower() in path.name.lower()
    ]

    if not preferred:
        print(f"Skipping {preferred_name} monitor CSV: not found in {logs_dir}")
        return None

    path = preferred[0]

    try:
        df = pd.read_csv(path, comment="#")
    except pd.errors.EmptyDataError:
        print(f"Skipping {path}: empty CSV")
        return None

    if df.empty:
        print(f"Skipping {path}: no rows")
        return None

    df = df.reset_index(drop=True)
    df["episode"] = np.arange(1, len(df) + 1)

    if "l" in df.columns:
        df["cumulative_timesteps"] = df["l"].cumsum()

    print(f"Loaded {preferred_name} monitor CSV: {path}")
    return df


# ---------------------------------------------------------------------
# Generic plotting helpers
# ---------------------------------------------------------------------


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Iterable[str],
    title: str,
    ylabel: str,
    path: Path,
    rolling_window: int | None = None,
) -> None:
    y_cols = [col for col in y_cols if col in df.columns]

    if not y_cols:
        print(f"Skipping {path.name}: no matching columns")
        return

    plt.figure(figsize=(10, 5))

    for col in y_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        if rolling_window is not None and rolling_window > 1:
            series = series.rolling(rolling_window, min_periods=1).mean()

        plt.plot(df[x_col], series, label=col)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping {path.name}: missing {x_col} or {y_col}")
        return

    plt.figure(figsize=(7, 7))
    plt.scatter(df[x_col], df[y_col], s=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# progress.json plots
# ---------------------------------------------------------------------


def plot_training_reward(df: pd.DataFrame, plots_dir: Path) -> None:
    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["rollout/ep_rew_mean"],
        title="Training Episode Reward Mean",
        ylabel="Mean episode reward",
        path=plots_dir / "01_training_reward_mean.png",
    )


def plot_eval_reward(df: pd.DataFrame, plots_dir: Path) -> None:
    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["eval/mean_reward"],
        title="Evaluation Mean Reward",
        ylabel="Mean evaluation reward",
        path=plots_dir / "02_eval_mean_reward.png",
    )


def plot_episode_length(df: pd.DataFrame, plots_dir: Path) -> None:
    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["rollout/ep_len_mean", "eval/mean_ep_length"],
        title="Episode Length",
        ylabel="Episode length",
        path=plots_dir / "03_episode_length.png",
    )


def plot_custom_diagnostics(df: pd.DataFrame, plots_dir: Path) -> None:
    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["custom/lap_progress_percent"],
        title="Lap Progress Percentage",
        ylabel="Lap progress (%)",
        path=plots_dir / "04_lap_progress_percent.png",
    )

    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["custom/speed"],
        title="Speed",
        ylabel="Speed",
        path=plots_dir / "05_speed.png",
    )

    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["custom/radial_error", "custom/abs_radial_error"],
        title="Radial Error",
        ylabel="Distance from centerline",
        path=plots_dir / "06_radial_error.png",
    )

    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["custom/step_reward"],
        title="Custom Step Reward",
        ylabel="Reward per logged step",
        path=plots_dir / "07_custom_step_reward.png",
    )


def plot_actions(df: pd.DataFrame, plots_dir: Path) -> None:
    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["custom/steer", "custom/throttle"],
        title="Policy Actions: Steering and Throttle",
        ylabel="Action value",
        path=plots_dir / "08_actions_steer_throttle.png",
    )

    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["custom/gas", "custom/brake"],
        title="Gas and Brake",
        ylabel="Action value",
        path=plots_dir / "09_gas_brake.png",
    )


def plot_losses(df: pd.DataFrame, plots_dir: Path) -> None:
    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=[
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef_loss",
        ],
        title="SAC Training Losses",
        ylabel="Loss",
        path=plots_dir / "10_training_losses.png",
    )

    save_line_plot(
        df=df,
        x_col="plot_step",
        y_cols=["train/ent_coef"],
        title="SAC Entropy Coefficient",
        ylabel="Entropy coefficient",
        path=plots_dir / "11_entropy_coefficient.png",
    )


def plot_done_reasons(df: pd.DataFrame, plots_dir: Path) -> None:
    col = "custom/done_reason"

    if col not in df.columns:
        print("Skipping done reason plot: custom/done_reason not found")
        return

    reasons = df[col].dropna()

    if reasons.empty:
        print("Skipping done reason plot: no done reasons")
        return

    counts = reasons.value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Logged Done Reasons")
    plt.xlabel("Done reason")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "12_done_reasons.png", dpi=150)
    plt.close()


def plot_trajectory(df: pd.DataFrame, plots_dir: Path) -> None:
    save_scatter_plot(
        df=df.dropna(subset=["custom/x", "custom/y"]) if "custom/x" in df.columns and "custom/y" in df.columns else df,
        x_col="custom/x",
        y_col="custom/y",
        title="Logged Car Positions",
        xlabel="x",
        ylabel="y",
        path=plots_dir / "13_logged_trajectory_xy.png",
    )


# ---------------------------------------------------------------------
# Monitor CSV plots
# ---------------------------------------------------------------------


def plot_monitor_rewards(df: pd.DataFrame, plots_dir: Path, name: str) -> None:
    if "r" not in df.columns:
        print(f"Skipping {name} monitor rewards: column 'r' missing")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["r"], label="episode reward")

    if len(df) >= 5:
        rolling = df["r"].rolling(10, min_periods=1).mean()
        plt.plot(df["episode"], rolling, label="rolling mean")

    plt.title(f"{name.capitalize()} Monitor Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"14_{name}_monitor_rewards.png", dpi=150)
    plt.close()


def plot_monitor_lengths(df: pd.DataFrame, plots_dir: Path, name: str) -> None:
    if "l" not in df.columns:
        print(f"Skipping {name} monitor lengths: column 'l' missing")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["l"], label="episode length")

    if len(df) >= 5:
        rolling = df["l"].rolling(10, min_periods=1).mean()
        plt.plot(df["episode"], rolling, label="rolling mean")

    plt.title(f"{name.capitalize()} Monitor Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"15_{name}_monitor_lengths.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def write_summary(
    run_dir: Path,
    plots_dir: Path,
    progress_df: pd.DataFrame | None,
    train_monitor_df: pd.DataFrame | None,
    eval_monitor_df: pd.DataFrame | None,
) -> None:
    lines = []
    lines.append(f"Run directory: {run_dir}")
    lines.append("")

    if progress_df is not None:
        lines.append("progress.json")
        lines.append(f"- rows: {len(progress_df)}")

        for col in [
            "time/total_timesteps",
            "rollout/ep_rew_mean",
            "eval/mean_reward",
            "custom/lap_progress_percent",
            "custom/speed",
            "custom/abs_radial_error",
        ]:
            if col in progress_df.columns:
                series = pd.to_numeric(progress_df[col], errors="coerce").dropna()
                if not series.empty:
                    lines.append(
                        f"- {col}: last={series.iloc[-1]:.3f}, "
                        f"max={series.max():.3f}, "
                        f"mean={series.mean():.3f}"
                    )

        lines.append("")

    if train_monitor_df is not None:
        lines.append("train monitor")
        lines.append(f"- episodes: {len(train_monitor_df)}")

        if "r" in train_monitor_df.columns:
            lines.append(f"- final reward: {train_monitor_df['r'].iloc[-1]:.3f}")
            lines.append(f"- max reward: {train_monitor_df['r'].max():.3f}")

        if "l" in train_monitor_df.columns:
            lines.append(f"- final length: {train_monitor_df['l'].iloc[-1]:.3f}")
            lines.append(f"- max length: {train_monitor_df['l'].max():.3f}")

        lines.append("")

    if eval_monitor_df is not None:
        lines.append("eval monitor")
        lines.append(f"- episodes: {len(eval_monitor_df)}")

        if "r" in eval_monitor_df.columns:
            lines.append(f"- mean eval reward: {eval_monitor_df['r'].mean():.3f}")
            lines.append(f"- max eval reward: {eval_monitor_df['r'].max():.3f}")

        if "l" in eval_monitor_df.columns:
            lines.append(f"- mean eval length: {eval_monitor_df['l'].mean():.3f}")

        lines.append("")

    summary_path = plots_dir / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Command-line usage
# ---------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to one run directory, e.g. artifacts/runs/the_testt_20260503_155000",
    )
    args = parser.parse_args()

    make_training_plots(args.run_dir)