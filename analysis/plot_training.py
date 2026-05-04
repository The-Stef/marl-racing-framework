from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Save plots without opening windows

import matplotlib.pyplot as plt


def make_training_plots(run_dir: str | Path) -> None:
    """
    Create plots for one training run using the new per-episode logger.

    Expected run structure:
        run_dir/
        ├── logs/
        │   └── episode_metrics.jsonl
        ├── results/
        └── plots/

    Usage:
        make_training_plots("artifacts/runs/the_testt_20260503_155000")
    """

    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    episode_df = load_episode_metrics_jsonl(logs_dir / "episode_metrics.jsonl")

    if episode_df is None:
        print("No episode metrics found. No plots created.")
        return

    # Per-episode summary plots
    plot_episode_reward(episode_df, plots_dir)
    plot_episode_length(episode_df, plots_dir)
    plot_lap_progress(episode_df, plots_dir)
    plot_speed(episode_df, plots_dir)
    plot_radial_error(episode_df, plots_dir)
    plot_termination_reasons(episode_df, plots_dir)

    # Plots using saved series data
    plot_saved_trajectory(episode_df, plots_dir, mode="progress")
    plot_saved_trajectory(episode_df, plots_dir, mode="reward")
    plot_saved_trajectory(episode_df, plots_dir, mode="length")

    plot_saved_episode_speed_trace(episode_df, plots_dir, mode="progress")
    plot_saved_episode_action_trace(episode_df, plots_dir, mode="progress")

    write_summary(
        run_dir=run_dir,
        plots_dir=plots_dir,
        episode_df=episode_df,
    )

    print(f"Saved plots to: {plots_dir}")


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------


def load_episode_metrics_jsonl(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Skipping episode_metrics.jsonl: not found at {path}")
        return None

    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        print(f"Skipping {path}: could not read JSONL")
        return None

    if df.empty:
        print(f"Skipping {path}: no rows")
        return None

    df = df.reset_index(drop=True)

    if "episode" not in df.columns:
        df["episode"] = np.arange(1, len(df) + 1)

    print(f"Loaded episode metrics: {path}")
    return df


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def is_non_empty_list(value) -> bool:
    return isinstance(value, list) and len(value) > 0


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_saved_series_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows that actually contain saved x/y trajectory lists.
    """

    if "x_pos" not in df.columns or "y_pos" not in df.columns:
        return df.iloc[0:0]

    mask = (
        df["x_pos"].apply(is_non_empty_list)
        & df["y_pos"].apply(is_non_empty_list)
    )

    return df[mask].copy()


def choose_saved_episode(df: pd.DataFrame, mode: str = "progress") -> Optional[pd.Series]:
    """
    Pick one episode that has saved series data.

    mode options:
        progress: highest lap_progress_percent
        reward: highest cumulative_reward
        length: longest episode
        latest: latest saved episode
    """

    saved = get_saved_series_rows(df)

    if saved.empty:
        print("No saved series episodes found")
        return None

    if mode == "progress":
        if "lap_progress_percent" not in saved.columns:
            return saved.iloc[-1]
        idx = safe_numeric(saved["lap_progress_percent"]).idxmax()
        return saved.loc[idx]

    if mode == "reward":
        if "cumulative_reward" not in saved.columns:
            return saved.iloc[-1]
        idx = safe_numeric(saved["cumulative_reward"]).idxmax()
        return saved.loc[idx]

    if mode == "length":
        if "length" not in saved.columns:
            return saved.iloc[-1]
        idx = safe_numeric(saved["length"]).idxmax()
        return saved.loc[idx]

    if mode == "latest":
        return saved.iloc[-1]

    raise ValueError(f"Unknown mode: {mode}")


def save_metric_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Iterable[str],
    title: str,
    ylabel: str,
    path: Path,
    rolling_window: int | None = 10,
) -> None:
    y_cols = [col for col in y_cols if col in df.columns]

    if not y_cols:
        print(f"Skipping {path.name}: no matching columns")
        return

    if x_col not in df.columns:
        print(f"Skipping {path.name}: missing x column {x_col}")
        return

    plt.figure(figsize=(10, 5))

    for col in y_cols:
        values = safe_numeric(df[col])

        plt.plot(df[x_col], values, alpha=0.35, label=f"{col} raw")

        if rolling_window is not None and rolling_window > 1:
            rolling = values.rolling(rolling_window, min_periods=1).mean()
            plt.plot(df[x_col], rolling, linewidth=2, label=f"{col} rolling")

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Per-episode plots
# ---------------------------------------------------------------------


def plot_episode_reward(df: pd.DataFrame, plots_dir: Path) -> None:
    save_metric_plot(
        df=df,
        x_col="episode",
        y_cols=["cumulative_reward"],
        title="Episode Cumulative Reward",
        ylabel="Cumulative reward",
        path=plots_dir / "01_episode_cumulative_reward.png",
        rolling_window=10,
    )


def plot_episode_length(df: pd.DataFrame, plots_dir: Path) -> None:
    save_metric_plot(
        df=df,
        x_col="episode",
        y_cols=["length"],
        title="Episode Length",
        ylabel="Steps",
        path=plots_dir / "02_episode_length.png",
        rolling_window=10,
    )


def plot_lap_progress(df: pd.DataFrame, plots_dir: Path) -> None:
    save_metric_plot(
        df=df,
        x_col="episode",
        y_cols=["lap_progress_percent"],
        title="Lap Progress Per Episode",
        ylabel="Lap progress (%)",
        path=plots_dir / "03_lap_progress_percent.png",
        rolling_window=10,
    )


def plot_speed(df: pd.DataFrame, plots_dir: Path) -> None:
    save_metric_plot(
        df=df,
        x_col="episode",
        y_cols=["speed_mean", "speed_max"],
        title="Episode Speed",
        ylabel="Speed",
        path=plots_dir / "04_speed_mean_max.png",
        rolling_window=10,
    )


def plot_radial_error(df: pd.DataFrame, plots_dir: Path) -> None:
    save_metric_plot(
        df=df,
        x_col="episode",
        y_cols=["radial_error_mean", "radial_error_max"],
        title="Episode Radial Error",
        ylabel="Distance from centerline",
        path=plots_dir / "05_radial_error_mean_max.png",
        rolling_window=10,
    )


def plot_termination_reasons(df: pd.DataFrame, plots_dir: Path) -> None:
    col = "termination_reason"

    if col not in df.columns:
        print("Skipping termination reason plot: termination_reason not found")
        return

    reasons = df[col].dropna()

    if reasons.empty:
        print("Skipping termination reason plot: no termination reasons")
        return

    counts = reasons.value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Episode Termination Reasons")
    plt.xlabel("Termination reason")
    plt.ylabel("Episode count")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "06_termination_reasons.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Saved-series plots
# ---------------------------------------------------------------------


def plot_saved_trajectory(
    df: pd.DataFrame,
    plots_dir: Path,
    mode: str = "progress",
) -> None:
    row = choose_saved_episode(df, mode=mode)

    if row is None:
        print(f"Skipping trajectory by {mode}: no saved series")
        return

    x = row.get("x_pos", [])
    y = row.get("y_pos", [])

    if not is_non_empty_list(x) or not is_non_empty_list(y):
        print(f"Skipping trajectory by {mode}: empty x/y")
        return

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    episode = row.get("episode", "unknown")
    reward = row.get("cumulative_reward", None)
    length = row.get("length", None)
    progress = row.get("lap_progress_percent", None)
    reason = row.get("termination_reason", "unknown")

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1, label="trajectory")
    plt.scatter([x[0]], [y[0]], s=60, label="start")
    plt.scatter([x[-1]], [y[-1]], s=60, label="end")

    plt.title(
        f"Saved Trajectory Selected by {mode}\n"
        f"episode={episode}, reward={fmt(reward)}, length={length}, "
        f"progress={fmt(progress)}%, reason={reason}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"07_trajectory_by_{mode}.png", dpi=150)
    plt.close()


def plot_saved_episode_speed_trace(
    df: pd.DataFrame,
    plots_dir: Path,
    mode: str = "progress",
) -> None:
    row = choose_saved_episode(df, mode=mode)

    if row is None:
        print(f"Skipping speed trace by {mode}: no saved series")
        return

    speed = row.get("speed", [])

    if not is_non_empty_list(speed):
        print(f"Skipping speed trace by {mode}: empty speed series")
        return

    episode = row.get("episode", "unknown")
    reward = row.get("cumulative_reward", None)
    progress = row.get("lap_progress_percent", None)
    reason = row.get("termination_reason", "unknown")

    steps = np.arange(len(speed))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, speed, label="speed")
    plt.title(
        f"Speed Trace Selected by {mode}\n"
        f"episode={episode}, reward={fmt(reward)}, "
        f"progress={fmt(progress)}%, reason={reason}"
    )
    plt.xlabel("Saved step index")
    plt.ylabel("Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"08_speed_trace_by_{mode}.png", dpi=150)
    plt.close()


def plot_saved_episode_action_trace(
    df: pd.DataFrame,
    plots_dir: Path,
    mode: str = "progress",
) -> None:
    row = choose_saved_episode(df, mode=mode)

    if row is None:
        print(f"Skipping action trace by {mode}: no saved series")
        return

    candidate_cols = ["throttle", "gas", "brake"]

    series = {
        col: row.get(col, [])
        for col in candidate_cols
        if is_non_empty_list(row.get(col, []))
    }

    if not series:
        print(f"Skipping action trace by {mode}: no action series found")
        return

    episode = row.get("episode", "unknown")
    reward = row.get("cumulative_reward", None)
    progress = row.get("lap_progress_percent", None)
    reason = row.get("termination_reason", "unknown")

    plt.figure(figsize=(10, 5))

    for name, values in series.items():
        steps = np.arange(len(values))
        plt.plot(steps, values, label=name)

    plt.title(
        f"Action Trace Selected by {mode}\n"
        f"episode={episode}, reward={fmt(reward)}, "
        f"progress={fmt(progress)}%, reason={reason}"
    )
    plt.xlabel("Saved step index")
    plt.ylabel("Action value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"09_action_trace_by_{mode}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def write_summary(
    run_dir: Path,
    plots_dir: Path,
    episode_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append(f"Run directory: {run_dir}")
    lines.append("")
    lines.append("episode_metrics.jsonl")
    lines.append(f"- episodes: {len(episode_df)}")

    numeric_summary_cols = [
        "cumulative_reward",
        "length",
        "lap_progress_percent",
        "speed_mean",
        "speed_max",
        "radial_error_mean",
        "radial_error_max",
    ]

    for col in numeric_summary_cols:
        if col not in episode_df.columns:
            continue

        series = safe_numeric(episode_df[col]).dropna()

        if series.empty:
            continue

        lines.append(
            f"- {col}: "
            f"last={series.iloc[-1]:.3f}, "
            f"max={series.max():.3f}, "
            f"mean={series.mean():.3f}"
        )

    if "termination_reason" in episode_df.columns:
        lines.append("")
        lines.append("termination reasons:")

        counts = episode_df["termination_reason"].dropna().value_counts()

        for reason, count in counts.items():
            lines.append(f"- {reason}: {count}")

    saved = get_saved_series_rows(episode_df)
    lines.append("")
    lines.append(f"saved series episodes: {len(saved)}")

    if not saved.empty:
        best_progress = choose_saved_episode(episode_df, mode="progress")
        best_reward = choose_saved_episode(episode_df, mode="reward")
        longest = choose_saved_episode(episode_df, mode="length")

        if best_progress is not None:
            lines.append(
                "best saved progress episode: "
                f"episode={best_progress.get('episode')}, "
                f"progress={fmt(best_progress.get('lap_progress_percent'))}%, "
                f"reward={fmt(best_progress.get('cumulative_reward'))}, "
                f"length={best_progress.get('length')}, "
                f"reason={best_progress.get('termination_reason')}"
            )

        if best_reward is not None:
            lines.append(
                "best saved reward episode: "
                f"episode={best_reward.get('episode')}, "
                f"reward={fmt(best_reward.get('cumulative_reward'))}, "
                f"progress={fmt(best_reward.get('lap_progress_percent'))}%, "
                f"length={best_reward.get('length')}, "
                f"reason={best_reward.get('termination_reason')}"
            )

        if longest is not None:
            lines.append(
                "longest saved episode: "
                f"episode={longest.get('episode')}, "
                f"length={longest.get('length')}, "
                f"reward={fmt(longest.get('cumulative_reward'))}, "
                f"progress={fmt(longest.get('lap_progress_percent'))}%, "
                f"reason={longest.get('termination_reason')}"
            )

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