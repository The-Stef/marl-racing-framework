import json
from pathlib import Path
from numbers import Real

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLoggerCallback(BaseCallback):
    """
    Writes one JSON object per completed episode.

    Output file:
        episode_metrics.jsonl

    Most episodes contain summary metrics only.
    Every Nth episode also contains downsampled per-step series.
    """

    def __init__(
        self,
        log_path,
        save_series_every: int = 20,
        series_stride: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.log_path = Path(log_path)

        # Every 20th episode gets x/y/speed/action lists.
        # Set to 0 or None later if you want no series saved.
        self.save_series_every = save_series_every

        # Store every 5th step inside selected episodes.
        self.series_stride = max(1, int(series_stride))

        self.file = None
        self.episodes = None
        self.global_episode_count = 0

    def _make_empty_episode(self):
        episode_number = self.global_episode_count + 1

        return {
            "episode_number": episode_number,
            "save_series": self._should_save_series_for_episode_number(episode_number),

            # Basic episode counters
            "length": 0,
            "cumulative_reward": 0.0,

            # Final values from info
            "termination_reason": "unknown",
            "lap_progress_radians": None,
            "lap_progress_percent": None,

            # Running stats
            "speed_sum": 0.0,
            "speed_count": 0,
            "speed_max": None,

            "radial_error_sum": 0.0,
            "radial_error_count": 0,
            "radial_error_max": None,

            # Optional per-step lists
            "x_pos": [],
            "y_pos": [],
            "speed": [],
            "throttle": [],
            "brake": [],
            "gas": [],

            # Internal step index inside this episode
            "step_index": 0,
        }

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        n_envs = self.training_env.num_envs
        self.episodes = [self._make_empty_episode() for _ in range(n_envs)]

        self.file = self.log_path.open("w", encoding="utf-8")

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _clean_value(self, value):
        """
        Convert numpy/Python values into JSON-safe values.

        Returns None for things we do not want to save,
        such as arrays, dicts, lists, or objects.
        """

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool):
            return bool(value)

        if isinstance(value, Real):
            return float(value)

        if isinstance(value, str):
            return value

        return None

    def _update_mean_max(self, episode, name, value):
        """
        Track mean and max for a numeric value.

        Example:
            name = "speed"

        This updates:
            speed_sum
            speed_count
            speed_max
        """

        sum_key = f"{name}_sum"
        count_key = f"{name}_count"
        max_key = f"{name}_max"

        episode[sum_key] += value
        episode[count_key] += 1

        if episode[max_key] is None:
            episode[max_key] = value
        else:
            episode[max_key] = max(episode[max_key], value)

    def _should_save_series_for_episode_number(self, episode_number: int) -> bool:
        """
        Decide whether a specific episode number should save per-step series.

        Example:
            save_series_every = 20

        Then episodes 20, 40, 60, ... save series.
        """

        if self.save_series_every is None:
            return False

        if self.save_series_every <= 0:
            return False

        return episode_number % self.save_series_every == 0

    def _maybe_add_series_value(self, episode, key, value):
        """
        Add a value to a per-step list, but only every series_stride steps.

        Example:
            series_stride = 5

        Then we save step 0, 5, 10, 15, ...
        """

        step_index = episode["step_index"]

        if step_index % self.series_stride != 0:
            return

        if key not in episode:
            return

        episode[key].append(value)

    def _write_episode(self, episode):
        """
        Turn the temporary episode tracker into one JSON object
        and write it as one line.
        """

        self.global_episode_count += 1

        speed_mean = None
        if episode["speed_count"] > 0:
            speed_mean = episode["speed_sum"] / episode["speed_count"]

        radial_error_mean = None
        if episode["radial_error_count"] > 0:
            radial_error_mean = episode["radial_error_sum"] / episode["radial_error_count"]

        row = {
            "episode": self.global_episode_count,
            "total_timesteps": int(self.num_timesteps),

            "length": int(episode["length"]),
            "cumulative_reward": float(episode["cumulative_reward"]),
            "termination_reason": episode["termination_reason"],

            "lap_progress_radians": episode["lap_progress_radians"],
            "lap_progress_percent": episode["lap_progress_percent"],

            "speed_mean": speed_mean,
            "speed_max": episode["speed_max"],

            "radial_error_mean": radial_error_mean,
            "radial_error_max": episode["radial_error_max"],

            "series_saved": bool(episode["save_series"]),
        }

        should_save_series = episode["save_series"]

        if should_save_series:
            row["x_pos"] = episode["x_pos"]
            row["y_pos"] = episode["y_pos"]
            row["speed"] = episode["speed"]
            row["throttle"] = episode["throttle"]
            row["brake"] = episode["brake"]
            row["gas"] = episode["gas"]

        self.file.write(json.dumps(row) + "\n")
        self.file.flush()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        This collects step values into the current episode.
        When the episode ends, it writes one JSON row.

        :return: If the callback returns False, training is aborted early.
        """

        # Store latest values from self.locals
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            episode = self.episodes[env_idx]

            episode["length"] += 1

            if env_idx < len(rewards):
                episode["cumulative_reward"] += float(rewards[env_idx])

            should_save_series = episode["save_series"]

            # -------------------------
            # Final/simple info values
            # -------------------------

            if "done_reason" in info:
                value = self._clean_value(info["done_reason"])
                if value is not None:
                    episode["termination_reason"] = value

            if "lap_progress_radians" in info:
                value = self._clean_value(info["lap_progress_radians"])
                if value is not None:
                    episode["lap_progress_radians"] = value

            if "lap_progress_percent" in info:
                value = self._clean_value(info["lap_progress_percent"])
                if value is not None:
                    episode["lap_progress_percent"] = value

            # -------------------------
            # Speed stats
            # -------------------------

            if "speed" in info:
                speed = self._clean_value(info["speed"])
                if speed is not None:
                    self._update_mean_max(episode, "speed", speed)

                    if should_save_series:
                        self._maybe_add_series_value(episode, "speed", speed)

            # -------------------------
            # Radial error stats
            # -------------------------
            # If abs_radial_error is missing, fall back to abs(radial_error).

            radial_error = None

            if "abs_radial_error" in info:
                radial_error = self._clean_value(info["abs_radial_error"])
            elif "radial_error" in info:
                raw_radial_error = self._clean_value(info["radial_error"])
                if raw_radial_error is not None:
                    radial_error = abs(raw_radial_error)

            if radial_error is not None:
                self._update_mean_max(episode, "radial_error", radial_error)

            # -------------------------
            # Optional trajectory/action series
            # -------------------------

            if should_save_series:
                if "x" in info:
                    value = self._clean_value(info["x"])
                    if value is not None:
                        self._maybe_add_series_value(episode, "x_pos", value)

                if "y" in info:
                    value = self._clean_value(info["y"])
                    if value is not None:
                        self._maybe_add_series_value(episode, "y_pos", value)

                if "throttle" in info:
                    value = self._clean_value(info["throttle"])
                    if value is not None:
                        self._maybe_add_series_value(episode, "throttle", value)

                if "brake" in info:
                    value = self._clean_value(info["brake"])
                    if value is not None:
                        self._maybe_add_series_value(episode, "brake", value)

                if "gas" in info:
                    value = self._clean_value(info["gas"])
                    if value is not None:
                        self._maybe_add_series_value(episode, "gas", value)

            episode["step_index"] += 1

            # -------------------------
            # Episode ended
            # -------------------------

            if env_idx < len(dones) and dones[env_idx]:
                self._write_episode(episode)
                self.episodes[env_idx] = self._make_empty_episode()

        return True