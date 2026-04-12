import matplotlib.pyplot as plt
from gymnasium import spaces
import gymnasium as gym
import numpy as np


class SimpleRacingEnv(gym.Env):
    """Custom racing environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def _get_observation(self):
        return np.array(
            [self.x, self.y, self.velocity, self.direction, self.radial_error],
            dtype=np.float32
        )

    def _compute_radial_error(self):
        distance_from_center = np.sqrt(
            (self.track_center_x - self.x) ** 2 +
            (self.track_center_y - self.y) ** 2
        )
        return distance_from_center - self.track_radius

    def _apply_action(self, action):
        if action == 1:  # accelerate
            self.velocity += self.acceleration_amount
        elif action == 2:  # brake
            self.velocity -= self.brake_amount
        elif action == 3:  # turn left
            self.direction += self.turn_amount
        elif action == 4:  # turn right
            self.direction -= self.turn_amount

        self.velocity = np.clip(self.velocity, 0.0, self.max_velocity)

    def _wrap_direction(self):
        if self.direction > np.pi:
            self.direction -= 2 * np.pi
        elif self.direction < -np.pi:
            self.direction += 2 * np.pi

    def _move_agent(self):
        self.x += self.velocity * np.cos(self.direction) * self.dt
        self.y += self.velocity * np.sin(self.direction) * self.dt

    def _compute_reward(self):
        dense_reward = self.velocity - 0.5 * abs(self.radial_error)
        sparse_reward = 100.0 if self.finished_lap else 0.0
        reward = self.alpha * dense_reward + self.beta * sparse_reward

        if abs(self.radial_error) > self.track_half_width:
            reward -= 5.0

        return reward

    def _check_lap_completion(self):
        near_start_x = abs(self.x - self.start_x) < 1.0
        crossed_y = self.prev_y < self.finish_line_y <= self.y
        return self.steps > 50 and near_start_x and crossed_y

    def __init__(self, render_mode=None, alpha=1.0, beta=0.0):
        super().__init__()
        self.render_mode = render_mode

        """
        0 - do nothing
        1 - accelerate
        2 - brake
        3 - turn left
        4 - turn right
        """
        self.action_space = spaces.Discrete(5)

        # Observable state
        self.x = None
        self.y = None
        self.velocity = None
        self.direction = None
        self.radial_error = None

        self.steps = None

        self.fig = None
        self.ax = None

        self.prev_x = None
        self.prev_y = None
        self.finished_lap = False

        self.alpha = alpha
        self.beta = beta

        # Track parameters
        self.track_center_x = 0.0
        self.track_center_y = 0.0
        self.track_radius = 10.0
        self.track_half_width = 2.0

        # Physics parameters
        self.acceleration_amount = 0.2
        self.brake_amount = 0.2
        self.turn_amount = np.pi / 18  # 10 degrees
        self.max_velocity = 5.0
        self.dt = 0.2
        self.max_steps = 500

        # Start / finish parameters
        self.start_x = -10.0
        self.start_y = 0.0
        self.start_direction = np.pi / 2

        self.finish_line_x_min = -12.0
        self.finish_line_x_max = -8.0
        self.finish_line_y = 0.0

        self.observation_space = spaces.Box(
            low=np.array([-20, -20, 0, -np.pi, -10], dtype=np.float32),
            high=np.array([20, 20, 5, np.pi, 10], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )

    def step(self, action):
        self.steps += 1

        self._apply_action(action)
        self._wrap_direction()

        self.prev_x = self.x
        self.prev_y = self.y

        self._move_agent()

        self.finished_lap = self._check_lap_completion()
        self.radial_error = self._compute_radial_error()

        observation = self._get_observation()
        reward = self._compute_reward()

        terminated = False
        truncated = False

        if abs(self.radial_error) > self.track_half_width:
            terminated = True

        if self.finished_lap:
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        base_x = self.start_x
        base_y = self.start_y
        base_direction = self.start_direction

        self.x = base_x + self.np_random.uniform(-0.3, 0.3)
        self.y = base_y + self.np_random.uniform(-0.3, 0.3)
        self.velocity = 0.0
        self.direction = base_direction + self.np_random.uniform(-0.3, 0.3)

        self.steps = 0
        self.prev_x = self.x
        self.prev_y = self.y
        self.finished_lap = False

        self.radial_error = self._compute_radial_error()

        return self._get_observation(), {}

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()

        # Draw finish line
        self.ax.plot(
            [self.finish_line_x_min, self.finish_line_x_max],
            [self.finish_line_y, self.finish_line_y],
            color="red",
            linewidth=2
        )

        # Draw ideal track circle
        outer = plt.Circle(
            (self.track_center_x, self.track_center_y),
            self.track_radius + self.track_half_width,
            fill=False
        )
        inner = plt.Circle(
            (self.track_center_x, self.track_center_y),
            self.track_radius - self.track_half_width,
            fill=False
        )
        ideal = plt.Circle(
            (self.track_center_x, self.track_center_y),
            self.track_radius,
            fill=False,
            linestyle="--",
            color="g",
        )

        self.ax.add_patch(outer)
        self.ax.add_patch(inner)
        self.ax.add_patch(ideal)

        # Draw agent
        self.ax.plot(self.x, self.y, "o")

        # Draw direction arrow
        arrow_length = 1.5
        dx = arrow_length * np.cos(self.direction)
        dy = arrow_length * np.sin(self.direction)
        self.ax.arrow(self.x, self.y, dx, dy, head_width=0.4)

        # Set limits and aspect
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_aspect("equal")

        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None