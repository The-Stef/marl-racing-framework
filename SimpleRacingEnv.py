import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt

class SimpleRacingEnv(gym.Env):
    """Custom racing environment that follows gym interface."""

    def __init__(self):
        super().__init__()

        """
        0 - do nothing
        1 - accelerate
        2 - brake
        3 - turn left
        4 - turn right
        """
        self.action_space = spaces.Discrete(5)

        """ Observable actions:
        x - position on x axis
        y - position on y axis
        velocity - speed in given direction
        direction - direction of agent
        radial_error - distance from ideal track
        """
        self.x = None
        self.y = None
        self.velocity = None
        self.direction = None
        self.radial_error = None

        self.steps = None

        self.fig = None
        self.ax = None

        # To avoid magic numbers later down the road, define some hyperparameters
        self.track_center_x = 0.0
        self.track_center_y = 0.0
        self.track_radius = 10.0

        self.acceleration_amount = 0.2
        self.brake_amount = 0.2
        self.turn_amount = np.pi / 18  # 10 degrees
        self.max_velocity = 5.0
        self.dt = 0.2
        self.track_half_width = 2.0
        self.max_steps = 500

        self.observation_space = spaces.Box(
            low=np.array([-20, -20, 0, -np.pi, -10], dtype=np.float32),
            high=np.array([20, 20, 5, np.pi, 10], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )

    def step(self, action):
        self.steps += 1

        if action == 1: # Accelerate
            self.velocity += self.acceleration_amount
        if action == 2: # Brake
            self.velocity -= self.acceleration_amount
        if action == 3: # Turn left
            self.direction += self.turn_amount
        if action == 4: # Turn right
            self.direction -= self.turn_amount

        # Keep velocity in range [0, max_velocity]
        self.velocity = np.clip(self.velocity, 0.0, self.max_velocity)

        # Keep direction in range [-pi, pi]
        if self.direction > np.pi:
            self.direction -= 2 * np.pi
        elif self.direction < -np.pi:
            self.direction += 2 * np.pi

        # Move agent
        self.x += self.velocity * np.cos(self.direction) * self.dt
        self.y += self.velocity * np.sin(self.direction) * self.dt

        # Recompute radial error
        distance_from_center = np.sqrt(
            (self.track_center_x - self.x) ** 2 +
            (self.track_center_y - self.y) ** 2
        )
        self.radial_error = distance_from_center - self.track_radius

        observation = np.array(
            [self.x, self.y, self.velocity, self.direction, self.radial_error],
            dtype=np.float32
        )

        reward = self.velocity - 0.5 * abs(self.radial_error)

        # Termination logic
        terminated = False
        truncated = False

        # No termination for making a lap at the moment
        # Model
        if abs(self.radial_error) > self.track_half_width:
            reward -= 5.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Set the initial state of the environment."""
        super().reset(seed=seed)

        self.x = -10
        self.y = 0
        self.velocity = 0
        self.direction = np.pi / 2

        # Compute radial error
        distance_from_center = np.sqrt((self.track_center_x - self.x) ** 2 + (self.track_center_y - self.y) ** 2)
        self.radial_error = distance_from_center - self.track_radius

        self.steps = 0

        observation = np.array(
            [self.x, self.y, self.velocity, self.direction, self.radial_error],
            dtype=np.float32
        )
        info = {} # Used for supplementary diagnostic information (later for debugging?)

        return observation, info

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()

        # draw ideal track circle
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
            color='g',
        )

        self.ax.add_patch(outer)
        self.ax.add_patch(inner)
        self.ax.add_patch(ideal)

        # draw agent
        self.ax.plot(self.x, self.y, "o")

        # draw direction arrow
        arrow_length = 1.5
        dx = arrow_length * np.cos(self.direction)
        dy = arrow_length * np.sin(self.direction)
        self.ax.arrow(self.x, self.y, dx, dy, head_width=0.4)

        # set limits and aspect
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_aspect("equal")

        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None