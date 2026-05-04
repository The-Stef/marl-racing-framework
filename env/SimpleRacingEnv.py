import Box2D
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .car_dynamics import Car
from .rendering import render_env
from .rewards import compute_reward
from .track import (
    wrap_angle,
    car_heading,
    compute_radial_error,
    compute_desired_direction,
    tangential_velocity,
    current_tile,
    populate_dictionary_with_info
)
from configs import default as cfg

class SimpleRacingEnv(gym.Env):
    """Custom racing environment that follows gym interface. (TAKE TWO)"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, alpha=1.0, beta=0.0):
        """Initializes the environment."""
        super().__init__()
        self.render_mode = render_mode

        # World
        self.WORLD = None

        # Time
        self.PHYSICS_FPS = cfg.PHYSICS_FPS
        self.DT = 1.0 / self.PHYSICS_FPS

        # Steps
        self.STEPS = 0
        self.MAX_STEPS = cfg.MAX_STEPS

        # Track
        self.TRACK_RADIUS = cfg.TRACK_RADIUS
        self.TRACK_CENTER_X = cfg.TRACK_CENTER_X
        self.TRACK_CENTER_Y = cfg.TRACK_CENTER_Y
        self.TRACK_HALF_WIDTH = cfg.TRACK_HALF_WIDTH

        # Car
        self.CAR = None
        self.MAX_SPEED = cfg.MAX_SPEED
        self.START_DIRECTION = cfg.START_DIRECTION
        self.CAR_START_POSITION_Y = self.TRACK_CENTER_Y
        self.CAR_START_POSITION_X = self.TRACK_CENTER_X - self.TRACK_RADIUS

        # Screen (pygame rendering)
        self.SCREEN = None
        self.SCREEN_SIZE = cfg.SCREEN_SIZE
        self.ZOOM = cfg.ZOOM

        # Clock (pygame rendering)
        self.CLOCK = None

        # Hyperparameters
        self.ALPHA = alpha
        self.BETA = beta

        # Lap Completion
        self.PREV_THETA = None # Accumulates angular progress
        self.LAP_PROGRESS = 0.0
        self.LAP_COUNT = 0
        self.MAX_LAPS = cfg.MAX_LAPS

        # Extra
        self.ACTION_REPEAT = cfg.ACTION_REPEAT

        # Tiles
        self.NUM_TILES = cfg.NUM_TILES
        self.VISITED_TILES = None

        # Steer [-1,1], Throttle [-1.0, 1.0]
        # Initially, it was Steer [-1,1], Gas [0,1] & Brake [0,1] but
        # PPO can put out both gas & brake so that led to issues.
        # So throttle > 0 -> gas, throttle < 0 -> brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Speed, Heading error, Radial error, angular velocity
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, -np.pi, -self.TRACK_HALF_WIDTH, -20.0],
                dtype=np.float32
            ),
            high=np.array(
                [self.MAX_SPEED, np.pi, self.TRACK_HALF_WIDTH, 20.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

    def step(self, action):
        steer = float(np.tanh(action[0]))
        throttle = float(np.tanh(action[1]))

        gas = max(throttle, 0.0)
        brake = max(-throttle, 0.0)

        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.ACTION_REPEAT):
            self.STEPS += 1

            self.CAR.steer(steer)
            self.CAR.gas(gas)
            self.CAR.brake(brake)

            self.CAR.step(self.DT)
            self.WORLD.Step(self.DT, 6, 2)

            theta = np.arctan2(
                self.CAR.hull.position[1] - self.TRACK_CENTER_Y,
                self.CAR.hull.position[0] - self.TRACK_CENTER_X
            )

            dtheta = theta - self.PREV_THETA
            if dtheta > np.pi:
                dtheta -= 2 * np.pi
            elif dtheta < -np.pi:
                dtheta += 2 * np.pi

            self.LAP_PROGRESS += dtheta
            self.PREV_THETA = theta

            # Reward is now handled by rewards.py
            total_reward += compute_reward(self)

            done_reason = "not_done"

            # If the car crashes by going off-track
            if abs(self._compute_radial_error()) > self.TRACK_HALF_WIDTH:
                terminated = True
                done_reason = "car_crash"
                break

            # If the car performs a full lap
            if self.LAP_PROGRESS <= -2 * np.pi:
                terminated = True
                done_reason = "lap_completed"
                break

            # If the episode is taking too long
            if self.STEPS >= self.MAX_STEPS:
                truncated = True
                done_reason = "timeout"
                break

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        info = self._populate_dictionary_with_info(
            steer=steer,
            throttle=throttle,
            gas=gas,
            brake=brake,
            reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            done_reason=done_reason,
        )

        return observation, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.STEPS = 0

        if self.CAR is not None:
            self.CAR.destroy()
            self.CAR = None

        # Create the world
        self.WORLD = Box2D.b2World((0, 0))

        # Create the car
        self.CAR = Car(
            self.WORLD,
            self.START_DIRECTION,
            self.CAR_START_POSITION_X,
            self.CAR_START_POSITION_Y
        )

        # Set progress around track to 0
        self.PREV_THETA = np.arctan2(
            self.CAR.hull.position[1] - self.TRACK_CENTER_Y,
            self.CAR.hull.position[0] - self.TRACK_CENTER_X
        )
        self.LAP_PROGRESS = 0.0

        # Create the tiles, with the starting tile already visited
        self.VISITED_TILES = {self._current_tile()}

        # Compute first observation
        observation = self._get_obs()

        info = {}
        return observation, info

    def render(self):
        render_env(self)

    def close(self):
        """Closes everything."""
        if self.CAR is not None:
            self.CAR.destroy()
            self.CAR = None

        if self.SCREEN is not None:
            pygame.quit()
            self.SCREEN = None
            self.CLOCK = None

############################### HELPERS ###################################

    def _get_obs(self):
        """Return the observation. Values computed using info straight from the Car object."""
        velocity_x = float(self.CAR.hull.linearVelocity[0])
        velocity_y = float(self.CAR.hull.linearVelocity[1])
        velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

        current_direction = self._car_heading()
        desired_direction = self._compute_desired_direction()
        heading_error = self._wrap_angle(desired_direction - current_direction)

        radial_error = np.clip(
            self._compute_radial_error(),
            -self.TRACK_HALF_WIDTH,
            self.TRACK_HALF_WIDTH
        )

        angular_velocity = float(self.CAR.hull.angularVelocity)

        return np.array(
            [velocity, heading_error, radial_error, angular_velocity],
            dtype=np.float32
        )

    def _compute_radial_error(self):
        return compute_radial_error(self)

    def _compute_desired_direction(self):
        return compute_desired_direction(self)

    def _tangential_velocity(self):
        return tangential_velocity(self)

    def _wrap_angle(self, angle):
        return wrap_angle(angle)

    def _car_heading(self):
        return car_heading(self)

    def _current_tile(self):
        return current_tile(self)

    def _populate_dictionary_with_info(
            self,
            steer: float,
            throttle: float,
            gas: float,
            brake: float,
            reward: float,
            terminated: bool,
            truncated: bool,
            done_reason: str | None,
    ) -> dict:
        return populate_dictionary_with_info(
            self,
            steer,
            throttle,
            gas,
            brake,
            reward,
            terminated,
            truncated,
            done_reason,
        )