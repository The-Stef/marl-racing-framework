import functools

from pettingzoo import ParallelEnv
import Box2D
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


from configs import default as cfg

class MARLRacingEnv(ParallelEnv):
    """Multi Agent version of the SimpleRacingEnv."""

    metadata = {
        "name": "marl_racing_environment_v0",
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None, **kwargs):
        """Initializes the environment."""
        self.render_mode = render_mode

        self.NUM_AGENTS = kwargs.get("num_agents", cfg.NUM_AGENTS)
        self.possible_agents = [
            "car_" + str(r)
            for r in range(self.NUM_AGENTS)
        ]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

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
        self.CARS = {}
        self.MAX_SPEED = cfg.MAX_SPEED
        self.START_DIRECTION = cfg.START_DIRECTION
        # self.CAR_START_POSITION_Y = self.TRACK_CENTER_Y
        # self.CAR_START_POSITION_X = self.TRACK_CENTER_X - self.TRACK_RADIUS

        # Screen (pygame rendering)
        self.SCREEN = None
        self.SCREEN_SIZE = cfg.SCREEN_SIZE
        self.ZOOM = cfg.ZOOM

        # Clock (pygame rendering)
        self.CLOCK = None

        # Hyperparameters
        # self.ALPHA = alpha
        # self.BETA = beta

        # Lap Completion
        self.PREV_THETA = {}  # Accumulates angular progress
        self.LAP_PROGRESS = {}
        self.LAP_COUNT = {}
        self.MAX_LAPS = cfg.MAX_LAPS

        # Extra
        self.ACTION_REPEAT = cfg.ACTION_REPEAT

        # Tiles
        self.NUM_TILES = cfg.NUM_TILES
        self.VISITED_TILES = {}

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Speed, Heading error, Radial error, angular velocity
        return spaces.Box(
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

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Steer [-1,1], Throttle [-1.0, 1.0]
        return spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32
    )