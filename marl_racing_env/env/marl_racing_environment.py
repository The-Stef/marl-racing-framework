import functools
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
import Box2D
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from ...env.car_dynamics import Car
from ...env.rendering import render_env
from ...env.rewards import compute_reward
from ...env.track import (
    wrap_angle,
    car_heading,
    compute_radial_error,
    compute_desired_direction,
    tangential_velocity,
    current_tile,
    populate_dictionary_with_info
)
from configs import default as cfg

from .helpers.helpers import current_tile

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

        # TODO: Delete these?
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
        """Reset the environment to a starting point."""
        # Unlike gymnasium's Env, the environment is responsible for setting the random seed explicitly.
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]

        self.STEPS = 0

        # Create the world
        self.WORLD = Box2D.b2World((0, 0))

        # Reset per-agent dictionaries
        self.CARS = {}
        self.PREV_THETA = {}
        self.LAP_PROGRESS = {}
        self.LAP_COUNT = {}
        self.VISITED_TILES = {}

        for i, agent in enumerate(self.agents):
            # Set up each car
            self.CARS[agent] = Car(
                self.WORLD,
                self.START_DIRECTION,
                # TODO figure out positions
                # self.CAR_START_POSITION_X,
                # self.CAR_START_POSITION_Y
            )

            # Set up each prev_theta
            self.PREV_THETA[agent] = np.arctan2(
                self.CARS[agent].hull.position[1]- self.TRACK_CENTER_Y,
                self.CARS[agent].hull.position[0] - self.TRACK_CENTER_X,
            )

            # Set up each agent's lap progress
            self.LAP_PROGRESS[agent] = 0.0

            # Set up each agent's lap count
            self.LAP_COUNT[agent] = 0

            self.VISITED_TILES[agent] = {current_tile(self, agent)}

        # the observations should be numpy arrays even if there is only one value
        observations = {
            # TODO agent: self._get_obs(agent)
            agent: None
            for agent in self.agents
        }
        infos = {
            agent: {}
            for agent in self.agents
        }

        self.state = observations

        return observations, infos

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