import functools
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
import Box2D
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

# from ...env.car_dynamics import Car
# from ...env.rendering import render_env
# from ...env.rewards import compute_reward

from ..car_dynamics import Car

# from ...env.track import (
#     wrap_angle,
#     car_heading,
#     compute_radial_error,
#     compute_desired_direction,
#     tangential_velocity,
#     current_tile,
#     populate_dictionary_with_info
# )
from configs import default as cfg

from .helpers.helpers import current_tile, get_obs, compute_car_start_position, render_env, compute_radial_error, compute_reward

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

        # Cars
        self.CARS = {}
        self.MAX_SPEED = cfg.MAX_SPEED
        self.START_DIRECTION = cfg.START_DIRECTION

        # Screen (pygame rendering)
        self.SCREEN = None
        self.SCREEN_SIZE = cfg.SCREEN_SIZE
        self.ZOOM = cfg.ZOOM

        # Clock (pygame rendering)
        self.CLOCK = None

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
            car_start_position_x, car_start_position_y = compute_car_start_position(self, agent, i)

            # Set up each car
            self.CARS[agent] = Car(
                self.WORLD,
                self.START_DIRECTION,
                car_start_position_x,
                car_start_position_y
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
            agent: get_obs(self, agent)
            for agent in self.agents
        }
        infos = {
            agent: {}
            for agent in self.agents
        }

        self.state = observations

        return observations, infos

    def step(self, actions):
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        live_agents = self.agents[:]

        # Rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: 0.0 for agent in live_agents}

        # Same for the observations
        observations = {}

        terminations = {agent: False for agent in live_agents}
        truncations = {agent: False for agent in live_agents}
        done_reasons = {agent: "not_done" for agent in live_agents}

        next_lap_target = {}

        for _ in range(self.ACTION_REPEAT):
            self.STEPS += 1

            for agent in live_agents:
                if terminations[agent] or truncations[agent]:
                    continue

                steer = float(np.tanh(actions[agent][0]))
                throttle = float(np.tanh(actions[agent][1]))

                gas = max(throttle, 0.0)
                brake = max(-throttle, 0.0)

                self.CARS[agent].steer(steer)
                self.CARS[agent].gas(gas)
                self.CARS[agent].brake(brake)

                self.CARS[agent].step(self.DT)
                self.WORLD.Step(self.DT, 6, 2) #TODO I'm guessing this one goes OOTL?

                theta = np.arctan2(
                    self.CARS[agent].hull.position[1] - self.TRACK_CENTER_Y,
                    self.CARS[agent].hull.position[0] - self.TRACK_CENTER_X
                )

                dtheta = theta - self.PREV_THETA[agent]
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                self.LAP_PROGRESS[agent] += dtheta
                self.PREV_THETA[agent] = theta

                # Reward is now handled by helpers.py
                rewards[agent] += compute_reward(self, agent)

                # If the car crashes by going off-track
                if abs(compute_radial_error(self, agent)) > self.TRACK_HALF_WIDTH:
                    terminations[agent] = True
                    done_reasons[agent] = "car_crash"
                    break

                # If the car completes another lap
                next_lap_target[agent] = -2 * np.pi * (self.LAP_COUNT[agent] + 1)

                if self.LAP_PROGRESS[agent] <= next_lap_target[agent]:
                    self.LAP_COUNT[agent] += 1

                    # Reset tile rewards for the new lap
                    self.VISITED_TILES[agent] = {current_tile(self, agent)}

                    #TODO - does this fixed lap # logic stay the same for multi-agent settings?
                    # # Fixed-lap mode, e.g. MAX_LAPS = 1
                    # if self.MAX_LAPS is not None and self.LAP_COUNT >= self.MAX_LAPS:
                    #     terminated = True
                    #     done_reason = "max_laps_reached"
                    #     break

                    # Endurance mode: lap completed, but episode continues
                    done_reasons[agent] = "not_done"

                # If the episode is taking too long
                if self.STEPS >= self.MAX_STEPS:
                    for agent in live_agents:
                        if not terminations[agent]:
                            truncations[agent] = True
                            done_reasons[agent] = "timeout"
                    break

                if all(terminations[a] or truncations[a] for a in live_agents):
                    break

                #TODO - reintroduce info dictionary
                # info = self._populate_dictionary_with_info(
                #     steer=steer,
                #     throttle=throttle,
                #     gas=gas,
                #     brake=brake,
                #     reward=total_reward,
                #     terminated=terminated,
                #     truncated=truncated,
                #     done_reason=done_reason,
                #     lap_count=self.LAP_COUNT,
                # )

        observations = {
            agent: get_obs(self, agent)
            for agent in live_agents
        }

        infos = {
            agent: {"done_reason": done_reasons[agent]}
            for agent in live_agents
        }

        self.agents = [
            agent
            for agent in live_agents
            if not terminations[agent] and not truncations[agent]
        ]

        self.state = observations

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        render_env(self)

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