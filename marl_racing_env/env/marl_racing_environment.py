from .helpers.track_helpers import current_tile_lane, compute_radial_error, compute_car_start_pose
from .helpers.reward_helpers import compute_reward
from .helpers.observation_helpers import get_obs
from .helpers.render_helpers import render_env
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from configs import default as cfg
from ..car_dynamics import Car
from gymnasium import spaces
import numpy as np
import functools
import Box2D

class MARLRacingEnv(ParallelEnv):
    """Multi-Agent racing environment with a top-down view."""

    metadata = {
        "name": "marl_racing_environment_v0",
        "render_modes": ["human"],
        "render_fps": 30
    }

    def _map_agent_names_and_ids(self, agent_no):
        """
        Create PettingZoo agent names and ID mapping.
        :param agent_no: Number of agents to be created.
        """
        self.possible_agents = [
            "car_" + str(r)
            for r in range(agent_no)
        ]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    def _shuffle_agent_order(self):
        """For training robustness. Shuffles the order of agents."""
        shuffled_agents = self.agents[:]
        self.np_random.shuffle(shuffled_agents)

        return shuffled_agents

    def _initialize_rng(self, seed):
        """
        Using the provided seed, initialize random number generator.
        :param seed: Random seed.
        """
        if seed is not None or not hasattr(self, "np_random"):
            self.np_random, self.np_random_seed = seeding.np_random(seed)

    def _reset_per_agent_dictionaries(self):
        """Clear dictionaries that hold per-agent data."""
        self.CARS = {}
        self.PREV_THETA = {}
        self.LAST_D_THETA = {}
        self.LAP_PROGRESS = {}
        self.LAP_COUNT = {}
        self.VISITED_TILES = {}
        self.CURRENT_LAP_STEPS = {}

    def _spawn_agents(self, agent_container, cars_per_row, lateral_spacing, longitudinal_spacing):
        """
        Set data of individual agents in dictionaries that hold per-agent data.
        :param agent_container: Data structure which holds a list of agents.
        :param cars_per_row: Number of cars per row.
        :param lateral_spacing: Lateral spacing of the agents.
        :param longitudinal_spacing: Longitudinal spacing of the agents.
        """
        for idx, agent in enumerate(agent_container):
            car_start_position_x, car_start_position_y, car_start_direction = compute_car_start_pose(self, agent, idx, cars_per_row, lateral_spacing, longitudinal_spacing)

            self.CARS[agent] = Car(
                self.WORLD,
                car_start_direction,
                car_start_position_x,
                car_start_position_y,
            )
            self.CARS[agent].hull.userData['agent'] = agent

            # Set up each prev_theta
            self.PREV_THETA[agent] = np.arctan2(
                self.CARS[agent].hull.position[1] - self.TRACK_CENTER_Y,
                self.CARS[agent].hull.position[0] - self.TRACK_CENTER_X,
            )

            self.LAP_PROGRESS[agent] = np.float32(0.0)
            self.LAST_D_THETA[agent] = np.float32(0.0)
            self.LAP_COUNT[agent] = 0
            self.CURRENT_LAP_STEPS[agent] = 0
            self.VISITED_TILES[agent] = {current_tile_lane(self, agent)}

    def _set_variables_from_config(self):
        """Initialize variables with values from a config file."""
        self.PHYSICS_FPS = cfg.PHYSICS_FPS
        self.DT = 1.0 / self.PHYSICS_FPS  # Sole exception

        self.ACTION_REPEAT = cfg.ACTION_REPEAT
        self.HEADING_ERROR_PENALTY = cfg.HEADING_ERROR_PENALTY
        self.LIDAR_FOV = cfg.LIDAR_FOV
        self.LIDAR_MAX_DISTANCE = cfg.LIDAR_MAX_DISTANCE
        self.LIDAR_NUM_RAYS = cfg.LIDAR_NUM_RAYS
        self.MAX_LAPS = cfg.MAX_LAPS
        self.MAX_SPEED = cfg.MAX_SPEED
        self.MAX_STEPS = cfg.MAX_STEPS
        self.NUM_TILES = cfg.NUM_TILES
        self.OPPONENT_DETECTION_RADIUS = cfg.OPPONENT_DETECTION_RADIUS
        self.SCREEN_SIZE = cfg.SCREEN_SIZE
        self.START_DIRECTION = cfg.START_DIRECTION
        self.TRACK_CENTER_X = cfg.TRACK_CENTER_X
        self.TRACK_CENTER_Y = cfg.TRACK_CENTER_Y
        self.TRACK_HALF_WIDTH = cfg.TRACK_HALF_WIDTH
        self.TRACK_RADIUS = cfg.TRACK_RADIUS
        self.ZOOM = cfg.ZOOM

    def _apply_actions_to_live_agents(self, live_agents, terminations, truncations, actions, steers, throttles, gases, brakes):
        """
        For each live agent, apply action values to `steer`, `gas`, and `brake`.
        :param live_agents: Agents that are currently still training.
        :param terminations: Dictionary containing termination information.
        :param truncations: Dictionary containing truncation information.
        :param actions: Actions to be applied.
        :param steers: Per-agent applied steering values, updated in-place.
        :param throttles: Per-agent applied throttle values, updated in-place.
        :param gases: Per-agent applied gas values, updated in-place.
        :param brakes: Per-agent applied brake values, updated in-place.
        """
        for agent in live_agents:
            if terminations[agent] or truncations[agent]:
                continue

            self.CURRENT_LAP_STEPS[agent] += 1

            steer = float(np.tanh(actions[agent][0]))
            throttle = float(np.tanh(actions[agent][1]))
            gas = max(throttle, 0.0)
            brake = max(-throttle, 0.0)

            steers[agent] = steer
            throttles[agent] = throttle
            gases[agent] = gas
            brakes[agent] = brake

            self.CARS[agent].steer(steer)
            self.CARS[agent].gas(gas)
            self.CARS[agent].brake(brake)
            self.CARS[agent].step(self.DT)

    def _check_for_agent_collisions(self, rewards, terminations, truncations, done_reasons):
        """
        Check if agents collide. Uses Box2D-based collision.
        :param rewards: Dictionary containing per-agent rewards.
        :param terminations: Dictionary containing termination information.
        :param truncations: Dictionary containing truncation information.
        :param done_reasons: Dictionary containing reasons why individual agents finished training.
        """
        for contact in self.WORLD.contacts:
            if contact.touching:
                u1 = contact.fixtureA.body.userData
                u2 = contact.fixtureB.body.userData
                if isinstance(u1, dict) and u1.get('type') == 'hull' and \
                        isinstance(u2, dict) and u2.get('type') == 'hull':
                    a1 = u1.get('agent')
                    a2 = u2.get('agent')
                    if a1 in rewards and a2 in rewards:
                        # Apply penalty to both agents if they are still live
                        if not terminations[a1] and not truncations[a1]:
                            rewards[a1] -= cfg.COLLISION_PENALTY / self.ACTION_REPEAT
                            terminations[a1] = True
                            done_reasons[a1] = "collision"
                        if not terminations[a2] and not truncations[a2]:
                            rewards[a2] -= cfg.COLLISION_PENALTY / self.ACTION_REPEAT
                            terminations[a2] = True
                            done_reasons[a2] = "collision"

    def _update_lap_progress(self, agent):
        """
        Keep track of each agent's progress around the lap.
        :param agent: Agent whose progress is being updated.
        """
        theta = np.arctan2(
            self.CARS[agent].hull.position[1] - self.TRACK_CENTER_Y,
            self.CARS[agent].hull.position[0] - self.TRACK_CENTER_X
        )

        d_theta = theta - self.PREV_THETA[agent]
        if d_theta > np.pi:
            d_theta -= 2 * np.pi
        elif d_theta < -np.pi:
            d_theta += 2 * np.pi

        self.LAST_D_THETA[agent] = np.float32(d_theta)
        self.LAP_PROGRESS[agent] = np.float32(self.LAP_PROGRESS[agent] + d_theta)
        self.PREV_THETA[agent] = np.float32(theta)

    def _build_observations(self, agent_container):
        """
        Build the ``observations`` dictionary.
        :param agent_container: Data structure which holds a list of agents.
        """
        return \
        {
            agent: get_obs(self, agent).astype(np.float32)
            for agent in agent_container
        }

    def _agent_is_off_track(self, agent):
        """Check if agent is off-track.
        :param agent: Agent under suspicion of being off-track.
        :return: True if agent is off-track, False otherwise.
        """
        return abs(compute_radial_error(self, agent)) > self.TRACK_HALF_WIDTH

    def _agent_has_completed_lap(self, agent):
        """Check if agent has completed a lap.
        Currently, progress is tracked in radians. Agents move clockwise (0 to -2pi), and as such, progress is negatively measured.

        :param agent: Agent under suspicion of having completed a track.
        """
        next_lap_target = -2 * np.pi * (self.LAP_COUNT[agent] + 1) # Threshold to be passed

        return self.LAP_PROGRESS[agent] <= next_lap_target

    @staticmethod
    def _terminate_agent(agent, terminations, done_reasons, actual_reason):
        """
        Terminate agent with ``done_reason``.
        :param agent: Agent that is being terminated.
        :param terminations: Dictionary containing termination information.
        :param done_reasons: Dictionary containing reasons why individual agents finished.
        :param actual_reason: Actual reason why agent finished.
        """
        terminations[agent] = True
        done_reasons[agent] = actual_reason

    @staticmethod
    def _truncate_agent(agent, truncations, done_reasons, actual_reason):
        """
        Truncate agent with ``done_reason``.
        :param agent: Agent that is being truncated.
        :param truncations: Dictionary containing truncation information.
        :param done_reasons: Dictionary containing reasons why individual agents got truncated.
        :param actual_reason: Actual reason why agent was truncated.
        """
        truncations[agent] = True
        done_reasons[agent] = actual_reason

    @staticmethod
    def _build_infos(agent_container, **fields):
        """
        Build infos for the given agents from optional per-agent fields.
        :param agent_container: Data structure which holds a list of agents.
        :param fields: Optional per-agent fields.
        """
        return \
        {
            agent: {
                key: value[agent]
                for key, value in fields.items()
                if value is not None
            }
            for agent in agent_container
        }

    @staticmethod
    def _initialize_step_dictionaries(live_agents):
        """
        Initialize dictionaries that hold per-agent data for each step.
        :param live_agents: Data structure which holds a list of agents.
        """
        rewards = {agent: 0.0 for agent in live_agents}
        terminations = {agent: False for agent in live_agents}
        truncations = {agent: False for agent in live_agents}
        done_reasons = {agent: "not_done" for agent in live_agents}

        steers = {agent: 0.0 for agent in live_agents}
        throttles = {agent: 0.0 for agent in live_agents}
        gases = {agent: 0.0 for agent in live_agents}
        brakes = {agent: 0.0 for agent in live_agents}

        return rewards, terminations, truncations, done_reasons, steers, throttles, gases, brakes

    def __init__(self, render_mode=None, **kwargs):
        """Initialize the variables contained within the environment."""
        self.render_mode = render_mode

        self.NUM_AGENTS = kwargs.get("num_agents", cfg.NUM_AGENTS)
        self._map_agent_names_and_ids(self.NUM_AGENTS)

        # Initialize all necessary variables
        self.state = None
        self.WORLD = None
        self.SCREEN = None
        self.CLOCK = None
        self.STEPS = 0

        self._reset_per_agent_dictionaries()
        self._set_variables_from_config()

    def reset(self, seed=None, options=None):
        """Reset the environment to its starting point."""
        self._initialize_rng(seed)

        self.agents = self.possible_agents[:]
        self.STEPS = 0
        self.WORLD = Box2D.b2World((0, 0))

        self._reset_per_agent_dictionaries()
        shuffled_agents = self._shuffle_agent_order()

        # Set random positions for agents on the track (within certain ranges)
        cars_per_row = int(self.np_random.choice([1, 2], p=[0.9, 0.1]))
        lateral_spacing = float(self.np_random.uniform(1.0, 3.5))
        longitudinal_spacing = float(self.np_random.uniform(3.0, 12.0))

        self._spawn_agents(shuffled_agents, cars_per_row, lateral_spacing, longitudinal_spacing)

        observations = self._build_observations(self.agents)
        infos = self._build_infos(self.agents)
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        Apply agent actions, advance physics, and return the next environment state.
        :param actions: List of agents' actions.
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        live_agents = self.agents[:]
        rewards, terminations, truncations, done_reasons, steers, throttles, gases, brakes = self._initialize_step_dictionaries(live_agents)

        for _ in range(self.ACTION_REPEAT):
            self.STEPS += 1

            self._apply_actions_to_live_agents(live_agents, terminations, truncations, actions, steers, throttles, gases, brakes)
            self.WORLD.Step(self.DT, 6, 2)
            self._check_for_agent_collisions(rewards, terminations, truncations, done_reasons)

            for agent in live_agents:
                if terminations[agent] or truncations[agent]:
                    continue

                self._update_lap_progress(agent)

                rewards[agent] += compute_reward(self, agent)

                if self._agent_is_off_track(agent):
                    self._terminate_agent(agent, terminations, done_reasons, "car_crash")
                    continue

                if self._agent_has_completed_lap(agent):
                    self.LAP_COUNT[agent] += 1
                    self.CURRENT_LAP_STEPS[agent] = 0

                    # Reset tile rewards for the new lap
                    self.VISITED_TILES[agent] = {current_tile_lane(self, agent)}

                    # Fixed-lap mode, e.g. MAX_LAPS = 1
                    if self.MAX_LAPS is not None and self.LAP_COUNT[agent] >= self.MAX_LAPS:
                        terminations[agent] = True
                        done_reasons[agent] = "max_laps_reached"
                    else:
                        # Endurance mode: lap completed, but episode continues
                        done_reasons[agent] = "not_done"

            # If the episode is taking too long
            if self.STEPS >= self.MAX_STEPS:
                for agent in live_agents:
                    if not terminations[agent]:
                        self._truncate_agent(agent, truncations, done_reasons, "timeout")
                break

            if all(terminations[a] or truncations[a] for a in live_agents):
                break

        observations = self._build_observations(live_agents)
        infos = self._build_infos(
            live_agents,
            done_reason=done_reasons,
            lap_progress=self.LAP_PROGRESS,
            terminated=terminations,
            truncated=truncations,
            lap_count=self.LAP_COUNT,
            reward=rewards,
            steer=steers,
            throttle=throttles,
            gas=gases,
            brake=brakes,
        )

        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                self.CARS[agent].destroy()
                del self.CARS[agent]

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
        """Invoke the helper function which renders the environment."""
        render_env(self)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Speed, Heading error, Radial error, angular velocity
        return spaces.Box(
            low=np.array(
                [0.0, -np.pi, -self.TRACK_HALF_WIDTH, -20.0] + [0.0] * self.LIDAR_NUM_RAYS,
                dtype=np.float32
            ),
            high=np.array(
                [self.MAX_SPEED, np.pi, self.TRACK_HALF_WIDTH, 20.0] + [1.0] * self.LIDAR_NUM_RAYS,
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