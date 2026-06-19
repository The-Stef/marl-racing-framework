from .track_helpers import car_heading, compute_desired_direction, wrap_angle, compute_radial_error
import numpy as np

"""
Helper functions related to environment observations are added here.
"""

def get_obs(env, agent):
    """Return the agent's observation. Values computed using info straight from the Car object."""

    velocity_x = float(env.CARS[agent].hull.linearVelocity[0])
    velocity_y = float(env.CARS[agent].hull.linearVelocity[1])
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    current_direction = car_heading(env, agent)
    desired_direction = compute_desired_direction(env, agent)
    heading_error = wrap_angle(desired_direction - current_direction)

    radial_error = np.clip(
        compute_radial_error(env, agent),
        -env.TRACK_HALF_WIDTH,
        env.TRACK_HALF_WIDTH
    )

    angular_velocity = float(env.CARS[agent].hull.angularVelocity)

    distance_to_other_agent = compute_distance_to_other_agent(env, agent)

    observation = np.array(
        [velocity, heading_error, radial_error, angular_velocity, distance_to_other_agent],
        dtype=np.float32
    )

    return np.clip(
        observation,
        env.observation_space(agent).low,
        env.observation_space(agent).high,
    )

def compute_distance_to_other_agent(env, agent):
    """In a two agent setting, get current agent's distance to other agent."""
    own_pos = env.CARS[agent].hull.position

    other_agents = [
        other_agent
        for other_agent in env.CARS.keys()
        if other_agent != agent
    ]

    if not other_agents:
        return 1.0

    other_pos = env.CARS[other_agents[0]].hull.position

    dx = other_pos[0] - own_pos[0]
    dy = other_pos[1] - own_pos[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)

    return float(distance / env.TRACK_RADIUS)