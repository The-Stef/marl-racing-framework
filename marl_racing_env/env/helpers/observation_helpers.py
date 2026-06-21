from .track_helpers import car_heading, compute_desired_direction, wrap_angle, compute_radial_error
from configs import default as cfg
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

    lidar = compute_lidar(env, agent)

    observation = np.concatenate(
        [
            np.array(
                [velocity, heading_error, radial_error, angular_velocity],
                dtype=np.float32
            ),
            lidar,
        ]
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

def compute_lidar(env, agent):
    """Cast LIDAR rays out from agent to check distance from border or other agents."""
    own_pos = env.CARS[agent].hull.position
    heading = car_heading(env, agent)

    ray_angles = np.linspace(
        0.0,
        env.LIDAR_FOV,
        env.LIDAR_NUM_RAYS,
        endpoint=False,
    )

    distances = []

    for relative_angle in ray_angles:
        # Get ray orientation & directional vector from angle
        angle = heading + relative_angle
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        nearest = env.LIDAR_MAX_DISTANCE

        # Track borders
        inner_radius = env.TRACK_RADIUS - env.TRACK_HALF_WIDTH
        outer_radius = env.TRACK_RADIUS + env.TRACK_HALF_WIDTH

        # Check if LIDAR rays are hitting the borders
        for radius in [inner_radius, outer_radius]:
            border_distance_hit = ray_circle_distance(
                origin=own_pos,
                direction=direction,
                center=np.array([env.TRACK_CENTER_X, env.TRACK_CENTER_Y], dtype=np.float32),
                radius=radius,
            )

            if border_distance_hit is not None:
                nearest = min(nearest, border_distance_hit)

        # Opponent cars, approximated as circles
        for other_agent, other_car in env.CARS.items():
            if other_agent == agent:
                continue

            other_pos = np.array(other_car.hull.position, dtype=np.float32)

            opponent_distance_hit = ray_circle_distance(
                origin=own_pos,
                direction=direction,
                center=other_pos,
                radius=env.OPPONENT_DETECTION_RADIUS,
            )

            if opponent_distance_hit is not None:
                nearest = min(nearest, opponent_distance_hit)

        distances.append(nearest / env.LIDAR_MAX_DISTANCE)

    return np.array(distances, dtype=np.float32)

def ray_circle_distance(origin, direction, center, radius):
    """Return the closest positive distance where a ray intersects a circle. Used for LIDAR checks against circular track borders and circular
approximations of opponent agents. Returns ``None`` if the ray does not hit the circle."""
    offset = origin - center

    a = np.dot(direction, direction)
    b = 2.0 * np.dot(offset, direction)
    c = np.dot(offset, offset) - radius ** 2

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    candidates = [t for t in [t1, t2] if 0.0 <= t <= 1e9]

    if not candidates:
        return None

    return min(candidates)