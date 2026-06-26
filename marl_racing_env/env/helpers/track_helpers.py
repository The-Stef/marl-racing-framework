import numpy as np

"""
Helper functions related to the track are added here.
"""

def wrap_angle(angle):
    """Wrap angle in [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def car_heading(env, agent):
    """Return the current car's real forward heading."""
    return wrap_angle(float(env.CARS[agent].hull.angle) + np.pi / 2)

def current_tile(env, agent):
    """Return current angular tile index for current agent around the circular track."""

    # Get car position as an angle relative to center of track
    theta = np.arctan2(
        env.CARS[agent].hull.position[1] - env.TRACK_CENTER_Y,
        env.CARS[agent].hull.position[0] - env.TRACK_CENTER_X
    )

    # Convert angular position [-pi, pi] to [0, 2pi]
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # Map position to specific tile
    tile = int(theta / (2 * np.pi) * env.NUM_TILES)
    return tile

def current_tile_lane(env, agent):
    """Break the angular tile into two, lane-based tiles."""
    angular_tile = current_tile(env, agent)
    radial_error = compute_radial_error(env, agent)
    lane = 0 if radial_error < 0.0 else 1

    return angular_tile * 2 + lane

def compute_desired_direction(env, agent):
    """Return the tangent direction angle the current car should follow."""
    rx = env.CARS[agent].hull.position[0] - env.TRACK_CENTER_X
    ry = env.CARS[agent].hull.position[1] - env.TRACK_CENTER_Y

    # Clockwise tangent, center-to-car position rotated by 90 degrees to right
    tx = ry
    ty = -rx

    return np.arctan2(ty, tx)

def compute_radial_error(env, agent):
    """Compute signed distance from the ideal circular centerline."""
    distance_from_center = np.sqrt(
        (env.TRACK_CENTER_X - env.CARS[agent].hull.position[0]) ** 2 +
        (env.TRACK_CENTER_Y - env.CARS[agent].hull.position[1]) ** 2
    )
    return distance_from_center - env.TRACK_RADIUS

def compute_car_start_pose(
        env,
        agent,
        idx,
        start_theta=np.pi,
        centerline_offset=0.0,
        lateral_spacing=2.25,
        longitudinal_spacing=3.0,
        orientation_offset=0.0,
):
    """
    Compute start position and heading for:
    - 1 agent: centered on the track centerline
    - 2 agents: side-by-side
    - 3+ agents: two-column grid, with each row following the circular centerline
    """

    num_agents = env.NUM_AGENTS

    if num_agents == 1:
        cars_per_row = 1
        row = 0
        lateral_offset = 0.0

    elif num_agents == 2:
        cars_per_row = 2
        row = 0

        if idx == 0:
            lateral_offset = -lateral_spacing / 2.0
        else:
            lateral_offset = lateral_spacing / 2.0

    else:
        cars_per_row = 2

        row = idx // cars_per_row
        col = idx % cars_per_row

        agents_before_row = row * cars_per_row
        agents_in_this_row = min(cars_per_row, num_agents - agents_before_row)

        # Center each row.
        # Two cars: -spacing/2, +spacing/2
        # One car: 0
        lateral_offset = (col - 0.5) * lateral_spacing

    # Each row moves along the circular centerline, not backward along a straight tangent.
    theta = start_theta + (centerline_offset + row * longitudinal_spacing) / env.TRACK_RADIUS

    # Centerline point for this row
    centerline_x = env.TRACK_CENTER_X + env.TRACK_RADIUS * np.cos(theta)
    centerline_y = env.TRACK_CENTER_Y + env.TRACK_RADIUS * np.sin(theta)

    # Radial direction = sideways from track center
    radial_x = np.cos(theta)
    radial_y = np.sin(theta)

    # Tangent direction = forward direction along circular track
    tangent_x = np.sin(theta)
    tangent_y = -np.cos(theta)

    # Apply lateral offset from centerline
    x = centerline_x + lateral_offset * radial_x
    y = centerline_y + lateral_offset * radial_y

    # Proper heading for this point on the circular track
    agent_orientation = np.arctan2(-tangent_x, tangent_y)

    # Optional heading offset
    agent_orientation += orientation_offset

    return x, y, agent_orientation

def tangential_velocity(env, agent):
    """Project car velocity onto the clockwise tangent direction."""
    rx = env.CARS[agent].hull.position[0] - env.TRACK_CENTER_X
    ry = env.CARS[agent].hull.position[1] - env.TRACK_CENTER_Y
    r = np.sqrt(rx * rx + ry * ry) + 1e-8

    # Clockwise tangent
    tx = ry / r
    ty = -rx / r

    vx = env.CARS[agent].hull.linearVelocity[0]
    vy = env.CARS[agent].hull.linearVelocity[1]

    return vx * tx + vy * ty