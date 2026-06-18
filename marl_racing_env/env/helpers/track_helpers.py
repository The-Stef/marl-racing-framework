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
        cars_per_row = 1,
        lateral_spacing = 2.5,
        longitudinal_spacing = 12.0
):
    """Compute the car's starting position and heading on the circular track."""

    row = idx // cars_per_row
    col = idx % cars_per_row

    start_theta = np.pi

    # Move row along the circular track
    theta_offset = row * longitudinal_spacing / env.TRACK_RADIUS
    theta = start_theta + theta_offset

    # Radial direction at this theta
    radial_x = np.cos(theta)
    radial_y = np.sin(theta)

    # Centerline point for this row
    centerline_x = env.TRACK_CENTER_X + env.TRACK_RADIUS * np.cos(theta)
    centerline_y = env.TRACK_CENTER_Y + env.TRACK_RADIUS * np.sin(theta)

    # Side-by-side lane placement
    lateral_offset = (col - 0.5) * lateral_spacing

    x = centerline_x + lateral_offset * radial_x
    y = centerline_y + lateral_offset * radial_y

    # Tangent direction = forward direction along the circle
    tangent_x = np.sin(theta)
    tangent_y = -np.cos(theta)

    start_direction = np.arctan2(-tangent_x, tangent_y)

    return x, y, start_direction

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