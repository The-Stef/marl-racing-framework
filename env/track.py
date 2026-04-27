import numpy as np

def wrap_angle(angle):
    """Wrap angle in [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def car_heading(env):
    """Return the car's real forward heading."""
    return wrap_angle(float(env.CAR.hull.angle) + np.pi / 2)

def compute_radial_error(env):
    """Compute signed distance from the ideal circular centerline."""
    distance_from_center = np.sqrt(
        (env.TRACK_CENTER_X - env.CAR.hull.position[0]) ** 2 +
        (env.TRACK_CENTER_Y - env.CAR.hull.position[1]) ** 2
    )
    return distance_from_center - env.TRACK_RADIUS

def compute_desired_direction(env):
    """Return the tangent direction angle the car should follow."""
    rx = env.CAR.hull.position[0] - env.TRACK_CENTER_X
    ry = env.CAR.hull.position[1] - env.TRACK_CENTER_Y

    # Clockwise tangent, center-to-car position rotated by 90 degrees to right
    tx = ry
    ty = -rx

    return np.arctan2(ty, tx)

def tangential_velocity(env):
    """Project car velocity onto the clockwise tangent direction."""
    rx = env.CAR.hull.position[0] - env.TRACK_CENTER_X
    ry = env.CAR.hull.position[1] - env.TRACK_CENTER_Y
    r = np.sqrt(rx * rx + ry * ry) + 1e-8

    # Clockwise tangent
    tx = ry / r
    ty = -rx / r

    vx = env.CAR.hull.linearVelocity[0]
    vy = env.CAR.hull.linearVelocity[1]

    return vx * tx + vy * ty

def current_tile(env):
    """Return current angular tile index around the circular track."""

    # Get car position as an angle relative to center of track
    theta = np.arctan2(
        env.CAR.hull.position[1] - env.TRACK_CENTER_Y,
        env.CAR.hull.position[0] - env.TRACK_CENTER_X
    )

    # Convert angular position [-pi, pi] to [0, 2pi]
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # Map position to specific tile
    tile = int(theta / (2 * np.pi) * env.NUM_TILES)
    return tile