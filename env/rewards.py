import numpy as np
from configs import default as cfg


def compute_reward(env):
    """Compute reward for the current environment state."""
    radial_error = env._compute_radial_error()
    tangential_speed = env._tangential_velocity()
    angular_velocity = abs(float(env.CAR.hull.angularVelocity))

    # Check whether car is still on the track
    on_track = abs(radial_error) <= env.TRACK_HALF_WIDTH

    reward = 0.0

    # Reward exploration, but only while on track
    tile = env._current_tile()
    new_tile_reward = 0.0

    if on_track and tile not in env.VISITED_TILES:
        env.VISITED_TILES.add(tile)
        new_tile_reward = cfg.NEW_TILE_REWARD

    reward += new_tile_reward

    # Reward real clockwise motion, punish backward motion
    reward += cfg.TANGENTIAL_SPEED_WEIGHT * tangential_speed

    # Stay near centerline
    reward -= cfg.RADIAL_ERROR_WEIGHT * abs(radial_error)

    # Punish spinning in place
    reward -= cfg.ANGULAR_VELOCITY_WEIGHT * angular_velocity

    # Big crash penalty
    if not on_track:
        reward -= cfg.OFF_TRACK_PENALTY

    # Lap bonus
    if env.LAP_PROGRESS <= -2 * np.pi:
        reward += cfg.LAP_BONUS

    return reward