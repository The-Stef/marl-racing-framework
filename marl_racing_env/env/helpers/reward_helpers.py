from .track_helpers import compute_radial_error, current_tile_lane, car_heading, compute_desired_direction, wrap_angle
from configs import default as cfg
import numpy as np

"""
Helper functions related to the agent reward are added here.
"""

def compute_reward(env, agent):
    """Compute reward for the current environment state & current agent."""
    radial_error = compute_radial_error(env, agent)

    # Check whether car is still on the track
    on_track = abs(radial_error) <= env.TRACK_HALF_WIDTH

    reward = 0.0

    # Reward exploration, but only while on track
    tile = current_tile_lane(env, agent)
    new_tile_reward = 0.0

    forward_progress = env.LAST_D_THETA[agent] < -1e-4

    if on_track and forward_progress and tile not in env.VISITED_TILES[agent]:
        env.VISITED_TILES[agent].add(tile)
        new_tile_reward = cfg.NEW_TILE_REWARD

    reward += new_tile_reward

    current_direction = car_heading(env, agent)
    desired_direction = compute_desired_direction(env, agent)
    heading_error = wrap_angle(desired_direction - current_direction)

    if on_track:
        reward -= env.HEADING_ERROR_PENALTY * abs(heading_error)

    # Big crash penalty
    if not on_track:
        reward -= cfg.OFF_TRACK_PENALTY

    # Lap bonus
    if env.LAP_PROGRESS[agent] <= -2 * np.pi * (env.LAP_COUNT[agent] + 1):
        reward += cfg.GAMMA_DISCOUNT ** env.CURRENT_LAP_STEPS[agent] * cfg.LAP_BONUS

    return reward