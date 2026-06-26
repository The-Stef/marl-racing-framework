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
        lateral_offset=0.0,
        lateral_spacing=2.25,
        longitudinal_spacing=3.0,
        orientation_offset=0.0,
):
    """
    Start placement logic:
    - 1 agent: start from one default centerline point, then apply offsets
    - 2 agents: start side-by-side around that centerline point
    - 3+ agents: use a two-column grid, with rows placed along the circular track
    :param centerline_offset: Centerline offset for agents.
    :param lateral_offset: Lateral offset for agents.
    :param orientation_offset: Orientation offset for agents.
    """

    num_agents = getattr(env, "NUM_AGENTS", None)
    if num_agents is None:
        num_agents = env.num_agents

    # Theta controls default centerline point.
    # centerline_offset moves the formation forward/backward along the circular centerline.
    base_theta = start_theta + centerline_offset / env.TRACK_RADIUS

    if num_agents == 1:
        row = 0

        # One agent starts on the centerline, plus optional lateral offset.
        agent_lateral_offset = lateral_offset

    elif num_agents == 2:
        row = 0

        # Two agents start side-by-side in separate lanes.
        if idx == 0:
            agent_lateral_offset = -lateral_spacing / 2.0
        else:
            agent_lateral_offset = lateral_spacing / 2.0

        # Optional shared lateral shift for the whole pair.
        agent_lateral_offset += lateral_offset

    else:
        # 3+ agents use a two-column grid.
        cars_per_row = 2

        row = idx // cars_per_row
        col = idx % cars_per_row

        # Fixed two-column placement:
        # col 0 -> left lane
        # col 1 -> right lane
        agent_lateral_offset = (col - 0.5) * lateral_spacing

        # Optional shared lateral shift for the whole grid.
        agent_lateral_offset += lateral_offset

    # For grid rows, move each row further along the circular centerline.
    theta = base_theta + row * longitudinal_spacing / env.TRACK_RADIUS

    # Default centerline point at this theta
    centerline_x = env.TRACK_CENTER_X + env.TRACK_RADIUS * np.cos(theta)
    centerline_y = env.TRACK_CENTER_Y + env.TRACK_RADIUS * np.sin(theta)

    # Radial direction = left/right from centerline
    radial_x = np.cos(theta)
    radial_y = np.sin(theta)

    # Tangent direction = forward direction along track
    tangent_x = np.sin(theta)
    tangent_y = -np.cos(theta)

    # Apply lateral offset
    x = centerline_x + agent_lateral_offset * radial_x
    y = centerline_y + agent_lateral_offset * radial_y

    # Correct orientation for this point on the track
    agent_orientation = np.arctan2(-tangent_x, tangent_y)

    # Optional orientation noise / offset
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