import numpy as np
import pygame

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

    return np.array(
        [velocity, heading_error, radial_error, angular_velocity],
        dtype=np.float32
    )

def wrap_angle(angle):
    """Wrap angle in [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def car_heading(env, agent):
    """Return the current car's real forward heading."""
    return wrap_angle(float(env.CARS[agent].hull.angle) + np.pi / 2)

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

def compute_car_start_position(env, agent, idx):
    """Compute the current car's starting position in a grid that has 2 cars per row."""

    cars_per_row = 2
    lateral_spacing = 2.5
    longitudinal_spacing = 4.0

    # Current agent's grid position
    row = idx // cars_per_row
    col = idx % cars_per_row

    # Grid starts from this position
    start_theta = np.pi

    # Middle of the road at the starting line
    centerline_x = env.TRACK_CENTER_X + env.TRACK_RADIUS * np.cos(start_theta)
    centerline_y = env.TRACK_CENTER_Y + env.TRACK_RADIUS * np.sin(start_theta)

    # The track's sideways & forward directions (used for car placements)
    radial_x = np.cos(start_theta)
    radial_y = np.sin(start_theta)

    tangent_x = np.sin(start_theta)
    tangent_y = -np.cos(start_theta)

    # How faw sideways the car should be
    lateral_offset = (col - 0.5) * lateral_spacing

    # How far backwards the car should be
    backward_offset = row * longitudinal_spacing

    # Compute final x & y
    x = centerline_x + lateral_offset * radial_x - backward_offset * tangent_x
    y = centerline_y + lateral_offset * radial_y - backward_offset * tangent_y

    return x, y

def render_env(env):
    if env.SCREEN is None:
        pygame.init()
        env.SCREEN = pygame.display.set_mode(env.SCREEN_SIZE)
        pygame.display.set_caption("Multi Agent Racing")
        env.CLOCK = pygame.time.Clock()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            return

    canvas = pygame.Surface(env.SCREEN_SIZE)
    canvas.fill((255, 255, 255))

    center = (env.SCREEN_SIZE[0] // 2, env.SCREEN_SIZE[1] // 2)

    # Draw track
    outer_r = int((env.TRACK_RADIUS + env.TRACK_HALF_WIDTH) * env.ZOOM)
    inner_r = int((env.TRACK_RADIUS - env.TRACK_HALF_WIDTH) * env.ZOOM)
    ideal_r = int(env.TRACK_RADIUS * env.ZOOM)

    pygame.draw.circle(canvas, (0, 0, 0), center, outer_r, width=2)
    pygame.draw.circle(canvas, (0, 0, 0), center, inner_r, width=2)
    pygame.draw.circle(canvas, (0, 180, 0), center, ideal_r, width=1)

    # Draw car onto the canvas
    translation = center
    camera_angle = 0.0

    for agent in env.CARS:
        env.CARS[agent].draw(
            canvas,
            zoom=env.ZOOM,
            translation=translation,
            angle=camera_angle,
            draw_particles=True,
        )

    # Flip vertically so world-up looks like screen-up
    flipped = pygame.transform.flip(canvas, False, True)
    env.SCREEN.blit(flipped, (0, 0))

    pygame.display.flip()
    env.CLOCK.tick(env.metadata["render_fps"])