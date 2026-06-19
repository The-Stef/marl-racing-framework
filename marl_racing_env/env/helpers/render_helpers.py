import pygame

"""
Helper functions related to rendering the environment are added here.
"""

def render_env(env):
    """Renders the environment for visual inspection."""
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