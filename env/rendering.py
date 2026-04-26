import pygame

def render_env(self):
    if self.SCREEN is None:
        pygame.init()
        self.SCREEN = pygame.display.set_mode(self.SCREEN_SIZE)
        pygame.display.set_caption("NewEnv")
        self.CLOCK = pygame.time.Clock()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.close()
            return

    canvas = pygame.Surface(self.SCREEN_SIZE)
    canvas.fill((255, 255, 255))

    center = (self.SCREEN_SIZE[0] // 2, self.SCREEN_SIZE[1] // 2)

    # Draw track
    outer_r = int((self.TRACK_RADIUS + self.TRACK_HALF_WIDTH) * self.ZOOM)
    inner_r = int((self.TRACK_RADIUS - self.TRACK_HALF_WIDTH) * self.ZOOM)
    ideal_r = int(self.TRACK_RADIUS * self.ZOOM)

    pygame.draw.circle(canvas, (0, 0, 0), center, outer_r, width=2)
    pygame.draw.circle(canvas, (0, 0, 0), center, inner_r, width=2)
    pygame.draw.circle(canvas, (0, 180, 0), center, ideal_r, width=1)

    # Draw car onto the canvas
    translation = center
    camera_angle = 0.0
    self.CAR.draw(
        canvas,
        zoom=self.ZOOM,
        translation=translation,
        angle=camera_angle,
        draw_particles=True,
    )

    # Flip vertically so world-up looks like screen-up
    flipped = pygame.transform.flip(canvas, False, True)
    self.SCREEN.blit(flipped, (0, 0))

    pygame.display.flip()
    self.CLOCK.tick(self.metadata["render_fps"])