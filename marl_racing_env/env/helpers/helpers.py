import numpy as np

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

def get_obs(self, agent):
    """Return the agent's observation. Values computed using info straight from the Car object."""

    velocity_x = float(self.CARS[agent].hull.linearVelocity[0])
    velocity_y = float(self.CARS[agent].hull.linearVelocity[1])
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    # current_direction = self._car_heading()
    # desired_direction = self._compute_desired_direction()
    # heading_error = self._wrap_angle(desired_direction - current_direction)

    radial_error = np.clip(
        self._compute_radial_error(),
        -self.TRACK_HALF_WIDTH,
        self.TRACK_HALF_WIDTH
    )

    angular_velocity = float(self.CARS[agent].hull.angularVelocity)

    return np.array(
        [velocity, heading_error, radial_error, angular_velocity],
        dtype=np.float32
    )