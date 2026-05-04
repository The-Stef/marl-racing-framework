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

def populate_dictionary_with_info(
        self,
        steer: float,
        throttle: float,
        gas: float,
        brake: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        done_reason: str | None,
        lap_count: int,
) -> dict:
    """
    Takes information from the step() function and adds it to the info dictionary.
    """
    car = self.CAR

    # Position
    x = float(car.hull.position[0])
    y = float(car.hull.position[1])

    # Velocity
    vx = float(car.hull.linearVelocity[0])
    vy = float(car.hull.linearVelocity[1])
    speed = float(np.sqrt(vx ** 2 + vy ** 2))

    # Track / lap progress
    radial_error = float(self._compute_radial_error())

    lap_progress_radians = float(self.LAP_PROGRESS)

    # Total progress across all laps. 100 = one lap, 200 = two laps, etc.
    lap_progress_percent = float(
        max((-self.LAP_PROGRESS / (2 * np.pi)) * 100.0, 0.0)
    )

    # Progress within the current lap only. This stays between 0 and 100.
    current_lap_progress_percent = float(
        lap_progress_percent % 100.0
    )

    info = {
        # Timing
        "steps": int(self.STEPS),

        # Action
        "steer": float(steer),
        "throttle": float(throttle),
        "gas": float(gas),
        "brake": float(brake),

        # Car state
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,

        # Track state
        "radial_error": radial_error,
        "abs_radial_error": abs(radial_error),
        "lap_count": int(lap_count),
        "lap_progress_radians": lap_progress_radians,
        "lap_progress_percent": lap_progress_percent,
        "current_lap_progress_percent": current_lap_progress_percent,

        # Reward
        "step_reward": float(reward),

        # Episode ending
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done_reason": done_reason if done_reason is not None else "not_done",
    }

    return info