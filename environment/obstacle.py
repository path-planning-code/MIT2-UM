"""Dynamic obstacles: three maneuver types and nominal extrapolation.

 motion models:
  FIXED          - constant direction, constant speed
  RANDOM         - direction rotates by N(0, 0.3) each step
  HIGH_MANEUVER  - direction rotates by N(0, 0.8), speed +/-20% per step
The speed perturbation is resampled around the base speed every step so it
stays bounded.  Obstacles fly horizontally at fixed z and reflect at the
world boundary so they remain inside the scenario.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

import config

class Obstacle:
    def __init__(
        self,
        idx: int,
        obs_type: str,
        pos: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        speed: float | None = None,
        motion_seed: int | None = None,
    ) -> None:
        self.idx = idx
        self.type_key = obs_type
        self.params = config.OBSTACLE_TYPES[obs_type]
        self.pos = np.asarray(pos, dtype=float)
        self.dir = np.asarray(direction, dtype=float)
        self.dir /= np.linalg.norm(self.dir)
        self.base_speed = self.params.speed
        self.speed = self.base_speed if speed is None else speed
        self._motion_rng = (
            np.random.default_rng(motion_seed)
            if motion_seed is not None
            else None
        )

    # ------------------------------------------------------------------
    def update(
        self,
        rng: np.random.Generator | None = None,
        dt: float = config.DT,
    ) -> None:
        """Advance the obstacle by one simulation step.

        Each obstacle may own a dedicated motion RNG (sub-offset seed); when
        one is provided, it is used instead of the shared world RNG so the
        obstacle's noise stream is independent of scenario composition.
        """
        rng = self._motion_rng if self._motion_rng is not None else rng
        if rng is None:
            raise ValueError("obstacle update requires a random generator")
        if self.type_key in ("RANDOM", "HIGH_MANEUVER"):
            theta = float(rng.normal(0.0, 0.3 if self.type_key == "RANDOM" else 0.8))
            self._rotate_direction(theta)
        if self.type_key == "HIGH_MANEUVER":
            perturbation = float(rng.uniform(-0.2, 0.2))
            self.speed = self.base_speed * (1.0 + perturbation)

        self.pos += self.speed * self.dir * dt
        self._reflect_boundary()

    def predict_position(
        self, t: int, horizon: int, dt: float = config.DT
    ) -> np.ndarray:
        """Nominal extrapolation: current direction and speed, no noise.

        Parameter `t` is reserved for future time-varying velocity models;
        the current implementation only depends on `horizon` (steps ahead).
        """
        return self.pos + self.speed * self.dir * (horizon * dt)

    # ------------------------------------------------------------------
    def _rotate_direction(self, theta: float) -> None:
        c, s = math.cos(theta), math.sin(theta)
        x, y = self.dir[0], self.dir[1]
        self.dir[0] = c * x - s * y
        self.dir[1] = s * x + c * y
        self.dir /= np.linalg.norm(self.dir)

    def _reflect_boundary(self) -> None:
        if self.pos[0] < config.WORLD_X_MIN:
            self.pos[0] = 2.0 * config.WORLD_X_MIN - self.pos[0]
            self.dir[0] = -self.dir[0]
        elif self.pos[0] > config.WORLD_X_MAX:
            self.pos[0] = 2.0 * config.WORLD_X_MAX - self.pos[0]
            self.dir[0] = -self.dir[0]
        if self.pos[1] < config.WORLD_Y_MIN:
            self.pos[1] = 2.0 * config.WORLD_Y_MIN - self.pos[1]
            self.dir[1] = -self.dir[1]
        elif self.pos[1] > config.WORLD_Y_MAX:
            self.pos[1] = 2.0 * config.WORLD_Y_MAX - self.pos[1]
            self.dir[1] = -self.dir[1]
