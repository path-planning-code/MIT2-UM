"""Thin terrain uncertainty wrapper used by MIT2-UM."""

from __future__ import annotations

import numpy as np

import config

class TerrainUncertainty:
    def __init__(self, terrain) -> None:
        self.terrain = terrain

    def h_true(self, x: float, y: float) -> float:
        return self.terrain.h_true(x, y)

    def delta_terr(self, x: float, y: float) -> float:
        return self.terrain.delta_terr(x, y)

    def h_low(self, x: float, y: float) -> float:
        return self.terrain.h_low(x, y)

    def h_high(self, x: float, y: float) -> float:
        return self.terrain.h_high(x, y)

    def r_terr(self, pos) -> float:
        """ terrain risk at a position."""
        margin = max(0.0, self.h_high(pos[0], pos[1]) - pos[2])
        return 1.0 - np.exp(-margin / 10.0)
