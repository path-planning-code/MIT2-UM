"""ID-IT2-FDTM adapter exposing the project Terrain interface."""

from __future__ import annotations

from typing import Tuple

import numpy as np

import config
from environment.terrain import Terrain
from environment.dem_loader import bilinear_sample
from id_it2_fdtm import FuzzyDynamicTerrain

class FDTMTerrainAdapter(Terrain):
    """Wrap any base terrain (synthetic or real DEM) with ID-IT2-FDTM.

    h_true is passed through to the base terrain, which settles monotonically
    over flight time.  The UAV-visible interval is the union of the epicenter
    band epicenter_delta[+/-delta_terr] and the FDTM band [h_L, h_U]; the FDTM band
    degrades through its own iteration-driven reduction (Table 2) in flight.
    """

    def __init__(
        self,
        base_terrain,
        evolve_iters: int = config.TERRAIN_EVOLVE_ITERS,
        interval_mode: str = config.TERRAIN_INTERVAL_MODE,
        alpha: float = config.FDTM_ALPHA,
        T1: int = config.FDTM_T1,
        T2: int = config.FDTM_T2,
        T_update: int = config.FDTM_T_UPDATE,
        s_max: float = config.FDTM_S_MAX,
        grid_size: int = config.FDTM_GRID_SIZE,
        map_range: Tuple[float, float, float] = (
            config.WORLD_X_MAX,
            config.WORLD_Y_MAX,
            config.WORLD_Z_MAX,
        ),
    ) -> None:
        if interval_mode not in ("epicenter_delta", "fdtm", "hybrid"):
            raise ValueError(
                f"unknown terrain interval mode {interval_mode!r}; "
                "use 'epicenter_delta', 'fdtm' or 'hybrid'"
            )
        super().__init__(
            epicenter=tuple(float(v) for v in base_terrain.epicenter)
        )
        self.base = base_terrain
        self.interval_mode = interval_mode
        self.alpha = float(alpha)
        self.fdtm = FuzzyDynamicTerrain(
            base_terrain,
            alpha=alpha,
            T1=T1,
            T2=T2,
            T_update=T_update,
            s_max=s_max,
            grid_size=grid_size,
            map_range=tuple(map_range),
        )
        self.evolve_iters = int(evolve_iters)
        self.iter_count = 0
        self._evolve_step = 0
        # Pre-bake the fuzzy degradation only for the static-world baseline
        # (frozen terrain).  When the environment evolves in flight, the FDTM
        # interval degrades from its fresh h0 over the mission steps.
        if self.evolve_iters > 0 and not config.ENV_EVOLUTION_ENABLED:
            for it in range(1, self.evolve_iters + 1):
                self.fdtm.evolve(it)
            self.iter_count = self.evolve_iters

    def h_true(self, x: float, y: float) -> float:
        if self._grid_h_true is not None:
            return self._grid_lookup(self._grid_h_true, x, y)
        # Evolving terrain passes through to the moving physical surface; the
        # static baseline keeps the original FDTM h0 lookups for bit-exactness.
        if config.ENV_EVOLUTION_ENABLED:
            return float(self.base.h_true(x, y))
        return float(self.fdtm.get_h0(x, y))

    def _h_true_raw(self, x: float, y: float) -> float:
        """Raw elevation for base class delta calculations."""
        if config.ENV_EVOLUTION_ENABLED:
            return float(self.base.h_true(x, y))
        return float(self.fdtm.get_h0(x, y))

    def _fdtm_low(self, x: float, y: float) -> float:
        """FDTM lower bound anchored to the *current* physical surface h_true(t).

        The band must ride the settling surface, not the frozen t=0 grid:
        using the frozen h0 made h_L collapse and blew up the hybrid width.
        """
        return float(self.base.h_true(x, y)) * (1.0 - self.alpha)

    def _fdtm_high(self, x: float, y: float) -> float:
        """FDTM upper bound anchored to the *current* physical surface h_true(t)."""
        return float(self.base.h_true(x, y)) * (1.0 + self.alpha)

    def h_low(self, x: float, y: float) -> float:
        if self._grid_h_low is not None:
            return self._grid_lookup(self._grid_h_low, x, y)
        if self.interval_mode == "fdtm":
            return self._fdtm_low(x, y)
        epi_low = super().h_low(x, y)
        if self.interval_mode == "epicenter_delta":
            return epi_low
        return min(epi_low, self._fdtm_low(x, y))

    def h_high(self, x: float, y: float) -> float:
        if self._grid_h_high is not None:
            return self._grid_lookup(self._grid_h_high, x, y)
        if self.interval_mode == "fdtm":
            return self._fdtm_high(x, y)
        epi_high = super().h_high(x, y)
        if self.interval_mode == "epicenter_delta":
            return epi_high
        return max(epi_high, self._fdtm_high(x, y))

    def precompute_grid(self, resolution: int = 200) -> None:
        """Precompute elevation grid for fast lookup.

        First delegates to the base terrain for its own grid (used by
        the epicenter_delta fallback), then computes the adapter's own grid
        using its h_true/h_high/h_low methods.
        """
        # Let base terrain precompute its own grid if supported
        if hasattr(self.base, 'precompute_grid'):
            self.base.precompute_grid(resolution)

        # Now precompute adapter's own grid
        xs = np.linspace(0.0, config.WORLD_X_MAX, resolution)
        ys = np.linspace(0.0, config.WORLD_Y_MAX, resolution)
        xx, yy = np.meshgrid(xs, ys)

        flat_x = xx.ravel()
        flat_y = yy.ravel()

        true_vals = np.array([float(self.fdtm.get_h0(float(x), float(y))) for x, y in zip(flat_x, flat_y)])

        if self.interval_mode == "fdtm":
            low_vals = np.array([float(self.fdtm.get_hL(float(x), float(y))) for x, y in zip(flat_x, flat_y)])
            high_vals = np.array([float(self.fdtm.get_hU(float(x), float(y))) for x, y in zip(flat_x, flat_y)])
        elif self.interval_mode == "epicenter_delta":
            delta_vals = np.array([self._delta_terr_raw(float(x), float(y)) for x, y in zip(flat_x, flat_y)])
            low_vals = true_vals - delta_vals
            high_vals = true_vals + delta_vals
        else:  # hybrid
            fdtm_low = np.array([float(self.fdtm.get_hL(float(x), float(y))) for x, y in zip(flat_x, flat_y)])
            fdtm_high = np.array([float(self.fdtm.get_hU(float(x), float(y))) for x, y in zip(flat_x, flat_y)])
            delta_vals = np.array([self._delta_terr_raw(float(x), float(y)) for x, y in zip(flat_x, flat_y)])
            epi_low = true_vals - delta_vals
            epi_high = true_vals + delta_vals
            low_vals = np.minimum(epi_low, fdtm_low)
            high_vals = np.maximum(epi_high, fdtm_high)

        self._grid_h_true = true_vals.reshape(resolution, resolution)
        self._grid_h_low = low_vals.reshape(resolution, resolution)
        self._grid_h_high = high_vals.reshape(resolution, resolution)
        self._grid_resolution = resolution

    def _grid_lookup(self, grid: np.ndarray, x: float, y: float) -> float:
        """Bilinear lookup on the precomputed grid."""
        res = self._grid_resolution
        fx = (float(np.clip(x, 0.0, config.WORLD_X_MAX)) / config.WORLD_X_MAX) * (res - 1)
        fy = (float(np.clip(y, 0.0, config.WORLD_Y_MAX)) / config.WORLD_Y_MAX) * (res - 1)
        return bilinear_sample(grid, fx, fy)

    def delta_terr(self, x: float, y: float) -> float:
        return super().delta_terr(x, y)

    def get_elevation(self, x: float, y: float) -> float:
        return self.h_true(x, y)

    def evolve(self, iter_count: int | None = None) -> bool:
        """Advance terrain one step.

        The physical base terrain settles down through ``base.evolve()`` (the
        monotonic post-quake subsidence matching the Table-2 degradation
        direction); the fuzzy interval then tracks it via ``_fdtm_low/high``.
        ``super().evolve()`` bumps the clock and drops the stale cached grids.
        """
        if not config.ENV_EVOLUTION_ENABLED:
            return False
        if hasattr(self.base, "evolve"):
            self.base.evolve()
        super().evolve()
        return True

    def get_fuzzy_bounds(self):
        return self.fdtm.get_fuzzy_bounds()
