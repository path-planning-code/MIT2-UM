"""Terrain: synthetic Gaussian peaks plus the  two-layer uncertainty.

Physics uses h_true (ground truth); UAVs only know [h_low, h_high].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import config
from environment.dem_loader import (
    DEMInfo,
    bilinear_sample,
    crop_geometry,
    fill_nodata_nearest,
    load_dem_array,
    normalize_elevation,
)

@dataclass(frozen=True)
class GaussianPeak:
    x: float
    y: float
    height: float
    scale: float

class Terrain:
    def __init__(self, epicenter: Tuple[float, float] = (500.0, 500.0)) -> None:
        self.epicenter = np.asarray(epicenter, dtype=float)
        # Cached grid for fast lookup (populated by precompute_grid)
        self._grid_resolution: int = 0
        self._grid_h_true: np.ndarray | None = None
        self._grid_h_high: np.ndarray | None = None
        self._grid_h_low: np.ndarray | None = None
        # Evolution clock (seconds since episode start); advanced once per step.
        self._t: float = 0.0
        # Whether the physical surface settles over flight time.
        self._evolve_enabled: bool = config.ENV_EVOLUTION_ENABLED

    def precompute_grid(self, resolution: int = 200) -> None:
        """Precompute elevation on a regular grid for fast bilinear lookup.

        This dramatically speeds up repeated queries (e.g. during PSO optimization)
        by replacing per-query peak summation with a single bilinear interpolation.

        Parameters
        ----------
        resolution : int
            Grid resolution (resolution x resolution points).
        """
        xs = np.linspace(0.0, config.WORLD_X_MAX, resolution)
        ys = np.linspace(0.0, config.WORLD_Y_MAX, resolution)
        xx, yy = np.meshgrid(xs, ys)

        # Compute h_true via the original (slow) method, vectorized
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        true_vals = np.array([self._h_true_raw(float(x), float(y)) for x, y in zip(flat_x, flat_y)])
        delta_vals = np.array([self._delta_terr_raw(float(x), float(y)) for x, y in zip(flat_x, flat_y)])

        self._grid_h_true = true_vals.reshape(resolution, resolution)
        self._grid_h_high = (true_vals + delta_vals).reshape(resolution, resolution)
        self._grid_h_low = (true_vals - delta_vals).reshape(resolution, resolution)
        self._grid_resolution = resolution

    def _h_true_raw(self, x: float, y: float) -> float:
        """Raw elevation computation (overridden by subclasses)."""
        raise NotImplementedError

    def _delta_terr_raw(self, x: float, y: float) -> float:
        """Raw terrain uncertainty computation (epicenter Gaussian model)."""
        d = np.linalg.norm(np.asarray([x, y], dtype=float) - self.epicenter)
        return config.TERRAIN_ERR_BASE + config.TERRAIN_ERR_AMP * np.exp(
            -(d ** 2) / (2.0 * config.TERRAIN_ERR_SIGMA ** 2)
        )

    def _settle_factor(self) -> float:
        """Monotonic post-quake subsidence factor in (0, 1].

        ``h(t) = h0 * factor``; the factor starts at 1.0 and decays toward
        1 - TERRAIN_SETTLE_FRAC.  Every peak / cell shares it, so the whole
        surface sinks together instead of a single peak moving.
        """
        if not self._evolve_enabled:
            return 1.0
        return 1.0 - config.TERRAIN_SETTLE_FRAC * (
            1.0 - np.exp(-self._t / config.TERRAIN_SETTLE_TAU)
        )

    def _grid_lookup(self, grid: np.ndarray, x: float, y: float) -> float:
        """Bilinear lookup on the precomputed grid."""
        res = self._grid_resolution
        fx = (float(np.clip(x, 0.0, config.WORLD_X_MAX)) / config.WORLD_X_MAX) * (res - 1)
        fy = (float(np.clip(y, 0.0, config.WORLD_Y_MAX)) / config.WORLD_Y_MAX) * (res - 1)
        return bilinear_sample(grid, fx, fy)

    def h_true(self, x: float, y: float) -> float:
        if self._grid_h_true is not None:
            return self._grid_lookup(self._grid_h_true, x, y)
        return self._h_true_raw(x, y)

    def delta_terr(self, x: float, y: float) -> float:
        return self._delta_terr_raw(x, y)

    def h_low(self, x: float, y: float) -> float:
        if self._grid_h_low is not None:
            return self._grid_lookup(self._grid_h_low, x, y)
        return self._h_true_raw(x, y) - self._delta_terr_raw(x, y)

    def h_high(self, x: float, y: float) -> float:
        if self._grid_h_high is not None:
            return self._grid_lookup(self._grid_h_high, x, y)
        return self._h_true_raw(x, y) + self._delta_terr_raw(x, y)

    def evolve(self) -> None:
        """Advance terrain by one control step (DT) and drop cached grids.

        Subclasses that add physical motion must call ``super().evolve()`` so
        the clock advances and the stale ``_grid_*`` caches are invalidated;
        subsequent ``h_true``/``h_low``/``h_high`` calls then fall through to
        the time-dependent raw lookups.
        """
        self._t += config.DT
        self._grid_resolution = 0
        self._grid_h_true = None
        self._grid_h_high = None
        self._grid_h_low = None

class SyntheticTerrain(Terrain):
    def __init__(
        self,
        peaks: Tuple[GaussianPeak, ...],
        epicenter: Tuple[float, float] = (500.0, 500.0),
    ) -> None:
        super().__init__(epicenter)
        self.peaks = peaks

    def _h_true_raw(self, x: float, y: float) -> float:
        factor = self._settle_factor()
        total = 0.0
        for peak in self.peaks:
            total += (peak.height * factor) * np.exp(
                -(((x - peak.x) / peak.scale) ** 2)
                - (((y - peak.y) / peak.scale) ** 2)
            )
        return float(total)

    def get_elevation(self, x: float, y: float) -> float:
        """Interface required by ID-IT2-FDTM."""
        return self.h_true(x, y)

    def precompute_grid(self, resolution: int = 200) -> None:
        """Vectorized grid precomputation for synthetic terrain.

        Much faster than the base class implementation because it sums
        peak contributions in a vectorized way instead of looping over
        every grid point.
        """
        xs = np.linspace(0.0, config.WORLD_X_MAX, resolution)
        ys = np.linspace(0.0, config.WORLD_Y_MAX, resolution)
        xx, yy = np.meshgrid(xs, ys)  # shape: (resolution, resolution)

        # Vectorized: sum all Gaussian peaks
        true_vals = np.zeros_like(xx)
        for peak in self.peaks:
            true_vals += peak.height * np.exp(
                -((xx - peak.x) / peak.scale) ** 2
                - ((yy - peak.y) / peak.scale) ** 2
            )

        # Vectorized terrain uncertainty (epicenter model)
        d = np.sqrt((xx - self.epicenter[0]) ** 2 + (yy - self.epicenter[1]) ** 2)
        delta_vals = config.TERRAIN_ERR_BASE + config.TERRAIN_ERR_AMP * np.exp(
            -(d ** 2) / (2.0 * config.TERRAIN_ERR_SIGMA ** 2)
        )

        self._grid_h_true = true_vals
        self._grid_h_high = true_vals + delta_vals
        self._grid_h_low = true_vals - delta_vals
        self._grid_resolution = resolution

    @staticmethod
    def from_scenario(
        scenario_key: str,
        rng: np.random.Generator,
        epicenter: Tuple[float, float] | None = None,
        n_peaks: int | None = None,
    ) -> "SyntheticTerrain":
        scenario = config.SCENARIO_BY_KEY[scenario_key]
        if n_peaks is not None:
            peak_count = int(n_peaks)
            scale_range = (4.0, 8.0) if peak_count <= 10 else (2.5, 6.0)
        elif scenario.terrain == "10peak":
            peak_count, scale_range = 10, (4.0, 8.0)
        elif scenario.terrain == "30peak":
            peak_count, scale_range = 30, (2.5, 6.0)
        else:
            raise ValueError(f"not a synthetic terrain: {scenario.terrain}")

        peaks = []
        for _ in range(peak_count):
            for _ in range(5000):
                x = rng.uniform(
                    config.WORLD_X_MIN + 0.1 * (config.WORLD_X_MAX - config.WORLD_X_MIN),
                    config.WORLD_X_MAX - 0.1 * (config.WORLD_X_MAX - config.WORLD_X_MIN),
                )
                y = rng.uniform(
                    config.WORLD_Y_MIN + 0.1 * (config.WORLD_Y_MAX - config.WORLD_Y_MIN),
                    config.WORLD_Y_MAX - 0.1 * (config.WORLD_Y_MAX - config.WORLD_Y_MIN),
                )
                if all(
                    np.hypot(x - p.x, y - p.y) >= 0.06 * (config.WORLD_X_MAX - config.WORLD_X_MIN) for p in peaks
                ):
                    break
            else:
                raise RuntimeError("could not place terrain peaks with spacing")
            height = rng.uniform(0.35 * config.WORLD_Z_MAX, 0.7 * config.WORLD_Z_MAX)
            scale = rng.uniform(*scale_range)
            peaks.append(GaussianPeak(x, y, height, scale))
        return SyntheticTerrain(tuple(peaks), epicenter)

class RealDEMTerrain(Terrain):
    """Real SRTM-style DEM terrain using the full tile (downsampled).

    Matches the approach from the previous paper:
    - Load the full DEM tile (3601x3601, 1 degree x 1 degree)
    - Downsample to match the world grid resolution
    - Fill NoData values
    - Normalize elevation to fit world height (70% of max)
    - Map directly to world coordinates
    """

    def __init__(
        self,
        dem_path: str = config.REAL_DEM_PATH,
        elevation_mode: str = config.DEM_ELEVATION_MODE,
        max_relief: float = config.DEM_MAX_RELIEF,
        nodata_values=config.DEM_NODATA_VALUES,
        epicenter: Tuple[float, float] = (50.0, 50.0),
        landslide_enabled: bool = config.DEM_LANDSLIDE_ENABLED,
        landslide_seed: int | None = None,
        grid_size: int = 100,
    ) -> None:
        super().__init__(epicenter)
        self.dem_path = dem_path
        self.elevation_mode = elevation_mode
        self.max_relief = float(max_relief)
        self.nodata_values = tuple(nodata_values)
        self.landslide_enabled = landslide_enabled
        self.landslide_seed = landslide_seed
        self.landslides: list = []
        self.grid_size = grid_size

        # Load full DEM tile
        self._array, self._info = load_dem_array(dem_path)
        full_rows, full_cols = self._array.shape

        # Downsample to grid_size x grid_size
        from scipy.ndimage import zoom as ndzoom
        zf = (grid_size / full_rows, grid_size / full_cols)
        data = self._array.astype(float)
        valid_mask = ~np.isin(data, self.nodata_values)
        data_filled = data.copy()
        data_filled[~valid_mask] = float(data[valid_mask].min()) if valid_mask.any() else 0.0
        data_ds = ndzoom(data_filled, zf, order=1)
        mask_ds = ndzoom(valid_mask.astype(float), zf, order=0) > 0.5
        data_ds[~mask_ds] = np.nan

        # Fill NoData on downsampled grid
        data_ds = self._fill_nodata_small(data_ds)

        # Apply landslides if enabled
        self._nrows, self._ncols = data_ds.shape
        if self.landslide_enabled and self.landslide_seed is not None:
            self._apply_landslides(data_ds, int(self.landslide_seed))

        # Store raw stats
        self._h_min = float(np.nanmin(data_ds))
        self._h_max = float(np.nanmax(data_ds))

        # Normalize elevation
        self._crop = normalize_elevation(
            data_ds, self.elevation_mode, self.max_relief * 0.7
        )

    def _fill_nodata_small(self, data: np.ndarray) -> np.ndarray:
        """Fill NoData values using linear interpolation (same as previous paper)."""
        mask = np.isnan(data)
        if not mask.any():
            return np.nan_to_num(data, nan=0.0)
        from scipy.interpolate import griddata
        rows, cols = data.shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        valid = ~mask
        if valid.sum() < 3:
            return np.nan_to_num(data, nan=0.0)
        points = np.column_stack([xx[valid].ravel(), yy[valid].ravel()])
        values = data[valid].ravel()
        xi = np.column_stack([xx[mask].ravel(), yy[mask].ravel()])
        try:
            filled = griddata(points, values, xi, method='linear')
            data[mask] = filled
        except Exception:
            pass
        return np.nan_to_num(data, nan=0.0)

    def _apply_landslides(self, crop: np.ndarray, seed: int) -> None:
        """Add 3-5 Gaussian landslide deformations."""
        rng = np.random.default_rng(seed)
        lo, hi = config.DEM_LANDSLIDE_COUNT_RANGE
        count = int(rng.integers(lo, hi + 1))
        xs = np.arange(self._ncols) * (config.WORLD_X_MAX / (self._ncols - 1))
        ys = np.arange(self._nrows) * (config.WORLD_Y_MAX / (self._nrows - 1))
        for _ in range(count):
            cx = float(rng.uniform(0.0, config.WORLD_X_MAX))
            cy = float(rng.uniform(0.0, config.WORLD_Y_MAX))
            radius = float(rng.uniform(*config.DEM_LANDSLIDE_RADIUS_RANGE))
            dh = float(rng.uniform(*config.DEM_LANDSLIDE_DH_RANGE))
            self.landslides.append(
                {"x": cx, "y": cy, "radius": radius, "dh": dh}
            )
            dx2 = (xs - cx) ** 2
            dy2 = (ys - cy) ** 2
            perturb = dh * np.exp(
                -(dx2[np.newaxis, :] + dy2[:, np.newaxis])
                / (2.0 * radius ** 2)
            )
            crop += perturb

    def get_landslides(self) -> list:
        """Return the sampled landslide summary in world coordinates."""
        return [dict(item) for item in self.landslides]

    def _h_true_raw(self, x: float, y: float) -> float:
        """World (x, y) -> normalized DEM elevation.

        The full DEM tile is scaled to fit the world dimensions.
        x in [0, WORLD_X_MAX] maps to full DEM width.
        y in [0, WORLD_Y_MAX] maps to full DEM height.
        """
        fx = (float(np.clip(x, 0.0, config.WORLD_X_MAX)) / config.WORLD_X_MAX) * (
            self._ncols - 1
        )
        fy = (float(np.clip(y, 0.0, config.WORLD_Y_MAX)) / config.WORLD_Y_MAX) * (
            self._nrows - 1
        )
        z = bilinear_sample(self._crop, fx, fy)
        return z * self._settle_factor()

    def get_elevation(self, x: float, y: float) -> float:
        """Interface required by ID-IT2-FDTM."""
        return self.h_true(x, y)

    def dem_info(self) -> DEMInfo:
        return self._info

    def crop_summary(self) -> dict:
        return {
            "path": self.dem_path,
            "grid_size": self.grid_size,
            "nrows": self._nrows,
            "ncols": self._ncols,
            "h_min": self._h_min,
            "h_max": self._h_max,
            "elevation_mode": self.elevation_mode,
            "max_relief": self.max_relief,
        }
