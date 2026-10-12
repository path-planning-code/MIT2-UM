"""
ID-IT2-FDTM: Iteration-Driven Interval Type-2 Fuzzy Dynamic Terrain Model
==========================================================================

Implements the dynamic terrain model described in:
  "Two-Stage Path Planning for Post-Disaster UAV Swarms"

Core equations (refer to paper Section III-C, Eqs. (3-1) to (3-8)):
  (3-1)  h_0(x,y)       — nominal DEM elevation
  (3-2)  h(x,y) = [h_L(x,y), h_U(x,y)]  — interval type-2 fuzzy elevation
  (3-3)  h_L(x,y) = h_0(x,y) * (1 - alpha)
  (3-4)  h_U(x,y) = h_0(x,y) * (1 + alpha)
  (3-5)  v(x,y) = tanh(||grad h_0(x,y)|| / s_max)  — slope vulnerability index
  (3-6)  h_L <- max(0, h_L - h_0 * r(iter, v))
  (3-7)  h_U <- max(0, h_U - h_0 * r(iter, v))
  (3-8)  z_j >= z_U(x_j, y_j) + d_safe  — robust safety constraint

Fuzzy rule table (Table 3-1): reduction rate r(iter, v) as fraction of h_0
               Low v (<0.3)   Mid v (0.3-0.7)   High v (>0.7)
  Early        0.5% (0.005)   3.0% (0.030)      6.0% (0.060)
  Middle       0.2% (0.002)   1.5% (0.015)      3.0% (0.030)
  Late         0.0% (0.000)   0.5% (0.005)      1.0% (0.010)

License: MIT
"""

import numpy as np

class _RegularGridInterpolator:
    """Minimal bilinear interpolator on a regular 2D grid (numpy only).

    Replaces scipy.interpolate.RegularGridInterpolator so ID-IT2-FDTM has
    no third-party dependency beyond numpy.  The calling convention and
    out-of-bounds fill behavior match the scipy class used by this module.
    """

    def __init__(self, points, values, bounds_error=False, fill_value=0.0):
        self.x1d = np.asarray(points[0], dtype=float)
        self.y1d = np.asarray(points[1], dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.bounds_error = bounds_error
        self.fill_value = float(fill_value)
        self.dx = (
            (self.x1d[-1] - self.x1d[0]) / (len(self.x1d) - 1)
            if len(self.x1d) > 1 else 0.0
        )
        self.dy = (
            (self.y1d[-1] - self.y1d[0]) / (len(self.y1d) - 1)
            if len(self.y1d) > 1 else 0.0
        )

    def __call__(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        out = np.full(pts.shape[0], self.fill_value, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]
        x0, x1 = self.x1d[0], self.x1d[-1]
        y0, y1 = self.y1d[0], self.y1d[-1]
        valid = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        xi = np.clip(x, x0, x1)
        yi = np.clip(y, y0, y1)
        i = np.floor((xi - x0) / self.dx).astype(int)
        j = np.floor((yi - y0) / self.dy).astype(int)
        i = np.minimum(i, len(self.x1d) - 2)
        j = np.minimum(j, len(self.y1d) - 2)
        fx = (xi - (x0 + i * self.dx)) / self.dx
        fy = (yi - (y0 + j * self.dy)) / self.dy
        v00 = self.values[i, j]
        v10 = self.values[i + 1, j]
        v01 = self.values[i, j + 1]
        v11 = self.values[i + 1, j + 1]
        out[valid] = (
            v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy
        )[valid]
        return out

RegularGridInterpolator = _RegularGridInterpolator

# ============================================================
# Configuration — all constants self-contained (paper defaults)
# ============================================================

MAP_RANGE = (100.0, 100.0, 80.0)   # (x_max, y_max, z_max) in meters
SAFE_ALTITUDE = 5.0                 # minimum safe clearance above terrain (m)
ALPHA = 0.15                        # global fuzziness coefficient
T1 = 30                             # early -> middle phase boundary (iterations)
T2 = 60                             # middle -> late phase boundary (iterations)
T_UPDATE = 10                       # terrain evolution update interval
S_MAX = 1.0                         # slope normalization reference (~45 deg)

# ============================================================
# SimpleTerrain: minimal synthetic terrain for demo / testing
# ============================================================

class SimpleTerrain:
    """Synthetic terrain built from Gaussian peaks.

    Provides the minimal interface required by FuzzyDynamicTerrain:
    a ``get_elevation(x, y)`` method returning elevation at any point.
    """

    def __init__(self, map_range=MAP_RANGE, num_peaks=8, seed=2024):
        rng = np.random.RandomState(seed)
        self.map_range = map_range
        self.num_peaks = num_peaks
        self.peaks = []
        for i in range(num_peaks):
            cx = map_range[0] * (0.1 + 0.8 * rng.random())
            cy = map_range[1] * (0.1 + 0.8 * rng.random())
            h = map_range[2] * rng.uniform(0.3, 0.8)
            xs = map_range[0] * rng.uniform(0.07, 0.14)
            ys = map_range[1] * rng.uniform(0.07, 0.14)
            self.peaks.append({"cx": cx, "cy": cy, "h": h, "xs": xs, "ys": ys})

    def get_elevation(self, x, y):
        """Return elevation at (x, y). Accepts scalars or arrays."""
        x = np.atleast_1d(np.array(x, dtype=float))
        y = np.atleast_1d(np.array(y, dtype=float))
        z = np.zeros_like(x)
        for p in self.peaks:
            z += p["h"] * np.exp(
                -((x - p["cx"]) / p["xs"]) ** 2
                - ((y - p["cy"]) / p["ys"]) ** 2
            )
        return float(z[0]) if z.size == 1 else z

    def get_peaks_summary(self):
        lines = []
        for i, p in enumerate(self.peaks):
            lines.append(
                f"  Peak {i+1}: center=({p['cx']:.1f},{p['cy']:.1f}), "
                f"height={p['h']:.1f} m, "
                f"x_slope={p['xs']:.1f}, y_slope={p['ys']:.1f}"
            )
        return "\n".join(lines)

# ============================================================
# FuzzyDynamicTerrain — core ID-IT2-FDTM
# ============================================================

class FuzzyDynamicTerrain:
    """Iteration-Driven Interval Type-2 Fuzzy Dynamic Terrain Model.

    Parameters
    ----------
    base_terrain : object
        Must provide ``get_elevation(x, y)`` returning nominal elevation h_0.
    alpha : float
        Global fuzziness coefficient (Eqs. 3-3, 3-4). Default 0.15.
    T1, T2 : int
        Phase boundaries: early (<T1), middle (T1..<T2), late (>=T2).
    T_update : int
        Terrain update interval in iterations.
    s_max : float
        Normalization reference slope (Eq. 3-5). Default 1.0 (~45 deg).
    grid_size : int or None
        Override grid resolution; None = 100.

    Attributes
    ----------
    iter_count : int
        Current evolution iteration counter.
    X_grid, Y_grid : ndarray (gs, gs)
        Horizontal grid coordinates.
    """

    # ---- Zero-order T-S fuzzy rule table (Table 3-1) ----
    #       rows: phase (early=0, middle=1, late=2)
    #       cols: vulnerability class (low=0, mid=1, high=2)
    _RULE_TABLE = np.array([
        [0.005, 0.030, 0.060],   # Early  phase
        [0.002, 0.015, 0.030],   # Middle phase
        [0.000, 0.005, 0.010],   # Late   phase
    ])

    def __init__(self, base_terrain, alpha=ALPHA, T1=T1, T2=T2,
                 T_update=T_UPDATE, s_max=S_MAX, grid_size=None,
                 map_range=MAP_RANGE):
        self.base = base_terrain
        self.alpha = np.clip(float(alpha), 0.0, 1.0)
        self.T1 = int(T1)
        self.T2 = int(T2)
        self.T_update = int(T_update)
        self.s_max = float(s_max)
        self.map_range = tuple(map_range)
        self.iter_count = 0

        # ---- Determine grid ----
        if hasattr(base_terrain, "X_grid") and base_terrain.X_grid is not None:
            self.X_grid = base_terrain.X_grid.copy()
            self.Y_grid = base_terrain.Y_grid.copy()
            self._gs = self.X_grid.shape[0]
        elif grid_size is not None:
            self._gs = int(grid_size)
        else:
            self._gs = 100

        mr = self.map_range
        xv = np.linspace(0, mr[0], self._gs)
        yv = np.linspace(0, mr[1], self._gs)
        self.X_grid, self.Y_grid = np.meshgrid(xv, yv)

        # ---- Step 1: Extract nominal DEM h_0(x,y)  (Eq. 3-1) ----
        self._h0 = np.zeros((self._gs, self._gs), dtype=np.float64)
        for iy in range(self._gs):
            for ix in range(self._gs):
                self._h0[iy, ix] = float(
                    base_terrain.get_elevation(
                        self.X_grid[iy, ix], self.Y_grid[iy, ix]))

        # ---- Step 2: Compute gradient magnitude ||grad h_0|| ----
        dy, dx = np.gradient(
            self._h0,
            self.Y_grid[1, 0] - self.Y_grid[0, 0],
            self.X_grid[0, 1] - self.X_grid[0, 0],
        )
        self._grad_norm = np.sqrt(dx ** 2 + dy ** 2)

        # ---- Step 3: Vulnerability index v(x,y)  (Eq. 3-5) ----
        self._vulnerability = np.tanh(self._grad_norm / self.s_max)

        # ---- Step 4: Initial IT2 fuzzy elevation  (Eqs. 3-3, 3-4) ----
        self._h_L = self._h0 * (1.0 - self.alpha)
        self._h_U = self._h0 * (1.0 + self.alpha)
        self._z_U = self._h_U.copy()

        # ---- Step 5: Build spatial interpolators ----
        self._build_interpolators()

    # ==================================================================
    # Internal: interpolator construction
    # ==================================================================

    def _build_interpolators(self):
        x1d = self.X_grid[0, :]
        y1d = self.Y_grid[:, 0]

        self._interp_h0 = RegularGridInterpolator(
            (x1d, y1d), self._h0.T, bounds_error=False, fill_value=0.0)
        self._interp_hL = RegularGridInterpolator(
            (x1d, y1d), self._h_L.T, bounds_error=False, fill_value=0.0)
        self._interp_hU = RegularGridInterpolator(
            (x1d, y1d), self._h_U.T, bounds_error=False, fill_value=0.0)
        self._interp_zU = self._interp_hU
        self._interp_vuln = RegularGridInterpolator(
            (x1d, y1d), self._vulnerability.T,
            bounds_error=False, fill_value=0.0)

    # ==================================================================
    # Internal: fuzzy logic helpers
    # ==================================================================

    def _get_phase(self, iter_count):
        """Classify iteration into early(0) / middle(1) / late(2)."""
        if iter_count < self.T1:
            return 0
        elif iter_count < self.T2:
            return 1
        else:
            return 2

    @staticmethod
    def _get_vulnerability_class(v_val):
        """Classify vulnerability into low(0) / mid(1) / high(2)."""
        if v_val < 0.3:
            return 0
        elif v_val <= 0.7:
            return 1
        else:
            return 2

    def _lookup_reduction_rate(self, iter_count, v_val):
        """Zero-order T-S fuzzy rule lookup (Table 3-1).

        Returns reduction rate r(iter, v) as a fraction of h_0.
        """
        phase = self._get_phase(iter_count)
        vcls = self._get_vulnerability_class(v_val)
        return self._RULE_TABLE[phase, vcls]

    # ==================================================================
    # Evolution  (Eqs. 3-6, 3-7)
    # ==================================================================

    def evolve(self, iter_count=None):
        """Execute one terrain evolution step.

        Triggered every ``T_update`` iterations.  Updates both h_L and h_U
        using the zero-order T-S fuzzy rule table.

        Parameters
        ----------
        iter_count : int or None
            External iteration counter.  If None, auto-increments.

        Returns
        -------
        changed : bool
            True if any grid cell's elevation was modified.
        """
        if iter_count is None:
            self.iter_count += 1
            iter_count = self.iter_count
        else:
            self.iter_count = int(iter_count)

        if self.iter_count % self.T_update != 0:
            return False

        changed = False
        for iy in range(self._gs):
            for ix in range(self._gs):
                v_val = self._vulnerability[iy, ix]
                r = self._lookup_reduction_rate(self.iter_count, v_val)
                if r <= 0:
                    continue
                delta = self._h0[iy, ix] * r

                new_hL = max(0.0, self._h_L[iy, ix] - delta)
                if abs(new_hL - self._h_L[iy, ix]) > 1e-6:
                    self._h_L[iy, ix] = new_hL
                    changed = True

                new_hU = max(0.0, self._h_U[iy, ix] - delta)
                if abs(new_hU - self._h_U[iy, ix]) > 1e-6:
                    self._h_U[iy, ix] = new_hU
                    changed = True

        if changed:
            self._z_U = self._h_U.copy()
            self._build_interpolators()
        return changed

    # ==================================================================
    # Public query interface
    # ==================================================================

    def get_h0(self, x, y):
        """Nominal elevation h_0(x,y)  (Eq. 3-1)."""
        xa = np.atleast_1d(np.clip(x, 0, self.map_range[0]))
        ya = np.atleast_1d(np.clip(y, 0, self.map_range[1]))
        z = self._interp_h0(np.column_stack([xa, ya]))
        return float(z[0]) if z.size == 1 else z

    def get_hL(self, x, y):
        """Lower bound fuzzy elevation h_L(x,y)  (Eq. 3-3)."""
        xa = np.atleast_1d(np.clip(x, 0, self.map_range[0]))
        ya = np.atleast_1d(np.clip(y, 0, self.map_range[1]))
        z = self._interp_hL(np.column_stack([xa, ya]))
        return float(z[0]) if z.size == 1 else z

    def get_hU(self, x, y):
        """Upper bound fuzzy elevation h_U(x,y)  (Eq. 3-4)."""
        xa = np.atleast_1d(np.clip(x, 0, self.map_range[0]))
        ya = np.atleast_1d(np.clip(y, 0, self.map_range[1]))
        z = self._interp_hU(np.column_stack([xa, ya]))
        return float(z[0]) if z.size == 1 else z

    def get_zU(self, x, y):
        """Worst-case elevation for robust safety constraint (Eq. 3-8)."""
        return self.get_hU(x, y)

    def get_vulnerability(self, x, y):
        """Terrain fragility index v(x,y)  (Eq. 3-5)."""
        xa = np.atleast_1d(np.clip(x, 0, self.map_range[0]))
        ya = np.atleast_1d(np.clip(y, 0, self.map_range[1]))
        z = self._interp_vuln(np.column_stack([xa, ya]))
        return float(z[0]) if z.size == 1 else z

    def get_elevation(self, x, y):
        """Convenience: returns nominal elevation (backward compatible)."""
        return self.get_h0(x, y)

    # ==================================================================
    # Robust safety constraint  (Eq. 3-8)
    # ==================================================================

    def check_safety(self, x, y, z, d_safe=None):
        """Check robust safety: z >= z_U(x,y) + d_safe  (Eq. 3-8).

        Returns (is_safe, required_min_z).
        """
        if d_safe is None:
            d_safe = SAFE_ALTITUDE
        zu = self.get_zU(x, y)
        return z >= zu + d_safe, zu + d_safe

    def get_min_safe_z(self, x, y, d_safe=None):
        """Minimum safe altitude at (x, y)."""
        if d_safe is None:
            d_safe = SAFE_ALTITUDE
        return self.get_zU(x, y) + d_safe

    # ==================================================================
    # Surface data (for visualization)
    # ==================================================================

    def get_surface_data(self):
        """Return (X_grid, Y_grid, z_U) for current worst-case surface."""
        return self.X_grid, self.Y_grid, self._z_U.copy()

    def get_base_surface(self):
        """Return (X_grid, Y_grid, h_0) for nominal surface."""
        return self.X_grid, self.Y_grid, self._h0.copy()

    def get_vulnerability_map(self):
        """Return (X_grid, Y_grid, v) vulnerability map."""
        return self.X_grid, self.Y_grid, self._vulnerability.copy()

    def get_fuzzy_bounds(self):
        """Return (X_grid, Y_grid, h_L, h_U) IT2 fuzzy bounds."""
        return self.X_grid, self.Y_grid, self._h_L.copy(), self._h_U.copy()

    def get_gradient_map(self):
        """Return (X_grid, Y_grid, grad_norm) slope magnitude."""
        return self.X_grid, self.Y_grid, self._grad_norm.copy()

    # ==================================================================
    # State summary
    # ==================================================================

    def get_summary(self):
        phase_names = ["early", "middle", "late"]
        return {
            "alpha": self.alpha,
            "iter_count": self.iter_count,
            "phase": phase_names[self._get_phase(self.iter_count)],
            "h0_range": (float(self._h0.min()), float(self._h0.max())),
            "hL_range": (float(self._h_L.min()), float(self._h_L.max())),
            "hU_range": (float(self._h_U.min()), float(self._h_U.max())),
            "vuln_range": (float(self._vulnerability.min()),
                           float(self._vulnerability.max())),
        }

    def __repr__(self):
        s = self.get_summary()
        return (
            f"FuzzyDynamicTerrain(alpha={s['alpha']:.2f}, "
            f"iter={s['iter_count']}, phase={s['phase']}, "
            f"h0=[{s['h0_range'][0]:.1f}, {s['h0_range'][1]:.1f}], "
            f"hU=[{s['hU_range'][0]:.1f}, {s['hU_range'][1]:.1f}])"
        )

# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ID-IT2-FDTM — Quick Self-Test")
    print("=" * 60)

    # Create terrain and model
    terrain = SimpleTerrain(num_peaks=6, seed=42)
    fdtm = FuzzyDynamicTerrain(terrain, alpha=0.15, T1=30, T2=60,
                                T_update=10, grid_size=80)
    print(fdtm)
    print()

    # Evolve through all phases
    for phase_name, phase_end in [("Early", 30), ("Middle", 60), ("Late", 100)]:
        for it in range(fdtm.iter_count, phase_end):
            fdtm.evolve()
        s = fdtm.get_summary()
        print(f"After {phase_name} phase (iter={s['iter_count']}):")
        print(f"  hU range: [{s['hU_range'][0]:.1f}, {s['hU_range'][1]:.1f}] m")
        print(f"  vuln range: [{s['vuln_range'][0]:.3f}, {s['vuln_range'][1]:.3f}]")
        print()

    print("Self-test passed.")
