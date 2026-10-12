"""
Reviewer-facing visualization of the Multi-modal Interval Type-2 Uncertainty
Model (MIT2-UM) and the Unified Risk Field (URF), corresponding to Section 3.3
of the manuscript.

The script instantiates the *unmodified* modelling modules shipped in this
repository and renders six figures into ``demo/figures/``:

  fig1_terrain_uncertainty.png
      Physical terrain surface, the perceived [h_low, h_high] interval, the
      epicenter Gaussian error band and the monotonic post-quake settlement.
  fig2_dynamic_spatiotemporal_fields.png
      Gaussian-mixture GPS / visibility interference fields and the in-flight
      monotonic degradation measured at an interference center.
  fig3_it2_fls.png
      Footprint of uncertainty (FOU) of the trapezoidal membership functions,
      the GPS localization error interval and the sensor ranging interval.
  fig4_obstacle_prediction.png
      Analytic diffusion of the obstacle prediction error over the prediction
      horizon for the three maneuver classes.
  fig5_communication_delay.png
      Sinusoidal communication delay at three damage levels and the peer-speed
      dependent position staleness.
  fig6_unified_risk_field.png
      First-layer single-source risk maps and the second-layer T-S fused
      R_min / R_max maps on a fixed flight plane.

Only numpy and matplotlib are required. Run from the repository root:

    python demo/visualize_mit2um.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from config import FieldCenter, ScenarioConfig
from environment.terrain import SyntheticTerrain, GaussianPeak
from environment.obstacle import Obstacle
from uncertainty.fdtm_adapter import FDTMTerrainAdapter
from uncertainty.dynamic_field import DynamicField
from uncertainty.gps_model import GPSModel
from uncertainty.sensor_model import SensorModel
from uncertainty.comm_model import CommModel
from uncertainty.membership import FuzzySet
from uncertainty.obstacle_pred import delta_obs
from uncertainty.risk_field import RiskField

FIG_DIR = os.path.join(REPO_ROOT, "demo", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GRID = np.linspace(0.0, config.WORLD_X_MAX, 120)
GX, GY = np.meshgrid(GRID, GRID)
RISK_CMAP = "viridis"

# A representative post-disaster scene: one severe GPS center, one severe
# visibility center, heavy communication damage and three maneuver classes.
DEMO_SCENARIO = ScenarioConfig(
    "DEMO", "10peak", ("A", "A", "A", "B", "C"),
    (FieldCenter(30.0, 30.0, "BAD"),),
    (FieldCenter(55.0, 45.0, "BAD"),),
    0.15, 0.08, 40.0,
    (("FIXED", 2), ("RANDOM", 6), ("HIGH_MANEUVER", 3)), "train",
    epicenter=(35.0, 35.0),
)


def _save(fig, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def build_terrain():
    """Synthetic Gaussian-peak terrain wrapped by the hybrid interval model."""
    rng = np.random.default_rng(7)
    peaks = []
    centers = [(22, 28, 42, 7), (52, 24, 48, 6), (74, 44, 40, 7),
               (34, 66, 52, 6), (66, 74, 46, 7), (46, 46, 34, 8)]
    for x, y, h, s in centers:
        peaks.append(GaussianPeak(float(x), float(y), float(h), float(s)))
    base = SyntheticTerrain(tuple(peaks), epicenter=DEMO_SCENARIO.epicenter)
    return FDTMTerrainAdapter(base, evolve_iters=0), rng


def sample_grid(func):
    return np.array([[func(float(x), float(y)) for x in GRID] for y in GRID])


# ---------------------------------------------------------------------------
# Figure 1: terrain uncertainty
# ---------------------------------------------------------------------------
def figure_terrain(terrain) -> None:
    h_true = sample_grid(terrain.h_true)
    h_high = sample_grid(terrain.h_high)
    h_low = sample_grid(terrain.h_low)
    bandwidth = h_high - h_low

    ex, ey = DEMO_SCENARIO.epicenter
    cross_x = np.linspace(0.0, config.WORLD_X_MAX, 300)

    def cross(selector):
        return np.array([selector(float(x), float(ey)) for x in cross_x])

    c_true_0, c_low_0, c_high_0 = cross(terrain.h_true), cross(terrain.h_low), cross(terrain.h_high)
    for _ in range(120):       # 120 steps * 0.5 s = 60 s
        terrain.evolve()
    c_true_60 = cross(terrain.h_true)
    for _ in range(120):       # to t = 120 s
        terrain.evolve()
    c_true_120 = cross(terrain.h_true)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    im0 = axes[0, 0].imshow(
        h_true, origin="lower", extent=[0, 100, 0, 100], cmap="terrain"
    )
    axes[0, 0].set_title("(a) physical elevation $h_{true}$ (m)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(
        bandwidth, origin="lower", extent=[0, 100, 0, 100], cmap="magma"
    )
    axes[0, 1].plot(ex, ey, marker="*", color="cyan", ms=14,
                    mec="k", mew=0.6, label="epicenter")
    for r in (config.TERRAIN_ERR_SIGMA, 2 * config.TERRAIN_ERR_SIGMA):
        axes[0, 1].add_patch(Circle((ex, ey), r, fill=False,
                                    ec="cyan", ls="--", lw=1.0))
    axes[0, 1].set_title("(b) perceived bandwidth $h_{high}-h_{low}$ (m)")
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    ax = axes[1, 0]
    ax.fill_between(cross_x, c_low_0, c_high_0, color="tab:orange",
                    alpha=0.30, label="perceived interval $[h_L,h_U]$")
    ax.plot(cross_x, c_true_0, color="k", lw=1.8, label="$h_{true}$")
    ax.plot(cross_x, c_high_0, color="tab:red", lw=0.9)
    ax.plot(cross_x, c_low_0, color="tab:blue", lw=0.9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"(c) cross-section at y = {ey:.0f} m")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(cross_x, c_true_0, label="t = 0 s", color="k")
    ax.plot(cross_x, c_true_60, label="t = 60 s", color="tab:green", ls="--")
    ax.plot(cross_x, c_true_120, label="t = 120 s", color="tab:red", ls=":")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("physical elevation (m)")
    ax.set_title("(d) monotonic post-quake settlement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig1_terrain_uncertainty.png")


# ---------------------------------------------------------------------------
# Figure 2: dynamic spatio-temporal fields
# ---------------------------------------------------------------------------
def figure_dynamic_fields() -> DynamicField:
    field = DynamicField(DEMO_SCENARIO, seed=7)
    nsat_map = np.array([[field.query_gps_raw(float(x), float(y))[0]
                          for x in GRID] for y in GRID])
    hdop_map = np.array([[field.query_gps_raw(float(x), float(y))[1]
                          for x in GRID] for y in GRID])
    vis_map = np.array([[field.query_visibility_raw(float(x), float(y))
                         for x in GRID] for y in GRID])

    steps = np.arange(0, 480)
    cx, cy = DEMO_SCENARIO.gps_centers[0].x, DEMO_SCENARIO.gps_centers[0].y
    n_t, h_t, s_t, v_t = [], [], [], []
    for step in steps:
        n, h, s = field.query_gps(cx, cy, step)
        n_t.append(n); h_t.append(h); s_t.append(s)
        v_t.append(field.query_visibility(cx, cy, step))
    t_sec = steps * config.DT

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, data, title, cmap in (
        (axes[0, 0], nsat_map, "(a) visible satellites $N_{sat}$", "viridis"),
        (axes[0, 1], hdop_map, "(b) HDOP", "magma"),
        (axes[1, 0], vis_map, "(c) visibility V", "cividis"),
    ):
        im = ax.imshow(data, origin="lower", extent=[0, 100, 0, 100], cmap=cmap)
        for c in (*DEMO_SCENARIO.gps_centers, *DEMO_SCENARIO.visibility_centers):
            ax.plot(c.x, c.y, marker="x", color="w", ms=10, mew=2)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    l1, = ax.plot(t_sec, n_t, color="tab:blue", label="$N_{sat}$")
    l2, = ax.plot(t_sec, s_t, color="tab:green", label="SNR (dB)")
    ax.set_xlabel("mission time (s)")
    ax.set_ylabel("$N_{sat}$ / SNR")
    ax2 = ax.twinx()
    l3, = ax2.plot(t_sec, h_t, color="tab:red", label="HDOP")
    l4, = ax2.plot(t_sec, np.array(v_t) * 30, color="tab:orange",
                   ls="--", label="visibility $\\times$ 30")
    ax2.set_ylabel("HDOP / scaled visibility")
    ax.set_title("(d) in-flight degradation at the interference center")
    ax.legend(handles=[l1, l2, l3, l4], fontsize=7, loc="center right")
    fig.tight_layout()
    _save(fig, "fig2_dynamic_spatiotemporal_fields.png")
    return field


# ---------------------------------------------------------------------------
# Figure 3: interval type-2 fuzzy logic systems
# ---------------------------------------------------------------------------
def figure_it2_fls() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    xs = np.linspace(0, 12, 400)
    colors = {"Low": "tab:red", "Med": "tab:orange", "High": "tab:green"}
    for label, fs in GPSModel().input_sets["N_sat"].items():
        up = np.array([fs.upper(float(x)) for x in xs])
        lo = np.array([fs.lower(float(x)) for x in xs])
        ax.fill_between(xs, lo, up, color=colors[label], alpha=0.25)
        ax.plot(xs, up, color=colors[label], lw=1.4,
                label=f"{label} (upper / lower)")
        ax.plot(xs, lo, color=colors[label], lw=0.9, ls="--")
    ax.set_xlabel("visible satellites $N_{sat}$")
    ax.set_ylabel("membership")
    ax.set_title("(a) FOU of trapezoidal memberships ($N_{sat}$)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    scores = np.arange(7)
    ax.bar(scores, config.GPS_Q_CENTROIDS, color="tab:blue", alpha=0.75)
    for q, c in zip(scores, config.GPS_Q_CENTROIDS):
        ax.text(q, c + 0.2, f"{c:.1f}", ha="center", fontsize=8)
    ax.set_xlabel("quality score Q (0 = worst, 6 = best)")
    ax.set_ylabel("consequent centroid (m)")
    ax.set_title("(b) GPS quality-score to error centroid")

    gps = GPSModel()
    nsat_sweep = np.linspace(0, 12, 100)
    ax = axes[1, 0]
    typed_curves = {}
    for uav_type, color in (("A", "tab:blue"), ("B", "tab:orange"), ("C", "tab:green")):
        low, up = [], []
        for n in nsat_sweep:
            dmin, dmax = gps.infer(float(n), 8.0, 18.0, uav_type)
            low.append(dmin); up.append(dmax)
        typed_curves[uav_type] = (np.array(low), np.array(up), color)
    for uav_type, (low, up, color) in typed_curves.items():
        ax.fill_between(nsat_sweep, low, up, color=color, alpha=0.15, zorder=1)
    styles = {"A": "-", "B": "--", "C": "-."}
    for uav_type, (low, up, color) in typed_curves.items():
        ax.plot(nsat_sweep, up, color=color, lw=2.0, ls=styles[uav_type],
                zorder=6, label=f"type {uav_type} upper bound")
    ax.set_xlabel("$N_{sat}$ (HDOP = 8, SNR = 18 dB)")
    ax.set_ylabel("localization error interval (m)")
    ax.set_title("(c) typed GPS localization interval")
    ax.legend(fontsize=8)

    sensor = SensorModel()
    dist_sweep = np.linspace(0, config.SENSOR_RANGE_MAX, 100)
    ax = axes[1, 1]
    sensor_curves = {}
    for vis, color in ((0.2, "tab:red"), (0.5, "tab:orange"), (0.9, "tab:green")):
        low, up = [], []
        for d in dist_sweep:
            dmin, dmax = sensor.infer(vis, float(d))
            low.append(dmin); up.append(dmax)
        sensor_curves[vis] = (np.array(low), np.array(up), color)
    for vis, (low, up, color) in sensor_curves.items():
        ax.fill_between(dist_sweep, low, up, color=color, alpha=0.15, zorder=1)
    for vis, (low, up, color) in sensor_curves.items():
        ax.plot(dist_sweep, up, color=color, lw=2.0, zorder=6,
                label=f"visibility = {vis:.1f}")
    ax.set_xlabel("target distance (m)")
    ax.set_ylabel("ranging error interval (m)")
    ax.set_title("(d) sensor ranging interval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig3_it2_fls.png")


# ---------------------------------------------------------------------------
# Figure 4: obstacle prediction diffusion
# ---------------------------------------------------------------------------
def figure_obstacle_prediction() -> None:
    horizons = np.arange(0, 41)
    obs_types = ("FIXED", "RANDOM", "HIGH_MANEUVER")
    colors = ("tab:blue", "tab:orange", "tab:red")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for otype, color in zip(obs_types, colors):
        obs = Obstacle(0, otype, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        vals = [delta_obs(obs, int(h), t=0.0) for h in horizons]
        ax.plot(horizons * config.DT, vals, color=color, lw=1.6, label=otype)
    for h, name in ((config.H_BRD, "$H_{BRD}$"), (config.H_META, "$H_{meta}$")):
        ax.axvline(h * config.DT, color="gray", ls=":", lw=1.0)
        ax.text(h * config.DT + 0.3, ax.get_ylim()[1] * 0.92, name, fontsize=9)
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("$\\delta_{obs}$ (m)")
    ax.set_title("(a) diffusion by maneuver class at t = 0 s")
    ax.legend(fontsize=8)

    ax = axes[1]
    obs = Obstacle(0, "HIGH_MANEUVER", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    for t, color in ((0.0, "tab:green"), (120.0, "tab:orange"), (240.0, "tab:red")):
        vals = [delta_obs(obs, int(h), t=t) for h in horizons]
        ax.plot(horizons * config.DT, vals, color=color, lw=1.6,
                label=f"t = {t:.0f} s")
    ax.axvline(config.H_BRD * config.DT, color="gray", ls=":", lw=1.0)
    ax.axvline(config.H_META * config.DT, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("$\\delta_{obs}$ (m)")
    ax.set_title("(b) high-maneuver growth over the mission")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig4_obstacle_prediction.png")


# ---------------------------------------------------------------------------
# Figure 5: communication delay uncertainty
# ---------------------------------------------------------------------------
def figure_communication() -> None:
    levels = (
        ("light  (0.05$\\pm$0.02 s, T=60 s)", 0.05, 0.02, 60.0, "tab:green"),
        ("medium (0.15$\\pm$0.08 s, T=40 s)", 0.15, 0.08, 40.0, "tab:orange"),
        ("severe (0.22$\\pm$0.12 s, T=25 s)", 0.22, 0.12, 25.0, "tab:red"),
    )
    t = np.linspace(0.0, 120.0, 600)
    steps = np.round(t / config.DT).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for label, base, amp, period, color in levels:
        scn = ScenarioConfig("X", "10peak", ("A",), (), (),
                             base, amp, period, (), "train")
        fld = DynamicField(scn, seed=1)
        delay = np.array([fld.delay(int(s)) for s in steps])
        ax.plot(t, delay, color=color, lw=1.4, label=label)
    ax.set_xlabel("mission time (s)")
    ax.set_ylabel("transmission delay (s)")
    ax.set_title("(a) sinusoidal delay at three damage levels")
    ax.legend(fontsize=7)

    ax = axes[1]
    comm = CommModel()
    speeds = np.linspace(0.0, 12.0, 100)
    for delay, color in ((0.05, "tab:green"), (0.15, "tab:orange"),
                         (0.27, "tab:red")):
        tau = comm.tau_comm(delay)
        ax.plot(speeds, speeds * tau, color=color, lw=1.6,
                label=f"$\\tau_{{comm}}$ = {tau:.2f} s")
    ax.set_xlabel("peer UAV speed (m/s)")
    ax.set_ylabel("position staleness $\\delta_{comm}$ (m)")
    ax.set_title("(b) speed-dependent position uncertainty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig5_communication_delay.png")


# ---------------------------------------------------------------------------
# Figure 6: unified risk field (two-layer fusion)
# ---------------------------------------------------------------------------
class _StubUAV:
    def __init__(self, type_key: str, v: float) -> None:
        self.type_key = type_key
        self.v = v
        self.active = True


class DemoSnapshot:
    """Minimal frozen snapshot exposing only the interface RiskField queries."""

    def __init__(self, terrain, obstacles, field, step: int) -> None:
        gps, sensor, comm = GPSModel(), SensorModel(), CommModel()
        self.world = type("World", (), {})()
        self.world.terrain = terrain
        self.world.obstacles = obstacles
        self.world.uavs = [_StubUAV("A", 5.0), _StubUAV("B", 3.0), _StubUAV("C", 2.0)]
        self._field, self._step = field, step
        self._gps, self._sensor, self._comm = gps, sensor, comm
        self._delay = field.delay(step)

    def query_gps_at(self, x, y, uav_type):
        nsat, hdop, snr = self._field.query_gps(x, y, self._step)
        return self._gps.infer(nsat, hdop, snr, uav_type)

    def query_sensor_at(self, x, y, range_m):
        vis = self._field.query_visibility(x, y, self._step)
        return self._sensor.infer(vis, range_m)

    def get_comm_max(self, idx):
        return self._comm.delta_comm_max(self.world.uavs, idx, self._delay)

    def delta_obs(self, obstacle_idx, horizon):
        return delta_obs(self.world.obstacles[obstacle_idx], horizon,
                         t=self._step * config.DT)


def figure_urf(terrain, field) -> None:
    config.RISK_QUERY_SPATIAL = True
    # Slow and moderately maneuverable obstacles keep the long-horizon
    # conservative blobs spatially readable on a single plane.
    obstacles = [
        Obstacle(0, "FIXED", (25.0, 30.0, 22.0), (1.0, 0.0, 0.0)),
        Obstacle(1, "FIXED", (72.0, 68.0, 22.0), (0.0, 1.0, 0.0)),
        Obstacle(2, "RANDOM", (58.0, 28.0, 22.0), (-1.0, 0.0, 0.0)),
    ]
    step = 40                                   # t = 20 s of mission time
    snapshot = DemoSnapshot(terrain, obstacles, field, step=step)
    rf = RiskField()
    plane_z = 22.0

    maps = {key: np.zeros_like(GX) for key in
            ("r_terr", "r_gps", "r_obs", "r_sens", "r_comm", "R_min", "R_max")}
    for iy in range(GX.shape[0]):
        for ix in range(GX.shape[1]):
            res = rf.query(snapshot, 0,
                           np.array([GX[iy, ix], GY[iy, ix], plane_z]))
            for key in maps:
                maps[key][iy, ix] = res[key]

    titles = (
        ("r_terr", "(a) terrain risk"),
        ("r_gps", "(b) localization risk"),
        ("r_obs", "(c) obstacle risk"),
        ("r_sens", "(d) sensing risk"),
        ("r_comm", "(e) communication risk (global per step)"),
        ("R_min", "(f) fused $R_{min}$"),
        ("R_max", "(g) conservative $R_{max}$"),
    )
    centers = (*DEMO_SCENARIO.gps_centers, *DEMO_SCENARIO.visibility_centers)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, (key, title) in zip(axes.flat, titles):
        im = ax.imshow(maps[key], origin="lower", extent=[0, 100, 0, 100],
                       cmap=RISK_CMAP, vmin=0.0, vmax=1.0)
        for c in centers:
            ax.plot(c.x, c.y, marker="+", color="w", ms=9, mew=1.5)
        for obs in obstacles:
            ax.plot(obs.pos[0], obs.pos[1], marker="o", mec="w",
                    mfc="none", ms=10, mew=1.4)
        if key == "r_comm":
            ax.text(2, 4, f"value = {maps[key].mean():.3f}", color="w",
                    fontsize=9, va="bottom")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes.flat[-1]
    # Vertical profile above the highest synthetic peak.
    px, py = 34.0, 66.0
    zs = np.linspace(0.0, 60.0, 200)
    profile = [RiskField.terrain_risk_at(
        snapshot, np.array([px, py, float(z)])) for z in zs]
    ax.plot(zs, profile, color="tab:blue", lw=1.8)
    ax.set_xlabel("flight altitude z (m)")
    ax.set_ylabel("terrain risk")
    ax.set_title(f"(h) vertical profile at ({px:.0f}, {py:.0f}) m")
    ax.grid(alpha=0.3)
    fig.suptitle(f"URF on the z = {plane_z:.0f} m flight plane (t = 20 s)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig6_unified_risk_field.png")


def main() -> None:
    np.random.seed(42)
    terrain, _ = build_terrain()
    figure_terrain(terrain)
    field = figure_dynamic_fields()
    figure_it2_fls()
    figure_obstacle_prediction()
    figure_communication()
    terrain2, _ = build_terrain()
    figure_urf(terrain2, field)
    print("All MIT2-UM / URF figures generated in", FIG_DIR)


if __name__ == "__main__":
    main()
