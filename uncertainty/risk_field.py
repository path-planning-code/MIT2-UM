"""URF unified risk field ().

First layer computes the five position-dependent risks at a point; the
second layer fuses them into R_min / R_max.  The field is stateless: it
only reads the frozen UncertaintySnapshot, so every query inside a BRD
call is deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import config
from uncertainty import variant

def _terrain_high(world, x: float, y: float) -> float:
    """Terrain upper surface the online risk model trusts.

    Deterministic / obstacle-only ablations disable the terrain-uncertainty
    interval, so the risk field uses the exact physical surface instead of the
    conservative upper bound.
    """
    if variant.terrain_uncertainty_on():
        return world.terrain.h_high(x, y)
    return world.terrain.h_true(x, y)

class RiskField:
    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mu_high(r: float) -> float:
        return float(np.clip(r, 0.0, 1.0))

    @staticmethod
    def _nearest_obstacle_range(snapshot, p: np.ndarray) -> float:
        obstacles = snapshot.world.obstacles
        if not obstacles:
            return config.SENSOR_RANGE_MAX
        dist = min(float(np.linalg.norm(p - o.pos)) for o in obstacles)
        return float(np.clip(dist, 0.0, config.SENSOR_RANGE_MAX))

    def _d_pos_max(self, snapshot, idx: int, p: np.ndarray) -> float:
        if config.RISK_QUERY_SPATIAL:
            uav_type = snapshot.world.uavs[idx].type_key
            return snapshot.query_gps_at(p[0], p[1], uav_type)[1]
        return snapshot.get_delta_pos_max(idx)

    def _d_sens_max(
        self, snapshot, idx: int, p: np.ndarray, range_m: float
    ) -> float:
        if config.RISK_QUERY_SPATIAL:
            return snapshot.query_sensor_at(p[0], p[1], range_m)[1]
        return snapshot.get_delta_sens_max(idx)

    @staticmethod
    def terrain_risk_at(snapshot, p) -> float:
        """Point-wise terrain risk r_terr (), used for path lookahead."""
        p = np.asarray(p, dtype=float)
        h_high = _terrain_high(snapshot.world, p[0], p[1])
        margin = max(0.0, float(h_high - p[2]))
        return float(1.0 - np.exp(-margin / 10.0))

    # ------------------------------------------------------------------
    # Layer 1
    # ------------------------------------------------------------------
    def first_layer(
        self, snapshot, idx: int, p
    ) -> Dict[str, float]:
        """Return r_terr, r_gps, r_obs, r_sens, r_comm at point p."""
        world = snapshot.world
        p = np.asarray(p, dtype=float)

        h_high = _terrain_high(world, p[0], p[1])
        #  r_terr = 1 - exp(-max(0, h_high - z)/10).
        margin = max(0.0, float(h_high - p[2]))
        r_terr = 1.0 - np.exp(-margin / 10.0)

        d_pos_max = self._d_pos_max(snapshot, idx, p)
        r_gps = float(np.clip(d_pos_max / config.RISK_DENOM_GPS, 0.0, 1.0))

        range_m = self._nearest_obstacle_range(snapshot, p)
        d_sens_max = self._d_sens_max(snapshot, idx, p, range_m)
        r_sens = float(np.clip(d_sens_max / config.RISK_DENOM_SENS, 0.0, 1.0))

        r_obs_terms: List[float] = []
        for k, obstacle in enumerate(world.obstacles):
            d_robust = max(
                0.0,
                float(np.linalg.norm(p - obstacle.pos))
                - d_pos_max
                - snapshot.delta_obs(k, config.H_META)
                - d_sens_max
                - obstacle.params.radius,
            )
            r_obs_terms.append(np.exp(-d_robust / config.RISK_OBS_SCALE))
        r_obs = float(max(r_obs_terms)) if r_obs_terms else 0.0

        r_comm = float(
            np.clip(snapshot.get_comm_max(idx) / config.RISK_DENOM_COMM, 0.0, 1.0)
        )

        return {
            "r_terr": float(r_terr),
            "r_gps": r_gps,
            "r_obs": r_obs,
            "r_sens": r_sens,
            "r_comm": r_comm,
        }

    # ------------------------------------------------------------------
    # Layer 2 (T-S fusion)
    # ------------------------------------------------------------------
    @staticmethod
    def second_layer(
        first: Dict[str, float],
    ) -> Tuple[float, float]:
        if config.UNCERTAINTY_VARIANT == "no_urf":
            mean = float(
                np.mean(
                    [
                        first["r_terr"],
                        first["r_gps"],
                        first["r_obs"],
                        first["r_sens"],
                        first["r_comm"],
                    ]
                )
            )
            return mean, mean
        r = first
        f = [
            RiskField._mu_high(r["r_terr"]),
            RiskField._mu_high(r["r_gps"]),
            RiskField._mu_high(r["r_obs"]),
            RiskField._mu_high(r["r_sens"]),
            RiskField._mu_high(r["r_comm"]),
        ]
        rule_outputs = [
            0.9 * r["r_terr"] + 0.05 * r["r_gps"] + 0.05 * r["r_obs"],
            0.1 * r["r_terr"] + 0.8 * r["r_gps"] + 0.05 * r["r_obs"]
            + 0.05 * r["r_sens"],
            0.05 * r["r_terr"] + 0.05 * r["r_gps"] + 0.8 * r["r_obs"]
            + 0.1 * r["r_comm"],
            0.05 * r["r_terr"] + 0.7 * r["r_sens"] + 0.25 * r["r_obs"],
            0.05 * r["r_terr"] + 0.7 * r["r_comm"] + 0.25 * r["r_obs"],
        ]
        denominator = sum(f) + config.EPS_SAFE
        r_min = sum(fi * out for fi, out in zip(f, rule_outputs)) / denominator
        r_max = max(rule_outputs)
        return float(r_min), float(r_max)

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------
    def query(self, snapshot, idx: int, p) -> Dict[str, float]:
        """Full URF query: first layer plus R_min / R_max."""
        first = self.first_layer(snapshot, idx, p)
        r_min, r_max = self.second_layer(first)
        result = dict(first)
        result["R_min"] = r_min
        result["R_max"] = r_max
        return result

    def grid_query(
        self,
        snapshot,
        idx: int,
        xs: np.ndarray,
        ys: np.ndarray,
        z: float,
    ) -> np.ndarray:
        """R_max over an (x, y) grid, used for heatmaps and acceptance."""
        grid = np.full((len(ys), len(xs)), np.nan)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                grid[iy, ix] = self.query(snapshot, idx, (x, y, z))["R_max"]
        return grid
