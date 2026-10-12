"""UncertaintySnapshot: one frozen measurement set per simulation step.

 every sim step samples the dynamic fields once; the whole BRD
call, E computation, f_c evaluation, URF queries and u features must read
the same snapshot.  Noise is never resampled inside a step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

import config
from uncertainty.comm_model import CommModel
from uncertainty.dynamic_field import DynamicField
from uncertainty.gps_model import GPSModel
from uncertainty.obstacle_pred import delta_obs
from uncertainty.sensor_model import SensorModel
from uncertainty.variant import gps_sens_comm_on, obstacle_uncertainty_on

if TYPE_CHECKING:
    from environment.world import World

_GPS_MODEL: GPSModel | None = None
_SENSOR_MODEL: SensorModel | None = None
_COMM_MODEL: CommModel | None = None

def _models() -> tuple[GPSModel, SensorModel, CommModel]:
    global _GPS_MODEL, _SENSOR_MODEL, _COMM_MODEL
    if _GPS_MODEL is None:
        _GPS_MODEL = GPSModel()
        _SENSOR_MODEL = SensorModel()
        _COMM_MODEL = CommModel()
    return _GPS_MODEL, _SENSOR_MODEL, _COMM_MODEL

class UncertaintySnapshot:
    def __init__(self, world: "World") -> None:
        self.world = world
        self.step = world.time
        self.dynamic_field: DynamicField = world.dynamic_field
        self.terrain = world.terrain
        self.delay = self.dynamic_field.delay(float(world.time))
        gps_model, sensor_model, comm_model = _models()

        self.per_uav: Dict[int, Dict[str, float]] = {}
        for idx in world.active_indices():
            uav = world.uavs[idx]
            nsat, hdop, snr = self.dynamic_field.query_gps(
                uav.pos[0], uav.pos[1], world.time
            )
            visibility = self.dynamic_field.query_visibility(
                uav.pos[0], uav.pos[1], world.time
            )
            d_pos_min, d_pos_max = gps_model.infer(
                nsat, hdop, snr, uav.type_key
            )
            range_m = self._nearest_obstacle_range(uav)
            d_sens_min, d_sens_max = sensor_model.infer(visibility, range_m)
            comm_max = comm_model.delta_comm_max(
                world.uavs, idx, self.delay
            )
            if not gps_sens_comm_on():
                d_pos_min = d_pos_max = 0.0
                d_sens_min = d_sens_max = 0.0
                comm_max = 0.0
            self.per_uav[idx] = {
                "nsat": nsat,
                "hdop": hdop,
                "snr": snr,
                "visibility": visibility,
                "range_m": range_m,
                "delta_pos_min": d_pos_min,
                "delta_pos_max": d_pos_max,
                "delta_sens_min": d_sens_min,
                "delta_sens_max": d_sens_max,
                "comm_max": comm_max,
            }

        self.obstacle_deltas: Dict[int, Dict[str, float]] = {}
        t_seconds = float(world.time) * config.DT
        for k, obstacle in enumerate(world.obstacles):
            if obstacle_uncertainty_on():
                self.obstacle_deltas[k] = {
                    "brd": delta_obs(obstacle, config.H_BRD, t=t_seconds),
                    "meta": delta_obs(obstacle, config.H_META, t=t_seconds),
                }
            else:
                self.obstacle_deltas[k] = {"brd": 0.0, "meta": 0.0}

    # ------------------------------------------------------------------
    def _nearest_obstacle_range(self, uav) -> float:
        if not self.world.obstacles:
            return config.SENSOR_RANGE_MAX
        dist = min(
            float(np.linalg.norm(uav.pos - obstacle.pos))
            for obstacle in self.world.obstacles
        )
        return float(min(max(dist, 0.0), config.SENSOR_RANGE_MAX))

    # ------------------------------------------------------------------
    # Frozen per-UAV accessors
    # ------------------------------------------------------------------
    def get_delta_pos_max(self, idx: int) -> float:
        return self.per_uav[idx]["delta_pos_max"]

    def get_delta_sens_max(self, idx: int) -> float:
        return self.per_uav[idx]["delta_sens_max"]

    def get_comm_max(self, idx: int) -> float:
        return self.per_uav[idx]["comm_max"]

    def get_measurement(self, idx: int, key: str) -> float:
        return self.per_uav[idx][key]

    def delta_obs(self, obstacle_idx: int, horizon: int) -> float:
        if horizon == config.H_BRD:
            return self.obstacle_deltas[obstacle_idx]["brd"]
        if horizon == config.H_META:
            return self.obstacle_deltas[obstacle_idx]["meta"]
        raise ValueError(f"unexpected horizon {horizon}")

    # ------------------------------------------------------------------
    # Deterministic spatial queries for URF at predicted positions.
    # They reuse the frozen per-step noise, so they never resample.
    # ------------------------------------------------------------------
    def query_gps_at(
        self, x: float, y: float, uav_type: str
    ) -> tuple[float, float]:
        gps_model, _, _ = _models()
        nsat, hdop, snr = self.dynamic_field.query_gps(x, y, self.step)
        return gps_model.infer(nsat, hdop, snr, uav_type)

    def query_visibility_at(self, x: float, y: float) -> float:
        return self.dynamic_field.query_visibility(x, y, self.step)

    def query_sensor_at(
        self, x: float, y: float, range_m: float
    ) -> tuple[float, float]:
        _, sensor_model, _ = _models()
        visibility = self.query_visibility_at(x, y)
        return sensor_model.infer(visibility, range_m)

def create_snapshot(world: "World") -> UncertaintySnapshot:
    """Create one frozen uncertainty snapshot for the current sim step."""
    return UncertaintySnapshot(world)
