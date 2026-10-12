"""Spatio-temporal uncertainty fields with per-step frozen noise."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import config

def prog_factor(t_seconds: float) -> float:
    """Monotonic degradation progression 0 -> 1 over the mission clock.

    Shared by the four evolving sources (GPS / sensor / obstacle / comm):
    ``prog(t) = 1 - exp(-t / FOUR_SRC_GROW_TAU)``.  Returns 0 when the
    environment is frozen or four-source evolution is disabled, so the static
    baseline is bit-exact.
    """
    if not (config.ENV_EVOLUTION_ENABLED and config.ENV_EVOLVE_FOUR_SOURCES):
        return 0.0
    return 1.0 - np.exp(-t_seconds / config.FOUR_SRC_GROW_TAU)

class DynamicField:
    def __init__(self, scenario: config.ScenarioConfig, seed: int) -> None:
        self.scenario = scenario
        self._rng = np.random.default_rng(seed)
        self._cached_step: int | None = None
        self._t_seconds: float = 0.0
        self.noise_offsets: Dict[str, float] = {
            "N": 0.0, "H": 0.0, "S": 0.0, "V": 0.0,
        }

    # ------------------------------------------------------------------
    def begin_step(self, step: int) -> None:
        """Sample the uncertainty noise once per simulation step."""
        if self._cached_step != step:
            self._cached_step = step
            self._t_seconds = step * config.DT
            self.noise_offsets = {
                key: float(self._rng.normal(0.0, std))
                for key, std in config.NOISE_STD.items()
            }

    # ------------------------------------------------------------------
    def _mix(self, x: float, y: float, centers, delta_key: str) -> float:
        total = 0.0
        for center in centers:
            d2 = (x - center.x) ** 2 + (y - center.y) ** 2
            total += (
                config.STRENGTH_DELTAS[center.strength][delta_key]
                * np.exp(-d2 / (2.0 * config.DYNAMIC_FIELD_SIGMA ** 2))
            )
        return float(total)

    def query_gps_raw(self, x: float, y: float) -> Tuple[float, float, float]:
        nsat = config.GPS_BASE_NSAT - self._mix(
            x, y, self.scenario.gps_centers, "dN"
        )
        hdop = config.GPS_BASE_HDOP + self._mix(
            x, y, self.scenario.gps_centers, "dH"
        )
        snr = config.GPS_BASE_SNR - self._mix(
            x, y, self.scenario.gps_centers, "dS"
        )
        nsat = float(np.clip(nsat, *config.GPS_CLIP[0]))
        hdop = float(np.clip(hdop, *config.GPS_CLIP[1]))
        snr = float(np.clip(snr, *config.GPS_CLIP[2]))
        return nsat, hdop, snr

    def query_visibility_raw(self, x: float, y: float) -> float:
        value = config.VISIBILITY_BASE - self._mix(
            x, y, self.scenario.visibility_centers, "dV"
        )
        return float(np.clip(value, *config.VISIBILITY_CLIP))

    def query_gps(
        self, x: float, y: float, step: int
    ) -> Tuple[float, float, float]:
        self.begin_step(step)
        nsat, hdop, snr = self.query_gps_raw(x, y)
        nsat = float(np.clip(nsat + self.noise_offsets["N"], *config.GPS_CLIP[0]))
        hdop = float(np.clip(hdop + self.noise_offsets["H"], *config.GPS_CLIP[1]))
        snr = float(np.clip(snr + self.noise_offsets["S"], *config.GPS_CLIP[2]))
        if config.ENV_EVOLUTION_ENABLED and config.ENV_EVOLVE_FOUR_SOURCES:
            p = prog_factor(self._t_seconds)
            nsat = float(np.clip(nsat - config.GPS_DEGRADE_NSAT * p, *config.GPS_CLIP[0]))
            hdop = float(np.clip(hdop + config.GPS_DEGRADE_HDOP * p, *config.GPS_CLIP[1]))
            snr = float(np.clip(snr - config.GPS_DEGRADE_SNR * p, *config.GPS_CLIP[2]))
        return nsat, hdop, snr

    def query_visibility(self, x: float, y: float, step: int) -> float:
        self.begin_step(step)
        value = self.query_visibility_raw(x, y) + self.noise_offsets["V"]
        if config.ENV_EVOLUTION_ENABLED and config.ENV_EVOLVE_FOUR_SOURCES:
            value -= config.VIS_DEGRADE * prog_factor(self._t_seconds)
        return float(np.clip(value, *config.VISIBILITY_CLIP))

    def delay(self, t: float) -> float:
        # t is in simulation steps; period is in seconds.
        t_seconds = t * config.DT
        base = self.scenario.delay_base
        if config.ENV_EVOLUTION_ENABLED and config.ENV_EVOLVE_FOUR_SOURCES:
            base = base * (1.0 + config.COMM_TAU_GROW * prog_factor(t_seconds))
        return (
            base
            + self.scenario.delay_amp
            * np.sin(2.0 * np.pi * t_seconds / self.scenario.delay_period)
        )
