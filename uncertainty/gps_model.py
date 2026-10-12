"""GPS IT2-FLS: 27-rule quality-score generator + typed max fusion.

 consequence interval [c-w, c+w] with w=0.5 m, then
delta_pos = max(per-type gps_base, interval).  All 27 input combinations
are enumerated so no rule blind spot exists.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import config
from uncertainty.membership import FuzzySet

class GPSModel:
    def __init__(self) -> None:
        self.input_sets: Dict[str, Dict[str, FuzzySet]] = {
            "N_sat": {
                "Low": FuzzySet("N_sat_Low", (0.0, 0.0, 3.0, 5.0)),
                "Med": FuzzySet("N_sat_Med", (3.0, 5.0, 7.0, 9.0)),
                "High": FuzzySet("N_sat_High", (7.0, 9.0, 12.0, 12.0)),
            },
            "HDOP": {
                "Low": FuzzySet("HDOP_Low", (0.0, 0.0, 2.0, 5.0)),
                "Med": FuzzySet("HDOP_Med", (2.0, 5.0, 8.0, 12.0)),
                "High": FuzzySet("HDOP_High", (8.0, 12.0, 30.0, 30.0)),
            },
            "SNR": {
                "Low": FuzzySet("SNR_Low", (0.0, 0.0, 10.0, 15.0)),
                "Med": FuzzySet("SNR_Med", (10.0, 15.0, 25.0, 30.0)),
                "High": FuzzySet("SNR_High", (25.0, 30.0, 40.0, 40.0)),
            },
        }
        self.quality: Dict[str, Dict[str, int]] = {
            "N_sat": {"Low": 0, "Med": 1, "High": 2},
            "HDOP": {"High": 0, "Med": 1, "Low": 2},
            "SNR": {"Low": 0, "Med": 1, "High": 2},
        }
        self.rules: List[Tuple[str, str, str]] = [
            (ns, hd, snr)
            for ns in ("Low", "Med", "High")
            for hd in ("Low", "Med", "High")
            for snr in ("Low", "Med", "High")
        ]

    # ------------------------------------------------------------------
    def infer(
        self, nsat: float, hdop: float, snr: float, uav_type: str = "A"
    ) -> Tuple[float, float]:
        """Return typed delta_pos interval [delta_min, delta_max] in meters."""
        f_lower_sum = 0.0
        f_upper_sum = 0.0
        num_lower = 0.0
        num_upper = 0.0
        best_upper = -1.0
        best_centroid = config.GPS_Q_CENTROIDS[0]

        for ns, hd, sn in self.rules:
            q = (
                self.quality["N_sat"][ns]
                + self.quality["HDOP"][hd]
                + self.quality["SNR"][sn]
            )
            centroid = config.GPS_Q_CENTROIDS[q]
            f_lower = min(
                self.input_sets["N_sat"][ns].lower(nsat),
                self.input_sets["HDOP"][hd].lower(hdop),
                self.input_sets["SNR"][sn].lower(snr),
            )
            f_upper = min(
                self.input_sets["N_sat"][ns].upper(nsat),
                self.input_sets["HDOP"][hd].upper(hdop),
                self.input_sets["SNR"][sn].upper(snr),
            )
            f_lower_sum += f_lower
            f_upper_sum += f_upper
            num_lower += f_lower * (centroid - config.GPS_RULES_W)
            num_upper += f_upper * (centroid + config.GPS_RULES_W)
            if f_upper > best_upper:
                best_upper = f_upper
                best_centroid = centroid

        if config.UNCERTAINTY_VARIANT == "t1fls":
            # Type-1 FLS: crisp centroid defuzzification. The IT2 interval (and
            # its conservative +w upper bound) collapses to a single point.
            if f_upper_sum >= config.EPS_SAFE:
                delta = num_upper / f_upper_sum - config.GPS_RULES_W
            else:
                delta = best_centroid
            base = config.UAV_TYPES[uav_type].gps_base
            delta = max(base, delta)
            return float(delta), float(delta)

        if f_lower_sum < config.EPS_SAFE or f_upper_sum < config.EPS_SAFE:
            delta_min = best_centroid - config.GPS_RULES_W
            delta_max = best_centroid + config.GPS_RULES_W
        else:
            delta_min = num_lower / f_lower_sum
            delta_max = num_upper / f_upper_sum

        if delta_min > delta_max:
            delta_min, delta_max = delta_max, delta_min

        base = config.UAV_TYPES[uav_type].gps_base
        delta_min = max(base, delta_min)
        delta_max = max(base, delta_max)
        if delta_min > delta_max:
            delta_min, delta_max = delta_max, delta_min
        return float(delta_min), float(delta_max)

    def get_delta_pos_max(
        self, nsat: float, hdop: float, snr: float, uav_type: str
    ) -> float:
        return self.infer(nsat, hdop, snr, uav_type)[1]
