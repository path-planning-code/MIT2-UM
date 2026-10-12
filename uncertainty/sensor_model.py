"""Sensor ranging IT2-FLS: 9 rules with physically non-negative output."""

from __future__ import annotations

from typing import Dict, List, Tuple

import config
from uncertainty.membership import FuzzySet

class SensorModel:
    def __init__(self) -> None:
        self.visibility_sets: Dict[str, FuzzySet] = {
            "Low": FuzzySet("vis_Low", (0.0, 0.0, 0.3, 0.5)),
            "Med": FuzzySet("vis_Med", (0.3, 0.5, 0.7, 0.9)),
            "High": FuzzySet("vis_High", (0.7, 0.9, 1.0, 1.0)),
        }
        self.range_sets: Dict[str, FuzzySet] = {
            "Near": FuzzySet("range_Near", (0.0, 0.0, 20.0, 40.0)),
            "Med": FuzzySet("range_Med", (30.0, 50.0, 80.0, 100.0)),
            "Far": FuzzySet("range_Far", (80.0, 100.0, 200.0, 200.0)),
        }
        #  order: visibility Low/Med/High x range Far/Med/Near.
        self.rules: List[Tuple[str, str, float]] = [
            ("Low", "Far", 4.5),
            ("Low", "Med", 3.0),
            ("Low", "Near", 1.5),
            ("Med", "Far", 2.5),
            ("Med", "Med", 1.5),
            ("Med", "Near", 0.8),
            ("High", "Far", 1.0),
            ("High", "Med", 0.5),
            ("High", "Near", 0.2),
        ]

    def infer(self, visibility: float, range_m: float) -> Tuple[float, float]:
        """Return [delta_min, delta_max] ranging error in meters."""
        range_m = float(min(max(range_m, 0.0), config.SENSOR_RANGE_MAX))
        f_lower_sum = 0.0
        f_upper_sum = 0.0
        num_lower = 0.0
        num_upper = 0.0
        best_upper = -1.0
        best_centroid = self.rules[0][2]

        for vis_label, range_label, centroid in self.rules:
            f_lower = min(
                self.visibility_sets[vis_label].lower(visibility),
                self.range_sets[range_label].lower(range_m),
            )
            f_upper = min(
                self.visibility_sets[vis_label].upper(visibility),
                self.range_sets[range_label].upper(range_m),
            )
            f_lower_sum += f_lower
            f_upper_sum += f_upper
            num_lower += f_lower * (centroid - config.SENSOR_RULES_W)
            num_upper += f_upper * (centroid + config.SENSOR_RULES_W)
            if f_upper > best_upper:
                best_upper = f_upper
                best_centroid = centroid

        if config.UNCERTAINTY_VARIANT == "t1fls":
            # Type-1 FLS: crisp centroid defuzzification (no IT2 interval).
            if f_upper_sum >= config.EPS_SAFE:
                delta = num_upper / f_upper_sum - config.SENSOR_RULES_W
            else:
                delta = best_centroid
            delta = max(0.0, delta)
            return float(delta), float(delta)

        if f_lower_sum < config.EPS_SAFE or f_upper_sum < config.EPS_SAFE:
            delta_min = best_centroid - config.SENSOR_RULES_W
            delta_max = best_centroid + config.SENSOR_RULES_W
        else:
            delta_min = num_lower / f_lower_sum
            delta_max = num_upper / f_upper_sum

        delta_min = max(0.0, delta_min)
        if delta_min > delta_max:
            delta_min, delta_max = delta_max, delta_min
        return float(delta_min), float(delta_max)

    def get_delta_max(self, visibility: float, range_m: float) -> float:
        return self.infer(visibility, range_m)[1]
