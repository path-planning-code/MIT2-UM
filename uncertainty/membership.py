"""Trapezoid membership functions with the  upper/lower FOU structure."""

from __future__ import annotations

from typing import Tuple

import config

def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid membership with correct left/right shoulder endpoints."""
    if x <= a:
        return 1.0 if a == b else 0.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    if x <= c:
        return 1.0
    if x < d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0

class FuzzySet:
    """A fuzzy set with upper membership and scaled lower membership."""

    def __init__(
        self,
        name: str,
        params: Tuple[float, float, float, float],
        lower_factor: float = config.LOWER_MF_FACTOR,
    ) -> None:
        self.name = name
        self.params = params
        self.lower_factor = lower_factor

    def upper(self, x: float) -> float:
        return trapezoid(x, *self.params)

    def lower(self, x: float) -> float:
        # Read the FOU shrink factor at query time so the T1-FLS ablation can
        # collapse the IT2 footprint of uncertainty to zero at runtime.
        return config.LOWER_MF_FACTOR * self.upper(x)
