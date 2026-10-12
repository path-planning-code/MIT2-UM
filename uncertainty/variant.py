"""Uncertainty-modelling ablation variants (dedicated experiment).

Central registry plus the runtime switches used across the uncertainty stack.
Every switch is read at query time (not import time), so one process can run
several variants back-to-back by reassigning ``config.UNCERTAINTY_VARIANT`` and
calling :func:`apply` once per episode.
"""

from __future__ import annotations

import config

VARIANTS: tuple = ("full", "deterministic", "obstacle_only", "t1fls", "no_urf")

_FOU_DEFAULT: float = 0.6
_FOU_T1: float = 1.0

TITLES = {
    "full": "Full (MIT2-UM + URF)",
    "deterministic": "Deterministic",
    "obstacle_only": "Obstacle-Only",
    "t1fls": "T1-FLS",
    "no_urf": "No-URF",
}

def current() -> str:
    v = config.UNCERTAINTY_VARIANT
    if v not in VARIANTS:
        raise ValueError(f"unknown UNCERTAINTY_VARIANT: {v!r}")
    return v

def apply() -> None:
    """Apply variant-wide config knock-ons before an episode runs.

    ``t1fls`` collapses the IT2 footprint of uncertainty (FOU) to zero by
    driving the lower-membership shrink factor to 1.0, i.e. type-1 fuzzy logic.
    """
    if current() == "t1fls":
        config.LOWER_MF_FACTOR = _FOU_T1
    else:
        config.LOWER_MF_FACTOR = _FOU_DEFAULT

def gps_sens_comm_on() -> bool:
    """GPS / sensor / comm uncertainty sources are active."""
    return current() not in ("deterministic", "obstacle_only")

def obstacle_uncertainty_on() -> bool:
    """Obstacle-motion prediction uncertainty is active."""
    return current() != "deterministic"

def terrain_uncertainty_on() -> bool:
    """Terrain elevation uncertainty interval (h_high vs h_true) is active."""
    return current() not in ("deterministic", "obstacle_only")