"""Obstacle trajectory prediction uncertainty (analytic model)."""

from __future__ import annotations

import config
from uncertainty.dynamic_field import prog_factor

def delta_obs(obstacle, horizon: int, dt: float = config.DT, t: float = 0.0) -> float:
    """ delta_obs = 2.0 + beta * v_obs * horizon * dt.

    Uses obstacle.speed (current speed) rather than params.speed (type
    baseline), so HIGH_MANEUVER obstacles with perturbed velocity get the
    correct prediction uncertainty.  Over the mission the diffusion rate
    ``beta_k`` grows (trajectories become less predictable), via the shared
    progression factor evaluated at elapsed seconds ``t``.
    """
    beta = obstacle.params.beta
    if config.ENV_EVOLUTION_ENABLED and config.ENV_EVOLVE_FOUR_SOURCES:
        beta = beta * (1.0 + config.OBS_BETA_GROW * prog_factor(t))
    return (
        config.OBS_DELTA_INIT
        + beta * obstacle.speed * horizon * dt
    )
