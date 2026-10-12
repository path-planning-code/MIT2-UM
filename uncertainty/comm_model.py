"""Communication delay uncertainty (analytic model)."""

from __future__ import annotations

from typing import Sequence

import config

class CommModel:
    def tau_comm(self, delay: float) -> float:
        return delay + config.COMM_T_LOSS

    def delta_comm_pair(self, v_j: float, delay: float) -> float:
        return v_j * self.tau_comm(delay)

    def delta_comm_max(
        self,
        uavs: Sequence,
        idx: int,
        delay: float,
        mode: str | None = None,
    ) -> float:
        """Per-UAV communication staleness used by u_comm / r_comm.

         writes delta_comm_max(i) = v_i * tau in the snapshot content and
        delta_comm(i,j) = v_j * tau in the model definition.  The default
        config mode "max_others" uses the fastest other active UAV, which is
        consistent with the pair-level formula; "own_speed" follows the
        snapshot text literally.
        """
        mode = config.COMM_MAX_MODE if mode is None else mode
        tau = self.tau_comm(delay)
        if mode == "own_speed":
            return uavs[idx].v * tau
        speeds = [
            uavs[j].v
            for j in range(len(uavs))
            if j != idx and uavs[j].active
        ]
        return max(speeds, default=0.0) * tau
