# MIT2-UM and URF Reference Implementation

This repository contains the reference implementation of the **Multi-modal
Interval Type-2 Uncertainty Model (MIT2-UM)** and the **Unified Risk Field
(URF)** used by our two-stage adaptive game-theoretic UAV swarm planner. It
corresponds to Section 3.3 (system modelling) of the manuscript. Every
algorithm file is the exact code executed by the experiments in the paper;
only internal development tags and machine-specific paths were removed.

## What is inside

```
.
├── config.py                  # all model constants and the 12 scenarios
├── uncertainty/               # MIT2-UM five-source modelling and URF
│   ├── dynamic_field.py       # 3.3.1 Gaussian-mixture spatio-temporal field
│   ├── membership.py          # trapezoidal upper/lower memberships (FOU)
│   ├── gps_model.py           # 3.3.3 GPS localization IT2-FLS (27 rules)
│   ├── sensor_model.py        # 3.3.5 ranging IT2-FLS (9 rules)
│   ├── comm_model.py          # 3.3.6 delay / loss / position staleness
│   ├── obstacle_pred.py       # 3.3.4 analytic prediction diffusion
│   ├── snapshot.py            # 3.3.7 per-step frozen uncertainty snapshot
│   ├── terrain_model.py       # terrain uncertainty interface
│   ├── fdtm_adapter.py        # 3.3.2 hybrid epicenter + TSI-TM interval
│   ├── risk_field.py          # 3.3.8 URF two-layer T-S fusion
│   └── variant.py             # ablation switches (Section 7.4)
├── id_it2_fdtm/               # iteration-driven interval type-2 fuzzy
│   └── id_it2_fdtm.py         # dynamic terrain model (slope vulnerability,
│                              # three-phase reduction, interval bounds)
├── environment/               # minimal simulation substrate
│   ├── terrain.py             # synthetic Gaussian peaks and real-DEM terrain
│   ├── dem_loader.py          # dependency-free SRTM GeoTIFF reader
│   └── obstacle.py            # FIXED / RANDOM / HIGH_MANEUVER obstacles
└── demo/
    ├── visualize_mit2um.py    # reviewer-facing six-figure walkthrough
    └── figures/               # generated figures
```

## Paper-to-code correspondence

| Manuscript | Implementation |
|---|---|
| Dynamic field, MED/BAD offsets, frozen noise, monotonic degradation | `uncertainty/dynamic_field.py`, constants in `config.py` |
| Epicenter Gaussian error band, TSI-TM settlement, interval union | `environment/terrain.py`, `id_it2_fdtm/`, `uncertainty/fdtm_adapter.py` |
| GPS IT2-FLS, 27 quality-score rules, typed hardware baseline | `uncertainty/gps_model.py`, `uncertainty/membership.py` |
| Obstacle diffusion with two prediction horizons | `uncertainty/obstacle_pred.py` |
| Sensor ranging IT2-FLS, nine rules, non-negative output | `uncertainty/sensor_model.py` |
| Sinusoidal delay, retransmission, peer-speed staleness | `uncertainty/comm_model.py` |
| Step-wise frozen snapshot and deterministic spatial queries | `uncertainty/snapshot.py` |
| First-layer single-source risks, five T-S rules, R_min / R_max | `uncertainty/risk_field.py` |
| Full / Deterministic / Obstacle-Only / T1-FLS / No-URF ablations | `uncertainty/variant.py` |

## Quick start

The code needs only NumPy and Matplotlib.

```bash
pip install -r requirements.txt
python demo/visualize_mit2um.py
```

Six figures are written to `demo/figures/`, covering the terrain interval,
the dynamic interference fields, both interval type-2 fuzzy systems, the
obstacle diffusion, the communication delay and the full two-layer URF on a
fixed flight plane. The standalone fuzzy-terrain walkthrough can be run with
`python -m id_it2_fdtm.demo`.

To switch an uncertainty ablation, set `UNCERTAINTY_VARIANT` in `config.py`
to one of `full`, `deterministic`, `obstacle_only`, `t1fls`, `no_urf`. Real
SRTM terrain additionally requires the GeoTIFF tile described in
`data/README.txt`.
