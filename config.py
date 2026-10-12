"""Centralized configuration for the  two-stage framework.

All numeric tables in this module are transcribed from framework-.
Later modules (environment, uncertainty, game, learning, experiments)
must import from here instead of hard-coding values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Time, world and episode
# ---------------------------------------------------------------------------
DT: float = 0.5                       # control period (s)
EPISODE_MAX_STEPS: int = 500          # 250 s cap (widened so slow C-type UAVs can finish the diagonal)
K: int = 10                           # PPO weight window (env step = K sim steps, 5 s)
H_BRD: int = 1                        # one-step decision horizon
H_META: int = 20                      # 10 s macro risk horizon
M_MAX: int = 10                       # BRD max sweeps per group

# BRD action hysteresis: suppress frequent action switching to reduce
# trajectory oscillation.  A new action is only adopted when its utility
# gain over the previous action exceeds the relative threshold.
BRD_HYSTERESIS_ENABLED: bool = True
BRD_HYSTERESIS_THRESHOLD: float = 0.02  # 2% relative utility gain required to switch

# BRD action cooldown: after switching to a new action, the UAV must hold
# it for at least COOLDOWN_STEPS simulation steps before switching again.
# This is a stronger guard against oscillation than hysteresis alone: even
# if a different action looks better, the UAV sticks with its recent choice
# long enough for the trajectory to settle.
BRD_COOLDOWN_ENABLED: bool = True
BRD_COOLDOWN_STEPS: int = 5   # 2.5 s cooldown at DT=0.5 (shorter=less zigzag)

WORLD_X_MIN, WORLD_X_MAX = 0.0, 100.0
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 100.0
WORLD_Z_MIN, WORLD_Z_MAX = 0.0, 80.0
WORLD_GRID: float = 2.0               # terrain grid resolution (m)

ARRIVAL_H_TOL: float = 2.0            # horizontal arrival tolerance (m)
ARRIVAL_V_TOL: float = 1.0            # vertical arrival tolerance (m)
GOAL_CLEARANCE: float = 0.0           # h_true + clearance < z_goal
START_MIN_SEP: float = 5.0            # min inter-UAV start distance
GOAL_MIN_SEP: float = 0.0             # min inter-UAV goal distance
START_GOAL_MIN_DIST: float = 50.0     # min start-goal distance for each UAV
INIT_Z_OFFSET: float = 10.0           # z_init = z_ref + offset
INIT_V: float = 0.8  # initial speed as fraction of v_max (0.8 = 80%), matching old framework
INIT_PSI: float = 0.0
INIT_GAMMA: float = 0.0  # start level (old framework: gamma=0.0) — start with climb attitude
EPS_SAFE: float = 1e-6                # division guard

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
N_ACTIONS: int = 7
# Discrete action space (strictly follows paper Table 1):
# a0=悬停, a1=左转, a2=右转, a3=爬升, a4=下降, a5=加速, a6=减速
HOVER, TURN_LEFT, TURN_RIGHT, CLIMB, DESCEND, ACCELERATE, DECELERATE = range(7)
ACTION_NAMES: Tuple[str, ...] = (
    "hover", "accelerate", "decelerate", "turn_left", "turn_right",
    "climb", "descend",
)

# ---------------------------------------------------------------------------
# UAV types 
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UAVTypeConfig:
    key: str
    name: str
    v_max: float          # m/s
    d_safe: float         # m
    delta_psi_ref: float  # deg/step (action increment)
    delta_gamma_ref: float  # deg/step (action increment)
    gamma_max: float      # deg (maximum climb angle)
    z_ref: float          # m (cruise/reference altitude)
    gps_base: float       # m
    sense_radius: float   # m
    a_max: float = 2.0          # m/s^2 (max acceleration)
    turn_rate_max: float = 30.0  # deg/s (max turn rate)
    climb_rate_max: float = 15.0 # deg/s (max climb rate)

    @property
    def dv(self) -> float:
        return 0.15 * self.v_max

    @property
    def dpsi(self) -> float:
        return self.delta_psi_ref

    @property
    def dgamma(self) -> float:
        return self.delta_gamma_ref

    @property
    def dpsi_rad(self) -> float:
        return math.radians(self.dpsi)

    @property
    def dgamma_rad(self) -> float:
        return math.radians(self.dgamma)

    @property
    def gamma_max_rad(self) -> float:
        return math.radians(self.gamma_max)

    @property
    def turn_rate_max_rad(self) -> float:
        """Max turn rate per step (rad/step) = deg/s * DT"""
        return math.radians(self.turn_rate_max) * DT

    @property
    def climb_rate_max_rad(self) -> float:
        """Max climb rate per step (rad/step) = deg/s * DT"""
        return math.radians(self.climb_rate_max) * DT

    @property
    def dv_max_step(self) -> float:
        """Max speed change per step (m/s) = a_max * DT"""
        return self.a_max * DT

UAV_TYPES: Dict[str, UAVTypeConfig] = {
    # Action increments (strictly follows paper Table 1):
    #   A型: Δψ=5°, Δγ=7.5°, Δv=0.75 m/s
    #   B型: Δψ=3.5°, Δγ=3.75°, Δv=0.45 m/s
    #   C型: Δψ=1.75°, Δγ=2.25°, Δv=0.3 m/s
    # d_safe kept at 2 m for all types.
    # gamma_max: maximum climb angle (degrees)
    "A": UAVTypeConfig("A", "recon", 5.0, 2.0, 5.0, 7.5, 30.0, 30.0, 1.0, 25.0),
    "B": UAVTypeConfig("B", "logistics", 3.0, 2.0, 3.5, 3.75, 25.0, 25.0, 3.0, 20.0),
    "C": UAVTypeConfig("C", "relay", 2.0, 2.0, 1.75, 2.25, 20.0, 20.0, 5.0, 15.0),
}

# ---------------------------------------------------------------------------
# Obstacle types ()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObstacleTypeConfig:
    key: str
    radius: float   # m
    speed: float    # m/s
    beta: float     # uncertainty diffusion rate

OBSTACLE_TYPES: Dict[str, ObstacleTypeConfig] = {
    "FIXED": ObstacleTypeConfig("FIXED", 1.5, 5.0, 0.05),
    "RANDOM": ObstacleTypeConfig("RANDOM", 2.0, 8.0, 0.15),
    "HIGH_MANEUVER": ObstacleTypeConfig("HIGH_MANEUVER", 2.5, 12.0, 0.30),
}

# ---------------------------------------------------------------------------
# Dynamic uncertainty fields ()
# ---------------------------------------------------------------------------
DYNAMIC_FIELD_SIGMA: float = 15.0
GPS_BASE_NSAT, GPS_BASE_HDOP, GPS_BASE_SNR = 10.0, 1.5, 32.0
VISIBILITY_BASE: float = 0.95

STRENGTH_DELTAS: Dict[str, Dict[str, float]] = {
    "MED": {"dN": 3.0, "dH": 3.5, "dS": 7.0, "dV": 0.25},
    "BAD": {"dN": 7.0, "dH": 13.5, "dS": 22.0, "dV": 0.55},
}

NOISE_STD: Dict[str, float] = {
    "N": 0.3,
    "H": 0.2,
    "S": 1.0,
    "V": 0.02,
}

GPS_CLIP: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (0.0, 12.0),
    (0.5, 30.0),
    (0.0, 40.0),
)
VISIBILITY_CLIP: Tuple[float, float] = (0.0, 1.0)

# ---------------------------------------------------------------------------
# Uncertainty model parameters ()
# ---------------------------------------------------------------------------
LOWER_MF_FACTOR: float = 0.6

# GPS IT2-FLS
GPS_RULES_W: float = 0.5              # consequence interval half-width (m)
GPS_Q_CENTROIDS: Tuple[float, ...] = (12.0, 10.0, 8.0, 6.0, 4.0, 2.5, 1.5)

# Sensor IT2-FLS
SENSOR_RULES_W: float = 0.2
SENSOR_CENTROIDS: Tuple[float, ...] = (
    4.5, 3.0, 1.5, 2.5, 1.5, 0.8, 1.0, 0.5, 0.2,
)
SENSOR_RANGE_MAX: float = 50.0        # clamp distance input to sensor model (scaled for 100x100 world)

# Obstacle prediction
OBS_DELTA_INIT: float = 2.0           # initial perception error (m)

# Communication
COMM_LOSS_PROB: float = 0.1
COMM_RETRANSMIT_TIME: float = 0.5     # s
COMM_T_LOSS: float = COMM_LOSS_PROB * COMM_RETRANSMIT_TIME

# Terrain uncertainty
TERRAIN_ERR_BASE: float = 0.5
TERRAIN_ERR_AMP: float = 3.0
TERRAIN_ERR_SIGMA: float = 30.0

# Real DEM integration (SRTM-like GeoTIFF)
REAL_DEM_PATH: str = r"data/n29_e102_1arc_v3.tif"  # place the SRTM GeoTIFF tile here
DEM_CROP_CENTER: Tuple[float, float] = (29.32, 102.66)  # (lat, lon)
DEM_CROP_SIZE_M: float = 1000.0
DEM_NODATA_VALUES: Tuple[int, ...] = (32769,)
# "normalize": relative + linear scale to DEM_MAX_RELIEF
# "detrend":   subtract crop minimum, no scaling
# "absolute":  keep real DEM elevations as-is
DEM_ELEVATION_MODE: str = "normalize"
DEM_MAX_RELIEF: float = 80.0

# Post-quake landslide perturbation ( 3-5 Gaussian landslides,
# radius 10-30 m, elevation change in [-3, +3] m).
# Scaled down 10x from original 1000x1000 world.
DEM_LANDSLIDE_ENABLED: bool = True
DEM_LANDSLIDE_COUNT_RANGE: Tuple[int, int] = (3, 5)
DEM_LANDSLIDE_RADIUS_RANGE: Tuple[float, float] = (10.0, 30.0)
DEM_LANDSLIDE_DH_RANGE: Tuple[float, float] = (-3.0, 3.0)

# ID-IT2-FDTM integration
USE_FDTM: bool = True
# "epicenter_delta": h_true +/- delta_terr (, epicenter Gaussian only)
# "fdtm":      ID-IT2-FDTM h_L/h_U (multiplicative alpha bounds, slope-dependent)
# "hybrid":    union of both intervals (most conservative, uses both models)
TERRAIN_INTERVAL_MODE: str = "hybrid"
TERRAIN_EVOLVE_ITERS: int = 80      # evolve 80 iterations before episode (mid-late stage degradation)
FDTM_ALPHA: float = 0.15
FDTM_T1: int = 30
FDTM_T2: int = 60
FDTM_T_UPDATE: int = 10
FDTM_S_MAX: float = 1.0
FDTM_GRID_SIZE: int = 100           # 100 m / 100 = 1 m grid spacing

# ---------------------------------------------------------------------------
# Environment evolution (dynamic environment,  dynamic-terrain extension)
# ---------------------------------------------------------------------------
# Master switch: when False, the whole world is frozen for the episode (the
# original static baseline); when True, the five uncertainty sources evolve in
# flight — the terrain settles down (P1) and the GPS / sensor / obstacle / comm
# sources degrade up (P3), gated individually below.
ENV_EVOLUTION_ENABLED: bool = True

# Post-quake terrain settling: the physical surface subsides monotonically.
#   h(t) = h0 * (1 - SETTLE_FRAC * (1 - exp(-t / SETTLE_TAU)))
# Fast in the early aftershock phase, slow as the ground stabilizes (same
# direction as the fuzzy degradation in Table 2, where r decays over phases).
# The curve is shared by every peak / cell, so all peaks sink together.
TERRAIN_SETTLE_FRAC: float = 0.32   # total fractional subsidence (32%)
TERRAIN_SETTLE_TAU: float = 80.0    # seconds; smaller settles sooner

# Four-source evolution (P3).  Unlike the terrain, these uncertainties GROW
# over the mission clock (post-disaster conditions keep worsening).  Each one
# rides a shared monotonic progression   prog(t) = 1 - exp(-t / TAU)  (0 -> 1),
# so the degradation is fast early and saturates late, and every step stays
# deterministic (no new randomness, reproducible).
ENV_EVOLVE_FOUR_SOURCES: bool = True
FOUR_SRC_GROW_TAU: float = 120.0    # seconds; shared progression time constant

GPS_DEGRADE_NSAT: float = 4.0       # satellites lost over the mission (max)
GPS_DEGRADE_HDOP: float = 5.0       # HDOP worsening over the mission (max)
GPS_DEGRADE_SNR: float = 8.0        # SNR loss in dB over the mission (max)
VIS_DEGRADE: float = 0.40           # visibility drop (smoke/dust accumulation)
OBS_BETA_GROW: float = 0.5          # fractional growth of obstacle diffusion beta_k
COMM_TAU_GROW: float = 0.5          # fractional growth of communication base delay

# ---------------------------------------------------------------------------
# Uncertainty-modelling ablation variants (dedicated experiment)
# ---------------------------------------------------------------------------
# Switches the online uncertainty model while keeping the offline PSO global
# path, the potential-game replanner and the PPO meta-controller identical.
#   "full"          - MIT2-UM (5 sources, IT2) + URF two-layer T-S fusion
#   "deterministic" - every uncertainty magnitude set to nominal (0)
#   "obstacle_only" - only the obstacle-motion uncertainty is retained
#   "t1fls"         - IT2 footprint of uncertainty collapsed (type-1 fuzzy)
#   "no_urf"        - URF T-S fusion replaced by equal-weight averaging
UNCERTAINTY_VARIANT: str = "full"

# ---------------------------------------------------------------------------
# URF ()
# ---------------------------------------------------------------------------
RISK_DENOM_GPS: float = 15.0
RISK_OBS_SCALE: float = 30.0
RISK_DENOM_SENS: float = 5.0
RISK_DENOM_COMM: float = 10.0

META_DENOM: Dict[str, float] = {
    "u_terr": 1.0,
    "u_gps": 15.0,
    "u_obs": 40.0,
    "u_sens": 5.0,
    "u_comm": 10.0,
}

# ---------------------------------------------------------------------------
# Meta controller ()
# ---------------------------------------------------------------------------
META_OBS_DIM: int = 8
META_ACT_DIM: int = 6
WEIGHT_NAMES: Tuple[str, ...] = ("w_p", "w_c", "w_r", "w_s", "w_a", "w_rf")
#  initial weights (six-term utility around reference path):
#   w_p  — path deviation (primary tracking objective)
#   w_c  — collision repulsion (safety constraint)
#   w_r  — rejoin incentive (auxiliary guidance back to path)
#   w_s  — action smoothness (trajectory quality)
#   w_a  — altitude tracking (vertical profile following)
#   w_rf — unified risk field (uncertainty-aware safety)
FIXED_WEIGHTS: Tuple[float, ...] = (0.30, 0.25, 0.15, 0.05, 0.10, 0.15)

NET_HIDDEN: Tuple[int, ...] = (64, 64)

# ---------------------------------------------------------------------------
# Reward ()
# ---------------------------------------------------------------------------
REWARD_CRASH: float = -100.0
REWARD_UAV_UAV: float = -5.0
REWARD_DIST: float = -0.1
REWARD_SPEED: float = -0.01
REWARD_ARRIVE: float = 10.0
TARGET_SPEED_RATIO: float = 0.6

# ---------------------------------------------------------------------------
# Meta-learning reward ()
# ---------------------------------------------------------------------------
META_R_SAFETY: float = -5.0     # per collision (dominant outcome penalty)
META_R_PROGRESS: float = 0.0        # removed: monotonic projection made this weight-independent
META_R_TRACK: float = 0.001          # per meter path deviation (light shaping, keep outcome aligned)
META_R_SMOOTH: float = 0.001         # per unit weight change (small auxiliary term)
META_R_ARRIVAL: float = 5.0      # per UAV arrival (dominant outcome signal)
META_R_SURVIVAL: float = 0.0       # per step per active UAV (reward for staying alive)

# ---------------------------------------------------------------------------
# Training ()
# ---------------------------------------------------------------------------
TOTAL_EPISODES: int = 5000
TOTAL_ENV_STEPS: int = 75000          # = 5000 episodes x 15 env steps (legacy, K=20 now)
SB3_N_STEPS: int = 256
SB3_N_ENVS: int = 9
SB3_UPDATE_BATCH: int = SB3_N_STEPS * SB3_N_ENVS

PPO_LR: float = 3e-4
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_CLIP_RANGE: float = 0.2
PPO_N_EPOCHS: int = 10
PPO_BATCH_SIZE: int = 64
PPO_ENT_COEF: float = 0.001
PPO_SEED: int = 42
PPO_SAVE_PATH: str = "models/ppo_proposed"
PPO_SMOKE_SAVE_PATH: str = "models/ppo_proposed_smoke"
PPO_CHECKPOINT_DIR: str = "checkpoints/ppo_proposed"
PPO_CHECKPOINT_FREQ: int = 4096
PPO_TENSORBOARD_DIR: str = "runs/ppo_proposed"
SMOKE_ENV_STEPS: int = 512
SMOKE_N_ENVS: int = 2

B4_TOTAL_EPISODES: int = 5000
B4_OBS_DIM: int = 18
B4_PER_SIM_STEP: bool = True

# ---------------------------------------------------------------------------
# Scenario tables ( and 10.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldCenter:
    x: float
    y: float
    strength: str  # "MED" or "BAD"

@dataclass(frozen=True)
class ScenarioConfig:
    key: str
    terrain: str                      # "10peak" | "30peak" | "real_dem"
    formation: Tuple[str, ...]
    gps_centers: Tuple[FieldCenter, ...]
    visibility_centers: Tuple[FieldCenter, ...]
    delay_base: float
    delay_amp: float
    delay_period: float
    obstacle_composition: Tuple[Tuple[str, int], ...]
    use: str                          # "train" | "generalization"
    epicenter: Tuple[float, float] = (50.0, 50.0)
    real_dem_path: str = ""

@dataclass(frozen=True)
class ScenarioOverrides:
    """Parameter-layer overrides for building custom experiment scenes.

    All fields are optional; None keeps the  default behavior.
    """

    use_real_dem: bool | None = None
    real_dem_path: str | None = None
    dem_crop_center: Tuple[float, float] | None = None
    use_fdtm: bool | None = None
    terrain_evolve_iters: int | None = None
    n_peaks: int | None = None
    obstacle_composition: Dict[str, int] | None = None
    starts: Tuple[Tuple[float, float, float], ...] | None = None
    goals: Tuple[Tuple[float, float, float], ...] | None = None

SCENARIOS: Tuple[ScenarioConfig, ...] = (
    ScenarioConfig(
        "S1", "10peak", ("A", "A", "A", "A", "A"),
        (FieldCenter(30, 30, "MED"),),
        (FieldCenter(50, 50, "MED"),),
        0.05, 0.02, 60.0,
        (("FIXED", 3), ("RANDOM", 4)), "train",
    ),
    ScenarioConfig(
        "S2", "10peak", ("A", "A", "A", "B", "C"),
        (FieldCenter(30, 30, "MED"),),
        (FieldCenter(50, 50, "MED"),),
        0.05, 0.02, 60.0,
        (("FIXED", 3), ("RANDOM", 4)), "train",
    ),
    ScenarioConfig(
        "S3", "10peak", ("A", "A", "A", "B", "C"),
        (FieldCenter(30, 30, "BAD"),),
        (FieldCenter(50, 50, "BAD"),),
        0.10, 0.05, 50.0,
        (("FIXED", 2), ("RANDOM", 8), ("HIGH_MANEUVER", 3)), "train",
    ),
    ScenarioConfig(
        "S4", "10peak", ("A", "A", "B", "B", "C"),
        (FieldCenter(30, 30, "BAD"), FieldCenter(60, 20, "BAD")),
        (FieldCenter(40, 60, "BAD"),),
        0.15, 0.08, 40.0,
        (("FIXED", 3), ("RANDOM", 10), ("HIGH_MANEUVER", 5)), "train",
    ),
    ScenarioConfig(
        "S5", "30peak", ("A", "A", "A", "B", "C"),
        (FieldCenter(30, 30, "BAD"),),
        (FieldCenter(50, 50, "BAD"),),
        0.10, 0.05, 50.0,
        (("FIXED", 3), ("RANDOM", 5)), "train",
    ),
    ScenarioConfig(
        "S6", "30peak", ("A", "A", "B", "B", "C"),
        (FieldCenter(30, 30, "BAD"), FieldCenter(60, 20, "BAD")),
        (FieldCenter(40, 60, "BAD"),),
        0.15, 0.08, 40.0,
        (("FIXED", 3), ("RANDOM", 10), ("HIGH_MANEUVER", 5)), "train",
    ),
    ScenarioConfig(
        "S7", "real_dem", ("A", "A", "A", "B", "C"),
        (FieldCenter(30, 30, "BAD"),),
        (FieldCenter(50, 50, "MED"),),
        0.10, 0.05, 50.0,
        (("FIXED", 2), ("RANDOM", 3)), "generalization",
    ),
    ScenarioConfig(
        "S8", "real_dem", ("A", "A", "B", "C", "C"),
        (FieldCenter(30, 30, "BAD"), FieldCenter(60, 20, "BAD")),
        (FieldCenter(40, 60, "BAD"),),
        0.20, 0.10, 35.0,
        (("FIXED", 2), ("RANDOM", 4), ("HIGH_MANEUVER", 4)), "generalization",
    ),
    # --- Additional scenarios (S9-S12) for richer evaluation ---
    ScenarioConfig(
        "S9", "30peak", ("A", "A", "B", "B", "C"),
        (FieldCenter(25, 75, "BAD"), FieldCenter(75, 25, "BAD")),
        (FieldCenter(50, 50, "MED"),),
        0.12, 0.06, 45.0,
        (("FIXED", 5), ("RANDOM", 12), ("HIGH_MANEUVER", 8)), "train",
    ),
    ScenarioConfig(
        "S10", "30peak", ("A", "A", "B", "B", "C"),
        (FieldCenter(20, 20, "BAD"), FieldCenter(80, 20, "BAD"),
         FieldCenter(20, 80, "BAD"), FieldCenter(80, 80, "BAD")),
        (FieldCenter(30, 50, "BAD"), FieldCenter(70, 50, "BAD"),
         FieldCenter(50, 30, "BAD")),
        0.18, 0.10, 30.0,
        (("FIXED", 3), ("RANDOM", 10), ("HIGH_MANEUVER", 6)), "train",
    ),
    ScenarioConfig(
        "S11", "10peak", ("A", "A", "A", "A", "B", "B", "C", "C"),
        (FieldCenter(50, 50, "MED"),),
        (FieldCenter(50, 50, "MED"),),
        0.08, 0.04, 55.0,
        (("FIXED", 4), ("RANDOM", 10)), "train",
    ),
    ScenarioConfig(
        "S12", "30peak", ("A", "B", "B", "C", "C"),
        (FieldCenter(20, 30, "BAD"), FieldCenter(60, 70, "BAD"),
         FieldCenter(80, 20, "BAD")),
        (FieldCenter(40, 50, "BAD"), FieldCenter(70, 40, "BAD")),
        0.22, 0.12, 25.0,
        (("FIXED", 4), ("RANDOM", 6), ("HIGH_MANEUVER", 5)), "generalization",
    ),
)

SCENARIO_BY_KEY: Dict[str, ScenarioConfig] = {s.key: s for s in SCENARIOS}
TRAIN_SCENARIO_KEYS: Tuple[str, ...] = tuple(s.key for s in SCENARIOS if s.use == "train")
GENERALIZATION_SCENARIO_KEYS: Tuple[str, ...] = tuple(
    s.key for s in SCENARIOS if s.use == "generalization"
)

SCENARIO_BASE_SEEDS: Dict[str, int] = {
    s.key: 1000 + i for i, s in enumerate(SCENARIOS, start=1)
}

# ---------------------------------------------------------------------------
# Preset starts/goals.
# All UAVs start along the near edge from (1,1), spaced 10 m along +x,
# and share one goal at the far corner (100,100).
# Altitude:
#   synthetic terrain : z_start = z_goal = 1 m
#   real DEM          : dynamically adjusted to ground + GROUND_CLEARANCE (20m)
#                      in World.build(), so this value is mostly unused.
# ---------------------------------------------------------------------------
SYNTHETIC_START_GOAL_Z: float = 1.0
REAL_DEM_START_GOAL_Z: float = 65.0  # legacy value, dynamic adjustment is used instead

def _make_start_goal_preset(
    scenario: ScenarioConfig,
) -> Dict[str, Tuple[Tuple[float, float, float], ...]]:
    z = REAL_DEM_START_GOAL_Z if scenario.terrain == "real_dem" else SYNTHETIC_START_GOAL_Z
    n = len(scenario.formation)
    # Start from (1,1), spaced 10 m along +x axis
    starts = tuple((1.0 + i * 10.0, 1.0, z) for i in range(n))
    # All share one goal at (100, 100)
    goals = tuple((100.0, 100.0, z) for _ in range(n))
    return {"starts": starts, "goals": goals}

START_GOAL_PRESETS: Dict[str, Dict[str, Tuple[Tuple[float, float, float], ...]]] = {
    s.key: _make_start_goal_preset(s) for s in SCENARIOS
}

# ---------------------------------------------------------------------------
# Seed scheme ()
# ---------------------------------------------------------------------------
SEED_TERRAIN_OFFSET: int = 0
SEED_START_GOAL_OFFSET: int = 1000
SEED_OBSTACLE_OFFSET: int = 2000
SEED_DYNAMIC_OFFSET: int = 3000
TRAIN_INSTANCE_BASE: int = 1_000_000  # keep training instances disjoint from eval run_ids
OBSTACLE_MOTION_SEED_STRIDE: int = 100000  # per-obstacle motion sub-offset
OBSTACLE_CORRIDOR_BIAS: float = 0.6        # fraction of obstacles placed near the start->goal corridor
OBSTACLE_CORRIDOR_SIGMA: float = 10.0      # perpendicular std (m) for corridor placement

# ---------------------------------------------------------------------------
# Design switches for unresolved implementation choices.
# These centralize the assumptions made while coding; review before experiments.
# ---------------------------------------------------------------------------
COMM_MAX_MODE: str = "max_others"      # "max_others" | "own_speed"
RISK_QUERY_SPATIAL: bool = False       # R_max(p_hat) re-queries fields with frozen noise
CLAMP_XY: bool = True                  # clamp UAV x/y into the world

# f_g blend weight between heading alignment (launch gradient at v=0) and
# real closing progress (rewards speed toward goal).  Higher -> easier launch.
# NOTE: Legacy parameter from  f_goal term;  uses f_r (rejoin incentive) instead.
F_GOAL_ALIGN_WEIGHT: float = 0.3

# Terrain path lookahead (steps) folded into R_max so UAVs start climbing /
# turning before reaching a peak (BRD's 1-step horizon reacts too late).
TERRAIN_LOOKAHEAD_STEPS: int = 12

# Hard per-step safety filter: next waypoint must be above terrain and keep a
# 5 m margin from obstacles / other UAVs (see game.utility.action_feasible).
ENABLE_FEASIBILITY_MASK: bool = True

# Glide-slope descent range (m): horizontal distance from the goal over which
# f_a's reference altitude blends from the terrain-following band down to
# z_goal, so UAVs descend en route and reach the low goal with speed to spare.
DESCENT_RANGE: float = 40.0  # scaled for 100x100 world (was 80m for 1000x1000)

# Terrain-following altitude band: f_a tracks h_high + TERRAIN_CLEARANCE so the
# UAV hugs the terrain instead of over-climbing.  F_ALT_SCALE sets how sharply
# deviations are penalized (smaller = tighter hugging).
TERRAIN_CLEARANCE: float = 3.0
F_ALT_SCALE: float = 8.0

# Terrain threat detection :
#   - Takeoff immunity: skip terrain threat for first N steps to let UAV climb
#   - Look-ahead: check terrain at look-ahead distance instead of current position
#   - Reduced threshold: only trigger when truly close to terrain
TERRAIN_THREAT_THRESHOLD: float = 0.5      # terrain threat threshold (m), reduced from 1.5m
TERRAIN_LOOKAHEAD_DIST: float = 10.0       # terrain look-ahead distance (m)
TAKEOFF_IMMUNITY_STEPS: int = 20           # takeoff immunity steps (10s at DT=0.5)
# Altitude dead-band (m): errors smaller than this are not penalized, so the
# UAV holds a steady cruise instead of jittering climb/descend every step.
F_ALT_DEADBAND: float = 1.0
# Asymmetric penalty ratio for flying above the reference altitude.
# Below-reference errors have full weight (safety); above-reference errors
# are scaled by this factor (economy / path length).  <1.0 means the UAV
# prefers to err on the high side (safer) but still has incentive to descend.
F_ALT_UPPER_WEIGHT: float = 1.0
# Local-max radius (m) for the smooth terrain envelope f_a rides over, so the
# UAV clears a peak cluster at a steady height and starts climbing early.
TERRAIN_ENVELOPE_RADIUS: float = 10.0
# Baseline cruise altitude (m): f_a holds at least this height (or terrain
# envelope + clearance, whichever is higher) so the UAV flies a flat, smooth,
# low path that clears on-route terrain instead of hugging/grazing steep peaks.
CRUISE_ALT: float = 10.0

# Forbid a non-arrived UAV from idling on HOVER whenever a safe move exists, so
# it never freezes at v=0 short of the goal (v=0 cannot change altitude).
FORBID_IDLE_HOVER: bool = False

# Terminal landing descent rate (m/step): once a UAV is horizontally over its
# goal it descends vertically toward the goal altitude at this rate to touch
# down ( landing), decoupled from horizontal speed.
LANDING_DESCENT_RATE: float = 1.0

# Low-speed restart: if a non-arrived UAV's speed drops below this (m/s) while
# still far from the goal, force a (feasible) accelerate so it never freezes
# on hover mid-flight (the from-rest goal reward is too small to beat f_s).
RESTART_SPEED: float = 0.5

# Conditional trigger for replanning 
# Only run BRD when threat distance is below this ratio of (d_safe + R_obs)
REPLAN_TRIGGER_RATIO: float = 0.8

# ---------------------------------------------------------------------------
# Stuck detection & escape maneuver 
# ---------------------------------------------------------------------------
# Trigger escape when forward progress stalls for too long.
STUCK_THRESHOLD: float = 0.3          # min forward progress per check (m)
STUCK_CHECK_STEPS: int = 20           # consecutive steps below threshold to trigger
ESCAPE_STEPS: int = 15                # escape maneuver duration (steps)
ESCAPE_CLIMB_ANGLE: float = 20.0      # climb angle during escape (deg)

def scenario_obstacle_total(scenario: ScenarioConfig) -> int:
    """Total obstacle count for a scenario's composition."""
    return sum(count for _, count in scenario.obstacle_composition)

def scenario_formation_counts(scenario: ScenarioConfig) -> Dict[str, int]:
    """Formation composition counts, e.g. {'A': 3, 'B': 1, 'C': 1}."""
    counts = {key: 0 for key in UAV_TYPES}
    for uav_type in scenario.formation:
        counts[uav_type] += 1
    return counts

def all_uav_type_keys() -> List[str]:
    return list(UAV_TYPES)

def scenario_base_seed(key: str) -> int:
    """Deterministic per-scenario base seed ."""
    return SCENARIO_BASE_SEEDS[key]

# ---------------------------------------------------------------------------
# PSO Offline Global Path Planning 
# ---------------------------------------------------------------------------
# Particle Swarm Optimization for generating global reference paths.
# Path representation: N control points + cubic B-spline interpolation.
# Fitness: path length + heavy penalty for terrain / climb-angle violations.

PSO_NUM_WAYPOINTS: int = 5            # number of intermediate control points
PSO_PATH_SAMPLES: int = 100           # number of sampled points on B-spline path
PSO_N_PARTICLES: int = 50             # swarm size (full quality, for evaluation)
PSO_TRAIN_N_PARTICLES: int = 20       # reduced swarm size for training speed
PSO_MAX_ITER: int = 150               # maximum iterations (full quality, for evaluation)
PSO_TRAIN_MAX_ITER: int = 20          # reduced iterations for training speed
PSO_INERTIA_W: float = 0.8            # inertia weight w
PSO_C1: float = 2.0                   # cognitive (personal best) coefficient
PSO_C2: float = 2.0                   # social (global best) coefficient
PSO_VEL_SCALE: float = 0.15           # velocity bound = scale * position_range
PSO_INIT_RADIUS_RATIO: float = 0.50   # init perturbation radius / start-goal distance
PSO_TERRAIN_CLEARANCE: float = 15.0   # minimum terrain clearance (m) - increased for maximum safety
PSO_MAX_CLIMB_ANGLE: float = 25.0     # max climb angle (deg)
PSO_COLLISION_PENALTY: float = 1000.0  # penalty multiplier for infeasible solutions
PSO_MIN_INTER_UAV_DIST: float = 3.0   # minimum inter-UAV distance for coordinated planning

# Enable global path planning (: PSO offline planning)
ENABLE_GLOBAL_PLANNING: bool = True

# ---------------------------------------------------------------------------
# Path Tracking 
# ---------------------------------------------------------------------------
# Online path following with look-ahead projection method.
# Used when no threat is detected (pure path tracking mode).

PATH_LOOKAHEAD_DIST: float = 10.0      # look-ahead distance for path following (m)
PATH_REJOIN_THRESHOLD: float = 2.0     # lateral deviation threshold for rejoin detection (m)
PATH_PROGRESS_MONOTONIC: bool = True   # progress only advances, never goes backward

# ---------------------------------------------------------------------------
# Utility function parameters 
# ---------------------------------------------------------------------------
# Path deviation term f_p
PATH_DEV_SCALE: float = 15.0           # normalization scale for path deviation (m)

# Rejoin incentive term f_r
REJOIN_SIGMA: float = 8.0              # exponential scale for rejoin reward (m)

# Altitude tracking term f_a
ALT_TRACK_SCALE: float = 10.0          # normalization scale for altitude error (m)
ALT_TRACK_DEADBAND: float = 1.0        # dead-band for altitude error (m)
