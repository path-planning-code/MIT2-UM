"""ID-IT2-FDTM package facade for integration."""

from id_it2_fdtm.id_it2_fdtm import (
    ALPHA,
    MAP_RANGE,
    S_MAX,
    SAFE_ALTITUDE,
    T1,
    T2,
    T_UPDATE,
    FuzzyDynamicTerrain,
    SimpleTerrain,
)

__all__ = [
    "FuzzyDynamicTerrain",
    "SimpleTerrain",
    "MAP_RANGE",
    "SAFE_ALTITUDE",
    "ALPHA",
    "T1",
    "T2",
    "T_UPDATE",
    "S_MAX",
]
