"""Optimization strategy contracts and portable MILP backends."""

from .facility_location import (
    FacilityLocationConfig,
    FacilityLocationResult,
    solve_facility_location_by_island,
)
from .tsp import TspConfig, TspResult, build_tsp_distance_table, solve_tsp_from_distance_table

__all__ = [
    "FacilityLocationConfig",
    "FacilityLocationResult",
    "TspConfig",
    "TspResult",
    "build_tsp_distance_table",
    "solve_facility_location_by_island",
    "solve_tsp_from_distance_table",
]
