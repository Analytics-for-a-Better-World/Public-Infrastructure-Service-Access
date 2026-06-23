"""ABW maximum-cover optimization package."""

from .instance import MaxCoverInstance, build_instance, build_instance_from_facility_map
from .results import CurveComparison, HeuristicResult, MaxCoverCurve, MaxCoverResult
from .exact import GurobiConfig, PyomoConfig, solve_gurobi_curve, solve_pyomo_curve
from .heuristics import HeuristicConfig, run_heuristics
from ._incremental_core import (
    SparseSwapLocalSearch,
    add_delta,
    compute_coverage_and_objective,
    drop_delta,
    greedy_construct,
    greedy_then_local_search,
    select_by_marginal_gain,
    swap_delta,
    path_relink_fast,
)
from .deployment import greedy_deployment_sequence, optimize_then_greedy_deployment
from .pareto import approximate_pareto_curve, best_by_budget, compare_curves, exact_pareto_curve

__version__ = "0.1.0"

__all__ = [
    "CurveComparison",
    "GurobiConfig",
    "HeuristicConfig",
    "HeuristicResult",
    "MaxCoverCurve",
    "MaxCoverInstance",
    "MaxCoverResult",
    "PyomoConfig",
    "SparseSwapLocalSearch",
    "add_delta",
    "approximate_pareto_curve",
    "best_by_budget",
    "build_instance",
    "build_instance_from_facility_map",
    "compare_curves",
    "compute_coverage_and_objective",
    "drop_delta",
    "exact_pareto_curve",
    "greedy_construct",
    "greedy_deployment_sequence",
    "greedy_then_local_search",
    "optimize_then_greedy_deployment",
    "path_relink_fast",
    "run_heuristics",
    "select_by_marginal_gain",
    "solve_gurobi_curve",
    "solve_pyomo_curve",
    "swap_delta",
]
