"""ABW maximum-cover optimization package."""

from ._incremental_core import (
    SparseSwapLocalSearch,
    add_delta,
    compute_coverage_and_objective,
    drop_delta,
    greedy_construct,
    greedy_then_local_search,
    path_relink_fast,
    select_by_marginal_gain,
    swap_delta,
)
from .deployment import greedy_deployment_sequence, optimize_then_greedy_deployment
from .exact import GurobiConfig, PyomoConfig, solve_gurobi_curve, solve_pyomo_curve
from .heuristics import HeuristicConfig, run_heuristics
from .iterated import IteratedSearchConfig, run_iterated_search
from .instance import MaxCoverInstance, build_instance, build_instance_from_facility_map
from .pareto import approximate_pareto_curve, best_by_budget, compare_curves, exact_pareto_curve
from .results import CurveComparison, HeuristicResult, MaxCoverCurve, MaxCoverResult

__version__ = "0.2.1"

__all__ = [
    "CurveComparison",
    "GurobiConfig",
    "HeuristicConfig",
    "HeuristicResult",
    "IteratedSearchConfig",
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
    "run_iterated_search",
    "select_by_marginal_gain",
    "solve_gurobi_curve",
    "solve_pyomo_curve",
    "swap_delta",
]
