"""Regression tests for the exact solvers.

These tests need an optional solver and are skipped when none is available:
``gurobipy`` with a working licence for the Gurobi path, and ``pyomo`` with
``highspy`` for the Pyomo path.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import abw_maxcover as mc


def instance_with_redundant_facility() -> mc.MaxCoverInstance:
    """The README toy instance plus facility 4, which only covers demand 0.

    Demand 0 is already coverable by facilities 0 and 1, so facility 4 can
    never add coverage. Full coverage needs exactly three facilities
    ({0, 1, 3}); a parsimonious solve at a larger budget must not open more.
    """
    weights = np.array([10, 7, 5, 4, 3], dtype=np.int64)
    ij = [[0, 1, 4], [0, 2], [1, 2], [2, 3], [3]]
    ji = [[0, 1], [0, 2], [1, 2, 3], [3, 4], [0]]
    return mc.build_instance(weights, ij, ji, name="toy_redundant", validate_consistency=True)


def random_instance(seed: int, n_demand: int = 30, n_facilities: int = 12) -> mc.MaxCoverInstance:
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 40, size=n_demand, dtype=np.int64)
    ij: list[list[int]] = [[] for _ in range(n_demand)]
    ji: list[list[int]] = [[] for _ in range(n_facilities)]
    for facility in range(n_facilities):
        demand = np.flatnonzero(rng.random(n_demand) < 0.25)
        ji[facility] = demand.tolist()
        for item in demand:
            ij[int(item)].append(facility)
    return mc.build_instance(weights, ij, ji, name=f"random_{seed}", validate_consistency=True)


def brute_force_objective(instance: mc.MaxCoverInstance, budget: int) -> int:
    best = 0
    for size in range(min(budget, instance.n_facilities) + 1):
        for solution in itertools.combinations(range(instance.n_facilities), size):
            _, objective = mc.compute_coverage_and_objective(instance, list(solution))
            best = max(best, objective)
    return best


def _gurobi_available() -> bool:
    try:
        import gurobipy as gb

        model = gb.Model()
        model.Params.OutputFlag = 0
        model.addVar(vtype=gb.GRB.BINARY)
        model.optimize()
        return True
    except Exception:
        return False


def _highs_available() -> bool:
    try:
        import pyomo.environ as pyo

        return bool(pyo.SolverFactory(mc.PyomoConfig().solver).available(exception_flag=False))
    except Exception:
        return False


needs_gurobi = pytest.mark.skipif(not _gurobi_available(), reason="gurobipy or licence unavailable")
needs_highs = pytest.mark.skipif(not _highs_available(), reason="pyomo with highspy unavailable")


def _solve(solver: str, instance: mc.MaxCoverInstance, budgets: list[int], *, parsimonious: bool):
    if solver == "gurobi":
        return mc.exact_pareto_curve(
            instance,
            budgets,
            solver="gurobi",
            gurobi_config=mc.GurobiConfig(parsimonious=parsimonious, warm_start=False),
        )
    # The default PyomoConfig solver must be the one the [pyomo] extra installs.
    return mc.exact_pareto_curve(
        instance,
        budgets,
        solver="pyomo",
        pyomo_config=mc.PyomoConfig(parsimonious=parsimonious),
    )


def test_default_pyomo_solver_is_the_one_the_extra_installs() -> None:
    assert "highs" in mc.PyomoConfig().solver


def _check_parsimonious_solution_is_compact(solver: str) -> None:
    instance = instance_with_redundant_facility()

    curve = _solve(solver, instance, [5], parsimonious=True)
    result = curve.results[0]
    assert result.status == "optimal"
    assert result.objective == instance.total_weight
    assert len(result.solution) == 3
    assert 4 not in result.solution
    assert result.upper_bound is not None
    assert result.upper_bound >= result.objective
    assert result.upper_bound == float(instance.total_weight)
    assert result.mip_gap == 0.0

    plain = _solve(solver, instance, [5], parsimonious=False).results[0]
    assert plain.objective == instance.total_weight
    assert plain.upper_bound is not None
    assert plain.upper_bound >= plain.objective
    assert plain.mip_gap == 0.0


def _infeasible_config(solver: str):
    # Two fixed facilities counted against a budget of one: no feasible point.
    if solver == "gurobi":
        return mc.GurobiConfig(
            fixed_facilities=(0, 1), fixed_count_against_budget=True, warm_start=False
        )
    return mc.PyomoConfig(fixed_facilities=(0, 1), fixed_count_against_budget=True)


def _check_infeasible_solve_reports_no_incumbent(solver: str) -> None:
    instance = instance_with_redundant_facility()
    config = _infeasible_config(solver)
    if solver == "gurobi":
        curve = mc.exact_pareto_curve(instance, [1], solver="gurobi", gurobi_config=config)
    else:
        curve = mc.exact_pareto_curve(instance, [1], solver="pyomo", pyomo_config=config)
    result = curve.results[0]
    assert result.status == "infeasible"
    assert result.objective is None
    assert result.solution == []
    assert result.coverage is None
    assert result.upper_bound is None
    # No incumbent means no gap, never ``inf`` in a results table.
    assert result.mip_gap is None


@needs_gurobi
def test_gurobi_infeasible_solve_reports_no_incumbent() -> None:
    _check_infeasible_solve_reports_no_incumbent("gurobi")


@needs_highs
def test_pyomo_infeasible_solve_reports_no_incumbent() -> None:
    _check_infeasible_solve_reports_no_incumbent("pyomo")


def test_relative_gap_definition() -> None:
    from abw_maxcover.exact import _relative_gap

    assert _relative_gap(None, 10.0) is None
    assert _relative_gap(10, None) is None
    assert _relative_gap(26, 26.0) == 0.0
    assert _relative_gap(20, 26.0) == 0.3
    # A bound below the incumbent can only be noise; never report a negative gap.
    assert _relative_gap(26, 25.0) == 0.0
    assert _relative_gap(0, 0.0) == 0.0
    assert _relative_gap(0, 5.0) is None


def _check_bound_certifies_optimum(solver: str) -> None:
    for seed in range(3):
        instance = random_instance(seed)
        curve = _solve(solver, instance, [1, 2, 3], parsimonious=True)
        for result in curve.results:
            optimum = brute_force_objective(instance, result.budget)
            assert result.objective == optimum
            assert result.upper_bound is not None
            assert result.upper_bound >= optimum
            # A parsimony penalty below one can never push the bound past the
            # next integer, so an optimal solve certifies the optimum exactly.
            assert result.upper_bound == float(optimum)


@needs_gurobi
def test_gurobi_parsimonious_solution_is_compact() -> None:
    _check_parsimonious_solution_is_compact("gurobi")


@needs_gurobi
def test_gurobi_upper_bound_certifies_optimum() -> None:
    _check_bound_certifies_optimum("gurobi")


@needs_highs
def test_pyomo_parsimonious_solution_is_compact() -> None:
    _check_parsimonious_solution_is_compact("pyomo")


@needs_highs
def test_pyomo_upper_bound_certifies_optimum() -> None:
    _check_bound_certifies_optimum("pyomo")


def test_exact_pareto_curve_forwards_callback_to_both_solvers(monkeypatch) -> None:
    """The dispatcher must hand ``result_callback`` to whichever solver runs."""
    import abw_maxcover.pareto as pareto

    seen: dict[str, object] = {}

    def fake_gurobi(instance, budgets, *, config=None, progress=None, result_callback=None):
        seen["gurobi"] = result_callback
        return mc.MaxCoverCurve(instance_name=instance.name, kind="exact", results=[])

    def fake_pyomo(instance, budgets, *, config=None, progress=None, result_callback=None):
        seen["pyomo"] = result_callback
        return mc.MaxCoverCurve(instance_name=instance.name, kind="exact", results=[])

    monkeypatch.setattr(pareto, "solve_gurobi_curve", fake_gurobi)
    monkeypatch.setattr(pareto, "solve_pyomo_curve", fake_pyomo)
    instance = instance_with_redundant_facility()

    def callback(result: mc.MaxCoverResult) -> None:
        pass

    mc.exact_pareto_curve(instance, [1], solver="gurobi", result_callback=callback)
    mc.exact_pareto_curve(instance, [1], solver="pyomo", result_callback=callback)
    assert seen == {"gurobi": callback, "pyomo": callback}


@needs_highs
def test_pyomo_callback_checkpoints_every_budget_in_execution_order() -> None:
    instance = instance_with_redundant_facility()
    checkpoints: list[tuple[int, int | None]] = []

    curve = mc.exact_pareto_curve(
        instance,
        [3, 1, 2],
        solver="pyomo",
        result_callback=lambda result: checkpoints.append((result.budget, result.objective)),
    )

    assert [budget for budget, _ in checkpoints] == [1, 2, 3]
    assert curve.budgets() == [3, 1, 2]
    by_budget = {result.budget: result.objective for result in curve.results}
    assert checkpoints == [(budget, by_budget[budget]) for budget in (1, 2, 3)]


def test_parsimony_penalty_total_stays_below_one() -> None:
    from abw_maxcover.exact import _parsimony_penalty

    assert _parsimony_penalty(10, 2, False) == 0.0
    penalty = _parsimony_penalty(10, 2, True)
    assert 0.0 < penalty * (10 + 2) < 1.0


def test_coverage_upper_bound_undoes_penalty_and_tolerates_noise() -> None:
    from abw_maxcover.exact import _coverage_upper_bound

    # Penalised bound 25.667 with penalty 1/6 over at most 2 open facilities
    # corresponds to an integer coverage bound of 26.
    assert _coverage_upper_bound(25.6667, penalty=1.0 / 6.0, max_open=2, objective=None) == 26.0
    # Floating point noise just below an integer must not lose that integer.
    assert _coverage_upper_bound(203.9999999, penalty=0.0, max_open=3, objective=None) == 204.0
    # A known feasible objective is a valid lower bound on the optimum.
    assert _coverage_upper_bound(25.0, penalty=0.0, max_open=3, objective=26) == 26.0
    assert _coverage_upper_bound(None, penalty=0.0, max_open=3, objective=26) is None
    assert _coverage_upper_bound(float("inf"), penalty=0.0, max_open=3, objective=26) is None
