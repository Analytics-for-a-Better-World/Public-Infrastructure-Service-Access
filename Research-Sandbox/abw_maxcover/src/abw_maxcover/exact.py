"""Exact maximum-cover solvers and exact budget curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from ._budgets import normalise_budget_order
from .instance import MaxCoverInstance
from .results import MaxCoverCurve, MaxCoverResult
from ._incremental_core import (
    compute_coverage_and_objective,
    select_by_marginal_gain,
)

@dataclass(slots=True)
class GurobiConfig:
    time_limit_seconds: float = 300.0
    mip_gap: float = 1e-8
    trace: bool = False
    parsimonious: bool = False
    log_file: str | None = None
    warm_start: bool = True
    fixed_facilities: tuple[int, ...] = ()
    fixed_count_against_budget: bool = False


@dataclass(slots=True)
class PyomoConfig:
    solver: str = "gurobi"
    time_limit_seconds: float = 300.0
    mip_gap: float = 1e-8
    trace: bool = False
    parsimonious: bool = False
    fixed_facilities: tuple[int, ...] = ()
    fixed_count_against_budget: bool = False

def _greedy_mip_start(
    instance: MaxCoverInstance,
    budget: int,
    *,
    previous_solution: list[int] | None = None,
    fixed_facilities: tuple[int, ...] = (),
    fixed_count_against_budget: bool = False,
) -> tuple[list[int], np.ndarray, int, str]:
    max_selected = int(budget) if fixed_count_against_budget else int(budget) + len(fixed_facilities)
    start = select_by_marginal_gain(
        instance,
        max_selected,
        initial_solution=list(fixed_facilities) + list(previous_solution or []),
    )
    source = "previous_optimal_plus_greedy" if previous_solution else "greedy"
    return list(start.solution), start.coverage, int(start.objective), source


def _gurobi_status_name(gb_module: Any, status: int) -> str:
    code = gb_module.GRB
    mapping = {
        code.LOADED: "loaded",
        code.OPTIMAL: "optimal",
        code.INFEASIBLE: "infeasible",
        code.INF_OR_UNBD: "inf_or_unbd",
        code.UNBOUNDED: "unbounded",
        code.CUTOFF: "cutoff",
        code.ITERATION_LIMIT: "iteration_limit",
        code.NODE_LIMIT: "node_limit",
        code.TIME_LIMIT: "time_limit",
        code.SOLUTION_LIMIT: "solution_limit",
        code.INTERRUPTED: "interrupted",
        code.NUMERIC: "numeric",
        code.SUBOPTIMAL: "suboptimal",
        code.INPROGRESS: "in_progress",
        code.USER_OBJ_LIMIT: "user_obj_limit",
    }
    return mapping.get(status, f"unknown_{status}")


def solve_gurobi_curve(
    instance: MaxCoverInstance,
    budgets: list[int],
    *,
    config: GurobiConfig | None = None,
    progress: Callable[[Any], Any] = lambda iterable: iterable,
    result_callback: Callable[[MaxCoverResult], Any] | None = None,
) -> MaxCoverCurve:
    """Solve an exact Pareto curve with one reusable gurobipy model.

    When ``warm_start`` is enabled, each requested execution budget receives a
    MIP start from the previous optimal solution plus shared-core greedy
    additions.

    ``result_callback`` is invoked after every solved budget, enabling callers
    to checkpoint long-running exact curves without rebuilding the model.

    """
    cfg = config or GurobiConfig()
    import gurobipy as gb

    requested_budgets, execution_budgets = normalise_budget_order(budgets)
    target_demand = np.flatnonzero(
        (instance.weights > 0) & (instance.ij_indptr[1:] > instance.ij_indptr[:-1])
    ).astype(np.int32)

    model_start = perf_counter()
    model = gb.Model(f"{instance.name}_max_cover")
    model.ModelSense = gb.GRB.MAXIMIZE
    model.Params.OutputFlag = int(cfg.trace)
    model.Params.MIPGap = float(cfg.mip_gap)
    model.Params.TimeLimit = float(cfg.time_limit_seconds)
    if cfg.log_file:
        model.Params.LogFile = str(cfg.log_file)

    max_budget = max(execution_budgets) if execution_budgets else 0
    x_obj = -1.0 / (max_budget + len(cfg.fixed_facilities) + 1) if cfg.parsimonious else 0.0
    x = model.addVars(instance.n_facilities, obj=x_obj, vtype=gb.GRB.BINARY, name="x")
    y = model.addVars(target_demand.tolist(), vtype=gb.GRB.BINARY, name="y")

    for facility in cfg.fixed_facilities:
        facility_i = int(facility)
        if 0 <= facility_i < instance.n_facilities:
            x[facility_i].lb = 1.0
            x[facility_i].ub = 1.0

    for demand in target_demand:
        demand_i = int(demand)
        model.addConstr(
            y[demand_i] <= gb.quicksum(x[int(facility)] for facility in instance.facilities_of(demand_i)),
            name=f"cover_{demand_i}",
        )

    budget_constr = model.addConstr(x.sum() <= 0, name="budget")
    model.setObjective(
        gb.quicksum(float(instance.weights[int(demand)]) * y[int(demand)] for demand in target_demand),
        gb.GRB.MAXIMIZE,
    )
    model.update()
    model_seconds = float(perf_counter() - model_start)

    result_by_budget: dict[int, MaxCoverResult] = {}
    previous_optimal_solution: list[int] | None = None

    for budget in progress(execution_budgets):
        rhs = int(budget) if cfg.fixed_count_against_budget else int(budget) + len(cfg.fixed_facilities)
        budget_constr.RHS = rhs
        start_solution: list[int] = []
        start_objective: int | None = None
        start_source: str | None = None
        if cfg.warm_start:
            start_solution, start_coverage, start_objective, start_source = _greedy_mip_start(
                instance,
                budget,
                previous_solution=previous_optimal_solution,
                fixed_facilities=cfg.fixed_facilities,
                fixed_count_against_budget=cfg.fixed_count_against_budget,
            )
            selected_mask = np.zeros(instance.n_facilities, dtype=bool)
            selected_mask[start_solution] = True
            for facility in range(instance.n_facilities):
                x[facility].Start = 1.0 if selected_mask[facility] else 0.0
            for demand in target_demand:
                demand_i = int(demand)
                y[demand_i].Start = 1.0 if start_coverage[demand_i] > 0 else 0.0

        solve_start = perf_counter()
        model.optimize()
        solve_seconds = float(perf_counter() - solve_start)
        status = _gurobi_status_name(gb, int(model.Status))

        if model.SolCount > 0:
            solution = [facility for facility in range(instance.n_facilities) if x[facility].X >= 0.5]
            coverage, objective = compute_coverage_and_objective(instance, solution)
        else:
            solution = []
            coverage = None
            objective = None

        upper_bound = None
        if getattr(model, "ObjBound", None) is not None and isfinite(float(model.ObjBound)):
            upper_bound = float(floor(model.ObjBound))

        result = MaxCoverResult(
            budget=int(budget),
            method="gurobi_exact",
            objective=None if objective is None else int(objective),
            solution=solution,
            status=status,
            upper_bound=upper_bound,
            mip_gap=None if model.MIPGap == gb.GRB.INFINITY else float(model.MIPGap),
            coverage=coverage,
            model_seconds=model_seconds,
            solve_seconds=solve_seconds,
            total_seconds=model_seconds + solve_seconds,
            metadata={
                "warm_start_source": start_source,
                "warm_start_objective": start_objective,
                "warm_start_selected_count": len(start_solution),
                "target_demand_count": int(target_demand.size),
            },
        )
        result_by_budget[int(budget)] = result
        if result_callback is not None:
            result_callback(result)
        previous_optimal_solution = solution if status == "optimal" else None

    return MaxCoverCurve(
        instance_name=instance.name,
        kind="exact",
        results=[result_by_budget[budget] for budget in requested_budgets],
        metadata={
            "solver": "gurobi",
            "target_demand_count": int(target_demand.size),
            "requested_budgets": requested_budgets,
            "execution_budgets": execution_budgets,
        },
    )


def _make_pyomo_solver(name: str, *, time_limit_seconds: float, mip_gap: float):
    import pyomo.environ as pyo

    solver = pyo.SolverFactory(name)
    if time_limit_seconds is not None:
        if "cplex" in name:
            solver.options["timelimit"] = time_limit_seconds
        elif "cbc" in name:
            solver.options["sec"] = int(np.ceil(time_limit_seconds))
        elif "gurobi" in name:
            solver.options["TimeLimit"] = time_limit_seconds
        elif "highs" in name:
            solver.options["time_limit"] = time_limit_seconds
    if mip_gap is not None:
        if "cplex" in name:
            solver.options["mipgap"] = mip_gap
        elif "cbc" in name:
            solver.options["allowableGap"] = mip_gap
        elif "gurobi" in name:
            solver.options["MipGap"] = mip_gap
        elif "highs" in name:
            solver.options["mip_rel_gap"] = mip_gap
    return solver


def solve_pyomo_curve(
    instance: MaxCoverInstance,
    budgets: list[int],
    *,
    config: PyomoConfig | None = None,
    progress: Callable[[Any], Any] = lambda iterable: iterable,
) -> MaxCoverCurve:
    """Solve an exact curve through Pyomo using the same result schema."""
    cfg = config or PyomoConfig()
    import pyomo.environ as pyo

    requested_budgets, execution_budgets = normalise_budget_order(budgets)
    target_demand = instance.demand_with_candidates().tolist()
    facilities = list(range(instance.n_facilities))

    model_start = perf_counter()
    model = pyo.ConcreteModel("max_cover")
    model.I = pyo.Set(initialize=target_demand)
    model.J = pyo.Set(initialize=facilities)
    model.budget = pyo.Param(mutable=True, default=0)
    model.X = pyo.Var(model.J, domain=pyo.Binary)
    model.Y = pyo.Var(model.I, domain=pyo.Binary)

    for facility in cfg.fixed_facilities:
        facility_i = int(facility)
        if facility_i in facilities:
            model.X[facility_i].fix(1)

    @model.Expression()
    def n_open(m):
        return pyo.quicksum(m.X[j] for j in m.J)

    @model.Expression()
    def weighted_coverage(m):
        return pyo.quicksum(float(instance.weights[i]) * m.Y[i] for i in m.I)

    max_budget = max(execution_budgets) if execution_budgets else 0
    coef_x = -1.0 / (max_budget + len(cfg.fixed_facilities) + 1) if cfg.parsimonious else 0.0

    @model.Objective(sense=pyo.maximize)
    def objective(m):
        return m.weighted_coverage + coef_x * m.n_open

    @model.Constraint(model.I)
    def cover_if_open(m, demand):
        return m.Y[demand] <= pyo.quicksum(m.X[int(facility)] for facility in instance.facilities_of(int(demand)))

    @model.Constraint()
    def budget_limit(m):
        return m.n_open <= m.budget

    solver = _make_pyomo_solver(
        cfg.solver,
        time_limit_seconds=cfg.time_limit_seconds,
        mip_gap=cfg.mip_gap,
    )
    model_seconds = float(perf_counter() - model_start)
    result_by_budget: dict[int, MaxCoverResult] = {}
    for budget in progress(execution_budgets):
        model.budget.set_value(
            int(budget) if cfg.fixed_count_against_budget else int(budget) + len(cfg.fixed_facilities)
        )
        solve_start = perf_counter()
        solver_result = solver.solve(model, tee=cfg.trace)
        solve_seconds = float(perf_counter() - solve_start)
        solution = [j for j in facilities if pyo.value(model.X[j]) >= 0.5]
        coverage, objective_value = compute_coverage_and_objective(instance, solution)
        upper_bound = getattr(solver_result.problem, "upper_bound", None)
        result_by_budget[int(budget)] = MaxCoverResult(
            budget=int(budget),
            method=f"pyomo_{cfg.solver}_exact",
            objective=int(objective_value),
            solution=solution,
            status=str(solver_result.solver.termination_condition),
            upper_bound=None if upper_bound is None else float(upper_bound),
            coverage=coverage,
            model_seconds=model_seconds,
            solve_seconds=solve_seconds,
            total_seconds=model_seconds + solve_seconds,
        )

    return MaxCoverCurve(
        instance_name=instance.name,
        kind="exact",
        results=[result_by_budget[budget] for budget in requested_budgets],
        metadata={
            "solver": f"pyomo:{cfg.solver}",
            "target_demand_count": len(target_demand),
            "requested_budgets": requested_budgets,
            "execution_budgets": execution_budgets,
        },
    )
