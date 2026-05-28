from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import pandas as pd
from pyproj import Geod


SolverName = Literal["auto", "gurobi", "pyomo-highs"]


@dataclass(frozen=True)
class FacilityLocationConfig:
    """Configuration for the fixed-charge school facility-location model."""

    setup_km: float = 2.0
    mip_gap: float = 0.001
    solver: SolverName = "auto"
    tee: bool = False


@dataclass(frozen=True)
class FacilityLocationResult:
    island: str
    solver: str
    status_name: str
    runtime_seconds: float
    objective_value: float
    solver_gap: float | None
    selected: pd.DataFrame
    assignments: pd.DataFrame
    summary: dict[str, object]

    def as_notebook_dict(self) -> dict[str, object]:
        return {
            "island": self.island,
            "solver": self.solver,
            "status_name": self.status_name,
            "runtime_seconds": self.runtime_seconds,
            "objective_value": self.objective_value,
            "solver_gap": self.solver_gap,
            "selected": self.selected,
            "assignments": self.assignments,
            "summary": self.summary,
        }


GEOD = Geod(ellps="WGS84")


def solve_facility_location_by_island(
    island: str,
    demand_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    config: FacilityLocationConfig | None = None,
) -> FacilityLocationResult:
    """Solve one island's fixed-charge facility-location model.

    The auto strategy prefers Gurobi when import and license checks pass, then
    falls back to Pyomo's portable HiGHS interface (`appsi_highs`).
    """
    config = config or FacilityLocationConfig()
    demand = demand_table[demand_table["island"].eq(island)].reset_index(drop=True)
    candidates = candidate_table[candidate_table["island"].eq(island)].reset_index(drop=True)
    if demand.empty or candidates.empty:
        raise ValueError(f"No demand/candidates for island {island!r}")

    if config.solver == "gurobi":
        return _solve_with_gurobi(island, demand, candidates, config)
    if config.solver == "pyomo-highs":
        return _solve_with_pyomo_highs(island, demand, candidates, config)
    if config.solver != "auto":
        raise ValueError(f"Unknown facility-location solver: {config.solver!r}")

    try:
        return _solve_with_gurobi(island, demand, candidates, config)
    except Exception as gurobi_error:
        try:
            return _solve_with_pyomo_highs(island, demand, candidates, config)
        except Exception as highs_error:
            raise RuntimeError(
                "No usable facility-location MILP solver was available. "
                f"Gurobi failed with: {gurobi_error!r}. "
                f"Pyomo/HiGHS failed with: {highs_error!r}."
            ) from highs_error


def geodesic_distance_matrix_km(candidates: pd.DataFrame, demand: pd.DataFrame) -> dict[tuple[int, int], float]:
    distances: dict[tuple[int, int], float] = {}
    demand_lons = demand["demand_lon"].astype(float).to_numpy()
    demand_lats = demand["demand_lat"].astype(float).to_numpy()
    n_demand = len(demand)
    for i, candidate in candidates.iterrows():
        _, _, dist_m = GEOD.inv(
            [float(candidate["facility_lon"])] * n_demand,
            [float(candidate["facility_lat"])] * n_demand,
            demand_lons,
            demand_lats,
        )
        for j, km in enumerate(dist_m / 1000.0):
            distances[(int(i), int(j))] = float(km)
    return distances


def _solve_with_gurobi(
    island: str,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    config: FacilityLocationConfig,
) -> FacilityLocationResult:
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model(f"facility_location_{island}")
    model.Params.OutputFlag = 1 if config.tee else 0
    model.Params.MIPGap = config.mip_gap

    n_candidates = len(candidates)
    n_demand = len(demand)
    weights = demand["demand_weight"].astype(float).to_numpy()
    setup_cost = config.setup_km * float(weights.sum())
    distances = geodesic_distance_matrix_km(candidates, demand)

    open_var = model.addVars(n_candidates, vtype=GRB.BINARY, name="open")
    assign_var = model.addVars(n_candidates, n_demand, vtype=GRB.BINARY, name="assign")

    model.setObjective(
        gp.quicksum(setup_cost * open_var[i] for i in range(n_candidates))
        + gp.quicksum(
            float(weights[j]) * distances[(i, j)] * assign_var[i, j]
            for i in range(n_candidates)
            for j in range(n_demand)
        ),
        GRB.MINIMIZE,
    )
    for j in range(n_demand):
        model.addConstr(gp.quicksum(assign_var[i, j] for i in range(n_candidates)) == 1)
    for i in range(n_candidates):
        for j in range(n_demand):
            model.addConstr(assign_var[i, j] <= open_var[i])

    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi did not return a feasible solution; status={model.Status}")

    return _build_result(
        island=island,
        solver="gurobi",
        status_name="optimal" if model.Status == GRB.OPTIMAL else str(model.Status),
        runtime_seconds=float(model.Runtime),
        objective_value=float(model.ObjVal),
        solver_gap=float(model.MIPGap),
        demand=demand,
        candidates=candidates,
        distances=distances,
        is_open=lambda i: open_var[i].X > 0.5,
        is_assigned=lambda i, j: assign_var[i, j].X > 0.5,
    )


def _solve_with_pyomo_highs(
    island: str,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    config: FacilityLocationConfig,
) -> FacilityLocationResult:
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers import Highs

    solver = Highs()
    if not solver.available():
        raise RuntimeError("Pyomo solver 'appsi_highs' is not available; install pyomo and highspy.")
    solver.config.mip_gap = config.mip_gap
    solver.config.stream_solver = config.tee
    solver.highs_options["output_flag"] = bool(config.tee)
    solver.highs_options["log_to_console"] = bool(config.tee)

    n_candidates = len(candidates)
    n_demand = len(demand)
    candidate_ids = list(range(n_candidates))
    demand_ids = list(range(n_demand))
    weights = demand["demand_weight"].astype(float).to_dict()
    setup_cost = config.setup_km * float(sum(weights.values()))
    distances = geodesic_distance_matrix_km(candidates, demand)

    model = pyo.ConcreteModel(f"facility_location_{island}")
    model.I = pyo.Set(initialize=candidate_ids)
    model.J = pyo.Set(initialize=demand_ids)
    model.open = pyo.Var(model.I, domain=pyo.Binary)
    model.assign = pyo.Var(model.I, model.J, domain=pyo.Binary)

    @model.Objective(sense=pyo.minimize)
    def total_cost(m):
        return setup_cost * pyo.quicksum(m.open[i] for i in m.I) + pyo.quicksum(
            float(weights[j]) * distances[(i, j)] * m.assign[i, j]
            for i in m.I
            for j in m.J
        )

    @model.Constraint(model.J)
    def assign_each_demand(m, j):
        return pyo.quicksum(m.assign[i, j] for i in m.I) == 1

    @model.Constraint(model.I, model.J)
    def assign_only_to_open(m, i, j):
        return m.assign[i, j] <= m.open[i]

    start = perf_counter()
    results = solver.solve(model)
    runtime = perf_counter() - start
    termination_condition = results.termination_condition
    termination = getattr(termination_condition, "name", str(termination_condition)).lower()
    if termination not in {"optimal", "feasible"}:
        raise RuntimeError(f"Pyomo/HiGHS did not return a feasible solution; termination={termination}")

    objective_value = float(pyo.value(model.total_cost))
    return _build_result(
        island=island,
        solver="pyomo-highs",
        status_name=termination,
        runtime_seconds=float(runtime),
        objective_value=objective_value,
        solver_gap=None,
        demand=demand,
        candidates=candidates,
        distances=distances,
        is_open=lambda i: pyo.value(model.open[i]) > 0.5,
        is_assigned=lambda i, j: pyo.value(model.assign[i, j]) > 0.5,
    )


def _build_result(
    *,
    island: str,
    solver: str,
    status_name: str,
    runtime_seconds: float,
    objective_value: float,
    solver_gap: float | None,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    distances: dict[tuple[int, int], float],
    is_open,
    is_assigned,
) -> FacilityLocationResult:
    open_rows = []
    assignment_rows = []
    for i, candidate in candidates.iterrows():
        if is_open(int(i)):
            open_rows.append(candidate.to_dict())
        for j, demand_row in demand.iterrows():
            if is_assigned(int(i), int(j)):
                assignment_rows.append(_assignment_row(island, candidate, demand_row, distances[(int(i), int(j))]))

    selected = pd.DataFrame(open_rows)
    assignments = pd.DataFrame(assignment_rows)
    if assignments.empty:
        raise RuntimeError(f"{solver} returned no assignments for island {island!r}.")

    weighted_mean = float(
        (assignments["distance_km"] * assignments["students"]).sum() / assignments["students"].sum()
    )
    summary = {
        "island": island,
        "solver": solver,
        "status_name": status_name,
        "schools": int(len(demand)),
        "candidate_facilities": int(len(candidates)),
        "selected_facilities": int(len(selected)),
        "total_students_weight": float(assignments["students"].sum()),
        "mean_distance_km": float(assignments["distance_km"].mean()),
        "weighted_mean_distance_km": weighted_mean,
        "p90_distance_km": float(assignments["distance_km"].quantile(0.9)),
        "max_distance_km": float(assignments["distance_km"].max()),
        "objective_value": objective_value,
        "solver_gap": solver_gap,
        "runtime_seconds": runtime_seconds,
    }
    return FacilityLocationResult(
        island=island,
        solver=solver,
        status_name=status_name,
        runtime_seconds=runtime_seconds,
        objective_value=objective_value,
        solver_gap=solver_gap,
        selected=selected,
        assignments=assignments,
        summary=summary,
    )


def _assignment_row(island: str, candidate: pd.Series, demand_row: pd.Series, distance_km: float) -> dict[str, object]:
    return {
        "demand_id": demand_row["demand_id"],
        "source_id": demand_row.get("source_id"),
        "source_name": demand_row.get("source_name"),
        "island": island,
        "regency": demand_row.get("regency"),
        "students": float(demand_row.get("students", demand_row.get("demand_weight", 1.0))),
        "demand_lat": float(demand_row["demand_lat"]),
        "demand_lon": float(demand_row["demand_lon"]),
        "facility_candidate_id": candidate["candidate_id"],
        "facility_lat": float(candidate["facility_lat"]),
        "facility_lon": float(candidate["facility_lon"]),
        "distance_km": float(distance_km),
    }
