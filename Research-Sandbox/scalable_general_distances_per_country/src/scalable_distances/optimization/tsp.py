from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import pandas as pd
from pyproj import Geod


TspSolverName = Literal["auto", "gurobi", "pyomo-highs"]
GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class TspConfig:
    """Configuration for directed TSP solving over a complete stop-distance table."""

    solver: TspSolverName = "auto"
    mip_gap: float = 0.001
    tee: bool = False


@dataclass(frozen=True)
class TspResult:
    solver: str
    status_name: str
    runtime_seconds: float
    objective_distance: float
    solver_gap: float | None
    tour: list[str]
    arcs: pd.DataFrame
    summary: dict[str, object]


def build_tsp_distance_table(
    router: Any,
    stops: pd.DataFrame,
    *,
    stop_id_col: str = "stop_id",
    stop_type_col: str = "stop_type",
    fallback_multiplier: float = 1.35,
) -> pd.DataFrame:
    """Build a complete directed stop-to-stop distance table for TSP.

    `stops` must contain stop ids, lon/lat, and optional stop type. The router is
    used for road-network distances. Missing road paths are filled with geodesic
    fallback distances so the TSP remains solvable on disconnected components.
    """
    required = {stop_id_col, "lon", "lat"}
    missing = required.difference(stops.columns)
    if missing:
        raise ValueError(f"Stops table is missing required columns: {sorted(missing)}")

    normalized = stops.copy().reset_index(drop=True)
    normalized["stop_id"] = normalized[stop_id_col].astype(str)
    normalized["stop_type"] = normalized[stop_type_col].astype(str) if stop_type_col in normalized.columns else "stop"

    origins = normalized.rename(columns={"stop_id": "source_id", "stop_type": "source_type"})
    destinations = normalized.rename(columns={"stop_id": "target_id", "stop_type": "target_type"})
    origins["source_type"] = "tsp_stop"
    destinations["target_type"] = "tsp_stop"

    routed = router.route_many(origins, destinations)
    road_distances = {
        (str(row.source_id), str(row.target_id)): float(row.total_dist)
        for row in routed.itertuples(index=False)
    }

    coords = normalized.set_index("stop_id")[["lon", "lat"]].astype(float)
    rows: list[dict[str, object]] = []
    for from_id, from_row in coords.iterrows():
        for to_id, to_row in coords.iterrows():
            if from_id == to_id:
                distance_m = 0.0
                source = "identity"
            else:
                road_distance = road_distances.get((from_id, to_id))
                if road_distance is None:
                    _, _, geodesic_m = GEOD.inv(from_row["lon"], from_row["lat"], to_row["lon"], to_row["lat"])
                    distance_m = float(geodesic_m) * fallback_multiplier
                    source = "geodesic_fallback"
                else:
                    distance_m = road_distance
                    source = "road"
            rows.append(
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "distance_m": distance_m,
                    "distance_source": source,
                }
            )
    return pd.DataFrame(rows)


def solve_tsp_from_distance_table(
    distances: pd.DataFrame,
    *,
    depot_id: str,
    config: TspConfig | None = None,
) -> TspResult:
    """Solve a directed TSP from a complete long-form distance table."""
    config = config or TspConfig()
    _validate_tsp_distances(distances, depot_id)
    if config.solver == "gurobi":
        return _solve_tsp_with_gurobi(distances, depot_id=depot_id, config=config)
    if config.solver == "pyomo-highs":
        return _solve_tsp_with_pyomo_highs(distances, depot_id=depot_id, config=config)
    if config.solver != "auto":
        raise ValueError(f"Unknown TSP solver: {config.solver!r}")

    try:
        return _solve_tsp_with_gurobi(distances, depot_id=depot_id, config=config)
    except Exception as gurobi_error:
        try:
            return _solve_tsp_with_pyomo_highs(distances, depot_id=depot_id, config=config)
        except Exception as highs_error:
            raise RuntimeError(
                "No usable TSP MILP solver was available. "
                f"Gurobi failed with: {gurobi_error!r}. "
                f"Pyomo/HiGHS failed with: {highs_error!r}."
            ) from highs_error


def _validate_tsp_distances(distances: pd.DataFrame, depot_id: str) -> None:
    required = {"from_id", "to_id", "distance_m"}
    missing = required.difference(distances.columns)
    if missing:
        raise ValueError(f"TSP distances are missing required columns: {sorted(missing)}")
    ids = set(distances["from_id"].astype(str)).union(distances["to_id"].astype(str))
    if depot_id not in ids:
        raise ValueError(f"Depot {depot_id!r} is not present in TSP distances.")
    expected = {(i, j) for i in ids for j in ids if i != j}
    available = {
        (str(row.from_id), str(row.to_id))
        for row in distances[distances["from_id"].astype(str) != distances["to_id"].astype(str)].itertuples(index=False)
    }
    missing_pairs = expected.difference(available)
    if missing_pairs:
        examples = sorted(missing_pairs)[:5]
        raise ValueError(f"TSP distances are incomplete; examples of missing pairs: {examples}")


def _distance_lookup(distances: pd.DataFrame) -> tuple[list[str], dict[tuple[str, str], float]]:
    ids = sorted(set(distances["from_id"].astype(str)).union(distances["to_id"].astype(str)))
    lookup = {
        (str(row.from_id), str(row.to_id)): float(row.distance_m)
        for row in distances.itertuples(index=False)
        if str(row.from_id) != str(row.to_id)
    }
    return ids, lookup


def _solve_tsp_with_gurobi(distances: pd.DataFrame, *, depot_id: str, config: TspConfig) -> TspResult:
    import gurobipy as gp
    from gurobipy import GRB

    ids, distance = _distance_lookup(distances)
    non_depot = [node_id for node_id in ids if node_id != depot_id]
    model = gp.Model(f"tsp_{depot_id}")
    model.Params.OutputFlag = 1 if config.tee else 0
    model.Params.MIPGap = config.mip_gap
    x = model.addVars(distance.keys(), vtype=GRB.BINARY, name="arc")
    u = model.addVars(non_depot, lb=1, ub=max(1, len(ids) - 1), vtype=GRB.CONTINUOUS, name="order")
    model.setObjective(gp.quicksum(distance[arc] * x[arc] for arc in distance), GRB.MINIMIZE)
    for node_id in ids:
        model.addConstr(gp.quicksum(x[i, j] for i, j in distance if i == node_id) == 1)
        model.addConstr(gp.quicksum(x[i, j] for i, j in distance if j == node_id) == 1)
    n = len(ids)
    for i in non_depot:
        for j in non_depot:
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi did not return a feasible TSP solution; status={model.Status}")
    selected_arcs = [(i, j) for i, j in distance if x[i, j].X > 0.5]
    return _build_tsp_result(
        solver="gurobi",
        status_name="optimal" if model.Status == GRB.OPTIMAL else str(model.Status),
        runtime_seconds=float(model.Runtime),
        objective_distance=float(model.ObjVal),
        solver_gap=float(model.MIPGap),
        depot_id=depot_id,
        selected_arcs=selected_arcs,
        distance=distance,
        distances=distances,
    )


def _solve_tsp_with_pyomo_highs(distances: pd.DataFrame, *, depot_id: str, config: TspConfig) -> TspResult:
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers import Highs

    solver = Highs()
    if not solver.available():
        raise RuntimeError("Pyomo solver 'appsi_highs' is not available; install pyomo and highspy.")
    solver.config.mip_gap = config.mip_gap
    solver.config.stream_solver = config.tee
    solver.highs_options["output_flag"] = bool(config.tee)
    solver.highs_options["log_to_console"] = bool(config.tee)

    ids, distance = _distance_lookup(distances)
    non_depot = [node_id for node_id in ids if node_id != depot_id]
    arcs = list(distance)
    model = pyo.ConcreteModel(f"tsp_{depot_id}")
    model.N = pyo.Set(initialize=ids)
    model.K = pyo.Set(initialize=non_depot)
    model.A = pyo.Set(initialize=arcs, dimen=2)
    model.x = pyo.Var(model.A, domain=pyo.Binary)
    model.u = pyo.Var(model.K, bounds=(1, max(1, len(ids) - 1)))

    @model.Objective(sense=pyo.minimize)
    def total_distance(m):
        return pyo.quicksum(distance[i, j] * m.x[i, j] for i, j in m.A)

    @model.Constraint(model.N)
    def leave_once(m, node_id):
        return pyo.quicksum(m.x[i, j] for i, j in m.A if i == node_id) == 1

    @model.Constraint(model.N)
    def enter_once(m, node_id):
        return pyo.quicksum(m.x[i, j] for i, j in m.A if j == node_id) == 1

    @model.Constraint(model.K, model.K)
    def mtz(m, i, j):
        if i == j:
            return pyo.Constraint.Skip
        return m.u[i] - m.u[j] + len(ids) * m.x[i, j] <= len(ids) - 1

    start = perf_counter()
    results = solver.solve(model)
    runtime = perf_counter() - start
    termination_condition = results.termination_condition
    termination = getattr(termination_condition, "name", str(termination_condition)).lower()
    if termination not in {"optimal", "feasible"}:
        raise RuntimeError(f"Pyomo/HiGHS did not return a feasible TSP solution; termination={termination}")
    selected_arcs = [(i, j) for i, j in arcs if pyo.value(model.x[i, j]) > 0.5]
    return _build_tsp_result(
        solver="pyomo-highs",
        status_name=termination,
        runtime_seconds=float(runtime),
        objective_distance=float(pyo.value(model.total_distance)),
        solver_gap=None,
        depot_id=depot_id,
        selected_arcs=selected_arcs,
        distance=distance,
        distances=distances,
    )


def _build_tsp_result(
    *,
    solver: str,
    status_name: str,
    runtime_seconds: float,
    objective_distance: float,
    solver_gap: float | None,
    depot_id: str,
    selected_arcs: list[tuple[str, str]],
    distance: dict[tuple[str, str], float],
    distances: pd.DataFrame,
) -> TspResult:
    tour = _trace_tour(depot_id, selected_arcs)
    source_lookup = {
        (str(row.from_id), str(row.to_id)): getattr(row, "distance_source", "unknown")
        for row in distances.itertuples(index=False)
    }
    arc_rows = [
        {
            "sequence": idx,
            "from_id": i,
            "to_id": j,
            "distance_m": distance[i, j],
            "distance_source": source_lookup.get((i, j), "unknown"),
        }
        for idx, (i, j) in enumerate(zip(tour[:-1], tour[1:]))
    ]
    arcs = pd.DataFrame(arc_rows)
    summary = {
        "solver": solver,
        "status_name": status_name,
        "stops": len(tour) - 1,
        "tour_stops_including_return": len(tour),
        "objective_distance_m": objective_distance,
        "objective_distance_km": objective_distance / 1000.0,
        "fallback_arcs": int(arcs["distance_source"].eq("geodesic_fallback").sum()) if not arcs.empty else 0,
        "runtime_seconds": runtime_seconds,
        "solver_gap": solver_gap,
    }
    return TspResult(
        solver=solver,
        status_name=status_name,
        runtime_seconds=runtime_seconds,
        objective_distance=objective_distance,
        solver_gap=solver_gap,
        tour=tour,
        arcs=arcs,
        summary=summary,
    )


def _trace_tour(depot_id: str, arcs: list[tuple[str, str]]) -> list[str]:
    successor = {i: j for i, j in arcs}
    tour = [depot_id]
    current = depot_id
    for _ in range(len(arcs)):
        current = successor[current]
        tour.append(current)
        if current == depot_id:
            break
    if tour[-1] != depot_id or len(tour) != len(arcs) + 1:
        raise RuntimeError(f"Selected arcs do not form a single tour from depot {depot_id!r}.")
    return tour
