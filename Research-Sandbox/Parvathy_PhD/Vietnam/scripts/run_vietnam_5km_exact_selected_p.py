from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(r"C:\github\Public-Infrastructure-Service-Access")
VIETNAM_SCRIPT_DIR = REPO_ROOT / "Research-Sandbox" / "Parvathy_PhD" / "Vietnam" / "scripts"
sys.path.insert(0, str(VIETNAM_SCRIPT_DIR))

import run_dense_grid_straightline_analysis as dense  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coverage_percent(covered_population: float, total_population: float) -> float:
    return 100.0 * float(covered_population) / float(total_population) if total_population else math.nan


def status_name(status: int) -> str:
    return {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INTERRUPTED: "interrupted",
    }.get(status, str(status))


def build_target_candidate_lists(
    spatial: dense.SpatialMaxCover,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], int, float]:
    start = perf_counter()
    positive = np.flatnonzero(spatial.effective_weights > 0)
    target_indices: list[np.ndarray] = []
    target_weights: list[np.ndarray] = []
    target_candidate_lists: list[np.ndarray] = []
    candidate_to_targets: list[list[int]] = [[] for _ in range(spatial.n_candidates)]
    edge_rows = 0

    for begin in range(0, positive.size, int(chunk_size)):
        end = min(begin + int(chunk_size), positive.size)
        demand_indices = positive[begin:end]
        neighbours = spatial.candidate_tree.query_ball_point(
            spatial.demand.xy[demand_indices],
            spatial.threshold_m,
        )
        kept_indices: list[int] = []
        kept_weights: list[int] = []
        for demand_index, values in zip(demand_indices, neighbours):
            if not values:
                continue
            arr = np.asarray(values, dtype=np.int32)
            if arr.size == 0:
                continue
            target_pos = len(target_candidate_lists)
            target_candidate_lists.append(arr)
            for facility in arr:
                candidate_to_targets[int(facility)].append(target_pos)
            kept_indices.append(int(demand_index))
            kept_weights.append(int(spatial.effective_weights[int(demand_index)]))
            edge_rows += int(arr.size)
        if kept_indices:
            target_indices.append(np.asarray(kept_indices, dtype=np.int32))
            target_weights.append(np.asarray(kept_weights, dtype=np.int64))
        print(
            f"  incidence {end:,}/{positive.size:,} positive demand, "
            f"{len(target_candidate_lists):,} coverable targets, {edge_rows:,} edges",
            flush=True,
        )

    if target_indices:
        indices = np.concatenate(target_indices)
        weights = np.concatenate(target_weights)
    else:
        indices = np.empty(0, dtype=np.int32)
        weights = np.empty(0, dtype=np.int64)
    inverse = [np.asarray(values, dtype=np.int32) for values in candidate_to_targets]
    return indices, weights, target_candidate_lists, inverse, edge_rows, float(perf_counter() - start)


def greedy_start_solution(
    *,
    previous_solution: list[int] | None,
    budget: int,
    weights: np.ndarray,
    target_candidate_lists: list[np.ndarray],
    candidate_to_targets: list[np.ndarray],
) -> tuple[list[int], np.ndarray, int, str]:
    n_candidates = len(candidate_to_targets)
    n_targets = len(target_candidate_lists)
    selected: list[int] = []
    selected_mask = np.zeros(n_candidates, dtype=bool)
    covered = np.zeros(n_targets, dtype=bool)

    if previous_solution:
        for facility in previous_solution:
            facility_i = int(facility)
            if 0 <= facility_i < n_candidates and not selected_mask[facility_i]:
                selected.append(facility_i)
                selected_mask[facility_i] = True
                covered[candidate_to_targets[facility_i]] = True

    if len(selected) > int(budget):
        selected = selected[: int(budget)]
        selected_mask[:] = False
        covered[:] = False
        for facility_i in selected:
            selected_mask[facility_i] = True
            covered[candidate_to_targets[facility_i]] = True

    gains = np.zeros(n_candidates, dtype=np.int64)
    for target_i, facilities in enumerate(target_candidate_lists):
        if covered[target_i]:
            continue
        weight = int(weights[target_i])
        if weight <= 0:
            continue
        gains[facilities] += weight
    gains[selected_mask] = -1

    while len(selected) < int(budget):
        facility_i = int(np.argmax(gains))
        if int(gains[facility_i]) <= 0:
            break
        selected.append(facility_i)
        selected_mask[facility_i] = True
        gains[facility_i] = -1
        newly = candidate_to_targets[facility_i][~covered[candidate_to_targets[facility_i]]]
        if newly.size == 0:
            continue
        covered[newly] = True
        for target_i in newly:
            weight = int(weights[int(target_i)])
            if weight <= 0:
                continue
            facilities = target_candidate_lists[int(target_i)]
            active = facilities[~selected_mask[facilities]]
            if active.size:
                gains[active] -= weight

    objective = int(weights[covered].sum())
    source = "previous_optimal_plus_greedy" if previous_solution else "greedy"
    return selected, covered, objective, source


def apply_mip_start(
    *,
    x: gp.tupledict,
    y: gp.tupledict,
    selected: list[int],
    covered: np.ndarray,
    n_candidates: int,
) -> None:
    selected_mask = np.zeros(n_candidates, dtype=bool)
    for facility in selected:
        selected_mask[int(facility)] = True
    for facility in range(n_candidates):
        x[facility].Start = 1.0 if selected_mask[facility] else 0.0
    for target_i, is_covered in enumerate(covered):
        y[target_i].Start = 1.0 if bool(is_covered) else 0.0


def solve_selected_budgets(
    *,
    spatial: dense.SpatialMaxCover,
    case: str,
    threshold_km: float,
    budgets: list[int],
    output_dir: Path,
    time_limit_seconds: float,
    mip_gap: float,
    incidence_chunk_size: int,
) -> tuple[list[dict], list[dict], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building exact incidence for {case}", flush=True)
    target_indices, weights, target_candidate_lists, candidate_to_targets, edge_rows, incidence_seconds = build_target_candidate_lists(
        spatial,
        chunk_size=int(incidence_chunk_size),
    )
    coverable_incremental = float(weights.sum()) / float(spatial.weight_scale)
    all_candidate_covered = float(spatial.baseline_population + coverable_incremental)

    log_path = output_dir / "gurobi_logs" / f"{case}_exact_selected_p.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_start = perf_counter()
    model = gp.Model(f"{case}_exact_selected_p")
    model.Params.LogFile = str(log_path)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)

    n_candidates = int(spatial.n_candidates)
    n_targets = int(len(target_candidate_lists))
    x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_targets, vtype=GRB.BINARY, name="y")
    budget_constr = model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) <= 0, name="budget")
    for i, facilities in enumerate(target_candidate_lists):
        model.addConstr(y[i] <= gp.quicksum(x[int(j)] for j in facilities), name=f"cover_{i}")
    model.setObjective(gp.quicksum(float(weights[i]) * y[i] for i in range(n_targets)), GRB.MAXIMIZE)
    model.update()
    model_seconds = float(perf_counter() - model_start)

    rows: list[dict] = []
    selected_rows: list[dict] = []
    previous_optimal_solution: list[int] | None = None
    for budget in sorted({int(value) for value in budgets}):
        budget_constr.RHS = int(budget)
        start_solution, start_covered, start_objective, start_source = greedy_start_solution(
            previous_solution=previous_optimal_solution,
            budget=int(budget),
            weights=weights,
            target_candidate_lists=target_candidate_lists,
            candidate_to_targets=candidate_to_targets,
        )
        apply_mip_start(
            x=x,
            y=y,
            selected=start_solution,
            covered=start_covered,
            n_candidates=n_candidates,
        )
        solve_start = perf_counter()
        model.optimize()
        solve_seconds = float(perf_counter() - solve_start)
        incremental = float(model.ObjVal) / float(spatial.weight_scale) if model.SolCount else math.nan
        covered = float(spatial.baseline_population + incremental) if math.isfinite(incremental) else math.nan
        selected = [j for j in range(n_candidates) if model.SolCount and x[j].X > 0.5]
        row = {
            "case": case,
            "country": "Vietnam",
            "distance_model": "straight_line_projected_screening",
            "grid_spacing_m": 5000.0,
            "grid": "5 km",
            "threshold_km": float(threshold_km),
            "budget": int(budget),
            "method": "gurobi_exact_optimum",
            "status_name": status_name(model.Status),
            "status_code": int(model.Status),
            "n_population": int(spatial.n_population),
            "n_candidates": n_candidates,
            "n_targets_positive": int(np.count_nonzero(spatial.effective_weights > 0)),
            "n_targets_coverable": n_targets,
            "candidate_edge_rows": int(edge_rows),
            "baseline_covered_population": float(spatial.baseline_population),
            "available_incremental_population": float(spatial.available_incremental_population),
            "coverable_incremental_population": coverable_incremental,
            "all_candidate_covered_population": all_candidate_covered,
            "all_candidate_coverage_percent": coverage_percent(all_candidate_covered, spatial.total_population),
            "incremental_population": incremental,
            "total_covered_population": covered,
            "coverage_percent_total_population": coverage_percent(covered, spatial.total_population),
            "objective_weight_units": None if not model.SolCount else float(model.ObjVal),
            "objective_bound_weight_units": None if model.ObjBound == GRB.INFINITY else float(model.ObjBound),
            "mip_gap": None if model.MIPGap == GRB.INFINITY else float(model.MIPGap),
            "selected_candidates": int(len(selected)),
            "mip_start_source": start_source,
            "mip_start_selected_candidates": int(len(start_solution)),
            "mip_start_objective_weight_units": int(start_objective),
            "mip_start_covered_population": float(spatial.baseline_population + (float(start_objective) / float(spatial.weight_scale))),
            "incidence_seconds": incidence_seconds,
            "model_seconds": model_seconds,
            "seconds": solve_seconds,
            "cumulative_seconds": incidence_seconds + model_seconds + sum(float(r["seconds"]) for r in rows) + solve_seconds,
            "gurobi_log": str(log_path),
        }
        rows.append(row)
        for rank, j in enumerate(selected, start=1):
            selected_rows.append(
                {
                    "case": case,
                    "threshold_km": float(threshold_km),
                    "budget": int(budget),
                    "rank": int(rank),
                    "candidate_index": int(j),
                    "source_id": str(spatial.candidates.ids[j]),
                    "longitude": float(spatial.candidates.lon[j]),
                    "latitude": float(spatial.candidates.lat[j]),
                    "status_name": row["status_name"],
                }
            )
        write_csv(output_dir / "vietnam_5km_exact_selected_p.csv", rows)
        write_csv(output_dir / "vietnam_5km_exact_selected_candidates.csv", selected_rows)
        print(
            f"  p={budget}: {row['status_name']} {row['coverage_percent_total_population']:.6f}% "
            f"in {solve_seconds:.3f}s gap={row['mip_gap']} start={start_source}",
            flush=True,
        )
        previous_optimal_solution = selected if row["status_name"] == "optimal" else None

    stats = {
        "case": case,
        "threshold_km": float(threshold_km),
        "grid": "5 km",
        "n_population": int(spatial.n_population),
        "n_candidates": n_candidates,
        "n_targets_positive": int(np.count_nonzero(spatial.effective_weights > 0)),
        "n_targets_coverable": n_targets,
        "candidate_edge_rows": int(edge_rows),
        "baseline_covered_population": float(spatial.baseline_population),
        "available_incremental_population": float(spatial.available_incremental_population),
        "coverable_incremental_population": coverable_incremental,
        "all_candidate_covered_population": all_candidate_covered,
        "all_candidate_coverage_percent": coverage_percent(all_candidate_covered, spatial.total_population),
        "incidence_seconds": incidence_seconds,
        "model_seconds": model_seconds,
        "total_exact_seconds": incidence_seconds + model_seconds + sum(float(r["seconds"]) for r in rows),
        "gurobi_log": str(log_path),
    }
    return rows, selected_rows, stats


def best_heuristic_rows(path: Path, grid_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["grid"] = grid_label
    return (
        df.sort_values(
            ["threshold_km", "budget", "total_covered_population", "seconds"],
            ascending=[True, True, False, True],
        )
        .groupby(["threshold_km", "budget"], as_index=False)
        .head(1)
        .copy()
    )


def write_comparison(
    *,
    output_dir: Path,
    exact_rows: list[dict],
    heuristic_1km_path: Path,
    heuristic_5km_path: Path,
) -> pd.DataFrame:
    exact = pd.DataFrame(exact_rows)
    one = best_heuristic_rows(heuristic_1km_path, "1 km")
    five = best_heuristic_rows(heuristic_5km_path, "5 km")

    one = one.rename(
        columns={
            "method": "vietnam_1km_best_heuristic_method",
            "coverage_percent_total_population": "vietnam_1km_best_heuristic_coverage_percent",
            "total_covered_population": "vietnam_1km_best_heuristic_covered_population",
            "seconds": "vietnam_1km_best_heuristic_seconds",
        }
    )
    five = five.rename(
        columns={
            "method": "vietnam_5km_best_heuristic_method",
            "coverage_percent_total_population": "vietnam_5km_best_heuristic_coverage_percent",
            "total_covered_population": "vietnam_5km_best_heuristic_covered_population",
            "seconds": "vietnam_5km_best_heuristic_seconds",
        }
    )
    exact = exact.rename(
        columns={
            "coverage_percent_total_population": "vietnam_5km_exact_coverage_percent",
            "total_covered_population": "vietnam_5km_exact_covered_population",
            "seconds": "vietnam_5km_exact_seconds",
            "status_name": "vietnam_5km_exact_status",
        }
    )
    joined = exact.merge(
        one[
            [
                "threshold_km",
                "budget",
                "vietnam_1km_best_heuristic_method",
                "vietnam_1km_best_heuristic_coverage_percent",
                "vietnam_1km_best_heuristic_covered_population",
                "vietnam_1km_best_heuristic_seconds",
            ]
        ],
        on=["threshold_km", "budget"],
        how="left",
    )
    joined = joined.merge(
        five[
            [
                "threshold_km",
                "budget",
                "vietnam_5km_best_heuristic_method",
                "vietnam_5km_best_heuristic_coverage_percent",
                "vietnam_5km_best_heuristic_covered_population",
                "vietnam_5km_best_heuristic_seconds",
            ]
        ],
        on=["threshold_km", "budget"],
        how="left",
    )
    joined["vietnam_1km_heuristic_minus_5km_exact_pp"] = (
        joined["vietnam_1km_best_heuristic_coverage_percent"] - joined["vietnam_5km_exact_coverage_percent"]
    )
    joined["vietnam_5km_exact_minus_5km_heuristic_pp"] = (
        joined["vietnam_5km_exact_coverage_percent"] - joined["vietnam_5km_best_heuristic_coverage_percent"]
    )
    out = joined.sort_values(["threshold_km", "budget"])
    out.to_csv(output_dir / "vietnam_1km_heuristic_vs_5km_exact.csv", index=False)
    return out


def fmt_pct(value: float) -> str:
    return "" if pd.isna(value) else f"{float(value):.3f}%"


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if abs(number) < 0.0005:
        number = 0.0
    return f"{number:.3f}"


def markdown_table(df: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, comparison: pd.DataFrame, stats_rows: list[dict]) -> None:
    display = comparison.copy()
    for col in [
        "vietnam_5km_exact_coverage_percent",
        "vietnam_1km_best_heuristic_coverage_percent",
        "vietnam_5km_best_heuristic_coverage_percent",
    ]:
        display[col] = display[col].map(fmt_pct)
    for col in [
        "vietnam_1km_heuristic_minus_5km_exact_pp",
        "vietnam_5km_exact_minus_5km_heuristic_pp",
        "vietnam_5km_exact_seconds",
        "vietnam_1km_best_heuristic_seconds",
    ]:
        display[col] = display[col].map(fmt_num)

    stats = pd.DataFrame(stats_rows)
    for col in ["all_candidate_coverage_percent", "total_exact_seconds", "incidence_seconds", "model_seconds"]:
        stats[col] = stats[col].map(fmt_num)

    lines = [
        "# Vietnam 5 km Exact vs 1 km Heuristic",
        "",
        "This is the projected straight-line dense-grid comparison. The 5 km rows are Gurobi exact selected-p solves; the 1 km rows are the best existing dense-grid heuristic rows.",
        "",
        "## Exact Instance Summary",
        "",
        markdown_table(
            stats,
            [
                "threshold_km",
                "n_candidates",
                "n_targets_coverable",
                "candidate_edge_rows",
                "all_candidate_coverage_percent",
                "total_exact_seconds",
            ],
            ["Threshold", "Candidates", "Coverable Targets", "Edges", "All-Candidate Coverage", "Total Exact Seconds"],
        ),
        "",
        "## Comparison",
        "",
        markdown_table(
            display,
            [
                "threshold_km",
                "budget",
                "vietnam_5km_exact_status",
                "vietnam_5km_exact_coverage_percent",
                "vietnam_1km_best_heuristic_coverage_percent",
                "vietnam_1km_heuristic_minus_5km_exact_pp",
                "vietnam_5km_best_heuristic_coverage_percent",
                "vietnam_5km_exact_minus_5km_heuristic_pp",
                "vietnam_5km_exact_seconds",
                "vietnam_1km_best_heuristic_seconds",
                "vietnam_1km_best_heuristic_method",
            ],
            [
                "Threshold",
                "p",
                "5 km Exact Status",
                "5 km Exact",
                "1 km Heuristic",
                "1 km Heur - 5 km Exact pp",
                "5 km Heuristic",
                "5 km Exact - 5 km Heur pp",
                "5 km Exact Seconds",
                "1 km Heur Seconds",
                "1 km Method",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=ROOT / "runs" / "vietnam_20260619_0630" / "vietnam_data" / "outputs",
    )
    parser.add_argument(
        "--candidate-grid",
        type=Path,
        default=ROOT
        / "runs"
        / "vietnam_20260619_0630"
        / "vietnam_data"
        / "cache"
        / "vnm_candidate_sites_spacing_5000m_water_allowed_include_boundary_epsg_3405.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "vietnam_20260619_0630" / "dense_grid_straightline" / "grid_5000m_exact",
    )
    parser.add_argument("--run-tag-marker", default="maxdist_150000")
    parser.add_argument("--thresholds-km", type=float, nargs="+", default=[20.0, 50.0, 100.0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 80, 200])
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--incidence-chunk-size", type=int, default=1000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument(
        "--heuristic-1km",
        type=Path,
        default=ROOT
        / "outputs"
        / "vietnam_20260619_0630"
        / "dense_grid_straightline"
        / "grid_1000m"
        / "coverage_summary_by_budget.csv",
    )
    parser.add_argument(
        "--heuristic-5km",
        type=Path,
        default=ROOT
        / "outputs"
        / "vietnam_20260619_0630"
        / "dense_grid_straightline"
        / "grid_5000m"
        / "coverage_summary_by_budget.csv",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demand, candidates, existing, provenance = dense.load_inputs(
        outputs_dir=args.outputs_dir,
        marker=args.run_tag_marker,
        candidate_grid_path=args.candidate_grid,
        grid_spacing_m=5000.0,
        weight_scale=float(args.weight_scale),
    )

    all_rows: list[dict] = []
    all_selected: list[dict] = []
    stats_rows: list[dict] = []
    for threshold_km in args.thresholds_km:
        spatial = dense.SpatialMaxCover(
            demand=demand,
            candidates=candidates,
            existing=existing,
            threshold_m=float(threshold_km) * 1000.0,
            weight_scale=float(args.weight_scale),
            cache_size=int(args.cache_size),
            chunk_size=int(args.chunk_size),
        )
        case = f"vietnam_grid5km_straightline_{float(threshold_km):g}km"
        rows, selected, stats = solve_selected_budgets(
            spatial=spatial,
            case=case,
            threshold_km=float(threshold_km),
            budgets=[int(value) for value in args.budgets],
            output_dir=args.output_dir,
            time_limit_seconds=float(args.time_limit_seconds),
            mip_gap=float(args.mip_gap),
            incidence_chunk_size=int(args.incidence_chunk_size),
        )
        all_rows.extend(rows)
        all_selected.extend(selected)
        stats_rows.append(stats)
        write_csv(args.output_dir / "vietnam_5km_exact_selected_p.csv", all_rows)
        write_csv(args.output_dir / "vietnam_5km_exact_selected_candidates.csv", all_selected)
        write_csv(args.output_dir / "vietnam_5km_exact_instance_stats.csv", stats_rows)

    comparison = write_comparison(
        output_dir=args.output_dir,
        exact_rows=all_rows,
        heuristic_1km_path=args.heuristic_1km,
        heuristic_5km_path=args.heuristic_5km,
    )
    write_report(args.output_dir / "vietnam_5km_exact_vs_1km_heuristic_report.md", comparison, stats_rows)

    manifest = {
        "outputs_dir": str(args.outputs_dir),
        "candidate_grid": str(args.candidate_grid),
        "output_dir": str(args.output_dir),
        "thresholds_km": [float(value) for value in args.thresholds_km],
        "budgets": [int(value) for value in args.budgets],
        "distance_model": "straight_line_projected_screening",
        "grid_spacing_m": 5000.0,
        "weight_scale": float(args.weight_scale),
        "time_limit_seconds": float(args.time_limit_seconds),
        "mip_gap": float(args.mip_gap),
        "provenance": provenance,
        "outputs": {
            "exact_rows": str(args.output_dir / "vietnam_5km_exact_selected_p.csv"),
            "selected_candidates": str(args.output_dir / "vietnam_5km_exact_selected_candidates.csv"),
            "instance_stats": str(args.output_dir / "vietnam_5km_exact_instance_stats.csv"),
            "comparison": str(args.output_dir / "vietnam_1km_heuristic_vs_5km_exact.csv"),
            "report": str(args.output_dir / "vietnam_5km_exact_vs_1km_heuristic_report.md"),
        },
    }
    (args.output_dir / "vietnam_5km_exact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
