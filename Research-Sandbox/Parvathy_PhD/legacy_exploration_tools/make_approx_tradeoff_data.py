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
APPROX_SRC = (
    Path(r"C:\github\Public-Infrastructure-Service-Access")
    / "Research-Sandbox"
    / "approximated_tradeoff"
    / "src"
)
sys.path.insert(0, str(APPROX_SRC))
sys.path.insert(0, str(ROOT / "tools"))

import mc_heuristics as mch  # noqa: E402
from finish_timor_leste_fresh import build_gurobi_data, find_manifest  # noqa: E402


def load_vietnam_npz(path: Path) -> tuple[mch.MaxCoverInstance, dict]:
    data = np.load(path, allow_pickle=False)
    instance = mch.MaxCoverInstance(
        w=data["w"],
        ij_indptr=data["ij_indptr"],
        ij_indices=data["ij_indices"],
        ji_indptr=data["ji_indptr"],
        ji_indices=data["ji_indices"],
    )
    metadata = json.loads(str(data["metadata_json"]))
    return instance, metadata


def build_instance_from_timor_data(data: dict, weight_scale: float) -> mch.MaxCoverInstance:
    weights = np.rint(np.asarray(data["weights"], dtype=float) * weight_scale).astype(np.int64)
    ij_lists = [
        np.asarray(sorted(values), dtype=np.int32)
        for values in data["target_candidate_lists"]
    ]
    ji_lists: list[list[int]] = [[] for _ in range(len(data["candidate_sources"]))]
    for target_idx, facilities in enumerate(ij_lists):
        for facility in facilities:
            ji_lists[int(facility)].append(target_idx)
    ji_arrays = [np.asarray(values, dtype=np.int32) for values in ji_lists]
    return mch.build_instance(
        weights,
        ij_lists,
        ji_arrays,
        assume_unique_sorted=True,
    )


def curve_rows_from_lifted(
    *,
    case: str,
    country: str,
    grid: str,
    threshold_km: float,
    instance: mch.MaxCoverInstance,
    baseline_population: float,
    total_population: float,
    weight_scale: float,
) -> tuple[list[dict], dict]:
    t0 = perf_counter()
    greedy_result, reduced_result, restricted, lifted_result = mch.greedy_drop_greedy(instance)
    rows: list[dict] = []
    for budget, objective in enumerate(lifted_result.objectives):
        incremental = float(objective) / weight_scale
        total = baseline_population + incremental
        rows.append(
            {
                "case": case,
                "country": country,
                "grid": grid,
                "threshold_km": threshold_km,
                "budget": budget,
                "incremental_population": incremental,
                "covered_population": total,
                "coverage_percent": 100.0 * total / total_population if total_population else math.nan,
                "method": "approximated_tradeoff_greedy_drop_greedy",
                "seconds": lifted_result.times[budget] if budget < len(lifted_result.times) else math.nan,
            }
        )
    stats = {
        "case": case,
        "country": country,
        "grid": grid,
        "threshold_km": threshold_km,
        "n_population": int(instance.n_households),
        "n_candidates": int(instance.n_facilities),
        "greedy_facilities": int(len(greedy_result.solution)),
        "reduced_facilities": int(len(reduced_result.solution)),
        "lifted_facilities": int(len(lifted_result.solution)),
        "restricted_facilities": int(restricted.instance.n_facilities),
        "max_incremental_population": float(lifted_result.objective) / weight_scale,
        "baseline_population": float(baseline_population),
        "total_population": float(total_population),
        "total_seconds": float(perf_counter() - t0),
    }
    return rows, stats


def solve_timor_exact(
    *,
    data: dict,
    budgets: list[int],
    output_dir: Path,
    time_limit_seconds: float,
    mip_gap: float,
) -> list[dict]:
    log_path = output_dir / "timor_leste_approx_tradeoff_exact.log"
    n_candidates = len(data["candidate_sources"])
    n_targets = len(data["target_ids"])
    weights = np.asarray(data["weights"], dtype=float)
    lists = data["target_candidate_lists"]

    t0 = perf_counter()
    model = gp.Model("timor_leste_exact_curve")
    model.Params.LogFile = str(log_path)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_targets, vtype=GRB.BINARY, name="y")
    budget_constr = model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) <= 0, name="budget")
    for i, facilities in enumerate(lists):
        model.addConstr(y[i] <= gp.quicksum(x[j] for j in facilities), name=f"cover_{i}")
    model.setObjective(gp.quicksum(float(weights[i]) * y[i] for i in range(n_targets)), GRB.MAXIMIZE)
    model.update()
    modeling_seconds = perf_counter() - t0

    rows = []
    for budget in sorted(set(int(value) for value in budgets if int(value) >= 0)):
        budget_constr.RHS = int(budget)
        t1 = perf_counter()
        model.optimize()
        solving_seconds = perf_counter() - t1
        incremental = float(model.ObjVal) if model.SolCount else math.nan
        covered = float(data["baseline_population"] + incremental)
        rows.append(
            {
                "case": "timor_leste_2p5km_5km",
                "country": "Timor-Leste",
                "grid": "2.5 km",
                "threshold_km": 5.0,
                "budget": int(budget),
                "incremental_population": incremental,
                "covered_population": covered,
                "coverage_percent": 100.0 * covered / float(data["total_population"]),
                "method": "gurobi_exact_optimum",
                "status_name": {
                    GRB.OPTIMAL: "optimal",
                    GRB.TIME_LIMIT: "time_limit",
                    GRB.INFEASIBLE: "infeasible",
                    GRB.INTERRUPTED: "interrupted",
                }.get(model.Status, str(model.Status)),
                "modeling_seconds": float(modeling_seconds),
                "solving_seconds": float(solving_seconds),
                "mip_gap": None if model.MIPGap == GRB.INFINITY else float(model.MIPGap),
                "objective_bound": None if model.ObjBound == GRB.INFINITY else float(model.ObjBound),
                "log_path": str(log_path),
            }
        )
    return rows


def read_best_vietnam_heuristic(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    df = df.sort_values(
        ["threshold_km", "budget", "total_covered_population", "seconds"],
        ascending=[True, True, False, True],
    )
    best = df.groupby(["threshold_km", "budget"], as_index=False).head(1).copy()
    best["case"] = best["threshold_km"].map(lambda value: f"vietnam_10km_{value:g}km")
    best["country"] = "Vietnam"
    best["grid"] = "10 km"
    best["method"] = "best_existing_heuristic"
    best["coverage_percent"] = best["coverage_percent_total_population"]
    best["covered_population"] = best["total_covered_population"]
    return best[
        [
            "case",
            "country",
            "grid",
            "threshold_km",
            "budget",
            "covered_population",
            "coverage_percent",
            "method",
            "seconds",
        ]
    ].to_dict("records")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "approx_tradeoff")
    parser.add_argument(
        "--timor-fresh-root",
        type=Path,
        default=ROOT / "runs" / "TimorLeste_20260618_220002",
    )
    parser.add_argument(
        "--vietnam-output-root",
        type=Path,
        default=ROOT / "outputs" / "vietnam_20260619_0630",
    )
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--exact-time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--exact-mip-gap", type=float, default=1e-6)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Timor-Leste: same 2.5 km candidate instance used for the 174/175 exact check.
    timor_outputs = args.timor_fresh_root / "east-timor_data" / "outputs"
    timor_grid25_manifest = find_manifest(timor_outputs, has_candidates=True, spacing=2500.0)
    timor_data = build_gurobi_data(timor_grid25_manifest)
    timor_instance = build_instance_from_timor_data(timor_data, args.weight_scale)
    timor_curve, timor_stats = curve_rows_from_lifted(
        case="timor_leste_2p5km_5km",
        country="Timor-Leste",
        grid="2.5 km",
        threshold_km=5.0,
        instance=timor_instance,
        baseline_population=float(timor_data["baseline_population"]),
        total_population=float(timor_data["total_population"]),
        weight_scale=float(args.weight_scale),
    )
    exact_budgets = [0, 25, 50, 75, 100, 125, 150, 174, 175, 200]
    timor_exact = solve_timor_exact(
        data=timor_data,
        budgets=exact_budgets,
        output_dir=out,
        time_limit_seconds=float(args.exact_time_limit_seconds),
        mip_gap=float(args.exact_mip_gap),
    )

    # Vietnam: 10 km network-distance instances already built for this deck.
    vietnam_rows: list[dict] = []
    vietnam_stats: list[dict] = []
    for threshold in [20, 50, 100]:
        npz = args.vietnam_output_root / "optimization" / f"vietnam_10kmgrid_{threshold}km_threshold.npz"
        instance, metadata = load_vietnam_npz(npz)
        rows, stats = curve_rows_from_lifted(
            case=f"vietnam_10km_{threshold}km",
            country="Vietnam",
            grid="10 km",
            threshold_km=float(threshold),
            instance=instance,
            baseline_population=float(metadata["baseline_covered_population"]),
            total_population=float(metadata["total_population"]),
            weight_scale=float(metadata.get("weight_scale", args.weight_scale)),
        )
        vietnam_rows.extend(rows)
        vietnam_stats.append(stats)

    vietnam_best = read_best_vietnam_heuristic(
        args.vietnam_output_root / "fleur_style_10km_network" / "coverage_summary_by_budget.csv"
    )

    write_csv(out / "timor_leste_approx_curve.csv", timor_curve)
    write_csv(out / "timor_leste_exact_curve.csv", timor_exact)
    write_csv(out / "vietnam_10km_approx_curves.csv", vietnam_rows)
    write_csv(out / "vietnam_10km_best_existing_heuristic.csv", vietnam_best)

    approx_by_budget = {int(row["budget"]): row for row in timor_curve}
    comparison = []
    for exact in timor_exact:
        budget = int(exact["budget"])
        approx = approx_by_budget.get(budget)
        if approx is None:
            continue
        gap = float(exact["coverage_percent"]) - float(approx["coverage_percent"])
        comparison.append(
            {
                "case": exact["case"],
                "budget": budget,
                "exact_coverage_percent": exact["coverage_percent"],
                "approx_coverage_percent": approx["coverage_percent"],
                "gap_percentage_points": gap,
                "exact_status": exact["status_name"],
            }
        )
    write_csv(out / "timor_leste_approx_vs_exact.csv", comparison)

    manifest = {
        "generated_utc_note": "Local run date is recorded by filesystem metadata.",
        "approximated_tradeoff_source": str(APPROX_SRC.parent),
        "approximated_tradeoff_github": (
            "https://github.com/Analytics-for-a-Better-World/"
            "Public-Infrastructure-Service-Access/tree/main/Research-Sandbox/approximated_tradeoff"
        ),
        "method": "greedy_drop_greedy from approximated_tradeoff notebook/source",
        "timor": timor_stats,
        "vietnam": vietnam_stats,
        "exact_optimum_computed_for": ["Timor-Leste 2.5 km candidate grid, 5 km threshold"],
        "exact_optimum_not_computed_for": [
            "Vietnam national 10 km network-distance thresholds; full exact curve was not available in the local outputs"
        ],
        "outputs": {
            "timor_leste_approx_curve": str(out / "timor_leste_approx_curve.csv"),
            "timor_leste_exact_curve": str(out / "timor_leste_exact_curve.csv"),
            "timor_leste_approx_vs_exact": str(out / "timor_leste_approx_vs_exact.csv"),
            "vietnam_10km_approx_curves": str(out / "vietnam_10km_approx_curves.csv"),
            "vietnam_10km_best_existing_heuristic": str(out / "vietnam_10km_best_existing_heuristic.csv"),
        },
    }
    (out / "approx_tradeoff_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
