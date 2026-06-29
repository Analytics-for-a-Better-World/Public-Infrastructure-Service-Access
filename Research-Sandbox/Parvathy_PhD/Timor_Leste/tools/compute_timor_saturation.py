from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_timor_network_profiles_abw import (  # noqa: E402
    build_instance_from_manifest,
    clock_ms,
    load_manifests,
)

from abw_maxcover._incremental_core import budgeted_construct  # noqa: E402


PRIMARY_CASES = [
    "timor_drive_only_unsimplified_10km",
    "timor_drive_only_unsimplified_5km",
    "timor_drive_only_unsimplified_1km",
    "timor_drive_plus_walk_unsimplified_10km",
    "timor_drive_plus_walk_unsimplified_5km",
    "timor_drive_plus_walk_unsimplified_1km",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_greedy_saturation(instance: Any) -> dict[str, Any]:
    start = perf_counter()
    result = budgeted_construct(instance, instance.n_facilities, constructor="greedy", seed=42)
    elapsed = perf_counter() - start
    return {
        "greedy_saturation_budget": len(result.solution),
        "greedy_saturation_objective": int(result.objective),
        "greedy_saturation_seconds": elapsed,
        "greedy_saturation_clock_ms": clock_ms(elapsed),
        "greedy_steps_recorded": len(result.objectives) - 1,
    }


def build_cover_csr(instance: Any):
    import scipy.sparse as sp

    demand = instance.demand_with_candidates()
    lengths = instance.ij_indptr[demand + 1] - instance.ij_indptr[demand]
    row_index = np.repeat(np.arange(demand.size, dtype=np.int32), lengths.astype(np.int64))
    col_index = np.empty(int(lengths.sum()), dtype=np.int32)
    pos = 0
    for demand_i, width in zip(demand, lengths):
        start = int(instance.ij_indptr[int(demand_i)])
        end = int(instance.ij_indptr[int(demand_i) + 1])
        next_pos = pos + int(width)
        col_index[pos:next_pos] = instance.ij_indices[start:end]
        pos = next_pos
    data = np.ones(col_index.size, dtype=np.float64)
    matrix = sp.csr_matrix((data, (row_index, col_index)), shape=(demand.size, instance.n_facilities))
    return matrix


def compute_exact_set_cover_saturation(instance: Any, *, time_limit: float, threads: int) -> dict[str, Any]:
    start = perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return {
            "exact_saturation_status": f"import_failed: {exc}",
            "exact_saturation_seconds": perf_counter() - start,
        }

    matrix = build_cover_csr(instance)
    model = gp.Model("min_facilities_for_full_cover")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.Threads = int(threads)
    model.Params.MIPGap = 0.0
    x = model.addMVar(shape=instance.n_facilities, vtype=GRB.BINARY, name="x")
    model.setObjective(x.sum(), GRB.MINIMIZE)
    model.addMConstr(matrix, x, GRB.GREATER_EQUAL, np.ones(matrix.shape[0], dtype=np.float64))
    model.optimize()
    elapsed = perf_counter() - start
    status_name = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INTERRUPTED: "interrupted",
    }.get(model.Status, str(model.Status))
    budget = None
    selected_count = None
    if model.SolCount:
        selected = np.flatnonzero(np.asarray(x.X) > 0.5)
        selected_count = int(selected.size)
        budget = int(round(float(model.ObjVal)))
    return {
        "exact_saturation_status": status_name,
        "exact_saturation_budget": budget,
        "exact_saturation_selected_count": selected_count,
        "exact_saturation_bound": None if math.isnan(getattr(model, "ObjBound", math.nan)) else float(model.ObjBound),
        "exact_saturation_mip_gap": None if math.isnan(getattr(model, "MIPGap", math.nan)) else float(model.MIPGap),
        "exact_saturation_seconds": elapsed,
        "exact_saturation_clock_ms": clock_ms(elapsed),
        "exact_saturation_constraints": int(matrix.shape[0]),
        "exact_saturation_nnz": int(matrix.nnz),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("runs/timor_network_profile_20260623/east-timor_data/outputs"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timor_saturation_20260625"))
    parser.add_argument("--weight-scale", type=int, default=1000)
    parser.add_argument("--cases", nargs="*", default=PRIMARY_CASES)
    parser.add_argument("--exact-time-limit", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests(args.outputs_dir)
    rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        if case_id not in manifests:
            raise KeyError(f"case not found: {case_id}")
        manifest_path, manifest = manifests[case_id]
        print(f"Building {case_id}", flush=True)
        instance, stats, _, _ = build_instance_from_manifest(
            case_id,
            manifest_path,
            manifest,
            weight_scale=int(args.weight_scale),
        )
        row = {
            "case_id": case_id,
            "network_profile": stats["network_profile"],
            "simplify_network": stats["simplify_network"],
            "candidate_grid_spacing_m": stats["candidate_grid_spacing_m"],
            "n_candidates": instance.n_facilities,
            "n_demand": instance.n_demand,
            "n_coverable_uncovered_demand": stats["n_coverable_uncovered_population_points"],
            "n_arcs": instance.ji_indices.size,
            "baseline_population": stats["baseline_population"],
            "total_population": stats["total_population"],
            "baseline_percent": stats["baseline_percent"],
            "all_candidates_incremental_population": stats["all_candidates_incremental_population"],
            "all_candidates_total_population": stats["all_candidates_total_population"],
            "all_candidates_coverage_percent": stats["all_candidates_coverage_percent"],
            "build_seconds": stats["build_seconds"],
            "build_clock_ms": stats["build_clock_ms"],
        }
        row.update(compute_greedy_saturation(instance))
        if not args.skip_exact:
            row.update(
                compute_exact_set_cover_saturation(
                    instance,
                    time_limit=float(args.exact_time_limit),
                    threads=int(args.threads),
                )
            )
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)
        write_csv(args.output_dir / "timor_primary_saturation_summary.csv", rows)
    write_csv(args.output_dir / "timor_primary_saturation_summary.csv", rows)


if __name__ == "__main__":
    main()
