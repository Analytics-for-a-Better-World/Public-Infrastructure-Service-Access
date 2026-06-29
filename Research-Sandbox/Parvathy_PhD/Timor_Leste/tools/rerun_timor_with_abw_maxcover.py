from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abw_maxcover import (
    GurobiConfig,
    HeuristicConfig,
    approximate_pareto_curve,
    build_instance,
    exact_pareto_curve,
)
from abw_maxcover.results import MaxCoverCurve, MaxCoverResult


PROJECTED_EPSG = 32751
SERVICE_THRESHOLD_M = 5000.0
GRID_LABELS = {10000: "10 km", 5000: "5 km", 1000: "1 km"}
DEFAULT_EXACT_MAX_BUDGETS = {10000: 80, 5000: 105, 1000: 80}
DEFAULT_SELECTED_BUDGETS = [16, 32, 48, 64, 80]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolved(manifest: dict) -> dict:
    return manifest.get("resolved_parameters") or manifest.get("parameters", {}).get("resolved", {})


def resolve_copied_output(path_value: str, outputs_dir: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    copied = outputs_dir / path.name
    if copied.exists():
        return copied
    raise FileNotFoundError(f"Cannot resolve manifest output path: {path_value}")


def find_baseline_manifest(outputs_dir: Path) -> Path:
    candidates: list[Path] = []
    for path in outputs_dir.glob("run_manifest*.yaml"):
        manifest = read_yaml(path)
        params = resolved(manifest)
        if bool(params.get("has_candidates")) is False and params.get("candidate_grid_spacing_m") is None:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No baseline no-candidate manifest found in {outputs_dir}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def project_lon_lat(lon: Iterable[float], lat: Iterable[float]) -> np.ndarray:
    transformer = Transformer.from_crs(4326, PROJECTED_EPSG, always_xy=True)
    x, y = transformer.transform(np.asarray(list(lon), dtype=float), np.asarray(list(lat), dtype=float))
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def dataframe_xy(frame: pd.DataFrame) -> np.ndarray:
    if "geometry" in frame.columns and getattr(frame, "crs", None) is not None:
        projected = frame if frame.crs.to_epsg() == PROJECTED_EPSG else frame.to_crs(PROJECTED_EPSG)
        return np.column_stack([projected.geometry.x.to_numpy(dtype=float), projected.geometry.y.to_numpy(dtype=float)])
    return project_lon_lat(frame["Longitude"].to_numpy(dtype=float), frame["Latitude"].to_numpy(dtype=float))


def load_candidate_grid(fresh_root: Path, spacing: int) -> pd.DataFrame:
    path = (
        fresh_root
        / "east-timor_data"
        / "cache"
        / f"tls_candidate_sites_spacing_{spacing}m_no_water_include_boundary_epsg_32751.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_pickle(path).reset_index(drop=True)


def build_timor_data(fresh_root: Path, spacing: int) -> tuple[dict, dict]:
    outputs_dir = fresh_root / "east-timor_data" / "outputs"
    manifest_path = find_baseline_manifest(outputs_dir)
    manifest = read_yaml(manifest_path)
    population_path = resolve_copied_output(manifest["outputs"]["population"]["path"], outputs_dir)
    sources_path = resolve_copied_output(manifest["outputs"]["sources"]["path"], outputs_dir)

    population = pd.read_parquet(population_path).reset_index(drop=True)
    existing = pd.read_parquet(sources_path).reset_index(drop=True)
    candidates = load_candidate_grid(fresh_root, spacing)

    population["ID"] = population["ID"].astype(str)
    pop_xy = dataframe_xy(population)
    existing_xy = dataframe_xy(existing)
    candidate_xy = dataframe_xy(candidates)

    pop_tree = cKDTree(pop_xy)
    baseline_mask = np.zeros(len(population), dtype=bool)
    if len(existing_xy):
        for covered in pop_tree.query_ball_point(existing_xy, SERVICE_THRESHOLD_M):
            if covered:
                baseline_mask[np.asarray(covered, dtype=np.int64)] = True

    candidate_tree = cKDTree(candidate_xy)
    uncovered_indices = np.flatnonzero(~baseline_mask)
    candidate_lists_raw = candidate_tree.query_ball_point(pop_xy[uncovered_indices], SERVICE_THRESHOLD_M)

    target_ids: list[str] = []
    target_candidate_lists: list[np.ndarray] = []
    weights: list[float] = []
    for pop_idx, facilities in zip(uncovered_indices, candidate_lists_raw):
        if not facilities:
            continue
        target_ids.append(str(population.at[int(pop_idx), "ID"]))
        target_candidate_lists.append(np.asarray(sorted(int(value) for value in facilities), dtype=np.int32))
        weights.append(float(population.at[int(pop_idx), "population"]))

    if "ID" in candidates.columns:
        raw_candidate_ids = candidates["ID"].astype(str).to_list()
    else:
        raw_candidate_ids = [str(value) for value in range(len(candidates))]

    total_population = float(population["population"].sum())
    baseline_population = float(population.loc[baseline_mask, "population"].sum())
    all_candidate_incremental = float(np.asarray(weights, dtype=float).sum())
    data = {
        "baseline_manifest": str(manifest_path),
        "population_path": str(population_path),
        "sources_path": str(sources_path),
        "candidate_sources": [f"source_candidates_{value}" for value in raw_candidate_ids],
        "target_ids": target_ids,
        "weights_float": np.asarray(weights, dtype=float),
        "target_candidate_lists": target_candidate_lists,
        "total_population": total_population,
        "baseline_population": baseline_population,
        "baseline_share": baseline_population / total_population if total_population else math.nan,
        "existing_covered_points": int(baseline_mask.sum()),
        "candidate_edge_rows": int(sum(len(values) for values in target_candidate_lists)),
    }
    stats = {
        "grid": GRID_LABELS[spacing],
        "spacing_m": int(spacing),
        "n_candidates": int(len(candidates)),
        "n_targets_uncovered_coverable": int(len(target_candidate_lists)),
        "candidate_edge_rows": int(data["candidate_edge_rows"]),
        "population_points": int(len(population)),
        "retained_population": total_population,
        "existing_sources": int(len(existing)),
        "existing_covered_points": int(data["existing_covered_points"]),
        "existing_covered_population": baseline_population,
        "existing_covered_share": float(data["baseline_share"]),
        "all_candidate_covered_population": baseline_population + all_candidate_incremental,
        "all_candidate_coverage_percent": coverage_percent(
            baseline_population + all_candidate_incremental,
            total_population,
        ),
    }
    return data, stats


def build_abw_instance(data: dict, *, spacing: int, weight_scale: float):
    weights_scaled = np.rint(np.asarray(data["weights_float"], dtype=float) * weight_scale).astype(np.int64)
    ij_lists = [np.asarray(values, dtype=np.int32) for values in data["target_candidate_lists"]]
    ji_lists: list[list[int]] = [[] for _ in range(len(data["candidate_sources"]))]
    for demand_idx, facilities in enumerate(ij_lists):
        for facility in facilities:
            ji_lists[int(facility)].append(int(demand_idx))
    ji_arrays = [np.asarray(values, dtype=np.int32) for values in ji_lists]
    return build_instance(
        weights_scaled,
        ij_lists,
        ji_arrays,
        name=f"timor_leste_grid{spacing // 1000}km_5km_straightline",
        n_facilities=len(data["candidate_sources"]),
        assume_unique_sorted=True,
        validate_consistency=False,
        metadata={
            "country": "Timor-Leste",
            "grid": GRID_LABELS[spacing],
            "spacing_m": int(spacing),
            "threshold_m": SERVICE_THRESHOLD_M,
            "weight_scale": float(weight_scale),
            "baseline_population": float(data["baseline_population"]),
            "total_population": float(data["total_population"]),
        },
    )


def coverage_percent(covered_population: float, total_population: float) -> float:
    return 100.0 * float(covered_population) / float(total_population) if total_population else math.nan


def objective_population(data: dict, result: MaxCoverResult) -> float:
    if result.coverage is None:
        return math.nan
    return float(np.asarray(data["weights_float"], dtype=float)[result.coverage > 0].sum())


def curve_rows(
    *,
    curve: MaxCoverCurve,
    data: dict,
    grid: str,
    method_family: str,
    exact_cumulative: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    cumulative_solve = 0.0
    for result in curve.results:
        incremental = objective_population(data, result)
        if result.coverage is None and result.objective is not None:
            incremental = float(result.objective) / float(curve.metadata.get("weight_scale", 1.0))
        covered = float(data["baseline_population"] + incremental) if math.isfinite(incremental) else math.nan
        cumulative_solve += float(result.solve_seconds)
        rows.append(
            {
                "case": curve.instance_name,
                "country": "Timor-Leste",
                "grid": grid,
                "threshold_km": 5.0,
                "budget": int(result.budget),
                "method_family": method_family,
                "method": result.method,
                "status": result.status,
                "incremental_population": incremental,
                "covered_population": covered,
                "coverage_percent": coverage_percent(covered, float(data["total_population"])),
                "objective_weight_units": result.objective,
                "upper_bound_weight_units": result.upper_bound,
                "mip_gap": result.mip_gap,
                "selected_count": result.selected_count,
                "model_seconds": result.model_seconds,
                "solve_seconds": result.solve_seconds,
                "total_seconds": result.total_seconds,
                "cumulative_exact_seconds": result.model_seconds + cumulative_solve if exact_cumulative else math.nan,
                "construction_objective": result.construction_objective,
                "construction_seconds": result.construction_seconds,
                "local_search_moves": result.local_search_moves,
                "seed": result.seed,
                "repeat": result.repeat,
            }
        )
    return rows


def exact_stats_from_rows(stats: dict, rows: list[dict]) -> dict:
    max_coverage = float(stats["all_candidate_coverage_percent"])
    saturated = [
        row
        for row in rows
        if math.isfinite(float(row["coverage_percent"]))
        and float(row["coverage_percent"]) >= max_coverage - 1e-6
    ]
    exact_saturation_budget = None if not saturated else int(saturated[0]["budget"])
    total_seconds = 0.0 if not rows else float(rows[-1]["cumulative_exact_seconds"])
    return {
        **stats,
        "exact_saturation_budget": exact_saturation_budget,
        "exact_rows": int(len(rows)),
        "exact_total_seconds": total_seconds,
    }


def progress_printer(label: str):
    def progress(values):
        for value in values:
            print(f"{label}: solving budget {value}", flush=True)
            yield value

    return progress


def run_grid(
    *,
    fresh_root: Path,
    output_dir: Path,
    spacing: int,
    exact_budgets: list[int],
    selected_budgets: list[int],
    weight_scale: float,
    exact_time_limit_seconds: float,
    exact_mip_gap: float,
    heuristic_config: HeuristicConfig,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    grid = GRID_LABELS[spacing]
    print(f"\nBuilding {grid} Timor-Leste instance", flush=True)
    data, stats = build_timor_data(fresh_root, spacing)
    instance = build_abw_instance(data, spacing=spacing, weight_scale=weight_scale)
    stats.update(
        {
            "n_candidates": int(instance.n_facilities),
            "n_targets_uncovered_coverable": int(instance.n_demand),
            "scaled_total_weight": int(instance.total_weight),
            "weight_scale": float(weight_scale),
        }
    )
    print(
        f"{grid}: {instance.n_facilities} candidates, {instance.n_demand} demand rows, "
        f"{stats['candidate_edge_rows']} edges",
        flush=True,
    )

    log_file = output_dir / "gurobi_logs" / f"{instance.name}_abw_exact.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    exact_curve = exact_pareto_curve(
        instance,
        exact_budgets,
        solver="gurobi",
        gurobi_config=GurobiConfig(
            time_limit_seconds=float(exact_time_limit_seconds),
            mip_gap=float(exact_mip_gap),
            trace=False,
            log_file=str(log_file),
            warm_start=True,
        ),
        progress=progress_printer(f"{grid} exact"),
    )
    exact_curve.metadata["weight_scale"] = float(weight_scale)
    exact_rows = curve_rows(curve=exact_curve, data=data, grid=grid, method_family="abw_gurobi_exact", exact_cumulative=True)
    stats = exact_stats_from_rows(stats, exact_rows)

    print(f"{grid}: running new heuristic approximation on selected budgets {selected_budgets}", flush=True)
    heuristic_all_curve = approximate_pareto_curve(
        instance,
        selected_budgets,
        config=heuristic_config,
        select_best=False,
        progress=progress_printer(f"{grid} heuristic"),
    )
    heuristic_all_curve.metadata["weight_scale"] = float(weight_scale)
    heuristic_best_curve = approximate_pareto_curve(
        instance,
        selected_budgets,
        config=heuristic_config,
        select_best=True,
        progress=lambda values: values,
    )
    heuristic_best_curve.metadata["weight_scale"] = float(weight_scale)
    heuristic_all_rows = curve_rows(
        curve=heuristic_all_curve,
        data=data,
        grid=grid,
        method_family="abw_heuristic_all",
    )
    heuristic_best_rows = curve_rows(
        curve=heuristic_best_curve,
        data=data,
        grid=grid,
        method_family="abw_heuristic_best",
    )
    return exact_rows, stats, heuristic_all_rows, heuristic_best_rows


def rows_by_grid_budget(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(str(row["grid"]), int(row["budget"])): row for row in rows}


def build_selected_comparison(
    *,
    selected_budgets: list[int],
    exact_rows: list[dict],
    heuristic_best_rows: list[dict],
) -> list[dict]:
    exact = rows_by_grid_budget(exact_rows)
    heuristic = rows_by_grid_budget(heuristic_best_rows)
    rows: list[dict] = []
    for budget in selected_budgets:
        row10 = exact.get(("10 km", budget))
        row5 = exact.get(("5 km", budget))
        row1 = exact.get(("1 km", budget))
        heuristic1 = heuristic.get(("1 km", budget))
        rows.append(
            {
                "budget": int(budget),
                "timor_10km_exact_coverage_percent": None if row10 is None else row10["coverage_percent"],
                "timor_10km_exact_seconds": None if row10 is None else row10["solve_seconds"],
                "timor_5km_exact_coverage_percent": None if row5 is None else row5["coverage_percent"],
                "timor_5km_exact_seconds": None if row5 is None else row5["solve_seconds"],
                "timor_1km_exact_coverage_percent": None if row1 is None else row1["coverage_percent"],
                "timor_1km_exact_seconds": None if row1 is None else row1["solve_seconds"],
                "timor_1km_best_heuristic_coverage_percent": None if heuristic1 is None else heuristic1["coverage_percent"],
                "timor_1km_best_heuristic_seconds": None if heuristic1 is None else heuristic1["total_seconds"],
                "timor_1km_best_heuristic_method": None if heuristic1 is None else heuristic1["method"],
                "timor_1km_exact_minus_heuristic_percentage_points": (
                    None
                    if row1 is None or heuristic1 is None
                    else float(row1["coverage_percent"]) - float(heuristic1["coverage_percent"])
                ),
                "gain_10km_to_5km_percentage_points": (
                    None
                    if row10 is None or row5 is None
                    else float(row5["coverage_percent"]) - float(row10["coverage_percent"])
                ),
                "gain_5km_to_1km_percentage_points": (
                    None
                    if row5 is None or row1 is None
                    else float(row1["coverage_percent"]) - float(row5["coverage_percent"])
                ),
            }
        )
    return rows


def compare_with_previous(output_dir: Path, selected_rows: list[dict], stats_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    previous_dir = ROOT / "outputs" / "timor_three_grid_experiments"
    previous_selected_path = previous_dir / "timor_selected_budget_comparison.csv"
    previous_stats_path = previous_dir / "timor_exact_unit_stats_10km_5km_1km.csv"

    selected_compare: list[dict] = []
    if previous_selected_path.exists():
        previous = pd.read_csv(previous_selected_path).set_index("budget")
        for row in selected_rows:
            budget = int(row["budget"])
            record = {"budget": budget}
            if budget not in previous.index:
                continue
            old = previous.loc[budget]
            for grid_key in ("10km", "5km", "1km"):
                old_col = f"timor_{grid_key}_exact_coverage_percent"
                new_col = f"timor_{grid_key}_exact_coverage_percent"
                record[f"previous_{old_col}"] = float(old[old_col])
                record[f"new_{new_col}"] = float(row[new_col])
                record[f"delta_{grid_key}_exact_percentage_points"] = float(row[new_col]) - float(old[old_col])
            record["previous_1km_best_heuristic_coverage_percent"] = float(
                old["timor_1km_best_heuristic_coverage_percent"]
            )
            record["new_1km_best_heuristic_coverage_percent"] = float(
                row["timor_1km_best_heuristic_coverage_percent"]
            )
            record["delta_1km_best_heuristic_percentage_points"] = float(
                row["timor_1km_best_heuristic_coverage_percent"]
            ) - float(old["timor_1km_best_heuristic_coverage_percent"])
            selected_compare.append(record)

    stats_compare: list[dict] = []
    if previous_stats_path.exists():
        previous_stats = pd.read_csv(previous_stats_path).set_index("grid")
        for row in stats_rows:
            grid = str(row["grid"])
            if grid not in previous_stats.index:
                continue
            old = previous_stats.loc[grid]
            stats_compare.append(
                {
                    "grid": grid,
                    "previous_n_candidates": int(old["n_candidates"]),
                    "new_n_candidates": int(row["n_candidates"]),
                    "previous_candidate_edge_rows": int(old["candidate_edge_rows"]),
                    "new_candidate_edge_rows": int(row["candidate_edge_rows"]),
                    "previous_all_candidate_coverage_percent": float(old["all_candidate_coverage_percent"]),
                    "new_all_candidate_coverage_percent": float(row["all_candidate_coverage_percent"]),
                    "delta_all_candidate_coverage_percentage_points": float(row["all_candidate_coverage_percent"])
                    - float(old["all_candidate_coverage_percent"]),
                    "previous_exact_saturation_budget": int(old["exact_saturation_budget"]),
                    "new_exact_saturation_budget": int(row["exact_saturation_budget"]),
                    "previous_exact_total_seconds": float(old["exact_total_seconds"]),
                    "new_exact_total_seconds": float(row["exact_total_seconds"]),
                }
            )
    return selected_compare, stats_compare


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def format_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    *,
    path: Path,
    stats_rows: list[dict],
    selected_rows: list[dict],
    selected_compare_rows: list[dict],
    stats_compare_rows: list[dict],
) -> None:
    lines = [
        "# Timor-Leste rerun with abw_maxcover",
        "",
        "This rerun rebuilds the straight-line 5 km service instances from the cached Timor-Leste "
        "10 km, 5 km, and 1 km candidate grids, then solves them through the new `abw_maxcover` API.",
        "",
        "## Instance and exact curve summary",
        "",
        markdown_table(
            stats_rows,
            [
                "grid",
                "n_candidates",
                "candidate_edge_rows",
                "all_candidate_coverage_percent",
                "exact_saturation_budget",
                "exact_total_seconds",
            ],
        ),
        "",
        "## Selected budget comparison",
        "",
        markdown_table(
            selected_rows,
            [
                "budget",
                "timor_10km_exact_coverage_percent",
                "timor_5km_exact_coverage_percent",
                "timor_1km_exact_coverage_percent",
                "gain_10km_to_5km_percentage_points",
                "gain_5km_to_1km_percentage_points",
                "timor_1km_best_heuristic_coverage_percent",
                "timor_1km_exact_minus_heuristic_percentage_points",
                "timor_1km_best_heuristic_method",
            ],
        ),
        "",
        "## New exact values minus previous exact values",
        "",
        markdown_table(
            selected_compare_rows,
            [
                "budget",
                "delta_10km_exact_percentage_points",
                "delta_5km_exact_percentage_points",
                "delta_1km_exact_percentage_points",
                "delta_1km_best_heuristic_percentage_points",
            ],
        ),
        "",
        "## Instance-level comparison with previous run",
        "",
        markdown_table(
            stats_compare_rows,
            [
                "grid",
                "previous_n_candidates",
                "new_n_candidates",
                "previous_candidate_edge_rows",
                "new_candidate_edge_rows",
                "delta_all_candidate_coverage_percentage_points",
                "previous_exact_saturation_budget",
                "new_exact_saturation_budget",
                "previous_exact_total_seconds",
                "new_exact_total_seconds",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", type=Path, default=ROOT / "runs" / "TimorLeste_20260618_220002")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "timor_abw_maxcover_rerun")
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--exact-time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--exact-mip-gap", type=float, default=1e-6)
    parser.add_argument("--selected-budgets", nargs="+", type=int, default=DEFAULT_SELECTED_BUDGETS)
    parser.add_argument("--spacings", nargs="+", type=int, default=[10000, 5000, 1000])
    parser.add_argument("--heuristic-randomized-repeats", type=int, default=3)
    parser.add_argument("--heuristic-rcl-size", type=int, default=25)
    parser.add_argument("--heuristic-sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_budgets = [int(value) for value in args.selected_budgets]
    heuristic_config = HeuristicConfig(
        randomized_repeats=int(args.heuristic_randomized_repeats),
        rcl_size=int(args.heuristic_rcl_size),
        sample_size=int(args.heuristic_sample_size),
        seed=int(args.seed),
    )

    t0 = perf_counter()
    exact_rows: list[dict] = []
    stats_rows: list[dict] = []
    heuristic_all_rows: list[dict] = []
    heuristic_best_rows: list[dict] = []

    for spacing in args.spacings:
        max_budget = DEFAULT_EXACT_MAX_BUDGETS[int(spacing)]
        grid_exact_rows, stats, grid_heuristic_all, grid_heuristic_best = run_grid(
            fresh_root=args.fresh_root,
            output_dir=args.output_dir,
            spacing=int(spacing),
            exact_budgets=list(range(max_budget + 1)),
            selected_budgets=selected_budgets,
            weight_scale=float(args.weight_scale),
            exact_time_limit_seconds=float(args.exact_time_limit_seconds),
            exact_mip_gap=float(args.exact_mip_gap),
            heuristic_config=heuristic_config,
        )
        exact_rows.extend(grid_exact_rows)
        stats_rows.append(stats)
        heuristic_all_rows.extend(grid_heuristic_all)
        heuristic_best_rows.extend(grid_heuristic_best)

    selected_rows = build_selected_comparison(
        selected_budgets=selected_budgets,
        exact_rows=exact_rows,
        heuristic_best_rows=heuristic_best_rows,
    )
    selected_compare_rows, stats_compare_rows = compare_with_previous(args.output_dir, selected_rows, stats_rows)

    write_csv(args.output_dir / "timor_abw_exact_curves.csv", exact_rows)
    write_csv(args.output_dir / "timor_abw_exact_stats.csv", stats_rows)
    write_csv(args.output_dir / "timor_abw_heuristics_selected_all.csv", heuristic_all_rows)
    write_csv(args.output_dir / "timor_abw_heuristics_selected_best.csv", heuristic_best_rows)
    write_csv(args.output_dir / "timor_abw_selected_budget_comparison.csv", selected_rows)
    write_csv(args.output_dir / "timor_abw_previous_selected_delta.csv", selected_compare_rows)
    write_csv(args.output_dir / "timor_abw_previous_stats_delta.csv", stats_compare_rows)
    write_report(
        path=args.output_dir / "timor_abw_rerun_report.md",
        stats_rows=stats_rows,
        selected_rows=selected_rows,
        selected_compare_rows=selected_compare_rows,
        stats_compare_rows=stats_compare_rows,
    )

    manifest = {
        "fresh_root": str(args.fresh_root),
        "output_dir": str(args.output_dir),
        "spacings_m": [int(value) for value in args.spacings],
        "selected_budgets": selected_budgets,
        "exact_max_budgets": {str(key): value for key, value in DEFAULT_EXACT_MAX_BUDGETS.items()},
        "threshold_m": SERVICE_THRESHOLD_M,
        "distance_model": "straight_line_projected_screening",
        "optimization_package": "abw_maxcover",
        "weight_scale": float(args.weight_scale),
        "elapsed_seconds": float(perf_counter() - t0),
        "outputs": {
            "exact_curves": str(args.output_dir / "timor_abw_exact_curves.csv"),
            "exact_stats": str(args.output_dir / "timor_abw_exact_stats.csv"),
            "heuristics_all": str(args.output_dir / "timor_abw_heuristics_selected_all.csv"),
            "heuristics_best": str(args.output_dir / "timor_abw_heuristics_selected_best.csv"),
            "selected_budget_comparison": str(args.output_dir / "timor_abw_selected_budget_comparison.csv"),
            "previous_selected_delta": str(args.output_dir / "timor_abw_previous_selected_delta.csv"),
            "previous_stats_delta": str(args.output_dir / "timor_abw_previous_stats_delta.csv"),
            "report": str(args.output_dir / "timor_abw_rerun_report.md"),
        },
    }
    (args.output_dir / "timor_abw_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
