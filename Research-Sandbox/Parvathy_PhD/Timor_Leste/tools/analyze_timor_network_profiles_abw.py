from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(r"C:\github\Public-Infrastructure-Service-Access")
ABW_MAXCOVER_SRC = REPO_ROOT / "Research-Sandbox" / "abw_maxcover" / "src"
if str(ABW_MAXCOVER_SRC) not in sys.path:
    sys.path.insert(0, str(ABW_MAXCOVER_SRC))

from abw_maxcover import (  # noqa: E402
    GurobiConfig,
    HeuristicConfig,
    PyomoConfig,
    approximate_pareto_curve,
    build_instance_from_facility_map,
    exact_pareto_curve,
)


PROFILE_LABEL = {"driving": "drive_only", "driving_walk": "drive_plus_walk"}


def clock_ms(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    if not math.isfinite(float(seconds)):
        return None
    millis = int(round(float(seconds) * 1000.0))
    sign = "-" if millis < 0 else ""
    millis = abs(millis)
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_output(manifest: dict[str, Any], key: str) -> Path:
    path = Path(manifest["outputs"][key]["path"])
    if path.exists():
        return path
    raise FileNotFoundError(path)


def runtime_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("parameters", {}).get("runtime_settings", {})


def resolved_parameters(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("resolved_parameters") or manifest.get("parameters", {}).get("resolved", {})


def case_id_from_manifest(manifest: dict[str, Any]) -> str:
    runtime = runtime_settings(manifest)
    resolved = resolved_parameters(manifest)
    profile = str(runtime.get("network_profile", "unknown"))
    simplified = bool(runtime.get("simplify_network"))
    spacing = int(round(float(resolved.get("candidate_grid_spacing_m") or runtime.get("candidate_grid_spacing_m"))))
    simplify_label = "simplified" if simplified else "unsimplified"
    return f"timor_{PROFILE_LABEL.get(profile, profile)}_{simplify_label}_{spacing // 1000}km"


def load_manifests(outputs_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(outputs_dir.glob("run_manifest_*.yaml")):
        manifest = read_yaml(path)
        case_id = case_id_from_manifest(manifest)
        if case_id in manifests:
            raise ValueError(f"duplicate case id {case_id}: {path} and {manifests[case_id][0]}")
        manifests[case_id] = (path, manifest)
    return manifests


def load_case_timings(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def snap_records(case_id: str, group: str, values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    record: dict[str, Any] = {
        "case_id": case_id,
        "snap_group": group,
        "count": int(arr.size),
    }
    if arr.size == 0:
        for key in ["mean_m", "std_m", "min_m", "p50_m", "p90_m", "p95_m", "p99_m", "max_m"]:
            record[key] = math.nan
        return record
    record.update(
        {
            "mean_m": float(arr.mean()),
            "std_m": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min_m": float(arr.min()),
            "p50_m": float(np.percentile(arr, 50)),
            "p90_m": float(np.percentile(arr, 90)),
            "p95_m": float(np.percentile(arr, 95)),
            "p99_m": float(np.percentile(arr, 99)),
            "max_m": float(arr.max()),
        }
    )
    return record


def build_instance_from_manifest(
    case_id: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    weight_scale: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    build_start = perf_counter()
    population_path = resolve_output(manifest, "population")
    amenities_path = resolve_output(manifest, "distance_matrix_src_amenities_dst_population")
    candidates_path = resolve_output(manifest, "distance_matrix_src_candidates_dst_population")
    sources_path = resolve_output(manifest, "sources")
    existing_sources_path = resolve_output(manifest, "existing_sources")
    components_path = resolve_output(manifest, "connectivity_components")

    population = pd.read_parquet(population_path, columns=["ID", "population", "dist_snap_target", "component_id"])
    population["ID"] = population["ID"].astype(str)
    id_to_pos = pd.Series(np.arange(len(population), dtype=np.int32), index=population["ID"])
    pop_values = population["population"].to_numpy(dtype=float)
    weights = np.rint(pop_values * int(weight_scale)).astype(np.int64)

    amenities = pd.read_parquet(amenities_path, columns=["target_id"])
    amenity_target_ids = pd.Index(amenities["target_id"].astype(str).unique())
    covered_positions = id_to_pos.reindex(amenity_target_ids).dropna().to_numpy(dtype=np.int32)
    covered_set = set(int(value) for value in covered_positions.tolist())
    covered_mask = np.zeros(len(population), dtype=bool)
    if covered_positions.size:
        covered_mask[covered_positions] = True

    candidates = pd.read_parquet(candidates_path, columns=["source_id", "target_id"])
    source_codes, source_ids = pd.factorize(candidates["source_id"].astype(str), sort=False)
    target_positions = id_to_pos.reindex(candidates["target_id"].astype(str)).to_numpy(dtype=np.int32)
    uncovered_arc_mask = ~covered_mask[target_positions]
    source_codes = source_codes[uncovered_arc_mask].astype(np.int32, copy=False)
    target_positions = target_positions[uncovered_arc_mask].astype(np.int32, copy=False)

    facility_to_demand: dict[int, np.ndarray] = {}
    if source_codes.size:
        order = np.argsort(source_codes, kind="mergesort")
        sorted_sources = source_codes[order]
        sorted_targets = target_positions[order]
        starts = np.r_[0, np.flatnonzero(sorted_sources[1:] != sorted_sources[:-1]) + 1]
        ends = np.r_[starts[1:], sorted_sources.size]
        for start, end in zip(starts, ends):
            facility_to_demand[int(sorted_sources[int(start)])] = sorted_targets[int(start) : int(end)]

    runtime = runtime_settings(manifest)
    resolved = resolved_parameters(manifest)
    spacing_m = int(round(float(resolved.get("candidate_grid_spacing_m") or runtime.get("candidate_grid_spacing_m"))))
    profile = str(runtime.get("network_profile", "unknown"))
    simplify_network = bool(runtime.get("simplify_network"))
    threshold_m = float(runtime.get("max_total_dist"))

    instance = build_instance_from_facility_map(
        facility_to_demand,
        weights,
        covered=covered_set,
        name=f"{case_id}_threshold_{int(threshold_m)}m",
        n_facilities=len(source_ids),
        assume_unique_sorted=False,
        metadata={
            "case_id": case_id,
            "manifest_path": str(manifest_path),
            "population_path": str(population_path),
            "amenities_path": str(amenities_path),
            "candidates_path": str(candidates_path),
            "sources_path": str(sources_path),
            "existing_sources_path": str(existing_sources_path),
            "components_path": str(components_path),
            "weight_scale": int(weight_scale),
            "network_profile": profile,
            "simplify_network": simplify_network,
            "candidate_grid_spacing_m": spacing_m,
            "candidate_max_snap_dist_m": float(resolved.get("candidate_max_snap_dist_m") or runtime.get("candidate_max_snap_dist_m")),
            "max_total_dist_m": threshold_m,
            "pipeline_git_commit": manifest.get("pipeline_git_commit") or manifest.get("code", {}).get("git_commit"),
            "osm_pbf_sha256": manifest["inputs"]["osm_pbf"]["sha256"],
            "worldpop_sha256": manifest["inputs"]["worldpop_raster"]["sha256"],
            "created_utc": manifest["created_utc"],
        },
    )

    sources = pd.read_parquet(sources_path, columns=["ID", "dist_snap_source", "source_type", "component_id"])
    existing_sources = pd.read_parquet(existing_sources_path, columns=["ID", "dist_snap_source", "source_type", "component_id"])
    candidate_sources = sources.loc[sources["source_type"].astype(str).eq("candidates")]
    components = pd.read_parquet(components_path)
    top_components = components.sort_values("node_count", ascending=False).head(5)

    baseline_weight = int(weights[covered_positions].sum())
    total_weight = int(weights.sum())
    demand_with_candidates = instance.demand_with_candidates()
    available_incremental_weight = int(weights[demand_with_candidates].sum())
    stats = {
        **instance.metadata,
        "build_seconds": perf_counter() - build_start,
        "build_clock_ms": None,
        "n_population": int(len(population)),
        "n_population_components": int(population["component_id"].nunique(dropna=True)),
        "n_existing_sources": int(len(existing_sources)),
        "n_existing_source_components": int(existing_sources["component_id"].nunique(dropna=True)),
        "n_sources_total": int(len(sources)),
        "n_candidate_sources_in_sources_table": int(len(candidate_sources)),
        "n_candidate_sources_in_matrix": int(instance.n_facilities),
        "n_candidate_arcs_raw": int(len(candidates)),
        "n_candidate_arcs_after_existing_coverage_removed": int(target_positions.size),
        "n_candidate_arcs_after_deduplication": int(instance.ji_indices.size),
        "n_coverable_uncovered_population_points": int(demand_with_candidates.size),
        "baseline_weight": baseline_weight,
        "total_weight": total_weight,
        "all_candidates_incremental_weight": available_incremental_weight,
        "baseline_population": baseline_weight / weight_scale,
        "total_population": total_weight / weight_scale,
        "baseline_percent": 100.0 * baseline_weight / total_weight if total_weight else math.nan,
        "all_candidates_incremental_population": available_incremental_weight / weight_scale,
        "all_candidates_total_population": (baseline_weight + available_incremental_weight) / weight_scale,
        "all_candidates_coverage_percent": 100.0 * (baseline_weight + available_incremental_weight) / total_weight
        if total_weight
        else math.nan,
        "road_component_count": int(len(components)),
        "road_node_count": int(components["node_count"].sum()),
        "largest_component_nodes": int(top_components["node_count"].iloc[0]) if len(top_components) else 0,
        "largest_component_share": float(top_components["node_count"].iloc[0] / components["node_count"].sum())
        if len(top_components) and components["node_count"].sum()
        else math.nan,
        "top5_component_nodes": ";".join(str(int(value)) for value in top_components["node_count"].tolist()),
    }
    stats["build_clock_ms"] = clock_ms(stats["build_seconds"])

    snap_stats = [
        snap_records(case_id, "population_targets", population["dist_snap_target"].to_numpy(dtype=float)),
        snap_records(case_id, "existing_sources", existing_sources["dist_snap_source"].to_numpy(dtype=float)),
        snap_records(case_id, "candidate_sources", candidate_sources["dist_snap_source"].to_numpy(dtype=float)),
    ]
    return instance, stats, snap_stats, {"source_ids": source_ids.astype(str).tolist()}


def result_records(
    *,
    case_id: str,
    solver_family: str,
    stats: dict[str, Any],
    results: list[Any],
    exact_reference: dict[int, Any],
    weight_scale: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        objective = None if result.objective is None else int(result.objective)
        exact = exact_reference.get(int(result.budget))
        exact_objective = None if exact is None or exact.objective is None else int(exact.objective)
        gap_to_exact_population = None
        gap_to_exact_percent_points = None
        optimality_ratio = None
        if (
            objective is not None
            and exact_objective is not None
            and str(getattr(exact, "status", "")).lower() == "optimal"
            and exact_objective > 0
        ):
            gap_to_exact_population = (exact_objective - objective) / weight_scale
            gap_to_exact_percent_points = 100.0 * (exact_objective - objective) / stats["total_weight"]
            optimality_ratio = objective / exact_objective
        total_covered_weight = None if objective is None else stats["baseline_weight"] + objective
        total_seconds = float(result.total_seconds or 0.0)
        row = {
            "case_id": case_id,
            "network_profile": stats["network_profile"],
            "simplify_network": stats["simplify_network"],
            "candidate_grid_spacing_m": stats["candidate_grid_spacing_m"],
            "service_threshold_m": stats["max_total_dist_m"],
            "budget": int(result.budget),
            "solver_family": solver_family,
            "method": result.method,
            "status": result.status,
            "objective_population": None if objective is None else objective / weight_scale,
            "total_covered_population": None if total_covered_weight is None else total_covered_weight / weight_scale,
            "coverage_percent_total_population": None
            if total_covered_weight is None
            else 100.0 * total_covered_weight / stats["total_weight"],
            "exact_reference_objective_population": None if exact_objective is None else exact_objective / weight_scale,
            "gap_to_exact_population": gap_to_exact_population,
            "gap_to_exact_percent_points": gap_to_exact_percent_points,
            "optimality_ratio": optimality_ratio,
            "upper_bound_population": None if result.upper_bound is None else float(result.upper_bound) / weight_scale,
            "mip_gap": result.mip_gap,
            "selected_count": len(result.solution),
            "model_seconds": result.model_seconds,
            "model_clock_ms": clock_ms(result.model_seconds),
            "solve_seconds": result.solve_seconds,
            "solve_clock_ms": clock_ms(result.solve_seconds),
            "total_seconds": total_seconds,
            "total_clock_ms": clock_ms(total_seconds),
            "construction_seconds": result.construction_seconds,
            "construction_clock_ms": clock_ms(result.construction_seconds),
            "local_search_moves": result.local_search_moves,
            "seed": result.seed,
            "repeat": result.repeat,
            "baseline_population": stats["baseline_population"],
            "total_population": stats["total_population"],
            "baseline_percent": stats["baseline_percent"],
            "all_candidates_coverage_percent": stats["all_candidates_coverage_percent"],
            "n_candidate_sources_in_matrix": stats["n_candidate_sources_in_matrix"],
            "n_candidate_arcs_after_existing_coverage_removed": stats[
                "n_candidate_arcs_after_existing_coverage_removed"
            ],
            "manifest_path": stats["manifest_path"],
            "pipeline_git_commit": stats["pipeline_git_commit"],
            "osm_pbf_sha256": stats["osm_pbf_sha256"],
            "worldpop_sha256": stats["worldpop_sha256"],
        }
        for key, value in result.metadata.items():
            row[f"metadata_{key}"] = value
        rows.append(row)
    return rows


def should_run_policy(policy: str, stats: dict[str, Any]) -> bool:
    if policy == "none":
        return False
    if policy == "all":
        return True
    spacing = int(stats["candidate_grid_spacing_m"])
    primary = stats["network_profile"] == "driving_walk" and not bool(stats["simplify_network"])
    if policy == "primary":
        return primary
    if policy == "sparse":
        return spacing in {5000, 10000}
    if policy == "primary_sparse":
        return primary and spacing in {5000, 10000}
    raise ValueError(f"unknown policy: {policy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("runs/timor_network_profile_20260623/east-timor_data/outputs"),
    )
    parser.add_argument(
        "--case-timings",
        type=Path,
        default=Path("outputs/timor_network_profile_20260623/timor_network_profile_case_timings.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/timor_network_profile_optimization_20260623"),
    )
    parser.add_argument("--cases", nargs="*", default=["all"])
    parser.add_argument("--weight-scale", type=int, default=1000)
    parser.add_argument("--heuristic-budgets", type=int, nargs="+", default=[20, 60, 100, 175, 200])
    parser.add_argument("--gurobi-budgets", type=int, nargs="+", default=[20, 60, 100, 175, 200])
    parser.add_argument("--highs-budgets", type=int, nargs="+", default=[20, 60, 100, 175, 200])
    parser.add_argument(
        "--all-budget-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Replace all budget lists with range(START, STOP + 1, STEP).",
    )
    parser.add_argument(
        "--heuristic-budget-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Replace heuristic budgets with range(START, STOP + 1, STEP).",
    )
    parser.add_argument(
        "--gurobi-budget-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Replace Gurobi budgets with range(START, STOP + 1, STEP).",
    )
    parser.add_argument(
        "--highs-budget-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Replace HiGHS budgets with range(START, STOP + 1, STEP).",
    )
    parser.add_argument(
        "--gurobi-case-policy",
        choices=["none", "all", "primary", "sparse", "primary_sparse"],
        default="all",
    )
    parser.add_argument(
        "--highs-case-policy",
        choices=["none", "all", "primary", "sparse", "primary_sparse"],
        default="primary_sparse",
    )
    parser.add_argument("--gurobi-time-limit", type=float, default=300.0)
    parser.add_argument("--highs-time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--heuristic-repeats", type=int, default=5)
    parser.add_argument("--heuristic-seed", type=int, default=42)
    parser.add_argument("--trace-gurobi", action="store_true")
    parser.add_argument("--skip-heuristics", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    def expand(values: list[int]) -> list[int]:
        start, stop, step = [int(value) for value in values]
        if step <= 0:
            raise ValueError("budget range step must be positive")
        return list(range(start, stop + 1, step))

    if args.all_budget_range is not None:
        budgets = expand(args.all_budget_range)
        args.heuristic_budgets = budgets
        args.gurobi_budgets = budgets
        args.highs_budgets = budgets
    if args.heuristic_budget_range is not None:
        args.heuristic_budgets = expand(args.heuristic_budget_range)
    if args.gurobi_budget_range is not None:
        args.gurobi_budgets = expand(args.gurobi_budget_range)
    if args.highs_budget_range is not None:
        args.highs_budgets = expand(args.highs_budget_range)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gurobi_logs").mkdir(parents=True, exist_ok=True)

    manifests = load_manifests(args.outputs_dir)
    requested_cases = set(manifests) if args.cases == ["all"] else set(args.cases)
    missing = sorted(requested_cases - set(manifests))
    if missing:
        raise ValueError(f"unknown case ids: {missing}")

    case_timings = load_case_timings(args.case_timings)
    instance_rows: list[dict[str, Any]] = []
    snap_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    source_id_map: dict[str, Any] = {}

    ordered_cases = sorted(
        requested_cases,
        key=lambda case: (
            manifests[case][1].get("parameters", {}).get("runtime_settings", {}).get("network_profile", ""),
            bool(manifests[case][1].get("parameters", {}).get("runtime_settings", {}).get("simplify_network")),
            int(float(resolved_parameters(manifests[case][1]).get("candidate_grid_spacing_m", 0))),
        ),
    )

    for case_id in ordered_cases:
        manifest_path, manifest = manifests[case_id]
        print(f"\n=== {case_id} ===", flush=True)
        instance, stats, case_snap_rows, source_meta = build_instance_from_manifest(
            case_id,
            manifest_path,
            manifest,
            weight_scale=int(args.weight_scale),
        )
        timing = case_timings.get(case_id.replace("drive_only", "driving").replace("drive_plus_walk", "driving_walk"), {})
        # The pipeline runner used the older case names; keep both ids visible.
        old_case_id = (
            case_id.replace("drive_only", "driving")
            .replace("drive_plus_walk", "driving_walk")
            .replace("_10km", "_10km")
        )
        timing = case_timings.get(old_case_id, timing)
        stats["pipeline_case_id"] = old_case_id
        stats["pipeline_elapsed_seconds"] = number_or_none(timing.get("elapsed_seconds"))
        stats["pipeline_elapsed_clock_ms"] = timing.get("elapsed_clock_ms") or clock_ms(stats["pipeline_elapsed_seconds"])
        instance_rows.append(stats)
        snap_rows.extend(case_snap_rows)
        source_id_map[case_id] = source_meta
        print(
            json.dumps(
                {
                    "n_candidates": stats["n_candidate_sources_in_matrix"],
                    "arcs": stats["n_candidate_arcs_after_existing_coverage_removed"],
                    "baseline_percent": round(stats["baseline_percent"], 4),
                    "all_candidate_percent": round(stats["all_candidates_coverage_percent"], 4),
                    "build_time": stats["build_clock_ms"],
                },
                indent=2,
            ),
            flush=True,
        )

        exact_by_budget: dict[int, Any] = {}
        if not args.stats_only and should_run_policy(args.gurobi_case_policy, stats):
            print(f"Running Gurobi exact for {case_id}: {args.gurobi_budgets}", flush=True)
            gurobi_cfg = GurobiConfig(
                time_limit_seconds=float(args.gurobi_time_limit),
                mip_gap=float(args.mip_gap),
                trace=bool(args.trace_gurobi),
                log_file=str(args.output_dir / "gurobi_logs" / f"{case_id}_gurobi.log"),
                warm_start=True,
                parsimonious=False,
            )
            exact_curve = exact_pareto_curve(
                instance,
                [int(value) for value in args.gurobi_budgets],
                solver="gurobi",
                gurobi_config=gurobi_cfg,
            )
            exact_by_budget = {int(result.budget): result for result in exact_curve.results}
            result_rows.extend(
                result_records(
                    case_id=case_id,
                    solver_family="gurobi_exact",
                    stats=stats,
                    results=exact_curve.results,
                    exact_reference=exact_by_budget,
                    weight_scale=int(args.weight_scale),
                )
            )
            pd.DataFrame(result_rows).to_csv(args.output_dir / "timor_network_profile_results.csv", index=False)

        if not args.stats_only and should_run_policy(args.highs_case_policy, stats):
            print(f"Running Pyomo/HiGHS exact for {case_id}: {args.highs_budgets}", flush=True)
            pyomo_curve = exact_pareto_curve(
                instance,
                [int(value) for value in args.highs_budgets],
                solver="pyomo",
                pyomo_config=PyomoConfig(
                    solver="highs",
                    time_limit_seconds=float(args.highs_time_limit),
                    mip_gap=float(args.mip_gap),
                    trace=False,
                    parsimonious=False,
                ),
            )
            result_rows.extend(
                result_records(
                    case_id=case_id,
                    solver_family="pyomo_highs_exact",
                    stats=stats,
                    results=pyomo_curve.results,
                    exact_reference=exact_by_budget,
                    weight_scale=int(args.weight_scale),
                )
            )
            pd.DataFrame(result_rows).to_csv(args.output_dir / "timor_network_profile_results.csv", index=False)

        if not args.stats_only and not args.skip_heuristics:
            print(f"Running approximate Pareto heuristics for {case_id}: {args.heuristic_budgets}", flush=True)
            heuristic_cfg = HeuristicConfig(
                randomized_repeats=int(args.heuristic_repeats),
                seed=int(args.heuristic_seed),
                rcl_size=25,
                sample_size=250,
                local_search="first_sparse",
                use_path_relinking=True,
            )
            heuristic_curve = approximate_pareto_curve(
                instance,
                [int(value) for value in args.heuristic_budgets],
                config=heuristic_cfg,
                select_best=True,
            )
            result_rows.extend(
                result_records(
                    case_id=case_id,
                    solver_family="approximate_pareto",
                    stats=stats,
                    results=heuristic_curve.results,
                    exact_reference=exact_by_budget,
                    weight_scale=int(args.weight_scale),
                )
            )
            pd.DataFrame(result_rows).to_csv(args.output_dir / "timor_network_profile_results.csv", index=False)

        pd.DataFrame(instance_rows).to_csv(args.output_dir / "timor_network_profile_instance_statistics.csv", index=False)
        pd.DataFrame(snap_rows).to_csv(args.output_dir / "timor_network_profile_snap_statistics.csv", index=False)
        (args.output_dir / "timor_network_profile_source_ids.json").write_text(
            json.dumps(source_id_map, indent=2),
            encoding="utf-8",
        )

    manifest_out = {
        "script": str(Path(__file__).resolve()),
        "outputs_dir": str(args.outputs_dir),
        "case_timings": str(args.case_timings),
        "output_dir": str(args.output_dir),
        "cases": ordered_cases,
        "weight_scale": args.weight_scale,
        "heuristic_budgets": args.heuristic_budgets,
        "gurobi_budgets": args.gurobi_budgets,
        "highs_budgets": args.highs_budgets,
        "gurobi_case_policy": args.gurobi_case_policy,
        "highs_case_policy": args.highs_case_policy,
        "gurobi_time_limit": args.gurobi_time_limit,
        "highs_time_limit": args.highs_time_limit,
        "mip_gap": args.mip_gap,
        "heuristic_repeats": args.heuristic_repeats,
        "heuristic_seed": args.heuristic_seed,
        "optimization_package": "abw_maxcover",
        "abw_maxcover_src": str(ABW_MAXCOVER_SRC),
        "public_infrastructure_repo_commit": git_revision(REPO_ROOT),
        "abw_maxcover_config": {
            "gurobi": asdict(GurobiConfig()),
            "pyomo": asdict(PyomoConfig(solver="highs")),
            "heuristic": asdict(HeuristicConfig()),
        },
    }
    (args.output_dir / "timor_network_profile_optimization_manifest.json").write_text(
        json.dumps(manifest_out, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "result_rows": len(result_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
