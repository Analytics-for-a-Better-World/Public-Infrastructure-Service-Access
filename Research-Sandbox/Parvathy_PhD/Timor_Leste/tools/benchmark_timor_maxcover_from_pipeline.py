from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

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
    approximate_pareto_curve,
    build_instance_from_facility_map,
    exact_pareto_curve,
)


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_output(manifest: dict[str, Any], key: str) -> Path:
    return Path(manifest["outputs"][key]["path"])


def load_manifest_by_spacing(outputs_dir: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    manifests: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(outputs_dir.glob("run_manifest_*.yaml")):
        data = read_manifest(path)
        spacing = int(round(float(data["parameters"]["resolved"]["candidate_grid_spacing_m"])))
        manifests[spacing] = (path, data)
    return manifests


def build_instance_from_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    weight_scale: int,
) -> tuple[Any, dict[str, Any]]:
    t0 = perf_counter()
    population_path = resolve_output(manifest, "population")
    amenities_path = resolve_output(manifest, "distance_matrix_src_amenities_dst_population")
    candidates_path = resolve_output(manifest, "distance_matrix_src_candidates_dst_population")
    sources_path = resolve_output(manifest, "sources")

    population = pd.read_parquet(population_path, columns=["ID", "population"])
    id_to_pos = {str(value): int(pos) for pos, value in enumerate(population["ID"])}
    pop_values = population["population"].to_numpy(dtype=float)
    weights = np.rint(pop_values * weight_scale).astype(np.int64)

    amenities = pd.read_parquet(amenities_path, columns=["target_id"])
    covered_positions = np.fromiter(
        (id_to_pos[str(target_id)] for target_id in amenities["target_id"].unique()),
        dtype=np.int32,
    )
    covered_set = set(int(value) for value in covered_positions.tolist())

    candidates = pd.read_parquet(candidates_path, columns=["source_id", "target_id"])
    source_ids = pd.Index(candidates["source_id"].unique())
    facility_to_pos = {str(value): int(pos) for pos, value in enumerate(source_ids)}
    facility_to_demand: dict[int, list[int]] = {int(pos): [] for pos in range(len(source_ids))}

    for source_id, target_id in candidates[["source_id", "target_id"]].itertuples(index=False):
        demand_pos = id_to_pos[str(target_id)]
        if demand_pos in covered_set:
            continue
        facility_to_demand[facility_to_pos[str(source_id)]].append(demand_pos)

    instance = build_instance_from_facility_map(
        facility_to_demand,
        weights,
        covered=covered_set,
        name=f"timor_leste_{int(manifest['parameters']['resolved']['candidate_grid_spacing_m'])}m_grid",
        n_facilities=len(source_ids),
        assume_unique_sorted=False,
        metadata={
            "manifest_path": str(manifest_path),
            "population_path": str(population_path),
            "amenities_path": str(amenities_path),
            "candidates_path": str(candidates_path),
            "sources_path": str(sources_path),
            "weight_scale": int(weight_scale),
            "candidate_grid_spacing_m": int(manifest["parameters"]["resolved"]["candidate_grid_spacing_m"]),
            "candidate_max_snap_dist_m": float(manifest["parameters"]["resolved"]["candidate_max_snap_dist_m"]),
            "max_total_dist_m": float(manifest["runtime_settings"]["max_total_dist"]),
            "pipeline_git_commit": manifest.get("pipeline_git_commit") or manifest.get("code", {}).get("git_commit"),
            "osm_pbf_sha256": manifest["inputs"]["osm_pbf"]["sha256"],
            "worldpop_sha256": manifest["inputs"]["worldpop_raster"]["sha256"],
            "created_utc": manifest["created_utc"],
        },
    )

    baseline_weight = int(weights[covered_positions].sum())
    total_weight = int(weights.sum())
    available_incremental_weight = int(instance.weights[instance.ij_indptr[1:] > instance.ij_indptr[:-1]].sum())
    stats = {
        **instance.metadata,
        "build_seconds": perf_counter() - t0,
        "n_population": int(len(population)),
        "n_existing_sources": int(len(pd.read_parquet(resolve_output(manifest, "existing_sources"), columns=["ID"]))),
        "n_sources_total": int(len(pd.read_parquet(sources_path, columns=["ID"]))),
        "n_candidates": int(instance.n_facilities),
        "n_candidate_arcs_raw": int(len(candidates)),
        "n_candidate_arcs_after_existing_coverage_removed": int(instance.ji_indices.size),
        "baseline_weight": baseline_weight,
        "total_weight": total_weight,
        "baseline_population": baseline_weight / weight_scale,
        "total_population": total_weight / weight_scale,
        "baseline_percent": 100.0 * baseline_weight / total_weight,
        "available_incremental_population": available_incremental_weight / weight_scale,
    }
    return instance, stats


def result_records(
    *,
    grid_spacing_m: int,
    stats: dict[str, Any],
    exact_results: list[Any],
    heuristic_results: list[Any],
    weight_scale: int,
) -> list[dict[str, Any]]:
    exact_by_budget = {result.budget: result for result in exact_results}
    rows: list[dict[str, Any]] = []
    for result in exact_results + heuristic_results:
        exact = exact_by_budget.get(result.budget)
        objective = None if result.objective is None else int(result.objective)
        exact_objective = None if exact is None or exact.objective is None else int(exact.objective)
        gap_to_exact_population = None
        gap_to_exact_percent_points = None
        optimality_ratio = None
        if objective is not None and exact_objective is not None and exact.status == "optimal":
            gap_to_exact_population = (exact_objective - objective) / weight_scale
            gap_to_exact_percent_points = 100.0 * (exact_objective - objective) / stats["total_weight"]
            optimality_ratio = objective / exact_objective if exact_objective else None
        rows.append(
            {
                "grid_spacing_m": int(grid_spacing_m),
                "budget": int(result.budget),
                "method": result.method,
                "status": result.status,
                "objective_population": None if objective is None else objective / weight_scale,
                "total_covered_population": None
                if objective is None
                else (stats["baseline_weight"] + objective) / weight_scale,
                "coverage_percent_total_population": None
                if objective is None
                else 100.0 * (stats["baseline_weight"] + objective) / stats["total_weight"],
                "exact_objective_population": None if exact_objective is None else exact_objective / weight_scale,
                "gap_to_exact_population": gap_to_exact_population,
                "gap_to_exact_percent_points": gap_to_exact_percent_points,
                "optimality_ratio": optimality_ratio,
                "upper_bound_population": None if result.upper_bound is None else float(result.upper_bound) / weight_scale,
                "mip_gap": result.mip_gap,
                "selected_count": len(result.solution),
                "model_seconds": result.model_seconds,
                "solve_seconds": result.solve_seconds,
                "total_seconds": result.total_seconds,
                "construction_seconds": result.construction_seconds,
                "local_search_moves": result.local_search_moves,
                "seed": result.seed,
                "repeat": result.repeat,
                "baseline_population": stats["baseline_population"],
                "total_population": stats["total_population"],
                "baseline_percent": stats["baseline_percent"],
                "n_candidates": stats["n_candidates"],
                "n_candidate_arcs_after_existing_coverage_removed": stats[
                    "n_candidate_arcs_after_existing_coverage_removed"
                ],
                "manifest_path": stats["manifest_path"],
                "pipeline_git_commit": stats["pipeline_git_commit"],
                "osm_pbf_sha256": stats["osm_pbf_sha256"],
                "worldpop_sha256": stats["worldpop_sha256"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(
            r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis\runs\network_only_20260622_1645\east-timor_data\outputs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis\outputs\timor_maxcover_benchmark_20260623"),
    )
    parser.add_argument("--grids", type=int, nargs="+", default=[10000, 5000, 1000])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 60, 100, 175])
    parser.add_argument("--exact-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--weight-scale", type=int, default=1000)
    parser.add_argument("--heuristic-repeats", type=int, default=3)
    parser.add_argument("--trace-gurobi", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = load_manifest_by_spacing(args.outputs_dir)
    all_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for grid in args.grids:
        manifest_path, manifest = manifests[int(grid)]
        print(f"Building Timor-Leste {grid} m instance from {manifest_path.name}", flush=True)
        instance, stats = build_instance_from_manifest(
            manifest_path,
            manifest,
            weight_scale=int(args.weight_scale),
        )
        stats_rows.append(stats)
        print(
            json.dumps(
                {
                    "grid_spacing_m": grid,
                    "n_candidates": stats["n_candidates"],
                    "arcs": stats["n_candidate_arcs_after_existing_coverage_removed"],
                    "baseline_percent": stats["baseline_percent"],
                    "build_seconds": stats["build_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )

        heuristic_cfg = HeuristicConfig(
            randomized_repeats=int(args.heuristic_repeats),
            seed=42,
            rcl_size=25,
            sample_size=250,
            local_search="first_sparse",
            use_path_relinking=True,
        )
        print(f"Running heuristics for {grid} m", flush=True)
        heuristic_curve = approximate_pareto_curve(instance, args.budgets, config=heuristic_cfg, select_best=True)
        heuristic_results = heuristic_curve.results

        print(f"Running exact Gurobi for {grid} m", flush=True)
        gurobi_cfg = GurobiConfig(
            time_limit_seconds=float(args.exact_time_limit),
            mip_gap=float(args.mip_gap),
            trace=bool(args.trace_gurobi),
            log_file=str(args.output_dir / "gurobi_logs" / f"timor_{grid}m_exact.log"),
            warm_start=True,
            parsimonious=False,
        )
        exact_curve = exact_pareto_curve(instance, args.budgets, solver="gurobi", gurobi_config=gurobi_cfg)
        exact_results = exact_curve.results

        all_rows.extend(
            result_records(
                grid_spacing_m=grid,
                stats=stats,
                exact_results=exact_results,
                heuristic_results=heuristic_results,
                weight_scale=int(args.weight_scale),
            )
        )
        pd.DataFrame(all_rows).to_csv(args.output_dir / "timor_exact_vs_heuristic_results.csv", index=False)
        pd.DataFrame(stats_rows).to_csv(args.output_dir / "timor_instance_statistics.csv", index=False)

    manifest_out = {
        "script": str(Path(__file__).resolve()),
        "outputs_dir": str(args.outputs_dir),
        "output_dir": str(args.output_dir),
        "grids": args.grids,
        "budgets": args.budgets,
        "exact_time_limit": args.exact_time_limit,
        "mip_gap": args.mip_gap,
        "weight_scale": args.weight_scale,
        "heuristic_repeats": args.heuristic_repeats,
        "optimization_package": "abw_maxcover",
        "abw_maxcover_src": str(ABW_MAXCOVER_SRC),
        "public_infrastructure_repo_commit": git_revision(REPO_ROOT),
    }
    (args.output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
