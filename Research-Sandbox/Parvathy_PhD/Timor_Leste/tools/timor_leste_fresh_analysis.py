from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, vstack


TIME_RE = re.compile(
    r"(?P<label>Distance computation time|Total runtime|Candidate pipeline completed in|Built Pandana network in|Connectivity diagnostic completed in)"
    r"[: ]+(?P<seconds>[0-9.]+)s?(?:econds)?"
)


@dataclass
class RunSummary:
    label: str
    manifest: str
    log_file: str
    created_utc: str
    has_candidates: bool
    candidate_grid_spacing_m: float | None
    candidate_max_snap_dist_m: float | None
    population_points: int
    retained_population: float
    amenity_sources: int
    candidate_sources: int
    amenity_distance_rows: int
    candidate_distance_rows: int
    existing_covered_points: int
    existing_covered_population: float
    existing_covered_share: float
    all_candidates_covered_points: int | None
    all_candidates_covered_population: float | None
    all_candidates_covered_share: float | None
    distance_computation_seconds: float | None
    total_runtime_seconds: float | None
    candidate_pipeline_seconds: float | None
    pandana_build_seconds: float | None
    connectivity_seconds: float | None


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_timings(log_path: Path) -> dict[str, float]:
    timings: dict[str, float] = {}
    if not log_path.exists():
        return timings
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in TIME_RE.finditer(text):
        key = (
            match.group("label")
            .lower()
            .replace(" ", "_")
            .replace(":", "")
        )
        timings[key] = float(match.group("seconds"))
    return timings


def output_path(manifest: dict, key: str) -> Path | None:
    item = manifest.get("outputs", {}).get(key)
    if not item:
        return None
    return Path(item["path"])


def covered_population(population: pd.DataFrame, covered_ids: set[str]) -> tuple[int, float, float]:
    mask = population["ID"].isin(covered_ids)
    covered = float(population.loc[mask, "population"].sum())
    total = float(population["population"].sum())
    return int(mask.sum()), covered, covered / total if total else math.nan


def summarize_run(label: str, manifest_path: Path, log_path: Path) -> RunSummary:
    manifest = read_yaml(manifest_path)
    population_path = output_path(manifest, "population")
    sources_path = output_path(manifest, "sources")
    amenity_matrix_path = output_path(manifest, "distance_matrix_src_amenities_dst_population")
    candidate_matrix_path = output_path(manifest, "distance_matrix_src_candidates_dst_population")
    if population_path is None or sources_path is None or amenity_matrix_path is None:
        raise ValueError(f"Manifest {manifest_path} is missing required outputs.")

    population = pd.read_parquet(population_path)
    sources = pd.read_parquet(sources_path)
    amenity_dm = pd.read_parquet(amenity_matrix_path, columns=["target_id", "source_id", "total_dist"])
    candidate_dm = (
        pd.read_parquet(candidate_matrix_path, columns=["target_id", "source_id", "total_dist"])
        if candidate_matrix_path is not None
        else None
    )

    existing_ids = set(amenity_dm.loc[amenity_dm["total_dist"] <= 5000.0, "target_id"].astype(str))
    existing_points, existing_pop, existing_share = covered_population(population, existing_ids)

    all_points = None
    all_pop = None
    all_share = None
    candidate_rows = 0
    if candidate_dm is not None:
        candidate_rows = int(len(candidate_dm))
        all_ids = existing_ids | set(
            candidate_dm.loc[candidate_dm["total_dist"] <= 5000.0, "target_id"].astype(str)
        )
        all_points, all_pop, all_share = covered_population(population, all_ids)

    source_types = sources["source_type"].astype(str).value_counts().to_dict()
    timings = parse_timings(log_path)
    resolved = manifest.get("resolved_parameters", manifest.get("parameters", {}).get("resolved", {}))

    return RunSummary(
        label=label,
        manifest=str(manifest_path),
        log_file=str(log_path),
        created_utc=str(manifest.get("created_utc", "")),
        has_candidates=bool(resolved.get("has_candidates")),
        candidate_grid_spacing_m=resolved.get("candidate_grid_spacing_m"),
        candidate_max_snap_dist_m=resolved.get("candidate_max_snap_dist_m"),
        population_points=int(len(population)),
        retained_population=float(population["population"].sum()),
        amenity_sources=int(source_types.get("amenities", 0)),
        candidate_sources=int(source_types.get("candidates", 0)),
        amenity_distance_rows=int(len(amenity_dm)),
        candidate_distance_rows=candidate_rows,
        existing_covered_points=existing_points,
        existing_covered_population=existing_pop,
        existing_covered_share=existing_share,
        all_candidates_covered_points=all_points,
        all_candidates_covered_population=all_pop,
        all_candidates_covered_share=all_share,
        distance_computation_seconds=timings.get("distance_computation_time"),
        total_runtime_seconds=timings.get("total_runtime"),
        candidate_pipeline_seconds=timings.get("candidate_pipeline_completed_in"),
        pandana_build_seconds=timings.get("built_pandana_network_in"),
        connectivity_seconds=timings.get("connectivity_diagnostic_completed_in"),
    )


def build_candidate_model(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = read_yaml(manifest_path)
    population_path = output_path(manifest, "population")
    sources_path = output_path(manifest, "sources")
    amenity_matrix_path = output_path(manifest, "distance_matrix_src_amenities_dst_population")
    candidate_matrix_path = output_path(manifest, "distance_matrix_src_candidates_dst_population")
    if None in (population_path, sources_path, amenity_matrix_path, candidate_matrix_path):
        raise ValueError("Candidate solve requires population, sources, amenity matrix, and candidate matrix outputs.")
    return (
        pd.read_parquet(population_path),
        pd.read_parquet(sources_path),
        pd.read_parquet(amenity_matrix_path, columns=["target_id", "source_id", "total_dist"]),
        pd.read_parquet(candidate_matrix_path, columns=["target_id", "source_id", "total_dist"]),
    )


def solve_budget_with_scipy(
    manifest_path: Path,
    budget: int,
    *,
    time_limit_seconds: float,
    mip_rel_gap: float,
) -> dict:
    population, sources, amenity_dm, candidate_dm = build_candidate_model(manifest_path)
    total_population = float(population["population"].sum())
    existing_ids = set(amenity_dm.loc[amenity_dm["total_dist"] <= 5000.0, "target_id"].astype(str))
    existing_mask = population["ID"].astype(str).isin(existing_ids)
    baseline_population = float(population.loc[existing_mask, "population"].sum())

    candidate_sources = (
        sources.loc[sources["source_type"].astype(str) == "candidates", "ID"]
        .astype(str)
        .sort_values()
        .to_list()
    )
    source_to_x = {source_id: idx for idx, source_id in enumerate(candidate_sources)}

    uncovered_population = population.loc[~existing_mask, ["ID", "population"]].copy()
    reachable = candidate_dm.loc[
        (candidate_dm["total_dist"] <= 5000.0)
        & candidate_dm["target_id"].astype(str).isin(set(uncovered_population["ID"].astype(str)))
    ].copy()
    reachable["source_id"] = reachable["source_id"].astype(str)
    reachable["target_id"] = reachable["target_id"].astype(str)
    reachable = reachable.loc[reachable["source_id"].isin(source_to_x)]

    target_ids = sorted(reachable["target_id"].unique())
    target_to_y = {target_id: idx for idx, target_id in enumerate(target_ids)}
    weight_map = dict(zip(population["ID"].astype(str), population["population"].astype(float)))
    weights = np.array([weight_map[target_id] for target_id in target_ids], dtype=float)

    n_x = len(candidate_sources)
    n_y = len(target_ids)
    n_vars = n_x + n_y

    row_idx = np.arange(n_y, dtype=np.int64)
    col_idx = n_x + row_idx
    data = np.ones(n_y, dtype=float)

    edge_rows = reachable["target_id"].map(target_to_y).to_numpy(dtype=np.int64)
    edge_cols = reachable["source_id"].map(source_to_x).to_numpy(dtype=np.int64)
    rows = np.concatenate([row_idx, edge_rows])
    cols = np.concatenate([col_idx, edge_cols])
    vals = np.concatenate([data, -np.ones(len(edge_rows), dtype=float)])
    cover_constraints = coo_array((vals, (rows, cols)), shape=(n_y, n_vars)).tocsr()

    budget_cols = np.arange(n_x, dtype=np.int64)
    budget_constraint = coo_array(
        (np.ones(n_x, dtype=float), (np.zeros(n_x, dtype=np.int64), budget_cols)),
        shape=(1, n_vars),
    ).tocsr()
    constraints = LinearConstraint(
        vstack([cover_constraints, budget_constraint], format="csr"),
        lb=np.full(n_y + 1, -np.inf),
        ub=np.concatenate([np.zeros(n_y, dtype=float), np.array([budget], dtype=float)]),
    )

    c = np.zeros(n_vars, dtype=float)
    c[n_x:] = -weights
    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=constraints,
        options={"time_limit": time_limit_seconds, "mip_rel_gap": mip_rel_gap},
    )

    objective_incremental = -float(result.fun) if result.fun is not None else math.nan
    selected = []
    if result.x is not None:
        selected = [
            candidate_sources[i]
            for i, value in enumerate(result.x[:n_x])
            if value >= 0.5
        ]
    return {
        "budget": budget,
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "fun": None if result.fun is None else float(result.fun),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "mip_dual_bound": getattr(result, "mip_dual_bound", None),
        "mip_gap": getattr(result, "mip_gap", None),
        "baseline_population": baseline_population,
        "incremental_population": objective_incremental,
        "covered_population": baseline_population + objective_incremental,
        "coverage_share": (baseline_population + objective_incremental) / total_population,
        "selected_candidates": len(selected),
        "variables": n_vars,
        "candidate_variables": n_x,
        "demand_variables": n_y,
        "candidate_demand_edges": int(len(reachable)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-manifest", required=True, type=Path)
    parser.add_argument("--existing-log", required=True, type=Path)
    parser.add_argument("--grid5-manifest", required=True, type=Path)
    parser.add_argument("--grid5-log", required=True, type=Path)
    parser.add_argument("--grid25-manifest", required=True, type=Path)
    parser.add_argument("--grid25-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--solve-budgets", nargs="*", type=int, default=[])
    parser.add_argument("--solve-time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_run("existing_health_5km", args.existing_manifest, args.existing_log),
        summarize_run("candidate_grid_5000m", args.grid5_manifest, args.grid5_log),
        summarize_run("candidate_grid_2500m", args.grid25_manifest, args.grid25_log),
    ]

    summary_df = pd.DataFrame([asdict(item) for item in summaries])
    summary_csv = args.output_dir / "timor_leste_fresh_pipeline_summary.csv"
    summary_json = args.output_dir / "timor_leste_fresh_pipeline_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps([asdict(item) for item in summaries], indent=2),
        encoding="utf-8",
    )

    solve_rows = []
    for budget in args.solve_budgets:
        solve_rows.append(
            solve_budget_with_scipy(
                args.grid25_manifest,
                budget,
                time_limit_seconds=args.solve_time_limit_seconds,
                mip_rel_gap=args.mip_rel_gap,
            )
        )
        pd.DataFrame(solve_rows).to_csv(
            args.output_dir / "timor_leste_scipy_milp_budget_results.csv",
            index=False,
        )

    print(summary_df.to_string(index=False))
    if solve_rows:
        print(pd.DataFrame(solve_rows).to_string(index=False))
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
