from __future__ import annotations

import argparse
import copy
import csv
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(r"C:\github\Public-Infrastructure-Service-Access")
PIPELINE_DIR = REPO_ROOT / "Research-Sandbox" / "general_distances_per_country"
APPROX_SRC = REPO_ROOT / "Research-Sandbox" / "approximated_tradeoff" / "src"
VIETNAM_SCRIPT_DIR = REPO_ROOT / "Research-Sandbox" / "Parvathy_PhD" / "Vietnam" / "scripts"

sys.path.insert(0, str(APPROX_SRC))
sys.path.insert(0, str(VIETNAM_SCRIPT_DIR))
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(ROOT / "tools"))

import mc_heuristics as mch  # noqa: E402
from distance_pipeline.candidate_sites import (  # noqa: E402
    build_regular_grid_within_polygon,
    exclude_points_on_water,
    filter_snapped_candidates_by_distance,
)
from distance_pipeline.distance_matrix import compute_distances_polars  # noqa: E402
from distance_pipeline.snapping import snap_points_to_nodes  # noqa: E402
from finish_timor_leste_fresh import build_gurobi_data, find_manifest, summarize  # noqa: E402
from vietnam_grasp_heuristics import budgeted_construct, improve_local_search, run_grasp  # noqa: E402
from vietnam_sparse_local_search import SparseSwapLocalSearch  # noqa: E402


GRID_LABELS = {
    10000: "10 km",
    5000: "5 km",
    1000: "1 km",
}

PROJECTED_EPSG = 32751
SERVICE_THRESHOLD_M = 5000.0


def run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n>>> " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def file_meta(path: Path) -> dict:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def build_pandana_network_from_cache(nodes: pd.DataFrame, edges: pd.DataFrame) -> object:
    import pandana as pdna

    return pdna.Network(
        node_x=nodes["lon"],
        node_y=nodes["lat"],
        edge_from=edges["u"],
        edge_to=edges["v"],
        edge_weights=edges[["length"]],
    )


def candidate_sources_from_snapped(snapped: pd.DataFrame) -> pd.DataFrame:
    sources = snapped.rename(columns={"candidate_dist_road_estrada": "dist_snap_source"}).copy()
    sources["ID"] = "source_candidates_" + sources["ID"].astype(str)
    sources["source_type"] = "candidates"
    sources = sources.set_index("ID", drop=False)
    return sources


def ensure_cacheonly_timor_grid(
    *,
    fresh_root: Path,
    output_dir: Path,
    spacing: int,
) -> Path:
    country_dir = fresh_root / "east-timor_data"
    outputs_dir = country_dir / "outputs"
    cache_dir = country_dir / "cache"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = outputs_dir / f"run_manifest_cacheonly_timor_grid_{spacing}m.yaml"
    if manifest_path.exists():
        return manifest_path

    base_manifest_path = find_manifest(outputs_dir, has_candidates=False, spacing=None)
    base_manifest = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))

    candidate_grid_path = cache_dir / f"tls_candidate_sites_spacing_{spacing}m_no_water_include_boundary_epsg_32751.pkl"
    candidate_snapped_path = (
        cache_dir
        / (
            "tls_candidate_sites_snapped_spacing_"
            f"{spacing}m_no_water_include_boundary_candidate_dist_road_estrada_"
            f"max_snap_{spacing}m_epsg_32751.pkl"
        )
    )
    candidate_dm_path = outputs_dir / f"distance_matrix_src_candidates_dst_population_cacheonly_grid_{spacing}m.parquet"
    sources_path = outputs_dir / f"sources_cacheonly_grid_{spacing}m.parquet"

    boundary_path = cache_dir / "tls_country_boundary_epsg_32751.pkl"
    water_path = cache_dir / "east-timor-latest.osm_water_bodies_epsg_32751.pkl"
    nodes_path = cache_dir / "east-timor-latest.osm_nodes.pkl"
    edges_path = cache_dir / "east-timor-latest.osm_edges.pkl"
    population_path = Path(base_manifest["outputs"]["population"]["path"])
    amenity_dm_path = Path(base_manifest["outputs"]["distance_matrix_src_amenities_dst_population"]["path"])
    amenity_sources_path = Path(base_manifest["outputs"]["sources"]["path"])

    if not candidate_grid_path.exists():
        boundary = pd.read_pickle(boundary_path)
        water = pd.read_pickle(water_path)
        candidates = build_regular_grid_within_polygon(
            boundary,
            spacing_m=float(spacing),
            include_boundary=True,
            verbose=True,
        )
        candidates = exclude_points_on_water(candidates, water, verbose=True)
        candidates.to_pickle(candidate_grid_path)
    else:
        candidates = pd.read_pickle(candidate_grid_path)

    if not candidate_snapped_path.exists():
        nodes = pd.read_pickle(nodes_path)
        snapped = snap_points_to_nodes(
            candidates,
            nodes,
            id_col="ID",
            distance_col="candidate_dist_road_estrada",
            projected_epsg=32751,
            keep_geometry=False,
            verbose=True,
        )
        snapped = filter_snapped_candidates_by_distance(
            snapped,
            max_snap_dist_m=float(spacing),
            distance_col="candidate_dist_road_estrada",
            verbose=True,
        )
        snapped.to_pickle(candidate_snapped_path)
    else:
        snapped = pd.read_pickle(candidate_snapped_path)

    candidate_sources = candidate_sources_from_snapped(snapped)
    amenity_sources = pd.read_parquet(amenity_sources_path)
    sources = pd.concat([amenity_sources, candidate_sources], ignore_index=False, sort=False)
    sources.to_parquet(sources_path, index=False)

    if not candidate_dm_path.exists():
        population = pd.read_parquet(population_path).copy()
        population["ID"] = population["ID"].astype(str)
        population = population.set_index("ID", drop=False)

        nodes = pd.read_pickle(nodes_path)
        edges = pd.read_pickle(edges_path)
        network = build_pandana_network_from_cache(nodes, edges)
        node_pair_cache_dir = cache_dir / "node_pair_distances" / "east-timor-latest.osm_cost_length"
        dm = compute_distances_polars(
            targets=population,
            sources=candidate_sources,
            distance_threshold_largest=5.0,
            network=network,
            max_total_dist=5000.0,
            node_pair_cache_dir=node_pair_cache_dir,
            verbose=True,
        )
        dm.write_parquet(candidate_dm_path)

    manifest = copy.deepcopy(base_manifest)
    manifest["created_utc"] = datetime.now(UTC).isoformat()
    manifest["implementation"]["role"] = "cache_only_reconstruction"
    manifest["outputs"] = {
        "population": file_meta(population_path),
        "targets": file_meta(Path(base_manifest["outputs"]["targets"]["path"])),
        "sources": file_meta(sources_path),
        "existing_sources": file_meta(Path(base_manifest["outputs"]["existing_sources"]["path"])),
        "distance_matrix_src_amenities_dst_population": file_meta(amenity_dm_path),
        "distance_matrix_src_candidates_dst_population": file_meta(candidate_dm_path),
        "connectivity_components": file_meta(Path(base_manifest["outputs"]["connectivity_components"]["path"])),
    }
    resolved = {
        "aggregate_factor": None,
        "amenity_values": [
            "src_amenities-candidates",
            "dst_population",
            "amenity_clinic-doctors-hospital",
        ],
        "candidate_grid_spacing_m": float(spacing),
        "candidate_max_snap_dist_m": float(spacing),
        "has_candidates": True,
    }
    manifest.setdefault("parameters", {})["resolved"] = resolved
    manifest["resolved_parameters"] = resolved
    runtime_settings = manifest.setdefault("runtime_settings", {})
    runtime_settings.update(
        {
            "candidate_grid_spacing_m": float(spacing),
            "candidate_max_snap_dist_m": float(spacing),
            "force_recompute": False,
            "max_total_dist": 5000.0,
            "matrix_output_mode": "split",
            "matrix_shape": "sparse",
        }
    )
    manifest.setdefault("parameters", {})["runtime_settings"] = runtime_settings
    manifest.setdefault("diagnostics", {})["cache_only_note"] = (
        "Generated from cached boundary, water, snapped population, OSM nodes/edges, "
        "and node-pair distances because pyrosm.pbfreader was blocked by local "
        "application-control policy."
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def ensure_timor_pipeline_runs(
    *,
    python: Path,
    fresh_root: Path,
    output_dir: Path,
    spacings: list[int],
    force_pipeline: bool,
) -> None:
    outputs_dir = fresh_root / "east-timor_data" / "outputs"
    for spacing in spacings:
        try:
            if not force_pipeline:
                find_manifest(outputs_dir, has_candidates=True, spacing=float(spacing))
                print(f"Found existing Timor manifest for {spacing} m; skipping pipeline run.")
                continue
        except FileNotFoundError:
            pass

        print(f"Building cache-only Timor manifest for {spacing} m.")
        ensure_cacheonly_timor_grid(
            fresh_root=fresh_root,
            output_dir=output_dir,
            spacing=spacing,
        )


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
    return mch.build_instance(weights, ij_lists, ji_arrays, assume_unique_sorted=True)


def project_lon_lat(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    transformer = Transformer.from_crs(4326, PROJECTED_EPSG, always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def ensure_raw_candidate_grid(fresh_root: Path, spacing: int) -> pd.DataFrame:
    cache_dir = fresh_root / "east-timor_data" / "cache"
    candidate_grid_path = cache_dir / f"tls_candidate_sites_spacing_{spacing}m_no_water_include_boundary_epsg_32751.pkl"
    if candidate_grid_path.exists():
        return pd.read_pickle(candidate_grid_path).reset_index(drop=True)

    boundary = pd.read_pickle(cache_dir / "tls_country_boundary_epsg_32751.pkl")
    water = pd.read_pickle(cache_dir / "east-timor-latest.osm_water_bodies_epsg_32751.pkl")
    candidates = build_regular_grid_within_polygon(
        boundary,
        spacing_m=float(spacing),
        include_boundary=True,
        verbose=True,
    )
    candidates = exclude_points_on_water(candidates, water, verbose=True)
    candidates.to_pickle(candidate_grid_path)
    return candidates.reset_index(drop=True)


def candidate_xy(candidates: pd.DataFrame) -> np.ndarray:
    if "geometry" in candidates.columns and getattr(candidates, "crs", None) is not None:
        projected = candidates if candidates.crs.to_epsg() == PROJECTED_EPSG else candidates.to_crs(PROJECTED_EPSG)
        return np.column_stack([projected.geometry.x.to_numpy(dtype=float), projected.geometry.y.to_numpy(dtype=float)])
    return project_lon_lat(
        candidates["Longitude"].to_numpy(dtype=float),
        candidates["Latitude"].to_numpy(dtype=float),
    )


def find_existing_timor_manifest(fresh_root: Path) -> Path:
    return find_manifest(fresh_root / "east-timor_data" / "outputs", has_candidates=False, spacing=None)


def build_straightline_timor_data(
    *,
    fresh_root: Path,
    spacing: int,
) -> tuple[dict, dict]:
    base_manifest_path = find_existing_timor_manifest(fresh_root)
    base_manifest = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))
    population = pd.read_parquet(base_manifest["outputs"]["population"]["path"]).reset_index(drop=True)
    existing = pd.read_parquet(base_manifest["outputs"]["sources"]["path"]).reset_index(drop=True)
    candidates = ensure_raw_candidate_grid(fresh_root, spacing)

    population["ID"] = population["ID"].astype(str)
    pop_xy = project_lon_lat(
        population["Longitude"].to_numpy(dtype=float),
        population["Latitude"].to_numpy(dtype=float),
    )
    existing_xy = project_lon_lat(
        existing["Longitude"].to_numpy(dtype=float),
        existing["Latitude"].to_numpy(dtype=float),
    )
    cand_xy = candidate_xy(candidates)

    pop_tree = cKDTree(pop_xy)
    baseline_mask = np.zeros(len(population), dtype=bool)
    if len(existing_xy):
        for covered in pop_tree.query_ball_point(existing_xy, SERVICE_THRESHOLD_M):
            if covered:
                baseline_mask[np.asarray(covered, dtype=np.int64)] = True

    candidate_tree = cKDTree(cand_xy)
    uncovered_indices = np.flatnonzero(~baseline_mask)
    candidate_lists_raw = candidate_tree.query_ball_point(pop_xy[uncovered_indices], SERVICE_THRESHOLD_M)

    target_ids: list[str] = []
    target_candidate_lists: list[list[int]] = []
    weights: list[float] = []
    for pop_idx, facilities in zip(uncovered_indices, candidate_lists_raw):
        if not facilities:
            continue
        target_ids.append(str(population.at[int(pop_idx), "ID"]))
        target_candidate_lists.append(sorted(int(value) for value in facilities))
        weights.append(float(population.at[int(pop_idx), "population"]))

    raw_candidate_ids = (
        candidates["ID"].astype(str).to_list()
        if "ID" in candidates.columns
        else [str(value) for value in range(len(candidates))]
    )
    data = {
        "manifest": f"straight_line_projected_screening_spacing_{spacing}",
        "candidate_sources": [f"source_candidates_{value}" for value in raw_candidate_ids],
        "target_ids": target_ids,
        "weights": np.asarray(weights, dtype=float),
        "target_candidate_lists": target_candidate_lists,
        "total_population": float(population["population"].sum()),
        "baseline_population": float(population.loc[baseline_mask, "population"].sum()),
        "baseline_share": float(population.loc[baseline_mask, "population"].sum()) / float(population["population"].sum()),
        "existing_covered_points": int(baseline_mask.sum()),
        "candidate_edge_rows": int(sum(len(values) for values in target_candidate_lists)),
    }
    stats = {
        "label": f"straightline_candidate_grid_{spacing}m",
        "distance_model": "straight_line_projected_screening",
        "candidate_grid_spacing_m": float(spacing),
        "population_points": int(len(population)),
        "retained_population": float(population["population"].sum()),
        "amenity_sources": int(len(existing)),
        "candidate_sources": int(len(candidates)),
        "candidate_distance_rows": int(data["candidate_edge_rows"]),
        "existing_covered_points": int(data["existing_covered_points"]),
        "existing_covered_population": float(data["baseline_population"]),
        "existing_covered_share": float(data["baseline_share"]),
        "all_candidates_covered_population": float(data["baseline_population"] + data["weights"].sum()),
        "all_candidates_covered_share": float(data["baseline_population"] + data["weights"].sum()) / float(data["total_population"]),
    }
    return data, stats


def coverage_percent(total: float, denominator: float) -> float:
    return 100.0 * float(total) / float(denominator) if denominator else math.nan


def status_name(status: int) -> str:
    return {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INTERRUPTED: "interrupted",
    }.get(status, str(status))


def solve_exact_unit_curve(
    *,
    data: dict,
    case: str,
    grid: str,
    output_dir: Path,
    time_limit_seconds: float,
    mip_gap: float,
    max_budget: int | None,
) -> tuple[list[dict], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_candidates = len(data["candidate_sources"])
    n_targets = len(data["target_ids"])
    weights = np.asarray(data["weights"], dtype=float)
    target_lists = data["target_candidate_lists"]
    max_incremental = float(weights.sum())
    final_budget = n_candidates if max_budget is None else min(int(max_budget), n_candidates)
    log_path = output_dir / f"{case}_exact_gurobi.log"

    t0 = perf_counter()
    model = gp.Model(f"{case}_exact")
    model.Params.LogFile = str(log_path)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.MIPGap = float(mip_gap)
    x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_targets, vtype=GRB.BINARY, name="y")
    budget_constr = model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) <= 0, name="budget")
    for i, facilities in enumerate(target_lists):
        model.addConstr(y[i] <= gp.quicksum(x[j] for j in facilities), name=f"cover_{i}")
    model.setObjective(gp.quicksum(float(weights[i]) * y[i] for i in range(n_targets)), GRB.MAXIMIZE)
    model.update()
    modeling_seconds = float(perf_counter() - t0)

    rows: list[dict] = []
    saturated_budget: int | None = None
    for budget in range(final_budget + 1):
        budget_constr.RHS = int(budget)
        t1 = perf_counter()
        model.optimize()
        solving_seconds = float(perf_counter() - t1)
        incremental = float(model.ObjVal) if model.SolCount else math.nan
        covered = float(data["baseline_population"] + incremental) if math.isfinite(incremental) else math.nan
        saturated = math.isfinite(incremental) and incremental >= max_incremental - 1e-6
        if saturated and saturated_budget is None:
            saturated_budget = int(budget)
        rows.append(
            {
                "case": case,
                "country": "Timor-Leste",
                "grid": grid,
                "threshold_km": 5.0,
                "budget": int(budget),
                "incremental_population": incremental,
                "covered_population": covered,
                "coverage_percent": coverage_percent(covered, float(data["total_population"])),
                "method": "gurobi_exact_optimum",
                "status_name": status_name(model.Status),
                "modeling_seconds": modeling_seconds,
                "solving_seconds": solving_seconds,
                "cumulative_exact_seconds": float(sum(float(r["solving_seconds"]) for r in rows) + modeling_seconds),
                "mip_gap": None if model.MIPGap == GRB.INFINITY else float(model.MIPGap),
                "objective_bound": None if model.ObjBound == GRB.INFINITY else float(model.ObjBound),
                "saturated_all_candidate_coverage": bool(saturated),
                "gurobi_log": str(log_path),
            }
        )
        if saturated:
            break

    stats = {
        "case": case,
        "country": "Timor-Leste",
        "grid": grid,
        "threshold_km": 5.0,
        "n_candidates": int(n_candidates),
        "n_targets_uncovered_coverable": int(n_targets),
        "candidate_edge_rows": int(data["candidate_edge_rows"]),
        "baseline_population": float(data["baseline_population"]),
        "total_population": float(data["total_population"]),
        "max_incremental_population": max_incremental,
        "all_candidate_covered_population": float(data["baseline_population"] + max_incremental),
        "all_candidate_coverage_percent": coverage_percent(
            float(data["baseline_population"] + max_incremental),
            float(data["total_population"]),
        ),
        "exact_modeling_seconds": modeling_seconds,
        "exact_total_seconds": float(sum(float(r["solving_seconds"]) for r in rows) + modeling_seconds),
        "exact_rows": int(len(rows)),
        "exact_saturation_budget": saturated_budget,
    }
    return rows, stats


def run_approx_curve(
    *,
    data: dict,
    instance: mch.MaxCoverInstance,
    case: str,
    grid: str,
    weight_scale: float,
) -> tuple[list[dict], dict]:
    t0 = perf_counter()
    greedy_result, reduced_result, restricted, lifted_result = mch.greedy_drop_greedy(instance)
    total_seconds = float(perf_counter() - t0)

    rows: list[dict] = []
    for budget, objective in enumerate(lifted_result.objectives):
        incremental = float(objective) / float(weight_scale)
        covered = float(data["baseline_population"] + incremental)
        rows.append(
            {
                "case": case,
                "country": "Timor-Leste",
                "grid": grid,
                "threshold_km": 5.0,
                "budget": int(budget),
                "incremental_population": incremental,
                "covered_population": covered,
                "coverage_percent": coverage_percent(covered, float(data["total_population"])),
                "method": "approximated_tradeoff_greedy_drop_greedy",
                "seconds": float(lifted_result.times[budget]) if budget < len(lifted_result.times) else math.nan,
                "method_total_seconds": total_seconds,
            }
        )

    stats = {
        "case": case,
        "grid": grid,
        "n_population": int(instance.n_households),
        "n_candidates": int(instance.n_facilities),
        "greedy_facilities": int(len(greedy_result.solution)),
        "reduced_facilities": int(len(reduced_result.solution)),
        "restricted_facilities": int(restricted.instance.n_facilities),
        "lifted_facilities": int(len(lifted_result.solution)),
        "max_incremental_population": float(lifted_result.objective) / float(weight_scale),
        "max_covered_population": float(data["baseline_population"]) + float(lifted_result.objective) / float(weight_scale),
        "max_coverage_percent": coverage_percent(
            float(data["baseline_population"]) + float(lifted_result.objective) / float(weight_scale),
            float(data["total_population"]),
        ),
        "total_seconds": total_seconds,
    }
    return rows, stats


def result_row_from_objective(
    *,
    data: dict,
    case: str,
    grid: str,
    budget: int,
    method: str,
    objective: int | float,
    weight_scale: float,
    seconds: float,
    construction_objective: int | float | None = None,
    construction_seconds: float | None = None,
    local_search_moves: int | None = None,
    seed: int | None = None,
    repeat: int | None = None,
    status: str = "ok",
) -> dict:
    incremental = float(objective) / float(weight_scale)
    covered = float(data["baseline_population"] + incremental)
    return {
        "case": case,
        "country": "Timor-Leste",
        "grid": grid,
        "threshold_km": 5.0,
        "budget": int(budget),
        "method": method,
        "status": status,
        "seed": seed,
        "repeat": repeat,
        "construction_incremental_population": (
            float(construction_objective) / float(weight_scale)
            if construction_objective is not None
            else math.nan
        ),
        "incremental_population": incremental,
        "baseline_covered_population": float(data["baseline_population"]),
        "covered_population": covered,
        "coverage_percent": coverage_percent(covered, float(data["total_population"])),
        "objective_weight_units": int(objective),
        "construction_seconds": construction_seconds,
        "seconds": float(seconds),
        "local_search_moves": local_search_moves,
    }


def run_fleur_heuristics(
    *,
    data: dict,
    instance: mch.MaxCoverInstance,
    case: str,
    grid: str,
    budgets: list[int],
    weight_scale: float,
    randomized_repeats: int,
    grasp_time_limit_seconds: float,
    grasp_max_iterations: int,
    rcl_size: int,
    seed: int,
    run_randomized: bool,
) -> list[dict]:
    rows: list[dict] = []
    if not budgets:
        return rows

    sparse = SparseSwapLocalSearch.from_instance(instance)
    for budget in budgets:
        constructed = budgeted_construct(instance, budget, constructor="greedy")
        rows.append(
            result_row_from_objective(
                data=data,
                case=case,
                grid=grid,
                budget=budget,
                method="fleur_greedy_construction",
                objective=constructed.objective,
                weight_scale=weight_scale,
                seconds=constructed.total_time,
            )
        )
        improved = improve_local_search(
            instance,
            constructed,
            local_search="first_sparse",
            sparse_local_search=sparse,
        )
        rows.append(
            result_row_from_objective(
                data=data,
                case=case,
                grid=grid,
                budget=budget,
                method="fleur_greedy_first_sparse",
                objective=improved.objective,
                weight_scale=weight_scale,
                seconds=constructed.total_time + improved.total_time,
                construction_objective=constructed.objective,
                construction_seconds=constructed.total_time,
                local_search_moves=max(0, len(improved.objectives) - 1),
            )
        )

        if run_randomized and randomized_repeats > 0:
            for repeat in range(int(randomized_repeats)):
                run_seed = int(seed + 100 * budget + repeat)
                t0 = perf_counter()
                best, records = run_grasp(
                    instance,
                    budget,
                    time_limit_seconds=float(grasp_time_limit_seconds),
                    max_iterations=int(grasp_max_iterations),
                    constructor="randomized",
                    rcl_size=int(rcl_size),
                    sample_size=250,
                    local_search="first_sparse",
                    path_relinking=True,
                    path_relinking_method="fast",
                    seed=run_seed,
                    max_pool=8,
                )
                rows.append(
                    result_row_from_objective(
                        data=data,
                        case=case,
                        grid=grid,
                        budget=budget,
                        method="fleur_randomized_grasp_first_sparse_fast_path_relinking",
                        objective=best.objective,
                        weight_scale=weight_scale,
                        seconds=float(perf_counter() - t0),
                        seed=run_seed,
                        repeat=repeat,
                        local_search_moves=None,
                    )
                    | {"grasp_iterations": len(records)}
                )
    return rows


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


def choose_five_budgets(exact_anchor_rows: list[dict], anchor_grid: str) -> tuple[list[int], str]:
    saturated = [
        int(row["budget"])
        for row in exact_anchor_rows
        if row.get("saturated_all_candidate_coverage")
    ]
    if saturated:
        anchor = min(saturated)
        basis = f"first {anchor_grid} exact budget that attains all-candidate coverage"
    else:
        anchor = max(int(row["budget"]) for row in exact_anchor_rows)
        basis = f"largest solved {anchor_grid} exact budget"

    raw = [
        max(0, int(round(anchor * f)))
        for f in (0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    budgets: list[int] = []
    for value in raw:
        if value not in budgets:
            budgets.append(value)
    probe = 0
    while len(budgets) < 5 and probe <= anchor:
        if probe not in budgets:
            budgets.append(probe)
        probe += 1
    return sorted(budgets[:5]), basis


def row_at_budget(rows: list[dict], budget: int) -> dict | None:
    exact = [row for row in rows if int(row["budget"]) == int(budget)]
    if exact:
        return exact[0]
    lower = [row for row in rows if int(row["budget"]) <= int(budget)]
    saturated_lower = [row for row in lower if row.get("saturated_all_candidate_coverage")]
    if saturated_lower:
        return max(saturated_lower, key=lambda row: int(row["budget"]))
    return max(lower, key=lambda row: int(row["budget"])) if lower else None


def best_by_budget(rows: list[dict]) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for row in rows:
        budget = int(row["budget"])
        current = best.get(budget)
        if current is None:
            best[budget] = row
            continue
        if float(row["covered_population"]) > float(current["covered_population"]):
            best[budget] = row
        elif float(row["covered_population"]) == float(current["covered_population"]) and float(row["seconds"]) < float(current["seconds"]):
            best[budget] = row
    return best


def build_selected_comparison(
    *,
    selected_budgets: list[int],
    exact_rows_by_grid: dict[str, list[dict]],
    heuristic_1km_rows: list[dict],
) -> list[dict]:
    best_1km = best_by_budget(heuristic_1km_rows)
    rows: list[dict] = []
    for budget in selected_budgets:
        row10 = row_at_budget(exact_rows_by_grid["10 km"], budget)
        row5 = row_at_budget(exact_rows_by_grid["5 km"], budget)
        row1_exact = row_at_budget(exact_rows_by_grid["1 km"], budget)
        row1_heuristic = best_1km.get(int(budget))
        rows.append(
            {
                "budget": int(budget),
                "timor_10km_exact_coverage_percent": None if row10 is None else row10["coverage_percent"],
                "timor_10km_exact_seconds": None if row10 is None else row10["solving_seconds"],
                "timor_5km_exact_coverage_percent": None if row5 is None else row5["coverage_percent"],
                "timor_5km_exact_seconds": None if row5 is None else row5["solving_seconds"],
                "timor_1km_exact_coverage_percent": None if row1_exact is None else row1_exact["coverage_percent"],
                "timor_1km_exact_seconds": None if row1_exact is None else row1_exact["solving_seconds"],
                "timor_1km_best_heuristic_coverage_percent": None if row1_heuristic is None else row1_heuristic["coverage_percent"],
                "timor_1km_best_heuristic_seconds": None if row1_heuristic is None else row1_heuristic["seconds"],
                "timor_1km_best_heuristic_method": None if row1_heuristic is None else row1_heuristic["method"],
                "timor_1km_exact_minus_heuristic_percentage_points": (
                    None
                    if row1_exact is None or row1_heuristic is None
                    else float(row1_exact["coverage_percent"]) - float(row1_heuristic["coverage_percent"])
                ),
                "gain_10km_to_5km_percentage_points": (
                    None
                    if row10 is None or row5 is None
                    else float(row5["coverage_percent"]) - float(row10["coverage_percent"])
                ),
                "gain_5km_to_1km_percentage_points": (
                    None
                    if row5 is None or row1_exact is None
                    else float(row1_exact["coverage_percent"]) - float(row5["coverage_percent"])
                ),
                "gain_5km_to_1km_note": "1 km value is certified exact optimum",
            }
        )
    return rows


def build_approx_quality_rows(
    *,
    selected_budgets: list[int],
    exact_rows_by_grid: dict[str, list[dict]],
    approx_rows: list[dict],
) -> list[dict]:
    approx_by_grid: dict[str, list[dict]] = {}
    for row in approx_rows:
        approx_by_grid.setdefault(str(row["grid"]), []).append(row)

    rows: list[dict] = []
    for grid in ["10 km", "5 km", "1 km"]:
        for budget in selected_budgets:
            exact = row_at_budget(exact_rows_by_grid.get(grid, []), budget)
            approx = row_at_budget(approx_by_grid.get(grid, []), budget)
            rows.append(
                {
                    "grid": grid,
                    "budget": int(budget),
                    "exact_coverage_percent": None if exact is None else exact["coverage_percent"],
                    "approx_coverage_percent": None if approx is None else approx["coverage_percent"],
                    "exact_minus_approx_percentage_points": (
                        None
                        if exact is None or approx is None
                        else float(exact["coverage_percent"]) - float(approx["coverage_percent"])
                    ),
                    "exact_seconds": None if exact is None else exact["solving_seconds"],
                    "approx_seconds": None if approx is None else approx["seconds"],
                }
            )
    return rows


def build_approx_gain_rows(approx_rows: list[dict], selected_budgets: list[int]) -> list[dict]:
    by_grid: dict[str, list[dict]] = {}
    for row in approx_rows:
        by_grid.setdefault(str(row["grid"]), []).append(row)
    rows: list[dict] = []
    for budget in selected_budgets:
        row10 = row_at_budget(by_grid.get("10 km", []), budget)
        row5 = row_at_budget(by_grid.get("5 km", []), budget)
        row1 = row_at_budget(by_grid.get("1 km", []), budget)
        rows.append(
            {
                "budget": int(budget),
                "approx_10km_coverage_percent": None if row10 is None else row10["coverage_percent"],
                "approx_5km_coverage_percent": None if row5 is None else row5["coverage_percent"],
                "approx_1km_coverage_percent": None if row1 is None else row1["coverage_percent"],
                "approx_gain_10km_to_5km_percentage_points": (
                    None if row10 is None or row5 is None else float(row5["coverage_percent"]) - float(row10["coverage_percent"])
                ),
                "approx_gain_5km_to_1km_percentage_points": (
                    None if row5 is None or row1 is None else float(row1["coverage_percent"]) - float(row5["coverage_percent"])
                ),
            }
        )
    return rows


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    def fmt(value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, float):
            if abs(value) < 0.00005:
                value = 0.0
            return f"{value:.4f}"
        return str(value)

    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def write_report(
    *,
    path: Path,
    stats_rows: list[dict],
    selected_rows: list[dict],
    approx_quality_rows: list[dict],
    approx_gain_rows: list[dict],
    selected_basis: str,
) -> None:
    lines = [
        "# Timor-Leste 10/5/1 km Grid Experiment",
        "",
        f"Selected p values use the {selected_basis}.",
        "",
        "Exact optima are certified for the 10 km, 5 km, and 1 km Timor-Leste instances. The 1 km Fleur-style heuristic rows are retained only to assess heuristic quality against the exact optimum.",
        "",
        "## Instance Summary",
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
        "## Selected Budget Comparison",
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
        "## Approximate Greedy-Drop-Greedy Quality",
        "",
        markdown_table(
            approx_quality_rows,
            [
                "grid",
                "budget",
                "exact_coverage_percent",
                "approx_coverage_percent",
                "exact_minus_approx_percentage_points",
                "exact_seconds",
                "approx_seconds",
            ],
        ),
        "",
        "## Approximate Greedy-Drop-Greedy Gains",
        "",
        markdown_table(
            approx_gain_rows,
            [
                "budget",
                "approx_10km_coverage_percent",
                "approx_5km_coverage_percent",
                "approx_1km_coverage_percent",
                "approx_gain_10km_to_5km_percentage_points",
                "approx_gain_5km_to_1km_percentage_points",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--fresh-root", type=Path, default=ROOT / "runs" / "TimorLeste_20260618_220002")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "timor_three_grid_experiments")
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--exact-time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--exact-mip-gap", type=float, default=1e-6)
    parser.add_argument("--exact-max-budget", type=int, default=None)
    parser.add_argument("--force-pipeline", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument(
        "--distance-model",
        choices=["straightline", "network"],
        default="straightline",
        help="Default straightline avoids locally blocked pyrosm/pandana native routing extensions.",
    )
    parser.add_argument("--randomized-repeats", type=int, default=3)
    parser.add_argument("--grasp-time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--grasp-max-iterations", type=int, default=5)
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spacings = [10000, 5000, 1000]
    if args.distance_model == "network" and not args.skip_pipeline:
        ensure_timor_pipeline_runs(
            python=args.python,
            fresh_root=args.fresh_root,
            output_dir=args.output_dir,
            spacings=spacings,
            force_pipeline=bool(args.force_pipeline),
        )

    outputs_dir = args.fresh_root / "east-timor_data" / "outputs"
    logs_dir = args.fresh_root / "east-timor_data" / "logs"

    data_by_grid: dict[str, dict] = {}
    instance_by_grid: dict[str, mch.MaxCoverInstance] = {}
    summary_rows: list[dict] = []
    exact_rows: list[dict] = []
    exact_stats: list[dict] = []
    approx_rows: list[dict] = []
    approx_stats: list[dict] = []

    for spacing in spacings:
        grid = GRID_LABELS[spacing]
        suffix = "straightline" if args.distance_model == "straightline" else "network"
        case = f"timor_leste_grid{spacing // 1000:g}km_5km_{suffix}"

        if args.distance_model == "straightline":
            data, summary = build_straightline_timor_data(
                fresh_root=args.fresh_root,
                spacing=spacing,
            )
            summary_rows.append(summary)
        else:
            manifest = find_manifest(outputs_dir, has_candidates=True, spacing=float(spacing))
            log_candidates = [
                logs_dir / f"timor_candidate_health_5km_grid_{spacing}.log",
                args.output_dir / "pipeline_logs" / f"timor_candidate_health_5km_grid_{spacing}.log",
                args.output_dir / "pipeline_logs" / f"timor_candidate_health_5km_grid_{spacing}.console.log",
            ]
            log_path = next((path for path in log_candidates if path.exists()), log_candidates[-1])
            try:
                summary_rows.append(summarize(f"candidate_grid_{spacing}m", manifest, log_path))
            except Exception as exc:
                summary_rows.append({"label": f"candidate_grid_{spacing}m", "manifest": str(manifest), "summary_error": str(exc)})
            data = build_gurobi_data(manifest)

        instance = build_instance_from_timor_data(data, float(args.weight_scale))
        data_by_grid[grid] = data
        instance_by_grid[grid] = instance

        rows, stats = run_approx_curve(
            data=data,
            instance=instance,
            case=case,
            grid=grid,
            weight_scale=float(args.weight_scale),
        )
        approx_rows.extend(rows)
        approx_stats.append(stats)

        rows, stats = solve_exact_unit_curve(
            data=data,
            case=case,
            grid=grid,
            output_dir=args.output_dir / "gurobi_logs",
            time_limit_seconds=float(args.exact_time_limit_seconds),
            mip_gap=float(args.exact_mip_gap),
            max_budget=args.exact_max_budget,
        )
        exact_rows.extend(rows)
        exact_stats.append(stats)

    write_csv(args.output_dir / "timor_pipeline_summary.csv", summary_rows)
    write_csv(args.output_dir / "timor_approx_greedy_drop_greedy_curves.csv", approx_rows)
    write_csv(args.output_dir / "timor_approx_greedy_drop_greedy_stats.csv", approx_stats)
    exact_curves_path = args.output_dir / "timor_exact_unit_curves_10km_5km_1km.csv"
    exact_stats_path = args.output_dir / "timor_exact_unit_stats_10km_5km_1km.csv"
    write_csv(exact_curves_path, exact_rows)
    write_csv(exact_stats_path, exact_stats)
    write_csv(args.output_dir / "timor_exact_unit_curves_10km_5km.csv", exact_rows)
    write_csv(args.output_dir / "timor_exact_unit_stats_10km_5km.csv", exact_stats)

    exact_by_grid: dict[str, list[dict]] = {
        grid: [row for row in exact_rows if row["grid"] == grid]
        for grid in ("10 km", "5 km", "1 km")
    }
    selected_budgets, selected_basis = choose_five_budgets(exact_by_grid["1 km"], "1 km")
    selected_meta = [{"budget": value, "basis": selected_basis} for value in selected_budgets]
    write_csv(args.output_dir / "selected_p_values.csv", selected_meta)

    heuristic_1km_rows = run_fleur_heuristics(
        data=data_by_grid["1 km"],
        instance=instance_by_grid["1 km"],
        case="timor_leste_grid1km_5km",
        grid="1 km",
        budgets=selected_budgets,
        weight_scale=float(args.weight_scale),
        randomized_repeats=int(args.randomized_repeats),
        grasp_time_limit_seconds=float(args.grasp_time_limit_seconds),
        grasp_max_iterations=int(args.grasp_max_iterations),
        rcl_size=int(args.rcl_size),
        seed=int(args.seed),
        run_randomized=True,
    )
    write_csv(args.output_dir / "timor_1km_fleur_heuristics_selected_p.csv", heuristic_1km_rows)

    selected_rows = build_selected_comparison(
        selected_budgets=selected_budgets,
        exact_rows_by_grid=exact_by_grid,
        heuristic_1km_rows=heuristic_1km_rows,
    )
    approx_quality_rows = build_approx_quality_rows(
        selected_budgets=selected_budgets,
        exact_rows_by_grid=exact_by_grid,
        approx_rows=approx_rows,
    )
    approx_gain_rows = build_approx_gain_rows(approx_rows, selected_budgets)
    write_csv(args.output_dir / "timor_selected_budget_comparison.csv", selected_rows)
    write_csv(args.output_dir / "timor_approx_vs_exact_selected_p.csv", approx_quality_rows)
    write_csv(args.output_dir / "timor_approx_gain_selected_p.csv", approx_gain_rows)

    write_report(
        path=args.output_dir / "timor_three_grid_report.md",
        stats_rows=exact_stats,
        selected_rows=selected_rows,
        approx_quality_rows=approx_quality_rows,
        approx_gain_rows=approx_gain_rows,
        selected_basis=selected_basis,
    )

    manifest = {
        "fresh_root": str(args.fresh_root),
        "output_dir": str(args.output_dir),
        "candidate_spacings_m": spacings,
        "threshold_m": 5000,
        "distance_model": args.distance_model,
        "amenities": ["hospital", "clinic", "doctors"],
        "exact_grids": ["10 km", "5 km", "1 km"],
        "heuristic_grids": ["1 km"],
        "selected_budgets": selected_budgets,
        "selected_basis": selected_basis,
        "approximated_tradeoff_source": str(APPROX_SRC.parent),
        "fleur_style_source": str(VIETNAM_SCRIPT_DIR),
        "outputs": {
            "pipeline_summary": str(args.output_dir / "timor_pipeline_summary.csv"),
            "approx_curves": str(args.output_dir / "timor_approx_greedy_drop_greedy_curves.csv"),
            "exact_unit_curves": str(exact_curves_path),
            "exact_unit_stats": str(exact_stats_path),
            "heuristic_1km": str(args.output_dir / "timor_1km_fleur_heuristics_selected_p.csv"),
            "selected_comparison": str(args.output_dir / "timor_selected_budget_comparison.csv"),
            "approx_vs_exact_selected": str(args.output_dir / "timor_approx_vs_exact_selected_p.csv"),
            "approx_gain_selected": str(args.output_dir / "timor_approx_gain_selected_p.csv"),
            "report": str(args.output_dir / "timor_three_grid_report.md"),
        },
    }
    (args.output_dir / "timor_three_grid_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
