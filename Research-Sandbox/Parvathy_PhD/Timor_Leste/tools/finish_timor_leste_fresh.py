from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import yaml


TIME_RE = re.compile(
    r"(?P<label>Distance computation time|Total runtime|Candidate pipeline completed in|Built Pandana network in|Connectivity diagnostic completed in)"
    r"[: ]+(?P<seconds>[0-9.]+)s?(?:econds)?"
)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_timings(log_path: Path) -> dict[str, float]:
    timings: dict[str, float] = {}
    if not log_path.exists():
        return timings
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in TIME_RE.finditer(text):
        key = match.group("label").lower().replace(" ", "_")
        timings[key] = float(match.group("seconds"))
    return timings


def manifest_output(manifest: dict, key: str) -> Path:
    return Path(manifest["outputs"][key]["path"])


def resolved(manifest: dict) -> dict:
    return manifest.get("resolved_parameters") or manifest.get("parameters", {}).get("resolved", {})


def find_manifest(outputs_dir: Path, *, has_candidates: bool, spacing: float | None) -> Path:
    candidates = []
    for path in outputs_dir.glob("run_manifest*.yaml"):
        manifest = read_yaml(path)
        params = resolved(manifest)
        if bool(params.get("has_candidates")) != has_candidates:
            continue
        value = params.get("candidate_grid_spacing_m")
        if spacing is None:
            if value is None:
                candidates.append(path)
        elif value is not None and abs(float(value) - float(spacing)) < 1e-9:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No manifest found for has_candidates={has_candidates}, spacing={spacing}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def covered_stats(population: pd.DataFrame, covered_ids: set[str]) -> tuple[int, float, float]:
    mask = population["ID"].astype(str).isin(covered_ids)
    covered = float(population.loc[mask, "population"].sum())
    total = float(population["population"].sum())
    return int(mask.sum()), covered, covered / total if total else math.nan


def summarize(label: str, manifest_path: Path, log_path: Path) -> dict:
    manifest = read_yaml(manifest_path)
    population = pd.read_parquet(manifest_output(manifest, "population"))
    sources = pd.read_parquet(manifest_output(manifest, "sources"))
    amenity_dm = pd.read_parquet(
        manifest_output(manifest, "distance_matrix_src_amenities_dst_population"),
        columns=["target_id", "source_id", "total_dist"],
    )

    candidate_dm = None
    if "distance_matrix_src_candidates_dst_population" in manifest["outputs"]:
        candidate_dm = pd.read_parquet(
            manifest_output(manifest, "distance_matrix_src_candidates_dst_population"),
            columns=["target_id", "source_id", "total_dist"],
        )

    existing_ids = set(amenity_dm.loc[amenity_dm["total_dist"] <= 5000.0, "target_id"].astype(str))
    existing_points, existing_pop, existing_share = covered_stats(population, existing_ids)
    all_points = all_pop = all_share = None
    candidate_rows = 0
    if candidate_dm is not None:
        candidate_rows = int(len(candidate_dm))
        candidate_ids = set(candidate_dm.loc[candidate_dm["total_dist"] <= 5000.0, "target_id"].astype(str))
        all_points, all_pop, all_share = covered_stats(population, existing_ids | candidate_ids)

    source_counts = sources["source_type"].astype(str).value_counts().to_dict()
    timings = parse_timings(log_path)
    params = resolved(manifest)
    return {
        "label": label,
        "manifest": str(manifest_path),
        "log_file": str(log_path),
        "created_utc": manifest.get("created_utc"),
        "pbf_path": manifest["inputs"]["osm_pbf"]["path"],
        "pbf_sha256": manifest["inputs"]["osm_pbf"]["sha256"],
        "worldpop_path": manifest["inputs"]["worldpop_raster"]["path"],
        "worldpop_sha256": manifest["inputs"]["worldpop_raster"]["sha256"],
        "has_candidates": bool(params.get("has_candidates")),
        "candidate_grid_spacing_m": params.get("candidate_grid_spacing_m"),
        "candidate_max_snap_dist_m": params.get("candidate_max_snap_dist_m"),
        "population_points": int(len(population)),
        "retained_population": float(population["population"].sum()),
        "amenity_sources": int(source_counts.get("amenities", 0)),
        "candidate_sources": int(source_counts.get("candidates", 0)),
        "amenity_distance_rows": int(len(amenity_dm)),
        "candidate_distance_rows": candidate_rows,
        "existing_covered_points": existing_points,
        "existing_covered_population": existing_pop,
        "existing_covered_share": existing_share,
        "all_candidates_covered_points": all_points,
        "all_candidates_covered_population": all_pop,
        "all_candidates_covered_share": all_share,
        "distance_computation_seconds": timings.get("distance_computation_time"),
        "total_runtime_seconds": timings.get("total_runtime"),
        "candidate_pipeline_seconds": timings.get("candidate_pipeline_completed_in"),
        "pandana_build_seconds": timings.get("built_pandana_network_in"),
        "connectivity_seconds": timings.get("connectivity_diagnostic_completed_in"),
    }


def build_gurobi_data(manifest_path: Path) -> dict:
    manifest = read_yaml(manifest_path)
    population = pd.read_parquet(manifest_output(manifest, "population"))
    sources = pd.read_parquet(manifest_output(manifest, "sources"))
    amenity_dm = pd.read_parquet(
        manifest_output(manifest, "distance_matrix_src_amenities_dst_population"),
        columns=["target_id", "source_id", "total_dist"],
    )
    candidate_dm = pd.read_parquet(
        manifest_output(manifest, "distance_matrix_src_candidates_dst_population"),
        columns=["target_id", "source_id", "total_dist"],
    )

    population["ID"] = population["ID"].astype(str)
    total_population = float(population["population"].sum())
    weights_by_id = dict(zip(population["ID"], population["population"].astype(float)))

    existing_ids = set(amenity_dm.loc[amenity_dm["total_dist"] <= 5000.0, "target_id"].astype(str))
    existing_points, baseline_population, baseline_share = covered_stats(population, existing_ids)

    candidate_sources = (
        sources.loc[sources["source_type"].astype(str) == "candidates", "ID"]
        .astype(str)
        .sort_values()
        .to_list()
    )
    candidate_to_idx = {source_id: idx for idx, source_id in enumerate(candidate_sources)}

    rows = candidate_dm.loc[candidate_dm["total_dist"] <= 5000.0, ["source_id", "target_id"]].copy()
    rows["source_id"] = rows["source_id"].astype(str)
    rows["target_id"] = rows["target_id"].astype(str)
    rows = rows.loc[~rows["target_id"].isin(existing_ids)]
    rows = rows.loc[rows["source_id"].isin(candidate_to_idx)]

    target_to_candidates: dict[str, list[int]] = {}
    for target_id, group in rows.groupby("target_id", sort=False):
        js = sorted({candidate_to_idx[source_id] for source_id in group["source_id"]})
        if js:
            target_to_candidates[str(target_id)] = js

    target_ids = sorted(target_to_candidates)
    weights = np.array([weights_by_id[target_id] for target_id in target_ids], dtype=float)
    target_candidate_lists = [target_to_candidates[target_id] for target_id in target_ids]

    return {
        "manifest": str(manifest_path),
        "candidate_sources": candidate_sources,
        "target_ids": target_ids,
        "weights": weights,
        "target_candidate_lists": target_candidate_lists,
        "total_population": total_population,
        "baseline_population": baseline_population,
        "baseline_share": baseline_share,
        "existing_covered_points": existing_points,
        "candidate_edge_rows": int(len(rows)),
    }


def solve_budgets(data: dict, budgets: list[int], output_dir: Path, time_limit: float, mip_gap: float) -> list[dict]:
    log_path = output_dir / "timor_leste_gurobi_174_175.log"
    n_candidates = len(data["candidate_sources"])
    n_targets = len(data["target_ids"])
    weights = data["weights"]
    lists = data["target_candidate_lists"]

    model_start = perf_counter()
    model = gp.Model("timor_leste_max_cover_5km")
    model.Params.LogFile = str(log_path)
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    x = model.addVars(n_candidates, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_targets, vtype=GRB.BINARY, name="y")
    budget_constr = model.addConstr(gp.quicksum(x[j] for j in range(n_candidates)) <= 0, name="budget")
    for i, js in enumerate(lists):
        model.addConstr(y[i] <= gp.quicksum(x[j] for j in js), name=f"cover_{i}")
    model.setObjective(gp.quicksum(float(weights[i]) * y[i] for i in range(n_targets)), GRB.MAXIMIZE)
    model.update()
    modeling_seconds = perf_counter() - model_start

    results = []
    for budget in budgets:
        budget_constr.RHS = int(budget)
        solve_start = perf_counter()
        model.optimize()
        solving_seconds = perf_counter() - solve_start
        selected = [data["candidate_sources"][j] for j in range(n_candidates) if x[j].X >= 0.5]
        incremental = float(model.ObjVal) if model.SolCount else math.nan
        covered = float(data["baseline_population"] + incremental)
        results.append(
            {
                "budget": int(budget),
                "status": int(model.Status),
                "status_name": {
                    GRB.OPTIMAL: "optimal",
                    GRB.TIME_LIMIT: "time_limit",
                    GRB.INFEASIBLE: "infeasible",
                    GRB.INTERRUPTED: "interrupted",
                }.get(model.Status, str(model.Status)),
                "modeling_seconds": modeling_seconds,
                "solving_seconds": solving_seconds,
                "mip_gap": None if model.MIPGap == GRB.INFINITY else float(model.MIPGap),
                "objective_bound": None if model.ObjBound == GRB.INFINITY else float(model.ObjBound),
                "selected_candidates": len(selected),
                "incremental_population": incremental,
                "baseline_population": float(data["baseline_population"]),
                "covered_population": covered,
                "coverage_share": covered / float(data["total_population"]),
                "total_population": float(data["total_population"]),
                "n_candidate_variables": n_candidates,
                "n_demand_variables": n_targets,
                "candidate_edge_rows": int(data["candidate_edge_rows"]),
                "gurobi_log": str(log_path),
            }
        )
        pd.DataFrame(results).to_csv(output_dir / "timor_leste_gurobi_174_175_results.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[174, 175])
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    args = parser.parse_args()

    country_dir = args.fresh_root / "east-timor_data"
    outputs_dir = country_dir / "outputs"
    logs_dir = country_dir / "logs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = find_manifest(outputs_dir, has_candidates=False, spacing=None)
    grid5_manifest = find_manifest(outputs_dir, has_candidates=True, spacing=5000.0)
    grid25_manifest = find_manifest(outputs_dir, has_candidates=True, spacing=2500.0)

    summaries = [
        summarize("existing_health_5km", existing_manifest, logs_dir / "timor_existing_health_5km_full.log"),
        summarize("candidate_grid_5000m", grid5_manifest, logs_dir / "timor_candidate_health_5km_grid_5000.log"),
        summarize("candidate_grid_2500m", grid25_manifest, logs_dir / "timor_candidate_health_5km_grid_2500.log"),
    ]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "timor_leste_fresh_pipeline_summary.csv", index=False)
    (args.output_dir / "timor_leste_fresh_pipeline_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    data = build_gurobi_data(grid25_manifest)
    gurobi_results = solve_budgets(
        data,
        sorted(args.budgets),
        args.output_dir,
        time_limit=float(args.time_limit_seconds),
        mip_gap=float(args.mip_gap),
    )
    (args.output_dir / "timor_leste_gurobi_174_175_results.json").write_text(
        json.dumps(gurobi_results, indent=2),
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))
    print(pd.DataFrame(gurobi_results).to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
