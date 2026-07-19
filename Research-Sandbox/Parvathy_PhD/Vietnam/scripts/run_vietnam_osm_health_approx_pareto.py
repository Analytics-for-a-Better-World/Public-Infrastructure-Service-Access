from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml


def default_abw_maxcover_src(script_path: Path) -> Path:
    resolved = script_path.resolve()
    candidates = [
        # Sandbox layout: <workspace>/tools/<script>.py
        resolved.parents[1],
        resolved.parents[1] / "abw_maxcover" / "src",
        # Repository layout: Research-Sandbox/Parvathy_PhD/Vietnam/scripts/<script>.py
        resolved.parents[4] / "packages" / "abw_maxcover" / "src",
    ]
    for candidate in candidates:
        if (candidate / "abw_maxcover").exists():
            return candidate
    return candidates[0]


ROOT = Path(__file__).resolve().parents[1]
ABW_MAXCOVER_SRC = default_abw_maxcover_src(Path(__file__))
if str(ABW_MAXCOVER_SRC) not in sys.path:
    sys.path.insert(0, str(ABW_MAXCOVER_SRC))

RUN_OUTPUT = ROOT / "runs" / "vietnam_170_agg5_20260624_s20" / "vietnam_data" / "outputs"
OUT = ROOT / "outputs" / "vietnam_osm_health_approx_pareto_20260630"
THRESHOLD_M = 5_000.0
BUDGETS = [0, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]


def clock_ms(seconds: float) -> str:
    millis = int(round(seconds * 1000.0))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def load_manifest_candidate_paths(run_output: Path, spacings: list[int]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for manifest_path in sorted(run_output.glob("run_manifest_*.yaml")):
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        settings = data.get("runtime_settings", {}) or data.get("parameters", {}).get("runtime_settings", {})
        spacing = settings.get("candidate_grid_spacing_m")
        outputs = data.get("outputs", {})
        candidate = outputs.get("distance_matrix_src_candidates_dst_population")
        if spacing is None or not candidate:
            continue
        path = Path(candidate["path"])
        if path.exists():
            paths[int(round(float(spacing)))] = path
    missing = set(spacings) - set(paths)
    if missing:
        raise FileNotFoundError(f"Missing candidate matrix paths for spacings: {sorted(missing)}")
    return paths


def find_one(run_output: Path, pattern: str) -> Path:
    matches = sorted(run_output.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def read_filtered_matrix(path: Path, threshold_m: float) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=["target_id", "source_id", "total_dist"],
        filter=pc.field("total_dist") <= float(threshold_m),
    )
    return table.to_pandas()


def build_curve_for_spacing(
    *,
    spacing_m: int,
    candidate_path: Path,
    population: pd.DataFrame,
    baseline_covered_ids: set[str],
    total_weight: int,
    threshold_m: float,
    budgets: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from abw_maxcover import HeuristicConfig, approximate_pareto_curve
    from abw_maxcover.instance import build_instance_from_facility_map

    start_total = perf_counter()
    start = perf_counter()
    candidate = read_filtered_matrix(candidate_path, threshold_m)
    read_seconds = perf_counter() - start
    candidate_arcs_before_baseline = int(len(candidate))

    start = perf_counter()
    candidate = candidate.loc[~candidate["target_id"].isin(baseline_covered_ids), ["target_id", "source_id"]]
    candidate = candidate.drop_duplicates()
    residual_target_ids = pd.Index(candidate["target_id"].unique())
    residual_population = population.set_index("ID").loc[residual_target_ids, "population"]
    weights = np.rint(residual_population.to_numpy(dtype="float64")).astype(np.int64)
    demand_codes = pd.Series(np.arange(len(residual_target_ids), dtype=np.int32), index=residual_target_ids)
    facility_ids = pd.Index(candidate["source_id"].unique())
    facility_codes = pd.Series(np.arange(len(facility_ids), dtype=np.int32), index=facility_ids)
    candidate["demand_i"] = candidate["target_id"].map(demand_codes).astype("int32")
    candidate["facility_j"] = candidate["source_id"].map(facility_codes).astype("int32")
    facility_to_demand = {
        int(facility): group["demand_i"].to_numpy(dtype=np.int32)
        for facility, group in candidate.groupby("facility_j", sort=False)
    }
    instance = build_instance_from_facility_map(
        facility_to_demand,
        weights,
        name=f"vietnam_osm_health_{spacing_m}m",
        n_facilities=len(facility_ids),
        assume_unique_sorted=False,
        metadata={
            "spacing_m": spacing_m,
            "threshold_m": threshold_m,
            "candidate_path": str(candidate_path),
        },
    )
    build_seconds = perf_counter() - start

    max_budget = min(max(budgets), instance.n_facilities)
    budgets_for_run = [budget for budget in budgets if budget <= max_budget]
    if budgets_for_run[-1] != max_budget:
        budgets_for_run.append(max_budget)
    config = HeuristicConfig(
        constructors=("greedy", "compact", "regreedy"),
        randomized_repeats=0,
        local_search="none",
        use_path_relinking=False,
        seed=42,
    )
    start = perf_counter()
    curve = approximate_pareto_curve(instance, budgets_for_run, config=config)
    solve_seconds = perf_counter() - start

    records: list[dict[str, object]] = []
    for result in curve.results:
        records.append(
            {
                "spacing_m": spacing_m,
                "budget": int(result.budget),
                "method": result.method,
                "incremental_objective": int(result.objective),
                "incremental_coverage_pct_total": 100.0 * int(result.objective) / total_weight,
                "solve_seconds_result": result.total_seconds,
                "solve_clock_result": clock_ms(float(result.total_seconds)),
                "selected_count": len(result.solution),
            }
        )
    summary = {
        "spacing_m": spacing_m,
        "candidate_matrix_path": str(candidate_path),
        "candidate_arcs_within_threshold_before_baseline": candidate_arcs_before_baseline,
        "candidate_arcs_after_osm_baseline": int(len(candidate)),
        "residual_demand_points_with_candidate": int(instance.n_demand),
        "candidate_count_with_residual_arcs": int(instance.n_facilities),
        "read_seconds": read_seconds,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": perf_counter() - start_total,
        "read_clock": clock_ms(read_seconds),
        "build_clock": clock_ms(build_seconds),
        "solve_clock": clock_ms(solve_seconds),
        "total_clock": clock_ms(perf_counter() - start_total),
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output", type=Path, default=RUN_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--threshold-m", type=float, default=THRESHOLD_M)
    parser.add_argument("--spacings", type=int, nargs="+", default=[10000, 5000, 1000])
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    parser.add_argument(
        "--abw-maxcover-src",
        type=Path,
        default=ABW_MAXCOVER_SRC,
        help="Directory that contains the abw_maxcover package, or the package src directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.abw_maxcover_src) not in sys.path:
        sys.path.insert(0, str(args.abw_maxcover_src))
    run_output = args.run_output.resolve()
    output_root = args.output_root.resolve()
    threshold_m = float(args.threshold_m)
    spacings = [int(value) for value in args.spacings]
    budgets = sorted({int(value) for value in args.budgets})
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_paths = load_manifest_candidate_paths(run_output, spacings)
    baseline_matrix_path = find_one(run_output, "distance_matrix_src_amenities_dst_population_*af7db34de280.parquet")
    population_path = find_one(run_output, "population_*amenity_clinic-hospital*.parquet")
    population = pd.read_parquet(population_path)
    total_weight = int(np.rint(population["population"].sum()))

    baseline_matrix = read_filtered_matrix(baseline_matrix_path, threshold_m)
    baseline_covered_ids = set(baseline_matrix["target_id"].drop_duplicates().astype(str))
    baseline_weight = int(
        np.rint(population.loc[population["ID"].isin(baseline_covered_ids), "population"].sum())
    )
    baseline_pct = 100.0 * baseline_weight / total_weight

    all_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for spacing_m in spacings:
        print(f"Building OSM-baseline approximate Pareto for {spacing_m} m", flush=True)
        records, summary = build_curve_for_spacing(
            spacing_m=spacing_m,
            candidate_path=candidate_paths[spacing_m],
            population=population,
            baseline_covered_ids=baseline_covered_ids,
            total_weight=total_weight,
            threshold_m=threshold_m,
            budgets=budgets,
        )
        for record in records:
            record["baseline_objective"] = baseline_weight
            record["total_objective"] = baseline_weight + int(record["incremental_objective"])
            record["total_coverage_pct"] = 100.0 * int(record["total_objective"]) / total_weight
            record["baseline_coverage_pct"] = baseline_pct
        all_records.extend(records)
        summaries.append(summary)

    curve_df = pd.DataFrame(all_records)
    summary = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "threshold_m": threshold_m,
        "run_output": str(run_output),
        "output_root": str(output_root),
        "spacings": spacings,
        "budgets": budgets,
        "abw_maxcover_src": str(args.abw_maxcover_src),
        "baseline_matrix_path": str(baseline_matrix_path),
        "population_path": str(population_path),
        "baseline_covered_population": baseline_weight,
        "total_population_weight": total_weight,
        "baseline_coverage_pct": baseline_pct,
        "summaries": summaries,
    }
    curve_csv = output_root / "vietnam_osm_health_approx_pareto_curves.csv"
    summary_json = output_root / "vietnam_osm_health_approx_pareto_summary.json"
    figure_pdf = output_root / "vietnam_osm_health_approx_pareto_curves.pdf"
    curve_df.to_csv(curve_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    colors = {10000: "#4C78A8", 5000: "#F58518", 1000: "#54A24B"}
    for spacing_m, group in curve_df.groupby("spacing_m"):
        group = group.sort_values("budget")
        ax.plot(
            group["budget"],
            group["total_coverage_pct"],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=colors.get(int(spacing_m)),
            label=f"{int(spacing_m/1000)} km candidates",
        )
    ax.axhline(baseline_pct, color="#666666", linestyle="--", linewidth=1.2, label="OSM baseline")
    ax.set_xlabel("New facilities opened (p)")
    ax.set_ylabel(f"Population covered within {threshold_m / 1000:g} km (%)")
    ax.set_title("Vietnam OSM hospital/clinic baseline with candidate-grid expansion")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_pdf)
    fig.savefig(output_root / "vietnam_osm_health_approx_pareto_curves.png", dpi=220)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
