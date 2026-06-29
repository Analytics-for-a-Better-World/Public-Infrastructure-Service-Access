from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(r"C:\github\Public-Infrastructure-Service-Access")
VIETNAM_SCRIPT_DIR = REPO_ROOT / "Research-Sandbox" / "Parvathy_PhD" / "Vietnam" / "scripts"
sys.path.insert(0, str(VIETNAM_SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "tools"))

import run_dense_grid_straightline_analysis as dense  # noqa: E402
from run_vietnam_5km_exact_selected_p import solve_selected_budgets  # noqa: E402


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


def empty_existing() -> dense.ExistingFacilities:
    return dense.ExistingFacilities(
        lon=np.empty(0, dtype=float),
        lat=np.empty(0, dtype=float),
        xy=np.empty((0, 2), dtype=float),
    )


def load_candidate_set(
    *,
    outputs_dir: Path,
    run_tag_marker: str,
    candidate_grid: Path,
    spacing_m: float,
    weight_scale: float,
) -> tuple[dense.DemandData, dense.CandidateData, dict]:
    demand, candidates, _existing, provenance = dense.load_inputs(
        outputs_dir=outputs_dir,
        marker=run_tag_marker,
        candidate_grid_path=candidate_grid,
        grid_spacing_m=float(spacing_m),
        weight_scale=float(weight_scale),
    )
    provenance["existing_facilities_used"] = False
    provenance["objective_note"] = "No existing-facility baseline; objective is direct selected-site coverage."
    return demand, candidates, provenance


def summary_row(
    *,
    spatial: dense.SpatialMaxCover,
    grid: str,
    spacing_m: float,
    budget: int,
    method: str,
    objective: int,
    seconds: float,
    construction_objective: int | None = None,
    construction_seconds: float | None = None,
    moves: int | None = None,
    seed: int | None = None,
    repeat: int | None = None,
) -> dict:
    incremental = float(objective) / float(spatial.weight_scale)
    total = float(spatial.baseline_population + incremental)
    return {
        "case": f"vietnam_grid{float(spacing_m) / 1000.0:g}km_no_baseline_s20",
        "country": "Vietnam",
        "distance_model": "straight_line_projected_screening",
        "objective_definition": "selected_sites_total_population_no_existing_baseline",
        "grid": grid,
        "grid_spacing_m": float(spacing_m),
        "threshold_km": 20.0,
        "budget": int(budget),
        "method": method,
        "seed": seed,
        "repeat": repeat,
        "status": "ok",
        "n_population": int(spatial.n_population),
        "n_candidates": int(spatial.n_candidates),
        "baseline_covered_population": float(spatial.baseline_population),
        "incremental_population": incremental,
        "total_covered_population": total,
        "coverage_percent_total_population": 100.0 * total / float(spatial.total_population),
        "objective_weight_units": int(objective),
        "construction_incremental_population": None
        if construction_objective is None
        else float(construction_objective) / float(spatial.weight_scale),
        "construction_seconds": construction_seconds,
        "seconds": float(seconds),
        "local_search_moves": moves,
    }


def selected_frame(
    *,
    spatial: dense.SpatialMaxCover,
    grid: str,
    spacing_m: float,
    budget: int,
    method: str,
    solution: list[int],
    seed: int | None = None,
    repeat: int | None = None,
) -> pd.DataFrame:
    rows = []
    for rank, facility_i in enumerate(solution, start=1):
        rows.append(
            {
                "case": f"vietnam_grid{float(spacing_m) / 1000.0:g}km_no_baseline_s20",
                "grid": grid,
                "grid_spacing_m": float(spacing_m),
                "threshold_km": 20.0,
                "budget": int(budget),
                "method": method,
                "seed": seed,
                "repeat": repeat,
                "rank": int(rank),
                "candidate_index": int(facility_i),
                "source_id": str(spatial.candidates.ids[int(facility_i)]),
                "longitude": float(spatial.candidates.lon[int(facility_i)]),
                "latitude": float(spatial.candidates.lat[int(facility_i)]),
            }
        )
    return pd.DataFrame(rows)


def run_1km_heuristic(
    *,
    demand: dense.DemandData,
    candidates: dense.CandidateData,
    budgets: list[int],
    output_dir: Path,
    weight_scale: float,
    cache_size: int,
    chunk_size: int,
    local_max_moves: int,
    local_max_candidates_per_drop: int,
    elite_candidates: int,
    randomized_repeats: int,
    rcl_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spatial = dense.SpatialMaxCover(
        demand=demand,
        candidates=candidates,
        existing=empty_existing(),
        threshold_m=20_000.0,
        weight_scale=float(weight_scale),
        cache_size=int(cache_size),
        chunk_size=int(chunk_size),
    )
    max_budget = max(int(value) for value in budgets)
    print(f"Running no-baseline 1 km heuristic with {spatial.n_candidates:,} candidates", flush=True)
    initial = spatial.compute_initial_gains()
    positive = initial[initial > 0]
    if positive.size:
        elite_k = min(int(elite_candidates), int(positive.size))
        elite = np.argpartition(initial, -elite_k)[-elite_k:]
        elite = elite[initial[elite] > 0].astype(np.int32, copy=False)
    else:
        elite = np.empty(0, dtype=np.int32)

    rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    greedy = spatial.construct(budget=max_budget, randomized=False, rcl_size=int(rcl_size), seed=int(seed))
    for budget in budgets:
        prefix = spatial.prefix_result(greedy, int(budget))
        idx = min(int(budget), len(greedy.times) - 1)
        rows.append(
            summary_row(
                spatial=spatial,
                grid="1 km",
                spacing_m=1000.0,
                budget=int(budget),
                method="spatial_greedy",
                objective=prefix.objective,
                seconds=float(greedy.times[idx]) if idx < len(greedy.times) else greedy.total_time,
            )
        )
        selected_frames.append(
            selected_frame(
                spatial=spatial,
                grid="1 km",
                spacing_m=1000.0,
                budget=int(budget),
                method="spatial_greedy",
                solution=prefix.solution,
            )
        )

        start = perf_counter()
        improved = spatial.improve_local(
            prefix,
            max_moves=int(local_max_moves),
            max_candidates_per_drop=int(local_max_candidates_per_drop),
            elite_candidates=elite,
        )
        rows.append(
            summary_row(
                spatial=spatial,
                grid="1 km",
                spacing_m=1000.0,
                budget=int(budget),
                method="spatial_greedy_local",
                objective=improved.objective,
                seconds=float(perf_counter() - start),
                construction_objective=prefix.objective,
                construction_seconds=float(greedy.times[idx]) if idx < len(greedy.times) else greedy.total_time,
                moves=improved.moves,
            )
        )
        selected_frames.append(
            selected_frame(
                spatial=spatial,
                grid="1 km",
                spacing_m=1000.0,
                budget=int(budget),
                method="spatial_greedy_local",
                solution=improved.solution,
            )
        )

    for budget in budgets:
        for repeat in range(int(randomized_repeats)):
            run_seed = int(seed) + int(budget) * 1000 + repeat
            randomized = spatial.construct(
                budget=int(budget),
                randomized=True,
                rcl_size=int(rcl_size),
                seed=run_seed,
            )
            improved = spatial.improve_local(
                randomized,
                max_moves=int(local_max_moves),
                max_candidates_per_drop=int(local_max_candidates_per_drop),
                elite_candidates=elite,
            )
            rows.append(
                summary_row(
                    spatial=spatial,
                    grid="1 km",
                    spacing_m=1000.0,
                    budget=int(budget),
                    method="spatial_randomized_greedy_local",
                    objective=improved.objective,
                    seconds=float(randomized.total_time + improved.total_time),
                    construction_objective=randomized.objective,
                    construction_seconds=float(randomized.total_time),
                    moves=improved.moves,
                    seed=run_seed,
                    repeat=repeat,
                )
            )
            selected_frames.append(
                selected_frame(
                    spatial=spatial,
                    grid="1 km",
                    spacing_m=1000.0,
                    budget=int(budget),
                    method="spatial_randomized_greedy_local",
                    solution=improved.solution,
                    seed=run_seed,
                    repeat=repeat,
                )
            )
    summary = pd.DataFrame(rows)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary.to_csv(output_dir / "vietnam_1km_no_baseline_heuristic_s20.csv", index=False)
    selected.to_csv(output_dir / "vietnam_1km_no_baseline_heuristic_selected_candidates_s20.csv", index=False)
    return summary, selected


def best_rows(df: pd.DataFrame, coverage_col: str) -> pd.DataFrame:
    return (
        df.sort_values(["budget", coverage_col, "seconds"], ascending=[True, False, True])
        .groupby("budget", as_index=False)
        .head(1)
        .copy()
    )


def write_comparison(
    *,
    output_dir: Path,
    exact_5km: pd.DataFrame,
    heuristic_1km: pd.DataFrame,
) -> pd.DataFrame:
    five = exact_5km.copy().rename(
        columns={
            "status_name": "five_km_status",
            "coverage_percent_total_population": "five_km_exact_coverage_percent",
            "total_covered_population": "five_km_exact_covered_population",
            "seconds": "five_km_exact_seconds",
            "mip_gap": "five_km_mip_gap",
        }
    )
    one = best_rows(heuristic_1km, "coverage_percent_total_population").rename(
        columns={
            "method": "one_km_heuristic_method",
            "coverage_percent_total_population": "one_km_heuristic_coverage_percent",
            "total_covered_population": "one_km_heuristic_covered_population",
            "seconds": "one_km_heuristic_seconds",
        }
    )
    comparison = five.merge(
        one[
            [
                "budget",
                "one_km_heuristic_method",
                "one_km_heuristic_coverage_percent",
                "one_km_heuristic_covered_population",
                "one_km_heuristic_seconds",
            ]
        ],
        on="budget",
        how="left",
    )
    comparison["one_km_minus_five_km_pp"] = (
        comparison["one_km_heuristic_coverage_percent"] - comparison["five_km_exact_coverage_percent"]
    )
    comparison["one_km_extra_covered_population"] = (
        comparison["one_km_heuristic_covered_population"] - comparison["five_km_exact_covered_population"]
    )
    comparison.to_csv(output_dir / "vietnam_no_baseline_s20_1km_heuristic_vs_5km_exact.csv", index=False)
    return comparison


def fmt_pct(value: object) -> str:
    return "" if pd.isna(value) else f"{float(value):.3f}%"


def fmt_num(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, comparison: pd.DataFrame) -> None:
    display = comparison.copy()
    for col in ["five_km_exact_coverage_percent", "one_km_heuristic_coverage_percent"]:
        display[col] = display[col].map(fmt_pct)
    for col in [
        "one_km_minus_five_km_pp",
        "one_km_extra_covered_population",
        "five_km_exact_seconds",
        "one_km_heuristic_seconds",
        "five_km_mip_gap",
    ]:
        display[col] = display[col].map(lambda value: fmt_num(value, 4) if col.endswith("_pp") or col == "five_km_mip_gap" else fmt_num(value, 3))
    lines = [
        "# Vietnam Fleur-Style No-Baseline S=20",
        "",
        "This run matches the chart formulation: no existing-facility baseline, service threshold S=20 km, exact optimization on a 5 km grid, and best available heuristic on a 1 km grid.",
        "",
        markdown_table(
            display,
            [
                "budget",
                "five_km_status",
                "five_km_exact_coverage_percent",
                "one_km_heuristic_coverage_percent",
                "one_km_minus_five_km_pp",
                "one_km_extra_covered_population",
                "five_km_exact_seconds",
                "one_km_heuristic_seconds",
                "one_km_heuristic_method",
                "five_km_mip_gap",
            ],
            [
                "p",
                "5 km Status",
                "5 km Exact",
                "1 km Heuristic",
                "1 km - 5 km pp",
                "Extra Covered",
                "5 km Seconds",
                "1 km Seconds",
                "1 km Method",
                "5 km Gap",
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
        "--candidate-grid-5km",
        type=Path,
        default=ROOT
        / "runs"
        / "vietnam_20260619_0630"
        / "vietnam_data"
        / "cache"
        / "vnm_candidate_sites_spacing_5000m_water_allowed_include_boundary_epsg_3405.pkl",
    )
    parser.add_argument(
        "--candidate-grid-1km",
        type=Path,
        default=ROOT
        / "runs"
        / "vietnam_20260619_0630"
        / "vietnam_data"
        / "cache"
        / "vnm_candidate_sites_spacing_1000m_water_allowed_include_boundary_epsg_3405.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "vietnam_20260619_0630" / "fleur_no_baseline_s20",
    )
    parser.add_argument("--run-tag-marker", default="maxdist_150000")
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 60, 80, 100, 200])
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--incidence-chunk-size", type=int, default=1000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--local-max-moves", type=int, default=8)
    parser.add_argument("--local-max-candidates-per-drop", type=int, default=6000)
    parser.add_argument("--elite-candidates", type=int, default=500)
    parser.add_argument("--randomized-repeats", type=int, default=2)
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demand5, candidates5, provenance5 = load_candidate_set(
        outputs_dir=args.outputs_dir,
        run_tag_marker=args.run_tag_marker,
        candidate_grid=args.candidate_grid_5km,
        spacing_m=5000.0,
        weight_scale=float(args.weight_scale),
    )
    spatial5 = dense.SpatialMaxCover(
        demand=demand5,
        candidates=candidates5,
        existing=empty_existing(),
        threshold_m=20_000.0,
        weight_scale=float(args.weight_scale),
        cache_size=int(args.cache_size),
        chunk_size=int(args.chunk_size),
    )
    exact_rows, exact_selected, exact_stats = solve_selected_budgets(
        spatial=spatial5,
        case="vietnam_grid5km_no_baseline_s20",
        threshold_km=20.0,
        budgets=[int(value) for value in args.budgets],
        output_dir=args.output_dir,
        time_limit_seconds=float(args.time_limit_seconds),
        mip_gap=float(args.mip_gap),
        incidence_chunk_size=int(args.incidence_chunk_size),
    )
    exact_df = pd.DataFrame(exact_rows)
    exact_df.to_csv(args.output_dir / "vietnam_5km_no_baseline_exact_s20.csv", index=False)
    pd.DataFrame(exact_selected).to_csv(args.output_dir / "vietnam_5km_no_baseline_exact_selected_candidates_s20.csv", index=False)
    pd.DataFrame([exact_stats]).to_csv(args.output_dir / "vietnam_5km_no_baseline_exact_stats_s20.csv", index=False)

    demand1, candidates1, provenance1 = load_candidate_set(
        outputs_dir=args.outputs_dir,
        run_tag_marker=args.run_tag_marker,
        candidate_grid=args.candidate_grid_1km,
        spacing_m=1000.0,
        weight_scale=float(args.weight_scale),
    )
    heuristic_df, _selected = run_1km_heuristic(
        demand=demand1,
        candidates=candidates1,
        budgets=[int(value) for value in args.budgets],
        output_dir=args.output_dir,
        weight_scale=float(args.weight_scale),
        cache_size=int(args.cache_size),
        chunk_size=int(args.chunk_size),
        local_max_moves=int(args.local_max_moves),
        local_max_candidates_per_drop=int(args.local_max_candidates_per_drop),
        elite_candidates=int(args.elite_candidates),
        randomized_repeats=int(args.randomized_repeats),
        rcl_size=int(args.rcl_size),
        seed=int(args.seed),
    )
    comparison = write_comparison(output_dir=args.output_dir, exact_5km=exact_df, heuristic_1km=heuristic_df)
    write_report(args.output_dir / "vietnam_no_baseline_s20_report.md", comparison)

    manifest = {
        "objective_definition": "selected_sites_total_population_no_existing_baseline",
        "threshold_km": 20.0,
        "budgets": [int(value) for value in args.budgets],
        "output_dir": str(args.output_dir),
        "time_limit_seconds": float(args.time_limit_seconds),
        "mip_gap": float(args.mip_gap),
        "provenance_5km": provenance5,
        "provenance_1km": provenance1,
        "outputs": {
            "exact_5km": str(args.output_dir / "vietnam_5km_no_baseline_exact_s20.csv"),
            "heuristic_1km": str(args.output_dir / "vietnam_1km_no_baseline_heuristic_s20.csv"),
            "comparison": str(args.output_dir / "vietnam_no_baseline_s20_1km_heuristic_vs_5km_exact.csv"),
            "report": str(args.output_dir / "vietnam_no_baseline_s20_report.md"),
        },
    }
    (args.output_dir / "vietnam_no_baseline_s20_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
