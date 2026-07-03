from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from time import perf_counter as pc
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
APPROX_SRC = SCRIPT_DIR.parents[2] / "approximated_tradeoff" / "src"
sys.path.insert(0, str(APPROX_SRC))
sys.path.insert(0, str(SCRIPT_DIR))

import mc_heuristics as mch  # noqa: E402
from vietnam_grasp_heuristics import budgeted_construct, improve_local_search, run_grasp  # noqa: E402
from vietnam_sparse_local_search import SparseSwapLocalSearch  # noqa: E402


LOCAL_ROOT = Path(r"C:\local\Parvathy\Vietnam")
OUTPUTS_DIR = LOCAL_ROOT / "fresh_downloads" / "vietnam_data" / "outputs"
RESULT_DIR = LOCAL_ROOT / "component_comparison"


@dataclass(frozen=True)
class SnapScenario:
    key: str
    label: str
    marker: str
    snap_components: str
    component_filter: str | None = None


@dataclass(slots=True)
class BuiltScenarioInstance:
    scenario: SnapScenario
    metric: str
    threshold_m: float
    instance: mch.MaxCoverInstance
    population_ids: np.ndarray
    raw_population: np.ndarray
    baseline_mask: np.ndarray
    candidate_ids: np.ndarray
    candidate_longitude: np.ndarray
    candidate_latitude: np.ndarray
    source_path: Path
    population_path: Path
    existing_matrix_path: Path
    candidate_matrix_path: Path
    distance_rows_retained: int
    candidate_rows_retained: int
    scale: float

    @property
    def threshold_km(self) -> float:
        return float(self.threshold_m) / 1000.0

    @property
    def total_population(self) -> float:
        return float(self.raw_population.sum())

    @property
    def baseline_population(self) -> float:
        return float(self.raw_population[self.baseline_mask].sum())

    @property
    def available_incremental_population(self) -> float:
        return float(self.instance.w.sum() / self.scale)


SNAP_SCENARIOS = (
    SnapScenario(
        key="all_components",
        label="all components",
        marker="candidates_spacing_10000_maxsnap_5000_connectivity",
        snap_components="all",
    ),
    SnapScenario(
        key="component_filter_0_1",
        label="component filter 0,1",
        marker="candidates_spacing_10000_maxsnap_5000_connectivity",
        snap_components="posthoc_filter_0,1",
        component_filter="0,1",
    ),
)

METRIC_LABELS = {
    "road_distance": "road distance only",
    "total_dist": "road + snap distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Vietnam stroke-center access under all-component and "
            "component-aware snapping, including baseline and added-facility frontiers."
        )
    )
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--thresholds-m", type=float, nargs="+", default=[20000, 50000, 100000])
    parser.add_argument("--metrics", nargs="+", default=["road_distance", "total_dist"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100, 150, 175, 200])
    parser.add_argument("--local-search-budgets", type=int, nargs="*", default=[175])
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--population-sample", type=int, default=35000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-grasp-p175", action="store_true")
    parser.add_argument("--grasp-iterations", type=int, default=3)
    parser.add_argument("--grasp-time-limit-seconds", type=float, default=90.0)
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--skip-maps", action="store_true")
    return parser.parse_args()


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) >= 0})


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def read_matrix(path: Path, metric: str) -> pd.DataFrame:
    columns = ["source_id", "target_id", metric]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        frame = pd.read_parquet(path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}") from None
        frame = frame[columns]
    frame = frame.rename(columns={metric: "distance_m"})
    frame["source_id"] = frame["source_id"].astype(str)
    frame["target_id"] = frame["target_id"].astype(str)
    return frame


def coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    for lon, lat in (("longitude", "latitude"), ("Longitude", "Latitude"), ("lon", "lat")):
        if lon in df.columns and lat in df.columns:
            return lon, lat
    raise KeyError(f"No supported coordinate columns in {list(df.columns)}")


def parse_component_filter(value: str | None) -> set[int] | None:
    if value is None or str(value).strip().lower() in {"", "all", "none", "null"}:
        return None
    allowed: set[int] = set()
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_s, end_s = text.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            allowed.update(range(min(start, end), max(start, end) + 1))
        else:
            allowed.add(int(text))
    return allowed


def apply_component_filter(
    frame: pd.DataFrame,
    *,
    scenario: SnapScenario,
    source_component_by_id: dict[str, int],
    target_component_by_id: dict[str, int],
) -> pd.DataFrame:
    allowed = parse_component_filter(scenario.component_filter)
    if allowed is None:
        return frame
    source_component = frame["source_id"].map(source_component_by_id)
    target_component = frame["target_id"].map(target_component_by_id)
    return frame.loc[source_component.isin(allowed) & target_component.isin(allowed)].copy()


def build_instance(
    *,
    outputs_dir: Path,
    scenario: SnapScenario,
    metric: str,
    threshold_m: float,
    weight_scale: float,
) -> BuiltScenarioInstance:
    population_path = find_one(outputs_dir, f"population_*{scenario.marker}.parquet")
    sources_path = find_one(outputs_dir, f"sources_*{scenario.marker}.parquet")
    existing_matrix_path = find_one(outputs_dir, f"distance_matrix_src_table_dst_population_*{scenario.marker}.parquet")
    candidate_matrix_path = find_one(outputs_dir, f"distance_matrix_src_candidates_dst_population_*{scenario.marker}.parquet")

    population = pd.read_parquet(population_path).reset_index(drop=True)
    sources = pd.read_parquet(sources_path).reset_index(drop=True)
    if not {"ID", "population"}.issubset(population.columns):
        raise ValueError(f"{population_path} must contain ID and population columns")
    if not {"ID", "source_type"}.issubset(sources.columns):
        raise ValueError(f"{sources_path} must contain ID and source_type columns")

    population_ids = population["ID"].astype(str).to_numpy()
    pop_id_to_idx = {pid: idx for idx, pid in enumerate(population_ids)}
    raw_population = population["population"].to_numpy(dtype=float)
    weights = np.rint(raw_population * weight_scale).astype(np.int64)
    source_component_by_id = (
        sources.set_index(sources["ID"].astype(str))["component_id"]
        .astype("Int64")
        .dropna()
        .astype(int)
        .to_dict()
        if "component_id" in sources.columns
        else {}
    )
    target_component_by_id = (
        population.set_index(population["ID"].astype(str))["component_id"]
        .astype("Int64")
        .dropna()
        .astype(int)
        .to_dict()
        if "component_id" in population.columns
        else {}
    )

    existing = read_matrix(existing_matrix_path, metric)
    existing = apply_component_filter(
        existing,
        scenario=scenario,
        source_component_by_id=source_component_by_id,
        target_component_by_id=target_component_by_id,
    )
    existing = existing.loc[np.isfinite(existing["distance_m"]) & (existing["distance_m"] <= threshold_m)]
    baseline_mask = np.zeros(len(population_ids), dtype=bool)
    for target_id in existing["target_id"].unique():
        idx = pop_id_to_idx.get(str(target_id))
        if idx is not None:
            baseline_mask[idx] = True

    effective_weights = weights.copy()
    effective_weights[baseline_mask] = 0

    candidates = read_matrix(candidate_matrix_path, metric)
    candidates = apply_component_filter(
        candidates,
        scenario=scenario,
        source_component_by_id=source_component_by_id,
        target_component_by_id=target_component_by_id,
    )
    candidates = candidates.loc[np.isfinite(candidates["distance_m"]) & (candidates["distance_m"] <= threshold_m)]
    candidate_ids = sorted(candidates["source_id"].dropna().astype(str).unique().tolist())
    candidate_id_to_j = {source_id: j for j, source_id in enumerate(candidate_ids)}

    ji_lists: list[np.ndarray] = []
    ij_lists: list[list[int]] = [[] for _ in range(len(population_ids))]
    grouped = {str(source_id): group for source_id, group in candidates.groupby("source_id", sort=False)}
    for source_id in candidate_ids:
        group = grouped.get(source_id)
        if group is None:
            households = np.empty(0, dtype=np.int32)
        else:
            households = np.asarray(
                sorted({
                    pop_id_to_idx[str(target_id)]
                    for target_id in group["target_id"]
                    if str(target_id) in pop_id_to_idx
                }),
                dtype=np.int32,
            )
        ji_lists.append(households)
        facility = candidate_id_to_j[source_id]
        for household in households:
            ij_lists[int(household)].append(facility)

    ij_arrays = [np.asarray(sorted(values), dtype=np.int32) for values in ij_lists]
    instance = mch.build_instance(
        effective_weights,
        ij_arrays,
        ji_lists,
        assume_unique_sorted=True,
        validate_consistency=False,
    )

    candidate_sources = sources.loc[sources["ID"].astype(str).isin(candidate_ids)].copy()
    candidate_sources = candidate_sources.set_index(candidate_sources["ID"].astype(str))
    lon_col, lat_col = coordinate_columns(candidate_sources.reset_index(drop=True))
    candidate_lon = np.full(len(candidate_ids), np.nan)
    candidate_lat = np.full(len(candidate_ids), np.nan)
    for j, source_id in enumerate(candidate_ids):
        if source_id in candidate_sources.index:
            candidate_lon[j] = float(candidate_sources.at[source_id, lon_col])
            candidate_lat[j] = float(candidate_sources.at[source_id, lat_col])

    return BuiltScenarioInstance(
        scenario=scenario,
        metric=metric,
        threshold_m=float(threshold_m),
        instance=instance,
        population_ids=population_ids,
        raw_population=raw_population,
        baseline_mask=baseline_mask,
        candidate_ids=np.asarray(candidate_ids, dtype="U"),
        candidate_longitude=candidate_lon,
        candidate_latitude=candidate_lat,
        source_path=sources_path,
        population_path=population_path,
        existing_matrix_path=existing_matrix_path,
        candidate_matrix_path=candidate_matrix_path,
        distance_rows_retained=int(len(existing) + len(candidates)),
        candidate_rows_retained=int(len(candidates)),
        scale=float(weight_scale),
    )


def objective_population(built: BuiltScenarioInstance, objective: int | float) -> float:
    return float(objective) / built.scale


def result_row(
    built: BuiltScenarioInstance,
    *,
    method: str,
    budget: int,
    objective: int | float,
    seconds: float,
    construction_seconds: float | None = None,
    local_search_moves: int | None = None,
    seed: int | None = None,
    repeat: int | None = None,
) -> dict[str, object]:
    incremental = objective_population(built, objective)
    total = built.baseline_population + incremental
    total_population = built.total_population
    return {
        "snap_policy": built.scenario.key,
        "snap_label": built.scenario.label,
        "snap_components": built.scenario.snap_components,
        "distance_metric": built.metric,
        "distance_metric_label": METRIC_LABELS.get(built.metric, built.metric),
        "threshold_m": float(built.threshold_m),
        "threshold_km": built.threshold_km,
        "method": method,
        "budget": int(budget),
        "seed": seed,
        "repeat": repeat,
        "existing_facilities": 130,
        "added_facilities": int(budget),
        "n_population": int(built.instance.n_households),
        "n_candidates": int(built.instance.n_facilities),
        "distance_rows_retained": built.distance_rows_retained,
        "candidate_rows_retained": built.candidate_rows_retained,
        "baseline_covered_population": built.baseline_population,
        "baseline_coverage_percent": 100.0 * built.baseline_population / total_population,
        "incremental_population": incremental,
        "total_covered_population": total,
        "coverage_percent_total_population": 100.0 * total / total_population,
        "available_incremental_population": built.available_incremental_population,
        "objective_weight_units": int(objective),
        "seconds": float(seconds),
        "construction_seconds": construction_seconds,
        "local_search_moves": local_search_moves,
        "population_path": str(built.population_path),
        "sources_path": str(built.source_path),
        "existing_matrix_path": str(built.existing_matrix_path),
        "candidate_matrix_path": str(built.candidate_matrix_path),
    }


def selected_candidates_frame(
    built: BuiltScenarioInstance,
    solution: list[int],
    *,
    method: str,
    budget: int,
    seed: int | None = None,
    repeat: int | None = None,
) -> pd.DataFrame:
    rows = []
    for rank, facility in enumerate(solution, start=1):
        facility_i = int(facility)
        rows.append(
            {
                "snap_policy": built.scenario.key,
                "snap_label": built.scenario.label,
                "distance_metric": built.metric,
                "threshold_km": built.threshold_km,
                "method": method,
                "budget": int(budget),
                "seed": seed,
                "repeat": repeat,
                "rank": rank,
                "facility_index": facility_i,
                "source_id": str(built.candidate_ids[facility_i]) if facility_i < len(built.candidate_ids) else "",
                "longitude": float(built.candidate_longitude[facility_i]) if facility_i < len(built.candidate_longitude) else np.nan,
                "latitude": float(built.candidate_latitude[facility_i]) if facility_i < len(built.candidate_latitude) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_instance(
    built: BuiltScenarioInstance,
    *,
    budgets: list[int],
    local_search_budgets: list[int],
    run_grasp_p175: bool,
    grasp_iterations: int,
    grasp_time_limit_seconds: float,
    rcl_size: int,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    trace_rows: list[dict[str, object]] = []

    for budget in budgets:
        if budget == 0:
            rows.append(result_row(built, method="baseline_existing_only", budget=0, objective=0, seconds=0.0))

    max_budget = max(budgets) if budgets else 0
    if max_budget > 0:
        greedy = budgeted_construct(built.instance, max_budget, constructor="greedy")
        previous = 0
        for step, objective in enumerate(greedy.objectives):
            trace_rows.append(
                {
                    "snap_policy": built.scenario.key,
                    "snap_label": built.scenario.label,
                    "distance_metric": built.metric,
                    "threshold_km": built.threshold_km,
                    "step": int(step),
                    "incremental_population": objective_population(built, objective),
                    "marginal_incremental_population": objective_population(built, int(objective) - previous),
                    "total_covered_population": built.baseline_population + objective_population(built, objective),
                    "coverage_percent_total_population": 100.0
                    * (built.baseline_population + objective_population(built, objective))
                    / built.total_population,
                    "seconds": float(greedy.times[step]) if step < len(greedy.times) else np.nan,
                }
            )
            previous = int(objective)

        for budget in budgets:
            if budget <= 0:
                continue
            idx = min(int(budget), len(greedy.objectives) - 1)
            rows.append(
                result_row(
                    built,
                    method="greedy_construction",
                    budget=budget,
                    objective=greedy.objectives[idx],
                    seconds=float(greedy.times[idx]) if idx < len(greedy.times) else float(greedy.total_time),
                )
            )

    sparse_index = None
    if local_search_budgets:
        sparse_index = SparseSwapLocalSearch.from_instance(built.instance)
    for budget in local_search_budgets:
        if budget <= 0:
            continue
        constructed = budgeted_construct(built.instance, budget, constructor="greedy")
        improved = improve_local_search(
            built.instance,
            constructed,
            local_search="first_sparse",
            sparse_local_search=sparse_index,
        )
        method = "greedy_first_sparse"
        rows.append(
            result_row(
                built,
                method=method,
                budget=budget,
                objective=improved.objective,
                seconds=constructed.total_time + improved.total_time,
                construction_seconds=constructed.total_time,
                local_search_moves=max(0, len(improved.objectives) - 1),
            )
        )
        selected_frames.append(selected_candidates_frame(built, improved.solution, method=method, budget=budget))

    if run_grasp_p175:
        grasp_budget = 175
        run_seed = int(seed + 1000 * round(built.threshold_km) + (0 if built.scenario.key == "all_components" else 500000))
        t0 = pc()
        best, records = run_grasp(
            built.instance,
            grasp_budget,
            time_limit_seconds=grasp_time_limit_seconds,
            max_iterations=grasp_iterations,
            constructor="randomized",
            rcl_size=rcl_size,
            sample_size=sample_size,
            local_search="first_sparse",
            path_relinking=True,
            path_relinking_method="fast",
            seed=run_seed,
            max_pool=8,
        )
        rows.append(
            result_row(
                built,
                method="randomized_grasp_first_sparse_fast_path_relinking",
                budget=grasp_budget,
                objective=best.objective,
                seconds=pc() - t0,
                seed=run_seed,
                repeat=0,
            )
        )
        selected_frames.append(
            selected_candidates_frame(
                built,
                best.solution,
                method="randomized_grasp_first_sparse_fast_path_relinking",
                budget=grasp_budget,
                seed=run_seed,
                repeat=0,
            )
        )
        trace_rows.extend(
            {
                "snap_policy": built.scenario.key,
                "snap_label": built.scenario.label,
                "distance_metric": built.metric,
                "threshold_km": built.threshold_km,
                "step": int(record.iteration),
                "incremental_population": objective_population(built, record.best_objective),
                "marginal_incremental_population": np.nan,
                "total_covered_population": built.baseline_population + objective_population(built, record.best_objective),
                "coverage_percent_total_population": 100.0
                * (built.baseline_population + objective_population(built, record.best_objective))
                / built.total_population,
                "seconds": float(record.total_seconds),
                "trace_type": "grasp_p175",
            }
            for record in records
        )

    trace = pd.DataFrame(trace_rows)
    return rows, selected_frames, trace


def best_p175(summary: pd.DataFrame) -> pd.DataFrame:
    p175 = summary.loc[summary["budget"].eq(175)].copy()
    if p175.empty:
        return p175
    p175 = p175.sort_values(
        [
            "snap_policy",
            "distance_metric",
            "threshold_km",
            "coverage_percent_total_population",
            "seconds",
        ],
        ascending=[True, True, True, False, True],
    )
    return p175.groupby(["snap_policy", "distance_metric", "threshold_km"], as_index=False).head(1)


def write_existing_and_p175_summary(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline = summary.loc[summary["method"].eq("baseline_existing_only")].copy()
    chosen = best_p175(summary)
    rows = []
    for _, row in baseline.iterrows():
        rows.append({**row.to_dict(), "stage": "existing 130"})
    for _, row in chosen.iterrows():
        rows.append({**row.to_dict(), "stage": "existing 130 + selected 175"})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "vietnam_existing_and_p175_access_summary.csv", index=False)

    key_cols = ["distance_metric", "threshold_km", "stage"]
    pivot = out.pivot_table(
        index=key_cols,
        columns="snap_label",
        values="coverage_percent_total_population",
        aggfunc="first",
    ).reset_index()
    if {"all components", "component filter 0,1"}.issubset(pivot.columns):
        pivot["component_filter_minus_all_pp"] = pivot["component filter 0,1"] - pivot["all components"]
    pivot.to_csv(output_dir / "vietnam_component_metric_delta.csv", index=False)
    return out


def write_markdown_note(summary: pd.DataFrame, output_dir: Path, manifest: dict[str, object]) -> Path:
    chosen = best_p175(summary)
    note_path = output_dir / "vietnam_component_comparison_note.md"
    lines = [
        "# Vietnam Component-Aware Stroke Access Comparison",
        "",
        "This run compares the current 130 stroke centers with a 175-candidate addition under two snapping policies.",
        "",
        "- `all components`: original nearest-road snapping; every weak road component can receive a snapped point.",
        "- `component filter 0,1`: diagnostic fallback; it keeps only all-components matrix rows whose already-snapped source and target are on components 0 or 1.",
        "",
        "Important: this fallback is not the same as a true `--snap-components 0,1` rerun, because points on minor components are not re-snapped to the allowed components. A true constrained-snap run was attempted, but the national Vietnam Pandana graph build did not progress beyond `Building Pandana network` in the available run window.",
        "- `road_distance`: routed road distance only.",
        "- `total_dist`: routed road distance plus source/target snap distance, used as a drive-plus-access-distance metric.",
        "",
        "## Best 175-Addition Rows",
        "",
    ]
    if chosen.empty:
        lines.append("No budget-175 rows were generated.")
    else:
        cols = [
            "snap_label",
            "distance_metric",
            "threshold_km",
            "method",
            "coverage_percent_total_population",
            "total_covered_population",
            "seconds",
        ]
        display = chosen[cols].sort_values(["distance_metric", "threshold_km", "snap_label"])
        lines.append(display.to_markdown(index=False, floatfmt=".3f"))
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "The analysis inputs are the PISA split sparse matrices in:",
            "",
            f"`{manifest['outputs_dir']}`",
            "",
            "The run manifest is stored as `analysis_manifest.json` in this folder.",
            "",
        ]
    )
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def write_frontier_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    greedy = summary.loc[summary["method"].eq("greedy_construction")].copy()
    local = summary.loc[summary["budget"].eq(175) & ~summary["method"].eq("greedy_construction")].copy()
    metrics = [metric for metric in ["road_distance", "total_dist"] if metric in summary["distance_metric"].unique()]
    thresholds = sorted(summary["threshold_km"].unique())
    fig, axes = plt.subplots(len(metrics), len(thresholds), figsize=(4.4 * len(thresholds), 3.8 * len(metrics)), squeeze=False)
    colors = {"all_components": "#26547c", "component_filter_0_1": "#c44536"}
    for r, metric in enumerate(metrics):
        for c, threshold in enumerate(thresholds):
            ax = axes[r][c]
            for snap_policy, group in greedy.loc[
                greedy["distance_metric"].eq(metric) & greedy["threshold_km"].eq(threshold)
            ].groupby("snap_policy"):
                group = group.sort_values("budget")
                ax.plot(
                    group["budget"],
                    group["coverage_percent_total_population"],
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                    color=colors.get(snap_policy, "#666666"),
                    label=str(group["snap_label"].iloc[0]),
                )
            for snap_policy, group in local.loc[
                local["distance_metric"].eq(metric) & local["threshold_km"].eq(threshold)
            ].groupby("snap_policy"):
                best = group.sort_values(["coverage_percent_total_population", "seconds"], ascending=[False, True]).iloc[0]
                ax.scatter(
                    [best["budget"]],
                    [best["coverage_percent_total_population"]],
                    s=80,
                    marker="*",
                    color=colors.get(snap_policy, "#666666"),
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=5,
                )
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\n{threshold:g} km")
            ax.set_xlabel("Added candidate facilities")
            ax.set_ylabel("Covered population (%)")
            ax.grid(True, alpha=0.25)
            if r == 0 and c == len(thresholds) - 1:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Vietnam stroke access frontiers by snapping policy", y=1.02, fontsize=14)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"vietnam_component_frontiers_by_snap_metric.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_stage_bar_plot(stage_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if stage_summary.empty:
        return
    order = stage_summary.sort_values(["distance_metric", "threshold_km", "snap_label", "stage"])
    labels = [
        f"{row.distance_metric}\n{row.threshold_km:g}km\n{row.snap_label}"
        for row in order.itertuples()
    ]
    colors = ["#9aa4a8" if stage == "existing 130" else "#2a9d8f" for stage in order["stage"]]
    fig, ax = plt.subplots(figsize=(max(10, 0.46 * len(order)), 5.2))
    ax.bar(np.arange(len(order)), order["coverage_percent_total_population"], color=colors, width=0.82)
    ax.set_ylabel("Covered population (%)")
    ax.set_title("Vietnam existing 130 centers versus best 175-candidate addition")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(max(0, order["coverage_percent_total_population"].min() - 5), min(100, order["coverage_percent_total_population"].max() + 3))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"vietnam_existing_vs_p175_summary.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_delta_heatmap(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    path = output_dir / "vietnam_component_metric_delta.csv"
    if not path.exists():
        return
    delta = pd.read_csv(path)
    if "component_filter_minus_all_pp" not in delta.columns or delta.empty:
        return
    delta["row_label"] = delta["distance_metric"].astype(str) + " | " + delta["stage"].astype(str)
    pivot = delta.pivot_table(index="row_label", columns="threshold_km", values="component_filter_minus_all_pp", aggfunc="first")
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.55 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g} km" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Component-filter minus all-components coverage (percentage points)")
    fig.colorbar(image, ax=ax, label="pp")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"vietnam_component_delta_heatmap.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def method_slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower() or "method"


def selected_for_best(selected: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    mask = (
        selected["snap_policy"].astype(str).eq(str(best["snap_policy"]))
        & selected["distance_metric"].astype(str).eq(str(best["distance_metric"]))
        & np.isclose(selected["threshold_km"].astype(float), float(best["threshold_km"]))
        & selected["budget"].astype(int).eq(int(best["budget"]))
        & selected["method"].astype(str).eq(str(best["method"]))
    )
    return selected.loc[mask].sort_values("rank")


def write_solution_map(
    *,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    outputs_dir: Path,
    output_dir: Path,
    population_sample: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    if selected.empty:
        return
    chosen_rows = best_p175(summary)
    chosen_rows = chosen_rows.loc[
        chosen_rows["distance_metric"].eq("total_dist") & np.isclose(chosen_rows["threshold_km"], 50.0)
    ].copy()
    if chosen_rows.empty:
        return

    background_path = find_one(outputs_dir, f"population_*{SNAP_SCENARIOS[0].marker}.parquet")
    sources_path = find_one(outputs_dir, f"sources_*{SNAP_SCENARIOS[0].marker}.parquet")
    population = pd.read_parquet(background_path)
    if len(population) > population_sample:
        population = population.sample(population_sample, random_state=seed)
    sources = pd.read_parquet(sources_path)
    source_type = sources.get("source_type", pd.Series("", index=sources.index)).astype(str)
    existing = sources[source_type.isin(["table", "existing"])].copy()
    pop_lon, pop_lat = coordinate_columns(population)
    ex_lon, ex_lat = coordinate_columns(existing)

    fig, axes = plt.subplots(1, len(chosen_rows), figsize=(6.2 * len(chosen_rows), 8.2), squeeze=False)
    for ax, (_, best) in zip(axes[0], chosen_rows.sort_values("snap_policy").iterrows()):
        candidates = selected_for_best(selected, best)
        values = population["population"].to_numpy(dtype=float) if "population" in population.columns else np.ones(len(population))
        sizes = 1.2 + 10.0 * np.sqrt(values / max(values.max(), 1.0))
        ax.scatter(population[pop_lon], population[pop_lat], s=sizes, c="#cfcfcf", alpha=0.35, linewidths=0)
        ax.scatter(existing[ex_lon], existing[ex_lat], s=20, c="#111111", marker="x", linewidths=0.8, label="existing 130")
        ax.scatter(
            candidates["longitude"],
            candidates["latitude"],
            s=22,
            c="#d1495b",
            edgecolors="white",
            linewidths=0.4,
            label="selected 175",
            zorder=4,
        )
        ax.set_title(
            f"{best['snap_label']}\n50 km total_dist, {best['coverage_percent_total_population']:.2f}% covered",
            fontsize=10,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.18)
        ax.set_aspect("equal", adjustable="box")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Vietnam selected 175 candidate centers under snapping policies", y=0.99, fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"vietnam_p175_solution_map_totaldist_50km.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = unique_sorted(args.budgets)
    local_search_budgets = unique_sorted(args.local_search_budgets)
    manifest = {
        "outputs_dir": str(args.outputs_dir),
        "output_dir": str(args.output_dir),
        "snap_scenarios": [scenario.__dict__ for scenario in SNAP_SCENARIOS],
        "metrics": args.metrics,
        "thresholds_m": args.thresholds_m,
        "budgets": budgets,
        "local_search_budgets": local_search_budgets,
        "run_grasp_p175": bool(args.run_grasp_p175),
        "grasp_iterations": int(args.grasp_iterations),
        "grasp_time_limit_seconds": float(args.grasp_time_limit_seconds),
        "seed": int(args.seed),
        "created_by": "run_vietnam_component_comparison.py",
    }

    summary_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []

    for scenario in SNAP_SCENARIOS:
        for metric in args.metrics:
            for threshold_m in args.thresholds_m:
                print(f"Building {scenario.key}, {metric}, {threshold_m / 1000:g} km")
                built = build_instance(
                    outputs_dir=args.outputs_dir,
                    scenario=scenario,
                    metric=metric,
                    threshold_m=float(threshold_m),
                    weight_scale=float(args.weight_scale),
                )
                rows, selected, trace = run_instance(
                    built,
                    budgets=budgets,
                    local_search_budgets=local_search_budgets,
                    run_grasp_p175=bool(args.run_grasp_p175),
                    grasp_iterations=int(args.grasp_iterations),
                    grasp_time_limit_seconds=float(args.grasp_time_limit_seconds),
                    rcl_size=int(args.rcl_size),
                    sample_size=int(args.sample_size),
                    seed=int(args.seed),
                )
                summary_rows.extend(rows)
                selected_frames.extend(selected)
                traces.append(trace)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["distance_metric", "threshold_km", "snap_policy", "budget", "method"]
    )
    summary.to_csv(args.output_dir / "vietnam_frontier_by_budget.csv", index=False)

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    selected_all.to_csv(args.output_dir / "vietnam_selected_candidates_p175.csv", index=False)

    trace_all = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    trace_all.to_csv(args.output_dir / "vietnam_greedy_traces.csv", index=False)

    stage_summary = write_existing_and_p175_summary(summary, args.output_dir)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    note_path = write_markdown_note(summary, args.output_dir, manifest)

    write_frontier_plot(summary, args.output_dir)
    write_stage_bar_plot(stage_summary, args.output_dir)
    write_delta_heatmap(args.output_dir)
    if not args.skip_maps:
        write_solution_map(
            summary=summary,
            selected=selected_all,
            outputs_dir=args.outputs_dir,
            output_dir=args.output_dir,
            population_sample=int(args.population_sample),
            seed=int(args.seed),
        )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_rows": int(len(summary)),
                "selected_rows": int(len(selected_all)),
                "trace_rows": int(len(trace_all)),
                "note": str(note_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
