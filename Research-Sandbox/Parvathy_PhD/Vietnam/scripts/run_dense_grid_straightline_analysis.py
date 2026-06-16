from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter as pc

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


OUTPUT_ROOT = Path(r"C:\local\Parvathy\Vietnam")
DEFAULT_OUTPUTS_DIR = OUTPUT_ROOT / "fresh_downloads" / "vietnam_data" / "outputs"
DEFAULT_ANALYSIS_DIR = OUTPUT_ROOT / "dense_grid_straightline_analysis"
DEFAULT_RUN_TAG_MARKER = (
    "pop_1_sample_1_seed_42_max_none_agg_10_maxdist_150000_amenity_amenity_all-"
    "dst_population-src_table_stroke_facs_100_en-candidates_candidates_spacing_10000_"
    "maxsnap_5000_connectivity"
)


@dataclass(slots=True)
class DemandData:
    ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    xy: np.ndarray
    population: np.ndarray
    full_weights: np.ndarray


@dataclass(slots=True)
class CandidateData:
    ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    xy: np.ndarray


@dataclass(slots=True)
class ExistingFacilities:
    lon: np.ndarray
    lat: np.ndarray
    xy: np.ndarray


@dataclass(slots=True)
class SpatialResult:
    solution: list[int]
    objective: int
    coverage: np.ndarray
    objectives: list[int]
    times: list[float]
    total_time: float
    moves: int = 0


class SpatialMaxCover:
    def __init__(
        self,
        *,
        demand: DemandData,
        candidates: CandidateData,
        existing: ExistingFacilities,
        threshold_m: float,
        weight_scale: float,
        cache_size: int,
        chunk_size: int,
    ) -> None:
        self.demand = demand
        self.candidates = candidates
        self.existing = existing
        self.threshold_m = float(threshold_m)
        self.threshold_km = self.threshold_m / 1000.0
        self.weight_scale = float(weight_scale)
        self.pop_tree = cKDTree(demand.xy)
        self.candidate_tree = cKDTree(candidates.xy)
        self.cache_size = int(cache_size)
        self.chunk_size = int(chunk_size)
        self._cover_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.baseline_mask = self._compute_baseline_mask()
        self.effective_weights = demand.full_weights.copy()
        self.effective_weights[self.baseline_mask] = 0
        self.initial_gains: np.ndarray | None = None
        self.initial_gain_seconds: float | None = None

    @property
    def n_candidates(self) -> int:
        return int(len(self.candidates.ids))

    @property
    def n_population(self) -> int:
        return int(len(self.demand.ids))

    @property
    def total_population(self) -> float:
        return float(self.demand.population.sum())

    @property
    def baseline_population(self) -> float:
        return float(self.demand.population[self.baseline_mask].sum())

    @property
    def available_incremental_population(self) -> float:
        return float(self.effective_weights.sum() / self.weight_scale)

    def _compute_baseline_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_population, dtype=bool)
        if len(self.existing.xy) == 0:
            return mask
        neighbours = self.pop_tree.query_ball_point(self.existing.xy, self.threshold_m)
        for values in neighbours:
            if values:
                mask[np.asarray(values, dtype=np.int32)] = True
        return mask

    def households_of(self, facility: int) -> np.ndarray:
        facility_i = int(facility)
        cached = self._cover_cache.get(facility_i)
        if cached is not None:
            self._cover_cache.move_to_end(facility_i)
            return cached
        neighbours = self.pop_tree.query_ball_point(
            self.candidates.xy[facility_i],
            self.threshold_m,
        )
        out = np.asarray(neighbours, dtype=np.int32)
        self._cover_cache[facility_i] = out
        if len(self._cover_cache) > self.cache_size:
            self._cover_cache.popitem(last=False)
        return out

    def candidates_for_households(self, households: np.ndarray, max_candidates: int) -> np.ndarray:
        if households.size == 0:
            return np.empty(0, dtype=np.int32)
        touched: list[np.ndarray] = []
        for household in households:
            neighbours = self.candidate_tree.query_ball_point(
                self.demand.xy[int(household)],
                self.threshold_m,
            )
            if neighbours:
                touched.append(np.asarray(neighbours, dtype=np.int32))
        if not touched:
            return np.empty(0, dtype=np.int32)
        all_touched = np.concatenate(touched)
        if all_touched.size <= max_candidates:
            return np.unique(all_touched)
        unique, counts = np.unique(all_touched, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        return unique[order[:max_candidates]].astype(np.int32, copy=False)

    def compute_initial_gains(self, *, force: bool = False) -> np.ndarray:
        if self.initial_gains is not None and not force:
            return self.initial_gains.copy()
        gains = np.zeros(self.n_candidates, dtype=np.int64)
        start = pc()
        weights = self.effective_weights
        for begin in range(0, self.n_candidates, self.chunk_size):
            end = min(begin + self.chunk_size, self.n_candidates)
            neighbours = self.pop_tree.query_ball_point(
                self.candidates.xy[begin:end],
                self.threshold_m,
            )
            for offset, values in enumerate(neighbours):
                if values:
                    gains[begin + offset] = int(weights[np.asarray(values, dtype=np.int32)].sum())
            if begin == 0 or end == self.n_candidates or (begin // self.chunk_size) % 25 == 0:
                print(
                    f"  initial gains {end:,}/{self.n_candidates:,} "
                    f"for {self.threshold_km:g} km"
                )
        self.initial_gain_seconds = float(pc() - start)
        self.initial_gains = gains
        return gains.copy()

    def subtract_newly_covered(
        self,
        *,
        gains: np.ndarray,
        newly_covered: np.ndarray,
        selected_mask: np.ndarray,
    ) -> None:
        if newly_covered.size == 0:
            return
        touched_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for household in newly_covered:
            household_i = int(household)
            weight = int(self.effective_weights[household_i])
            if weight <= 0:
                continue
            neighbours = self.candidate_tree.query_ball_point(
                self.demand.xy[household_i],
                self.threshold_m,
            )
            if not neighbours:
                continue
            arr = np.asarray(neighbours, dtype=np.int32)
            keep = (~selected_mask[arr]) & (gains[arr] >= 0)
            if keep.any():
                kept = arr[keep]
                touched_parts.append(kept)
                weight_parts.append(np.full(kept.size, weight, dtype=np.int64))
        if not touched_parts:
            return
        touched = np.concatenate(touched_parts)
        weights = np.concatenate(weight_parts)
        np.add.at(gains, touched, -weights)

    def construct(
        self,
        *,
        budget: int,
        randomized: bool,
        rcl_size: int,
        seed: int,
    ) -> SpatialResult:
        gains = self.compute_initial_gains()
        uncovered = self.effective_weights > 0
        coverage = np.zeros(self.n_population, dtype=np.int16)
        selected_mask = np.zeros(self.n_candidates, dtype=bool)
        solution: list[int] = []
        objective = 0
        objectives = [0]
        times = [0.0]
        rng = np.random.default_rng(seed)
        start = pc()
        for _ in range(int(budget)):
            if randomized:
                positive = int(np.count_nonzero(gains > 0))
                if positive <= 0:
                    break
                k = min(int(rcl_size), positive)
                top = np.argpartition(gains, -k)[-k:]
                top = top[gains[top] > 0]
                if top.size == 0:
                    break
                facility = int(rng.choice(top))
            else:
                facility = int(np.argmax(gains))
            if int(gains[facility]) <= 0:
                break
            selected_mask[facility] = True
            gains[facility] = -1
            covered = self.households_of(facility)
            newly = covered[uncovered[covered]]
            if covered.size:
                coverage[covered] += 1
            if newly.size:
                uncovered[newly] = False
                objective += int(self.effective_weights[newly].sum())
                self.subtract_newly_covered(
                    gains=gains,
                    newly_covered=newly,
                    selected_mask=selected_mask,
                )
            solution.append(facility)
            objectives.append(int(objective))
            times.append(float(pc() - start))
        return SpatialResult(
            solution=solution,
            objective=int(objective),
            coverage=coverage,
            objectives=objectives,
            times=times,
            total_time=float(pc() - start),
        )

    def coverage_for_solution(self, solution: list[int]) -> tuple[np.ndarray, int]:
        coverage = np.zeros(self.n_population, dtype=np.int16)
        for facility in solution:
            covered = self.households_of(int(facility))
            if covered.size:
                coverage[covered] += 1
        objective = int(self.effective_weights[coverage > 0].sum())
        return coverage, objective

    def prefix_result(self, result: SpatialResult, budget: int) -> SpatialResult:
        solution = list(result.solution[: int(budget)])
        coverage, objective = self.coverage_for_solution(solution)
        idx = min(int(budget), len(result.objectives) - 1)
        return SpatialResult(
            solution=solution,
            objective=int(objective),
            coverage=coverage,
            objectives=list(result.objectives[: idx + 1]),
            times=list(result.times[: idx + 1]),
            total_time=float(result.times[idx]) if idx < len(result.times) else float(result.total_time),
        )

    def improve_local(
        self,
        result: SpatialResult,
        *,
        max_moves: int,
        max_candidates_per_drop: int,
        elite_candidates: np.ndarray,
    ) -> SpatialResult:
        sol = list(result.solution)
        if not sol:
            return result
        coverage = result.coverage.copy()
        objective = int(self.effective_weights[coverage > 0].sum())
        open_mask = np.zeros(self.n_candidates, dtype=bool)
        for facility in sol:
            open_mask[int(facility)] = True
        lost_marker = np.zeros(self.n_population, dtype=bool)
        objectives = [int(objective)]
        times = [0.0]
        moves = 0
        start = pc()

        for _ in range(int(max_moves)):
            improved = False
            for position, removed in enumerate(list(sol)):
                removed_i = int(removed)
                removed_cover = self.households_of(removed_i)
                if removed_cover.size == 0:
                    continue
                newly_uncovered = removed_cover[coverage[removed_cover] == 1]
                loss = int(self.effective_weights[newly_uncovered].sum()) if newly_uncovered.size else 0
                pool = self.candidates_for_households(
                    newly_uncovered,
                    max_candidates=max_candidates_per_drop,
                )
                if pool.size < max_candidates_per_drop and elite_candidates.size:
                    pool = np.unique(np.concatenate([pool, elite_candidates]))
                if pool.size == 0:
                    continue
                if newly_uncovered.size:
                    lost_marker[newly_uncovered] = True

                best_add = -1
                best_net = 0
                for add in pool:
                    add_i = int(add)
                    if open_mask[add_i]:
                        continue
                    add_cover = self.households_of(add_i)
                    if add_cover.size == 0:
                        continue
                    current_uncovered = add_cover[coverage[add_cover] == 0]
                    gain = int(self.effective_weights[current_uncovered].sum()) if current_uncovered.size else 0
                    if newly_uncovered.size:
                        recovered = add_cover[lost_marker[add_cover]]
                        if recovered.size:
                            gain += int(self.effective_weights[recovered].sum())
                    net = gain - loss
                    if net > best_net:
                        best_net = int(net)
                        best_add = add_i
                        break

                if newly_uncovered.size:
                    lost_marker[newly_uncovered] = False

                if best_add >= 0:
                    coverage[removed_cover] -= 1
                    add_cover = self.households_of(best_add)
                    if add_cover.size:
                        coverage[add_cover] += 1
                    open_mask[removed_i] = False
                    open_mask[best_add] = True
                    sol[position] = best_add
                    objective += best_net
                    moves += 1
                    objectives.append(int(objective))
                    times.append(float(pc() - start))
                    improved = True
                    break

            if not improved:
                break

        return SpatialResult(
            solution=sol,
            objective=int(objective),
            coverage=coverage,
            objectives=objectives,
            times=times,
            total_time=float(pc() - start),
            moves=moves,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Vietnam 5 km/1 km dense-grid straight-line screening with "
            "spatial greedy, local search, GRASP, and solution figures."
        )
    )
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--run-tag-marker", default=DEFAULT_RUN_TAG_MARKER)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--grid-spacing-m", type=float, required=True)
    parser.add_argument("--thresholds-km", type=float, nargs="+", default=[20.0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 80, 200])
    parser.add_argument("--local-search-budgets", type=int, nargs="*", default=[20, 80, 200])
    parser.add_argument("--randomized-budgets", type=int, nargs="*", default=[20])
    parser.add_argument("--randomized-repeats", type=int, default=2)
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--local-max-moves", type=int, default=8)
    parser.add_argument("--local-max-candidates-per-drop", type=int, default=6000)
    parser.add_argument("--elite-candidates", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--cache-size", type=int, default=20000)
    parser.add_argument("--weight-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--population-sample", type=int, default=30000)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def transformer() -> Transformer:
    return Transformer.from_crs("EPSG:4326", "EPSG:3405", always_xy=True)


def project(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x, y = transformer().transform(lon, lat)
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    for lon, lat in [
        ("Longitude", "Latitude"),
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("xcoord", "ycoord"),
    ]:
        if lon in df.columns and lat in df.columns:
            return lon, lat
    raise KeyError(f"No supported coordinate columns in {list(df.columns)}")


def load_inputs(
    *,
    outputs_dir: Path,
    marker: str,
    candidate_grid_path: Path,
    grid_spacing_m: float,
    weight_scale: float,
) -> tuple[DemandData, CandidateData, ExistingFacilities, dict]:
    population_path = find_one(outputs_dir, f"population_*{marker}*.parquet")
    sources_path = find_one(outputs_dir, f"sources_*{marker}*.parquet")
    population = pd.read_parquet(population_path).reset_index(drop=True)
    sources = pd.read_parquet(sources_path).reset_index(drop=True)
    candidates = pd.read_pickle(candidate_grid_path).reset_index(drop=True)

    pop_lon_col, pop_lat_col = coordinate_columns(population)
    cand_lon_col, cand_lat_col = coordinate_columns(candidates)
    source_lon_col, source_lat_col = coordinate_columns(sources)

    pop_lon = population[pop_lon_col].to_numpy(dtype=float)
    pop_lat = population[pop_lat_col].to_numpy(dtype=float)
    raw_population = population["population"].to_numpy(dtype=float)
    full_weights = np.rint(raw_population * float(weight_scale)).astype(np.int64)
    demand = DemandData(
        ids=population["ID"].astype(str).to_numpy(),
        lon=pop_lon,
        lat=pop_lat,
        xy=project(pop_lon, pop_lat),
        population=raw_population,
        full_weights=full_weights,
    )

    cand_lon = candidates[cand_lon_col].to_numpy(dtype=float)
    cand_lat = candidates[cand_lat_col].to_numpy(dtype=float)
    if "geometry" in candidates.columns and getattr(candidates, "crs", None) is not None:
        try:
            cand_xy = np.column_stack([candidates.geometry.x.to_numpy(), candidates.geometry.y.to_numpy()])
        except Exception:
            cand_xy = project(cand_lon, cand_lat)
    else:
        cand_xy = project(cand_lon, cand_lat)
    raw_ids = candidates["ID"].astype(str).to_numpy() if "ID" in candidates.columns else np.arange(len(candidates)).astype(str)
    candidate_data = CandidateData(
        ids=np.asarray([f"grid_{grid_spacing_m:g}m_{value}" for value in raw_ids], dtype=str),
        lon=cand_lon,
        lat=cand_lat,
        xy=np.asarray(cand_xy, dtype=float),
    )

    source_type = sources.get("source_type", pd.Series("", index=sources.index)).astype(str)
    existing = sources[source_type.isin(["table", "existing"])].copy()
    existing_lon = existing[source_lon_col].to_numpy(dtype=float)
    existing_lat = existing[source_lat_col].to_numpy(dtype=float)
    existing_facilities = ExistingFacilities(
        lon=existing_lon,
        lat=existing_lat,
        xy=project(existing_lon, existing_lat) if len(existing_lon) else np.empty((0, 2), dtype=float),
    )

    provenance = {
        "population_path": str(population_path),
        "sources_path": str(sources_path),
        "candidate_grid_path": str(candidate_grid_path),
        "n_existing_facilities": int(len(existing_facilities.lon)),
    }
    return demand, candidate_data, existing_facilities, provenance


def objective_to_population(value: int | float, scale: float) -> float:
    return float(value) / float(scale)


def summary_row(
    *,
    instance_name: str,
    marker: str,
    grid_spacing_m: float,
    distance_model: str,
    spatial: SpatialMaxCover,
    method: str,
    budget: int,
    objective: int | float,
    seconds: float,
    construction_objective: int | float | None = None,
    construction_seconds: float | None = None,
    local_search_moves: int | None = None,
    seed: int | None = None,
    repeat: int | None = None,
    status: str = "ok",
) -> dict:
    incremental = objective_to_population(objective, spatial.weight_scale)
    total = spatial.baseline_population + incremental
    return {
        "instance": instance_name,
        "run_tag_marker": marker,
        "distance_model": distance_model,
        "grid_spacing_m": float(grid_spacing_m),
        "threshold_km": spatial.threshold_km,
        "n_population": spatial.n_population,
        "n_candidates": spatial.n_candidates,
        "budget": int(budget),
        "method": method,
        "seed": seed,
        "repeat": repeat,
        "status": status,
        "construction_incremental_population": (
            objective_to_population(construction_objective, spatial.weight_scale)
            if construction_objective is not None
            else np.nan
        ),
        "incremental_population": incremental,
        "baseline_covered_population": spatial.baseline_population,
        "total_covered_population": total,
        "coverage_percent_total_population": (
            100.0 * total / spatial.total_population if spatial.total_population else np.nan
        ),
        "incremental_percent_total_population": (
            100.0 * incremental / spatial.total_population if spatial.total_population else np.nan
        ),
        "available_incremental_population": spatial.available_incremental_population,
        "objective_weight_units": int(objective),
        "construction_seconds": construction_seconds,
        "seconds": float(seconds),
        "initial_gain_seconds": spatial.initial_gain_seconds,
        "local_search_moves": local_search_moves,
    }


def selected_frame(
    *,
    candidates: CandidateData,
    instance_name: str,
    solution: list[int],
    method: str,
    budget: int,
    threshold_km: float,
    seed: int | None = None,
    repeat: int | None = None,
) -> pd.DataFrame:
    rows = []
    for rank, facility in enumerate(solution, start=1):
        facility_i = int(facility)
        rows.append(
            {
                "instance": instance_name,
                "rank": rank,
                "facility_index": facility_i,
                "source_id": str(candidates.ids[facility_i]),
                "longitude": float(candidates.lon[facility_i]),
                "latitude": float(candidates.lat[facility_i]),
                "method": method,
                "budget": int(budget),
                "threshold_km": float(threshold_km),
                "seed": seed,
                "repeat": repeat,
            }
        )
    return pd.DataFrame(rows)


def method_slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text or "method"


def choose_best_rows(summary: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    keep = summary[summary["budget"].isin(budgets)].copy()
    if keep.empty:
        return keep
    keep = keep.sort_values(
        ["grid_spacing_m", "threshold_km", "budget", "total_covered_population", "seconds"],
        ascending=[True, True, True, False, True],
    )
    return keep.groupby(["grid_spacing_m", "threshold_km", "budget"], as_index=False).head(1)


def selected_for_best(selected: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    mask = (
        (selected["instance"].astype(str) == str(best["instance"]))
        & np.isclose(selected["threshold_km"].astype(float), float(best["threshold_km"]))
        & (selected["budget"].astype(int) == int(best["budget"]))
        & (selected["method"].astype(str) == str(best["method"]))
    )
    if pd.notna(best.get("seed")):
        mask &= selected["seed"].fillna(-1).astype(float) == float(best["seed"])
    if pd.notna(best.get("repeat")):
        mask &= selected["repeat"].fillna(-1).astype(float) == float(best["repeat"])
    return selected[mask].sort_values("rank")


def write_solution_figures(
    *,
    output_dir: Path,
    demand: DemandData,
    existing: ExistingFacilities,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    budgets: list[int],
    population_sample: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "solution_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(demand.lon) > population_sample:
        sample_idx = rng.choice(len(demand.lon), size=int(population_sample), replace=False)
    else:
        sample_idx = np.arange(len(demand.lon))

    rows = []
    pop_sizes = 1.4 + 12.0 * np.sqrt(
        demand.population[sample_idx] / max(float(demand.population[sample_idx].max()), 1.0)
    )
    for _, best in choose_best_rows(summary, budgets).iterrows():
        chosen = selected_for_best(selected, best)
        if chosen.empty:
            rows.append({**best.to_dict(), "figure": "", "figure_status": "missing selected candidates"})
            continue
        grid_km = float(best["grid_spacing_m"]) / 1000.0
        threshold = float(best["threshold_km"])
        budget = int(best["budget"])
        method = str(best["method"])
        filename = (
            f"solution_grid{grid_km:g}km_threshold{threshold:g}km_"
            f"p{budget}_{method_slug(method)}.png"
        )
        outpath = figure_dir / filename
        fig, ax = plt.subplots(figsize=(7.5, 9.5))
        ax.scatter(
            demand.lon[sample_idx],
            demand.lat[sample_idx],
            s=pop_sizes,
            c="#d0d0d0",
            alpha=0.35,
            linewidths=0,
            label="population sample",
        )
        if len(existing.lon):
            ax.scatter(
                existing.lon,
                existing.lat,
                s=18,
                c="#111111",
                marker="x",
                linewidths=0.8,
                alpha=0.75,
                label="existing stroke facilities",
            )
        ax.scatter(
            chosen["longitude"],
            chosen["latitude"],
            s=28,
            c="#d62728",
            edgecolors="white",
            linewidths=0.5,
            label="selected candidates",
            zorder=5,
        )
        covered_m = float(best["total_covered_population"]) / 1_000_000.0
        seconds = float(best["seconds"])
        ax.set_title(
            f"Vietnam grid {grid_km:g} km, threshold {threshold:g} km, p={budget}\n"
            f"{method} | {covered_m:.2f}M covered | {seconds:.1f}s",
            fontsize=11,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="lower left", fontsize=8, frameon=True)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        rows.append({**best.to_dict(), "figure": str(outpath), "figure_status": "ok"})
    pd.DataFrame(rows).to_csv(figure_dir / "solution_figure_index.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distance_model = "straight_line_projected_screening"
    demand, candidates, existing, provenance = load_inputs(
        outputs_dir=args.outputs_dir,
        marker=args.run_tag_marker,
        candidate_grid_path=args.candidate_grid,
        grid_spacing_m=float(args.grid_spacing_m),
        weight_scale=float(args.weight_scale),
    )
    budgets = sorted({int(value) for value in args.budgets if int(value) > 0})
    local_budgets = sorted({int(value) for value in args.local_search_budgets if int(value) > 0})
    randomized_budgets = sorted({int(value) for value in args.randomized_budgets if int(value) > 0})
    max_budget = max(budgets + local_budgets + randomized_budgets) if (budgets or local_budgets or randomized_budgets) else 0

    summary_rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    manifest = {
        "fresh_data_only": True,
        "fleur_data_used": False,
        "distance_model": distance_model,
        "outputs_dir": str(args.outputs_dir),
        "run_tag_marker": args.run_tag_marker,
        "candidate_grid": str(args.candidate_grid),
        "grid_spacing_m": float(args.grid_spacing_m),
        "thresholds_km": [float(value) for value in args.thresholds_km],
        "budgets": budgets,
        "local_search_budgets": local_budgets,
        "randomized_budgets": randomized_budgets,
        "randomized_repeats": int(args.randomized_repeats),
        "rcl_size": int(args.rcl_size),
        "local_max_moves": int(args.local_max_moves),
        "local_max_candidates_per_drop": int(args.local_max_candidates_per_drop),
        "elite_candidates": int(args.elite_candidates),
        "weight_scale": float(args.weight_scale),
        "provenance": provenance,
    }

    for threshold_km in args.thresholds_km:
        threshold_m = float(threshold_km) * 1000.0
        instance_name = (
            f"vietnam_grid{float(args.grid_spacing_m) / 1000.0:g}km_"
            f"straightline_{float(threshold_km):g}km"
        )
        print(f"Running {instance_name} with {len(candidates.ids):,} candidates")
        spatial = SpatialMaxCover(
            demand=demand,
            candidates=candidates,
            existing=existing,
            threshold_m=threshold_m,
            weight_scale=float(args.weight_scale),
            cache_size=int(args.cache_size),
            chunk_size=int(args.chunk_size),
        )
        initial = spatial.compute_initial_gains()
        positive_initial = initial[initial > 0]
        if positive_initial.size:
            elite_k = min(int(args.elite_candidates), int(positive_initial.size))
            elite = np.argpartition(initial, -elite_k)[-elite_k:]
            elite = elite[initial[elite] > 0]
        else:
            elite = np.empty(0, dtype=np.int32)

        greedy = spatial.construct(
            budget=max_budget,
            randomized=False,
            rcl_size=int(args.rcl_size),
            seed=int(args.seed),
        )
        for budget in budgets:
            prefix = spatial.prefix_result(greedy, budget)
            idx = min(int(budget), len(greedy.times) - 1)
            summary_rows.append(
                summary_row(
                    instance_name=instance_name,
                    marker=args.run_tag_marker,
                    grid_spacing_m=float(args.grid_spacing_m),
                    distance_model=distance_model,
                    spatial=spatial,
                    method="spatial_greedy",
                    budget=budget,
                    objective=prefix.objective,
                    seconds=float(greedy.times[idx]) if idx < len(greedy.times) else greedy.total_time,
                )
            )
            selected_frames.append(
                selected_frame(
                    candidates=candidates,
                    instance_name=instance_name,
                    solution=prefix.solution,
                    method="spatial_greedy",
                    budget=budget,
                    threshold_km=spatial.threshold_km,
                )
            )

        for budget in local_budgets:
            base = spatial.prefix_result(greedy, budget)
            improved = spatial.improve_local(
                base,
                max_moves=int(args.local_max_moves),
                max_candidates_per_drop=int(args.local_max_candidates_per_drop),
                elite_candidates=elite.astype(np.int32, copy=False),
            )
            summary_rows.append(
                summary_row(
                    instance_name=instance_name,
                    marker=args.run_tag_marker,
                    grid_spacing_m=float(args.grid_spacing_m),
                    distance_model=distance_model,
                    spatial=spatial,
                    method="spatial_greedy_local",
                    budget=budget,
                    objective=improved.objective,
                    seconds=float(base.total_time + improved.total_time),
                    construction_objective=base.objective,
                    construction_seconds=base.total_time,
                    local_search_moves=improved.moves,
                )
            )
            selected_frames.append(
                selected_frame(
                    candidates=candidates,
                    instance_name=instance_name,
                    solution=improved.solution,
                    method="spatial_greedy_local",
                    budget=budget,
                    threshold_km=spatial.threshold_km,
                )
            )

        for budget in randomized_budgets:
            best_random: SpatialResult | None = None
            best_base: SpatialResult | None = None
            best_repeat = -1
            for repeat in range(int(args.randomized_repeats)):
                run_seed = int(args.seed) + 1000 * repeat + int(round(threshold_km * 10))
                constructed = spatial.construct(
                    budget=budget,
                    randomized=True,
                    rcl_size=int(args.rcl_size),
                    seed=run_seed,
                )
                improved = spatial.improve_local(
                    constructed,
                    max_moves=int(args.local_max_moves),
                    max_candidates_per_drop=int(args.local_max_candidates_per_drop),
                    elite_candidates=elite.astype(np.int32, copy=False),
                )
                total_seconds = constructed.total_time + improved.total_time
                summary_rows.append(
                    summary_row(
                        instance_name=instance_name,
                        marker=args.run_tag_marker,
                        grid_spacing_m=float(args.grid_spacing_m),
                        distance_model=distance_model,
                        spatial=spatial,
                        method="spatial_grasp_local",
                        budget=budget,
                        objective=improved.objective,
                        seconds=float(total_seconds),
                        construction_objective=constructed.objective,
                        construction_seconds=constructed.total_time,
                        local_search_moves=improved.moves,
                        seed=run_seed,
                        repeat=repeat,
                    )
                )
                selected_frames.append(
                    selected_frame(
                        candidates=candidates,
                        instance_name=instance_name,
                        solution=improved.solution,
                        method="spatial_grasp_local",
                        budget=budget,
                        threshold_km=spatial.threshold_km,
                        seed=run_seed,
                        repeat=repeat,
                    )
                )
                if best_random is None or improved.objective > best_random.objective:
                    best_random = improved
                    best_base = constructed
                    best_repeat = repeat
            if best_random is not None and best_base is not None:
                selected_frames.append(
                    selected_frame(
                        candidates=candidates,
                        instance_name=instance_name,
                        solution=best_random.solution,
                        method="spatial_grasp_best",
                        budget=budget,
                        threshold_km=spatial.threshold_km,
                        seed=int(args.seed),
                        repeat=best_repeat,
                    )
                )
                summary_rows.append(
                    summary_row(
                        instance_name=instance_name,
                        marker=args.run_tag_marker,
                        grid_spacing_m=float(args.grid_spacing_m),
                        distance_model=distance_model,
                        spatial=spatial,
                        method="spatial_grasp_best",
                        budget=budget,
                        objective=best_random.objective,
                        seconds=float(best_base.total_time + best_random.total_time),
                        construction_objective=best_base.objective,
                        construction_seconds=best_base.total_time,
                        local_search_moves=best_random.moves,
                        seed=int(args.seed),
                        repeat=best_repeat,
                    )
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["grid_spacing_m", "threshold_km", "budget", "method", "repeat"],
        na_position="last",
    )
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary.to_csv(args.output_dir / "coverage_summary_by_budget.csv", index=False)
    selected.to_csv(args.output_dir / "selected_candidates.csv", index=False)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_figures and not summary.empty and not selected.empty:
        write_solution_figures(
            output_dir=args.output_dir,
            demand=demand,
            existing=existing,
            summary=summary,
            selected=selected,
            budgets=budgets,
            population_sample=int(args.population_sample),
            seed=int(args.seed),
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_rows": int(len(summary)),
                "selected_rows": int(len(selected)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
