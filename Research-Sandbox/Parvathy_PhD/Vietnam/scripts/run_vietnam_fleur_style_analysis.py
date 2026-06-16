from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
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

OUTPUT_ROOT = Path(r"C:\local\Parvathy\Vietnam")


@dataclass(slots=True)
class LoadedInstance:
    path: Path
    instance: mch.MaxCoverInstance
    metadata: dict
    candidate_source_ids: np.ndarray
    candidate_longitude: np.ndarray
    candidate_latitude: np.ndarray

    @property
    def scale(self) -> float:
        return float(self.metadata.get("weight_scale", 1.0))

    @property
    def baseline(self) -> float:
        return float(self.metadata.get("baseline_covered_population", 0.0))

    @property
    def total_population(self) -> float:
        return float(self.metadata.get("total_population", 0.0))

    @property
    def threshold_km(self) -> float:
        return float(self.metadata.get("threshold_m", 0.0)) / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Fleur-style analysis tables on fresh PISA Vietnam max-cover instances. "
            "The script does not read Fleur npy/pickle data."
        )
    )
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="*",
        default=sorted((OUTPUT_ROOT / "optimization").glob("vietnam_10kmgrid_*km_threshold.npz")),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 60, 80, 100, 200])
    parser.add_argument(
        "--local-search-budgets",
        type=int,
        nargs="*",
        help="Budgets for greedy plus first-swap local search. Defaults to all budgets.",
    )
    parser.add_argument(
        "--randomized-budgets",
        type=int,
        nargs="*",
        default=[20],
        help="Budgets for repeated randomized GRASP checks. Use an empty value to skip.",
    )
    parser.add_argument("--randomized-repeats", type=int, default=3)
    parser.add_argument("--grasp-time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--grasp-max-iterations", type=int, default=5)
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--local-search", choices=["first", "first_sparse", "none"], default="first_sparse")
    parser.add_argument("--path-relinking-method", choices=["fast", "original"], default="fast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "fleur_style_analysis")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_instance(path: Path) -> LoadedInstance:
    data = np.load(path, allow_pickle=False)
    instance = mch.MaxCoverInstance(
        w=data["w"],
        ij_indptr=data["ij_indptr"],
        ij_indices=data["ij_indices"],
        ji_indptr=data["ji_indptr"],
        ji_indices=data["ji_indices"],
    )
    metadata = json.loads(str(data["metadata_json"]))
    return LoadedInstance(
        path=path,
        instance=instance,
        metadata=metadata,
        candidate_source_ids=data["candidate_source_ids"].astype(str),
        candidate_longitude=data["candidate_longitude"].astype(float),
        candidate_latitude=data["candidate_latitude"].astype(float),
    )


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) >= 0})


def objective_to_population(loaded: LoadedInstance, objective: int | float) -> float:
    return float(objective) / loaded.scale


def summary_row(
    loaded: LoadedInstance,
    *,
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
    incremental = objective_to_population(loaded, objective)
    total = loaded.baseline + incremental
    total_population = loaded.total_population
    return {
        "instance": loaded.path.name,
        "run_tag_marker": loaded.metadata.get("run_tag_marker", ""),
        "threshold_km": loaded.threshold_km,
        "n_candidates": int(loaded.instance.n_facilities),
        "budget": int(budget),
        "method": method,
        "seed": seed,
        "repeat": repeat,
        "status": status,
        "construction_incremental_population": (
            objective_to_population(loaded, construction_objective)
            if construction_objective is not None
            else np.nan
        ),
        "incremental_population": incremental,
        "baseline_covered_population": loaded.baseline,
        "total_covered_population": total,
        "coverage_percent_total_population": 100.0 * total / total_population if total_population else np.nan,
        "incremental_percent_total_population": 100.0 * incremental / total_population if total_population else np.nan,
        "available_incremental_population": float(loaded.instance.w.sum() / loaded.scale),
        "objective_weight_units": int(objective),
        "construction_seconds": construction_seconds,
        "seconds": float(seconds),
        "local_search_moves": local_search_moves,
    }


def selected_candidates_frame(
    loaded: LoadedInstance,
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
                "instance": loaded.path.name,
                "rank": rank,
                "facility_index": facility_i,
                "source_id": str(loaded.candidate_source_ids[facility_i])
                if facility_i < len(loaded.candidate_source_ids)
                else "",
                "longitude": float(loaded.candidate_longitude[facility_i])
                if facility_i < len(loaded.candidate_longitude)
                else np.nan,
                "latitude": float(loaded.candidate_latitude[facility_i])
                if facility_i < len(loaded.candidate_latitude)
                else np.nan,
                "method": method,
                "budget": int(budget),
                "threshold_km": loaded.threshold_km,
                "seed": seed,
                "repeat": repeat,
            }
        )
    return pd.DataFrame(rows)


def run_greedy_curve(loaded: LoadedInstance, budgets: list[int], outdir: Path) -> tuple[list[dict], list[pd.DataFrame]]:
    max_budget = max(budgets) if budgets else 0
    result = budgeted_construct(loaded.instance, max_budget, constructor="greedy")
    rows: list[dict] = []
    traces = []
    previous = 0
    trace_rows = []
    for step, objective in enumerate(result.objectives):
        incremental = objective_to_population(loaded, objective)
        trace_rows.append(
            {
                "instance": loaded.path.name,
                "threshold_km": loaded.threshold_km,
                "step": step,
                "incremental_population": incremental,
                "marginal_incremental_population": objective_to_population(loaded, objective - previous),
                "total_covered_population": loaded.baseline + incremental,
                "seconds": result.times[step] if step < len(result.times) else np.nan,
            }
        )
        previous = int(objective)
    trace = pd.DataFrame(trace_rows)
    trace.to_csv(outdir / f"{loaded.path.stem}_greedy_marginal_trace.csv", index=False)
    traces.append(trace)

    for budget in budgets:
        idx = min(int(budget), len(result.objectives) - 1)
        rows.append(
            summary_row(
                loaded,
                method="greedy_construction",
                budget=budget,
                objective=int(result.objectives[idx]),
                seconds=float(result.times[idx]) if idx < len(result.times) else float(result.total_time),
            )
        )
    return rows, traces


def run_local_search_sweep(
    loaded: LoadedInstance,
    budgets: list[int],
    outdir: Path,
    *,
    local_search: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    for budget in budgets:
        constructed = budgeted_construct(loaded.instance, budget, constructor="greedy")
        improved = improve_local_search(loaded.instance, constructed, local_search=local_search)
        method = f"greedy_{local_search}"
        trace = pd.DataFrame(
            {
                "instance": loaded.path.name,
                "threshold_km": loaded.threshold_km,
                "budget": int(budget),
                "move": range(len(improved.objectives)),
                "incremental_population": [
                    objective_to_population(loaded, objective) for objective in improved.objectives
                ],
                "total_covered_population": [
                    loaded.baseline + objective_to_population(loaded, objective)
                    for objective in improved.objectives
                ],
                "seconds": improved.times,
            }
        )
        trace.to_csv(outdir / f"{loaded.path.stem}_p{budget}_{local_search}_trace.csv", index=False)
        rows.append(
            summary_row(
                loaded,
                method=method,
                budget=budget,
                objective=improved.objective,
                seconds=constructed.total_time + improved.total_time,
                construction_objective=constructed.objective,
                construction_seconds=constructed.total_time,
                local_search_moves=max(0, len(improved.objectives) - 1),
            )
        )
        selected_frames.append(selected_candidates_frame(loaded, improved.solution, method=method, budget=budget))
    return rows, selected_frames


def run_randomized_checks(
    loaded: LoadedInstance,
    budgets: list[int],
    *,
    repeats: int,
    seed: int,
    time_limit_seconds: float,
    max_iterations: int,
    rcl_size: int,
    sample_size: int,
    local_search: str,
    path_relinking_method: str,
    outdir: Path,
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    if repeats <= 0:
        return rows, selected_frames
    for budget in budgets:
        for repeat in range(repeats):
            run_seed = int(seed + 1000 * round(loaded.threshold_km) + 100 * int(budget) + repeat)
            t0 = pc()
            best, records = run_grasp(
                loaded.instance,
                budget,
                time_limit_seconds=time_limit_seconds,
                max_iterations=max_iterations,
                constructor="randomized",
                rcl_size=rcl_size,
                sample_size=sample_size,
                local_search=local_search,
                path_relinking=True,
                path_relinking_method=path_relinking_method,
                seed=run_seed,
                max_pool=8,
            )
            seconds = pc() - t0
            trace = pd.DataFrame(
                [
                    {
                        "instance": loaded.path.name,
                        "threshold_km": loaded.threshold_km,
                        "budget": int(budget),
                        "seed": run_seed,
                        "repeat": repeat,
                        "iteration": record.iteration,
                        "construction_incremental_population": objective_to_population(
                            loaded, record.construction_objective
                        ),
                        "local_search_incremental_population": objective_to_population(
                            loaded, record.local_search_objective
                        ),
                        "path_relinking_incremental_population": objective_to_population(
                            loaded, record.path_relinking_objective
                        ),
                        "best_incremental_population": objective_to_population(loaded, record.best_objective),
                        "total_seconds": record.total_seconds,
                        "pool_size": record.pool_size,
                    }
                    for record in records
                ]
            )
            trace.to_csv(
                outdir / f"{loaded.path.stem}_p{budget}_randomized_{local_search}_{path_relinking_method}_repeat{repeat}_trace.csv",
                index=False,
            )
            rows.append(
                summary_row(
                    loaded,
                    method=f"randomized_grasp_{local_search}_{path_relinking_method}_path_relinking",
                    budget=budget,
                    objective=best.objective,
                    seconds=seconds,
                    seed=run_seed,
                    repeat=repeat,
                    local_search_moves=None,
                )
            )
            selected_frames.append(
                selected_candidates_frame(
                    loaded,
                    best.solution,
                    method=f"randomized_grasp_{local_search}_{path_relinking_method}_path_relinking",
                    budget=budget,
                    seed=run_seed,
                    repeat=repeat,
                )
            )
    return rows, selected_frames


def write_instance_stats(loaded: LoadedInstance) -> dict:
    total_population = loaded.total_population
    baseline = loaded.baseline
    incremental_available = float(loaded.instance.w.sum() / loaded.scale)
    return {
        "instance": loaded.path.name,
        "threshold_km": loaded.threshold_km,
        "n_population": int(loaded.instance.n_households),
        "n_candidates": int(loaded.instance.n_facilities),
        "distance_rows_retained": int(loaded.metadata.get("distance_rows_retained", 0)),
        "candidate_distance_rows_retained": int(loaded.metadata.get("candidate_distance_rows_retained", 0)),
        "baseline_covered_points": int(loaded.metadata.get("baseline_covered_points", 0)),
        "baseline_covered_population": baseline,
        "baseline_coverage_percent": 100.0 * baseline / total_population if total_population else np.nan,
        "incremental_population_available": incremental_available,
        "incremental_available_percent": 100.0 * incremental_available / total_population if total_population else np.nan,
        "total_population": total_population,
    }


def write_plots(summary: pd.DataFrame, greedy_traces: list[pd.DataFrame], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (outdir / "plot_status.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (threshold, method), group in summary.groupby(["threshold_km", "method"]):
            if group["repeat"].notna().any():
                group = group.groupby("budget", as_index=False)["total_covered_population"].max()
            else:
                group = group.sort_values("budget")
            ax.plot(
                group["budget"],
                group["total_covered_population"] / 1_000_000,
                marker="o",
                label=f"{threshold:g} km {method}",
            )
        ax.set_xlabel("New candidate facilities")
        ax.set_ylabel("Covered population (millions)")
        ax.set_title("Vietnam fresh-data coverage by budget")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / "coverage_by_budget.png", dpi=180)
        plt.close(fig)

    if greedy_traces:
        fig, ax = plt.subplots(figsize=(9, 5))
        for trace in greedy_traces:
            threshold = trace["threshold_km"].iloc[0]
            use = trace.loc[trace["step"] > 0]
            ax.plot(
                use["step"],
                use["marginal_incremental_population"] / 1_000_000,
                label=f"{threshold:g} km",
            )
        ax.set_xlabel("Greedy step")
        ax.set_ylabel("Marginal covered population (millions)")
        ax.set_title("Greedy marginal gains")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "greedy_marginal_gains.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = unique_sorted(args.budgets)
    local_budgets = unique_sorted(args.local_search_budgets if args.local_search_budgets is not None else budgets)
    randomized_budgets = unique_sorted(args.randomized_budgets)

    instance_stats = []
    summary_rows: list[dict] = []
    greedy_traces: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    manifest = {
        "fresh_data_only": True,
        "fleur_data_used": False,
        "instances": [str(path) for path in args.instances],
        "budgets": budgets,
        "local_search_budgets": local_budgets,
        "randomized_budgets": randomized_budgets,
        "randomized_repeats": int(args.randomized_repeats),
        "grasp_time_limit_seconds": float(args.grasp_time_limit_seconds),
        "grasp_max_iterations": int(args.grasp_max_iterations),
        "rcl_size": int(args.rcl_size),
        "sample_size": int(args.sample_size),
        "local_search": args.local_search,
        "path_relinking_method": args.path_relinking_method,
        "seed": int(args.seed),
    }

    for path in args.instances:
        loaded = load_instance(path)
        print(f"Analyzing {path.name} ({loaded.threshold_km:g} km)")
        instance_stats.append(write_instance_stats(loaded))
        greedy_rows, traces = run_greedy_curve(loaded, budgets, args.output_dir)
        summary_rows.extend(greedy_rows)
        greedy_traces.extend(traces)

        local_rows, local_selected = run_local_search_sweep(
            loaded,
            local_budgets,
            args.output_dir,
            local_search=args.local_search,
        )
        summary_rows.extend(local_rows)
        selected_frames.extend(local_selected)

        randomized_rows, randomized_selected = run_randomized_checks(
            loaded,
            randomized_budgets,
            repeats=args.randomized_repeats,
            seed=args.seed,
            time_limit_seconds=args.grasp_time_limit_seconds,
            max_iterations=args.grasp_max_iterations,
            rcl_size=args.rcl_size,
            sample_size=args.sample_size,
            local_search=args.local_search,
            path_relinking_method=args.path_relinking_method,
            outdir=args.output_dir,
        )
        summary_rows.extend(randomized_rows)
        selected_frames.extend(randomized_selected)

    stats = pd.DataFrame(instance_stats).sort_values("threshold_km")
    stats.to_csv(args.output_dir / "instance_statistics.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(["threshold_km", "budget", "method", "repeat"], na_position="last")
    summary.to_csv(args.output_dir / "coverage_summary_by_budget.csv", index=False)

    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True)
        selected.to_csv(args.output_dir / "selected_candidates.csv", index=False)

    if not args.skip_plots:
        write_plots(summary, greedy_traces, args.output_dir)

    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_rows": int(len(summary)),
                "instance_rows": int(len(stats)),
                "selected_candidate_rows": int(sum(len(frame) for frame in selected_frames)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
