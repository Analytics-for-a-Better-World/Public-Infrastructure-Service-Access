from __future__ import annotations

import argparse
from itertools import combinations
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from src.anthony_model import (
    apply_timetable_start,
    build_anthony_mip_model,
    load_default_data,
    mip_objective_value,
    placement_from_timetable,
    timetable_from_solution,
)
from src.full_heuristic import validate_full_solution
from src.lns_improvement import _normalize_timetable, _repair_subject_exam_order


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a seeded full IB MILP with bound tracking.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("full_seeded_mip_timetable.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("full_seeded_mip_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("full_seeded_mip.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("full_seeded_mip_bounds.png"))
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--nb-days", type=int, default=34)
    parser.add_argument("--objective-mode", choices=["formal", "anthony_appendix"], default="formal")
    parser.add_argument("--enforce-subject-exam-order", action="store_true")
    parser.add_argument("--y-binary", action="store_true")
    parser.add_argument("--proximity-at-most-one", action="store_true")
    parser.add_argument("--symmetry", type=int, default=None)
    parser.add_argument("--mip-focus", type=int, default=None)
    parser.add_argument("--cuts", type=int, default=None)
    parser.add_argument("--presolve", type=int, default=None)
    parser.add_argument("--callback-slot-swap-heuristic", action="store_true")
    parser.add_argument("--callback-heuristic-min-interval", type=float, default=60.0)
    parser.add_argument("--callback-heuristic-max-swaps", type=int, default=4)
    parser.add_argument("--dense-cluster-cuts", type=int, default=0)
    parser.add_argument("--dense-cluster-size", type=int, default=12)
    parser.add_argument("--dense-cluster-overlap", action="store_true")
    parser.add_argument("--dense-cluster-min-new-pair-share", type=float, default=0.20)
    parser.add_argument("--dense-cluster-time-limit", type=float, default=60.0)
    parser.add_argument("--dense-cluster-summary", type=Path, default=None)
    args = parser.parse_args()

    exams, days, pairs = load_default_data(args.data_dir)
    start = _normalize_timetable(pd.read_csv(args.start))

    built = build_anthony_mip_model(
        exams=exams,
        days=days,
        pairs=pairs,
        nb_days=args.nb_days,
        max_clashes=10_000,
        max_afternoon_minutes=180,
        max_daily_minutes=385,
        forbid_weekends=True,
        forbid_may_first=True,
        forbid_language_fridays=True,
        force_sbs_start=True,
        consecutive_subject_exams=True,
        consecutive_usable_subject_exams=False,
        first_half_subjects={"BUSINESS MANAGEMENT", "HISTORY", "ENGLISH A LAL"},
        recode_math_paper_three=True,
        y_binary=args.y_binary,
        proximity_at_most_one=args.proximity_at_most_one,
        enforce_subject_exam_order=args.enforce_subject_exam_order,
        objective_mode=args.objective_mode,
        output_flag=1,
        model_name="full_seeded_mip",
    )
    dense_cut_rows: list[dict[str, Any]] = []
    if args.dense_cluster_cuts:
        dense_cut_rows = _add_dense_cluster_cuts(
            built=built,
            exams=exams,
            days=days,
            pairs=pairs,
            nb_days=args.nb_days,
            count=args.dense_cluster_cuts,
            size=args.dense_cluster_size,
            allow_overlap=args.dense_cluster_overlap,
            min_new_pair_share=args.dense_cluster_min_new_pair_share,
            subproblem_time_limit=args.dense_cluster_time_limit,
            objective_mode=args.objective_mode,
            enforce_subject_exam_order=args.enforce_subject_exam_order,
        )
        if args.dense_cluster_summary is not None:
            pd.DataFrame(dense_cut_rows).to_csv(args.dense_cluster_summary, index=False)

    if args.enforce_subject_exam_order:
        start = _repair_subject_exam_order(start, built.data)

    validate_full_solution(
        start,
        built.data,
        max_clashes=10_000,
        max_afternoon_minutes=180,
        max_daily_minutes=385,
        first_half_subjects={"BUSINESS MANAGEMENT", "HISTORY", "ENGLISH A LAL"},
    )
    start_objective = mip_objective_value(start, built.data.pairs, built.data.days, mode=args.objective_mode)
    print("Start objective:", start_objective)

    apply_timetable_start(built, start)
    built.model.setParam("TimeLimit", args.time_limit)
    built.model.setParam("LogFile", str(args.log_output))
    if args.symmetry is not None:
        built.model.setParam("Symmetry", args.symmetry)
    if args.mip_focus is not None:
        built.model.setParam("MIPFocus", args.mip_focus)
    if args.cuts is not None:
        built.model.setParam("Cuts", args.cuts)
    if args.presolve is not None:
        built.model.setParam("Presolve", args.presolve)

    progress: list[dict[str, float]] = []
    heuristic_stats = {
        "attempts": 0,
        "accepted": 0,
        "best_injected": float("inf"),
        "last_time": -float("inf"),
    }

    def callback(model, where):
        import gurobipy as gp

        if where == gp.GRB.Callback.MIP:
            runtime = float(model.cbGet(gp.GRB.Callback.RUNTIME))
            incumbent = float(model.cbGet(gp.GRB.Callback.MIP_OBJBST))
            bound = float(model.cbGet(gp.GRB.Callback.MIP_OBJBND))
            if abs(incumbent) >= 1e100:
                gap = float("nan")
            else:
                gap = abs(incumbent - bound) / max(1.0, abs(incumbent))
            progress.append(
                {
                    "time_seconds": runtime,
                    "incumbent": incumbent,
                    "best_bound": bound,
                    "gap": gap,
                }
            )
        elif args.callback_slot_swap_heuristic and where == gp.GRB.Callback.MIPSOL:
            runtime = float(model.cbGet(gp.GRB.Callback.RUNTIME))
            if runtime - heuristic_stats["last_time"] < args.callback_heuristic_min_interval:
                return
            heuristic_stats["last_time"] = runtime
            heuristic_stats["attempts"] += 1
            incumbent = _timetable_from_callback_solution(built=built, model=model)
            if incumbent is None:
                return
            try:
                candidate, candidate_objective, start_objective_cb = _slot_swap_improvement(
                    incumbent,
                    built.data,
                    objective_mode=args.objective_mode,
                    max_swaps=args.callback_heuristic_max_swaps,
                )
            except Exception:
                return
            if candidate_objective + 1e-6 >= start_objective_cb:
                return
            if candidate_objective + 1e-6 >= heuristic_stats["best_injected"]:
                return
            _inject_timetable_solution(built=built, model=model, timetable=candidate)
            heuristic_stats["best_injected"] = candidate_objective
            heuristic_stats["accepted"] += 1

    t0 = time.perf_counter()
    built.model.optimize(callback)
    solve_seconds = time.perf_counter() - t0

    if built.model.SolCount:
        timetable = timetable_from_solution(built)
        objective = mip_objective_value(timetable, built.data.pairs, built.data.days, mode=args.objective_mode)
        timetable.to_csv(args.output, index=False)
    else:
        objective = None

    progress_df = _deduplicate_progress(pd.DataFrame(progress))
    progress_df.to_csv(args.progress_output, index=False)
    _plot_progress(progress_df, args.plot_output)

    print("MILP seconds:", round(solve_seconds, 6))
    print("MILP status:", int(built.model.Status))
    print("MILP incumbent:", objective)
    print("MILP best bound:", float(built.model.ObjBound) if built.model.SolCount else None)
    print("MILP gap:", float(built.model.MIPGap) if built.model.SolCount else None)
    if args.callback_slot_swap_heuristic:
        print("Callback heuristic attempts:", heuristic_stats["attempts"])
        print("Callback heuristic accepted:", heuristic_stats["accepted"])
        print("Callback heuristic best injected:", heuristic_stats["best_injected"])
    if dense_cut_rows:
        print("Dense cluster cuts:", len(dense_cut_rows))
        print(pd.DataFrame(dense_cut_rows).to_string(index=False))
    print(f"Saved timetable to {args.output}")
    print(f"Saved progress to {args.progress_output}")
    print(f"Saved log to {args.log_output}")
    print(f"Saved bound plot to {args.plot_output}")


def _add_dense_cluster_cuts(
    *,
    built: Any,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    nb_days: int,
    count: int,
    size: int,
    allow_overlap: bool,
    min_new_pair_share: float,
    subproblem_time_limit: float,
    objective_mode: str,
    enforce_subject_exam_order: bool,
) -> list[dict[str, Any]]:
    import gurobipy as gp

    clusters = _select_dense_clusters(
        built.data.pairs,
        count=count,
        size=size,
        allow_overlap=allow_overlap,
        min_new_pair_share=min_new_pair_share,
    )
    rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        sub_exams = exams.copy()
        if "Full Name" not in sub_exams.columns and "FULL_NAME" in sub_exams.columns:
            sub_exams = sub_exams.rename(columns={"FULL_NAME": "Full Name"})
        sub_exams = sub_exams[sub_exams["Full Name"].astype(str).str.strip().isin(cluster)].copy()
        sub_pairs = pairs.copy()
        if sub_pairs.columns[0] not in cluster:
            sub_pairs = sub_pairs.set_index(sub_pairs.columns[0])
        sub_pairs.index = sub_pairs.index.astype(str).str.strip()
        sub_pairs.columns = sub_pairs.columns.astype(str).str.strip()
        sub_pairs = sub_pairs.loc[cluster, cluster].reset_index()

        sub_built = build_anthony_mip_model(
            exams=sub_exams,
            days=days,
            pairs=sub_pairs,
            nb_days=nb_days,
            max_clashes=None,
            max_afternoon_minutes=180,
            max_daily_minutes=385,
            forbid_weekends=True,
            forbid_may_first=True,
            forbid_language_fridays=True,
            force_sbs_start=False,
            consecutive_subject_exams=True,
            consecutive_usable_subject_exams=False,
            first_half_subjects={"BUSINESS MANAGEMENT", "HISTORY", "ENGLISH A LAL"},
            recode_math_paper_three=True,
            enforce_subject_exam_order=enforce_subject_exam_order,
            objective_mode=objective_mode,
            output_flag=0,
            model_name=f"dense_cluster_lb_{cluster_index}",
        )
        sub_built.model.setParam("TimeLimit", subproblem_time_limit)
        sub_built.model.optimize()
        lower_bound = max(0.0, float(sub_built.model.ObjBound))
        lhs = gp.LinExpr()
        model_order = {exam: idx for idx, exam in enumerate(built.data.exam_names)}
        ordered_cluster = sorted(cluster, key=model_order.__getitem__)
        for pos, exam_i in enumerate(ordered_cluster):
            for exam_j in ordered_cluster[pos + 1 :]:
                cij = float(built.data.pairs.loc[exam_i, exam_j])
                if cij == 0:
                    continue
                for category, weight in {
                    "a": 64,
                    "b": 32,
                    "c": 16,
                    "d": 8,
                    "e": 4,
                    "f": 2,
                    "g": 1,
                }.items():
                    lhs.addTerms(cij * float(weight), built.y[(exam_i, exam_j, category)])
        if lower_bound > 1e-6:
            built.model.addConstr(lhs >= lower_bound - 1e-6, name=f"dense_cluster_lb[{cluster_index}]")
        rows.append(
            {
                "cluster": cluster_index,
                "size": len(cluster),
                "lower_bound": lower_bound,
                "pair_mass": _cluster_pair_mass(built.data.pairs, cluster),
                "sub_status": int(sub_built.model.Status),
                "sub_gap": float(sub_built.model.MIPGap) if sub_built.model.SolCount else None,
                "exams": ";".join(cluster),
            }
        )
    built.model.update()
    return rows


def _select_dense_clusters(
    pairs: pd.DataFrame,
    *,
    count: int,
    size: int,
    allow_overlap: bool,
    min_new_pair_share: float,
) -> list[list[str]]:
    matrix = pairs.copy().apply(pd.to_numeric, errors="raise")
    clusters: list[list[str]] = []
    total_scores = matrix.sum(axis=1).sort_values(ascending=False)
    covered_pairs: set[tuple[str, str]] = set()
    remaining = set(matrix.index.astype(str))
    for seed in total_scores.index:
        seed = str(seed)
        if not allow_overlap and seed not in remaining:
            continue
        cluster = [seed]
        if not allow_overlap:
            remaining.remove(seed)
        candidate_pool = set(matrix.index.astype(str))
        candidate_pool.discard(seed)
        while len(cluster) < size and candidate_pool:
            next_exam = max(
                candidate_pool,
                key=lambda exam: float(matrix.loc[exam, cluster].sum()),
            )
            cluster.append(next_exam)
            candidate_pool.remove(next_exam)
            if not allow_overlap and next_exam in remaining:
                remaining.remove(next_exam)
        if allow_overlap:
            cluster_pairs = _positive_cluster_pairs(matrix, cluster)
            if clusters:
                new_pairs = cluster_pairs - covered_pairs
                new_mass = _pair_set_mass(matrix, new_pairs)
                total_mass = max(1.0, _pair_set_mass(matrix, cluster_pairs))
                if new_mass / total_mass < min_new_pair_share:
                    continue
            covered_pairs.update(cluster_pairs)
        clusters.append(cluster)
        if len(clusters) >= count:
            break
    return clusters


def _positive_cluster_pairs(matrix: pd.DataFrame, cluster: list[str]) -> set[tuple[str, str]]:
    pair_set: set[tuple[str, str]] = set()
    for exam_i, exam_j in combinations(cluster, 2):
        if float(matrix.loc[exam_i, exam_j]) > 0:
            pair_set.add(tuple(sorted((exam_i, exam_j))))
    return pair_set


def _pair_set_mass(matrix: pd.DataFrame, pair_set: set[tuple[str, str]]) -> float:
    return float(sum(float(matrix.loc[exam_i, exam_j]) for exam_i, exam_j in pair_set))


def _cluster_pair_mass(matrix: pd.DataFrame, cluster: list[str]) -> float:
    return _pair_set_mass(matrix, _positive_cluster_pairs(matrix, cluster))


def _timetable_from_callback_solution(*, built: Any, model: Any) -> pd.DataFrame | None:
    import gurobipy as gp

    records: list[dict[str, Any]] = []
    day_lookup = built.data.days.set_index("Date")["DOW"].to_dict()
    for (exam, slot, date), var in built.x.items():
        value = model.cbGetSolution(var)
        if value > 0.5:
            records.append(
                {
                    "Day_of_Week": day_lookup.get(date),
                    "Date": date,
                    "Slot": slot,
                    "Exam_Name": exam,
                }
            )
    if len(records) != len(built.data.exam_names):
        return None
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["Date", "Slot", "Exam_Name"])
        .reset_index(drop=True)
    )


def _slot_swap_improvement(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
    max_swaps: int,
) -> tuple[pd.DataFrame, float, float]:
    current = _normalize_timetable(timetable)
    validate_full_solution(current, data)
    current_objective = mip_objective_value(current, data.pairs, data.days, mode=objective_mode)
    start_objective = current_objective

    for _ in range(max_swaps):
        best_candidate: pd.DataFrame | None = None
        best_objective = current_objective
        for date, day_df in current.groupby("Date", sort=True):
            am_exams = day_df.loc[day_df["Slot"] == "AM", "Exam_Name"].tolist()
            pm_exams = day_df.loc[day_df["Slot"] == "PM", "Exam_Name"].tolist()
            for left in am_exams:
                for right in pm_exams:
                    trial = current.copy()
                    trial.loc[trial["Exam_Name"] == left, "Slot"] = "PM"
                    trial.loc[trial["Exam_Name"] == right, "Slot"] = "AM"
                    try:
                        validate_full_solution(trial, data)
                    except ValueError:
                        continue
                    objective = mip_objective_value(trial, data.pairs, data.days, mode=objective_mode)
                    if objective + 1e-6 < best_objective:
                        best_candidate = trial
                        best_objective = objective
        if best_candidate is None:
            break
        current = _normalize_timetable(best_candidate)
        current_objective = best_objective
    return current, current_objective, start_objective


def _inject_timetable_solution(*, built: Any, model: Any, timetable: pd.DataFrame) -> None:
    placement = placement_from_timetable(timetable)
    variables = []
    values = []
    for key, var in built.x.items():
        exam, slot, date = key
        variables.append(var)
        values.append(1.0 if placement.get(exam) == (pd.Timestamp(date).normalize(), slot) else 0.0)
    model.cbSetSolution(variables, values)


def _deduplicate_progress(progress: pd.DataFrame) -> pd.DataFrame:
    if progress.empty:
        return progress
    progress = progress.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["incumbent", "best_bound"])
    progress["time_seconds"] = progress["time_seconds"].round(3)
    return progress.drop_duplicates(subset=["time_seconds", "incumbent", "best_bound"]).reset_index(drop=True)


def _plot_progress(progress: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Best bound", color="#1f77b4", linewidth=3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title("Full IB seeded MILP incumbent and bound evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _format_clock_time(seconds: float, _pos: int | None = None) -> str:
    if pd.isna(seconds):
        return ""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    main()
