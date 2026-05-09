from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from src.anthony_model import (
    apply_timetable_start,
    build_anthony_mip_model,
    mip_objective_value,
    prepare_toy_inputs,
    timetable_from_solution,
)
from src.toy_heuristic import solve_toy_heuristic
from run_full_mip_experiment import _cluster_pair_mass, _select_dense_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy heuristic and seeded toy MILP with bound tracking.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--heuristic-output", type=Path, default=Path("toy_best_heuristic_timetable.csv"))
    parser.add_argument("--mip-output", type=Path, default=Path("toy_seeded_mip_timetable.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("toy_seeded_mip_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("toy_seeded_mip.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_seeded_mip_bounds.png"))
    parser.add_argument("--start-mode", choices=["heuristic", "none"], default="heuristic")
    parser.add_argument("--use-default-solver-settings", action="store_true")
    parser.add_argument("--y-binary", action="store_true")
    parser.add_argument("--strengthen-y-upper-bounds", action="store_true")
    parser.add_argument("--proximity-at-most-one", action="store_true")
    parser.add_argument("--enforce-subject-exam-order", action="store_true")
    parser.add_argument("--symmetry", type=int, default=None)
    parser.add_argument("--dense-cluster-cuts", type=int, default=0)
    parser.add_argument("--dense-cluster-size", type=int, default=8)
    parser.add_argument("--dense-cluster-overlap", action="store_true")
    parser.add_argument("--dense-cluster-min-new-pair-share", type=float, default=0.05)
    parser.add_argument("--dense-cluster-time-limit", type=float, default=5.0)
    parser.add_argument("--dense-cluster-summary", type=Path, default=None)
    args = parser.parse_args()

    toy_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    toy_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")

    t0 = time.perf_counter()
    heuristic = solve_toy_heuristic(
        toy_exams,
        toy_pairs,
        max_rounds=100,
        objective_mode="anthony_appendix",
    )
    heuristic_seconds = time.perf_counter() - t0
    heuristic.timetable.to_csv(args.heuristic_output, index=False)

    print("Heuristic seconds:", round(heuristic_seconds, 6))
    print("Heuristic objective:", heuristic.objective_value)
    print(f"Saved heuristic timetable to {args.heuristic_output}")

    exams, days, pairs = prepare_toy_inputs(toy_exams, toy_pairs)
    built = build_anthony_mip_model(
        exams=exams,
        days=days,
        pairs=pairs,
        nb_days=len(days),
        max_clashes=15,
        max_afternoon_minutes=180,
        max_daily_minutes=375,
        forbid_weekends=True,
        forbid_may_first=True,
        forbid_language_fridays=True,
        forbid_language_friday_afternoons=False,
        consecutive_subject_exams=True,
        consecutive_usable_subject_exams=False,
        first_half_subjects={"Finance", "Law and Ethics"},
        recode_math_paper_three=False,
        y_binary=args.y_binary,
        strengthen_y_upper_bounds=args.strengthen_y_upper_bounds,
        proximity_at_most_one=args.proximity_at_most_one,
        enforce_subject_exam_order=args.enforce_subject_exam_order,
        objective_mode="anthony_appendix",
        output_flag=1,
        model_name="toy_seeded_mip",
    )
    dense_cut_rows: list[dict[str, object]] = []
    if args.dense_cluster_cuts:
        dense_cut_rows = _add_dense_cluster_cuts(
            built=built,
            exams=exams,
            days=days,
            pairs=pairs,
            count=args.dense_cluster_cuts,
            size=args.dense_cluster_size,
            allow_overlap=args.dense_cluster_overlap,
            min_new_pair_share=args.dense_cluster_min_new_pair_share,
            subproblem_time_limit=args.dense_cluster_time_limit,
            enforce_subject_exam_order=args.enforce_subject_exam_order,
        )
        if args.dense_cluster_summary is not None:
            pd.DataFrame(dense_cut_rows).to_csv(args.dense_cluster_summary, index=False)
    if args.start_mode == "heuristic":
        apply_timetable_start(built, heuristic.timetable)
        print("MILP start mode: heuristic")
    else:
        print("MILP start mode: none")
    if not args.use_default_solver_settings:
        built.model.setParam("TimeLimit", args.time_limit)
    built.model.setParam("LogFile", str(args.log_output))
    if args.symmetry is not None and not args.use_default_solver_settings:
        built.model.setParam("Symmetry", args.symmetry)

    progress: list[dict[str, float]] = []

    def callback(model, where):
        import gurobipy as gp

        if where == gp.GRB.Callback.MIP:
            runtime = float(model.cbGet(gp.GRB.Callback.RUNTIME))
            incumbent = float(model.cbGet(gp.GRB.Callback.MIP_OBJBST))
            bound = float(model.cbGet(gp.GRB.Callback.MIP_OBJBND))
            if abs(incumbent) >= 1e100:
                gap = float("nan")
            elif abs(incumbent) <= 1e-9:
                gap = 0.0
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

    t1 = time.perf_counter()
    built.model.optimize(callback)
    mip_seconds = time.perf_counter() - t1

    mip_timetable = timetable_from_solution(built)
    mip_objective = mip_objective_value(mip_timetable, built.data.pairs, built.data.days, mode="anthony_appendix")
    mip_timetable.to_csv(args.mip_output, index=False)

    progress_df = _deduplicate_progress(pd.DataFrame(progress))
    progress_df.to_csv(args.progress_output, index=False)
    _plot_progress(progress_df, args.plot_output)

    print("MILP seconds:", round(mip_seconds, 6))
    print("MILP status:", int(built.model.Status))
    print("MILP objective:", mip_objective)
    print("MILP best bound:", float(built.model.ObjBound) if built.model.SolCount else None)
    print("MILP gap:", float(built.model.MIPGap) if built.model.SolCount else None)
    if dense_cut_rows:
        print("Dense cluster cuts:", len(dense_cut_rows))
        print(pd.DataFrame(dense_cut_rows).to_string(index=False))
    print(f"Saved MILP timetable to {args.mip_output}")
    print(f"Saved MILP progress to {args.progress_output}")
    print(f"Saved MILP log to {args.log_output}")
    print(f"Saved bound plot to {args.plot_output}")


def _add_dense_cluster_cuts(
    *,
    built,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    count: int,
    size: int,
    allow_overlap: bool,
    min_new_pair_share: float,
    subproblem_time_limit: float,
    enforce_subject_exam_order: bool,
) -> list[dict[str, object]]:
    import gurobipy as gp

    clusters = _select_dense_clusters(
        built.data.pairs,
        count=count,
        size=size,
        allow_overlap=allow_overlap,
        min_new_pair_share=min_new_pair_share,
    )
    rows: list[dict[str, object]] = []
    weights = {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1}
    for cluster_index, cluster in enumerate(clusters, start=1):
        sub_exams = exams[exams["Full Name"].astype(str).str.strip().isin(cluster)].copy()
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
            nb_days=len(days),
            max_clashes=None,
            max_afternoon_minutes=180,
            max_daily_minutes=375,
            forbid_weekends=True,
            forbid_may_first=True,
            forbid_language_fridays=True,
            forbid_language_friday_afternoons=False,
            consecutive_subject_exams=True,
            consecutive_usable_subject_exams=False,
            first_half_subjects={"Finance", "Law and Ethics"},
            recode_math_paper_three=False,
            enforce_subject_exam_order=enforce_subject_exam_order,
            objective_mode="anthony_appendix",
            output_flag=0,
            model_name=f"toy_dense_cluster_lb_{cluster_index}",
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
                for category, weight in weights.items():
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


def _deduplicate_progress(progress: pd.DataFrame) -> pd.DataFrame:
    if progress.empty:
        return progress
    progress = progress.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["incumbent", "best_bound"])
    progress["time_seconds"] = progress["time_seconds"].round(3)
    return progress.drop_duplicates(subset=["time_seconds", "incumbent", "best_bound"]).reset_index(drop=True)


def _plot_progress(progress: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Best bound", color="#1f77b4", linewidth=3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title("Toy MILP incumbent and bound evolution")
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
