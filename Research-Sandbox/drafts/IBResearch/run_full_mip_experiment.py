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
    load_default_data,
    mip_objective_value,
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
    print(f"Saved timetable to {args.output}")
    print(f"Saved progress to {args.progress_output}")
    print(f"Saved log to {args.log_output}")
    print(f"Saved bound plot to {args.plot_output}")


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
