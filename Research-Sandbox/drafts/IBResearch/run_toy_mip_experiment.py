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
    print(f"Saved MILP timetable to {args.mip_output}")
    print(f"Saved MILP progress to {args.progress_output}")
    print(f"Saved MILP log to {args.log_output}")
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
