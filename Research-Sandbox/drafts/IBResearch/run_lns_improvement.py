from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.full_heuristic import solve_full_heuristic
from src.lns_improvement import improve_with_lns


def main() -> None:
    parser = argparse.ArgumentParser(description="Improve a full IB timetable with MILP LNS.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("full_lns_timetable.csv"))
    parser.add_argument("--history-output", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--neighborhood-sizes", type=str, default=None)
    parser.add_argument("--strategy-cycle", type=str, default=None)
    parser.add_argument("--fix-mode-cycle", type=str, default=None)
    parser.add_argument("--nb-days", type=int, default=23)
    parser.add_argument("--solution-pool-size", type=int, default=5)
    parser.add_argument("--no-adaptive-time", action="store_true")
    parser.add_argument("--heuristic-rounds", type=int, default=2)
    parser.add_argument("--load-acceptance-tolerance", type=float, default=0.0)
    parser.add_argument("--max-spread-regression", type=float, default=None)
    parser.add_argument("--target-max-day-exams", type=int, default=7)
    parser.add_argument("--target-max-slot-exams", type=int, default=4)
    parser.add_argument("--enforce-subject-exam-order", action="store_true")
    parser.add_argument("--y-binary", action="store_true")
    parser.add_argument("--symmetry", type=int, default=None)
    args = parser.parse_args()

    exams = pd.read_csv(args.data_dir / "M24 exam names and block lengths.csv")
    days = pd.read_csv(args.data_dir / "exam_days3.csv")
    pairs = pd.read_csv(args.data_dir / "Exam Pairs ABW-2.csv")

    if args.start is None:
        start = solve_full_heuristic(
            exams=exams,
            days=days,
            pairs=pairs,
            max_rounds=args.heuristic_rounds,
        ).timetable
    else:
        start = pd.read_csv(args.start)

    result = improve_with_lns(
        exams=exams,
        days=days,
        pairs=pairs,
        start_timetable=start,
        iterations=args.iterations,
        subjects_per_iteration=args.subjects,
        time_limit_per_iteration=args.time_limit,
        neighborhood_sizes=(
            [int(item) for item in args.neighborhood_sizes.split(",")]
            if args.neighborhood_sizes
            else None
        ),
        strategy_cycle=(
            [item.strip() for item in args.strategy_cycle.split(",")]
            if args.strategy_cycle
            else None
        ),
        fix_mode_cycle=(
            [item.strip() for item in args.fix_mode_cycle.split(",")]
            if args.fix_mode_cycle
            else None
        ),
        nb_days=args.nb_days,
        solution_pool_size=args.solution_pool_size,
        adaptive_time=not args.no_adaptive_time,
        load_acceptance_tolerance=args.load_acceptance_tolerance,
        max_spread_regression=args.max_spread_regression,
        target_max_day_exams=args.target_max_day_exams,
        target_max_slot_exams=args.target_max_slot_exams,
        enforce_subject_exam_order=args.enforce_subject_exam_order,
        y_binary=args.y_binary,
        symmetry=args.symmetry,
    )

    print("Final objective:", result.objective_value)
    history_rows = []
    for item in result.history:
        improvement = (
            item.start_objective - item.candidate_objective
            if item.candidate_objective is not None
            else None
        )
        accepted_improvement = improvement if item.accepted else 0.0
        print(
            f"iteration={item.iteration} strategy={item.strategy} "
            f"size={item.neighborhood_size} fix={item.fix_mode} accepted={item.accepted} "
            f"candidate={item.candidate_objective} improvement={improvement} "
            f"spread={item.start_spread_score}->{item.candidate_spread_score} "
            f"status={item.solver_status} "
            f"gap={item.solver_gap} subjects={item.selected_subjects}"
        )
        history_rows.append(
            {
                "iteration": item.iteration,
                "strategy": item.strategy,
                "neighborhood_size": item.neighborhood_size,
                "fix_mode": item.fix_mode,
                "start_objective": item.start_objective,
                "candidate_objective": item.candidate_objective,
                "raw_improvement": improvement,
                "start_spread_score": item.start_spread_score,
                "candidate_spread_score": item.candidate_spread_score,
                "spread_improvement": (
                    item.start_spread_score - item.candidate_spread_score
                    if item.candidate_spread_score is not None
                    else None
                ),
                "accepted": item.accepted,
                "accepted_improvement": accepted_improvement,
                "solver_status": item.solver_status,
                "solver_gap": item.solver_gap,
                "selected_subjects": ";".join(item.selected_subjects),
            }
        )

    result.timetable.to_csv(args.output, index=False)
    print(f"Saved timetable to {args.output}")

    if args.history_output is not None:
        pd.DataFrame(history_rows).to_csv(args.history_output, index=False)
        print(f"Saved history to {args.history_output}")


if __name__ == "__main__":
    main()
