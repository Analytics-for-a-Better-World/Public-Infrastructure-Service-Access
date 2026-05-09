from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.spread_improvement import spread_with_secondary_mip


def main() -> None:
    parser = argparse.ArgumentParser(description="Spread a full IB timetable with a secondary MILP objective.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("full_spread_timetable.csv"))
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--allowed-objective-increase", type=float, default=0.0)
    parser.add_argument("--target-day-load", type=int, default=5)
    parser.add_argument("--target-slot-load", type=int, default=3)
    parser.add_argument("--nb-days", type=int, default=23)
    args = parser.parse_args()

    exams = pd.read_csv(args.data_dir / "M24 exam names and block lengths.csv")
    days = pd.read_csv(args.data_dir / "exam_days3.csv")
    pairs = pd.read_csv(args.data_dir / "Exam Pairs ABW-2.csv")
    start = pd.read_csv(args.start)

    result = spread_with_secondary_mip(
        exams=exams,
        days=days,
        pairs=pairs,
        start_timetable=start,
        time_limit=args.time_limit,
        allowed_objective_increase=args.allowed_objective_increase,
        target_day_load=args.target_day_load,
        target_slot_load=args.target_slot_load,
        nb_days=args.nb_days,
    )

    result.timetable.to_csv(args.output, index=False)
    print(f"Saved timetable to {args.output}")
    print("Solver status:", result.solver_status)
    print("Solver gap:", result.solver_gap)
    print("Before:", asdict(result.diagnostics_before))
    print("After:", asdict(result.diagnostics_after))

    if args.summary_output is not None:
        pd.DataFrame(
            [
                {"stage": "before", **asdict(result.diagnostics_before)},
                {"stage": "after", **asdict(result.diagnostics_after)},
            ]
        ).to_csv(args.summary_output, index=False)
        print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
