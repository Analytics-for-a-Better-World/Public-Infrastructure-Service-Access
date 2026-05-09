from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.full_heuristic import solve_full_heuristic, validate_full_solution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full IB timetable heuristic.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    exams = pd.read_csv(args.data_dir / "M24 exam names and block lengths.csv")
    days = pd.read_csv(args.data_dir / "exam_days3.csv")
    pairs = pd.read_csv(args.data_dir / "Exam Pairs ABW-2.csv")

    result = solve_full_heuristic(
        exams=exams,
        days=days,
        pairs=pairs,
        max_rounds=args.max_rounds,
    )
    checks = validate_full_solution(result.timetable, result.data)

    print("Objective:", result.objective_value)
    print("Same-slot clashes:", checks["same_slot_clashes"])
    print("Rows:", len(result.timetable))
    print(result.timetable.to_string(index=False))

    if args.output is not None:
        result.timetable.to_csv(args.output, index=False)
        print(f"Saved timetable to {args.output}")


if __name__ == "__main__":
    main()
