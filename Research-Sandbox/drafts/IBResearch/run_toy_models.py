from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.anthony_model import mip_objective_value, solve_toy_mip
from src.toy_heuristic import solve_toy_heuristic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy IB heuristic and MIP models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--mip-time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--skip-mip", action="store_true")
    parser.add_argument("--heuristic-output", type=Path, default=None)
    parser.add_argument("--mip-output", type=Path, default=None)
    args = parser.parse_args()

    toy_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    toy_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    toy_days = pd.read_csv(args.data_dir / "exam_days3.csv")

    heuristic = solve_toy_heuristic(toy_exams, toy_pairs, toy_days=toy_days, year=args.year)
    print("Heuristic objective:", heuristic.objective_value)
    print(heuristic.timetable.to_string(index=False))
    if args.heuristic_output is not None:
        heuristic.timetable.to_csv(args.heuristic_output, index=False)

    if args.skip_mip:
        return

    mip_model, mip_timetable = solve_toy_mip(
        toy_exams,
        toy_pairs,
        toy_days=toy_days,
        year=args.year,
        time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        mip_start_timetable=heuristic.timetable,
        output_flag=1,
    )
    mip_objective = mip_objective_value(mip_timetable, heuristic.pairs, heuristic.days, mode="anthony_appendix")
    print("\nMIP status:", mip_model.model.Status)
    print("MIP model objective:", mip_model.model.ObjVal if mip_model.model.SolCount else None)
    print("MIP evaluated objective:", mip_objective if mip_model.model.SolCount else None)
    if mip_model.model.SolCount:
        print(mip_timetable.to_string(index=False))
    if args.mip_output is not None and mip_model.model.SolCount:
        mip_timetable.to_csv(args.mip_output, index=False)


if __name__ == "__main__":
    main()
