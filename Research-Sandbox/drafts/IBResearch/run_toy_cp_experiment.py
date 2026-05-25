from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from ortools.sat.python import cp_model

from run_toy_pattern_model import (
    CrossCost,
    Pattern,
    _build_blocks,
    _clean_pairs,
    _cross_cost,
    _generate_all_patterns,
    _same_slot_clashes,
)
from src.anthony_model import (
    mip_objective_value,
    parse_day_series,
    placement_from_timetable,
    prepare_toy_inputs,
    timetable_from_placement,
)
from src.toy_heuristic import solve_toy_heuristic


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the aligned TOY instance with OR-Tools CP-SAT on the "
            "subject/block pattern formulation."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--linearization-level", type=int, default=2)
    parser.add_argument("--hint", choices=["none", "heuristic"], default="heuristic")
    parser.add_argument(
        "--hint-file",
        type=Path,
        default=None,
        help="Optional timetable CSV in Exam_Name/Date/Slot format. Overrides --hint when supplied.",
    )
    parser.add_argument("--decision-strategy", choices=["none", "subject_order", "pair_mass"], default="pair_mass")
    parser.add_argument("--log-search-progress", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("toy_cp_timetable.csv"))
    parser.add_argument("--heuristic-output", type=Path, default=Path("toy_cp_heuristic_hint.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("toy_cp_summary.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("toy_cp_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("toy_cp.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_cp_bounds.png"))
    args = parser.parse_args()

    raw_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    raw_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    exams, days, pairs = prepare_toy_inputs(raw_exams, raw_pairs)
    pairs = _clean_pairs(pairs)
    days = days.copy()
    days["Date"] = parse_day_series(days["Date"])

    heuristic_timetable: pd.DataFrame | None = None
    heuristic_objective: float | None = None
    hint_source = args.hint
    if args.hint_file is not None:
        heuristic_timetable = pd.read_csv(args.hint_file)
        heuristic_objective = float(mip_objective_value(heuristic_timetable, pairs, days, mode="anthony_appendix"))
        hint_source = f"file:{args.hint_file}"
    elif args.hint == "heuristic":
        heuristic = solve_toy_heuristic(
            raw_exams,
            raw_pairs,
            max_rounds=100,
            objective_mode="anthony_appendix",
        )
        heuristic_timetable = heuristic.timetable
        heuristic_objective = float(heuristic.objective_value)
        heuristic_timetable.to_csv(args.heuristic_output, index=False)

    started = time.perf_counter()
    result = solve_toy_cp_sat(
        exams=exams,
        days=days,
        pairs=pairs,
        time_limit=args.time_limit,
        workers=args.workers,
        random_seed=args.random_seed,
        linearization_level=args.linearization_level,
        hint_timetable=heuristic_timetable,
        decision_strategy=args.decision_strategy,
        log_search_progress=args.log_search_progress,
    )
    elapsed_seconds = time.perf_counter() - started

    timetable = result["timetable"]
    timetable.to_csv(args.output, index=False)

    progress = pd.DataFrame(result["progress"])
    progress.to_csv(args.progress_output, index=False)
    _plot_progress(progress, args.plot_output)

    args.log_output.write_text(result["log_text"], encoding="utf-8")

    validation_objective = mip_objective_value(timetable, pairs, days, mode="anthony_appendix")
    placement = placement_from_timetable(timetable)
    same_slot_clashes = _same_slot_clashes(placement, pairs)

    summary = {
        "status": result["status_name"],
        "status_code": result["status_code"],
        "objective": result["objective"],
        "validation_objective": validation_objective,
        "best_bound": result["best_bound"],
        "gap": result["gap"],
        "same_slot_clashes": same_slot_clashes,
        "wall_time_seconds": result["wall_time_seconds"],
        "elapsed_seconds": elapsed_seconds,
        "branches": result["branches"],
        "conflicts": result["conflicts"],
        "patterns": result["pattern_count"],
        "mu_variables": result["mu_count"],
        "incompatible_mu_variables": result["incompatible_mu_count"],
        "hint": hint_source,
        "hint_matched_subjects": result["hint_matched_subjects"],
        "heuristic_objective": heuristic_objective,
        "decision_strategy": args.decision_strategy,
        "workers": args.workers,
        "random_seed": args.random_seed,
        "linearization_level": args.linearization_level,
    }
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("CP-SAT status:", result["status_name"])
    print("CP-SAT objective:", result["objective"])
    print("Validation objective:", validation_objective)
    print("Best bound:", result["best_bound"])
    print("Gap:", result["gap"])
    print("Same-slot clashes:", same_slot_clashes)
    print("Wall seconds:", result["wall_time_seconds"])
    print("Branches:", result["branches"])
    print("Conflicts:", result["conflicts"])
    print("Patterns:", result["pattern_count"])
    print("Mu variables:", result["mu_count"])
    print(f"Saved timetable to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved progress to {args.progress_output}")
    print(f"Saved log to {args.log_output}")
    print(f"Saved plot to {args.plot_output}")


def solve_toy_cp_sat(
    *,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    time_limit: float | None,
    workers: int,
    random_seed: int,
    linearization_level: int,
    hint_timetable: pd.DataFrame | None,
    decision_strategy: str,
    log_search_progress: bool,
) -> dict[str, Any]:
    dates = days["Date"].tolist()
    date_index = {date: idx for idx, date in enumerate(dates)}
    exam_lengths = dict(zip(exams["Full Name"].astype(str), exams["Length"].astype(float)))
    blocks = _build_blocks(exams)
    patterns = _generate_all_patterns(
        blocks=blocks,
        exams=exams,
        days=days,
        pairs=pairs,
        dates=dates,
        date_index=date_index,
        exam_lengths=exam_lengths,
    )
    subjects = list(patterns)

    model = cp_model.CpModel()
    lam: dict[tuple[str, int], cp_model.IntVar] = {}
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            lam[(subject, pattern.index)] = model.NewBoolVar(f"lambda[{_name(subject)},{pattern.index}]")

    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
    mu: dict[tuple[str, int, str, int], cp_model.IntVar] = {}
    incompatible_count = 0
    for left_pos, left_subject in enumerate(subjects):
        for right_subject in subjects[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                for right_pattern in patterns[right_subject]:
                    cost = _cross_cost(
                        left_pattern.assignment,
                        right_pattern.assignment,
                        pairs=pairs,
                        date_index=date_index,
                        exam_lengths=exam_lengths,
                    )
                    key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                    cross_costs[key] = cost
                    mu_var = model.NewBoolVar(
                        f"mu[{_name(left_subject)},{left_pattern.index},{_name(right_subject)},{right_pattern.index}]"
                    )
                    mu[key] = mu_var
                    if cost.daily_length_infeasible:
                        model.Add(mu_var == 0)
                        incompatible_count += 1

    for subject, subject_patterns in patterns.items():
        model.Add(sum(lam[(subject, pattern.index)] for pattern in subject_patterns) == 1)

    for left_pos, left_subject in enumerate(subjects):
        for right_subject in subjects[left_pos + 1 :]:
            right_patterns = patterns[right_subject]
            left_patterns = patterns[left_subject]
            for left_pattern in left_patterns:
                model.Add(
                    sum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for right_pattern in right_patterns
                    )
                    == lam[(left_subject, left_pattern.index)]
                )
            for right_pattern in right_patterns:
                model.Add(
                    sum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for left_pattern in left_patterns
                    )
                    == lam[(right_subject, right_pattern.index)]
                )

    clash_terms: list[cp_model.LinearExpr] = []
    objective_terms: list[cp_model.LinearExpr] = []
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            var = lam[(subject, pattern.index)]
            internal_objective = _as_int(pattern.internal_objective, "internal objective")
            internal_clashes = _as_int(pattern.internal_same_slot_clashes, "internal clash count")
            if internal_objective:
                objective_terms.append(internal_objective * var)
            if internal_clashes:
                clash_terms.append(internal_clashes * var)

    for key, var in mu.items():
        cost = cross_costs[key]
        objective = _as_int(cost.objective, "cross objective")
        clashes = _as_int(cost.same_slot_clashes, "cross clash count")
        if objective:
            objective_terms.append(objective * var)
        if clashes:
            clash_terms.append(clashes * var)

    model.Add(sum(clash_terms) <= 15)
    model.Minimize(sum(objective_terms))

    hint_matched_subjects = 0
    if hint_timetable is not None:
        hint_matched_subjects = _add_pattern_hint(
            model=model,
            lam=lam,
            mu=mu,
            patterns=patterns,
            subjects=subjects,
            hint_timetable=hint_timetable,
        )

    if decision_strategy != "none":
        ordered_lam = _ordered_lambda_vars(
            lam=lam,
            patterns=patterns,
            subjects=subjects,
            blocks=blocks,
            pairs=pairs,
            mode=decision_strategy,
        )
        model.AddDecisionStrategy(
            ordered_lam,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )

    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.linearization_level = int(linearization_level)
    solver.parameters.log_search_progress = bool(log_search_progress)
    solver.parameters.log_to_stdout = False

    log_lines: list[str] = []
    if hasattr(solver, "log_callback"):
        solver.log_callback = log_lines.append

    progress_callback = _CpProgressCallback()
    status_code = solver.Solve(model, progress_callback)

    selected_patterns: dict[str, Pattern] = {}
    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    if status_code in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        for subject, subject_patterns in patterns.items():
            selected = [
                pattern
                for pattern in subject_patterns
                if solver.BooleanValue(lam[(subject, pattern.index)])
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Expected exactly one selected pattern for {subject!r}, got {len(selected)}.")
            selected_pattern = selected[0]
            selected_patterns[subject] = selected_pattern
            placement.update(selected_pattern.assignment)
        timetable = timetable_from_placement(placement, days)
        objective = float(solver.ObjectiveValue())
    else:
        timetable = pd.DataFrame(columns=["Day_of_Week", "Date", "Slot", "Exam_Name"])
        objective = None

    best_bound = float(solver.BestObjectiveBound())
    if objective is None:
        gap = None
    else:
        gap = abs(objective - best_bound) / max(1.0, abs(objective))

    return {
        "status_code": int(status_code),
        "status_name": solver.StatusName(status_code),
        "objective": objective,
        "best_bound": best_bound,
        "gap": gap,
        "wall_time_seconds": float(solver.WallTime()),
        "branches": int(solver.NumBranches()),
        "conflicts": int(solver.NumConflicts()),
        "pattern_count": sum(len(subject_patterns) for subject_patterns in patterns.values()),
        "mu_count": len(mu),
        "incompatible_mu_count": incompatible_count,
        "hint_matched_subjects": hint_matched_subjects,
        "progress": progress_callback.rows,
        "log_text": "".join(log_lines),
        "selected_patterns": selected_patterns,
        "timetable": timetable,
    }


class _CpProgressCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, float]] = []

    def on_solution_callback(self) -> None:
        objective = float(self.ObjectiveValue())
        best_bound = float(self.BestObjectiveBound())
        self.rows.append(
            {
                "time_seconds": float(self.WallTime()),
                "incumbent": objective,
                "best_bound": best_bound,
                "gap": abs(objective - best_bound) / max(1.0, abs(objective)),
                "branches": float(self.NumBranches()),
                "conflicts": float(self.NumConflicts()),
            }
        )


def _add_pattern_hint(
    *,
    model: cp_model.CpModel,
    lam: dict[tuple[str, int], cp_model.IntVar],
    mu: dict[tuple[str, int, str, int], cp_model.IntVar],
    patterns: dict[str, list[Pattern]],
    subjects: list[str],
    hint_timetable: pd.DataFrame,
) -> int:
    hint_placement = placement_from_timetable(hint_timetable)
    selected_by_subject: dict[str, int] = {}
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            if _assignment_matches(pattern.assignment, hint_placement):
                selected_by_subject[subject] = pattern.index
                break

    for subject, subject_patterns in patterns.items():
        selected_index = selected_by_subject.get(subject)
        for pattern in subject_patterns:
            model.AddHint(lam[(subject, pattern.index)], int(pattern.index == selected_index))

    for left_pos, left_subject in enumerate(subjects):
        left_index = selected_by_subject.get(left_subject)
        for right_subject in subjects[left_pos + 1 :]:
            right_index = selected_by_subject.get(right_subject)
            for key, var in mu.items():
                key_left_subject, key_left_index, key_right_subject, key_right_index = key
                if key_left_subject != left_subject or key_right_subject != right_subject:
                    continue
                selected = left_index == key_left_index and right_index == key_right_index
                model.AddHint(var, int(selected))

    return len(selected_by_subject)


def _assignment_matches(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    placement: dict[str, tuple[pd.Timestamp, str]],
) -> bool:
    for exam, (date, slot) in assignment.items():
        if placement.get(exam) != (pd.Timestamp(date).normalize(), slot):
            return False
    return True


def _ordered_lambda_vars(
    *,
    lam: dict[tuple[str, int], cp_model.IntVar],
    patterns: dict[str, list[Pattern]],
    subjects: list[str],
    blocks: dict[str, list[str]],
    pairs: pd.DataFrame,
    mode: str,
) -> list[cp_model.IntVar]:
    if mode == "subject_order":
        ordered_subjects = subjects
    elif mode == "pair_mass":
        subject_scores = {
            subject: _subject_pair_mass(subject=subject, blocks=blocks, pairs=pairs)
            for subject in subjects
        }
        ordered_subjects = sorted(subjects, key=lambda subject: (-subject_scores[subject], subject))
    else:
        raise ValueError(f"Unknown decision strategy: {mode}")

    result: list[cp_model.IntVar] = []
    for subject in ordered_subjects:
        subject_patterns = sorted(
            patterns[subject],
            key=lambda pattern: (pattern.internal_objective, pattern.index),
        )
        result.extend(lam[(subject, pattern.index)] for pattern in subject_patterns)
    return result


def _subject_pair_mass(*, subject: str, blocks: dict[str, list[str]], pairs: pd.DataFrame) -> float:
    subject_exams = blocks[subject]
    other_exams = [exam for other_subject, exams in blocks.items() if other_subject != subject for exam in exams]
    return float(pairs.loc[subject_exams, other_exams].sum().sum())


def _as_int(value: float, label: str) -> int:
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1e-6:
        raise ValueError(f"CP-SAT requires integer {label}; got {value!r}.")
    return int(rounded)


def _plot_progress(progress: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Best bound", color="#1f77b4", linewidth=3.0)
        ax.legend(loc="best")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title("Toy CP-SAT incumbent and bound evolution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _format_clock_time(seconds: float, _pos: int | None = None) -> str:
    if pd.isna(seconds):
        return ""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _name(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text)


if __name__ == "__main__":
    main()
