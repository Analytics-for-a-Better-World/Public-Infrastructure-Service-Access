from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from src.anthony_model import DEFAULT_SLOTS, mip_objective_value, parse_day_series, prepare_toy_inputs, timetable_from_placement


WEIGHTS = {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1}


@dataclass(frozen=True)
class Pattern:
    subject: str
    index: int
    assignment: dict[str, tuple[pd.Timestamp, str]]
    internal_objective: float
    internal_same_slot_clashes: float


@dataclass(frozen=True)
class CrossCost:
    objective: float
    same_slot_clashes: float
    daily_length_infeasible: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the toy instance with a subject/block pattern-pair formulation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--node-limit", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("toy_pattern_model_timetable.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("toy_pattern_model_summary.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("toy_pattern_model_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("toy_pattern_model.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_pattern_model_bounds.png"))
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--relax-lambda", action="store_true")
    parser.add_argument("--mu-binary", action="store_true")
    args = parser.parse_args()

    raw_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    raw_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    exams, days, pairs = prepare_toy_inputs(raw_exams, raw_pairs)
    pairs = _clean_pairs(pairs)
    days = days.copy()
    days["Date"] = parse_day_series(days["Date"])

    started = time.perf_counter()
    result = solve_pattern_model(
        exams=exams,
        days=days,
        pairs=pairs,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        log_output=args.log_output,
        output_flag=args.output_flag,
        relax_lambda=args.relax_lambda,
        mu_binary=args.mu_binary,
        progress_output=args.progress_output,
        plot_output=args.plot_output,
    )
    elapsed = time.perf_counter() - started

    timetable = result["timetable"]
    timetable.to_csv(args.output, index=False)

    if args.relax_lambda:
        validation_value = None
        reported_same_slot_clashes = None
    else:
        validation_value = mip_objective_value(timetable, pairs, days, mode="anthony_appendix")
        reported_same_slot_clashes = result["same_slot_clashes"]
    summary = {
        "status": result["status"],
        "objective": result["objective"],
        "validation_objective": validation_value,
        "best_bound": result["best_bound"],
        "gap": result["gap"],
        "nodes": result["nodes"],
        "iterations": result["iterations"],
        "work": result["work"],
        "runtime_seconds": result["runtime_seconds"],
        "elapsed_seconds": elapsed,
        "pattern_count": result["pattern_count"],
        "mu_count": result["mu_count"],
        "incompatible_mu_count": result["incompatible_mu_count"],
        "same_slot_clashes": reported_same_slot_clashes,
        "max_clashes": 15,
        "relax_lambda": args.relax_lambda,
        "mu_binary": args.mu_binary,
    }
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("Pattern model seconds:", round(elapsed, 6))
    print("Pattern model status:", result["status"])
    print("Pattern model objective:", result["objective"])
    print("Validation objective:", validation_value)
    if args.relax_lambda:
        print("Relaxation run: extracted timetable is only a rounded diagnostic, not a feasible incumbent.")
    print("Best bound:", result["best_bound"])
    print("Gap:", result["gap"])
    print("Nodes:", result["nodes"])
    print("Iterations:", result["iterations"])
    print("Work:", result["work"])
    print("Patterns:", result["pattern_count"])
    print("Mu variables:", result["mu_count"])
    print("Incompatible mu variables fixed to zero:", result["incompatible_mu_count"])
    print("Same-slot clashes:", result["same_slot_clashes"])
    print(f"Saved timetable to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved progress to {args.progress_output}")
    print(f"Saved log to {args.log_output}")
    print(f"Saved bound plot to {args.plot_output}")


def solve_pattern_model(
    *,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    time_limit: float | None,
    node_limit: float | None,
    log_output: Path,
    output_flag: int,
    relax_lambda: bool,
    mu_binary: bool,
    progress_output: Path,
    plot_output: Path,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("run_toy_pattern_model.py requires gurobipy.") from exc

    dates = parse_day_series(days["Date"]).tolist()
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

    model = gp.Model("toy_subject_pattern_model")
    model.setParam("OutputFlag", output_flag)
    model.setParam("LogFile", str(log_output))
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if node_limit is not None:
        model.setParam("NodeLimit", node_limit)

    lambda_type = GRB.CONTINUOUS if relax_lambda else GRB.BINARY
    mu_type = GRB.BINARY if mu_binary else GRB.CONTINUOUS

    lam: dict[tuple[str, int], Any] = {}
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            lam[(subject, pattern.index)] = model.addVar(
                vtype=lambda_type,
                lb=0.0,
                ub=1.0,
                name=_var_name("lambda", subject, pattern.index),
            )

    subjects = list(patterns)
    mu: dict[tuple[str, int, str, int], Any] = {}
    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
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
                    ub = 0.0 if cost.daily_length_infeasible else 1.0
                    if cost.daily_length_infeasible:
                        incompatible_count += 1
                    mu[key] = model.addVar(
                        vtype=mu_type,
                        lb=0.0,
                        ub=ub,
                        name=_var_name("mu", left_subject, left_pattern.index, right_subject, right_pattern.index),
                    )

    model.update()

    for subject, subject_patterns in patterns.items():
        model.addConstr(
            gp.quicksum(lam[(subject, pattern.index)] for pattern in subject_patterns) == 1,
            name=_var_name("choose_pattern", subject),
        )

    for left_pos, left_subject in enumerate(subjects):
        for right_subject in subjects[left_pos + 1 :]:
            right_patterns = patterns[right_subject]
            left_patterns = patterns[left_subject]
            for left_pattern in left_patterns:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for right_pattern in right_patterns
                    )
                    == lam[(left_subject, left_pattern.index)],
                    name=_var_name("mu_left", left_subject, left_pattern.index, right_subject),
                )
            for right_pattern in right_patterns:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for left_pattern in left_patterns
                    )
                    == lam[(right_subject, right_pattern.index)],
                    name=_var_name("mu_right", left_subject, right_subject, right_pattern.index),
                )

    clash_expr = gp.LinExpr()
    objective = gp.LinExpr()
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            var = lam[(subject, pattern.index)]
            if pattern.internal_objective:
                objective.addTerms(pattern.internal_objective, var)
            if pattern.internal_same_slot_clashes:
                clash_expr.addTerms(pattern.internal_same_slot_clashes, var)

    for key, var in mu.items():
        cost = cross_costs[key]
        if cost.objective:
            objective.addTerms(cost.objective, var)
        if cost.same_slot_clashes:
            clash_expr.addTerms(cost.same_slot_clashes, var)

    model.addConstr(clash_expr <= 15, name="max_same_slot_clashes")
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    progress: list[dict[str, float]] = []

    def callback(cb_model: Any, where: int) -> None:
        if where == GRB.Callback.MIP:
            runtime = float(cb_model.cbGet(GRB.Callback.RUNTIME))
            node_count = float(cb_model.cbGet(GRB.Callback.MIP_NODCNT))
            incumbent = float(cb_model.cbGet(GRB.Callback.MIP_OBJBST))
            bound = float(cb_model.cbGet(GRB.Callback.MIP_OBJBND))
            if abs(incumbent) >= 1e100:
                gap = float("nan")
            else:
                gap = abs(incumbent - bound) / max(1.0, abs(incumbent))
            progress.append(
                {
                    "time_seconds": runtime,
                    "nodes": node_count,
                    "incumbent": incumbent,
                    "best_bound": bound,
                    "gap": gap,
                }
            )

    model.optimize(callback)

    progress_df = _deduplicate_progress(pd.DataFrame(progress))
    progress_df.to_csv(progress_output, index=False)
    _plot_progress(progress_df, plot_output)

    chosen: dict[str, Pattern] = {}
    for subject, subject_patterns in patterns.items():
        chosen_pattern = max(subject_patterns, key=lambda pattern: lam[(subject, pattern.index)].X)
        chosen[subject] = chosen_pattern

    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for pattern in chosen.values():
        placement.update(pattern.assignment)
    timetable = timetable_from_placement(placement, days)
    same_slot_clashes = _same_slot_clashes(placement, pairs)

    objective_value = _safe_model_attr(model, "ObjVal") if model.SolCount else None
    best_bound = _safe_model_attr(model, "ObjBound")
    if best_bound is None and objective_value is not None:
        best_bound = objective_value
    gap_value = _safe_model_attr(model, "MIPGap") if model.SolCount else None
    if gap_value is None and objective_value is not None and best_bound is not None:
        gap_value = abs(objective_value - best_bound) / max(1.0, abs(objective_value))

    return {
        "status": int(model.Status),
        "objective": objective_value,
        "best_bound": best_bound,
        "gap": gap_value,
        "nodes": float(model.NodeCount),
        "iterations": float(model.IterCount),
        "work": float(model.Work),
        "runtime_seconds": float(model.Runtime),
        "pattern_count": sum(len(subject_patterns) for subject_patterns in patterns.values()),
        "mu_count": len(mu),
        "incompatible_mu_count": incompatible_count,
        "same_slot_clashes": same_slot_clashes,
        "timetable": timetable,
    }


def _clean_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    result = pairs.copy()
    if result.columns[0].startswith("Unnamed") or result.columns[0] not in result.columns[1:]:
        result = result.set_index(result.columns[0])
    result.index = result.index.astype(str).str.strip()
    result.columns = result.columns.astype(str).str.strip()
    return result.apply(pd.to_numeric, errors="raise")


def _build_blocks(exams: pd.DataFrame) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for subject, subject_df in exams.groupby("Subject", sort=False):
        subject_df = subject_df.sort_values("Exam")
        blocks[str(subject)] = subject_df["Full Name"].astype(str).tolist()
    return blocks


def _generate_all_patterns(
    *,
    blocks: dict[str, list[str]],
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    dates: list[pd.Timestamp],
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
) -> dict[str, list[Pattern]]:
    by_exam_subject = dict(zip(exams["Full Name"].astype(str), exams["Subject"].astype(str)))
    usable_dates = set(_usable_dates(days))
    second_half_dates = set(dates[round(len(dates) / 2) :])
    first_half_subjects = {"Finance", "Law and Ethics"}
    first_two_dates = set(dates[:2])
    result: dict[str, list[Pattern]] = {}

    for subject, exam_names in blocks.items():
        if len(exam_names) != 2:
            raise ValueError(f"Toy pattern model expects two exams per subject; {subject!r} has {len(exam_names)}.")
        subject_patterns: list[Pattern] = []
        for start_idx in range(len(dates) - 1):
            left_date = dates[start_idx]
            right_date = dates[start_idx + 1]
            if left_date not in usable_dates or right_date not in usable_dates:
                continue
            for first_exam, second_exam in itertools.permutations(exam_names, 2):
                for first_slot in _allowed_slots(first_exam, subject, left_date, exam_lengths):
                    for second_slot in _allowed_slots(second_exam, subject, right_date, exam_lengths):
                        assignment = {
                            first_exam: (left_date, first_slot),
                            second_exam: (right_date, second_slot),
                        }
                        if not _assignment_passes_toy_global_rules(
                            assignment=assignment,
                            subject=subject,
                            dates=dates,
                            first_two_dates=first_two_dates,
                            second_half_dates=second_half_dates,
                            first_half_subjects=first_half_subjects,
                        ):
                            continue
                        internal_objective, internal_clashes = _assignment_cost(
                            assignment,
                            pairs=pairs,
                            date_index=date_index,
                        )
                        subject_patterns.append(
                            Pattern(
                                subject=subject,
                                index=len(subject_patterns),
                                assignment=assignment,
                                internal_objective=internal_objective,
                                internal_same_slot_clashes=internal_clashes,
                            )
                        )
        if not subject_patterns:
            raise ValueError(f"No feasible subject patterns generated for {subject!r} ({exam_names}).")
        result[subject] = subject_patterns
    return result


def _usable_dates(days: pd.DataFrame) -> list[pd.Timestamp]:
    dates = parse_day_series(days["Date"])
    is_weekend = dates.dt.dayofweek >= 5
    is_may_first = dates.dt.month.eq(5) & dates.dt.day.eq(1)
    return dates.loc[~(is_weekend | is_may_first)].tolist()


def _allowed_slots(exam: str, subject: str, date: pd.Timestamp, exam_lengths: dict[str, float]) -> list[str]:
    slots = list(DEFAULT_SLOTS)
    if float(exam_lengths[exam]) > 180:
        slots = [slot for slot in slots if slot != "PM"]
    if subject.upper() == "LANGUAGE A LITERATURE" and pd.Timestamp(date).day_name() == "Friday":
        return []
    return slots


def _assignment_passes_toy_global_rules(
    *,
    assignment: dict[str, tuple[pd.Timestamp, str]],
    subject: str,
    dates: list[pd.Timestamp],
    first_two_dates: set[pd.Timestamp],
    second_half_dates: set[pd.Timestamp],
    first_half_subjects: set[str],
) -> bool:
    if subject.upper() == "SBS":
        if assignment.get("SBS Exam 1") != (dates[0], "AM"):
            return False
    else:
        if any(date in first_two_dates for date, _slot in assignment.values()):
            return False

    if subject in first_half_subjects:
        if any(date in second_half_dates for date, _slot in assignment.values()):
            return False
    return True


def _assignment_cost(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
) -> tuple[float, float]:
    objective = 0.0
    clashes = 0.0
    exams = sorted(assignment)
    for pos, exam_i in enumerate(exams):
        for exam_j in exams[pos + 1 :]:
            contribution, same_slot = _pair_cost_and_clash(
                exam_i,
                assignment[exam_i],
                exam_j,
                assignment[exam_j],
                pairs=pairs,
                date_index=date_index,
            )
            objective += contribution
            clashes += same_slot
    return objective, clashes


def _cross_cost(
    left_assignment: dict[str, tuple[pd.Timestamp, str]],
    right_assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
) -> CrossCost:
    objective = 0.0
    same_slot_clashes = 0.0
    daily_length_infeasible = False
    for exam_i, placement_i in left_assignment.items():
        for exam_j, placement_j in right_assignment.items():
            if placement_i[0] == placement_j[0] and exam_lengths[exam_i] + exam_lengths[exam_j] > 375:
                daily_length_infeasible = True
            contribution, clash = _pair_cost_and_clash(
                exam_i,
                placement_i,
                exam_j,
                placement_j,
                pairs=pairs,
                date_index=date_index,
            )
            objective += contribution
            same_slot_clashes += clash
    return CrossCost(objective=objective, same_slot_clashes=same_slot_clashes, daily_length_infeasible=daily_length_infeasible)


def _pair_cost_and_clash(
    exam_i: str,
    placement_i: tuple[pd.Timestamp, str],
    exam_j: str,
    placement_j: tuple[pd.Timestamp, str],
    *,
    pairs: pd.DataFrame,
    date_index: dict[pd.Timestamp, int],
) -> tuple[float, float]:
    date_i, slot_i = placement_i
    date_j, slot_j = placement_j
    cij = float(pairs.loc[exam_i, exam_j])
    gap = abs(date_index[pd.Timestamp(date_i).normalize()] - date_index[pd.Timestamp(date_j).normalize()])

    objective = 0.0
    clash = 0.0
    if gap == 0 and slot_i == slot_j:
        objective += cij * WEIGHTS["a"]
        clash += cij
    if gap == 0:
        objective += cij * WEIGHTS["b"]
    elif gap == 1 or gap == 5:
        objective += cij * WEIGHTS["c"]
    elif gap == 2:
        objective += cij * WEIGHTS["d"]
    elif gap == 3:
        objective += cij * WEIGHTS["e"]
    elif gap == 4:
        objective += cij * WEIGHTS["f"]
    return objective, clash


def _same_slot_clashes(placement: dict[str, tuple[pd.Timestamp, str]], pairs: pd.DataFrame) -> float:
    total = 0.0
    exams = sorted(placement)
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            if date_i == date_j and slot_i == slot_j:
                total += float(pairs.loc[exam_i, exam_j])
    return total


def _deduplicate_progress(progress: pd.DataFrame) -> pd.DataFrame:
    if progress.empty:
        return progress
    progress = progress.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["incumbent", "best_bound"])
    progress["time_seconds"] = progress["time_seconds"].round(3)
    progress["nodes"] = progress["nodes"].round(0)
    return progress.drop_duplicates(subset=["time_seconds", "nodes", "incumbent", "best_bound"]).reset_index(drop=True)


def _plot_progress(progress: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Best bound", color="#1f77b4", linewidth=3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title("Toy subject-pattern MILP incumbent and bound evolution")
    ax.grid(True, alpha=0.25)
    if not progress.empty:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _safe_model_attr(model: Any, attr_name: str) -> float | None:
    try:
        return float(getattr(model, attr_name))
    except Exception:
        return None


def _format_clock_time(seconds: float, _pos: int | None = None) -> str:
    seconds_int = int(round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def _var_name(prefix: str, *parts: object) -> str:
    clean = [str(part).replace(" ", "_").replace("[", "(").replace("]", ")").replace(",", "_") for part in parts]
    return prefix + "[" + ",".join(clean) + "]"


if __name__ == "__main__":
    main()
