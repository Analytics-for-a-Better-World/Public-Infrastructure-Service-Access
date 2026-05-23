from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.anthony_model import (
    DEFAULT_WEIGHTS,
    load_default_data,
    mip_objective_value,
    parse_day_series,
    placement_from_timetable,
    prepare_anthony_model_data,
    timetable_from_placement,
)
from src.full_heuristic import (
    _assignment_same_slot_delta,
    _build_blocks,
    _generate_block_candidates,
    _pair_value_map,
    _same_slot_clashes,
    validate_full_solution,
)


@dataclass(frozen=True)
class Pattern:
    subject: str
    index: int
    assignment: dict[str, tuple[pd.Timestamp, str]]
    fixed_objective_delta: float
    fixed_same_slot_delta: float


@dataclass(frozen=True)
class CrossCost:
    objective: float
    same_slot_clashes: float
    daily_length_infeasible: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improve a full-instance timetable with a subject-pattern LNS subproblem."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, default=Path("full_lns_nb34_recommended_6x120_guarded.csv"))
    parser.add_argument("--nb-days", type=int, default=34)
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--selected-subject", action="append", default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--objective-mode", choices=["formal", "anthony_appendix"], default="formal")
    parser.add_argument("--output", type=Path, default=Path("full_pattern_lns_timetable.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("full_pattern_lns_summary.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("full_pattern_lns.log"))
    parser.add_argument("--output-flag", type=int, default=1)
    args = parser.parse_args()

    exams, days, pairs = load_default_data(args.data_dir)
    data = prepare_anthony_model_data(exams, days, pairs, nb_days=args.nb_days, recode_math_paper_three=True)
    start_timetable = pd.read_csv(args.start)
    start_timetable["Date"] = pd.to_datetime(start_timetable["Date"], dayfirst=True).dt.normalize()
    validate_full_solution(start_timetable, data)

    selected_subjects = args.selected_subject
    if selected_subjects is None:
        selected_subjects = select_high_contribution_subjects(
            start_timetable,
            data=data,
            count=args.subjects,
            objective_mode=args.objective_mode,
        )

    started = time.perf_counter()
    result = improve_with_pattern_lns(
        start_timetable=start_timetable,
        data=data,
        selected_subjects=selected_subjects,
        time_limit=args.time_limit,
        objective_mode=args.objective_mode,
        log_output=args.log_output,
        output_flag=args.output_flag,
    )
    elapsed = time.perf_counter() - started

    result["timetable"].to_csv(args.output, index=False)
    summary = {
        "start_file": str(args.start),
        "objective_mode": args.objective_mode,
        "selected_subjects": ";".join(selected_subjects),
        "start_objective": result["start_objective"],
        "candidate_objective": result["candidate_objective"],
        "solver_candidate_objective": result["solver_candidate_objective"],
        "accepted": result["accepted"],
        "improvement": result["start_objective"] - result["candidate_objective"],
        "start_same_slot_clashes": result["start_same_slot_clashes"],
        "candidate_same_slot_clashes": result["candidate_same_slot_clashes"],
        "status": result["status"],
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
    }
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("Selected subjects:", ", ".join(selected_subjects))
    print("Start objective:", result["start_objective"])
    print("Candidate objective:", result["candidate_objective"])
    print("Solver candidate objective:", result["solver_candidate_objective"])
    print("Accepted:", result["accepted"])
    print("Improvement:", result["start_objective"] - result["candidate_objective"])
    print("Start same-slot clashes:", result["start_same_slot_clashes"])
    print("Candidate same-slot clashes:", result["candidate_same_slot_clashes"])
    print("Status:", result["status"])
    print("Best bound:", result["best_bound"])
    print("Gap:", result["gap"])
    print("Nodes:", result["nodes"])
    print("Iterations:", result["iterations"])
    print("Work:", result["work"])
    print("Patterns:", result["pattern_count"])
    print("Mu variables:", result["mu_count"])
    print("Incompatible mu variables fixed to zero:", result["incompatible_mu_count"])
    print("Elapsed seconds:", round(elapsed, 6))
    print(f"Saved timetable to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved log to {args.log_output}")


def improve_with_pattern_lns(
    *,
    start_timetable: pd.DataFrame,
    data: Any,
    selected_subjects: list[str],
    time_limit: float,
    objective_mode: str,
    log_output: Path,
    output_flag: int,
) -> dict[str, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("run_full_pattern_lns.py requires gurobipy.") from exc

    weights = dict(DEFAULT_WEIGHTS)
    pair_values = _pair_value_map(data.pairs)
    day_index = {date: idx for idx, date in enumerate(data.dates)}
    exam_lengths = dict(zip(data.exams["Full Name"], data.exams["Length"]))
    exam_subjects = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    full_placement = placement_from_timetable(start_timetable)
    selected_subjects = list(dict.fromkeys(selected_subjects))
    selected_exams = set(
        data.exams.loc[data.exams["Subject"].isin(selected_subjects), "Full Name"].tolist()
    )
    fixed_placement = {
        exam: placement
        for exam, placement in full_placement.items()
        if exam not in selected_exams
    }
    first_half_subjects = {"BUSINESS MANAGEMENT", "HISTORY", "ENGLISH A LAL"}
    blocks = _build_blocks(data.exams)

    patterns: dict[str, list[Pattern]] = {}
    for subject in selected_subjects:
        raw_candidates = _generate_block_candidates(
            block_id=subject,
            block=blocks[subject],
            dates=data.dates,
            days=data.days,
            slots=tuple(data.slots),
            exam_lengths=exam_lengths,
            exam_subjects=exam_subjects,
            max_afternoon_minutes=180,
            first_half_subjects=first_half_subjects,
        )
        subject_patterns: list[Pattern] = []
        for candidate in raw_candidates:
            assignment = candidate["assignment"]
            if _violates_fixed_daily_length(assignment, fixed_placement, exam_lengths):
                continue
            subject_patterns.append(
                Pattern(
                    subject=subject,
                    index=len(subject_patterns),
                    assignment=assignment,
                    fixed_objective_delta=_assignment_objective_delta_exact(
                        assignment,
                        base_placement=fixed_placement,
                        pair_values=pair_values,
                        day_index=day_index,
                        objective_mode=objective_mode,
                        weights=weights,
                    ),
                    fixed_same_slot_delta=_assignment_same_slot_delta(
                        assignment,
                        base_placement=fixed_placement,
                        pair_values=pair_values,
                    ),
                )
            )
        current_assignment = {
            exam: full_placement[exam]
            for exam in blocks[subject]["exams"]
        }
        current_signature = _assignment_signature(current_assignment)
        if not any(_assignment_signature(pattern.assignment) == current_signature for pattern in subject_patterns):
            if _violates_fixed_daily_length(current_assignment, fixed_placement, exam_lengths):
                raise ValueError(f"Current assignment for selected subject {subject!r} is infeasible.")
            subject_patterns.append(
                Pattern(
                    subject=subject,
                    index=len(subject_patterns),
                    assignment=current_assignment,
                    fixed_objective_delta=_assignment_objective_delta_exact(
                        current_assignment,
                        base_placement=fixed_placement,
                        pair_values=pair_values,
                        day_index=day_index,
                        objective_mode=objective_mode,
                        weights=weights,
                    ),
                    fixed_same_slot_delta=_assignment_same_slot_delta(
                        current_assignment,
                        base_placement=fixed_placement,
                        pair_values=pair_values,
                    ),
                )
            )
        if not subject_patterns:
            raise ValueError(f"No feasible patterns for selected subject {subject!r}.")
        patterns[subject] = subject_patterns

    fixed_timetable = timetable_from_placement(fixed_placement, data.days)
    fixed_objective = mip_objective_value(fixed_timetable, data.pairs, data.days, mode=objective_mode)
    fixed_same_slot = _same_slot_clashes(fixed_placement, data.pairs, pair_values=pair_values)
    start_objective = mip_objective_value(start_timetable, data.pairs, data.days, mode=objective_mode)
    start_same_slot = _same_slot_clashes(full_placement, data.pairs, pair_values=pair_values)

    model = gp.Model("full_pattern_lns")
    model.setParam("OutputFlag", output_flag)
    model.setParam("LogFile", str(log_output))
    model.setParam("TimeLimit", time_limit)

    lam: dict[tuple[str, int], Any] = {}
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            lam[(subject, pattern.index)] = model.addVar(
                vtype=GRB.BINARY,
                lb=0.0,
                ub=1.0,
                name=_var_name("lambda", subject, pattern.index),
            )

    mu: dict[tuple[str, int, str, int], Any] = {}
    cross_costs: dict[tuple[str, int, str, int], CrossCost] = {}
    incompatible_mu_count = 0
    for left_pos, left_subject in enumerate(selected_subjects):
        for right_subject in selected_subjects[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                for right_pattern in patterns[right_subject]:
                    cost = _cross_cost(
                        left_pattern.assignment,
                        right_pattern.assignment,
                        pair_values=pair_values,
                        day_index=day_index,
                        exam_lengths=exam_lengths,
                        objective_mode=objective_mode,
                        weights=weights,
                    )
                    key = (left_subject, left_pattern.index, right_subject, right_pattern.index)
                    cross_costs[key] = cost
                    ub = 0.0 if cost.daily_length_infeasible else 1.0
                    if cost.daily_length_infeasible:
                        incompatible_mu_count += 1
                    mu[key] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0,
                        ub=ub,
                        name=_var_name("mu", left_subject, left_pattern.index, right_subject, right_pattern.index),
                    )

    model.update()

    start_pattern: dict[str, int] = {}
    for subject, subject_patterns in patterns.items():
        current_assignment = {
            exam: full_placement[exam]
            for exam in blocks[subject]["exams"]
        }
        current_signature = _assignment_signature(current_assignment)
        for pattern in subject_patterns:
            is_current = _assignment_signature(pattern.assignment) == current_signature
            lam[(subject, pattern.index)].Start = 1.0 if is_current else 0.0
            if is_current:
                start_pattern[subject] = pattern.index

    for key, var in mu.items():
        left_subject, left_index, right_subject, right_index = key
        var.Start = 1.0 if (
            start_pattern.get(left_subject) == left_index
            and start_pattern.get(right_subject) == right_index
        ) else 0.0

    for subject, subject_patterns in patterns.items():
        model.addConstr(
            gp.quicksum(lam[(subject, pattern.index)] for pattern in subject_patterns) == 1,
            name=_var_name("choose", subject),
        )

    for left_pos, left_subject in enumerate(selected_subjects):
        for right_subject in selected_subjects[left_pos + 1 :]:
            for left_pattern in patterns[left_subject]:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for right_pattern in patterns[right_subject]
                    )
                    == lam[(left_subject, left_pattern.index)],
                    name=_var_name("mu_left", left_subject, left_pattern.index, right_subject),
                )
            for right_pattern in patterns[right_subject]:
                model.addConstr(
                    gp.quicksum(
                        mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)]
                        for left_pattern in patterns[left_subject]
                    )
                    == lam[(right_subject, right_pattern.index)],
                    name=_var_name("mu_right", left_subject, right_subject, right_pattern.index),
                )

    objective = gp.LinExpr(fixed_objective)
    same_slot_expr = gp.LinExpr(fixed_same_slot)
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            var = lam[(subject, pattern.index)]
            objective.addTerms(pattern.fixed_objective_delta, var)
            same_slot_expr.addTerms(pattern.fixed_same_slot_delta, var)

    for key, var in mu.items():
        cost = cross_costs[key]
        if cost.objective:
            objective.addTerms(cost.objective, var)
        if cost.same_slot_clashes:
            same_slot_expr.addTerms(cost.same_slot_clashes, var)

    model.addConstr(same_slot_expr <= 10_000, name="max_same_slot_clashes")
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()
    model.optimize()

    chosen: dict[str, Pattern] = {}
    for subject, subject_patterns in patterns.items():
        chosen[subject] = max(subject_patterns, key=lambda pattern: lam[(subject, pattern.index)].X)
    candidate_placement = dict(fixed_placement)
    for pattern in chosen.values():
        candidate_placement.update(pattern.assignment)
    solver_candidate_timetable = timetable_from_placement(candidate_placement, data.days)
    validate_full_solution(solver_candidate_timetable, data)

    solver_candidate_objective = mip_objective_value(
        solver_candidate_timetable,
        data.pairs,
        data.days,
        mode=objective_mode,
    )
    solver_candidate_same_slot = _same_slot_clashes(candidate_placement, data.pairs, pair_values=pair_values)
    accepted = solver_candidate_objective <= start_objective + 1e-9
    if accepted:
        output_timetable = solver_candidate_timetable
        output_objective = solver_candidate_objective
        output_same_slot = solver_candidate_same_slot
    else:
        output_timetable = start_timetable
        output_objective = start_objective
        output_same_slot = start_same_slot

    return {
        "timetable": output_timetable,
        "start_objective": start_objective,
        "candidate_objective": output_objective,
        "solver_candidate_objective": solver_candidate_objective,
        "accepted": accepted,
        "start_same_slot_clashes": start_same_slot,
        "candidate_same_slot_clashes": output_same_slot,
        "status": int(model.Status),
        "best_bound": float(model.ObjBound) if model.SolCount else None,
        "gap": float(model.MIPGap) if model.SolCount else None,
        "nodes": float(model.NodeCount),
        "iterations": float(model.IterCount),
        "work": float(model.Work),
        "runtime_seconds": float(model.Runtime),
        "pattern_count": sum(len(subject_patterns) for subject_patterns in patterns.values()),
        "mu_count": len(mu),
        "incompatible_mu_count": incompatible_mu_count,
    }


def select_high_contribution_subjects(
    timetable: pd.DataFrame,
    *,
    data: Any,
    count: int,
    objective_mode: str,
) -> list[str]:
    placement = placement_from_timetable(timetable)
    day_index = {date: idx for idx, date in enumerate(data.dates)}
    exam_subject = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    contributions: dict[str, float] = {subject: 0.0 for subject in data.exams["Subject"].unique()}
    exams = sorted(placement)
    for idx, exam_i in enumerate(exams):
        for exam_j in exams[idx + 1 :]:
            value = _pair_objective(
                exam_i,
                placement[exam_i],
                exam_j,
                placement[exam_j],
                pair_values=None,
                pairs=data.pairs,
                day_index=day_index,
                objective_mode=objective_mode,
                weights=DEFAULT_WEIGHTS,
            )
            if value == 0:
                continue
            subject_i = exam_subject[exam_i]
            subject_j = exam_subject[exam_j]
            if subject_i == subject_j:
                contributions[subject_i] += value
            else:
                contributions[subject_i] += 0.5 * value
                contributions[subject_j] += 0.5 * value
    return [
        subject
        for subject, _value in sorted(contributions.items(), key=lambda item: (-item[1], item[0]))[:count]
    ]


def _assignment_objective_delta_exact(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    base_placement: dict[str, tuple[pd.Timestamp, str]],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    objective_mode: str,
    weights: dict[str, float],
) -> float:
    total = 0.0
    assignment_items = list(assignment.items())
    for idx, (exam_i, placement_i) in enumerate(assignment_items):
        for exam_j, placement_j in assignment_items[idx + 1 :]:
            total += _pair_objective(
                exam_i,
                placement_i,
                exam_j,
                placement_j,
                pair_values=pair_values,
                pairs=None,
                day_index=day_index,
                objective_mode=objective_mode,
                weights=weights,
            )
        for exam_j, placement_j in base_placement.items():
            total += _pair_objective(
                exam_i,
                placement_i,
                exam_j,
                placement_j,
                pair_values=pair_values,
                pairs=None,
                day_index=day_index,
                objective_mode=objective_mode,
                weights=weights,
            )
    return total


def _cross_cost(
    left_assignment: dict[str, tuple[pd.Timestamp, str]],
    right_assignment: dict[str, tuple[pd.Timestamp, str]],
    *,
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    exam_lengths: dict[str, float],
    objective_mode: str,
    weights: dict[str, float],
) -> CrossCost:
    objective = 0.0
    same_slot_clashes = 0.0
    daily_length_infeasible = False
    for exam_i, placement_i in left_assignment.items():
        for exam_j, placement_j in right_assignment.items():
            if placement_i[0] == placement_j[0] and exam_lengths[exam_i] + exam_lengths[exam_j] > 385:
                daily_length_infeasible = True
            objective += _pair_objective(
                exam_i,
                placement_i,
                exam_j,
                placement_j,
                pair_values=pair_values,
                pairs=None,
                day_index=day_index,
                objective_mode=objective_mode,
                weights=weights,
            )
            if placement_i[0] == placement_j[0] and placement_i[1] == placement_j[1]:
                objective_pair = tuple(sorted((exam_i, exam_j)))
                same_slot_clashes += pair_values[objective_pair]
    return CrossCost(
        objective=objective,
        same_slot_clashes=same_slot_clashes,
        daily_length_infeasible=daily_length_infeasible,
    )


def _pair_objective(
    exam_i: str,
    placement_i: tuple[pd.Timestamp, str],
    exam_j: str,
    placement_j: tuple[pd.Timestamp, str],
    *,
    pair_values: dict[tuple[str, str], float] | None,
    pairs: pd.DataFrame | None,
    day_index: dict[pd.Timestamp, int],
    objective_mode: str,
    weights: dict[str, float],
) -> float:
    if pair_values is None:
        if pairs is None:
            raise ValueError("Either pair_values or pairs must be supplied.")
        cij = float(pairs.loc[exam_i, exam_j])
    else:
        cij = pair_values[tuple(sorted((exam_i, exam_j)))]
    if cij == 0:
        return 0.0
    date_i, slot_i = placement_i
    date_j, slot_j = placement_j
    gap = abs(day_index[date_i] - day_index[date_j])
    total = 0.0
    if gap == 0 and slot_i == slot_j:
        total += cij * weights["a"]
    if objective_mode == "anthony_appendix":
        if gap == 0:
            total += cij * weights["b"]
        elif gap == 1 or gap == 5:
            total += cij * weights["c"]
        elif gap == 2:
            total += cij * weights["d"]
        elif gap == 3:
            total += cij * weights["e"]
        elif gap == 4:
            total += cij * weights["f"]
    else:
        if gap == 0 and slot_i != slot_j:
            total += cij * weights["b"]
        elif gap == 1:
            total += cij * weights["c"]
        elif gap == 2:
            total += cij * weights["d"]
        elif gap == 3:
            total += cij * weights["e"]
        elif gap == 4:
            total += cij * weights["f"]
        elif gap == 5:
            total += cij * weights["g"]
    return total


def _violates_fixed_daily_length(
    assignment: dict[str, tuple[pd.Timestamp, str]],
    fixed_placement: dict[str, tuple[pd.Timestamp, str]],
    exam_lengths: dict[str, float],
) -> bool:
    for exam_i, (date_i, _slot_i) in assignment.items():
        for exam_j, (date_j, _slot_j) in fixed_placement.items():
            if date_i == date_j and exam_lengths[exam_i] + exam_lengths[exam_j] > 385:
                return True
    return False


def _assignment_signature(
    assignment: dict[str, tuple[pd.Timestamp, str]],
) -> tuple[tuple[str, pd.Timestamp, str], ...]:
    return tuple(
        sorted(
            (exam, pd.Timestamp(date).normalize(), str(slot))
            for exam, (date, slot) in assignment.items()
        )
    )


def _var_name(prefix: str, *parts: object) -> str:
    clean_parts = [
        str(part)
        .replace(" ", "_")
        .replace("[", "(")
        .replace("]", ")")
        .replace(",", "_")
        for part in parts
    ]
    return prefix + "[" + ",".join(clean_parts) + "]"


if __name__ == "__main__":
    main()
