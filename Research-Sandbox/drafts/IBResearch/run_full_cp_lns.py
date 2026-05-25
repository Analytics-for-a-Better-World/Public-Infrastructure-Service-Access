from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
from ortools.sat.python import cp_model

from run_full_pattern_lns import (
    CrossCost,
    Pattern,
    _assignment_objective_delta_exact,
    _assignment_signature,
    _cross_cost,
    _pair_objective,
    _violates_fixed_daily_length,
)
from run_pattern_lns_sweep import _interaction_neighborhoods, _subject_contribution_order
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Improve a full-instance timetable with an OR-Tools CP-SAT "
            "subject-pattern LNS neighborhood."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--nb-days", type=int, default=34)
    parser.add_argument("--objective-mode", choices=["formal", "anthony_appendix"], default="formal")
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--selected-subject", action="append", default=None)
    parser.add_argument("--selection", choices=["contribution", "interaction"], default="interaction")
    parser.add_argument("--interaction-index", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--linearization-level", type=int, default=2)
    parser.add_argument("--hint", action="store_true")
    parser.add_argument("--decision-strategy", choices=["none", "pattern_cost"], default="none")
    parser.add_argument("--log-search-progress", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("full_cp_lns_timetable.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("full_cp_lns_summary.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("full_cp_lns.log"))
    args = parser.parse_args()

    exams, days, pairs = load_default_data(args.data_dir)
    data = prepare_anthony_model_data(exams, days, pairs, nb_days=args.nb_days, recode_math_paper_three=True)
    start_timetable = pd.read_csv(args.start)
    start_timetable["Date"] = parse_day_series(start_timetable["Date"])
    validate_full_solution(start_timetable, data)

    selected_subjects = args.selected_subject
    if selected_subjects is None:
        selected_subjects = _select_subjects(
            timetable=start_timetable,
            data=data,
            objective_mode=args.objective_mode,
            selection=args.selection,
            subjects=args.subjects,
            interaction_index=args.interaction_index,
        )

    started = time.perf_counter()
    result = improve_with_cp_lns(
        start_timetable=start_timetable,
        data=data,
        selected_subjects=selected_subjects,
        time_limit=args.time_limit,
        workers=args.workers,
        random_seed=args.random_seed,
        linearization_level=args.linearization_level,
        objective_mode=args.objective_mode,
        use_hint=args.hint,
        decision_strategy=args.decision_strategy,
        log_search_progress=args.log_search_progress,
    )
    elapsed = time.perf_counter() - started

    result["timetable"].to_csv(args.output, index=False)
    args.log_output.write_text(result["log_text"], encoding="utf-8")

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
        "status": result["status_name"],
        "status_code": result["status_code"],
        "best_bound": result["best_bound"],
        "gap": result["gap"],
        "branches": result["branches"],
        "conflicts": result["conflicts"],
        "wall_time_seconds": result["wall_time_seconds"],
        "elapsed_seconds": elapsed,
        "pattern_count": result["pattern_count"],
        "mu_count": result["mu_count"],
        "incompatible_mu_count": result["incompatible_mu_count"],
        "workers": args.workers,
        "random_seed": args.random_seed,
        "linearization_level": args.linearization_level,
        "hint": args.hint,
        "decision_strategy": args.decision_strategy,
        "selection": args.selection,
        "interaction_index": args.interaction_index,
    }
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print("Selected subjects:", ", ".join(selected_subjects))
    print("Start objective:", result["start_objective"])
    print("Candidate objective:", result["candidate_objective"])
    print("Solver candidate objective:", result["solver_candidate_objective"])
    print("Accepted:", result["accepted"])
    print("Improvement:", result["start_objective"] - result["candidate_objective"])
    print("Status:", result["status_name"])
    print("Best bound:", result["best_bound"])
    print("Gap:", result["gap"])
    print("Branches:", result["branches"])
    print("Conflicts:", result["conflicts"])
    print("Patterns:", result["pattern_count"])
    print("Mu variables:", result["mu_count"])
    print("Incompatible mu variables fixed to zero:", result["incompatible_mu_count"])
    print("Elapsed seconds:", round(elapsed, 6))
    print(f"Saved timetable to {args.output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved log to {args.log_output}")


def improve_with_cp_lns(
    *,
    start_timetable: pd.DataFrame,
    data: Any,
    selected_subjects: list[str],
    time_limit: float,
    workers: int,
    random_seed: int,
    linearization_level: int,
    objective_mode: str,
    use_hint: bool,
    decision_strategy: str,
    log_search_progress: bool,
) -> dict[str, Any]:
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

    patterns = _generate_patterns(
        selected_subjects=selected_subjects,
        blocks=blocks,
        data=data,
        fixed_placement=fixed_placement,
        full_placement=full_placement,
        exam_lengths=exam_lengths,
        exam_subjects=exam_subjects,
        pair_values=pair_values,
        day_index=day_index,
        objective_mode=objective_mode,
        weights=weights,
        first_half_subjects=first_half_subjects,
    )

    fixed_timetable = timetable_from_placement(fixed_placement, data.days)
    fixed_objective = mip_objective_value(fixed_timetable, data.pairs, data.days, mode=objective_mode)
    fixed_same_slot = _same_slot_clashes(fixed_placement, data.pairs, pair_values=pair_values)
    start_objective = mip_objective_value(start_timetable, data.pairs, data.days, mode=objective_mode)
    start_same_slot = _same_slot_clashes(full_placement, data.pairs, pair_values=pair_values)

    model = cp_model.CpModel()
    lam: dict[tuple[str, int], cp_model.IntVar] = {}
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            lam[(subject, pattern.index)] = model.NewBoolVar(f"lambda[{_name(subject)},{pattern.index}]")

    mu: dict[tuple[str, int, str, int], cp_model.IntVar] = {}
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
                    var = model.NewBoolVar(
                        f"mu[{_name(left_subject)},{left_pattern.index},{_name(right_subject)},{right_pattern.index}]"
                    )
                    mu[key] = var
                    if cost.daily_length_infeasible:
                        model.Add(var == 0)
                        incompatible_mu_count += 1

    current_pattern: dict[str, int] = {}
    for subject, subject_patterns in patterns.items():
        model.Add(sum(lam[(subject, pattern.index)] for pattern in subject_patterns) == 1)
        current_assignment = {exam: full_placement[exam] for exam in blocks[subject]["exams"]}
        current_signature = _assignment_signature(current_assignment)
        for pattern in subject_patterns:
            if _assignment_signature(pattern.assignment) == current_signature:
                current_pattern[subject] = pattern.index

    for left_pos, left_subject in enumerate(selected_subjects):
        for right_subject in selected_subjects[left_pos + 1 :]:
            left_patterns = patterns[left_subject]
            right_patterns = patterns[right_subject]
            for left_pattern in left_patterns:
                model.Add(
                    sum(mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)] for right_pattern in right_patterns)
                    == lam[(left_subject, left_pattern.index)]
                )
            for right_pattern in right_patterns:
                model.Add(
                    sum(mu[(left_subject, left_pattern.index, right_subject, right_pattern.index)] for left_pattern in left_patterns)
                    == lam[(right_subject, right_pattern.index)]
                )

    objective_terms: list[cp_model.LinearExpr] = []
    same_slot_terms: list[cp_model.LinearExpr] = [_as_int(fixed_same_slot, "fixed same-slot clashes")]
    objective_constant = _as_int(fixed_objective, "fixed objective")
    for subject, subject_patterns in patterns.items():
        for pattern in subject_patterns:
            var = lam[(subject, pattern.index)]
            delta = _as_int(pattern.fixed_objective_delta, "fixed objective delta")
            clashes = _as_int(pattern.fixed_same_slot_delta, "fixed same-slot delta")
            if delta:
                objective_terms.append(delta * var)
            if clashes:
                same_slot_terms.append(clashes * var)
    for key, var in mu.items():
        cost = cross_costs[key]
        objective = _as_int(cost.objective, "cross objective")
        clashes = _as_int(cost.same_slot_clashes, "cross same-slot clashes")
        if objective:
            objective_terms.append(objective * var)
        if clashes:
            same_slot_terms.append(clashes * var)

    model.Add(sum(same_slot_terms) <= 10_000)
    model.Minimize(objective_constant + sum(objective_terms))

    if use_hint:
        for subject, subject_patterns in patterns.items():
            selected_index = current_pattern.get(subject)
            for pattern in subject_patterns:
                model.AddHint(lam[(subject, pattern.index)], int(pattern.index == selected_index))
        for key, var in mu.items():
            left_subject, left_index, right_subject, right_index = key
            selected = current_pattern.get(left_subject) == left_index and current_pattern.get(right_subject) == right_index
            model.AddHint(var, int(selected))

    if decision_strategy == "pattern_cost":
        ordered = [
            lam[(subject, pattern.index)]
            for subject in selected_subjects
            for pattern in sorted(patterns[subject], key=lambda p: (p.fixed_objective_delta, p.index))
        ]
        model.AddDecisionStrategy(ordered, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
    elif decision_strategy != "none":
        raise ValueError(f"Unknown decision strategy: {decision_strategy}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.linearization_level = int(linearization_level)
    solver.parameters.log_search_progress = bool(log_search_progress)
    solver.parameters.log_to_stdout = False
    log_lines: list[str] = []
    if hasattr(solver, "log_callback"):
        solver.log_callback = log_lines.append

    status_code = solver.Solve(model)
    has_solution = status_code in {cp_model.OPTIMAL, cp_model.FEASIBLE}

    if has_solution:
        candidate_placement = dict(fixed_placement)
        for subject, subject_patterns in patterns.items():
            selected = [
                pattern
                for pattern in subject_patterns
                if solver.BooleanValue(lam[(subject, pattern.index)])
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Expected one selected pattern for {subject!r}, got {len(selected)}.")
            candidate_placement.update(selected[0].assignment)
        solver_candidate_timetable = timetable_from_placement(candidate_placement, data.days)
        validate_full_solution(solver_candidate_timetable, data)
        solver_candidate_objective = mip_objective_value(solver_candidate_timetable, data.pairs, data.days, mode=objective_mode)
        solver_candidate_same_slot = _same_slot_clashes(candidate_placement, data.pairs, pair_values=pair_values)
    else:
        solver_candidate_timetable = start_timetable
        solver_candidate_objective = start_objective
        solver_candidate_same_slot = start_same_slot

    accepted = solver_candidate_objective <= start_objective + 1e-9
    if accepted:
        output_timetable = solver_candidate_timetable
        output_objective = solver_candidate_objective
        output_same_slot = solver_candidate_same_slot
    else:
        output_timetable = start_timetable
        output_objective = start_objective
        output_same_slot = start_same_slot

    best_bound = float(solver.BestObjectiveBound())
    gap = None if not has_solution else abs(float(solver.ObjectiveValue()) - best_bound) / max(1.0, abs(float(solver.ObjectiveValue())))

    return {
        "timetable": output_timetable,
        "start_objective": start_objective,
        "candidate_objective": output_objective,
        "solver_candidate_objective": solver_candidate_objective,
        "accepted": accepted,
        "start_same_slot_clashes": start_same_slot,
        "candidate_same_slot_clashes": output_same_slot,
        "status_code": int(status_code),
        "status_name": solver.StatusName(status_code),
        "best_bound": best_bound,
        "gap": gap,
        "branches": int(solver.NumBranches()),
        "conflicts": int(solver.NumConflicts()),
        "wall_time_seconds": float(solver.WallTime()),
        "pattern_count": sum(len(subject_patterns) for subject_patterns in patterns.values()),
        "mu_count": len(mu),
        "incompatible_mu_count": incompatible_mu_count,
        "log_text": "".join(log_lines),
    }


def _generate_patterns(
    *,
    selected_subjects: list[str],
    blocks: dict[str, dict[str, Any]],
    data: Any,
    fixed_placement: dict[str, tuple[pd.Timestamp, str]],
    full_placement: dict[str, tuple[pd.Timestamp, str]],
    exam_lengths: dict[str, float],
    exam_subjects: dict[str, str],
    pair_values: dict[tuple[str, str], float],
    day_index: dict[pd.Timestamp, int],
    objective_mode: str,
    weights: dict[str, float],
    first_half_subjects: set[str],
) -> dict[str, list[Pattern]]:
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

        current_assignment = {exam: full_placement[exam] for exam in blocks[subject]["exams"]}
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
    return patterns


def _select_subjects(
    *,
    timetable: pd.DataFrame,
    data: Any,
    objective_mode: str,
    selection: str,
    subjects: int,
    interaction_index: int,
) -> list[str]:
    if selection == "contribution":
        return _subject_contribution_order(timetable, data=data, objective_mode=objective_mode)[:subjects]
    if selection == "interaction":
        neighborhoods = _interaction_neighborhoods(
            timetable,
            data=data,
            objective_mode=objective_mode,
            size=subjects,
            limit=max(1, interaction_index + 1),
        )
        if not neighborhoods:
            raise ValueError("No interaction neighborhoods were generated.")
        return neighborhoods[min(interaction_index, len(neighborhoods) - 1)]
    raise ValueError(f"Unknown selection method: {selection}")


def _as_int(value: float, label: str) -> int:
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1e-6:
        raise ValueError(f"CP-SAT requires integer {label}; got {value!r}.")
    return int(rounded)


def _name(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text)


if __name__ == "__main__":
    main()
