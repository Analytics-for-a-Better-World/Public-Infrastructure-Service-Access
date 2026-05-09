from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.anthony_model import (
    AnthonyModel,
    apply_timetable_start,
    build_anthony_mip_model,
    _exam_order_key,
    mip_objective_value,
    placement_from_timetable,
    prepare_anthony_model_data,
    timetable_from_solution,
)
from src.full_heuristic import validate_full_solution


@dataclass(frozen=True)
class LnsIteration:
    """Summary of one large-neighborhood-search iteration."""

    iteration: int
    strategy: str
    neighborhood_size: int
    fix_mode: str
    selected_subjects: list[str]
    start_objective: float
    candidate_objective: float | None
    start_spread_score: float
    candidate_spread_score: float | None
    accepted: bool
    solver_status: int
    solver_gap: float | None


@dataclass(frozen=True)
class LnsResult:
    """Result returned by :func:`improve_with_lns`."""

    timetable: pd.DataFrame
    objective_value: float
    history: list[LnsIteration]
    data: Any


def improve_with_lns(
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    start_timetable: pd.DataFrame,
    *,
    iterations: int = 5,
    subjects_per_iteration: int = 6,
    time_limit_per_iteration: float = 60.0,
    neighborhood_sizes: list[int] | None = None,
    strategy_cycle: list[str] | None = None,
    fix_mode_cycle: list[str] | None = None,
    solution_pool_size: int = 5,
    adaptive_time: bool = True,
    nb_days: int = 23,
    objective_mode: str = "formal",
    max_clashes: int | None = 10_000,
    max_afternoon_minutes: int = 180,
    max_daily_minutes: int = 385,
    first_half_subjects: set[str] | None = None,
    load_acceptance_tolerance: float = 0.0,
    max_spread_regression: float | None = None,
    target_max_day_exams: int = 7,
    target_max_slot_exams: int = 4,
    enforce_subject_exam_order: bool = False,
    y_binary: bool = False,
    symmetry: int | None = None,
    output_flag: int = 0,
) -> LnsResult:
    """
    Improve a full-instance timetable by reoptimizing small MILP neighborhoods.

    Each iteration selects high-contribution subjects, fixes all other exams to
    their current placements, and lets Gurobi reoptimize only the selected
    subjects for a short time limit.
    """
    if neighborhood_sizes is None:
        neighborhood_sizes = [
            max(4, subjects_per_iteration - 2),
            subjects_per_iteration,
            subjects_per_iteration + 4,
        ]
    if strategy_cycle is None:
        strategy_cycle = [
            "worst_subjects",
            "worst_pairs",
            "same_slot_clashes",
            "crowded_days",
            "date_window",
            "load_aware_date_window",
            "conflict_neighbors",
            "same_slot_clashes",
        ]
    if fix_mode_cycle is None:
        fix_mode_cycle = [
            "exact",
            "date_window_slots_free",
            "selected_days_slots_free",
        ]

    if first_half_subjects is None:
        first_half_subjects = {
            "BUSINESS MANAGEMENT",
            "HISTORY",
            "ENGLISH A LAL",
        }

    data = prepare_anthony_model_data(
        exams=exams,
        days=days,
        pairs=pairs,
        nb_days=nb_days,
        recode_math_paper_three=True,
    )

    current = _normalize_timetable(start_timetable)
    if enforce_subject_exam_order:
        current = _repair_subject_exam_order(current, data)
    validate_full_solution(
        current,
        data,
        max_clashes=max_clashes,
        max_afternoon_minutes=max_afternoon_minutes,
        max_daily_minutes=max_daily_minutes,
        first_half_subjects=first_half_subjects,
    )
    current_objective = mip_objective_value(current, data.pairs, data.days, mode=objective_mode)

    history: list[LnsIteration] = []
    recently_selected: list[str] = []
    incumbent_pool: list[tuple[float, pd.DataFrame]] = [(current_objective, current)]
    no_improvement_streak = 0

    for iteration in range(1, iterations + 1):
        start_objective = current_objective
        start_spread_score = _spread_score(
            current,
            target_max_day_exams=target_max_day_exams,
            target_max_slot_exams=target_max_slot_exams,
        )
        strategy = strategy_cycle[(iteration - 1) % len(strategy_cycle)]
        neighborhood_size = neighborhood_sizes[(iteration - 1) % len(neighborhood_sizes)]
        fix_mode = fix_mode_cycle[(iteration - 1) % len(fix_mode_cycle)]
        if strategy in {"same_slot_clashes", "load_aware_date_window"}:
            neighborhood_size = max(neighborhood_size, subjects_per_iteration + 4)
        if no_improvement_streak >= 2:
            neighborhood_size = min(len(data.exams["Subject"].unique()), neighborhood_size + 2)

        selected_subjects = _select_subjects_for_neighborhood(
            timetable=current,
            data=data,
            subjects_per_iteration=neighborhood_size,
            objective_mode=objective_mode,
            recently_selected=recently_selected,
            strategy=strategy,
        )
        recently_selected = selected_subjects
        selected_dates = _dates_for_subjects(current, data, set(selected_subjects))
        iteration_time_limit = _adaptive_time_limit(
            base_time_limit=time_limit_per_iteration,
            strategy=strategy,
            no_improvement_streak=no_improvement_streak,
            adaptive_time=adaptive_time,
        )

        built = build_anthony_mip_model(
            exams=data.exams,
            days=data.days,
            pairs=data.pairs,
            nb_days=len(data.days),
            max_clashes=max_clashes,
            max_afternoon_minutes=max_afternoon_minutes,
            max_daily_minutes=max_daily_minutes,
            forbid_weekends=True,
            forbid_may_first=True,
            forbid_language_fridays=True,
            force_sbs_start=True,
            consecutive_subject_exams=True,
            consecutive_usable_subject_exams=False,
            first_half_subjects=first_half_subjects,
            recode_math_paper_three=False,
            y_binary=y_binary,
            enforce_subject_exam_order=enforce_subject_exam_order,
            objective_mode=objective_mode,
            output_flag=output_flag,
            model_name=f"full_lns_{iteration}",
        )
        apply_timetable_start(built, current)
        _fix_unselected_exams(
            built=built,
            current_timetable=current,
            selected_subjects=set(selected_subjects),
            selected_dates=selected_dates,
            fix_mode=fix_mode,
        )
        built.model.setParam("TimeLimit", iteration_time_limit)
        if symmetry is not None:
            built.model.setParam("Symmetry", symmetry)
        if solution_pool_size > 1:
            built.model.setParam("PoolSolutions", solution_pool_size)
            built.model.setParam("PoolSearchMode", 1)
        built.model.optimize()

        accepted = False
        candidate_objective: float | None = None
        candidate_spread_score: float | None = None
        candidate: pd.DataFrame | None = None
        if built.model.SolCount:
            candidate, candidate_objective = _best_solution_from_pool(
                built=built,
                data=data,
                objective_mode=objective_mode,
                start_objective=current_objective,
                start_spread_score=start_spread_score,
                load_acceptance_tolerance=load_acceptance_tolerance,
                target_max_day_exams=target_max_day_exams,
                target_max_slot_exams=target_max_slot_exams,
            )
            candidate_spread_score = _spread_score(
                candidate,
                target_max_day_exams=target_max_day_exams,
                target_max_slot_exams=target_max_slot_exams,
            )
            improves_primary = candidate_objective + 1e-9 < current_objective
            spread_regression_ok = (
                max_spread_regression is None
                or candidate_spread_score <= start_spread_score + max_spread_regression + 1e-9
            )
            improves_spread_within_budget = (
                load_acceptance_tolerance > 0.0
                and candidate_objective <= current_objective + load_acceptance_tolerance
                and candidate_spread_score + 1e-9 < start_spread_score
            )
            if (improves_primary and spread_regression_ok) or improves_spread_within_budget:
                validate_full_solution(
                    candidate,
                    data,
                    max_clashes=max_clashes,
                    max_afternoon_minutes=max_afternoon_minutes,
                    max_daily_minutes=max_daily_minutes,
                    first_half_subjects=first_half_subjects,
                )
                current = candidate
                current_objective = candidate_objective
                accepted = True
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
            if candidate is not None and candidate_objective is not None:
                incumbent_pool = _update_pool(
                    pool=incumbent_pool,
                    objective=candidate_objective,
                    timetable=candidate,
                    max_size=5,
                )
        else:
            no_improvement_streak += 1

        history.append(
            LnsIteration(
                iteration=iteration,
                strategy=strategy,
                neighborhood_size=neighborhood_size,
                fix_mode=fix_mode,
                selected_subjects=selected_subjects,
                start_objective=start_objective,
                candidate_objective=candidate_objective,
                start_spread_score=start_spread_score,
                candidate_spread_score=candidate_spread_score,
                accepted=accepted,
                solver_status=int(built.model.Status),
                solver_gap=float(built.model.MIPGap) if built.model.SolCount else None,
            )
        )

    best_objective, best_timetable = min(incumbent_pool, key=lambda item: item[0])
    return LnsResult(
        timetable=best_timetable,
        objective_value=best_objective,
        history=history,
        data=data,
    )


def _normalize_timetable(timetable: pd.DataFrame) -> pd.DataFrame:
    tt = timetable.copy()
    date_text = tt["Date"].astype(str)
    if date_text.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        tt["Date"] = pd.to_datetime(tt["Date"]).dt.normalize()
    else:
        tt["Date"] = pd.to_datetime(tt["Date"], dayfirst=True).dt.normalize()
    return tt[["Day_of_Week", "Date", "Slot", "Exam_Name"]].sort_values(
        ["Date", "Slot", "Exam_Name"]
    ).reset_index(drop=True)


def _repair_subject_exam_order(timetable: pd.DataFrame, data: Any) -> pd.DataFrame:
    """Assign earlier placements to earlier papers within each subject."""
    repaired = timetable.copy()
    day_lookup = data.days.set_index("Date")["DOW"].to_dict()
    for _, subject_exams_df in data.exams.groupby("Subject", sort=False):
        ordered_exams = sorted(subject_exams_df["Full Name"].tolist(), key=_exam_order_key)
        placements: list[tuple[pd.Timestamp, str]] = []
        present_exams: list[str] = []
        for exam in ordered_exams:
            row = repaired.loc[repaired["Exam_Name"] == exam]
            if row.empty:
                continue
            present_exams.append(exam)
            placements.append((pd.Timestamp(row.iloc[0]["Date"]).normalize(), str(row.iloc[0]["Slot"])))
        if len(placements) < 2:
            continue
        placements = sorted(placements, key=lambda item: (item[0], item[1]))
        for exam, (date, slot) in zip(present_exams, placements):
            mask = repaired["Exam_Name"] == exam
            repaired.loc[mask, "Date"] = date
            repaired.loc[mask, "Slot"] = slot
            repaired.loc[mask, "Day_of_Week"] = day_lookup.get(date, date.strftime("%a"))
    return _normalize_timetable(repaired)


def _fix_unselected_exams(
    *,
    built: AnthonyModel,
    current_timetable: pd.DataFrame,
    selected_subjects: set[str],
    selected_dates: set[pd.Timestamp],
    fix_mode: str,
) -> None:
    placement = placement_from_timetable(current_timetable)
    subject_by_exam = dict(zip(built.data.exams["Full Name"], built.data.exams["Subject"]))
    for exam, (date, slot) in placement.items():
        if subject_by_exam[exam] in selected_subjects:
            continue

        normalized_date = pd.Timestamp(date).normalize()
        if fix_mode == "date_window_slots_free" and normalized_date in selected_dates:
            _fix_exam_date(built, exam, normalized_date)
        elif fix_mode == "selected_days_slots_free" and _near_selected_date(normalized_date, selected_dates):
            _fix_exam_date(built, exam, normalized_date)
        else:
            var = built.x.get((exam, slot, normalized_date))
            if var is None:
                raise KeyError(f"Could not find x variable for fixed placement {(exam, slot, date)}")
            built.model.addConstr(var == 1.0, name=f"fix_start[{exam}]")
    built.model.update()


def _select_subjects_for_neighborhood(
    *,
    timetable: pd.DataFrame,
    data: Any,
    subjects_per_iteration: int,
    objective_mode: str,
    recently_selected: list[str],
    strategy: str,
) -> list[str]:
    if strategy == "worst_pairs":
        ordered = _subjects_from_worst_pairs(timetable, data, objective_mode=objective_mode)
    elif strategy == "same_slot_clashes":
        ordered = _subjects_from_same_slot_clashes(timetable, data)
    elif strategy == "crowded_days":
        ordered = _subjects_from_crowded_days(timetable, data, objective_mode=objective_mode)
    elif strategy == "date_window":
        ordered = _subjects_from_worst_date_window(timetable, data, objective_mode=objective_mode)
    elif strategy == "load_aware_date_window":
        ordered = _subjects_from_load_aware_date_window(
            timetable,
            data,
            objective_mode=objective_mode,
        )
    elif strategy == "conflict_neighbors":
        ordered = _subjects_from_conflict_neighbors(timetable, data, objective_mode=objective_mode)
    else:
        scores = _subject_contribution_scores(timetable, data, objective_mode=objective_mode)
        ordered = sorted(scores, key=lambda subject: (-scores[subject], subject))

    ordered = _append_missing_subjects(ordered, data)

    selected: list[str] = []
    for subject in ordered:
        if subject in recently_selected and len(ordered) > subjects_per_iteration:
            continue
        selected.append(subject)
        if len(selected) == subjects_per_iteration:
            return selected

    for subject in ordered:
        if subject not in selected:
            selected.append(subject)
        if len(selected) == subjects_per_iteration:
            break
    return selected


def _fix_exam_date(built: AnthonyModel, exam: str, date: pd.Timestamp) -> None:
    expr = None
    for slot in built.data.slots:
        var = built.x.get((exam, slot, date))
        if var is None:
            continue
        expr = var if expr is None else expr + var
    if expr is None:
        raise KeyError(f"Could not find x variables for {(exam, date)}")
    built.model.addConstr(expr == 1.0, name=f"fix_date[{exam},{date:%Y%m%d}]")


def _near_selected_date(date: pd.Timestamp, selected_dates: set[pd.Timestamp]) -> bool:
    return any(abs((date - selected_date).days) <= 1 for selected_date in selected_dates)


def _dates_for_subjects(
    timetable: pd.DataFrame,
    data: Any,
    selected_subjects: set[str],
) -> set[pd.Timestamp]:
    subject_by_exam = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    dates: set[pd.Timestamp] = set()
    for row in timetable.itertuples(index=False):
        if subject_by_exam[row.Exam_Name] in selected_subjects:
            dates.add(pd.Timestamp(row.Date).normalize())
    return dates


def _adaptive_time_limit(
    *,
    base_time_limit: float,
    strategy: str,
    no_improvement_streak: int,
    adaptive_time: bool,
) -> float:
    if not adaptive_time:
        return base_time_limit
    multiplier = 1.0
    if strategy in {"worst_pairs", "conflict_neighbors", "date_window"}:
        multiplier += 0.5
    if no_improvement_streak >= 2:
        multiplier += 0.5
    return base_time_limit * multiplier


def _best_solution_from_pool(
    *,
    built: AnthonyModel,
    data: Any,
    objective_mode: str,
    start_objective: float | None = None,
    start_spread_score: float | None = None,
    load_acceptance_tolerance: float = 0.0,
    target_max_day_exams: int = 7,
    target_max_slot_exams: int = 4,
) -> tuple[pd.DataFrame, float]:
    best_timetable = timetable_from_solution(built)
    best_objective = mip_objective_value(best_timetable, data.pairs, data.days, mode=objective_mode)
    best_key = _candidate_key(
        best_timetable,
        best_objective,
        start_objective=start_objective,
        start_spread_score=start_spread_score,
        load_acceptance_tolerance=load_acceptance_tolerance,
        target_max_day_exams=target_max_day_exams,
        target_max_slot_exams=target_max_slot_exams,
    )

    for solution_number in range(1, int(built.model.SolCount)):
        built.model.setParam("SolutionNumber", solution_number)
        candidate = _timetable_from_pool_solution(built)
        objective = mip_objective_value(candidate, data.pairs, data.days, mode=objective_mode)
        candidate_key = _candidate_key(
            candidate,
            objective,
            start_objective=start_objective,
            start_spread_score=start_spread_score,
            load_acceptance_tolerance=load_acceptance_tolerance,
            target_max_day_exams=target_max_day_exams,
            target_max_slot_exams=target_max_slot_exams,
        )
        if candidate_key < best_key:
            best_timetable = candidate
            best_objective = objective
            best_key = candidate_key

    built.model.setParam("SolutionNumber", 0)
    return best_timetable, best_objective


def _candidate_key(
    timetable: pd.DataFrame,
    objective: float,
    *,
    start_objective: float | None,
    start_spread_score: float | None,
    load_acceptance_tolerance: float,
    target_max_day_exams: int,
    target_max_slot_exams: int,
) -> tuple[int, float, float]:
    spread_score = _spread_score(
        timetable,
        target_max_day_exams=target_max_day_exams,
        target_max_slot_exams=target_max_slot_exams,
    )
    if (
        start_objective is not None
        and start_spread_score is not None
        and load_acceptance_tolerance > 0.0
        and objective <= start_objective + load_acceptance_tolerance
        and spread_score + 1e-9 < start_spread_score
    ):
        return (0, spread_score, objective)
    return (1, objective, spread_score)


def _timetable_from_pool_solution(built: AnthonyModel, *, threshold: float = 0.5) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    day_lookup = built.data.days.set_index("Date")["DOW"].to_dict()
    for (exam, slot, date), var in built.x.items():
        if var.Xn > threshold:
            records.append(
                {
                    "Day_of_Week": day_lookup.get(date),
                    "Date": date,
                    "Slot": slot,
                    "Exam_Name": exam,
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["Date", "Slot", "Exam_Name"])
        .reset_index(drop=True)
    )


def _update_pool(
    *,
    pool: list[tuple[float, pd.DataFrame]],
    objective: float,
    timetable: pd.DataFrame,
    max_size: int,
) -> list[tuple[float, pd.DataFrame]]:
    updated = [*pool, (objective, timetable)]
    unique: dict[float, pd.DataFrame] = {}
    for obj, tt in updated:
        unique.setdefault(round(obj, 6), tt)
    return sorted(((obj, tt) for obj, tt in unique.items()), key=lambda item: item[0])[:max_size]


def _subject_contribution_scores(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> dict[str, float]:
    placement = placement_from_timetable(timetable)
    subject_by_exam = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    day_index = {date: idx for idx, date in enumerate(data.days["Date"].tolist())}
    scores = {subject: 0.0 for subject in data.exams["Subject"].unique()}

    exams = sorted(placement)
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            cij = float(data.pairs.loc[exam_i, exam_j])
            if cij == 0:
                continue
            gap = abs(day_index[date_i] - day_index[date_j])
            penalty = _pair_penalty(cij, gap, slot_i == slot_j, objective_mode)
            if penalty == 0:
                continue
            scores[subject_by_exam[exam_i]] += penalty / 2.0
            scores[subject_by_exam[exam_j]] += penalty / 2.0
    return scores


def _subjects_from_worst_pairs(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> list[str]:
    pairs = _pair_contributions(timetable, data, objective_mode=objective_mode)
    selected: list[str] = []
    for _, exam_i, exam_j in pairs:
        for exam in (exam_i, exam_j):
            subject = _subject_for_exam(data, exam)
            if subject not in selected:
                selected.append(subject)
    return selected


def _subjects_from_same_slot_clashes(
    timetable: pd.DataFrame,
    data: Any,
) -> list[str]:
    placement = placement_from_timetable(timetable)
    subject_scores = {subject: 0.0 for subject in data.exams["Subject"].unique()}
    exams = sorted(placement)
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            if date_i != date_j or slot_i != slot_j:
                continue
            cij = float(data.pairs.loc[exam_i, exam_j])
            subject_scores[_subject_for_exam(data, exam_i)] += cij / 2.0
            subject_scores[_subject_for_exam(data, exam_j)] += cij / 2.0
    return sorted(subject_scores, key=lambda subject: (-subject_scores[subject], subject))


def _subjects_from_crowded_days(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> list[str]:
    day_scores = _date_contribution_scores(timetable, data, objective_mode=objective_mode)
    ordered_dates = sorted(day_scores, key=lambda date: (-day_scores[date], date))
    return _subjects_on_dates(timetable, data, ordered_dates[:3])


def _subjects_from_worst_date_window(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> list[str]:
    day_scores = _date_contribution_scores(timetable, data, objective_mode=objective_mode)
    dates = data.days["Date"].tolist()
    best_start = 0
    best_score = -1.0
    window_size = 4
    for start in range(0, max(1, len(dates) - window_size + 1)):
        window = dates[start : start + window_size]
        score = sum(day_scores.get(date, 0.0) for date in window)
        if score > best_score:
            best_score = score
            best_start = start
    return _subjects_on_dates(timetable, data, dates[best_start : best_start + window_size])


def _subjects_from_load_aware_date_window(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> list[str]:
    day_scores = _date_contribution_scores(timetable, data, objective_mode=objective_mode)
    day_loads = timetable.groupby("Date")["Exam_Name"].count().to_dict()
    slot_loads = timetable.groupby(["Date", "Slot"])["Exam_Name"].count().to_dict()
    dates = data.days["Date"].tolist()
    best_start = 0
    best_score = -1.0
    window_size = 5
    for start in range(0, max(1, len(dates) - window_size + 1)):
        window = dates[start : start + window_size]
        primary_score = sum(day_scores.get(date, 0.0) for date in window)
        load_score = 0.0
        for date in window:
            day_load = float(day_loads.get(date, 0))
            load_score += day_load * day_load
            for slot in data.slots:
                slot_load = float(slot_loads.get((date, slot), 0))
                load_score += 0.5 * slot_load * slot_load
        score = primary_score + 10_000.0 * load_score
        if score > best_score:
            best_score = score
            best_start = start
    return _subjects_on_dates(timetable, data, dates[best_start : best_start + window_size])


def _subjects_from_conflict_neighbors(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> list[str]:
    scores = _subject_contribution_scores(timetable, data, objective_mode=objective_mode)
    seed = min(scores, key=lambda subject: (-scores[subject], subject))
    subject_exams = data.exams.groupby("Subject")["Full Name"].apply(list).to_dict()
    neighbor_scores: dict[str, float] = {}
    for subject, exams in subject_exams.items():
        if subject == seed:
            continue
        total = 0.0
        for exam_i in subject_exams[seed]:
            for exam_j in exams:
                total += float(data.pairs.loc[exam_i, exam_j])
        neighbor_scores[subject] = total
    ordered = [seed]
    ordered.extend(sorted(neighbor_scores, key=lambda subject: (-neighbor_scores[subject], subject)))
    return ordered


def _pair_contributions(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> list[tuple[float, str, str]]:
    placement = placement_from_timetable(timetable)
    day_index = {date: idx for idx, date in enumerate(data.days["Date"].tolist())}
    contributions: list[tuple[float, str, str]] = []
    exams = sorted(placement)
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            cij = float(data.pairs.loc[exam_i, exam_j])
            if cij == 0:
                continue
            gap = abs(day_index[date_i] - day_index[date_j])
            penalty = _pair_penalty(cij, gap, slot_i == slot_j, objective_mode)
            if penalty > 0:
                contributions.append((penalty, exam_i, exam_j))
    return sorted(contributions, key=lambda item: (-item[0], item[1], item[2]))


def _date_contribution_scores(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str,
) -> dict[pd.Timestamp, float]:
    placement = placement_from_timetable(timetable)
    day_index = {date: idx for idx, date in enumerate(data.days["Date"].tolist())}
    scores = {date: 0.0 for date in data.days["Date"]}
    exams = sorted(placement)
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            cij = float(data.pairs.loc[exam_i, exam_j])
            if cij == 0:
                continue
            gap = abs(day_index[date_i] - day_index[date_j])
            penalty = _pair_penalty(cij, gap, slot_i == slot_j, objective_mode)
            if penalty == 0:
                continue
            scores[date_i] += penalty / 2.0
            scores[date_j] += penalty / 2.0
    return scores


def _subjects_on_dates(
    timetable: pd.DataFrame,
    data: Any,
    dates: list[pd.Timestamp],
) -> list[str]:
    date_set = {pd.Timestamp(date).normalize() for date in dates}
    subject_by_exam = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    subjects: list[str] = []
    for row in timetable.itertuples(index=False):
        if pd.Timestamp(row.Date).normalize() in date_set:
            subject = subject_by_exam[row.Exam_Name]
            if subject not in subjects:
                subjects.append(subject)
    return subjects


def _append_missing_subjects(subjects: list[str], data: Any) -> list[str]:
    result = list(subjects)
    for subject in sorted(data.exams["Subject"].unique()):
        if subject not in result:
            result.append(subject)
    return result


def _spread_score(
    timetable: pd.DataFrame,
    *,
    target_max_day_exams: int,
    target_max_slot_exams: int,
) -> float:
    day_loads = timetable.groupby("Date")["Exam_Name"].count()
    slot_loads = timetable.groupby(["Date", "Slot"])["Exam_Name"].count()
    day_overload = sum(max(0, int(load) - target_max_day_exams) ** 2 for load in day_loads)
    slot_overload = sum(max(0, int(load) - target_max_slot_exams) ** 2 for load in slot_loads)
    max_day = int(day_loads.max()) if not day_loads.empty else 0
    max_slot = int(slot_loads.max()) if not slot_loads.empty else 0
    return float(100 * day_overload + 100 * slot_overload + 10 * max_day + max_slot)


def _subject_for_exam(data: Any, exam: str) -> str:
    subject = data.exams.loc[data.exams["Full Name"] == exam, "Subject"]
    if subject.empty:
        raise KeyError(f"Unknown exam {exam!r}")
    return str(subject.iloc[0])


def _pair_penalty(cij: float, gap: int, same_slot: bool, objective_mode: str) -> float:
    weights = {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1}
    total = 0.0
    if gap == 0 and same_slot:
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
        if gap == 0 and not same_slot:
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
