from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any

import pandas as pd

from src.anthony_model import (
    DEFAULT_SLOTS,
    mip_objective_value,
    prepare_anthony_model_data,
    timetable_from_placement,
)


@dataclass(frozen=True)
class FullHeuristicResult:
    """Result returned by :func:`solve_full_heuristic`."""

    timetable: pd.DataFrame
    objective_value: float
    placement: dict[str, tuple[pd.Timestamp, str]]
    block_choice: dict[str, dict[str, Any]]
    data: Any


def solve_full_heuristic(
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    nb_days: int = 23,
    slots: tuple[str, str] = DEFAULT_SLOTS,
    max_rounds: int = 20,
    max_clashes: float | None = 10_000,
    max_afternoon_minutes: int = 180,
    max_daily_minutes: int = 385,
    first_half_subjects: set[str] | None = None,
    objective_mode: str = "formal",
    clash_cap_penalty: float = 10_000.0,
) -> FullHeuristicResult:
    """
    Build a feasible full-instance timetable with greedy construction and local search.

    The heuristic uses the same prepared data conventions as the MIP model,
    including the Mathematics Paper 3 subject recoding.
    """
    data = prepare_anthony_model_data(
        exams=exams,
        days=days,
        pairs=pairs,
        nb_days=nb_days,
        slots=slots,
        recode_math_paper_three=True,
    )
    if first_half_subjects is None:
        first_half_subjects = {
            "BUS MAN",
            "BUSINESS MANAGEMENT",
            "HISTORY",
            "ENGLISH A LAL",
        }

    exam_lengths = dict(zip(data.exams["Full Name"], data.exams["Length"]))
    exam_subjects = dict(zip(data.exams["Full Name"], data.exams["Subject"]))
    candidate_blocks = _build_blocks(data.exams)
    candidates = {
        block_id: _generate_block_candidates(
            block_id=block_id,
            block=block,
            dates=data.dates,
            days=data.days,
            slots=slots,
            exam_lengths=exam_lengths,
            exam_subjects=exam_subjects,
            max_afternoon_minutes=max_afternoon_minutes,
            first_half_subjects=first_half_subjects,
        )
        for block_id, block in candidate_blocks.items()
    }

    empty_blocks = [block_id for block_id, block_candidates in candidates.items() if not block_candidates]
    if empty_blocks:
        raise ValueError(f"No feasible candidates for blocks: {empty_blocks}")

    order = sorted(
        candidate_blocks,
        key=lambda block_id: (
            len(candidates[block_id]),
            -_block_conflict_mass(candidate_blocks[block_id], data.pairs),
            block_id,
        ),
    )

    block_choice: dict[str, dict[str, Any]] = {}
    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for block_id in order:
        best_candidate = min(
            candidates[block_id],
            key=lambda candidate: _candidate_score(
                candidate=candidate,
                partial_placement=placement,
                pairs=data.pairs,
                days=data.days,
                exam_lengths=exam_lengths,
                max_daily_minutes=max_daily_minutes,
                max_clashes=max_clashes,
                objective_mode=objective_mode,
                clash_cap_penalty=clash_cap_penalty,
            ),
        )
        if _candidate_score(
            candidate=best_candidate,
            partial_placement=placement,
            pairs=data.pairs,
            days=data.days,
            exam_lengths=exam_lengths,
            max_daily_minutes=max_daily_minutes,
            max_clashes=max_clashes,
            objective_mode=objective_mode,
            clash_cap_penalty=clash_cap_penalty,
        ) == float("inf"):
            raise ValueError(f"Could not place block {block_id!r} without violating hard limits.")
        block_choice[block_id] = best_candidate
        placement.update(best_candidate["assignment"])

    block_choice, placement = _local_search(
        block_choice=block_choice,
        candidates=candidates,
        order=order,
        pairs=data.pairs,
        days=data.days,
        exam_lengths=exam_lengths,
        max_daily_minutes=max_daily_minutes,
        max_clashes=max_clashes,
        objective_mode=objective_mode,
        clash_cap_penalty=clash_cap_penalty,
        max_rounds=max_rounds,
    )

    timetable = timetable_from_placement(placement, data.days)
    objective_value = mip_objective_value(timetable, data.pairs, data.days, mode=objective_mode)
    return FullHeuristicResult(
        timetable=timetable,
        objective_value=objective_value,
        placement=placement,
        block_choice=block_choice,
        data=data,
    )


def validate_full_solution(
    timetable: pd.DataFrame,
    data: Any,
    *,
    max_clashes: float | None = 10_000,
    max_afternoon_minutes: int = 180,
    max_daily_minutes: int = 385,
    first_half_subjects: set[str] | None = None,
) -> dict[str, float]:
    """Validate the full-instance hard constraints used by the heuristic."""
    if first_half_subjects is None:
        first_half_subjects = {
            "BUS MAN",
            "BUSINESS MANAGEMENT",
            "HISTORY",
            "ENGLISH A LAL",
        }

    tt = timetable.copy()
    tt["Date"] = pd.to_datetime(tt["Date"], dayfirst=True).dt.normalize()
    placement = {row.Exam_Name: (row.Date, row.Slot) for row in tt.itertuples(index=False)}
    exam_lengths = dict(zip(data.exams["Full Name"], data.exams["Length"]))
    exam_subjects = dict(zip(data.exams["Full Name"], data.exams["Subject"]))

    if set(placement) != set(data.exam_names):
        missing = set(data.exam_names) - set(placement)
        extra = set(placement) - set(data.exam_names)
        raise ValueError(f"Timetable exam mismatch. Missing={sorted(missing)}, extra={sorted(extra)}")

    forbidden_dates = set(data.days.loc[data.days["DOW"].isin(["Sat", "Sun"]), "Date"])
    forbidden_dates.update(data.days.loc[(data.days["Date"].dt.month == 5) & (data.days["Date"].dt.day == 1), "Date"])
    if set(tt["Date"]).intersection(forbidden_dates):
        raise ValueError("Timetable contains exams on a weekend or May 1.")

    first_day = data.dates[0]
    first_two_days = set(data.dates[:2])
    if placement.get("SBS PAPER ONE") != (first_day, "AM"):
        raise ValueError("SBS PAPER ONE is not fixed on the first morning.")
    for exam, (date, _) in placement.items():
        if exam_subjects[exam] != "SBS" and date in first_two_days:
            raise ValueError(f"Non-SBS exam {exam!r} is placed in the opening SBS-only window.")

    for exam, (date, slot) in placement.items():
        if slot == "PM" and float(exam_lengths[exam]) > max_afternoon_minutes:
            raise ValueError(f"Long exam {exam!r} is placed in the afternoon.")
        if _is_language_subject(exam_subjects[exam]) and pd.Timestamp(date).day_name() == "Friday":
            raise ValueError(f"Language exam {exam!r} is placed on a Friday.")

    second_half_dates = set(data.dates[round(len(data.dates) / 2) :])
    for exam, (date, _) in placement.items():
        if exam_subjects[exam] in first_half_subjects and date in second_half_dates:
            raise ValueError(f"First-half subject exam {exam!r} is in the second half.")

    for subject, subject_df in data.exams.groupby("Subject"):
        subject_exams = subject_df["Full Name"].tolist()
        if len(subject_exams) < 2:
            continue
        subject_positions = sorted(data.dates.index(placement[exam][0]) for exam in subject_exams)
        if subject_positions[-1] - subject_positions[0] != len(subject_positions) - 1:
            raise ValueError(f"Subject {subject!r} is not placed on consecutive calendar days.")

    same_slot_clashes = 0.0
    exams = list(placement)
    for idx, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[idx + 1 :]:
            date_j, slot_j = placement[exam_j]
            same_day = date_i == date_j
            if same_day and float(exam_lengths[exam_i]) + float(exam_lengths[exam_j]) > max_daily_minutes:
                raise ValueError(f"Daily length limit violated by {exam_i!r} and {exam_j!r}.")
            if same_day and slot_i == slot_j:
                same_slot_clashes += float(data.pairs.loc[exam_i, exam_j])

    if max_clashes is not None and same_slot_clashes > max_clashes:
        raise ValueError(f"Same-slot clashes {same_slot_clashes} exceed {max_clashes}.")

    return {
        "same_slot_clashes": same_slot_clashes,
        "objective_formal": mip_objective_value(timetable, data.pairs, data.days, mode="formal"),
    }


def _build_blocks(exams: pd.DataFrame) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for subject, subject_df in exams.groupby("Subject", sort=False):
        subject_exams = subject_df["Full Name"].tolist()
        blocks[str(subject)] = {"subject": str(subject), "exams": subject_exams}
    return blocks


def _generate_block_candidates(
    *,
    block_id: str,
    block: dict[str, Any],
    dates: list[pd.Timestamp],
    days: pd.DataFrame,
    slots: tuple[str, str],
    exam_lengths: dict[str, float],
    exam_subjects: dict[str, str],
    max_afternoon_minutes: int,
    first_half_subjects: set[str],
) -> list[dict[str, Any]]:
    exams = block["exams"]
    subject = block["subject"]
    n_exams = len(exams)
    if n_exams > 3:
        raise ValueError(f"Block {block_id!r} has {n_exams} exams; this heuristic supports up to 3.")

    forbidden_dates = set(days.loc[days["DOW"].isin(["Sat", "Sun"]), "Date"])
    forbidden_dates.update(days.loc[(days["Date"].dt.month == 5) & (days["Date"].dt.day == 1), "Date"])
    second_half_dates = set(dates[round(len(dates) / 2) :])
    first_two_days = set(dates[:2])

    candidates: list[dict[str, Any]] = []
    for start_idx in range(len(dates) - n_exams + 1):
        block_dates = dates[start_idx : start_idx + n_exams]
        if any(date in forbidden_dates for date in block_dates):
            continue

        if subject == "SBS":
            if block_dates[0] != dates[0]:
                continue
        elif any(date in first_two_days for date in block_dates):
            continue

        if subject in first_half_subjects and any(date in second_half_dates for date in block_dates):
            continue

        for exam_order in _exam_orders(exams, subject):
            slot_options: list[list[str]] = []
            feasible = True
            for exam, date in zip(exam_order, block_dates):
                allowed = _allowed_slots(
                    exam=exam,
                    subject=exam_subjects[exam],
                    date=date,
                    length=float(exam_lengths[exam]),
                    slots=slots,
                    max_afternoon_minutes=max_afternoon_minutes,
                )
                if subject == "SBS" and exam == "SBS PAPER ONE" and date == dates[0]:
                    allowed = ["AM"] if "AM" in allowed else []
                if not allowed:
                    feasible = False
                    break
                slot_options.append(allowed)
            if not feasible:
                continue

            for chosen_slots in product(*slot_options):
                assignment = {
                    exam: (date, slot)
                    for exam, date, slot in zip(exam_order, block_dates, chosen_slots)
                }
                candidates.append(
                    {
                        "block_id": block_id,
                        "subject": subject,
                        "assignment": assignment,
                    }
                )

    return candidates


def _exam_orders(exams: list[str], subject: str) -> list[tuple[str, ...]]:
    if subject == "SBS" and "SBS PAPER ONE" in exams:
        rest = [exam for exam in exams if exam != "SBS PAPER ONE"]
        return [(tuple(["SBS PAPER ONE", *order])) for order in permutations(rest)]
    return list(permutations(exams))


def _allowed_slots(
    *,
    exam: str,
    subject: str,
    date: pd.Timestamp,
    length: float,
    slots: tuple[str, str],
    max_afternoon_minutes: int,
) -> list[str]:
    allowed = list(slots)
    if length > max_afternoon_minutes:
        allowed = [slot for slot in allowed if slot != "PM"]
    if _is_language_subject(subject) and pd.Timestamp(date).day_name() == "Friday":
        return []
    return allowed


def _is_language_subject(subject: str) -> bool:
    return subject in {
        "LANG LIT",
        "LANG ACQ",
        "LIT",
        "LANGUAGE A LITERATURE",
        "LANGUAGE ACQUISITION",
    }


def _block_conflict_mass(block: dict[str, Any], pairs: pd.DataFrame) -> float:
    return float(sum(pairs.loc[exam].sum() for exam in block["exams"]))


def _candidate_score(
    *,
    candidate: dict[str, Any],
    partial_placement: dict[str, tuple[pd.Timestamp, str]],
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    exam_lengths: dict[str, float],
    max_daily_minutes: int,
    max_clashes: float | None,
    objective_mode: str,
    clash_cap_penalty: float,
) -> float:
    placement = dict(partial_placement)
    placement.update(candidate["assignment"])
    if _violates_pair_limits(
        placement,
        pairs=pairs,
        exam_lengths=exam_lengths,
        max_daily_minutes=max_daily_minutes,
    ):
        return float("inf")
    timetable = timetable_from_placement(placement, days)
    value = mip_objective_value(timetable, pairs, days, mode=objective_mode)
    if max_clashes is not None:
        value += clash_cap_penalty * max(0.0, _same_slot_clashes(placement, pairs) - max_clashes)
    return value


def _violates_pair_limits(
    placement: dict[str, tuple[pd.Timestamp, str]],
    *,
    pairs: pd.DataFrame,
    exam_lengths: dict[str, float],
    max_daily_minutes: int,
) -> bool:
    exams = list(placement)
    for idx, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[idx + 1 :]:
            date_j, slot_j = placement[exam_j]
            same_day = date_i == date_j
            if same_day and float(exam_lengths[exam_i]) + float(exam_lengths[exam_j]) > max_daily_minutes:
                return True
    return False


def _same_slot_clashes(
    placement: dict[str, tuple[pd.Timestamp, str]],
    pairs: pd.DataFrame,
) -> float:
    total = 0.0
    exams = list(placement)
    for idx, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[idx + 1 :]:
            date_j, slot_j = placement[exam_j]
            if date_i == date_j and slot_i == slot_j:
                total += float(pairs.loc[exam_i, exam_j])
    return total


def _build_placement(block_choice: dict[str, dict[str, Any]]) -> dict[str, tuple[pd.Timestamp, str]]:
    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for candidate in block_choice.values():
        placement.update(candidate["assignment"])
    return placement


def _placement_value(
    placement: dict[str, tuple[pd.Timestamp, str]],
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    objective_mode: str,
    clash_cap_penalty: float,
) -> float:
    timetable = timetable_from_placement(placement, days)
    value = mip_objective_value(timetable, pairs, days, mode=objective_mode)
    return value


def _local_search(
    *,
    block_choice: dict[str, dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    order: list[str],
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    exam_lengths: dict[str, float],
    max_daily_minutes: int,
    max_clashes: float | None,
    objective_mode: str,
    clash_cap_penalty: float,
    max_rounds: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[pd.Timestamp, str]]]:
    block_choice = dict(block_choice)
    current_placement = _build_placement(block_choice)
    current_value = _candidate_score(
        candidate={"assignment": {}},
        partial_placement=current_placement,
        pairs=pairs,
        days=days,
        exam_lengths=exam_lengths,
        max_daily_minutes=max_daily_minutes,
        max_clashes=max_clashes,
        objective_mode=objective_mode,
        clash_cap_penalty=clash_cap_penalty,
    )

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        for block_id in order:
            original_candidate = block_choice[block_id]
            best_candidate = original_candidate
            best_value = current_value

            base_choice = dict(block_choice)
            base_choice.pop(block_id)
            base_placement = _build_placement(base_choice)

            for candidate in candidates[block_id]:
                trial_placement = dict(base_placement)
                trial_placement.update(candidate["assignment"])
                if _violates_pair_limits(
                    trial_placement,
                    pairs=pairs,
                    exam_lengths=exam_lengths,
                    max_daily_minutes=max_daily_minutes,
                ):
                    continue
                trial_value = _candidate_score(
                    candidate={"assignment": {}},
                    partial_placement=trial_placement,
                    pairs=pairs,
                    days=days,
                    exam_lengths=exam_lengths,
                    max_daily_minutes=max_daily_minutes,
                    max_clashes=max_clashes,
                    objective_mode=objective_mode,
                    clash_cap_penalty=clash_cap_penalty,
                )
                if trial_value + 1e-9 < best_value:
                    best_candidate = candidate
                    best_value = trial_value

            if best_candidate is not original_candidate:
                block_choice[block_id] = best_candidate
                current_value = best_value
                current_placement = _build_placement(block_choice)
                improved = True

    return block_choice, current_placement
