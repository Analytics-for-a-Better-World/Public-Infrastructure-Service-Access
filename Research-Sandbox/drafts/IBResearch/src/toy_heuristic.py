from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from src.anthony_model import (
    DEFAULT_SLOTS,
    mip_objective_value,
    parse_day_series,
    prepare_toy_inputs,
    timetable_from_placement,
)


@dataclass(frozen=True)
class HeuristicResult:
    """Result returned by :func:`solve_toy_heuristic`."""

    timetable: pd.DataFrame
    objective_value: float
    placement: dict[str, tuple[pd.Timestamp, str]]
    block_choice: dict[str, dict[str, Any]]
    days: pd.DataFrame
    exams: pd.DataFrame
    pairs: pd.DataFrame


def solve_toy_heuristic(
    toy_exams: pd.DataFrame,
    toy_pairs: pd.DataFrame,
    *,
    toy_days: pd.DataFrame | None = None,
    year: int = 2026,
    max_rounds: int = 30,
    slots: tuple[str, str] = DEFAULT_SLOTS,
    max_daily_minutes: int = 375,
    max_clashes: float | None = 15,
    first_half_subjects: set[str] | None = None,
    objective_mode: str = "anthony_appendix",
) -> HeuristicResult:
    """
    Build a toy timetable with the notebook's greedy + local-search heuristic.

    The returned timetable has the same columns as the MIP output:
    ``Day_of_Week``, ``Date``, ``Slot``, and ``Exam_Name``.
    """
    exams, days, pairs = prepare_toy_inputs(toy_exams, toy_pairs, toy_days=toy_days, year=year)
    pairs = pairs.set_index(pairs.columns[0])
    pairs.index = pairs.index.astype(str).str.strip()
    pairs.columns = pairs.columns.astype(str).str.strip()
    pairs = pairs.apply(pd.to_numeric, errors="raise")
    dates = parse_day_series(days["Date"]).tolist()
    usable_days = _usable_days(days)
    if first_half_subjects is None:
        first_half_subjects = {"Finance", "Law and Ethics"}
    second_half_dates = set(parse_day_series(days["Date"]).tolist()[round(len(days) / 2) :])
    blocks = _build_subject_blocks(exams)
    candidates = {
        subject: _generate_candidates(
            block,
            dates=dates,
            usable_days=usable_days,
            slots=slots,
            first_day=usable_days[0],
            second_half_dates=second_half_dates,
            first_half_subjects=first_half_subjects,
        )
        for subject, block in blocks.items()
    }
    empty_subjects = [subject for subject, subject_candidates in candidates.items() if not subject_candidates]
    if empty_subjects:
        raise ValueError(f"No feasible heuristic candidates for subjects: {empty_subjects}")

    order = sorted(
        blocks,
        key=lambda subject: (-_block_difficulty(subject, blocks[subject], pairs), subject),
    )

    block_choice: dict[str, dict[str, Any]] = {}
    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for subject in order:
        best_candidate = min(
            candidates[subject],
            key=lambda candidate: _objective_for_candidate(
                candidate=candidate,
                partial_placement=placement,
                pairs=pairs,
                days=days,
                exam_lengths=dict(zip(exams["Full Name"], exams["Length"])),
                max_daily_minutes=max_daily_minutes,
                max_clashes=max_clashes,
                objective_mode=objective_mode,
            ),
        )
        block_choice[subject] = best_candidate
        placement.update(best_candidate["assignment"])

    block_choice, placement = _relocate_improve(
        block_choice=block_choice,
        candidates=candidates,
        order=order,
        pairs=pairs,
        days=days,
        exam_lengths=dict(zip(exams["Full Name"], exams["Length"])),
        max_daily_minutes=max_daily_minutes,
        max_clashes=max_clashes,
        objective_mode=objective_mode,
        max_rounds=max_rounds,
    )

    timetable = timetable_from_placement(placement, days)
    objective = mip_objective_value(timetable, pairs, days, mode=objective_mode)
    return HeuristicResult(
        timetable=timetable,
        objective_value=objective,
        placement=placement,
        block_choice=block_choice,
        days=days,
        exams=exams,
        pairs=pairs,
    )


def _usable_days(days: pd.DataFrame) -> list[pd.Timestamp]:
    dates = parse_day_series(days["Date"])
    is_weekend = dates.dt.dayofweek >= 5
    is_may_first = dates.dt.month.eq(5) & dates.dt.day.eq(1)
    return dates.loc[~(is_weekend | is_may_first)].tolist()


def _build_subject_blocks(exams: pd.DataFrame) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for subject, subject_df in exams.groupby("Subject", sort=False):
        subject_df = subject_df.sort_values("Exam")
        exam_names = subject_df["Full Name"].tolist()
        lengths = dict(zip(subject_df["Full Name"], subject_df["Length"]))
        blocks[str(subject)] = {
            "subject": str(subject),
            "exams": exam_names,
            "lengths": lengths,
        }
    return blocks


def _generate_candidates(
    block: dict[str, Any],
    *,
    dates: list[pd.Timestamp],
    usable_days: list[pd.Timestamp],
    slots: tuple[str, str],
    first_day: pd.Timestamp,
    second_half_dates: set[pd.Timestamp],
    first_half_subjects: set[str],
) -> list[dict[str, Any]]:
    subject = block["subject"]
    exams = block["exams"]
    if len(exams) != 2:
        raise ValueError(f"Toy heuristic expects exactly two exams per subject, got {subject!r}.")

    exam_1, exam_2 = exams
    length_1 = int(block["lengths"][exam_1])
    length_2 = int(block["lengths"][exam_2])
    candidates: list[dict[str, Any]] = []

    usable_day_set = set(usable_days)
    for day_idx in range(len(dates) - 1):
        day_1 = pd.Timestamp(dates[day_idx]).normalize()
        day_2 = pd.Timestamp(dates[day_idx + 1]).normalize()

        if day_1 not in usable_day_set or day_2 not in usable_day_set:
            continue

        if subject.upper() == "SBS" and day_1 != first_day:
            continue
        if day_1 == first_day and subject.upper() != "SBS":
            continue
        if subject in first_half_subjects and (day_1 in second_half_dates or day_2 in second_half_dates):
            continue

        sessions_1 = _allowed_sessions(subject, day_1, length_1, slots)
        sessions_2 = _allowed_sessions(subject, day_2, length_2, slots)
        for slot_1, slot_2 in product(sessions_1, sessions_2):
            candidates.append(
                {
                    "subject": subject,
                    "start_day": day_1,
                    "assignment": {
                        exam_1: (day_1, slot_1),
                        exam_2: (day_2, slot_2),
                    },
                }
            )
    return candidates


def _allowed_sessions(
    subject: str,
    date: pd.Timestamp,
    length_minutes: int,
    slots: tuple[str, str],
) -> list[str]:
    sessions = list(slots)
    if length_minutes > 180:
        sessions = [slot for slot in sessions if slot != "PM"]
    if subject.upper() == "LANGUAGE A LITERATURE" and pd.Timestamp(date).day_name() == "Friday":
        sessions = []
    return sessions


def _block_difficulty(subject: str, block: dict[str, Any], pairs: pd.DataFrame) -> float:
    total_conflict = sum(float(pairs.loc[exam].sum()) for exam in block["exams"])

    restrictiveness = 0
    if subject.upper() == "SBS":
        restrictiveness += 1
    if subject.upper() == "LANGUAGE A LITERATURE":
        restrictiveness += 1
    if any(int(minutes) > 135 for minutes in block["lengths"].values()):
        restrictiveness += 1

    return total_conflict + 3 * restrictiveness


def _objective_for_candidate(
    *,
    candidate: dict[str, Any],
    partial_placement: dict[str, tuple[pd.Timestamp, str]],
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    exam_lengths: dict[str, int],
    max_daily_minutes: int,
    max_clashes: float | None,
    objective_mode: str,
) -> float:
    placement = dict(partial_placement)
    placement.update(candidate["assignment"])
    if _violates_hard_pair_limits(
        placement,
        pairs=pairs,
        exam_lengths=exam_lengths,
        max_daily_minutes=max_daily_minutes,
        max_clashes=max_clashes,
    ):
        return float("inf")
    timetable = timetable_from_placement(placement, days)
    return mip_objective_value(timetable, pairs, days, mode=objective_mode)


def _build_placement_from_blocks(
    block_choice: dict[str, dict[str, Any]],
) -> dict[str, tuple[pd.Timestamp, str]]:
    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for candidate in block_choice.values():
        placement.update(candidate["assignment"])
    return placement


def _placement_objective(
    placement: dict[str, tuple[pd.Timestamp, str]],
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    objective_mode: str,
) -> float:
    timetable = timetable_from_placement(placement, days)
    return mip_objective_value(timetable, pairs, days, mode=objective_mode)


def _relocate_improve(
    *,
    block_choice: dict[str, dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    order: list[str],
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    exam_lengths: dict[str, int],
    max_daily_minutes: int,
    max_clashes: float | None,
    objective_mode: str,
    max_rounds: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[pd.Timestamp, str]]]:
    block_choice = dict(block_choice)
    current_placement = _build_placement_from_blocks(block_choice)
    current_cost = _placement_objective(current_placement, pairs, days, objective_mode)

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        for subject in order:
            original_candidate = block_choice[subject]
            best_candidate = original_candidate
            best_cost = current_cost

            trial_choice = dict(block_choice)
            trial_choice.pop(subject)
            base_placement = _build_placement_from_blocks(trial_choice)

            for candidate in candidates[subject]:
                trial_placement = dict(base_placement)
                trial_placement.update(candidate["assignment"])
                if _violates_hard_pair_limits(
                    trial_placement,
                    pairs=pairs,
                    exam_lengths=exam_lengths,
                    max_daily_minutes=max_daily_minutes,
                    max_clashes=max_clashes,
                ):
                    continue
                trial_cost = _placement_objective(trial_placement, pairs, days, objective_mode)
                if trial_cost + 1e-9 < best_cost:
                    best_candidate = candidate
                    best_cost = trial_cost

            if best_candidate is not original_candidate:
                block_choice[subject] = best_candidate
                current_cost = best_cost
                improved = True

        current_placement = _build_placement_from_blocks(block_choice)

    return block_choice, current_placement


def _violates_hard_pair_limits(
    placement: dict[str, tuple[pd.Timestamp, str]],
    *,
    pairs: pd.DataFrame,
    exam_lengths: dict[str, int],
    max_daily_minutes: int,
    max_clashes: float | None,
) -> bool:
    exams = list(placement)
    same_slot_clashes = 0.0
    for idx, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[idx + 1 :]:
            date_j, slot_j = placement[exam_j]
            same_day = pd.Timestamp(date_i).normalize() == pd.Timestamp(date_j).normalize()
            if same_day and int(exam_lengths[exam_i]) + int(exam_lengths[exam_j]) > max_daily_minutes:
                return True
            if same_day and slot_i == slot_j:
                same_slot_clashes += float(pairs.loc[exam_i, exam_j])
                if max_clashes is not None and same_slot_clashes > max_clashes:
                    return True
    return False
