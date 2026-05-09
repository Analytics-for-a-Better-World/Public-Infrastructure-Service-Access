from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.anthony_model import (
    AnthonyModel,
    apply_timetable_start,
    build_anthony_mip_model,
    mip_objective_value,
    prepare_anthony_model_data,
    timetable_from_solution,
)
from src.full_heuristic import validate_full_solution


@dataclass(frozen=True)
class SpreadDiagnostics:
    """Compact load diagnostics for a timetable."""

    objective_value: float
    used_usable_days: int
    max_day_exams: int
    max_slot_exams: int
    overloaded_day_exams: int
    overloaded_slot_exams: int
    same_slot_clashes: float


@dataclass(frozen=True)
class SpreadResult:
    """Result returned by :func:`spread_with_secondary_mip`."""

    timetable: pd.DataFrame
    diagnostics_before: SpreadDiagnostics
    diagnostics_after: SpreadDiagnostics
    data: Any
    solver_status: int
    solver_gap: float | None


def spread_with_secondary_mip(
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    start_timetable: pd.DataFrame,
    *,
    time_limit: float = 300.0,
    allowed_objective_increase: float = 0.0,
    target_day_load: int = 5,
    target_slot_load: int = 3,
    nb_days: int = 23,
    objective_mode: str = "formal",
    max_clashes: int | None = 10_000,
    max_afternoon_minutes: int = 180,
    max_daily_minutes: int = 385,
    first_half_subjects: set[str] | None = None,
    output_flag: int = 0,
) -> SpreadResult:
    """
    Reoptimize timetable load while preserving Antony's objective value.

    The primary objective is not replaced. Instead, this pass constrains the
    Antony objective to remain within ``allowed_objective_increase`` of the
    starting timetable, then minimizes a deterministic load/spread objective.
    """
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
    start = _normalize_timetable(start_timetable)
    validate_full_solution(
        start,
        data,
        max_clashes=max_clashes,
        max_afternoon_minutes=max_afternoon_minutes,
        max_daily_minutes=max_daily_minutes,
        first_half_subjects=first_half_subjects,
    )
    start_objective = mip_objective_value(start, data.pairs, data.days, mode=objective_mode)
    before = spread_diagnostics(
        start,
        data,
        objective_mode=objective_mode,
        target_day_load=target_day_load,
        target_slot_load=target_slot_load,
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
        objective_mode=objective_mode,
        output_flag=output_flag,
        model_name="full_spread_secondary",
    )
    apply_timetable_start(built, start)
    _add_primary_objective_cap(
        built=built,
        cap=start_objective + allowed_objective_increase,
    )
    _set_spread_objective(
        built=built,
        target_day_load=target_day_load,
        target_slot_load=target_slot_load,
    )
    built.model.setParam("TimeLimit", time_limit)
    built.model.optimize()

    if built.model.SolCount:
        candidate = timetable_from_solution(built)
    else:
        candidate = start

    validate_full_solution(
        candidate,
        data,
        max_clashes=max_clashes,
        max_afternoon_minutes=max_afternoon_minutes,
        max_daily_minutes=max_daily_minutes,
        first_half_subjects=first_half_subjects,
    )
    after = spread_diagnostics(
        candidate,
        data,
        objective_mode=objective_mode,
        target_day_load=target_day_load,
        target_slot_load=target_slot_load,
    )
    return SpreadResult(
        timetable=candidate,
        diagnostics_before=before,
        diagnostics_after=after,
        data=data,
        solver_status=int(built.model.Status),
        solver_gap=float(built.model.MIPGap) if built.model.SolCount else None,
    )


def spread_diagnostics(
    timetable: pd.DataFrame,
    data: Any,
    *,
    objective_mode: str = "formal",
    target_day_load: int = 5,
    target_slot_load: int = 3,
) -> SpreadDiagnostics:
    """Compute objective and load diagnostics for a timetable."""
    tt = _normalize_timetable(timetable)
    usable_dates = _usable_dates(data)
    day_counts = tt.groupby("Date").size().reindex(usable_dates, fill_value=0)
    slot_index = pd.MultiIndex.from_product([usable_dates, data.slots], names=["Date", "Slot"])
    slot_counts = tt.groupby(["Date", "Slot"]).size().reindex(slot_index, fill_value=0)
    validation = validate_full_solution(tt, data)
    return SpreadDiagnostics(
        objective_value=mip_objective_value(tt, data.pairs, data.days, mode=objective_mode),
        used_usable_days=int((day_counts > 0).sum()),
        max_day_exams=int(day_counts.max()),
        max_slot_exams=int(slot_counts.max()),
        overloaded_day_exams=int((day_counts - target_day_load).clip(lower=0).sum()),
        overloaded_slot_exams=int((slot_counts - target_slot_load).clip(lower=0).sum()),
        same_slot_clashes=float(validation["same_slot_clashes"]),
    )


def _add_primary_objective_cap(*, built: AnthonyModel, cap: float) -> None:
    objective = built.model.getObjective()
    built.model.addConstr(objective <= cap + 1e-6, name="keep_anthony_objective")
    built.model.update()


def _set_spread_objective(
    *,
    built: AnthonyModel,
    target_day_load: int,
    target_slot_load: int,
) -> None:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("spread_with_secondary_mip requires gurobipy.") from exc

    model = built.model
    usable_dates = _usable_dates(built.data)
    exam_count = len(built.data.exam_names)

    max_day = model.addVar(vtype=GRB.INTEGER, lb=0, ub=exam_count, name="spread_max_day")
    max_slot = model.addVar(vtype=GRB.INTEGER, lb=0, ub=exam_count, name="spread_max_slot")
    objective = gp.LinExpr()

    for date in usable_dates:
        day_load = gp.quicksum(
            built.x[(exam, slot, date)]
            for exam in built.data.exam_names
            for slot in built.data.slots
        )
        day_over = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"spread_day_over[{date:%Y%m%d}]")
        used_day = model.addVar(vtype=GRB.BINARY, name=f"spread_used_day[{date:%Y%m%d}]")
        model.addConstr(day_load <= max_day, name=f"spread_max_day_link[{date:%Y%m%d}]")
        model.addConstr(day_over >= day_load - target_day_load, name=f"spread_day_over_link[{date:%Y%m%d}]")
        model.addConstr(day_load <= exam_count * used_day, name=f"spread_used_day_upper[{date:%Y%m%d}]")
        model.addConstr(day_load >= used_day, name=f"spread_used_day_lower[{date:%Y%m%d}]")
        objective.addTerms(1000.0, day_over)
        objective.addTerms(-40.0, used_day)

        for slot in built.data.slots:
            slot_load = gp.quicksum(
                built.x[(exam, slot, date)]
                for exam in built.data.exam_names
            )
            slot_over = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"spread_slot_over[{date:%Y%m%d},{slot}]",
            )
            model.addConstr(slot_load <= max_slot, name=f"spread_max_slot_link[{date:%Y%m%d},{slot}]")
            model.addConstr(
                slot_over >= slot_load - target_slot_load,
                name=f"spread_slot_over_link[{date:%Y%m%d},{slot}]",
            )
            objective.addTerms(300.0, slot_over)

    objective.addTerms(10_000.0, max_day)
    objective.addTerms(2_000.0, max_slot)
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()


def _usable_dates(data: Any) -> list[pd.Timestamp]:
    dates = data.days["Date"].tolist()
    blocked = set(data.days.loc[data.days["DOW"].isin(["Sat", "Sun"]), "Date"])
    blocked.update(data.days.loc[(data.days["Date"].dt.month == 5) & (data.days["Date"].dt.day == 1), "Date"])
    return [date for date in dates if date not in blocked]


def _normalize_timetable(timetable: pd.DataFrame) -> pd.DataFrame:
    tt = timetable.copy()
    tt["Date"] = pd.to_datetime(tt["Date"], dayfirst=True).dt.normalize()
    return tt[["Day_of_Week", "Date", "Slot", "Exam_Name"]].sort_values(
        ["Date", "Slot", "Exam_Name"]
    ).reset_index(drop=True)
