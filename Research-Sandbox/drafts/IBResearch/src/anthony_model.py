from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_WEIGHTS: dict[str, int] = {
    "a": 64,
    "b": 32,
    "c": 16,
    "d": 8,
    "e": 4,
    "f": 2,
    "g": 1,
}

DEFAULT_SLOTS: tuple[str, str] = ("AM", "PM")


def parse_day_series(series: pd.Series) -> pd.Series:
    """Parse IB day strings consistently as day-first dates."""
    return pd.to_datetime(series, dayfirst=True).dt.normalize()


@dataclass(frozen=True)
class AnthonyModelData:
    """Cleaned data and index mappings used to build Anthony's IB model."""

    exams: pd.DataFrame
    days: pd.DataFrame
    pairs: pd.DataFrame
    exam_names: list[str]
    dates: list[pd.Timestamp]
    slots: tuple[str, str]


@dataclass(frozen=True)
class AnthonyModel:
    """Container returned by :func:`build_anthony_mip_model`."""

    model: Any
    x: dict[tuple[str, str, pd.Timestamp], Any]
    y: dict[tuple[str, str, str], Any]
    data: AnthonyModelData


def load_default_data(
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three CSV files needed by Anthony's model.

    Parameters
    ----------
    data_dir:
        Folder containing ``M24 exam names and block lengths.csv``,
        ``exam_days3.csv``, and ``Exam Pairs ABW-2.csv``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(exams, days, pairs)`` as raw DataFrames.
    """
    data_path = Path(data_dir)
    exams = pd.read_csv(data_path / "M24 exam names and block lengths.csv")
    days = pd.read_csv(data_path / "exam_days3.csv")
    pairs = pd.read_csv(data_path / "Exam Pairs ABW-2.csv")
    return exams, days, pairs


def prepare_anthony_model_data(
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    nb_days: int = 23,
    slots: tuple[str, str] = DEFAULT_SLOTS,
    recode_math_paper_three: bool = True,
) -> AnthonyModelData:
    """
    Clean and align the data for Anthony Furlong's IB timetabling model.

    The function keeps the data conventions from Anthony's notebook: exams are
    sorted by full exam name, dates are truncated to the first ``nb_days``, and
    the pair matrix is aligned to the cleaned exam order.
    """
    exams_clean = exams.copy()
    if "Full Name" not in exams_clean.columns and "FULL_NAME" in exams_clean.columns:
        exams_clean = exams_clean.rename(columns={"FULL_NAME": "Full Name"})
    required_exam_cols = {"Subject", "Full Name", "Length"}
    missing_exam_cols = required_exam_cols.difference(exams_clean.columns)
    if missing_exam_cols:
        raise ValueError(f"Exam data is missing required columns: {sorted(missing_exam_cols)}")

    exams_clean["Full Name"] = exams_clean["Full Name"].astype(str).str.strip()
    exams_clean["Subject"] = exams_clean["Subject"].astype(str).str.strip()
    exams_clean["Length"] = pd.to_numeric(exams_clean["Length"], errors="raise")

    if recode_math_paper_three:
        exams_clean.loc[
            exams_clean["Full Name"] == "MATHEMATICS APPLICATIONS AND INTERP PAPER THREE",
            "Subject",
        ] = "MATH APPS2"
        exams_clean.loc[
            exams_clean["Full Name"] == "MATHEMATICS ANALYSIS AND APPROACHES PAPER THREE",
            "Subject",
        ] = "MATH ANALYSIS2"

    exams_clean = (
        exams_clean.sort_values("Full Name")
        .drop_duplicates(subset=["Full Name"], keep="first")
        .reset_index(drop=True)
    )
    exam_names = exams_clean["Full Name"].tolist()

    days_clean = days.copy()
    if "Date" not in days_clean.columns or "DOW" not in days_clean.columns:
        raise ValueError("Day data must contain 'DOW' and 'Date' columns.")
    days_clean["Date"] = parse_day_series(days_clean["Date"])
    days_clean["DOW"] = days_clean["DOW"].astype(str).str.strip()
    days_clean = days_clean.head(nb_days).reset_index(drop=True)
    dates = days_clean["Date"].tolist()

    pairs_clean = pairs.copy()
    if pairs_clean.columns[0] not in exam_names:
        pairs_clean = pairs_clean.set_index(pairs_clean.columns[0])
    pairs_clean.index = pairs_clean.index.astype(str).str.strip()
    pairs_clean.columns = pairs_clean.columns.astype(str).str.strip()

    missing_pairs = sorted(set(exam_names).difference(pairs_clean.index).union(set(exam_names).difference(pairs_clean.columns)))
    if missing_pairs:
        raise ValueError(f"Pair matrix is missing exams used by the exam list: {missing_pairs}")

    pairs_clean = pairs_clean.loc[exam_names, exam_names].apply(pd.to_numeric, errors="raise")

    return AnthonyModelData(
        exams=exams_clean,
        days=days_clean,
        pairs=pairs_clean,
        exam_names=exam_names,
        dates=dates,
        slots=slots,
    )


def make_toy_days(
    year: int = 2026,
    *,
    total_days: int = 16,
    start_month: int = 4,
    start_day: int = 23,
) -> pd.DataFrame:
    """Create the toy examination window reported in Anthony's thesis."""
    start_date = pd.Timestamp(year=year, month=start_month, day=start_day).normalize()
    dates = pd.date_range(start_date, periods=total_days, freq="D")
    return pd.DataFrame({"DOW": dates.strftime("%a"), "Date": dates})


def prepare_toy_inputs(
    toy_exams: pd.DataFrame,
    toy_pairs: pd.DataFrame,
    *,
    toy_days: pd.DataFrame | None = None,
    year: int = 2026,
    total_days: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize the toy CSVs so they can be used by the same MIP API.

    The toy exam lengths are stored in hours. Anthony's model works in
    minutes for the IB data, so this function converts ``Length`` to minutes.
    """
    exams = toy_exams.copy()
    exams["Length"] = (pd.to_numeric(exams["Length"], errors="raise") * 60).round().astype(int)

    pairs = toy_pairs.copy()
    if pairs.columns[0].startswith("Unnamed"):
        pairs = pairs.rename(columns={pairs.columns[0]: "exam_name1"})

    days = make_toy_days(year=year, total_days=total_days) if toy_days is None else toy_days.copy()
    return exams, days.head(total_days).reset_index(drop=True), pairs


def build_anthony_mip_model(
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    nb_days: int = 23,
    slots: tuple[str, str] = DEFAULT_SLOTS,
    weights: dict[str, int] | None = None,
    max_clashes: int | None = 10_000,
    max_afternoon_minutes: int = 180,
    max_daily_minutes: int = 385,
    forbid_weekends: bool = True,
    forbid_may_first: bool = True,
    forbid_language_fridays: bool = True,
    forbid_language_friday_afternoons: bool = False,
    force_sbs_start: bool = True,
    consecutive_subject_exams: bool = True,
    consecutive_usable_subject_exams: bool = False,
    first_half_subjects: set[str] | None = None,
    recode_math_paper_three: bool = True,
    y_binary: bool = False,
    strengthen_y_upper_bounds: bool = False,
    proximity_at_most_one: bool = False,
    enforce_subject_exam_order: bool = False,
    objective_mode: str = "formal",
    output_flag: int = 1,
    model_name: str = "anthony_ib_mip",
) -> AnthonyModel:
    """
    Build Anthony Furlong's mixed-integer IB exam timetabling model.

    The model uses binary assignment variables ``x[exam, slot, date]`` and
    proximity variables ``y[exam_i, exam_j, category]`` for categories:

    ``a`` same slot, ``b`` same day/different slot, ``c`` next day,
    ``d`` one-day gap, ``e`` two-day gap, ``f`` three-day gap, and
    ``g`` four-day gap.

    The objective minimizes ``sum(C_ij * W_category * y_ij_category)``.
    Hard constraints follow Anthony's thesis/notebook rules for the supplied
    IB data: one slot per exam, no weekends, no May 1, no long PM exams,
    language exams not on Fridays, SBS opening days, same-subject consecutive
    days, and a maximum same-slot clash threshold.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("build_anthony_mip_model requires gurobipy.") from exc

    if len(slots) != 2:
        raise ValueError("Anthony's model assumes exactly two slots per day.")
    if objective_mode not in {"formal", "anthony_appendix"}:
        raise ValueError("objective_mode must be either 'formal' or 'anthony_appendix'.")

    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    missing_weights = set(DEFAULT_WEIGHTS).difference(weights)
    if missing_weights:
        raise ValueError(f"Missing objective weights for categories: {sorted(missing_weights)}")

    model_data = prepare_anthony_model_data(
        exams=exams,
        days=days,
        pairs=pairs,
        nb_days=nb_days,
        slots=slots,
        recode_math_paper_three=recode_math_paper_three,
    )
    exam_names = model_data.exam_names
    dates = model_data.dates
    days_clean = model_data.days
    exams_clean = model_data.exams
    pairs_clean = model_data.pairs
    slot_am, slot_pm = slots

    model = gp.Model(model_name)
    model.setParam("OutputFlag", output_flag)

    x: dict[tuple[str, str, pd.Timestamp], gp.Var] = {}
    for exam in exam_names:
        for slot in slots:
            for date in dates:
                x[(exam, slot, date)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=_var_name("x", exam, slot, date.strftime("%Y%m%d")),
                )

    y_type = GRB.BINARY if y_binary else GRB.CONTINUOUS
    y: dict[tuple[str, str, str], gp.Var] = {}
    for i, exam_i in enumerate(exam_names):
        for exam_j in exam_names[i + 1 :]:
            for category in DEFAULT_WEIGHTS:
                y[(exam_i, exam_j, category)] = model.addVar(
                    vtype=y_type,
                    lb=0.0,
                    ub=1.0,
                    name=_var_name(f"y_{category}", exam_i, exam_j),
                )

    model.update()

    for exam in exam_names:
        model.addConstr(
            gp.quicksum(x[(exam, slot, date)] for slot in slots for date in dates) == 1,
            name=_var_name("assign", exam),
        )

    for i, exam_i in enumerate(exam_names):
        for exam_j in exam_names[i + 1 :]:
            for date in dates:
                for slot in slots:
                    model.addConstr(
                        y[(exam_i, exam_j, "a")] >= x[(exam_i, slot, date)] + x[(exam_j, slot, date)] - 1,
                        name=_var_name("link_a", exam_i, exam_j, slot, date.strftime("%Y%m%d")),
                    )

                if objective_mode == "anthony_appendix":
                    model.addConstr(
                        y[(exam_i, exam_j, "b")]
                        >= x[(exam_i, slot_am, date)]
                        + x[(exam_j, slot_am, date)]
                        + x[(exam_i, slot_pm, date)]
                        + x[(exam_j, slot_pm, date)]
                        - 1,
                        name=_var_name("link_b", exam_i, exam_j, "same_day", date.strftime("%Y%m%d")),
                    )
                else:
                    model.addConstr(
                        y[(exam_i, exam_j, "b")] >= x[(exam_i, slot_am, date)] + x[(exam_j, slot_pm, date)] - 1,
                        name=_var_name("link_b", exam_i, exam_j, slot_am, slot_pm, date.strftime("%Y%m%d")),
                    )
                    model.addConstr(
                        y[(exam_i, exam_j, "b")] >= x[(exam_i, slot_pm, date)] + x[(exam_j, slot_am, date)] - 1,
                        name=_var_name("link_b", exam_i, exam_j, slot_pm, slot_am, date.strftime("%Y%m%d")),
                    )

            gap_to_category = {1: "c", 2: "d", 3: "e", 4: "f", 5: "g"}
            for gap, category in gap_to_category.items():
                target_category = "c" if objective_mode == "anthony_appendix" and gap == 5 else category
                for left_idx in range(len(dates) - gap):
                    left = dates[left_idx]
                    right = dates[left_idx + gap]
                    model.addConstr(
                        y[(exam_i, exam_j, target_category)]
                        >= gp.quicksum(x[(exam_i, slot, left)] for slot in slots)
                        + gp.quicksum(x[(exam_j, slot, right)] for slot in slots)
                        - 1,
                        name=_var_name("link_" + target_category, exam_i, exam_j, "fwd", left_idx),
                    )
                    model.addConstr(
                        y[(exam_i, exam_j, target_category)]
                        >= gp.quicksum(x[(exam_i, slot, right)] for slot in slots)
                        + gp.quicksum(x[(exam_j, slot, left)] for slot in slots)
                        - 1,
                        name=_var_name("link_" + target_category, exam_i, exam_j, "back", left_idx),
                    )

    if strengthen_y_upper_bounds:
        _add_y_upper_bound_strengthening(
            model=model,
            x=x,
            y=y,
            exam_names=exam_names,
            dates=dates,
            slots=slots,
            objective_mode=objective_mode,
        )

    if proximity_at_most_one:
        if objective_mode != "formal":
            raise ValueError("proximity_at_most_one is only valid for objective_mode='formal'.")
        for i, exam_i in enumerate(exam_names):
            for exam_j in exam_names[i + 1 :]:
                model.addConstr(
                    gp.quicksum(y[(exam_i, exam_j, category)] for category in DEFAULT_WEIGHTS) <= 1,
                    name=_var_name("proximity_at_most_one", exam_i, exam_j),
                )

    if forbid_weekends:
        weekend_dates = days_clean.loc[days_clean["DOW"].isin(["Sat", "Sun"]), "Date"].tolist()
        for exam in exam_names:
            for slot in slots:
                for date in weekend_dates:
                    model.addConstr(x[(exam, slot, date)] == 0, name=_var_name("no_weekend", exam, slot, date.strftime("%Y%m%d")))

    if forbid_language_fridays:
        language_subjects = {
            "LANG LIT",
            "LANG ACQ",
            "LIT",
            "LANGUAGE A LITERATURE",
            "LANGUAGE ACQUISITION",
            "LANGUAGE A LITERATURE",
        }
        friday_dates = days_clean.loc[days_clean["DOW"] == "Fri", "Date"].tolist()
        language_exams = exams_clean.loc[exams_clean["Subject"].str.upper().isin(language_subjects), "Full Name"].tolist()
        for exam in language_exams:
            for slot in slots:
                for date in friday_dates:
                    model.addConstr(x[(exam, slot, date)] == 0, name=_var_name("no_language_friday", exam, slot, date.strftime("%Y%m%d")))

    if forbid_language_friday_afternoons:
        language_subjects = {
            "LANG LIT",
            "LANG ACQ",
            "LIT",
            "LANGUAGE A LITERATURE",
            "LANGUAGE ACQUISITION",
        }
        friday_dates = days_clean.loc[days_clean["DOW"] == "Fri", "Date"].tolist()
        language_exams = exams_clean.loc[exams_clean["Subject"].str.upper().isin(language_subjects), "Full Name"].tolist()
        for exam in language_exams:
            for date in friday_dates:
                model.addConstr(
                    x[(exam, slot_pm, date)] == 0,
                    name=_var_name("no_language_friday_pm", exam, date.strftime("%Y%m%d")),
                )

    if forbid_may_first:
        may_first_dates = days_clean.loc[(days_clean["Date"].dt.month == 5) & (days_clean["Date"].dt.day == 1), "Date"].tolist()
        for exam in exam_names:
            for slot in slots:
                for date in may_first_dates:
                    model.addConstr(x[(exam, slot, date)] == 0, name=_var_name("no_may_first", exam, slot, date.strftime("%Y%m%d")))

    if max_clashes is not None:
        model.addConstr(
            gp.quicksum(
                float(pairs_clean.loc[exam_i, exam_j]) * y[(exam_i, exam_j, "a")]
                for i, exam_i in enumerate(exam_names)
                for exam_j in exam_names[i + 1 :]
            )
            <= max_clashes,
            name="max_same_slot_clashes",
        )

    for exam in exams_clean.loc[exams_clean["Length"] > max_afternoon_minutes, "Full Name"].tolist():
        for date in dates:
            model.addConstr(x[(exam, slot_pm, date)] == 0, name=_var_name("no_long_pm", exam, date.strftime("%Y%m%d")))

    for i, exam_i in enumerate(exam_names):
        length_i = float(exams_clean.loc[exams_clean["Full Name"] == exam_i, "Length"].iloc[0])
        for exam_j in exam_names[i + 1 :]:
            length_j = float(exams_clean.loc[exams_clean["Full Name"] == exam_j, "Length"].iloc[0])
            if length_i + length_j > max_daily_minutes:
                for date in dates:
                    model.addConstr(
                        gp.quicksum(x[(exam_i, slot, date)] for slot in slots)
                        + gp.quicksum(x[(exam_j, slot, date)] for slot in slots)
                        <= 1,
                        name=_var_name("max_daily_minutes", exam_i, exam_j, date.strftime("%Y%m%d")),
                    )

    if force_sbs_start and dates:
        sbs_one = exams_clean.loc[
            exams_clean["Full Name"].str.upper().isin({"SBS PAPER ONE", "SBS EXAM 1"}),
            "Full Name",
        ].tolist()
        for exam in sbs_one:
            model.addConstr(x[(exam, slot_am, dates[0])] == 1, name="sbs_paper_one_start")

        if len(dates) >= 2:
            blocked_dates = dates[:2]
            non_sbs_exams = exams_clean.loc[exams_clean["Subject"].str.upper() != "SBS", "Full Name"].tolist()
            for exam in non_sbs_exams:
                for slot in slots:
                    for date in blocked_dates:
                        model.addConstr(x[(exam, slot, date)] == 0, name=_var_name("only_sbs_opening", exam, slot, date.strftime("%Y%m%d")))

    if consecutive_subject_exams:
        if consecutive_usable_subject_exams:
            blocked_dates = set()
            if forbid_weekends:
                blocked_dates.update(days_clean.loc[days_clean["DOW"].isin(["Sat", "Sun"]), "Date"].tolist())
            if forbid_may_first:
                blocked_dates.update(
                    days_clean.loc[
                        (days_clean["Date"].dt.month == 5) & (days_clean["Date"].dt.day == 1),
                        "Date",
                    ].tolist()
                )
            sequence_dates = [date for date in dates if date not in blocked_dates]
        else:
            sequence_dates = dates

        for subject, subject_exams_df in exams_clean.groupby("Subject", sort=False):
            subject_exams = subject_exams_df["Full Name"].tolist()
            for i, exam_i in enumerate(subject_exams):
                for exam_j in subject_exams[i + 1 :]:
                    for date_idx, date in enumerate(dates):
                        if consecutive_usable_subject_exams and date not in sequence_dates:
                            adjacent_dates: set[pd.Timestamp] = set()
                        else:
                            pos = sequence_dates.index(date)
                            adjacent_dates = {
                                sequence_dates[k]
                                for k in (pos - 1, pos + 1)
                                if 0 <= k < len(sequence_dates)
                            }
                        model.addConstr(
                            gp.quicksum(x[(exam_j, slot, date)] for slot in slots)
                            + gp.quicksum(
                                x[(exam_i, slot, other_date)]
                                for slot in slots
                                for other_date in dates
                                if other_date not in adjacent_dates
                            )
                            <= 1,
                            name=_var_name("consecutive_subject", subject, exam_i, exam_j, date_idx),
                        )

    if first_half_subjects:
        second_half_dates = dates[round(len(dates) / 2) :]
        first_half_exams = exams_clean.loc[exams_clean["Subject"].isin(first_half_subjects), "Full Name"].tolist()
        for exam in first_half_exams:
            model.addConstr(
                gp.quicksum(x[(exam, slot, date)] for slot in slots for date in second_half_dates) == 0,
                name=_var_name("first_half", exam),
            )

    if enforce_subject_exam_order:
        date_position = {date: idx for idx, date in enumerate(dates)}
        for subject, subject_exams_df in exams_clean.groupby("Subject", sort=False):
            ordered_exams = sorted(subject_exams_df["Full Name"].tolist(), key=_exam_order_key)
            for left_exam, right_exam in zip(ordered_exams, ordered_exams[1:]):
                model.addConstr(
                    gp.quicksum(
                        date_position[date] * x[(left_exam, slot, date)]
                        for slot in slots
                        for date in dates
                    )
                    <= gp.quicksum(
                        date_position[date] * x[(right_exam, slot, date)]
                        for slot in slots
                        for date in dates
                    ),
                    name=_var_name("subject_exam_order", subject, left_exam, right_exam),
                )

    objective = gp.LinExpr()
    for i, exam_i in enumerate(exam_names):
        for exam_j in exam_names[i + 1 :]:
            cij = float(pairs_clean.loc[exam_i, exam_j])
            if cij == 0:
                continue
            for category, weight in weights.items():
                objective.addTerms(cij * float(weight), y[(exam_i, exam_j, category)])

    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    return AnthonyModel(model=model, x=x, y=y, data=model_data)


def _add_y_upper_bound_strengthening(
    *,
    model: Any,
    x: dict[tuple[str, str, pd.Timestamp], Any],
    y: dict[tuple[str, str, str], Any],
    exam_names: list[str],
    dates: list[pd.Timestamp],
    slots: tuple[str, str],
    objective_mode: str,
) -> None:
    """
    Add auxiliary AND variables so each proximity variable also has an upper bound.

    The base model only needs lower trigger constraints because all proximity
    variables have positive objective coefficients. This extended formulation is
    not required for correctness, but it can strengthen the LP relaxation and
    therefore improve proof of optimality on small experiments.
    """
    import gurobipy as gp

    slot_am, slot_pm = slots

    def day_expr(exam: str, date: pd.Timestamp) -> Any:
        return gp.quicksum(x[(exam, slot, date)] for slot in slots)

    def add_and_var(name: str, left: Any, right: Any) -> Any:
        z = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name=name)
        model.addConstr(z <= left, name=f"{name}_ub_left")
        model.addConstr(z <= right, name=f"{name}_ub_right")
        model.addConstr(z >= left + right - 1.0, name=f"{name}_lb")
        return z

    for i, exam_i in enumerate(exam_names):
        for exam_j in exam_names[i + 1 :]:
            same_slot_terms = []
            same_day_terms = []
            cross_slot_terms = []
            gap_terms: dict[int, list[Any]] = {gap: [] for gap in range(1, 6)}

            for date in dates:
                for slot in slots:
                    same_slot_terms.append(
                        add_and_var(
                            _var_name("z_same_slot", exam_i, exam_j, slot, date.strftime("%Y%m%d")),
                            x[(exam_i, slot, date)],
                            x[(exam_j, slot, date)],
                        )
                    )

                if objective_mode == "anthony_appendix":
                    same_day_terms.append(
                        add_and_var(
                            _var_name("z_same_day", exam_i, exam_j, date.strftime("%Y%m%d")),
                            day_expr(exam_i, date),
                            day_expr(exam_j, date),
                        )
                    )
                else:
                    cross_slot_terms.append(
                        add_and_var(
                            _var_name("z_cross_slot", exam_i, exam_j, slot_am, slot_pm, date.strftime("%Y%m%d")),
                            x[(exam_i, slot_am, date)],
                            x[(exam_j, slot_pm, date)],
                        )
                    )
                    cross_slot_terms.append(
                        add_and_var(
                            _var_name("z_cross_slot", exam_i, exam_j, slot_pm, slot_am, date.strftime("%Y%m%d")),
                            x[(exam_i, slot_pm, date)],
                            x[(exam_j, slot_am, date)],
                        )
                    )

            for gap in range(1, 6):
                for left_idx in range(len(dates) - gap):
                    left = dates[left_idx]
                    right = dates[left_idx + gap]
                    gap_terms[gap].append(
                        add_and_var(
                            _var_name("z_gap", exam_i, exam_j, gap, "fwd", left_idx),
                            day_expr(exam_i, left),
                            day_expr(exam_j, right),
                        )
                    )
                    gap_terms[gap].append(
                        add_and_var(
                            _var_name("z_gap", exam_i, exam_j, gap, "back", left_idx),
                            day_expr(exam_i, right),
                            day_expr(exam_j, left),
                        )
                    )

            model.addConstr(
                y[(exam_i, exam_j, "a")] <= gp.quicksum(same_slot_terms),
                name=_var_name("ub_y_a", exam_i, exam_j),
            )
            if objective_mode == "anthony_appendix":
                model.addConstr(
                    y[(exam_i, exam_j, "b")] <= gp.quicksum(same_day_terms),
                    name=_var_name("ub_y_b", exam_i, exam_j),
                )
                model.addConstr(
                    y[(exam_i, exam_j, "c")] <= gp.quicksum(gap_terms[1] + gap_terms[5]),
                    name=_var_name("ub_y_c", exam_i, exam_j),
                )
            else:
                model.addConstr(
                    y[(exam_i, exam_j, "b")] <= gp.quicksum(cross_slot_terms),
                    name=_var_name("ub_y_b", exam_i, exam_j),
                )
                model.addConstr(
                    y[(exam_i, exam_j, "c")] <= gp.quicksum(gap_terms[1]),
                    name=_var_name("ub_y_c", exam_i, exam_j),
                )

            for gap, category in {2: "d", 3: "e", 4: "f"}.items():
                model.addConstr(
                    y[(exam_i, exam_j, category)] <= gp.quicksum(gap_terms[gap]),
                    name=_var_name("ub_y_" + category, exam_i, exam_j),
                )
            if objective_mode == "formal":
                model.addConstr(
                    y[(exam_i, exam_j, "g")] <= gp.quicksum(gap_terms[5]),
                    name=_var_name("ub_y_g", exam_i, exam_j),
                )

    model.update()


def _exam_order_key(exam_name: str) -> tuple[int, str]:
    upper = exam_name.upper()
    paper_order = {
        "PAPER ONE": 1,
        "PAPER TWO": 2,
        "PAPER THREE": 3,
        "P1": 1,
        "P2": 2,
        "P3": 3,
        "EXAM 1": 1,
        "EXAM 2": 2,
        "EXAM 3": 3,
    }
    for marker, rank in paper_order.items():
        if marker in upper:
            return rank, upper
    return 99, upper


def solve_anthony_mip(
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    mip_start_timetable: pd.DataFrame | None = None,
    **model_kwargs: Any,
) -> tuple[AnthonyModel, pd.DataFrame]:
    """
    Build and solve Anthony's model, returning the model and timetable.

    Extra keyword arguments are passed to :func:`build_anthony_mip_model`.
    """
    built = build_anthony_mip_model(exams=exams, days=days, pairs=pairs, **model_kwargs)
    if time_limit is not None:
        built.model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        built.model.setParam("MIPGap", mip_gap)
    if mip_start_timetable is not None:
        apply_timetable_start(built, mip_start_timetable)

    built.model.optimize()
    return built, timetable_from_solution(built)


def solve_toy_mip(
    toy_exams: pd.DataFrame,
    toy_pairs: pd.DataFrame,
    *,
    toy_days: pd.DataFrame | None = None,
    year: int = 2026,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    mip_start_timetable: pd.DataFrame | None = None,
    **model_kwargs: Any,
) -> tuple[AnthonyModel, pd.DataFrame]:
    """Apply Anthony's MIP model to the toy CSV data."""
    exams, days, pairs = prepare_toy_inputs(toy_exams, toy_pairs, toy_days=toy_days, year=year)
    kwargs = {
        "nb_days": len(days),
        "max_clashes": 15,
        "max_afternoon_minutes": 180,
        "max_daily_minutes": 375,
        "forbid_language_fridays": True,
        "forbid_language_friday_afternoons": False,
        "consecutive_usable_subject_exams": False,
        "first_half_subjects": {"Finance", "Law and Ethics"},
        "recode_math_paper_three": False,
        "objective_mode": "anthony_appendix",
    }
    kwargs.update(model_kwargs)
    return solve_anthony_mip(
        exams=exams,
        days=days,
        pairs=pairs,
        time_limit=time_limit,
        mip_gap=mip_gap,
        mip_start_timetable=mip_start_timetable,
        **kwargs,
    )


def apply_timetable_start(built: AnthonyModel, timetable: pd.DataFrame) -> None:
    """Set Gurobi ``Start`` values from a shared-format timetable."""
    placement = placement_from_timetable(timetable)
    for (exam, slot, date), var in built.x.items():
        assigned = placement.get(exam)
        var.Start = 1.0 if assigned == (pd.Timestamp(date).normalize(), slot) else 0.0


def timetable_from_solution(built: AnthonyModel, *, threshold: float = 0.5) -> pd.DataFrame:
    """Convert solved ``x`` variables into Anthony's timetable format."""
    records: list[dict[str, Any]] = []
    day_lookup = built.data.days.set_index("Date")["DOW"].to_dict()
    for (exam, slot, date), var in built.x.items():
        if var.X > threshold:
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


def placement_from_timetable(timetable: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, str]]:
    """Convert the shared timetable DataFrame format into an exam placement."""
    required = {"Exam_Name", "Date", "Slot"}
    missing = required.difference(timetable.columns)
    if missing:
        raise ValueError(f"Timetable is missing columns: {sorted(missing)}")

    placement: dict[str, tuple[pd.Timestamp, str]] = {}
    for row in timetable.itertuples(index=False):
        exam = getattr(row, "Exam_Name")
        if exam in placement:
            raise ValueError(f"Exam {exam!r} appears more than once in the timetable.")
        placement[exam] = (pd.Timestamp(getattr(row, "Date")).normalize(), str(getattr(row, "Slot")))
    return placement


def timetable_from_placement(
    placement: dict[str, tuple[pd.Timestamp, str]],
    days: pd.DataFrame,
) -> pd.DataFrame:
    """Convert an exam placement into the shared solution DataFrame format."""
    days_clean = days.copy()
    days_clean["Date"] = parse_day_series(days_clean["Date"])
    day_lookup = days_clean.set_index("Date")["DOW"].to_dict()
    records = [
        {
            "Day_of_Week": day_lookup.get(pd.Timestamp(date).normalize()),
            "Date": pd.Timestamp(date).normalize(),
            "Slot": slot,
            "Exam_Name": exam,
        }
        for exam, (date, slot) in placement.items()
    ]
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["Date", "Slot", "Exam_Name"])
        .reset_index(drop=True)
    )


def mip_objective_value(
    timetable: pd.DataFrame,
    pairs: pd.DataFrame,
    days: pd.DataFrame,
    *,
    weights: dict[str, int] | None = None,
    mode: str = "formal",
) -> float:
    """
    Evaluate a timetable with the exact weighted proximity objective used by the MIP.

    Categories are identical to the model: same slot, same day/different slot,
    and absolute date-index gaps of one through five days.

    Use ``mode='anthony_appendix'`` to mimic the implementation in thesis
    Appendix 1.3, where same-day pairs are penalized with ``b`` regardless of
    slot and five-day gaps are linked to ``c`` rather than ``g``.
    """
    if mode not in {"formal", "anthony_appendix"}:
        raise ValueError("mode must be either 'formal' or 'anthony_appendix'.")

    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    placement = placement_from_timetable(timetable)

    days_clean = days.copy()
    days_clean["Date"] = parse_day_series(days_clean["Date"])
    day_index = {date: idx for idx, date in enumerate(days_clean["Date"].tolist())}

    pairs_clean = pairs.copy()
    if not set(placement).issubset(set(pairs_clean.index)):
        pairs_clean = pairs_clean.set_index(pairs_clean.columns[0])
    pairs_clean.index = pairs_clean.index.astype(str).str.strip()
    pairs_clean.columns = pairs_clean.columns.astype(str).str.strip()

    total = 0.0
    exams = sorted(placement)
    for idx, exam_i in enumerate(exams):
        for exam_j in exams[idx + 1 :]:
            cij = float(pairs_clean.loc[exam_i, exam_j])
            if cij == 0:
                continue

            date_i, slot_i = placement[exam_i]
            date_j, slot_j = placement[exam_j]
            gap = abs(day_index[date_i] - day_index[date_j])

            if gap == 0 and slot_i == slot_j:
                total += cij * weights["a"]
            if mode == "anthony_appendix" and gap == 0:
                total += cij * weights["b"]
            elif mode == "formal" and gap == 0 and slot_i != slot_j:
                total += cij * weights["b"]
            elif gap == 1 or (mode == "anthony_appendix" and gap == 5):
                total += cij * weights["c"]
            elif gap == 2:
                total += cij * weights["d"]
            elif gap == 3:
                total += cij * weights["e"]
            elif gap == 4:
                total += cij * weights["f"]
            elif mode == "formal" and gap == 5:
                total += cij * weights["g"]
    return total


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
