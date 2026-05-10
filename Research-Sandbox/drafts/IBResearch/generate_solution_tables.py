from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.anthony_model import load_default_data, prepare_toy_inputs

WEIGHTS = {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1}
LABELS = {
    "a": "same slot",
    "b": "same day, different slot",
    "c": "consecutive days",
    "d": "one-day gap",
    "e": "two-day gap",
    "f": "three-day gap",
    "g": "four-day gap",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX schedule and penalty tables for saved best solutions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clean-run-dir", type=Path, default=Path("clean_runs_20260508"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--toy-solution", type=Path, default=None)
    parser.add_argument("--full-solution", type=Path, default=None)
    args = parser.parse_args()

    args.table_dir.mkdir(parents=True, exist_ok=True)
    toy_solution = args.toy_solution or args.clean_run_dir / "toy_antony_verbatim_mip.csv"
    full_solution = args.full_solution or args.clean_run_dir / "full_lns_nb34_guarded_6x120.csv"

    toy_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    toy_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    toy_exams, toy_days, toy_pairs = prepare_toy_inputs(toy_exams, toy_pairs)
    toy_timetable = pd.read_csv(toy_solution)
    write_schedule_table(
        toy_timetable,
        args.table_dir / "toy_best_schedule_table.tex",
        "Toy best timetable, proven optimal for the currently available toy data.",
        "tab:toy-best-schedule",
    )
    toy_breakdown, toy_total = penalty_breakdown(toy_timetable, toy_pairs, toy_days, mode="anthony_appendix")
    write_penalty_table(
        toy_breakdown,
        toy_total,
        args.table_dir / "toy_best_penalty_breakdown.tex",
        "Penalty breakdown for the toy best timetable under the Appendix-style objective.",
        "tab:toy-best-penalty",
    )

    _full_exams, full_days, full_pairs = load_default_data(args.data_dir)
    full_timetable = pd.read_csv(full_solution)
    write_schedule_table(
        full_timetable,
        args.table_dir / "full_best_schedule_table.tex",
        "Full-instance best timetable found so far; feasible but not proven optimal.",
        "tab:full-best-schedule",
    )
    full_breakdown, full_total = penalty_breakdown(full_timetable, full_pairs, full_days.head(34), mode="formal")
    write_penalty_table(
        full_breakdown,
        full_total,
        args.table_dir / "full_best_penalty_breakdown.tex",
        "Penalty breakdown for the full-instance best timetable under the formal objective.",
        "tab:full-best-penalty",
    )

    print(f"toy_appendix_objective={toy_total:,.0f}")
    print(f"full_formal_objective={full_total:,.0f}")
    print(f"wrote_tables={args.table_dir}")


def normalize_pairs(pairs: pd.DataFrame, exam_names: list[str]) -> pd.DataFrame:
    pair_matrix = pairs.copy()
    if not set(exam_names).issubset(set(pair_matrix.index)):
        pair_matrix = pair_matrix.set_index(pair_matrix.columns[0])
    pair_matrix.index = pair_matrix.index.astype(str).str.strip()
    pair_matrix.columns = pair_matrix.columns.astype(str).str.strip()
    return pair_matrix.apply(pd.to_numeric, errors="raise")


def write_schedule_table(timetable: pd.DataFrame, output: Path, caption: str, label: str) -> None:
    table = timetable.copy()
    table["Date"] = pd.to_datetime(table["Date"]).dt.normalize()
    table["Slot"] = table["Slot"].astype(str).str.upper()
    rows = []
    for date, group in table.sort_values(["Date", "Slot", "Exam_Name"]).groupby("Date", sort=True):
        am = "; ".join(group.loc[group["Slot"] == "AM", "Exam_Name"]) or "--"
        pm = "; ".join(group.loc[group["Slot"] == "PM", "Exam_Name"]) or "--"
        day = group["Day_of_Week"].iloc[0] if "Day_of_Week" in group else date.strftime("%a")
        rows.append((date.strftime("%Y-%m-%d"), day, am, pm))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabularx}{\textwidth}{llXX}",
        r"\toprule",
        r"Date & Day & AM & PM \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def penalty_breakdown(timetable: pd.DataFrame, pairs: pd.DataFrame, days: pd.DataFrame, *, mode: str) -> tuple[dict, float]:
    table = timetable.copy()
    table["Date"] = pd.to_datetime(table["Date"]).dt.normalize()
    table["Slot"] = table["Slot"].astype(str).str.upper()
    placement = {row.Exam_Name: (row.Date, row.Slot) for row in table.itertuples(index=False)}
    exams = sorted(placement)
    pair_matrix = normalize_pairs(pairs, exams)
    day_list = pd.to_datetime(days["Date"], dayfirst=True).dt.normalize().tolist()
    day_index = {date: index for index, date in enumerate(day_list)}

    rows = {key: {"pairs": 0, "student_mass": 0.0, "contribution": 0.0} for key in WEIGHTS}
    total = 0.0
    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]
        for exam_j in exams[pos + 1 :]:
            date_j, slot_j = placement[exam_j]
            cij = float(pair_matrix.loc[exam_i, exam_j])
            if cij == 0:
                continue
            gap = abs(day_index[date_i] - day_index[date_j])
            for category in active_categories(gap, slot_i, slot_j, mode):
                rows[category]["pairs"] += 1
                rows[category]["student_mass"] += cij
                rows[category]["contribution"] += cij * WEIGHTS[category]
                total += cij * WEIGHTS[category]
    return rows, total


def active_categories(gap: int, slot_i: str, slot_j: str, mode: str) -> list[str]:
    categories = []
    if gap == 0 and slot_i == slot_j:
        categories.append("a")
    if mode == "anthony_appendix":
        if gap == 0:
            categories.append("b")
        elif gap == 1 or gap == 5:
            categories.append("c")
        elif gap == 2:
            categories.append("d")
        elif gap == 3:
            categories.append("e")
        elif gap == 4:
            categories.append("f")
    elif mode == "formal":
        if gap == 0 and slot_i != slot_j:
            categories.append("b")
        elif gap == 1:
            categories.append("c")
        elif gap == 2:
            categories.append("d")
        elif gap == 3:
            categories.append("e")
        elif gap == 4:
            categories.append("f")
        elif gap == 5:
            categories.append("g")
    else:
        raise ValueError("mode must be 'formal' or 'anthony_appendix'.")
    return categories


def write_penalty_table(rows: dict, total: float, output: Path, caption: str, label: str) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Category & Interpretation & Pairs & Student mass & Penalty \\",
        r"\midrule",
    ]
    for category in ["a", "b", "c", "d", "e", "f", "g"]:
        row = rows[category]
        lines.append(
            f"{category} & {LABELS[category]} & {row['pairs']:,} & "
            f"{row['student_mass']:,.0f} & {row['contribution']:,.0f} \\\\"
        )
    lines.extend([r"\midrule", f"Total & & & & {total:,.0f} \\\\", r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: object) -> str:
    text = str(value)
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


if __name__ == "__main__":
    main()
