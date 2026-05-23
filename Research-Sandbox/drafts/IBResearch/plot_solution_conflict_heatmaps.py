from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

from src.anthony_model import load_default_data, parse_day_series, prepare_toy_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot solution assignment heatmaps with same-slot conflicts highlighted.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clean-run-dir", type=Path, default=Path("clean_runs_20260508"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--toy-solution", type=Path, default=None)
    parser.add_argument("--full-solution", type=Path, default=None)
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)

    toy_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    toy_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    toy_exams, toy_days, toy_pairs = prepare_toy_inputs(toy_exams, toy_pairs)
    toy_solution = args.toy_solution or args.clean_run_dir / "toy_antony_verbatim_mip.csv"
    plot_solution_conflict_heatmap(
        timetable=pd.read_csv(toy_solution),
        exams=toy_exams,
        days=toy_days,
        pairs=toy_pairs,
        output=args.figure_dir / "toy_best_solution_conflict_heatmap.png",
        title="Toy best solution (proven optimal)",
        inactive_weekends=True,
        inactive_may_first=True,
        max_label_chars=28,
    )

    full_exams, full_days, full_pairs = load_default_data(args.data_dir)
    full_solution = args.full_solution or args.clean_run_dir / "full_lns_nb34_guarded_6x120.csv"
    plot_solution_conflict_heatmap(
        timetable=pd.read_csv(full_solution),
        exams=full_exams,
        days=full_days.head(34),
        pairs=full_pairs,
        output=args.figure_dir / "full_best_solution_conflict_heatmap.png",
        title="Full-instance best solution found so far (not proven optimal)",
        inactive_weekends=True,
        inactive_may_first=True,
        max_label_chars=34,
    )


def plot_solution_conflict_heatmap(
    *,
    timetable: pd.DataFrame,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    output: Path,
    title: str,
    inactive_weekends: bool,
    inactive_may_first: bool,
    max_label_chars: int,
) -> None:
    tt = _normalize_timetable(timetable)
    days_clean = days.copy()
    days_clean["Date"] = parse_day_series(days_clean["Date"])
    days_clean = days_clean.reset_index(drop=True)
    pairs_clean = _normalize_pairs(pairs, tt["Exam_Name"].tolist())

    exam_names = _exam_order(exams, tt)
    columns = [(date, slot) for date in days_clean["Date"] for slot in ("AM", "PM")]
    inactive_columns = _inactive_columns(days_clean, inactive_weekends, inactive_may_first)

    placement = {row.Exam_Name: (row.Date, row.Slot) for row in tt.itertuples(index=False)}
    conflict_exams = _same_slot_conflict_exams(tt, pairs_clean)

    values = []
    assigned_cells: list[tuple[int, int]] = []
    conflict_cells: list[tuple[int, int]] = []
    for row_idx, exam in enumerate(exam_names):
        row = []
        assigned = placement.get(exam)
        for col_idx, column in enumerate(columns):
            if column in inactive_columns:
                value = 1
            else:
                value = 0
            if assigned == column:
                value = 3 if exam in conflict_exams else 2
                assigned_cells.append((row_idx, col_idx))
                if exam in conflict_exams:
                    conflict_cells.append((row_idx, col_idx))
            row.append(value)
        values.append(row)

    width = max(10.0, min(22.0, len(columns) * 0.32))
    height = max(5.0, min(24.0, len(exam_names) * 0.24))
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = ListedColormap(["#f4f4f4", "#d9d9d9", "#4e79a7", "#d62728"])
    ax.imshow(values, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=3)

    for row_idx, col_idx in assigned_cells:
        ax.add_patch(
            mpatches.Rectangle(
                (col_idx - 0.5, row_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#1f1f1f",
                linewidth=0.45,
            )
        )
    for row_idx, col_idx in conflict_cells:
        ax.add_patch(
            mpatches.Rectangle(
                (col_idx - 0.5, row_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#111111",
                linewidth=1.7,
            )
        )

    ax.set_title(title)
    ax.set_xlabel("Date and slot")
    ax.set_ylabel("Exam")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{date.strftime('%m-%d')}\n{slot}" for date, slot in columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(exam_names)))
    ax.set_yticklabels([_shorten(name, max_label_chars) for name in exam_names], fontsize=6)
    ax.set_xticks([idx - 0.5 for idx in range(len(columns) + 1)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(len(exam_names) + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_items = [
        mpatches.Patch(facecolor="#f4f4f4", edgecolor="white", label="Active unassigned cell"),
        mpatches.Patch(facecolor="#d9d9d9", edgecolor="white", label="Inactive cell"),
        mpatches.Patch(facecolor="#4e79a7", edgecolor="#1f1f1f", label="Assigned exam"),
        mpatches.Patch(facecolor="#d62728", edgecolor="#111111", label="Assigned exam in same-slot conflict"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    conflict_mass = _same_slot_conflict_mass(tt, pairs_clean)
    print(f"Saved {output}")
    print(f"  assigned exams: {len(tt)}")
    print(f"  exams in same-slot conflicts: {len(conflict_exams)}")
    print(f"  same-slot conflict mass: {conflict_mass:g}")


def _normalize_timetable(timetable: pd.DataFrame) -> pd.DataFrame:
    tt = timetable.copy()
    required = {"Exam_Name", "Date", "Slot"}
    missing = required.difference(tt.columns)
    if missing:
        raise ValueError(f"Timetable is missing columns: {sorted(missing)}")
    tt["Exam_Name"] = tt["Exam_Name"].astype(str).str.strip()
    tt["Date"] = pd.to_datetime(tt["Date"]).dt.normalize()
    tt["Slot"] = tt["Slot"].astype(str).str.strip().str.upper()
    return tt.sort_values(["Date", "Slot", "Exam_Name"]).reset_index(drop=True)


def _normalize_pairs(pairs: pd.DataFrame, exam_names: list[str]) -> pd.DataFrame:
    pair_matrix = pairs.copy()
    if not set(exam_names).issubset(set(pair_matrix.index)):
        pair_matrix = pair_matrix.set_index(pair_matrix.columns[0])
    pair_matrix.index = pair_matrix.index.astype(str).str.strip()
    pair_matrix.columns = pair_matrix.columns.astype(str).str.strip()
    return pair_matrix.apply(pd.to_numeric, errors="raise")


def _exam_order(exams: pd.DataFrame, timetable: pd.DataFrame) -> list[str]:
    if "Full Name" in exams.columns:
        known = exams["Full Name"].astype(str).str.strip().tolist()
    elif "FULL_NAME" in exams.columns:
        known = exams["FULL_NAME"].astype(str).str.strip().tolist()
    elif "Exam_Name" in exams.columns:
        known = exams["Exam_Name"].astype(str).str.strip().tolist()
    else:
        known = timetable["Exam_Name"].astype(str).str.strip().tolist()
    scheduled = set(timetable["Exam_Name"])
    ordered = [exam for exam in known if exam in scheduled]
    extras = sorted(scheduled.difference(ordered))
    return ordered + extras


def _inactive_columns(days: pd.DataFrame, inactive_weekends: bool, inactive_may_first: bool) -> set[tuple[pd.Timestamp, str]]:
    inactive_dates: set[pd.Timestamp] = set()
    if inactive_weekends:
        inactive_dates.update(days.loc[days["DOW"].isin(["Sat", "Sun"]), "Date"].tolist())
    if inactive_may_first:
        inactive_dates.update(days.loc[(days["Date"].dt.month == 5) & (days["Date"].dt.day == 1), "Date"].tolist())
    return {(date, slot) for date in inactive_dates for slot in ("AM", "PM")}


def _same_slot_conflict_exams(timetable: pd.DataFrame, pairs: pd.DataFrame) -> set[str]:
    conflict_exams: set[str] = set()
    for (_date, _slot), group in timetable.groupby(["Date", "Slot"], sort=False):
        exams = group["Exam_Name"].tolist()
        for i, exam_i in enumerate(exams):
            for exam_j in exams[i + 1 :]:
                if float(pairs.loc[exam_i, exam_j]) > 0:
                    conflict_exams.update([exam_i, exam_j])
    return conflict_exams


def _same_slot_conflict_mass(timetable: pd.DataFrame, pairs: pd.DataFrame) -> float:
    total = 0.0
    for (_date, _slot), group in timetable.groupby(["Date", "Slot"], sort=False):
        exams = group["Exam_Name"].tolist()
        for i, exam_i in enumerate(exams):
            for exam_j in exams[i + 1 :]:
                total += float(pairs.loc[exam_i, exam_j])
    return total


def _shorten(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


if __name__ == "__main__":
    main()
