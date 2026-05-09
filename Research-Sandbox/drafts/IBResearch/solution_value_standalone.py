import pandas as pd


DEFAULT_WEIGHTS = {
    "a": 64,
    "b": 32,
    "c": 16,
    "d": 8,
    "e": 4,
    "f": 2,
    "g": 1,
}


def solution_value(
    timetable,
    pairs,
    days,
    *,
    weights=None,
    mode="anthony_appendix",
):
    """
    Score an exam timetable using Anthony Furlong's weighted pair objective.

    The timetable must contain these columns:
    - Exam_Name
    - Date
    - Slot

    The pairs matrix should contain shared candidate counts, with exam names
    as both rows and columns. If the first column contains exam names, this
    function will set it as the index automatically.

    The days table must contain a Date column in chronological order.

    mode="anthony_appendix" mimics the implementation in Anthony's Appendix
    1.3 code. mode="formal" uses the cleaner written model interpretation.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if mode not in {"anthony_appendix", "formal"}:
        raise ValueError("mode must be 'anthony_appendix' or 'formal'.")

    required_cols = {"Exam_Name", "Date", "Slot"}
    missing = required_cols - set(timetable.columns)
    if missing:
        raise ValueError(f"Timetable is missing columns: {sorted(missing)}")

    tt = timetable.copy()
    tt["Date"] = pd.to_datetime(tt["Date"], dayfirst=True).dt.normalize()

    if tt["Exam_Name"].duplicated().any():
        duplicated = tt.loc[tt["Exam_Name"].duplicated(), "Exam_Name"].tolist()
        raise ValueError(f"Duplicate exams in timetable: {duplicated}")

    placement = {
        row.Exam_Name: (row.Date, str(row.Slot))
        for row in tt.itertuples(index=False)
    }

    day_list = pd.to_datetime(days["Date"], dayfirst=True).dt.normalize().tolist()
    day_index = {date: i for i, date in enumerate(day_list)}

    pair_matrix = pairs.copy()
    if not set(placement).issubset(set(pair_matrix.index)):
        pair_matrix = pair_matrix.set_index(pair_matrix.columns[0])

    pair_matrix.index = pair_matrix.index.astype(str).str.strip()
    pair_matrix.columns = pair_matrix.columns.astype(str).str.strip()
    pair_matrix = pair_matrix.apply(pd.to_numeric, errors="raise")

    total = 0.0
    exams = sorted(placement)

    for pos, exam_i in enumerate(exams):
        date_i, slot_i = placement[exam_i]

        if date_i not in day_index:
            raise ValueError(f"Date {date_i} for {exam_i} is not in days table.")

        for exam_j in exams[pos + 1:]:
            date_j, slot_j = placement[exam_j]

            if date_j not in day_index:
                raise ValueError(f"Date {date_j} for {exam_j} is not in days table.")

            cij = float(pair_matrix.loc[exam_i, exam_j])
            if cij == 0:
                continue

            gap = abs(day_index[date_i] - day_index[date_j])

            if gap == 0 and slot_i == slot_j:
                total += cij * weights["a"]

            if mode == "anthony_appendix":
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
