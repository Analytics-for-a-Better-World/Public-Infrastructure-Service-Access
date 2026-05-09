from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the toy MILP as close as practical to Antony Furlong's Appendix 1.3 code."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("toy_antony_verbatim_mip.csv"))
    parser.add_argument("--progress-output", type=Path, default=Path("toy_antony_verbatim_progress.csv"))
    parser.add_argument("--log-output", type=Path, default=Path("toy_antony_verbatim.log"))
    parser.add_argument("--plot-output", type=Path, default=Path("toy_antony_verbatim_bounds.png"))
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()

    import gurobipy as gp
    from gurobipy import GRB

    exams = pd.read_csv(args.data_dir / "Toy exam list.csv").head(20).reset_index(drop=True)
    pairs = pd.read_csv(args.data_dir / "Exam pairs.csv", index_col=0)
    days = pd.read_csv(args.data_dir / "exam_days3.csv").head(16).reset_index(drop=True)

    nb_exam = 20
    nb_day = 16
    nb_slot = 2
    max_clashes = 15
    weights = {"a": 64, "b": 32, "c": 16, "d": 8, "e": 4, "f": 2, "g": 1}
    c = pairs.to_numpy()

    weekend_days = days[days["DOW"].isin(["Sat", "Sun"])].index.tolist()
    may_first = days[days["Date"] == "01/05/2026"].index.tolist()

    model = gp.Model("toy_antony_verbatim")
    model.setParam("OutputFlag", 1)
    model.setParam("LogFile", str(args.log_output))
    if args.time_limit is not None:
        model.setParam("TimeLimit", args.time_limit)

    x = {}
    for i in range(nb_exam):
        for p in range(nb_slot):
            for m in range(nb_day):
                x[i, p, m] = model.addVar(vtype=GRB.BINARY, name=f"x[{i},{p},{m}]")

    y = {}
    for category in weights:
        for i in range(nb_exam):
            for j in range(nb_exam):
                if i < j:
                    y[category, i, j] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0,
                        ub=1.0,
                        name=f"y_{category}[{i},{j}]",
                    )

    model.update()

    objective = gp.LinExpr()
    for i in range(nb_exam):
        for j in range(nb_exam):
            if i < j:
                for category, weight in weights.items():
                    objective += float(c[i, j]) * weight * y[category, i, j]
    model.setObjective(objective, GRB.MINIMIZE)

    for i in range(nb_exam):
        for j in range(nb_exam):
            if i < j:
                for p in range(nb_slot):
                    for m in range(nb_day):
                        model.addConstr(y["a", i, j] >= x[i, p, m] + x[j, p, m] - 1)

                for m in range(nb_day):
                    model.addConstr(
                        y["b", i, j] >= x[i, 0, m] + x[j, 0, m] + x[i, 1, m] + x[j, 1, m] - 1
                    )

                for m in range(nb_day - 1):
                    model.addConstr(
                        y["c", i, j] >= x[i, 0, m] + x[j, 0, m + 1] + x[i, 1, m] + x[j, 1, m + 1] - 1
                    )
                    model.addConstr(
                        y["c", i, j] >= x[i, 0, m + 1] + x[j, 0, m] + x[i, 1, m + 1] + x[j, 1, m] - 1
                    )

                for m in range(nb_day - 2):
                    model.addConstr(
                        y["d", i, j] >= x[i, 0, m] + x[j, 0, m + 2] + x[i, 1, m] + x[j, 1, m + 2] - 1
                    )
                    model.addConstr(
                        y["d", i, j] >= x[i, 0, m + 2] + x[j, 0, m] + x[i, 1, m + 2] + x[j, 1, m] - 1
                    )

                for m in range(nb_day - 3):
                    model.addConstr(
                        y["e", i, j] >= x[i, 0, m] + x[j, 0, m + 3] + x[i, 1, m] + x[j, 1, m + 3] - 1
                    )
                    model.addConstr(
                        y["e", i, j] >= x[i, 0, m + 3] + x[j, 0, m] + x[i, 1, m + 3] + x[j, 1, m] - 1
                    )

                for m in range(nb_day - 4):
                    model.addConstr(
                        y["f", i, j] >= x[i, 0, m] + x[j, 0, m + 4] + x[i, 1, m] + x[j, 1, m + 4] - 1
                    )
                    model.addConstr(
                        y["f", i, j] >= x[i, 0, m + 4] + x[j, 0, m] + x[i, 1, m + 4] + x[j, 1, m] - 1
                    )

                # Appendix 1.3 labels this block y_g but links the constraints to y_c.
                for m in range(nb_day - 5):
                    model.addConstr(
                        y["c", i, j] >= x[i, 0, m] + x[j, 0, m + 5] + x[i, 1, m] + x[j, 1, m + 5] - 1
                    )
                    model.addConstr(
                        y["c", i, j] >= x[i, 0, m + 5] + x[j, 0, m] + x[i, 1, m + 5] + x[j, 1, m] - 1
                    )

    for i in range(nb_exam):
        model.addConstr(gp.quicksum(x[i, p, m] for p in range(nb_slot) for m in range(nb_day)) == 1)

    for i in range(nb_exam):
        for p in range(nb_slot):
            for m in weekend_days:
                model.addConstr(x[i, p, m] == 0)

    for i in exams[exams["Subject"] == "Language A Literature"].index.tolist():
        for p in range(nb_slot):
            for m in days[days["DOW"] == "Fri"].index.tolist():
                model.addConstr(x[i, p, m] == 0)

    for i in range(nb_exam):
        for p in range(nb_slot):
            for m in may_first:
                model.addConstr(x[i, p, m] == 0)

    model.addConstr(
        gp.quicksum(float(c[i, j]) * y["a", i, j] for i in range(nb_exam) for j in range(nb_exam) if i < j)
        <= max_clashes
    )

    for i in range(nb_exam):
        for m in range(nb_day):
            if float(exams.loc[i, "Length"]) > 3:
                model.addConstr(x[i, 1, m] == 0)

    for i in range(nb_exam):
        for j in range(nb_exam):
            if i < j and float(exams.loc[i, "Length"]) + float(exams.loc[j, "Length"]) > 6.25:
                model.addConstr(y["b", i, j] == 0)

    for i in exams[exams["Full Name"] == "SBS Exam 1"].index.tolist():
        model.addConstr(x[i, 0, 0] == 1)

    for i in exams[exams["Subject"] != "SBS"].index.tolist():
        for p in range(nb_slot):
            model.addConstr(x[i, p, 0] + x[i, p, 1] == 0)

    for i in range(nb_exam):
        for j in range(nb_exam):
            if i < j and exams.loc[i, "Subject"] == exams.loc[j, "Subject"]:
                for p in range(nb_slot):
                    for m in range(nb_day):
                        model.addConstr(
                            x[j, p, m]
                            + gp.quicksum(
                                x[i, q, n]
                                for q in range(nb_slot)
                                for n in range(nb_day)
                                if n not in (m - 1, m + 1)
                            )
                            <= 1
                        )

    for i in exams[exams["Subject"].isin(["Finance", "Law and Ethics"])].index.tolist():
        model.addConstr(
            gp.quicksum(x[i, p, m] for p in range(nb_slot) for m in range(round(nb_day / 2), nb_day)) == 0
        )

    progress: list[dict[str, float]] = []

    def callback(cb_model, where):
        if where == GRB.Callback.MIP:
            runtime = float(cb_model.cbGet(GRB.Callback.RUNTIME))
            incumbent = float(cb_model.cbGet(GRB.Callback.MIP_OBJBST))
            bound = float(cb_model.cbGet(GRB.Callback.MIP_OBJBND))
            if abs(incumbent) >= 1e100:
                gap = float("nan")
            else:
                gap = abs(incumbent - bound) / max(1.0, abs(incumbent))
            progress.append(
                {
                    "time_seconds": runtime,
                    "incumbent": incumbent,
                    "best_bound": bound,
                    "gap": gap,
                }
            )

    start = time.perf_counter()
    model.optimize(callback)
    elapsed = time.perf_counter() - start

    rows = []
    if model.SolCount:
        for i in range(nb_exam):
            for p in range(nb_slot):
                for m in range(nb_day):
                    if x[i, p, m].X > 0.5:
                        rows.append(
                            {
                                "Day_of_Week": days.loc[m, "DOW"],
                                "Date": pd.to_datetime(days.loc[m, "Date"], dayfirst=True).date().isoformat(),
                                "Slot": "AM" if p == 0 else "PM",
                                "Exam_Name": exams.loc[i, "Full Name"],
                            }
                        )
    pd.DataFrame(rows).to_csv(args.output, index=False)

    progress_df = pd.DataFrame(progress)
    if not progress_df.empty:
        progress_df = progress_df.replace([float("inf"), -float("inf")], pd.NA).dropna(
            subset=["incumbent", "best_bound"]
        )
        progress_df["time_seconds"] = progress_df["time_seconds"].round(3)
        progress_df = progress_df.drop_duplicates(subset=["time_seconds", "incumbent", "best_bound"]).reset_index(
            drop=True
        )
    progress_df.to_csv(args.progress_output, index=False)
    _plot_progress(progress_df, args.plot_output)

    print("MILP seconds:", round(elapsed, 6))
    print("MILP status:", int(model.Status))
    print("MILP objective:", float(model.ObjVal) if model.SolCount else None)
    print("MILP best bound:", float(model.ObjBound) if model.SolCount else None)
    print("MILP gap:", float(model.MIPGap) if model.SolCount else None)
    print(f"Saved MILP timetable to {args.output}")
    print(f"Saved MILP progress to {args.progress_output}")
    print(f"Saved MILP log to {args.log_output}")
    print(f"Saved bound plot to {args.plot_output}")


def _plot_progress(progress: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not progress.empty:
        ax.plot(progress["time_seconds"], progress["incumbent"], label="Incumbent", color="#d62728", linewidth=3.0)
        ax.plot(progress["time_seconds"], progress["best_bound"], label="Best bound", color="#1f77b4", linewidth=3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_clock_time))
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Objective")
    ax.set_title("Toy Appendix-style MILP incumbent and bound evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _format_clock_time(seconds: float, _pos: int | None = None) -> str:
    if pd.isna(seconds):
        return ""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    main()
