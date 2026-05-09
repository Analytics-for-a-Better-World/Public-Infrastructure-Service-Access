from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.anthony_model import load_default_data, prepare_anthony_model_data
from src.spread_improvement import spread_diagnostics


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "clean_runs_20260508"


def clock_formatter(value: float, _pos: int) -> str:
    seconds = max(0, int(round(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def parse_log(path: Path) -> dict[str, float | int | str | bool | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float | int | str | bool | None] = {
        "log": path.name,
        "runtime_seconds": None,
        "nodes": None,
        "iterations": None,
        "incumbent": None,
        "best_bound": None,
        "gap_percent": None,
        "proved": "Optimal solution found" in text,
        "time_limit": "Time limit reached" in text,
    }
    explored = re.search(
        r"Explored\s+([0-9]+)\s+nodes\s+\(([0-9]+)\s+simplex iterations\)\s+in\s+([0-9.]+)\s+seconds",
        text,
    )
    if explored:
        result["nodes"] = int(explored.group(1))
        result["iterations"] = int(explored.group(2))
        result["runtime_seconds"] = float(explored.group(3))

    best = re.search(
        r"Best objective\s+([0-9.eE+-]+), best bound\s+([0-9.eE+-]+), gap\s+([0-9.]+)%",
        text,
    )
    if best:
        result["incumbent"] = float(best.group(1))
        result["best_bound"] = float(best.group(2))
        result["gap_percent"] = float(best.group(3))
    return result


def summarize_mip_logs() -> None:
    toy_logs = {
        "Appendix-style verbatim": "toy_antony_verbatim.log",
        "Reusable baseline": "toy_antony_baseline.log",
        "Seeded 10 min": "toy_seeded_10min.log",
        "Baseline 300s": "toy_exp_baseline_300s.log",
        "Binary y 300s": "toy_exp_ybin_300s.log",
        "Upper-link bounds 300s": "toy_exp_yub_300s.log",
        "Paper order 300s": "toy_exp_order_300s.log",
        "Binary y + paper order 300s": "toy_exp_ybin_order_300s.log",
        "Paper order + Symmetry=2 300s": "toy_exp_order_sym2_300s.log",
    }
    full_logs = {
        "Guarded start + paper order + Symmetry=2, 10 min": "full_mip_guarded_order_sym2_10min.log",
        "Guarded start + binary y + paper order + Symmetry=2, 5 min": "full_mip_guarded_ybin_order_sym2_5min.log",
        "Guarded start + proximity at most one, 5 min": "full_mip_guarded_prox1_5min.log",
        "Guarded start + MIPFocus=3, Cuts=2, 5 min": "full_mip_guarded_focus3_cuts2_5min.log",
    }
    for name, mapping in [("toy_mip_summary.csv", toy_logs), ("full_mip_summary.csv", full_logs)]:
        rows = []
        for label, filename in mapping.items():
            row = parse_log(RUN_DIR / filename)
            row["run"] = label
            rows.append(row)
        pd.DataFrame(rows).to_csv(RUN_DIR / name, index=False)


def summarize_lns_histories() -> None:
    histories = {
        "23-day mixed 8x90": "full_lns_23day_8x90_history.csv",
        "23-day focused 4x120": "full_lns_23day_focused_4x120_history.csv",
        "34-day pilot 3x120": "full_lns_nb34_pilot_3x120_history.csv",
        "34-day recommended 6x90": "full_lns_nb34_recommended_6x90_history.csv",
        "34-day guarded 6x120": "full_lns_nb34_guarded_6x120_history.csv",
    }
    all_rows = []
    for run, filename in histories.items():
        df = pd.read_csv(RUN_DIR / filename)
        df.insert(0, "run", run)
        all_rows.append(df)
    histories_df = pd.concat(all_rows, ignore_index=True)
    histories_df.to_csv(RUN_DIR / "clean_lns_history_all.csv", index=False)

    accepted = histories_df[histories_df["accepted"] == True].copy()
    accepted["improvement"] = accepted["start_objective"] - accepted["candidate_objective"]
    strategy = (
        accepted.groupby("strategy", as_index=False)["improvement"]
        .sum()
        .sort_values("improvement", ascending=False)
    )
    strategy.to_csv(RUN_DIR / "clean_lns_strategy_contributions.csv", index=False)

    run_summary = (
        histories_df.groupby("run", as_index=False)
        .agg(
            start_objective=("start_objective", "first"),
            final_objective=("candidate_objective", lambda s: float(s.dropna().iloc[-1])),
            accepted_moves=("accepted", "sum"),
        )
        .sort_values("run")
    )
    run_summary["improvement"] = run_summary["start_objective"] - run_summary["final_objective"]
    run_summary.to_csv(RUN_DIR / "clean_lns_run_summary.csv", index=False)


def summarize_solution_diagnostics() -> None:
    exams, days, pairs = load_default_data(ROOT / "data")
    data23 = prepare_anthony_model_data(exams, days, pairs, nb_days=23)
    data34 = prepare_anthony_model_data(exams, days, pairs, nb_days=34)
    timetables = [
        ("Initial heuristic, 23 days", "full_heuristic_rounds2.csv", data23),
        ("LNS mixed, 23 days", "full_lns_23day_8x90.csv", data23),
        ("LNS focused, 23 days", "full_lns_23day_focused_4x120.csv", data23),
        ("LNS pilot, 34 days", "full_lns_nb34_pilot_3x120.csv", data34),
        ("Recommended LNS, 34 days", "full_lns_nb34_recommended_6x90.csv", data34),
        ("Guarded LNS, 34 days", "full_lns_nb34_guarded_6x120.csv", data34),
    ]
    rows = []
    for label, filename, data in timetables:
        tt = pd.read_csv(RUN_DIR / filename)
        diag = spread_diagnostics(tt, data, objective_mode="formal")
        rows.append({"run": label, **diag.__dict__})
    pd.DataFrame(rows).to_csv(RUN_DIR / "clean_solution_diagnostics.csv", index=False)


def plot_toy_strengthening() -> None:
    variants = [
        ("baseline", "toy_exp_baseline_300s_progress.csv", "#1f77b4"),
        ("y binary", "toy_exp_ybin_300s_progress.csv", "#ff7f0e"),
        ("y upper bounds", "toy_exp_yub_300s_progress.csv", "#2ca02c"),
        ("paper order", "toy_exp_order_300s_progress.csv", "#d62728"),
        ("y binary + paper order", "toy_exp_ybin_order_300s_progress.csv", "#9467bd"),
        ("paper order + symmetry=2", "toy_exp_order_sym2_300s_progress.csv", "#8c564b"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, filename, color in variants:
        df = pd.read_csv(RUN_DIR / filename)
        if "best_bound" in df and not df["best_bound"].dropna().empty:
            ax.step(df["time_seconds"], df["best_bound"], where="post", label=label, linewidth=3.0, color=color)
    ax.axhline(25190, color="black", linestyle="--", linewidth=2.0, label="optimum 25190")
    ax.set_xlabel("Time")
    ax.set_ylabel("Best bound")
    ax.set_title("Toy MILP bound comparison by strengthening variant")
    ax.xaxis.set_major_formatter(FuncFormatter(clock_formatter))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "toy_strengthening_bound_comparison_clean.png", dpi=200)
    plt.close(fig)


def plot_full_summary() -> None:
    df = pd.read_csv(RUN_DIR / "clean_solution_diagnostics.csv")
    labels = [
        "Initial\n23d",
        "Mixed\n23d",
        "Focused\n23d",
        "Pilot\n34d",
        "Recommended\n34d",
        "Guarded\n34d",
    ]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(labels, df["objective_value"] / 1_000_000, marker="o", linewidth=3.0, color="#1f77b4", label="Objective (millions)")
    ax1.set_ylabel("Objective (millions)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(labels, df["same_slot_clashes"], marker="s", linewidth=3.0, color="#d62728", label="Same-slot clashes")
    ax2.set_ylabel("Same-slot clashes")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax1.set_title("Clean full-instance rerun summary")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "full_instance_summary_clean.png", dpi=200)
    plt.close(fig)


def main() -> None:
    summarize_mip_logs()
    summarize_lns_histories()
    summarize_solution_diagnostics()
    plot_toy_strengthening()
    plot_full_summary()
    print(f"Wrote clean summaries and plots to {RUN_DIR}")


if __name__ == "__main__":
    main()
