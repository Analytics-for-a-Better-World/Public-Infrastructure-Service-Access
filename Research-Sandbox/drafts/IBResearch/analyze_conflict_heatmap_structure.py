from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_solution_conflict_heatmaps import _exam_order, _normalize_pairs, _normalize_timetable
from src.anthony_model import load_default_data, parse_day_series, prepare_toy_inputs


SLOTS = ("AM", "PM")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ordering and conflict patterns in solution heatmaps.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clean-run-dir", type=Path, default=Path("clean_runs_20260508"))
    parser.add_argument("--output-dir", type=Path, default=Path("clean_runs_20260508/conflict_heatmap_diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    toy_exams = pd.read_csv(args.data_dir / "Toy exam list.csv")
    toy_pairs = pd.read_csv(args.data_dir / "Exam pairs.csv")
    toy_exams, toy_days, toy_pairs = prepare_toy_inputs(toy_exams, toy_pairs)
    toy = analyse_instance(
        name="toy",
        timetable=pd.read_csv(args.clean_run_dir / "toy_antony_verbatim_mip.csv"),
        exams=toy_exams,
        days=toy_days,
        pairs=toy_pairs,
        output_dir=args.output_dir,
    )

    full_exams, full_days, full_pairs = load_default_data(args.data_dir)
    full = analyse_instance(
        name="full",
        timetable=pd.read_csv(args.clean_run_dir / "full_lns_nb34_guarded_6x120.csv"),
        exams=full_exams,
        days=full_days.head(34),
        pairs=full_pairs,
        output_dir=args.output_dir,
    )

    summary = [toy, full]
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_diagonal_distance(summary, args.output_dir / "assignment_distance_from_diagonal.png")
    plot_pair_distance(summary, args.output_dir / "coenrollment_weight_by_source_row_distance.png")
    print(json.dumps(summary, indent=2))


def analyse_instance(
    *,
    name: str,
    timetable: pd.DataFrame,
    exams: pd.DataFrame,
    days: pd.DataFrame,
    pairs: pd.DataFrame,
    output_dir: Path,
) -> dict:
    tt = _normalize_timetable(timetable)
    days_clean = days.copy()
    days_clean["Date"] = parse_day_series(days_clean["Date"])
    days_clean = days_clean.reset_index(drop=True)
    pairs_clean = _normalize_pairs(pairs, tt["Exam_Name"].tolist())

    source_order = _exam_order(exams, tt)
    chronological_order = tt.sort_values(["Date", "Slot", "Exam_Name"])["Exam_Name"].tolist()
    columns = [(date, slot) for date in days_clean["Date"] for slot in SLOTS]
    col_index = {col: idx for idx, col in enumerate(columns)}
    source_index = {exam: idx for idx, exam in enumerate(source_order)}
    chrono_index = {exam: idx for idx, exam in enumerate(chronological_order)}
    placement = {row.Exam_Name: (row.Date, row.Slot) for row in tt.itertuples(index=False)}

    conflict_pairs = same_slot_conflict_pairs(tt, pairs_clean)
    conflict_exams = sorted({exam for item in conflict_pairs for exam in (item["exam_i"], item["exam_j"])})
    pd.DataFrame(conflict_pairs).to_csv(output_dir / f"{name}_same_slot_conflict_pairs.csv", index=False)

    assigned_source = [(source_index[exam], col_index[placement[exam]]) for exam in source_order if exam in placement]
    conflict_source = [(source_index[exam], col_index[placement[exam]]) for exam in conflict_exams]

    plot_assignment_diagnostic(
        row_count=len(source_order),
        col_count=len(columns),
        assigned=assigned_source,
        conflict=conflict_source,
        output=output_dir / f"{name}_source_order_assignment_diagnostic.png",
        title=f"{name.title()} solution in source-list row order",
    )

    assigned_chrono = [(chrono_index[exam], col_index[placement[exam]]) for exam in chronological_order if exam in placement]
    conflict_chrono = [(chrono_index[exam], col_index[placement[exam]]) for exam in conflict_exams]
    plot_assignment_diagnostic(
        row_count=len(chronological_order),
        col_count=len(columns),
        assigned=assigned_chrono,
        conflict=conflict_chrono,
        output=output_dir / f"{name}_chronological_row_assignment_diagnostic.png",
        title=f"{name.title()} solution in chronological row order",
    )

    return {
        "instance": name,
        "n_exams": len(source_order),
        "n_date_slots": len(columns),
        "same_slot_conflict_exam_count": len(conflict_exams),
        "same_slot_conflict_pair_count": len(conflict_pairs),
        "same_slot_conflict_mass": float(sum(item["weight"] for item in conflict_pairs)),
        "assigned": diagonal_stats(assigned_source, len(source_order), len(columns)),
        "assigned_random_diag_distance": random_diagonal_distance(len(assigned_source), len(source_order), len(columns), seed=17),
        "conflict": diagonal_stats(conflict_source, len(source_order), len(columns)),
        "conflict_random_diag_distance": random_diagonal_distance(len(conflict_source), len(source_order), len(columns), seed=29)
        if conflict_source
        else None,
        "pair_order": pair_order_stats(source_order, pairs_clean),
        "conflict_components": conflict_components(conflict_pairs),
    }


def same_slot_conflict_pairs(timetable: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    conflicts = []
    for (date, slot), group in timetable.groupby(["Date", "Slot"], sort=False):
        exams = group["Exam_Name"].tolist()
        for i, exam_i in enumerate(exams):
            for exam_j in exams[i + 1 :]:
                weight = float(pairs.loc[exam_i, exam_j])
                if weight > 0:
                    conflicts.append(
                        {
                            "date": date.date().isoformat(),
                            "slot": slot,
                            "exam_i": exam_i,
                            "exam_j": exam_j,
                            "weight": weight,
                        }
                    )
    return conflicts


def diagonal_stats(cells: list[tuple[int, int]], n_rows: int, n_cols: int) -> dict:
    if not cells:
        return {"count": 0, "pearson": None, "mean_norm_diag_dist": None, "median_norm_diag_dist": None}
    rows = np.array([row for row, _col in cells], dtype=float)
    cols = np.array([col for _row, col in cells], dtype=float)
    row_norm = rows / max(n_rows - 1, 1)
    col_norm = cols / max(n_cols - 1, 1)
    pearson = float(np.corrcoef(row_norm, col_norm)[0, 1]) if len(cells) > 1 else None
    distances = np.abs(row_norm - col_norm)
    return {
        "count": len(cells),
        "pearson": pearson,
        "mean_norm_diag_dist": float(distances.mean()),
        "median_norm_diag_dist": float(np.median(distances)),
    }


def random_diagonal_distance(count: int, n_rows: int, n_cols: int, *, seed: int, samples: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        rows = rng.choice(n_rows, size=count, replace=False if count <= n_rows else True)
        cols = rng.choice(n_cols, size=count, replace=False if count <= n_cols else True)
        row_norm = rows / max(n_rows - 1, 1)
        col_norm = cols / max(n_cols - 1, 1)
        means.append(float(np.abs(row_norm - col_norm).mean()))
    values = np.array(means)
    return {
        "random_mean": float(values.mean()),
        "random_p10": float(np.percentile(values, 10)),
        "random_p90": float(np.percentile(values, 90)),
    }


def pair_order_stats(exam_order: list[str], pairs: pd.DataFrame) -> dict:
    exam_index = {exam: idx for idx, exam in enumerate(exam_order)}
    weighted_sum = 0.0
    total_weight = 0.0
    positive_pairs = 0
    threshold_weights = defaultdict(float)
    for pos, exam_i in enumerate(exam_order):
        for exam_j in exam_order[pos + 1 :]:
            weight = float(pairs.loc[exam_i, exam_j])
            if weight <= 0:
                continue
            positive_pairs += 1
            distance = abs(exam_index[exam_i] - exam_index[exam_j])
            weighted_sum += weight * distance
            total_weight += weight
            for threshold in (1, 3, 6):
                if distance <= threshold:
                    threshold_weights[threshold] += weight
    return {
        "positive_pairs": positive_pairs,
        "weighted_mean_row_distance": weighted_sum / total_weight if total_weight else 0.0,
        "share_weight_distance_le_1": threshold_weights[1] / total_weight if total_weight else 0.0,
        "share_weight_distance_le_3": threshold_weights[3] / total_weight if total_weight else 0.0,
        "share_weight_distance_le_6": threshold_weights[6] / total_weight if total_weight else 0.0,
    }


def conflict_components(conflicts: list[dict]) -> list[dict]:
    graph: dict[str, set[str]] = defaultdict(set)
    weights: dict[tuple[str, str], float] = {}
    for item in conflicts:
        a, b = item["exam_i"], item["exam_j"]
        graph[a].add(b)
        graph[b].add(a)
        weights[tuple(sorted((a, b)))] = float(item["weight"])
    seen: set[str] = set()
    components = []
    for start in sorted(graph):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        nodes = []
        while queue:
            node = queue.popleft()
            nodes.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        mass = sum(weight for (a, b), weight in weights.items() if a in nodes and b in nodes)
        components.append({"size": len(nodes), "mass": mass, "exams": sorted(nodes)})
    return sorted(components, key=lambda item: (-item["size"], -item["mass"], item["exams"]))


def plot_assignment_diagnostic(
    *,
    row_count: int,
    col_count: int,
    assigned: list[tuple[int, int]],
    conflict: list[tuple[int, int]],
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if assigned:
        rows, cols = zip(*assigned)
        ax.scatter(cols, rows, s=28, color="#4e79a7", label="assigned", alpha=0.85)
    if conflict:
        rows, cols = zip(*conflict)
        ax.scatter(cols, rows, s=62, facecolors="none", edgecolors="#d62728", linewidths=1.8, label="same-slot conflict")
    ax.plot([0, col_count - 1], [0, row_count - 1], color="#333333", linewidth=1.0, linestyle="--", label="normalized diagonal")
    ax.set_xlim(-1, col_count)
    ax.set_ylim(row_count, -1)
    ax.set_xlabel("Date-slot column")
    ax.set_ylabel("Exam row")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_diagonal_distance(summary: list[dict], output: Path) -> None:
    labels = []
    values = []
    colors = []
    for item in summary:
        labels.extend([f"{item['instance']} assigned", f"{item['instance']} random"])
        values.extend([item["assigned"]["mean_norm_diag_dist"], item["assigned_random_diag_distance"]["random_mean"]])
        colors.extend(["#4e79a7", "#bab0ac"])
        if item["conflict"]["count"]:
            labels.extend([f"{item['instance']} conflict", f"{item['instance']} conflict random"])
            values.extend([item["conflict"]["mean_norm_diag_dist"], item["conflict_random_diag_distance"]["random_mean"]])
            colors.extend(["#d62728", "#f2a6a6"])
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean normalized distance from diagonal")
    ax.set_title("Heatmap diagonal structure versus random placement")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_pair_distance(summary: list[dict], output: Path) -> None:
    labels = [item["instance"] for item in summary]
    le1 = [item["pair_order"]["share_weight_distance_le_1"] for item in summary]
    le3 = [item["pair_order"]["share_weight_distance_le_3"] for item in summary]
    le6 = [item["pair_order"]["share_weight_distance_le_6"] for item in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - 0.24, le1, width=0.24, label="distance <= 1")
    ax.bar(x, le3, width=0.24, label="distance <= 3")
    ax.bar(x + 0.24, le6, width=0.24, label="distance <= 6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of co-enrolment weight")
    ax.set_title("Co-enrolment locality in source-list order")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
