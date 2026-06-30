from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def pct(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}%"


def num(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], labels: list[str] | None = None) -> str:
    labels = labels or columns
    rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(rows)


def expand_budget_rows(df: pd.DataFrame, budgets: list[int], group_col: str) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(group_col):
        group = group.sort_values("budget")
        for budget in budgets:
            exact = group[group["budget"].astype(int) == int(budget)]
            if not exact.empty:
                row = exact.iloc[-1].to_dict()
                row["effective_approx_budget"] = int(row["budget"])
            else:
                lower = group[group["budget"].astype(int) <= int(budget)]
                if lower.empty:
                    continue
                row = lower.iloc[-1].to_dict()
                row["effective_approx_budget"] = int(row["budget"])
                row["budget"] = int(budget)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "grid_experiment_report")
    parser.add_argument(
        "--timor-dir",
        type=Path,
        default=ROOT / "outputs" / "timor_three_grid_experiments",
    )
    parser.add_argument(
        "--vietnam-selected-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "vietnam_20260619_0630"
        / "fleur_style_10km_network_selected_p_deterministic",
    )
    parser.add_argument(
        "--vietnam-approx",
        type=Path,
        default=ROOT / "outputs" / "approx_tradeoff" / "vietnam_10km_approx_curves.csv",
    )
    parser.add_argument(
        "--vietnam-dense-exact-dir",
        type=Path,
        default=ROOT / "outputs" / "vietnam_20260619_0630" / "dense_grid_straightline" / "grid_5000m_exact",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timor_selected = pd.read_csv(args.timor_dir / "timor_selected_budget_comparison.csv")
    timor_stats_path = args.timor_dir / "timor_exact_unit_stats_10km_5km_1km.csv"
    if not timor_stats_path.exists():
        timor_stats_path = args.timor_dir / "timor_exact_unit_stats_10km_5km.csv"
    timor_stats = pd.read_csv(timor_stats_path)
    timor_approx_gain = pd.read_csv(args.timor_dir / "timor_approx_gain_selected_p.csv")
    timor_approx_quality = pd.read_csv(args.timor_dir / "timor_approx_vs_exact_selected_p.csv")

    vietnam_summary = pd.read_csv(args.vietnam_selected_dir / "coverage_summary_by_budget.csv")
    vietnam_best = vietnam_summary.sort_values(
        ["threshold_km", "budget", "total_covered_population", "seconds"],
        ascending=[True, True, False, True],
    ).groupby(["threshold_km", "budget"], as_index=False).head(1)
    vietnam_best = vietnam_best.rename(
        columns={
            "coverage_percent_total_population": "fleur_coverage_percent",
            "total_covered_population": "fleur_covered_population",
            "seconds": "fleur_seconds",
            "method": "fleur_method",
        }
    )

    vietnam_approx = pd.read_csv(args.vietnam_approx)
    selected_budgets = sorted(timor_selected["budget"].astype(int).unique())
    vietnam_approx = expand_budget_rows(vietnam_approx, selected_budgets, "threshold_km")
    vietnam_approx = vietnam_approx.rename(
        columns={
            "coverage_percent": "approx_coverage_percent",
            "covered_population": "approx_covered_population",
            "seconds": "approx_seconds",
            "method": "approx_method",
        }
    )
    vietnam_join = vietnam_approx.merge(
        vietnam_best[
            [
                "threshold_km",
                "budget",
                "fleur_method",
                "fleur_coverage_percent",
                "fleur_covered_population",
                "fleur_seconds",
                "construction_seconds",
                "local_search_moves",
            ]
        ],
        on=["threshold_km", "budget"],
        how="left",
    )
    vietnam_join["fleur_minus_approx_pp"] = (
        vietnam_join["fleur_coverage_percent"] - vietnam_join["approx_coverage_percent"]
    )
    vietnam_join.to_csv(args.output_dir / "vietnam_10km_selected_p_approx_vs_fleur.csv", index=False)

    timor_selected.to_csv(args.output_dir / "timor_selected_p_10_5_1_gain.csv", index=False)
    timor_approx_gain.to_csv(args.output_dir / "timor_selected_p_approx_gain.csv", index=False)
    timor_approx_quality.to_csv(args.output_dir / "timor_approx_vs_exact_selected_p.csv", index=False)
    vietnam_dense_exact_path = args.vietnam_dense_exact_dir / "vietnam_1km_heuristic_vs_5km_exact.csv"
    vietnam_dense_exact = pd.read_csv(vietnam_dense_exact_path) if vietnam_dense_exact_path.exists() else pd.DataFrame()
    if not vietnam_dense_exact.empty:
        vietnam_dense_exact.to_csv(args.output_dir / "vietnam_1km_heuristic_vs_5km_exact.csv", index=False)

    timor_display = timor_selected.copy()
    for col in [
        "timor_10km_exact_coverage_percent",
        "timor_5km_exact_coverage_percent",
        "timor_1km_exact_coverage_percent",
        "timor_1km_best_heuristic_coverage_percent",
    ]:
        timor_display[col] = timor_display[col].map(pct)
    for col in [
        "gain_10km_to_5km_percentage_points",
        "gain_5km_to_1km_percentage_points",
        "timor_10km_exact_seconds",
        "timor_5km_exact_seconds",
        "timor_1km_exact_seconds",
        "timor_1km_best_heuristic_seconds",
        "timor_1km_exact_minus_heuristic_percentage_points",
    ]:
        timor_display[col] = timor_display[col].map(num)

    timor_quality_display = timor_approx_quality.copy()
    for col in ["exact_coverage_percent", "approx_coverage_percent"]:
        timor_quality_display[col] = timor_quality_display[col].map(pct)
    for col in ["exact_minus_approx_percentage_points", "exact_seconds", "approx_seconds"]:
        timor_quality_display[col] = timor_quality_display[col].map(num)

    vietnam_display = vietnam_join.copy()
    vietnam_display["threshold_km"] = vietnam_display["threshold_km"].map(lambda v: f"{float(v):g} km")
    vietnam_display["effective_approx_budget"] = vietnam_display["effective_approx_budget"].map(
        lambda v: "" if pd.isna(v) else str(int(v))
    )
    for col in ["approx_coverage_percent", "fleur_coverage_percent"]:
        vietnam_display[col] = vietnam_display[col].map(pct)
    for col in ["fleur_minus_approx_pp", "approx_seconds", "fleur_seconds"]:
        vietnam_display[col] = vietnam_display[col].map(num)

    vietnam_dense_display = vietnam_dense_exact.copy()
    if not vietnam_dense_display.empty:
        for col in ["vietnam_5km_exact_coverage_percent", "vietnam_1km_best_heuristic_coverage_percent"]:
            vietnam_dense_display[col] = vietnam_dense_display[col].map(pct)
        for col in ["vietnam_1km_heuristic_minus_5km_exact_pp"]:
            vietnam_dense_display[col] = vietnam_dense_display[col].map(lambda value: num(value, 4))
        for col in ["vietnam_5km_exact_seconds", "vietnam_1km_best_heuristic_seconds"]:
            vietnam_dense_display[col] = vietnam_dense_display[col].map(num)

    stats_display = timor_stats.copy()
    for col in ["all_candidate_coverage_percent", "exact_total_seconds"]:
        stats_display[col] = stats_display[col].map(num)

    report = "\n".join(
        [
            "# Timor/Vietnam Grid Experiment Report",
            "",
            "Timor-Leste is reported as projected straight-line 5 km screening because local application-control policy blocked the native `pyrosm` and `pandana` routing extensions during this run. Exact optima are certified for the straight-line 10 km, 5 km, and 1 km Timor instances; heuristic rows are retained only for quality comparison.",
            "",
            "Vietnam uses the existing 10 km network-distance instances, the GitHub `approximated_tradeoff` greedy-drop-greedy curve, and deterministic Fleur-style greedy plus first-sparse local search at the selected p values.",
            "",
            "## Timor Exact Instance Summary",
            "",
            markdown_table(
                stats_display,
                [
                    "grid",
                    "n_candidates",
                    "candidate_edge_rows",
                    "all_candidate_coverage_percent",
                    "exact_saturation_budget",
                    "exact_total_seconds",
                ],
                ["Grid", "Candidates", "Edges", "All-Candidate Coverage", "Saturation p", "Exact Total Seconds"],
            ),
            "",
            "## Timor Selected p Gains",
            "",
            markdown_table(
                timor_display,
                [
                    "budget",
                    "timor_10km_exact_coverage_percent",
                    "timor_5km_exact_coverage_percent",
                    "timor_1km_exact_coverage_percent",
                    "gain_10km_to_5km_percentage_points",
                    "gain_5km_to_1km_percentage_points",
                    "timor_1km_best_heuristic_coverage_percent",
                    "timor_1km_exact_minus_heuristic_percentage_points",
                    "timor_1km_exact_seconds",
                    "timor_1km_best_heuristic_seconds",
                ],
                [
                    "p",
                    "10 km Exact",
                    "5 km Exact",
                    "1 km Exact",
                    "10 to 5 Gain pp",
                    "5 to 1 Gain pp",
                    "1 km Fleur Best",
                    "Exact-Fleur pp",
                    "1 km Exact Seconds",
                    "1 km Fleur Seconds",
                ],
            ),
            "",
            "## Timor Approximate Quality",
            "",
            markdown_table(
                timor_quality_display,
                [
                    "grid",
                    "budget",
                    "exact_coverage_percent",
                    "approx_coverage_percent",
                    "exact_minus_approx_percentage_points",
                    "exact_seconds",
                    "approx_seconds",
                ],
                ["Grid", "p", "Exact", "Approx", "Exact-Approx pp", "Exact Seconds", "Approx Seconds"],
            ),
            "",
            "## Vietnam 10 km Selected p Metrics",
            "",
            markdown_table(
                vietnam_display,
                [
                    "threshold_km",
                    "budget",
                    "approx_coverage_percent",
                    "fleur_coverage_percent",
                    "fleur_minus_approx_pp",
                    "approx_seconds",
                    "fleur_seconds",
                    "effective_approx_budget",
                    "fleur_method",
                ],
                [
                    "Threshold",
                    "p",
                    "Approx",
                    "Fleur Best",
                    "Fleur-Approx pp",
                    "Approx Seconds",
                    "Fleur Seconds",
                    "Approx Eff p",
                    "Fleur Method",
                ],
            ),
            "",
            "## Vietnam Dense 1 km Heuristic vs 5 km Exact",
            "",
            "These rows use the projected straight-line dense-grid model. `time_limit` means Gurobi kept the best incumbent and bound after the configured cap; that row is not a certified optimum.",
            "",
            markdown_table(
                vietnam_dense_display,
                [
                    "threshold_km",
                    "budget",
                    "vietnam_5km_exact_status",
                    "vietnam_5km_exact_coverage_percent",
                    "vietnam_1km_best_heuristic_coverage_percent",
                    "vietnam_1km_heuristic_minus_5km_exact_pp",
                    "vietnam_5km_exact_seconds",
                    "vietnam_1km_best_heuristic_seconds",
                    "vietnam_1km_best_heuristic_method",
                ],
                [
                    "Threshold",
                    "p",
                    "5 km Status",
                    "5 km Exact",
                    "1 km Heuristic",
                    "1 km - 5 km pp",
                    "5 km Seconds",
                    "1 km Seconds",
                    "1 km Method",
                ],
            )
            if not vietnam_dense_display.empty
            else "_Vietnam dense exact comparison not found._",
            "",
        ]
    )
    report_path = args.output_dir / "combined_timor_vietnam_grid_report.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "outputs": {
            "report": str(report_path),
            "timor_selected": str(args.output_dir / "timor_selected_p_10_5_1_gain.csv"),
            "timor_approx_gain": str(args.output_dir / "timor_selected_p_approx_gain.csv"),
            "timor_approx_quality": str(args.output_dir / "timor_approx_vs_exact_selected_p.csv"),
            "vietnam_selected": str(args.output_dir / "vietnam_10km_selected_p_approx_vs_fleur.csv"),
            "vietnam_dense_exact": str(args.output_dir / "vietnam_1km_heuristic_vs_5km_exact.csv"),
        },
        "inputs": {
            "timor_dir": str(args.timor_dir),
            "vietnam_selected_dir": str(args.vietnam_selected_dir),
            "vietnam_approx": str(args.vietnam_approx),
        },
    }
    (args.output_dir / "combined_grid_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
