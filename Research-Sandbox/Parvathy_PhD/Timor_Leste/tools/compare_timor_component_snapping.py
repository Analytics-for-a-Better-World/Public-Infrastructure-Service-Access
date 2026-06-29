from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PRIMARY_CASES = [
    "timor_drive_only_unsimplified_10km",
    "timor_drive_only_unsimplified_5km",
    "timor_drive_only_unsimplified_1km",
    "timor_drive_plus_walk_unsimplified_10km",
    "timor_drive_plus_walk_unsimplified_5km",
    "timor_drive_plus_walk_unsimplified_1km",
]


def read_stats(path: Path, policy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["snapping_policy"] = policy
    keep = [
        "case_id",
        "network_profile",
        "simplify_network",
        "candidate_grid_spacing_m",
        "baseline_percent",
        "all_candidates_coverage_percent",
        "n_candidate_sources_in_matrix",
        "n_candidate_arcs_after_existing_coverage_removed",
        "n_population_components",
        "n_existing_source_components",
        "pipeline_elapsed_clock_ms",
    ]
    return df[keep + ["snapping_policy"]]


def read_results(path: Path, policy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["snapping_policy"] = policy
    return df


def save_primary_plot(comparison: pd.DataFrame, out: Path) -> None:
    primary = comparison[comparison["case_id"].isin(PRIMARY_CASES)].copy()
    policy_order = ["unrestricted", "component_0_1_2"]
    colors = {"unrestricted": "#7A7A7A", "component_0_1_2": "#0072B2"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, profile in zip(axes, ["driving", "driving_walk"]):
        subset = primary[primary["network_profile"] == profile].copy()
        subset["grid_order"] = subset["candidate_grid_spacing_m"].map({10000: 0, 5000: 1, 1000: 2})
        subset = subset.sort_values("grid_order")
        x_base = [0, 1, 2]
        width = 0.34
        for offset, policy in [(-width / 2, policy_order[0]), (width / 2, policy_order[1])]:
            rows = subset.set_index("candidate_grid_spacing_m")
            values = [
                float(rows.loc[spacing, f"all_candidates_coverage_percent_{policy}"])
                for spacing in [10000, 5000, 1000]
            ]
            ax.bar([x + offset for x in x_base], values, width=width, color=colors[policy], edgecolor="white")
        ax.set_title("Drive only" if profile == "driving" else "Drive + walk")
        ax.set_xticks(x_base, ["10 km", "5 km", "1 km"])
        ax.set_ylim(78, 101)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("All-candidate coverage (%)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-stats", type=Path, required=True)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--new-stats", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--component-geography", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_stats = read_stats(args.old_stats, "unrestricted")
    new_stats = read_stats(args.new_stats, "component_0_1_2")
    stats = pd.concat([old_stats, new_stats], ignore_index=True)
    stats.to_csv(args.output_dir / "timor_component_snapping_instance_stats.csv", index=False)

    pivot = stats.pivot_table(
        index=["case_id", "network_profile", "simplify_network", "candidate_grid_spacing_m"],
        columns="snapping_policy",
        values=[
            "baseline_percent",
            "all_candidates_coverage_percent",
            "n_candidate_sources_in_matrix",
            "n_candidate_arcs_after_existing_coverage_removed",
            "n_population_components",
            "n_existing_source_components",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{policy}" for metric, policy in pivot.columns]
    pivot = pivot.reset_index()
    pivot["delta_baseline_percent_points"] = (
        pivot["baseline_percent_component_0_1_2"] - pivot["baseline_percent_unrestricted"]
    )
    pivot["delta_all_candidate_percent_points"] = (
        pivot["all_candidates_coverage_percent_component_0_1_2"]
        - pivot["all_candidates_coverage_percent_unrestricted"]
    )
    pivot["delta_candidates"] = (
        pivot["n_candidate_sources_in_matrix_component_0_1_2"]
        - pivot["n_candidate_sources_in_matrix_unrestricted"]
    )
    pivot["delta_arcs"] = (
        pivot["n_candidate_arcs_after_existing_coverage_removed_component_0_1_2"]
        - pivot["n_candidate_arcs_after_existing_coverage_removed_unrestricted"]
    )
    pivot.to_csv(args.output_dir / "timor_component_snapping_instance_comparison.csv", index=False)

    old_results = read_results(args.old_results, "unrestricted")
    new_results = read_results(args.new_results, "component_0_1_2")
    results = pd.concat([old_results, new_results], ignore_index=True)
    results.to_csv(args.output_dir / "timor_component_snapping_results_long.csv", index=False)

    exact = results[results["solver_family"].eq("gurobi_exact")].copy()
    exact_pivot = exact.pivot_table(
        index=["case_id", "network_profile", "simplify_network", "candidate_grid_spacing_m", "budget"],
        columns="snapping_policy",
        values=["coverage_percent_total_population", "total_covered_population", "total_seconds"],
        aggfunc="first",
    )
    exact_pivot.columns = [f"{metric}_{policy}" for metric, policy in exact_pivot.columns]
    exact_pivot = exact_pivot.reset_index()
    exact_pivot["delta_exact_coverage_percent_points"] = (
        exact_pivot["coverage_percent_total_population_component_0_1_2"]
        - exact_pivot["coverage_percent_total_population_unrestricted"]
    )
    exact_pivot["delta_exact_covered_population"] = (
        exact_pivot["total_covered_population_component_0_1_2"]
        - exact_pivot["total_covered_population_unrestricted"]
    )
    exact_pivot.to_csv(args.output_dir / "timor_component_snapping_exact_comparison.csv", index=False)

    heur = results[results["solver_family"].eq("approximate_pareto")].copy()
    heur.to_csv(args.output_dir / "timor_component_snapping_heuristics_long.csv", index=False)

    geography = pd.read_csv(args.component_geography)
    focus = geography[
        (geography["component_id"].isin([0, 1, 2]))
        & geography["network"].isin(
            ["driving_unsimplified", "driving_simplified", "driving_walk_unsimplified", "driving_walk_simplified"]
        )
    ].copy()
    focus.to_csv(args.output_dir / "timor_component_012_geography.csv", index=False)

    save_primary_plot(pivot, args.output_dir / "timor_component_snapping_all_candidate_coverage.pdf")
    save_primary_plot(pivot, args.output_dir / "timor_component_snapping_all_candidate_coverage.png")

    primary_exact = exact_pivot[
        exact_pivot["case_id"].isin(PRIMARY_CASES) & exact_pivot["budget"].isin([20, 100, 175, 200])
    ].copy()
    primary_exact.to_csv(args.output_dir / "timor_component_snapping_primary_exact.csv", index=False)

    print(
        {
            "stats_rows": len(stats),
            "comparison_rows": len(pivot),
            "exact_rows": len(exact_pivot),
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()
