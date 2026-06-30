from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis")
ARTICLE = ROOT / "articles" / "seps_access_optimization"
TIMOR_RESULTS = ROOT / "outputs" / "timor_maxcover_benchmark_20260623" / "timor_exact_vs_heuristic_results.csv"
TIMOR_STATS = ROOT / "outputs" / "timor_maxcover_benchmark_20260623" / "timor_instance_statistics.csv"
VIETNAM_RESULTS = (
    ROOT
    / "reference_cache"
    / "notes"
    / "vietnam"
    / "results"
    / "fleur_style_fast_replication"
    / "coverage_summary_by_budget.csv"
)
TIMOR_FIG_CACHE = ROOT / "reference_cache" / "figures" / "timor_leste"
VIETNAM_FIG_CACHE = ROOT / "reference_cache" / "figures" / "vietnam"


COLORS = {
    10000: "#4C78A8",
    5000: "#59A14F",
    1000: "#E15759",
    20.0: "#4C78A8",
    50.0: "#F28E2B",
    100.0: "#59A14F",
}


def finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def copy_map_figures() -> None:
    copies = {
        TIMOR_FIG_CACHE / "timor_leste_polygons_osm_subtle.pdf": ARTICLE / "fig_timor_leste_osm_context.pdf",
        TIMOR_FIG_CACHE / "timor_leste_selected_175_centers_map.pdf": ARTICLE / "fig_timor_leste_selected_175_centers_map.pdf",
        TIMOR_FIG_CACHE
        / "timor_leste_coverage_frontiers_by_threshold.pdf": ARTICLE / "fig_timor_leste_coverage_frontiers_by_threshold.pdf",
        VIETNAM_FIG_CACHE / "vietnam_polygons_osm_basemap.pdf": ARTICLE / "fig_vietnam_osm_context.pdf",
    }
    for source, destination in copies.items():
        if source.exists():
            shutil.copy2(source, destination)


def timor_grid_quality() -> None:
    data = pd.read_csv(TIMOR_RESULTS)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for grid, label in [(10000, "10 km grid"), (5000, "5 km grid"), (1000, "1 km grid")]:
        exact = data[(data.grid_spacing_m == grid) & (data.method == "gurobi_exact")].sort_values("budget")
        heur = data[(data.grid_spacing_m == grid) & (data.method != "gurobi_exact")].sort_values("budget")
        color = COLORS[grid]
        ax.plot(
            exact["budget"],
            exact["coverage_percent_total_population"],
            color=color,
            linewidth=2.4,
            marker="o",
            label=f"{label}: exact",
        )
        ax.plot(
            heur["budget"],
            heur["coverage_percent_total_population"],
            color=color,
            linewidth=0,
            marker="s",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=f"{label}: heuristic",
        )
    ax.set_xlabel("Added centers")
    ax.set_ylabel("Population within 5 km (%)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    finish_figure(fig, ARTICLE / "fig_timor_exact_heuristic_grid_quality.pdf")


def timor_gap_runtime() -> None:
    data = pd.read_csv(TIMOR_RESULTS)
    heur = data[data.method != "gurobi_exact"].copy()
    exact = data[data.method == "gurobi_exact"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))

    one_km = heur[heur.grid_spacing_m == 1000].sort_values("budget")
    axes[0].bar(
        one_km["budget"].astype(str),
        one_km["gap_to_exact_percent_points"].fillna(0.0),
        color="#E15759",
        width=0.65,
    )
    axes[0].set_xlabel("Added centers")
    axes[0].set_ylabel("Heuristic gap (percentage points)")
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    exact_times = exact.groupby("grid_spacing_m", as_index=False)["total_seconds"].sum()
    heur_times = heur.groupby("grid_spacing_m", as_index=False)["total_seconds"].sum()
    x = np.arange(len(exact_times))
    labels = [f"{int(v / 1000)} km" for v in exact_times["grid_spacing_m"]]
    axes[1].bar(x - 0.18, exact_times["total_seconds"], width=0.36, label="Exact", color="#4C78A8")
    axes[1].bar(x + 0.18, heur_times["total_seconds"], width=0.36, label="Heuristic", color="#59A14F")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Candidate grid")
    axes[1].set_ylabel("Total benchmark time (s)")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()
    finish_figure(fig, ARTICLE / "fig_timor_heuristic_quality_runtime.pdf")


def timor_instance_size() -> None:
    stats = pd.read_csv(TIMOR_STATS).sort_values("candidate_grid_spacing_m", ascending=False)
    fig, ax1 = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(stats))
    labels = [f"{int(v / 1000)} km" for v in stats["candidate_grid_spacing_m"]]
    ax1.bar(x - 0.18, stats["n_candidates"], width=0.36, color="#4C78A8", label="Candidates")
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.18,
        stats["n_candidate_arcs_after_existing_coverage_removed"],
        width=0.36,
        color="#F28E2B",
        label="Candidate-demand arcs",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Candidate grid")
    ax1.set_ylabel("Candidates")
    ax2.set_ylabel("Candidate-demand arcs")
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.16, 0.5), frameon=False)
    finish_figure(fig, ARTICLE / "fig_timor_instance_size_by_grid.pdf")


def vietnam_fleur_style() -> None:
    data = pd.read_csv(VIETNAM_RESULTS)
    best = (
        data.sort_values(["threshold_km", "budget", "coverage_percent_total_population"], ascending=[True, True, False])
        .groupby(["threshold_km", "budget"], as_index=False)
        .head(1)
        .sort_values(["threshold_km", "budget"])
    )
    greedy = data[data["method"].isin(["greedy_construction", "greedy_first_sparse"])].copy()
    greedy_best = (
        greedy.sort_values(["threshold_km", "budget", "coverage_percent_total_population"], ascending=[True, True, False])
        .groupby(["threshold_km", "budget"], as_index=False)
        .head(1)
    )
    merged = best.merge(
        greedy_best[["threshold_km", "budget", "coverage_percent_total_population"]],
        on=["threshold_km", "budget"],
        suffixes=("_best", "_greedy"),
    )
    merged["gain_pp_over_greedy"] = (
        merged["coverage_percent_total_population_best"] - merged["coverage_percent_total_population_greedy"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for threshold, group in best.groupby("threshold_km"):
        color = COLORS[float(threshold)]
        axes[0].plot(
            group["budget"],
            group["coverage_percent_total_population"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{threshold:g} km threshold",
        )
    axes[0].set_xlabel("Added stroke centers")
    axes[0].set_ylabel("Population within threshold (%)")
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    for threshold, group in merged.groupby("threshold_km"):
        color = COLORS[float(threshold)]
        axes[1].plot(
            group["budget"],
            group["gain_pp_over_greedy"],
            marker="s",
            linewidth=2.0,
            color=color,
            label=f"{threshold:g} km threshold",
        )
    axes[1].set_xlabel("Added stroke centers")
    axes[1].set_ylabel("Best heuristic gain over greedy (pp)")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    finish_figure(fig, ARTICLE / "fig_vietnam_fleur_style_10km_curves.pdf")


def export_summary_tables() -> None:
    data = pd.read_csv(TIMOR_RESULTS)
    summary = data[
        [
            "grid_spacing_m",
            "budget",
            "method",
            "status",
            "coverage_percent_total_population",
            "gap_to_exact_percent_points",
            "total_seconds",
            "mip_gap",
        ]
    ].copy()
    summary.to_csv(ARTICLE / "table_timor_exact_heuristic_quality.csv", index=False)

    vietnam = pd.read_csv(VIETNAM_RESULTS)
    best = (
        vietnam.sort_values(
            ["threshold_km", "budget", "coverage_percent_total_population"],
            ascending=[True, True, False],
        )
        .groupby(["threshold_km", "budget"], as_index=False)
        .head(1)
    )
    best.to_csv(ARTICLE / "table_vietnam_best_fleur_style_10km.csv", index=False)


def main() -> None:
    ARTICLE.mkdir(parents=True, exist_ok=True)
    copy_map_figures()
    timor_grid_quality()
    timor_gap_runtime()
    timor_instance_size()
    vietnam_fleur_style()
    export_summary_tables()


if __name__ == "__main__":
    main()
