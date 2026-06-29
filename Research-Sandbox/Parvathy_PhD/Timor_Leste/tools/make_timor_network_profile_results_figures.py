from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


SELECTED_DIR = Path("outputs/timor_network_profile_component012_optimization_20260626")
DENSE_DIR = Path("outputs/timor_network_profile_component012_optimization_20260626")
SATURATION_DIR = Path("outputs/timor_component012_saturation_20260626")
ARTICLE_DIR = Path("articles/seps_access_optimization")
FIGURE_DIR = ARTICLE_DIR / "figures"

PROFILE_LABELS = {
    "driving": "Drive-only",
    "driving_walk": "Drive + walk",
}
PROFILE_SHORT = {
    "driving": "Drive",
    "driving_walk": "Drive + walk",
}
PROFILE_COLORS = {
    "driving": "#2166ac",
    "driving_walk": "#009e73",
}
SIMPLIFY_LABELS = {
    "True": "Simplified",
    "False": "Unsimplified",
    True: "Simplified",
    False: "Unsimplified",
}
SIMPLIFY_LINESTYLES = {
    "True": (0, (4, 2)),
    "False": "solid",
    True: (0, (4, 2)),
    False: "solid",
}
GRID_LABELS = {
    10000: "10 km",
    5000: "5 km",
    1000: "1 km",
}
GRID_ORDER = [10000, 5000, 1000]
KEY_BUDGETS = [0, 20, 60, 100, 175, 200]
CASE_COLORS = {
    (10000, "driving"): "#1f77b4",
    (10000, "driving_walk"): "#17becf",
    (5000, "driving"): "#ff7f0e",
    (5000, "driving_walk"): "#b58900",
    (1000, "driving"): "#2ca02c",
    (1000, "driving_walk"): "#9467bd",
}
CASE_LABEL_OFFSETS = {
    (10000, "driving"): -0.28,
    (10000, "driving_walk"): 0.28,
    (5000, "driving"): -0.32,
    (5000, "driving_walk"): 0.32,
    (1000, "driving"): -0.26,
    (1000, "driving_walk"): 0.26,
}


def clock_label(seconds: float, _pos: object = None) -> str:
    if not math.isfinite(float(seconds)):
        return ""
    millis = int(round(float(seconds) * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    if minutes:
        return f"{minutes:d}:{secs:02d}.{ms:03d}"
    return f"{secs:d}.{ms:03d}s"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(SELECTED_DIR / "timor_network_profile_results.csv")
    dense = pd.read_csv(DENSE_DIR / "timor_network_profile_results.csv")
    stats = pd.read_csv(SELECTED_DIR / "timor_network_profile_instance_statistics.csv")
    snap = pd.read_csv(SELECTED_DIR / "timor_network_profile_snap_statistics.csv")
    saturation_summary_path = SATURATION_DIR / "timor_primary_saturation_summary.csv"
    saturation_curves_path = SATURATION_DIR / "timor_primary_curves_to_saturation.csv"
    saturation_summary = pd.read_csv(saturation_summary_path) if saturation_summary_path.exists() else pd.DataFrame()
    saturation_curves = pd.read_csv(saturation_curves_path) if saturation_curves_path.exists() else pd.DataFrame()
    for frame in [selected, dense, stats, saturation_summary, saturation_curves]:
        if "candidate_grid_spacing_m" in frame.columns:
            frame["candidate_grid_spacing_m"] = frame["candidate_grid_spacing_m"].astype(int)
    return selected, dense, stats, snap, saturation_summary, saturation_curves


def savefig(fig: plt.Figure, stem: str) -> None:
    pdf = ARTICLE_DIR / f"{stem}.pdf"
    png = ARTICLE_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def add_case_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["grid_label"] = result["candidate_grid_spacing_m"].map(GRID_LABELS)
    result["profile_label"] = result["network_profile"].map(PROFILE_LABELS)
    result["profile_short"] = result["network_profile"].map(PROFILE_SHORT)
    result["simplify_label"] = result["simplify_network"].map(SIMPLIFY_LABELS)
    return result


def make_primary_dense_pareto(dense: pd.DataFrame, saturation_curves: pd.DataFrame) -> None:
    if not saturation_curves.empty:
        data = add_case_labels(saturation_curves.copy())
        data["solver_family"] = data["curve_kind"]
    else:
        data = add_case_labels(dense.loc[dense["solver_family"].eq("gurobi_exact")].copy())
        data["curve_kind"] = "exact"
    largest_saturation = int(data["budget"].max())
    x_end = int(math.ceil((largest_saturation * 1.08) / 10.0) * 10)

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    for grid in GRID_ORDER:
        for profile in ["driving", "driving_walk"]:
            color = CASE_COLORS[(grid, profile)]
            label_offset = CASE_LABEL_OFFSETS[(grid, profile)]
            label_text = f"{GRID_LABELS[grid]} {PROFILE_SHORT[profile]}"
            for curve_kind, linestyle, linewidth, alpha in [
                ("exact", "solid", 2.2, 0.98),
                ("approx_regreedy", (0, (4, 2)), 1.65, 0.88),
            ]:
                sub = data.loc[
                    data["candidate_grid_spacing_m"].eq(grid)
                    & data["network_profile"].eq(profile)
                    & data["curve_kind"].eq(curve_kind)
                ].sort_values("budget")
                if sub.empty:
                    continue
                x = sub["budget"].to_numpy(dtype=float)
                y = sub["coverage_percent_total_population"].to_numpy(dtype=float)
                if x.size and x[-1] < x_end:
                    x = np.r_[x, float(x_end)]
                    y = np.r_[y, y[-1]]
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                if curve_kind == "exact":
                    ax.scatter([x[-1]], [y[-1]], color=color, s=18, zorder=4)
                    ax.text(
                        x_end - 4,
                        y[-1] + label_offset,
                        label_text,
                        color=color,
                        ha="right",
                        va="center",
                        fontsize=8.2,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
                    )
    ax.set_xlabel("Budget p")
    ax.set_ylabel("Population covered (%)")
    ax.set_xlim(0, x_end)
    ax.set_ylim(74, 100.7)
    ax.grid(True, axis="both")
    savefig(fig, "fig_timor_primary_exact_pareto_dense")


def make_sensitivity_p175(selected: pd.DataFrame) -> None:
    data = add_case_labels(
        selected.loc[
            selected["solver_family"].eq("gurobi_exact")
            & selected["budget"].eq(175)
        ].copy()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True)
    offsets = {
        ("driving", True): -0.27,
        ("driving", False): -0.09,
        ("driving_walk", True): 0.09,
        ("driving_walk", False): 0.27,
    }
    labels = {
        ("driving", True): "Drive-only / simplified",
        ("driving", False): "Drive-only / unsimplified",
        ("driving_walk", True): "Drive + walk / simplified",
        ("driving_walk", False): "Drive + walk / unsimplified",
    }
    for ax, grid in zip(axes, GRID_ORDER):
        for (profile, simplified), offset in offsets.items():
            row = data.loc[
                data["candidate_grid_spacing_m"].eq(grid)
                & data["network_profile"].eq(profile)
                & data["simplify_network"].astype(str).eq(str(simplified))
            ]
            if row.empty:
                continue
            hatch = "///" if simplified else None
            ax.bar(
                [0 + offset],
                row["coverage_percent_total_population"].iloc[0],
                width=0.16,
                color=PROFILE_COLORS[profile],
                alpha=0.88 if not simplified else 0.55,
                hatch=hatch,
                edgecolor="#333333",
                linewidth=0.4,
                label=labels[(profile, simplified)] if grid == GRID_ORDER[0] else None,
            )
        ax.set_title(GRID_LABELS[grid])
        ax.set_xticks([])
        ax.set_xlabel("p = 175")
        ax.set_ylim(76, 99)
    axes[0].set_ylabel("Population covered (%)")
    fig.legend(loc="center left", bbox_to_anchor=(0.91, 0.5))
    fig.subplots_adjust(right=0.88, wspace=0.18)
    savefig(fig, "fig_timor_network_sensitivity_p175")


def make_instance_size(stats: pd.DataFrame) -> None:
    data = add_case_labels(stats)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    metrics = [
        ("n_candidate_sources_in_matrix", "Candidates in matrix"),
        ("n_candidate_arcs_after_existing_coverage_removed", "Candidate-demand pairs after existing coverage"),
    ]
    x_base = np.arange(len(GRID_ORDER))
    width = 0.18
    variants = [
        ("driving", True, -1.5 * width, "Drive-only / simplified"),
        ("driving", False, -0.5 * width, "Drive-only / unsimplified"),
        ("driving_walk", True, 0.5 * width, "Drive + walk / simplified"),
        ("driving_walk", False, 1.5 * width, "Drive + walk / unsimplified"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for profile, simplified, offset, label in variants:
            values = []
            for grid in GRID_ORDER:
                row = data.loc[
                    data["candidate_grid_spacing_m"].eq(grid)
                    & data["network_profile"].eq(profile)
                    & data["simplify_network"].astype(str).eq(str(simplified))
                ]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x_base + offset,
                values,
                width=width,
                color=PROFILE_COLORS[profile],
                alpha=0.88 if not simplified else 0.55,
                hatch="///" if simplified else None,
                edgecolor="#333333",
                linewidth=0.35,
                label=label if metric == metrics[0][0] else None,
            )
        ax.set_yscale("log")
        ax.set_xticks(x_base, [GRID_LABELS[grid] for grid in GRID_ORDER])
        ax.set_ylabel(ylabel)
    fig.legend(loc="center left", bbox_to_anchor=(0.91, 0.5))
    fig.subplots_adjust(right=0.88, wspace=0.26)
    savefig(fig, "fig_timor_instance_size_by_network_grid")


def make_snap_distance(snap: pd.DataFrame) -> None:
    # Snap stats are identical across grids for population/existing sources within a network representation.
    data = snap.loc[snap["case_id"].str.endswith("_1km")].copy()
    records = []
    for _, row in data.iterrows():
        case = str(row["case_id"])
        records.append(
            {
                "case_id": case,
                "profile": "driving_walk" if "drive_plus_walk" in case else "driving",
                "simplified": "simplified" in case and "unsimplified" not in case,
                "snap_group": row["snap_group"],
                "mean_m": float(row["mean_m"]),
                "p95_m": float(row["p95_m"]),
            }
        )
    plot = pd.DataFrame(records)
    groups = ["population_targets", "existing_sources", "candidate_sources"]
    group_labels = ["Population", "Existing sources", "Candidate sites"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=False)
    x = np.arange(4)
    variant_order = [
        ("driving", True, "Drive\nS"),
        ("driving", False, "Drive\nU"),
        ("driving_walk", True, "D+W\nS"),
        ("driving_walk", False, "D+W\nU"),
    ]
    for ax, group, group_label in zip(axes, groups, group_labels):
        subset = plot.loc[plot["snap_group"].eq(group)]
        means = []
        p95s = []
        colors = []
        hatches = []
        for profile, simplified, _label in variant_order:
            row = subset.loc[subset["profile"].eq(profile) & subset["simplified"].eq(simplified)]
            means.append(float(row["mean_m"].iloc[0]))
            p95s.append(float(row["p95_m"].iloc[0]))
            colors.append(PROFILE_COLORS[profile])
            hatches.append("///" if simplified else None)
        bars = ax.bar(x, p95s, color=colors, alpha=0.72, edgecolor="#333333", linewidth=0.35)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        ax.scatter(x, means, s=20, color="#111111", zorder=3)
        ax.set_title(group_label)
        ax.set_xticks(x, [label for _, _, label in variant_order], rotation=0)
        ax.set_ylabel("Snap distance (m)")
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", linestyle="none", color="#111111", markersize=4, label="Mean"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#555555", hatch="///", label="Simplified"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#555555", label="Unsimplified"),
        ],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
    )
    fig.subplots_adjust(right=0.88, wspace=0.22)
    savefig(fig, "fig_timor_snap_distance_sensitivity")


def make_heuristic_quality(selected: pd.DataFrame) -> None:
    data = add_case_labels(
        selected.loc[
            selected["solver_family"].eq("approximate_pareto")
            & selected["gap_to_exact_percent_points"].notna()
        ].copy()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True)
    for ax, grid in zip(axes, GRID_ORDER):
        sub_grid = data.loc[data["candidate_grid_spacing_m"].eq(grid)]
        for profile in ["driving", "driving_walk"]:
            for simplified in [True, False]:
                sub = sub_grid.loc[
                    sub_grid["network_profile"].eq(profile)
                    & sub_grid["simplify_network"].astype(str).eq(str(simplified))
                ].sort_values("budget")
                if sub.empty:
                    continue
                ax.plot(
                    sub["budget"],
                    sub["gap_to_exact_percent_points"],
                    color=PROFILE_COLORS[profile],
                    linestyle=SIMPLIFY_LINESTYLES[simplified],
                    marker="o" if not simplified else "s",
                    markersize=3.5,
                    linewidth=1.5,
                    label=f"{PROFILE_SHORT[profile]} / {SIMPLIFY_LABELS[simplified]}" if grid == GRID_ORDER[0] else None,
                )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(GRID_LABELS[grid])
        ax.set_xlabel("Budget p")
    axes[0].set_ylabel("Heuristic gap to exact (percentage points)")
    axes[0].set_ylim(-0.002, max(0.04, float(data["gap_to_exact_percent_points"].max()) * 1.25))
    fig.legend(loc="center left", bbox_to_anchor=(0.91, 0.5))
    fig.subplots_adjust(right=0.88, wspace=0.18)
    savefig(fig, "fig_timor_heuristic_quality_selected")


def make_runtime(selected: pd.DataFrame) -> None:
    data = add_case_labels(selected.copy())
    key = data.loc[data["budget"].eq(175)].copy()
    key["case_label"] = (
        key["grid_label"]
        + " | "
        + key["profile_short"]
        + " | "
        + key["simplify_label"].str.replace("Unsimplified", "Unsimp.").str.replace("Simplified", "Simp.")
        + " | "
        + key["solver_family"].str.replace("_", " ")
    )
    key = key.sort_values("total_seconds", ascending=True).tail(24)
    fig, ax = plt.subplots(figsize=(10.6, 7.8))
    colors = key["solver_family"].map(
        {
            "gurobi_exact": "#4c78a8",
            "pyomo_highs_exact": "#f58518",
            "approximate_pareto": "#54a24b",
        }
    ).fillna("#777777")
    ax.barh(np.arange(len(key)), key["total_seconds"], color=colors, alpha=0.88)
    ax.set_yticks(np.arange(len(key)), key["case_label"])
    ax.set_xlabel("Total time")
    ax.xaxis.set_major_formatter(FuncFormatter(clock_label))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    savefig(fig, "fig_timor_runtime_selected_p175")


def write_tables(
    selected: pd.DataFrame,
    dense: pd.DataFrame,
    stats: pd.DataFrame,
    snap: pd.DataFrame,
    saturation_summary: pd.DataFrame,
    saturation_curves: pd.DataFrame,
) -> None:
    key_dense = dense.loc[dense["budget"].isin(KEY_BUDGETS)].copy()
    key_dense.to_csv(ARTICLE_DIR / "table_timor_primary_dense_exact_key_budgets.csv", index=False)

    selected.to_csv(ARTICLE_DIR / "table_timor_network_profile_selected_results.csv", index=False)
    stats.to_csv(ARTICLE_DIR / "table_timor_network_profile_instance_statistics.csv", index=False)
    snap.to_csv(ARTICLE_DIR / "table_timor_network_profile_snap_statistics.csv", index=False)
    if not saturation_summary.empty:
        saturation_summary.to_csv(ARTICLE_DIR / "table_timor_primary_saturation_summary.csv", index=False)
    if not saturation_curves.empty:
        saturation_curves.to_csv(ARTICLE_DIR / "table_timor_primary_curves_to_saturation.csv", index=False)

    heur = selected.loc[
        selected["solver_family"].eq("approximate_pareto")
        & selected["gap_to_exact_percent_points"].notna()
    ].copy()
    summary = (
        heur.groupby(["case_id", "candidate_grid_spacing_m"], as_index=False)
        .agg(
            max_gap_pp=("gap_to_exact_percent_points", "max"),
            mean_gap_pp=("gap_to_exact_percent_points", "mean"),
            max_runtime_s=("total_seconds", "max"),
        )
        .sort_values(["candidate_grid_spacing_m", "max_gap_pp"], ascending=[False, False])
    )
    summary["max_runtime_clock_ms"] = summary["max_runtime_s"].map(clock_label)
    summary.to_csv(ARTICLE_DIR / "table_timor_heuristic_quality_summary.csv", index=False)

    primary = dense.loc[
        dense["solver_family"].eq("gurobi_exact")
        & dense["budget"].isin([175, 200])
    ].copy()
    gains = []
    for profile in ["driving", "driving_walk"]:
        for budget in [175, 200]:
            rows = primary.loc[primary["network_profile"].eq(profile) & primary["budget"].eq(budget)]
            by_grid = rows.set_index("candidate_grid_spacing_m")["coverage_percent_total_population"].to_dict()
            gains.append(
                {
                    "network_profile": profile,
                    "budget": budget,
                    "coverage_10km_percent": by_grid.get(10000),
                    "coverage_5km_percent": by_grid.get(5000),
                    "coverage_1km_percent": by_grid.get(1000),
                    "gain_10_to_5_pp": None
                    if 10000 not in by_grid or 5000 not in by_grid
                    else by_grid[5000] - by_grid[10000],
                    "gain_5_to_1_pp": None
                    if 5000 not in by_grid or 1000 not in by_grid
                    else by_grid[1000] - by_grid[5000],
                }
            )
    pd.DataFrame(gains).to_csv(ARTICLE_DIR / "table_timor_coverage_gain_richer_grids.csv", index=False)


def main() -> None:
    apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected, dense, stats, snap, saturation_summary, saturation_curves = load_results()
    write_tables(selected, dense, stats, snap, saturation_summary, saturation_curves)
    make_primary_dense_pareto(dense, saturation_curves)
    make_sensitivity_p175(selected)
    make_instance_size(stats)
    make_snap_distance(snap)
    make_heuristic_quality(selected)
    make_runtime(selected)
    print(
        {
            "figures": [
                "fig_timor_primary_exact_pareto_dense",
                "fig_timor_network_sensitivity_p175",
                "fig_timor_instance_size_by_network_grid",
                "fig_timor_snap_distance_sensitivity",
                "fig_timor_heuristic_quality_selected",
                "fig_timor_runtime_selected_p175",
            ],
            "tables": [
                "table_timor_primary_dense_exact_key_budgets.csv",
                "table_timor_network_profile_selected_results.csv",
                "table_timor_network_profile_instance_statistics.csv",
                "table_timor_network_profile_snap_statistics.csv",
                "table_timor_heuristic_quality_summary.csv",
                "table_timor_coverage_gain_richer_grids.csv",
            ],
        }
    )


if __name__ == "__main__":
    main()
