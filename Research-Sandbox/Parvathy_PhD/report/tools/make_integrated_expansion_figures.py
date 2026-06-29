from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_FIG = ROOT / "articles" / "integrated_access_report" / "figures"
DECK_FIG = ROOT / "presentations" / "integrated_access_deck" / "figures"
OUT = ROOT / "outputs" / "integrated_report_expansion"

BLUE = "#2F6FBB"
GREEN = "#1B9E77"
ORANGE = "#E69F00"
RED = "#D55E00"
PURPLE = "#7B61FF"
GRAY = "#6E6E6E"
LIGHT = "#F5F7FA"


def setup() -> None:
    ARTICLE_FIG.mkdir(parents=True, exist_ok=True)
    DECK_FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D7DCE2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
        }
    )


def clock(seconds: float) -> str:
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    total = int(math.floor(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def savefig(name: str) -> None:
    pdf = ARTICLE_FIG / f"{name}.pdf"
    png = ARTICLE_FIG / f"{name}.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight")
    plt.close()
    shutil.copy2(pdf, DECK_FIG / pdf.name)
    shutil.copy2(png, DECK_FIG / png.name)


def representation(row: pd.Series) -> str:
    profile = "Drive+walk" if row["network_profile"] == "driving_walk" else "Drive"
    simp = "S" if bool(row["simplify_network"]) else "U"
    return f"{profile} {simp}"


def network_label(name: str) -> str:
    return {
        "driving_unsimplified": "Drive U",
        "driving_simplified": "Drive S",
        "driving_walk_unsimplified": "Drive+walk U",
        "driving_walk_simplified": "Drive+walk S",
    }.get(name, name)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def component_policy_figure(summary: dict) -> None:
    stats = pd.read_csv(
        ROOT
        / "outputs"
        / "timor_component_snapping_comparison_20260626"
        / "timor_component_snapping_instance_stats.csv"
    )
    stats["simplify_network"] = stats["simplify_network"].astype(str).str.lower().eq("true")
    stats["repr"] = stats.apply(representation, axis=1)
    rows = stats[stats["candidate_grid_spacing_m"] == 1000].copy()
    order = ["Drive U", "Drive S", "Drive+walk U", "Drive+walk S"]
    policies = ["unrestricted", "component_0_1_2"]
    colors = {"unrestricted": ORANGE, "component_0_1_2": BLUE}
    labels = {"unrestricted": "Unrestricted", "component_0_1_2": "Component-aware"}

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36

    for idx, policy in enumerate(policies):
        data = rows[rows["snapping_policy"] == policy].set_index("repr").reindex(order)
        axes[0].bar(
            x + (idx - 0.5) * width,
            data["n_population_components"],
            width,
            color=colors[policy],
            label=labels[policy],
        )
    axes[0].set_title("Population components")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=25, ha="right")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, loc="upper left")

    for idx, policy in enumerate(policies):
        data = rows[rows["snapping_policy"] == policy].set_index("repr").reindex(order)
        axes[1].bar(
            x + (idx - 0.5) * width,
            data["n_candidate_arcs_after_existing_coverage_removed"] / 1e6,
            width,
            color=colors[policy],
        )
    axes[1].set_title("Candidate-demand arcs")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=25, ha="right")
    axes[1].set_ylabel("Million arcs")

    for idx, policy in enumerate(policies):
        data = rows[rows["snapping_policy"] == policy].set_index("repr").reindex(order)
        axes[2].bar(
            x + (idx - 0.5) * width,
            data["all_candidates_coverage_percent"],
            width,
            color=colors[policy],
        )
    aware = rows[rows["snapping_policy"] == "component_0_1_2"].set_index("repr")
    plain = rows[rows["snapping_policy"] == "unrestricted"].set_index("repr")
    for i, label in enumerate(order):
        delta = aware.loc[label, "all_candidates_coverage_percent"] - plain.loc[
            label, "all_candidates_coverage_percent"
        ]
        top = max(
            aware.loc[label, "all_candidates_coverage_percent"],
            plain.loc[label, "all_candidates_coverage_percent"],
        )
        axes[2].text(i, top + 0.20, f"+{delta:.2f} pp", ha="center", va="bottom", fontsize=8)
    axes[2].set_title("All-candidate ceiling")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(order, rotation=25, ha="right")
    axes[2].set_ylabel("Covered population (%)")
    axes[2].set_ylim(72, 101.5)

    fig.suptitle("Timor-Leste: component-aware snapping changes the access matrix", y=1.06)
    savefig("fig_timor_component_policy_matrix_effect")

    comp_summary = {}
    for label in order:
        a = aware.loc[label]
        p = plain.loc[label]
        comp_summary[label] = {
            "population_components_unrestricted": int(p["n_population_components"]),
            "population_components_component_aware": int(a["n_population_components"]),
            "existing_components_unrestricted": int(p["n_existing_source_components"]),
            "existing_components_component_aware": int(a["n_existing_source_components"]),
            "arcs_unrestricted": int(p["n_candidate_arcs_after_existing_coverage_removed"]),
            "arcs_component_aware": int(a["n_candidate_arcs_after_existing_coverage_removed"]),
            "coverage_unrestricted_percent": float(p["all_candidates_coverage_percent"]),
            "coverage_component_aware_percent": float(a["all_candidates_coverage_percent"]),
            "coverage_gain_pp": float(
                a["all_candidates_coverage_percent"] - p["all_candidates_coverage_percent"]
            ),
        }
    summary["timor_component_policy_1km"] = comp_summary


def component_tail_figure(summary: dict) -> None:
    geo = pd.read_csv(
        ROOT
        / "outputs"
        / "timor_component_geography_20260626"
        / "timor_component_geography.csv"
    )
    geo["network_label"] = geo["network"].map(network_label)
    order = ["Drive U", "Drive S", "Drive+walk U", "Drive+walk S"]
    bins = [0, 2, 10, 100, 1000, np.inf]
    names = ["1-2", "3-10", "11-100", "101-1000", ">1000"]
    geo["size_bin"] = pd.cut(geo["node_count"], bins=bins, labels=names, include_lowest=True)
    pivot = (
        geo.pivot_table(
            index="network_label",
            columns="size_bin",
            values="component_id",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .reindex(order)
        .astype(int)
    )

    total_nodes = geo.groupby("network_label")["node_count"].sum().reindex(order)
    top3_nodes = (
        geo.sort_values(["network_label", "node_count"], ascending=[True, False])
        .groupby("network_label")
        .head(3)
        .groupby("network_label")["node_count"]
        .sum()
        .reindex(order)
    )
    top3_share = 100.0 * top3_nodes / total_nodes

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    bottom = np.zeros(len(order))
    bin_colors = ["#D55E00", "#E69F00", "#A6CEE3", "#2F6FBB", "#1B9E77"]
    x = np.arange(len(order))
    for color, name in zip(bin_colors, names):
        vals = pivot[name].to_numpy()
        axes[0].bar(x, vals, bottom=bottom, label=name, color=color)
        bottom += vals
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=25, ha="right")
    axes[0].set_ylabel("Number of components")
    axes[0].set_title("Tail of detached graph components")
    axes[0].legend(title="Nodes", frameon=False, ncol=1)

    total_components = geo.groupby("network_label")["component_id"].count().reindex(order)
    tiny_components = (
        geo[geo["node_count"] <= 10]
        .groupby("network_label")["component_id"]
        .count()
        .reindex(order)
        .fillna(0)
    )
    axes[1].bar(x - 0.18, total_components, 0.36, color=BLUE, label="All components")
    axes[1].bar(x + 0.18, tiny_components, 0.36, color=ORANGE, label="Components <= 10 nodes")
    for i, share in enumerate(top3_share):
        axes[1].text(i, total_components.iloc[i] + 8, f"top 3: {share:.2f}% nodes", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=25, ha="right")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Most components are tiny fragments")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0, max(total_components) * 1.24)

    fig.suptitle("Timor-Leste component statistics before snapping policy", y=1.06)
    savefig("fig_timor_component_size_tail")

    summary["timor_component_tail"] = {
        label: {
            "components": int(total_components.loc[label]),
            "components_le_10_nodes": int(tiny_components.loc[label]),
            "top3_node_share_percent": float(top3_share.loc[label]),
        }
        for label in order
    }


def sparse_memory_figure(summary: dict) -> None:
    meta = load_json(
        ROOT
        / "outputs"
        / "vietnam_1km_5km_approx_pareto_20260629"
        / "vietnam_170_1km_5km_component01_instance_metadata.json"
    )
    npz_path = (
        ROOT
        / "outputs"
        / "vietnam_1km_5km_approx_pareto_20260629"
        / "vietnam_170_1km_5km_component01_maxcover_instance.npz"
    )
    candidates = int(meta["n_candidate_sources_with_incremental_coverage"])
    demand = int(meta["n_coverable_incremental_demand"])
    dense_pairs = candidates * demand
    full_matrix_rows = 386_955_841
    opt_arcs = int(meta["n_arcs"])
    npz_mb = npz_path.stat().st_size / 1024 / 1024
    dense_float64_tb = dense_pairs * 8 / 1e12
    dense_bool_gb = dense_pairs / 1e9
    sparse_density = 100.0 * opt_arcs / dense_pairs
    compression_pairs = dense_pairs / opt_arcs

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), constrained_layout=True)

    names = ["Dense candidate x demand", "Finite routed rows", "Optimization arcs"]
    vals = [dense_pairs, full_matrix_rows, opt_arcs]
    bars = axes[0].barh(names, vals, color=[GRAY, BLUE, GREEN])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Rows or pairs (log scale)")
    axes[0].set_title("The instance is sparse after filtering")
    for bar, val in zip(bars, vals):
        axes[0].text(val * 1.08, bar.get_y() + bar.get_height() / 2, f"{val:,.0f}", va="center", fontsize=8)

    mem_names = ["Dense float64 distances", "Dense boolean cover", "Stored max-cover NPZ"]
    mem_bytes = [dense_pairs * 8, dense_pairs, npz_path.stat().st_size]
    colors = [RED, ORANGE, GREEN]
    bars = axes[1].barh(mem_names, mem_bytes, color=colors)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Bytes (log scale)")
    axes[1].set_title("Memory avoided by sparse construction")
    mem_labels = [f"{dense_float64_tb:.2f} TB", f"{dense_bool_gb:.1f} GB", f"{npz_mb:.1f} MB"]
    for bar, label, val in zip(bars, mem_labels, mem_bytes):
        axes[1].text(val * 1.10, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)

    fig.suptitle("Vietnam 170-center 1 km, 5 km-threshold max-cover instance", y=1.06)
    savefig("fig_vietnam_sparse_memory_funnel")

    summary["vietnam_sparse_memory"] = {
        "candidates_after_filter": candidates,
        "coverable_demand": demand,
        "dense_pairs": dense_pairs,
        "finite_routed_rows_on_disk": full_matrix_rows,
        "optimization_arcs": opt_arcs,
        "sparse_density_percent": sparse_density,
        "dense_pair_to_arc_ratio": compression_pairs,
        "dense_float64_tb": dense_float64_tb,
        "dense_boolean_gb": dense_bool_gb,
        "maxcover_npz_mb": npz_mb,
    }


def runtime_breakdown_figure(summary: dict) -> None:
    meta = load_json(
        ROOT
        / "outputs"
        / "vietnam_1km_5km_approx_pareto_20260629"
        / "vietnam_170_1km_5km_component01_instance_metadata.json"
    )
    stage_names = [
        "Candidate scan",
        "ID mapping",
        "Sparse build",
        "Greedy",
        "Zero-loss drop",
        "Regreedy",
    ]
    stage_seconds = [
        float(meta["candidate_scan_seconds"]),
        float(meta["mapping_seconds"]),
        float(meta["build_seconds"]),
        float(meta["greedy_seconds"]),
        float(meta["compact_seconds"]),
        float(meta["regreedy_seconds"]),
    ]

    timing_csv = (
        ROOT
        / "outputs"
        / "vietnam_road_agg5_20260624_s20_chunked"
        / "vietnam_road_pipeline_case_timings.csv"
    )
    pipeline_seconds = None
    if timing_csv.exists():
        timing = pd.read_csv(timing_csv)
        pipeline_seconds = float(timing.iloc[0]["elapsed_seconds"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), constrained_layout=True)

    x = np.arange(len(stage_names))
    bars = axes[0].bar(x, stage_seconds, color=[BLUE, BLUE, BLUE, GREEN, ORANGE, GREEN])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stage_names, rotation=30, ha="right")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Vietnam 170 optimization-stage timing")
    for bar, sec in zip(bars, stage_seconds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, sec + 1.0, clock(sec), ha="center", va="bottom", fontsize=7)
    axes[0].set_ylim(0, max(stage_seconds) * 1.28)

    labels = ["Greedy saturation", "After zero-loss drop", "Regreedy saturation"]
    selected = [
        int(meta["greedy_selected_count"]),
        int(meta["compact_selected_count"]),
        int(meta["regreedy_selected_count"]),
    ]
    coverage = [
        float(meta["all_candidate_coverage_percent"]),
        float(meta["all_candidate_coverage_percent"]),
        float(meta["all_candidate_coverage_percent"]),
    ]
    bars = axes[1].bar(labels, selected, color=[GREEN, ORANGE, BLUE])
    axes[1].set_ylabel("Selected sites at saturation")
    axes[1].set_title("Zero-loss compacting preserves coverage")
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    for bar, sites, cov in zip(bars, selected, coverage):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            sites + 650,
            f"{sites:,}\n{cov:.4f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_ylim(0, max(selected) * 1.18)

    fig.suptitle("Approximate Pareto engineering: build once, update incrementally", y=1.06)
    savefig("fig_vietnam_engineering_runtime_breakdown")

    summary["vietnam_engineering_runtime"] = {
        "stage_seconds": dict(zip(stage_names, stage_seconds)),
        "stage_clocks": dict(zip(stage_names, [clock(s) for s in stage_seconds])),
        "distance_pipeline_comparison_seconds": pipeline_seconds,
        "distance_pipeline_comparison_clock": None if pipeline_seconds is None else clock(pipeline_seconds),
        "greedy_selected_count": int(meta["greedy_selected_count"]),
        "compact_selected_count": int(meta["compact_selected_count"]),
        "removed_without_coverage_loss": int(meta["greedy_selected_count"] - meta["compact_selected_count"]),
        "coverage_percent": float(meta["all_candidate_coverage_percent"]),
    }


def approximation_figure(summary: dict) -> None:
    curve = pd.read_csv(
        ROOT
        / "outputs"
        / "vietnam_1km_5km_approx_pareto_20260629"
        / "vietnam_1km_5km_approx_pareto_curve.csv"
    )
    meta = load_json(
        ROOT
        / "outputs"
        / "vietnam_1km_5km_approx_pareto_20260629"
        / "vietnam_170_1km_5km_component01_instance_metadata.json"
    )
    # Downsample only for plotting; exact CSV is left untouched.
    max_budget = int(curve["budget"].max())
    sample_budgets = set(range(0, max_budget + 1, max(1, max_budget // 350)))
    sample_budgets.update(
        [
            0,
            20,
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            int(meta["compact_selected_count"]),
            int(meta["greedy_selected_count"]),
            max_budget,
        ]
    )
    curve_small = curve[curve["budget"].isin(sample_budgets)].copy()
    stages = ["greedy", "regreedy_restricted", "pointwise_max"]
    colors = {"greedy": GREEN, "regreedy_restricted": BLUE, "pointwise_max": RED}
    labels = {
        "greedy": "Greedy",
        "regreedy_restricted": "Regreedy after drop",
        "pointwise_max": "Pointwise maximum",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.0), constrained_layout=True)
    for stage in stages:
        data = curve_small[curve_small["stage"] == stage]
        if not data.empty:
            axes[0].plot(
                data["budget"],
                data["coverage_percent_total_population"],
                color=colors[stage],
                label=labels[stage],
                linewidth=2.0 if stage == "pointwise_max" else 1.6,
            )
    axes[0].axhline(meta["all_candidate_coverage_percent"], color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_xlabel("New sites")
    axes[0].set_ylabel("Covered population (%)")
    axes[0].set_title("Approximate Pareto curve to saturation")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_xlim(0, max_budget * 1.02)
    axes[0].set_ylim(20, 101)

    stage_times = [
        float(meta["greedy_seconds"]),
        float(meta["compact_seconds"]),
        float(meta["regreedy_seconds"]),
    ]
    bars = axes[1].bar(["Greedy", "Drop", "Regreedy"], stage_times, color=[GREEN, ORANGE, BLUE])
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Approximation stages are seconds-scale")
    for bar, sec in zip(bars, stage_times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, sec + 0.12, clock(sec), ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(stage_times) * 1.28)

    fig.suptitle("Vietnam 170-center 1 km approximation behavior", y=1.06)
    savefig("fig_vietnam_approximation_behavior")

    summary["vietnam_approximation_behavior"] = {
        "greedy_seconds": float(meta["greedy_seconds"]),
        "compact_seconds": float(meta["compact_seconds"]),
        "regreedy_seconds": float(meta["regreedy_seconds"]),
        "greedy_saturation_p": int(meta["greedy_selected_count"]),
        "compact_saturation_p": int(meta["compact_selected_count"]),
        "pointwise_max_saturation_p": int(meta["pointwise_max_selected_count_at_saturation"]),
        "coverage_percent": float(meta["all_candidate_coverage_percent"]),
    }


def vietnam_component_manifest_summary(summary: dict) -> None:
    manifest = load_json(ROOT / "outputs" / "article_components" / "vietnam_component_snapping_manifest.json")
    facts = manifest["component_facts"]
    summary["vietnam_component_facts_130"] = facts

    tt80 = ROOT / "outputs" / "article_components" / "vietnam_tt80_pipeline_component_inset_manifest.json"
    if tt80.exists():
        summary["vietnam_tt80_pipeline_manifest"] = load_json(tt80)


def main() -> None:
    setup()
    summary: dict = {}
    component_policy_figure(summary)
    component_tail_figure(summary)
    sparse_memory_figure(summary)
    runtime_breakdown_figure(summary)
    approximation_figure(summary)
    vietnam_component_manifest_summary(summary)

    with (OUT / "integrated_expansion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
