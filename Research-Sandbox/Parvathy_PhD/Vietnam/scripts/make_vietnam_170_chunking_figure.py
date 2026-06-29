from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis")
ARTICLE = ROOT / "articles" / "seps_access_optimization"
SUCCESS = (
    ROOT
    / "runs"
    / "vietnam_170_agg5_20260624_s20"
    / "vietnam_data"
    / "outputs"
    / "distance_matrix_src_candidates_dst_population_pop_1_sample_1_seed_42_max_none_agg_5_maxdist_20000_a_15fe840aed2e.parquet_parts"
    / "_SUCCESS.json"
)
OUT = ARTICLE / "fig_vietnam_170_1km_adaptive_chunking.pdf"
OUT_PNG = ARTICLE / "fig_vietnam_170_1km_adaptive_chunking.png"


def main() -> None:
    summary = json.loads(SUCCESS.read_text(encoding="utf-8"))
    chunks = pd.DataFrame(summary["chunks"]).reset_index(names="chunk")
    chunks["spatial_pairs_m"] = (
        chunks["estimated_spatial_candidate_pairs"] / 1_000_000
    )
    chunks["sparse_rows_m"] = chunks["sparse_row_count"] / 1_000_000
    chunks["unique_node_pairs_m"] = chunks["unique_node_pair_count"] / 1_000_000
    chunks["target_count_k"] = chunks["target_count"] / 1_000

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )

    ax = axes[0]
    ax.bar(
        chunks["chunk"],
        chunks["target_count_k"],
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_ylabel("targets per chunk (thousand)")
    ax.set_title("Automatic target chunk sizes")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)

    ax = axes[1]
    ax.plot(
        chunks["chunk"],
        chunks["spatial_pairs_m"],
        color="#0072B2",
        linewidth=1.7,
        label="estimated spatial pairs",
    )
    ax.plot(
        chunks["chunk"],
        chunks["unique_node_pairs_m"],
        color="#D55E00",
        linewidth=1.5,
        label="unique road-node pairs",
    )
    ax.plot(
        chunks["chunk"],
        chunks["sparse_rows_m"],
        color="#009E73",
        linewidth=1.5,
        label="written sparse rows",
    )
    ax.axhline(
        summary["max_spatial_pairs_per_chunk"] / 1_000_000,
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        label="25M pair cap",
    )
    ax.set_xlabel("matrix chunk")
    ax.set_ylabel("count (million)")
    ax.set_title("Pair volume and output rows")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.legend(loc="upper left", ncol=2, frameon=False)

    fig.suptitle(
        "Vietnam 170-center 1 km road matrix: realized adaptive chunks",
        fontsize=11,
        fontweight="bold",
        x=0.02,
        y=0.99,
        ha="left",
    )
    fig.text(
        0.02,
        0.012,
        (
            "Source: pipeline _SUCCESS.json. "
            "386,955,841 sparse candidate-population rows in 51 chunks; "
            "50 chunks adjusted from a 500,000-target ceiling."
        ),
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT)
    fig.savefig(OUT_PNG, dpi=220)
    print(OUT)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
