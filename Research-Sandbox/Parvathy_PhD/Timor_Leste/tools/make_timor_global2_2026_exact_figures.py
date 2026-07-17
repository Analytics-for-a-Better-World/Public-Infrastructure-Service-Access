from __future__ import annotations

import json
import math
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(
    os.environ.get(
        "PARVATHY_REPLICATION_ROOT",
        r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis",
    )
)
OUTPUT = Path(
    os.environ.get(
        "TIMOR_GLOBAL2_2026_PARETO_OUTPUT",
        str(ROOT / "outputs" / "timor_global2_2026_exact_pareto_20260717"),
    )
)
CASE_ROOT = OUTPUT / "cases"
FIGURES = OUTPUT / "figures"
PIPELINE_OUTPUT = Path(
    os.environ.get(
        "TIMOR_GLOBAL2_2026_PIPELINE_OUTPUT",
        str(
            ROOT
            / "runs"
            / "timor_global2_2026_20260717_clean"
            / "east-timor_data"
            / "outputs"
        ),
    )
)

GRID_COLORS = {10: "#0072B2", 5: "#D55E00", 1: "#009E73"}
THRESHOLD_COLORS = {2: "#7B2CBF", 5: "#D55E00", 10: "#0072B2"}
CANDIDATE_COLOR = "#0057B8"
EXISTING_COLOR = "#E8590C"
SELECTED_COLOR = "#169B62"


def case_id(grid_km: int, threshold_km: int) -> str:
    return f"timor_global2_2026_grid_{grid_km}km_threshold_{threshold_km}km"


def case_dir(grid_km: int, threshold_km: int) -> Path:
    return CASE_ROOT / case_id(grid_km, threshold_km)


def resolve_sources_path() -> Path:
    override = os.environ.get("TIMOR_GLOBAL2_2026_SOURCES_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    metadata = json.loads(
        (case_dir(1, 5) / "instance_metadata.json").read_text(encoding="utf-8")
    )
    manifest_path = Path(metadata["manifest_path"])
    if manifest_path.exists():
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        path = Path(manifest["outputs"]["sources"]["path"])
        if path.exists():
            return path

    matches = sorted(PIPELINE_OUTPUT.glob("sources*.parquet"))
    if not matches:
        raise FileNotFoundError(
            "No source catalog found. Set TIMOR_GLOBAL2_2026_SOURCES_PATH."
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def read_latest_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["budget"])] = record
    return records


def save(fig: plt.Figure, name: str, *, dpi: int = 240) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    plt.close(fig)


def style_plot(ax: plt.Axes) -> None:
    ax.grid(color="#D9DEE3", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def padded_bounds(
    layers: list[gpd.GeoDataFrame], x_fraction: float = 0.035, y_fraction: float = 0.055
) -> tuple[float, float, float, float]:
    bounds = np.vstack([layer.total_bounds for layer in layers])
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)
    xpad = max((xmax - xmin) * x_fraction, 1.0)
    ypad = max((ymax - ymin) * y_fraction, 1.0)
    return xmin - xpad, ymin - ypad, xmax + xpad, ymax + ypad


def add_osm_basemap(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 8,
    washout: float = 0.18,
) -> str | None:
    import contextily as ctx

    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_facecolor("#E8EDF0")
    error = None
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.OpenStreetMap.Mapnik,
            crs="EPSG:3857",
            zoom=zoom,
            attribution_size=4,
        )
    except Exception as exc:
        error = repr(exc)
    if washout:
        ax.axhspan(ymin, ymax, facecolor="white", alpha=washout, zorder=2)
    ax.set_axis_off()
    return error


def load_geography() -> tuple[dict[int, gpd.GeoDataFrame], gpd.GeoDataFrame]:
    candidates = {}
    for grid_km in (10, 5, 1):
        catalog = pd.read_csv(case_dir(grid_km, 10) / "candidate_catalog.csv")
        candidates[grid_km] = gpd.GeoDataFrame(
            catalog,
            geometry=gpd.points_from_xy(catalog["Longitude"], catalog["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

    sources = pd.read_parquet(resolve_sources_path())
    amenities = sources.loc[sources["source_type"].astype(str) == "amenities"].copy()
    existing = gpd.GeoDataFrame(
        amenities,
        geometry=gpd.points_from_xy(amenities["Longitude"], amenities["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    return candidates, existing


def draw_candidate_layer(
    ax: plt.Axes,
    candidates: gpd.GeoDataFrame,
    existing: gpd.GeoDataFrame,
) -> None:
    ax.plot(
        candidates.geometry.x,
        candidates.geometry.y,
        linestyle="None",
        marker=".",
        markersize=1.35,
        markeredgewidth=0,
        color=CANDIDATE_COLOR,
        alpha=0.76,
        rasterized=True,
        zorder=3,
    )
    ax.scatter(
        existing.geometry.x,
        existing.geometry.y,
        marker="D",
        s=15,
        facecolor=EXISTING_COLOR,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.94,
        zorder=5,
    )


def make_candidate_maps() -> list[str]:
    candidates, existing = load_geography()
    bounds = padded_bounds(list(candidates.values()) + [existing])
    errors: list[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 7.8), gridspec_kw={"hspace": 0.035})
    for ax, grid_km in zip(axes, (10, 5, 1)):
        error = add_osm_basemap(ax, bounds)
        if error:
            errors.append(error)
        draw_candidate_layer(ax, candidates[grid_km], existing)
        ax.text(
            0.015,
            0.97,
            f"{grid_km} km grid: {len(candidates[grid_km]):,} candidates",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color="#172026",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.2},
            zorder=10,
        )
    save(fig, "timor_global2_2026_candidate_grids_stacked", dpi=260)

    for grid_km in (10, 5, 1):
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        error = add_osm_basemap(ax, bounds)
        if error:
            errors.append(error)
        draw_candidate_layer(ax, candidates[grid_km], existing)
        save(fig, f"timor_global2_2026_candidate_grid_{grid_km}km", dpi=300)
    return errors


def make_typical_solution() -> dict:
    grid_km = 1
    threshold_km = 5
    budget = 100
    directory = case_dir(grid_km, threshold_km)
    solutions = read_latest_jsonl(directory / "solutions.jsonl")
    frontier = read_latest_jsonl(directory / "frontier.jsonl")
    solution = solutions[budget]
    point = frontier[budget]

    catalog = pd.read_csv(directory / "candidate_catalog.csv")
    catalog["candidate_id"] = catalog["candidate_id"].astype(str)
    candidate_gdf = gpd.GeoDataFrame(
        catalog,
        geometry=gpd.points_from_xy(catalog["Longitude"], catalog["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    selected_ids = set(map(str, solution["candidate_ids"]))
    selected = candidate_gdf.loc[candidate_gdf["candidate_id"].isin(selected_ids)].copy()
    if len(selected) != int(solution["budget"]):
        raise RuntimeError("Stored solution does not map one-to-one to candidate coordinates")

    sources = pd.read_parquet(resolve_sources_path())
    amenities = sources.loc[sources["source_type"].astype(str) == "amenities"].copy()
    existing = gpd.GeoDataFrame(
        amenities,
        geometry=gpd.points_from_xy(amenities["Longitude"], amenities["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    bounds = padded_bounds([candidate_gdf, existing])

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    basemap_error = add_osm_basemap(ax, bounds, washout=0.13)
    ax.plot(
        candidate_gdf.geometry.x,
        candidate_gdf.geometry.y,
        linestyle="None",
        marker=".",
        markersize=0.65,
        markeredgewidth=0,
        color=CANDIDATE_COLOR,
        alpha=0.28,
        rasterized=True,
        zorder=3,
    )
    ax.scatter(
        existing.geometry.x,
        existing.geometry.y,
        marker="D",
        s=16,
        facecolor=EXISTING_COLOR,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
        zorder=4,
    )
    ax.scatter(
        selected.geometry.x,
        selected.geometry.y,
        marker="o",
        s=28,
        facecolor=SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.98,
        zorder=5,
    )
    save(fig, "timor_global2_2026_exact_solution_grid1km_threshold5km_p100", dpi=300)

    metadata = {
        "grid_km": grid_km,
        "threshold_km": threshold_km,
        "budget": budget,
        "selected_count": len(selected),
        "coverage_pct": float(point["coverage_pct"]),
        "total_covered_population": float(point["total_covered_population"]),
        "baseline_coverage_pct": float(
            json.loads((directory / "instance_metadata.json").read_text(encoding="utf-8"))[
                "baseline_coverage_pct"
            ]
        ),
        "basemap_error": basemap_error,
    }
    (FIGURES / "timor_global2_2026_exact_solution_p100_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def make_five_km_threshold_overlay() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    maximum_saturation = 0
    series = {}
    for grid_km in (10, 5, 1):
        records = read_latest_jsonl(case_dir(grid_km, 5) / "frontier.jsonl")
        frame = pd.DataFrame(records.values()).sort_values("budget")
        saturation_budget = int(frame["budget"].max())
        maximum_saturation = max(maximum_saturation, saturation_budget)
        series[grid_km] = (frame, saturation_budget)
    x_end = int(math.ceil(maximum_saturation * 1.08 / 50.0) * 50)

    for grid_km in (10, 5, 1):
        frame, saturation_budget = series[grid_km]
        saturation_coverage = float(frame.iloc[-1]["coverage_pct"])
        color = GRID_COLORS[grid_km]
        ax.plot(
            frame["budget"],
            frame["coverage_pct"],
            color=color,
            linewidth=2.0,
            label=f"{grid_km} km candidate grid",
        )
        ax.scatter([saturation_budget], [saturation_coverage], color=color, s=30, zorder=4)
        if saturation_budget < x_end:
            ax.plot(
                [saturation_budget, x_end],
                [saturation_coverage, saturation_coverage],
                color=color,
                linewidth=1.5,
                linestyle="--",
            )
    ax.set_xlim(0, x_end)
    ax.set_xlabel("New facilities")
    ax.set_ylabel("Covered population (%)")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    style_plot(ax)
    save(fig, "timor_global2_2026_exact_pareto_threshold5km_by_grid")


def make_saturation_resolution_figure() -> None:
    summary = pd.read_csv(OUTPUT / "exact_frontier_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    x = np.arange(3)
    grid_order = [10, 5, 1]
    for threshold_km in (2, 5, 10):
        group = summary.loc[summary["threshold_km"] == threshold_km].set_index("grid_km")
        values = [float(group.loc[grid, "saturation_coverage_pct"]) for grid in grid_order]
        budgets = [int(group.loc[grid, "exact_saturation_budget"]) for grid in grid_order]
        color = THRESHOLD_COLORS[threshold_km]
        axes[0].plot(x, values, marker="o", linewidth=2.0, color=color, label=f"{threshold_km} km threshold")
        axes[1].plot(x, budgets, marker="o", linewidth=2.0, color=color, label=f"{threshold_km} km threshold")
    for ax in axes:
        ax.set_xticks(x, ["10 km", "5 km", "1 km"])
        ax.set_xlabel("Candidate-grid spacing")
        style_plot(ax)
    axes[0].set_ylabel("Exact saturation coverage (%)")
    axes[1].set_ylabel("Exact saturation budget")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, loc="best")
    save(fig, "timor_global2_2026_saturation_by_resolution")


def clock_tick(seconds: float, _position: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remaining = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}:{remaining:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{remaining:02d}"


def make_runtime_figure() -> None:
    summary = pd.read_csv(OUTPUT / "exact_frontier_summary.csv")
    summary = summary.sort_values(["grid_km", "threshold_km"], ascending=[True, False])
    labels = [f"{row.grid_km:g} km grid / {row.threshold_km:g} km threshold" for row in summary.itertuples()]
    solve = summary["gurobi_solve_seconds"].to_numpy(float)
    overhead = np.maximum(summary["exact_wall_seconds"].to_numpy(float) - solve, 0.0)
    y = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(y, solve, color="#0072B2", label="Gurobi solve")
    ax.barh(y, overhead, left=solve, color="#B8C2CC", label="checkpoint and session overhead")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Elapsed time")
    ax.xaxis.set_major_formatter(FuncFormatter(clock_tick))
    ax.legend(frameon=False, loc="lower right")
    style_plot(ax)
    save(fig, "timor_global2_2026_exact_runtime_by_case")


def write_figure_manifest(solution_metadata: dict, basemap_errors: list[str]) -> None:
    rows = [
        ("timor_global2_2026_exact_pareto_3x3", "Nine complete exact Pareto frontiers."),
        ("timor_global2_2026_exact_pareto_threshold5km_by_grid", "Exact 5 km-threshold comparison across candidate-grid resolutions."),
        ("timor_global2_2026_saturation_by_resolution", "Coverage ceilings and saturation budgets as spatial resolution increases."),
        ("timor_global2_2026_exact_runtime_by_case", "Exact solve and durable-checkpoint wall times."),
        ("timor_global2_2026_candidate_grids_stacked", "Current candidate grids with OSM health amenities."),
        ("timor_global2_2026_candidate_grid_10km", "Large individual 10 km candidate-grid map."),
        ("timor_global2_2026_candidate_grid_5km", "Large individual 5 km candidate-grid map."),
        ("timor_global2_2026_candidate_grid_1km", "Large individual 1 km candidate-grid map."),
        ("timor_global2_2026_exact_solution_grid1km_threshold5km_p100", "Representative exact p=100 deployment on the 1 km grid."),
    ]
    lines = ["# Timor-Leste WorldPop 2026 figure set", ""]
    for stem, description in rows:
        lines.append(f"- `{stem}.pdf` and `.png`: {description}")
    lines.extend(
        [
            "",
            "## Representative solution",
            "",
            f"The mapped exact solution uses the 1 km candidate grid, a 5 km service threshold, and p={solution_metadata['budget']}. ",
            f"It covers {solution_metadata['coverage_pct']:.3f}% of the population, compared with a baseline of {solution_metadata['baseline_coverage_pct']:.3f}% from existing OSM health amenities.",
            "Blue points are candidate sites, orange diamonds are existing OSM health amenities, and green circles are selected new sites.",
            "",
            f"OSM basemap errors: {len(basemap_errors) + int(solution_metadata['basemap_error'] is not None)}",
        ]
    )
    (FIGURES / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    basemap_errors = make_candidate_maps()
    solution_metadata = make_typical_solution()
    make_five_km_threshold_overlay()
    make_saturation_resolution_figure()
    make_runtime_figure()
    write_figure_manifest(solution_metadata, basemap_errors)
    print(FIGURES)


if __name__ == "__main__":
    main()
