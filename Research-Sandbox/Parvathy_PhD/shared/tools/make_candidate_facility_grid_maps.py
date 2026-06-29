from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis")
RUNS = ROOT / "runs"
ARTICLE = ROOT / "articles" / "seps_access_optimization"
ASSET_DIR = ROOT / "outputs" / "deck_assets" / "maps"


@dataclass(frozen=True)
class GridLayer:
    label: str
    spacing_m: int
    path: Path


@dataclass(frozen=True)
class CountryMap:
    key: str
    country: str
    crs_epsg: int
    zoom: int
    grids: tuple[GridLayer, ...]
    facility_path: Path
    facility_source_type: str
    facility_label: str
    composite_size: tuple[float, float]
    single_size: tuple[float, float]
    basemap_washout: float
    candidate_markersize: float
    candidate_alpha: float
    facility_size: float


COUNTRIES = (
    CountryMap(
        key="timor",
        country="Timor-Leste",
        crs_epsg=32751,
        zoom=8,
        grids=(
            GridLayer(
                "10 km",
                10000,
                RUNS
                / "timor_network_profile_20260623"
                / "east-timor_data"
                / "cache"
                / "tls_candidate_sites_spacing_10000m_no_water_include_boundary_epsg_32751.pkl",
            ),
            GridLayer(
                "5 km",
                5000,
                RUNS
                / "timor_network_profile_20260623"
                / "east-timor_data"
                / "cache"
                / "tls_candidate_sites_spacing_5000m_no_water_include_boundary_epsg_32751.pkl",
            ),
            GridLayer(
                "1 km",
                1000,
                RUNS
                / "timor_network_profile_20260623"
                / "east-timor_data"
                / "cache"
                / "tls_candidate_sites_spacing_1000m_no_water_include_boundary_epsg_32751.pkl",
            ),
        ),
        facility_path=RUNS
        / "timor_network_profile_20260623"
        / "east-timor_data"
        / "outputs"
        / "sources_pop_1_sample_1_seed_42_max_none_agg_none_maxdist_5000_amenity_amenity_all-dst_population-sr_df8c27ffbca5.parquet",
        facility_source_type="amenities",
        facility_label="existing health amenities",
        composite_size=(11.2, 2.4),
        single_size=(7.2, 3.2),
        basemap_washout=0.42,
        candidate_markersize=1.25,
        candidate_alpha=0.76,
        facility_size=16.0,
    ),
    CountryMap(
        key="vietnam",
        country="Vietnam",
        crs_epsg=3405,
        zoom=6,
        grids=(
            GridLayer(
                "10 km",
                10000,
                RUNS
                / "vietnam_170_agg5_20260624_s20"
                / "vietnam_data"
                / "cache"
                / "vnm_candidate_sites_spacing_10000m_water_allowed_include_boundary_epsg_3405.pkl",
            ),
            GridLayer(
                "5 km",
                5000,
                RUNS
                / "vietnam_170_agg5_20260624_s20"
                / "vietnam_data"
                / "cache"
                / "vnm_candidate_sites_spacing_5000m_water_allowed_include_boundary_epsg_3405.pkl",
            ),
            GridLayer(
                "1 km",
                1000,
                RUNS
                / "vietnam_170_agg5_20260624_s20"
                / "vietnam_data"
                / "cache"
                / "vnm_candidate_sites_spacing_1000m_water_allowed_include_boundary_epsg_3405.pkl",
            ),
        ),
        facility_path=RUNS
        / "vietnam_170_agg5_20260624_s20"
        / "vietnam_data"
        / "outputs"
        / "sources_pop_1_sample_1_seed_42_max_none_agg_5_maxdist_20000_amenity_amenity_all-dst_population-src__c200c8f9e0da.parquet",
        facility_source_type="table",
        facility_label="stroke-service facilities",
        composite_size=(11.2, 6.8),
        single_size=(4.6, 7.6),
        basemap_washout=0.50,
        candidate_markersize=1.15,
        candidate_alpha=0.82,
        facility_size=30.0,
    ),
)


def read_candidate_grid(layer: GridLayer, epsg: int) -> gpd.GeoDataFrame:
    obj = pd.read_pickle(layer.path)
    if not isinstance(obj, gpd.GeoDataFrame):
        if isinstance(obj, pd.DataFrame) and "geometry" in obj.columns:
            obj = gpd.GeoDataFrame(obj, geometry="geometry")
        else:
            raise TypeError(f"{layer.path} is not a GeoDataFrame-like pickle")
    gdf = obj.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    return gdf.loc[gdf.geometry.notna()].to_crs(epsg=3857)


def read_facilities(country: CountryMap) -> gpd.GeoDataFrame:
    frame = pd.read_parquet(country.facility_path)
    frame = frame.loc[frame["source_type"].astype(str) == country.facility_source_type].copy()
    frame = frame.dropna(subset=["Longitude", "Latitude"])
    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
        crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=3857)


def padded_bounds(layers: list[gpd.GeoDataFrame], x_pad_frac: float = 0.06, y_pad_frac: float = 0.06) -> tuple[float, float, float, float]:
    bounds = pd.DataFrame([gdf.total_bounds for gdf in layers], columns=["xmin", "ymin", "xmax", "ymax"])
    xmin = float(bounds["xmin"].min())
    ymin = float(bounds["ymin"].min())
    xmax = float(bounds["xmax"].max())
    ymax = float(bounds["ymax"].max())
    xpad = max((xmax - xmin) * x_pad_frac, 1.0)
    ypad = max((ymax - ymin) * y_pad_frac, 1.0)
    return xmin - xpad, ymin - ypad, xmax + xpad, ymax + ypad


def add_basemap_and_washout(ax: plt.Axes, bounds: tuple[float, float, float, float], zoom: int, washout: float) -> str | None:
    import contextily as cx

    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_facecolor("#eef2f4")

    basemap_error = None
    try:
        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=cx.providers.OpenStreetMap.Mapnik,
            zoom=zoom,
            attribution_size=4,
        )
    except Exception as exc:
        basemap_error = repr(exc)

    if washout:
        ax.axhspan(ymin, ymax, xmin=0, xmax=1, facecolor="white", alpha=washout, zorder=2)
    ax.set_axis_off()
    return basemap_error


def plot_layers(
    ax: plt.Axes,
    candidates: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    country: CountryMap,
) -> None:
    ax.plot(
        candidates.geometry.x.to_numpy(),
        candidates.geometry.y.to_numpy(),
        linestyle="None",
        marker=".",
        markersize=country.candidate_markersize,
        markeredgewidth=0,
        color="#0047ff",
        alpha=country.candidate_alpha,
        rasterized=True,
        zorder=3,
    )
    ax.scatter(
        facilities.geometry.x.to_numpy(),
        facilities.geometry.y.to_numpy(),
        marker="D",
        s=country.facility_size,
        facecolor="#e8590c",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.96,
        zorder=5,
    )


def legend_handles(country: CountryMap) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=".",
            color="none",
            markerfacecolor="#0047ff",
            markeredgewidth=0,
            markersize=8,
            alpha=country.candidate_alpha,
            label="candidate grid",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#e8590c",
            markeredgecolor="white",
            markersize=6,
            label=country.facility_label,
        ),
    ]


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=240, facecolor="white")


def make_country_maps(country: CountryMap) -> dict:
    grids = [(layer, read_candidate_grid(layer, country.crs_epsg)) for layer in country.grids]
    facilities = read_facilities(country)
    bounds = padded_bounds([gdf for _, gdf in grids] + [facilities])
    basemap_errors: list[str] = []

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(country.composite_size[0] * 0.86, country.composite_size[1]),
        dpi=180,
        gridspec_kw={"wspace": 0.03},
    )
    fig.patch.set_facecolor("white")
    for ax, (layer, candidates) in zip(axes, grids):
        error = add_basemap_and_washout(ax, bounds, country.zoom, country.basemap_washout)
        if error:
            basemap_errors.append(error)
        plot_layers(ax, candidates, facilities, country)
        ax.text(
            0.02,
            0.98,
            layer.label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#172026",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 2.5},
            zorder=10,
        )

    composite_base = ARTICLE / f"fig_{country.key}_candidate_grids_facilities"
    save_figure(fig, composite_base)
    plt.close(fig)

    stacked_base = None
    if country.key == "timor":
        fig_stacked, axes_stacked = plt.subplots(
            3,
            1,
            figsize=(7.2, 6.2),
            dpi=180,
            gridspec_kw={"hspace": 0.05},
        )
        fig_stacked.patch.set_facecolor("white")
        for row, (layer, candidates) in enumerate(grids):
            ax = axes_stacked[row]
            error = add_basemap_and_washout(ax, bounds, country.zoom, country.basemap_washout)
            if error:
                basemap_errors.append(error)
            plot_layers(ax, candidates, facilities, country)
            ax.text(
                0.02,
                0.96,
                layer.label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="#172026",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 2.5},
                zorder=10,
            )
        stacked_base = ARTICLE / f"fig_{country.key}_candidate_grids_facilities_stacked"
        save_figure(fig_stacked, stacked_base)
        plt.close(fig_stacked)

    single_records = []
    for layer, candidates in grids:
        fig_single, (ax, legend_ax_single) = plt.subplots(
            1,
            2,
            figsize=country.single_size,
            dpi=180,
            gridspec_kw={"width_ratios": [1, 0.24], "wspace": 0.02},
        )
        fig_single.patch.set_facecolor("white")
        error = add_basemap_and_washout(ax, bounds, country.zoom, country.basemap_washout)
        if error:
            basemap_errors.append(error)
        plot_layers(ax, candidates, facilities, country)
        legend_ax_single.axis("off")
        legend_ax_single.legend(
            handles=legend_handles(country),
            loc="center left",
            frameon=False,
            fontsize=8.0,
            handletextpad=0.6,
            borderaxespad=0,
        )
        single_base = ARTICLE / f"fig_{country.key}_candidate_grid_{layer.spacing_m // 1000}km_facilities"
        save_figure(fig_single, single_base)
        plt.close(fig_single)
        single_records.append(
            {
                "spacing": layer.label,
                "candidate_count": int(len(candidates)),
                "output_pdf": str(single_base.with_suffix(".pdf")),
                "output_png": str(single_base.with_suffix(".png")),
            }
        )

    return {
        "country": country.country,
        "key": country.key,
        "facility_count": int(len(facilities)),
        "facility_label": country.facility_label,
        "facility_source": str(country.facility_path),
        "facility_source_type": country.facility_source_type,
        "composite_pdf": str(composite_base.with_suffix(".pdf")),
        "composite_png": str(composite_base.with_suffix(".png")),
        "stacked_pdf": str(stacked_base.with_suffix(".pdf")) if stacked_base is not None else None,
        "stacked_png": str(stacked_base.with_suffix(".png")) if stacked_base is not None else None,
        "grids": single_records,
        "basemap": "OpenStreetMap.Mapnik via contextily",
        "basemap_errors": sorted(set(basemap_errors)),
    }


def main() -> None:
    ARTICLE.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    records = [make_country_maps(country) for country in COUNTRIES]
    manifest = ARTICLE / "candidate_grid_facility_maps_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
