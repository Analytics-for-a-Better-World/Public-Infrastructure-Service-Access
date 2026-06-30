from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis")
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "outputs" / "deck_assets" / "maps"

CASES = [
    {
        "key": "timor_5km",
        "country": "Timor-Leste",
        "label": "5 km grid",
        "spacing": "5 km",
        "crs": 32751,
        "zoom": 9,
        "color": "#005f73",
        "size": 3.5,
        "alpha": 0.88,
        "path": RUNS
        / "TimorLeste_20260618_220002"
        / "east-timor_data"
        / "cache"
        / "tls_candidate_sites_spacing_5000m_no_water_include_boundary_epsg_32751.pkl",
    },
    {
        "key": "timor_2p5km",
        "country": "Timor-Leste",
        "label": "2.5 km grid",
        "spacing": "2.5 km",
        "crs": 32751,
        "zoom": 9,
        "color": "#0a9396",
        "size": 2.2,
        "alpha": 0.80,
        "path": RUNS
        / "TimorLeste_20260618_220002"
        / "east-timor_data"
        / "cache"
        / "tls_candidate_sites_spacing_2500m_no_water_include_boundary_epsg_32751.pkl",
    },
    {
        "key": "vietnam_10km",
        "country": "Vietnam",
        "label": "10 km grid",
        "spacing": "10 km",
        "crs": 3405,
        "zoom": 6,
        "color": "#0047FF",
        "size": 0.8,
        "alpha": 0.95,
        "figsize": (5.4, 8.6),
        "annotate": False,
        "marker": ".",
        "plot_markersize": 1.25,
        "basemap_washout": 0.50,
        "slide_size": (250, 386),
        "slide_markersize": 1.15,
        "slide_basemap_washout": 0.56,
        "path": RUNS
        / "vietnam_20260619_0630"
        / "vietnam_data"
        / "cache"
        / "vnm_candidate_sites_spacing_10000m_water_allowed_include_boundary_epsg_3405.pkl",
    },
    {
        "key": "vietnam_5km",
        "country": "Vietnam",
        "label": "5 km grid",
        "spacing": "5 km",
        "crs": 3405,
        "zoom": 6,
        "color": "#0047FF",
        "size": 0.8,
        "alpha": 0.95,
        "figsize": (5.4, 8.6),
        "annotate": False,
        "marker": ".",
        "plot_markersize": 1.25,
        "basemap_washout": 0.50,
        "slide_size": (250, 386),
        "slide_markersize": 1.15,
        "slide_basemap_washout": 0.56,
        "path": RUNS
        / "vietnam_20260619_0630"
        / "vietnam_data"
        / "cache"
        / "vnm_candidate_sites_spacing_5000m_water_allowed_include_boundary_epsg_3405.pkl",
    },
    {
        "key": "vietnam_1km",
        "country": "Vietnam",
        "label": "1 km grid",
        "spacing": "1 km",
        "crs": 3405,
        "zoom": 6,
        "color": "#0047FF",
        "size": 0.8,
        "alpha": 0.95,
        "figsize": (5.4, 8.6),
        "annotate": False,
        "marker": ".",
        "plot_markersize": 1.25,
        "basemap_washout": 0.50,
        "slide_size": (250, 386),
        "slide_markersize": 1.15,
        "slide_basemap_washout": 0.56,
        "path": RUNS
        / "vietnam_20260619_0630"
        / "vietnam_data"
        / "cache"
        / "vnm_candidate_sites_spacing_1000m_water_allowed_include_boundary_epsg_3405.pkl",
    },
]


def to_geodataframe(obj: object, epsg: int) -> gpd.GeoDataFrame:
    if isinstance(obj, gpd.GeoDataFrame):
        gdf = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if "geometry" in obj.columns:
            gdf = gpd.GeoDataFrame(obj.copy(), geometry="geometry")
        elif {"lon", "lat"}.issubset(obj.columns):
            gdf = gpd.GeoDataFrame(
                obj.copy(),
                geometry=gpd.points_from_xy(obj["lon"], obj["lat"]),
                crs="EPSG:4326",
            )
        elif {"longitude", "latitude"}.issubset(obj.columns):
            gdf = gpd.GeoDataFrame(
                obj.copy(),
                geometry=gpd.points_from_xy(obj["longitude"], obj["latitude"]),
                crs="EPSG:4326",
            )
        else:
            raise ValueError(f"DataFrame has no recognizable geometry columns: {obj.columns}")
    elif isinstance(obj, dict):
        for value in obj.values():
            try:
                return to_geodataframe(value, epsg)
            except Exception:
                continue
        raise ValueError("No GeoDataFrame-like value found in dictionary pickle")
    else:
        raise TypeError(f"Unsupported pickle payload: {type(obj)!r}")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    return gdf[gdf.geometry.notna()].copy()


def plot_case(case: dict) -> dict:
    import contextily as cx

    src = Path(case["path"])
    gdf = to_geodataframe(pd.read_pickle(src), int(case["crs"]))
    gdf3857 = gdf.to_crs(epsg=3857)

    xmin, ymin, xmax, ymax = gdf3857.total_bounds
    xpad = max((xmax - xmin) * 0.07, 1)
    ypad = max((ymax - ymin) * 0.07, 1)

    fig, ax = plt.subplots(figsize=case.get("figsize", (10.4, 7.2)), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef2f4")
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")

    basemap_error = None
    try:
        cx.add_basemap(
            ax,
            crs=gdf3857.crs,
            source=cx.providers.OpenStreetMap.Mapnik,
            zoom=int(case["zoom"]),
            attribution_size=6,
        )
    except Exception as exc:  # Keep a readable fallback and record the failure.
        basemap_error = repr(exc)

    if case.get("basemap_washout", 0):
        ax.axhspan(
            ymin - ypad,
            ymax + ypad,
            xmin=0,
            xmax=1,
            facecolor="white",
            alpha=float(case["basemap_washout"]),
            zorder=2,
        )

    x = gdf3857.geometry.x.to_numpy()
    y = gdf3857.geometry.y.to_numpy()
    if "plot_markersize" in case:
        ax.plot(
            x,
            y,
            linestyle="None",
            marker=case.get("marker", "."),
            markersize=float(case["plot_markersize"]),
            markeredgewidth=0,
            color=case["color"],
            alpha=float(case["alpha"]),
            rasterized=True,
            zorder=3,
        )
    else:
        ax.scatter(
            x,
            y,
            s=float(case["size"]),
            c=case["color"],
            alpha=float(case["alpha"]),
            marker=case.get("marker", "o"),
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    ax.set_axis_off()
    if case.get("annotate", True):
        title = f"{case['country']} candidate grid: {case['label']}"
        subtitle = f"{len(gdf):,} candidate sites | OSM basemap | CRS EPSG:{case['crs']}"
        ax.text(
            0.012,
            0.988,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#172026",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d5dde2", alpha=0.94),
        )
        ax.text(
            0.012,
            0.928,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d5dde2", alpha=0.90),
        )

    out_path = OUT_DIR / f"{case['key']}_osm_candidates.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    slide_out_path = None
    if "slide_size" in case:
        slide_w, slide_h = case["slide_size"]
        fig2 = plt.figure(figsize=(slide_w / 100, slide_h / 100), dpi=100)
        fig2.patch.set_facecolor("white")
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.set_facecolor("#eef2f4")
        ax2.set_xlim(xmin - xpad, xmax + xpad)
        ax2.set_ylim(ymin - ypad, ymax + ypad)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_anchor("C")
        try:
            cx.add_basemap(
                ax2,
                crs=gdf3857.crs,
                source=cx.providers.OpenStreetMap.Mapnik,
                zoom=int(case["zoom"]),
                attribution_size=3,
            )
        except Exception:
            pass
        if case.get("slide_basemap_washout", 0):
            ax2.axhspan(
                ymin - ypad,
                ymax + ypad,
                xmin=0,
                xmax=1,
                facecolor="white",
                alpha=float(case["slide_basemap_washout"]),
                zorder=2,
            )
        ax2.plot(
            x,
            y,
            linestyle="None",
            marker=case.get("marker", "."),
            markersize=float(case.get("slide_markersize", case.get("plot_markersize", 1))),
            markeredgewidth=0,
            color=case["color"],
            alpha=float(case["alpha"]),
            rasterized=True,
            zorder=3,
        )
        ax2.set_axis_off()
        slide_out_path = OUT_DIR / f"{case['key']}_osm_candidates_slide.png"
        fig2.savefig(slide_out_path, dpi=100, facecolor="white")
        plt.close(fig2)

    record = {
        "key": case["key"],
        "country": case["country"],
        "label": case["label"],
        "spacing": case["spacing"],
        "source": str(src),
        "output": str(out_path),
        "candidate_count": int(len(gdf)),
        "crs": f"EPSG:{case['crs']}",
        "zoom": int(case["zoom"]),
        "basemap": "OpenStreetMap.Mapnik",
        "basemap_error": basemap_error,
    }
    if slide_out_path is not None:
        record["slide_output"] = str(slide_out_path)
    return record


def make_composite(records: list[dict], keys: list[str], out_path: Path) -> None:
    images = [
        Image.open(next(r.get("slide_output", r["output"]) for r in records if r["key"] == key)).convert("RGB")
        for key in keys
    ]
    target_height = min(img.height for img in images)
    resized = [
        img.resize((int(img.width * target_height / img.height), target_height), Image.LANCZOS)
        for img in images
    ]
    gap = 18
    width = sum(img.width for img in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (width, target_height), "white")
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    canvas.save(out_path, quality=95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [plot_case(case) for case in CASES]
    make_composite(records, ["timor_5km", "timor_2p5km"], OUT_DIR / "timor_candidate_grids_osm_composite.png")
    make_composite(
        records,
        ["vietnam_10km", "vietnam_5km", "vietnam_1km"],
        OUT_DIR / "vietnam_candidate_grids_osm_composite.png",
    )
    (OUT_DIR / "candidate_osm_maps_manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
