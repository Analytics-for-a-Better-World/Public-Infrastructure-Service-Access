from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from pyproj import Transformer

from make_vietnam_component_figure import WORKBOOK, read_first_sheet


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(r"C:\github\Public-Infrastructure-Service-Access\Research-Sandbox\general_distances_per_country")
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from countries.vietnam import CFG as VIETNAM_CFG  # noqa: E402
from distance_pipeline.cache import CacheManager  # noqa: E402
from distance_pipeline.io import download_file  # noqa: E402
from distance_pipeline.network import load_osm_network_data  # noqa: E402
from distance_pipeline.snapping import snap_points_to_nodes  # noqa: E402
from run_pipeline import compute_network_component_labels, filter_nodes_to_components  # noqa: E402


SOURCE_TT = 80
SOURCE_ID = f"source_table_{SOURCE_TT}"
NETWORK_BACKEND = "osmium"
SIMPLIFY_NETWORK = False

# Local crop around Bệnh viện Đa khoa tỉnh Gia Lai / source_table_80.
BBOX = (108.0198, 13.9756, 108.0364, 13.9894)

RUN_ROOT = ROOT / "runs" / "vietnam_tt80_component_pipeline"
OUT_DIR = ROOT / "outputs" / "article_components"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "pipeline_component_inset"
CLI_RUN_MANIFEST = OUT_DIR / "vietnam_tt80_pipeline_cli_runs.json"

DIAGNOSTIC_FACTS = {
    "source": "Mail exchange with Trang Luu, 2026-06-04 to 2026-06-08; corrected component-aware run inspected 2026-06-22",
    "network_nodes_original_diagnostic": 28_918_800,
    "network_edges_original_diagnostic": 58_794_098,
    "weak_components_original_diagnostic": 10_288,
    "giant_component_nodes_original_diagnostic": 28_184_668,
    "giant_component_share_percent_original_diagnostic": 97.46,
    "isolated_component_nodes_original_diagnostic": 15,
    "corrected_snap_node": 7_780_318_361,
    "corrected_snap_component": 0,
    "corrected_snap_distance_m": 67.887876,
    "retained_destination_rows_under_150km": 29_719,
}


def source_record() -> dict[str, object]:
    rows = read_first_sheet(WORKBOOK)
    for row in rows:
        if int(row["TT"]) == SOURCE_TT:
            return row
    raise RuntimeError(f"Could not find TT {SOURCE_TT} in {WORKBOOK}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def url_head_metadata(url: str) -> dict[str, str]:
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=30) as response:
            return {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"date", "etag", "last-modified", "content-length", "content-type"}
            }
    except (URLError, TimeoutError, OSError) as exc:
        return {"head_error": f"{type(exc).__name__}: {exc}"}


def collect_provenance(cfg) -> dict[str, object]:
    pbf_path = Path(cfg.PBF_PATH)
    stat = pbf_path.stat()
    cli_runs = None
    if CLI_RUN_MANIFEST.exists():
        cli_runs = json.loads(CLI_RUN_MANIFEST.read_text(encoding="utf-8"))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pipeline_git_commit": git_commit(PIPELINE_DIR),
        "pipeline_git_dirty": git_dirty(PIPELINE_DIR),
        "pipeline_dir": str(PIPELINE_DIR),
        "package_versions": package_versions(
            [
                "geopandas",
                "matplotlib",
                "networkx",
                "numpy",
                "osmium",
                "pandas",
                "pyproj",
                "pyrosm",
                "scipy",
                "shapely",
            ]
        ),
        "osm_pbf": {
            "url": cfg.PBF_URL,
            "path": str(pbf_path),
            "filename": cfg.resolved_pbf_filename,
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256_file(pbf_path),
            "http_head": url_head_metadata(cfg.PBF_URL),
        },
        "pipeline_cli_runs": cli_runs,
    }


def source_points_frame(record: dict[str, object]) -> pd.DataFrame:
    name = str(record.get("Name_English") or record.get("Name") or f"TT {SOURCE_TT}")
    return pd.DataFrame(
        {
            "ID": [SOURCE_ID],
            "Name": [name],
            "Longitude": [float(record["longitude"])],
            "Latitude": [float(record["latitude"])],
        }
    )


def point_gdf(record: dict[str, object]):
    import geopandas as gpd

    frame = source_points_frame(record)
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
        crs="EPSG:4326",
    )


def distance_m(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x0 - x1, y0 - y1)


def unique_undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    edge_pairs = edges[["u", "v", "length", "highway"]].copy()
    edge_pairs["a"] = edge_pairs[["u", "v"]].min(axis=1)
    edge_pairs["b"] = edge_pairs[["u", "v"]].max(axis=1)
    return edge_pairs.drop_duplicates(subset=["a", "b"])[["u", "v", "length", "highway"]].copy()


def component_edge_rows(edges: pd.DataFrame, component_ids: pd.Series, component_id: int | None) -> pd.DataFrame:
    if component_id is None:
        return edges
    labels_u = component_ids.reindex(edges["u"].astype("int64")).to_numpy()
    labels_v = component_ids.reindex(edges["v"].astype("int64")).to_numpy()
    return edges.loc[(labels_u == component_id) & (labels_v == component_id)].copy()


def plot_edges(ax: plt.Axes, edges: pd.DataFrame, nodes_xy: pd.DataFrame, *, color: str, lw: float, alpha: float, zorder: int) -> None:
    for row in edges.itertuples(index=False):
        try:
            left = nodes_xy.loc[int(row.u)]
            right = nodes_xy.loc[int(row.v)]
        except KeyError:
            continue
        ax.plot([left.x, right.x], [left.y, right.y], color=color, lw=lw, alpha=alpha, zorder=zorder)


def set_bbox(ax: plt.Axes, transformer: Transformer) -> dict[str, float]:
    min_lon, min_lat, max_lon, max_lat = BBOX
    x0, y0 = transformer.transform(min_lon, min_lat)
    x1, y1 = transformer.transform(max_lon, max_lat)
    bounds = {"xmin": min(x0, x1), "xmax": max(x0, x1), "ymin": min(y0, y1), "ymax": max(y0, y1)}
    ax.set_xlim(bounds["xmin"], bounds["xmax"])
    ax.set_ylim(bounds["ymin"], bounds["ymax"])
    return bounds


def add_scale_bar(ax: plt.Axes, bounds: dict[str, float], length_m: float = 250.0) -> None:
    x = bounds["xmin"] + 0.065 * (bounds["xmax"] - bounds["xmin"])
    y = bounds["ymin"] + 0.07 * (bounds["ymax"] - bounds["ymin"])
    ax.plot([x, x + length_m], [y, y], color="#263238", lw=2.2, solid_capstyle="butt", zorder=20)
    ax.text(x + length_m / 2, y + 16, f"{int(length_m)} m", ha="center", va="bottom", fontsize=8, color="#263238", zorder=21)


def write_local_edge_geojson(nodes: pd.DataFrame, edges: pd.DataFrame, component_ids: pd.Series, bad_component: int, main_component: int) -> Path:
    features = []
    for row in edges.itertuples(index=False):
        if int(row.u) not in nodes.index or int(row.v) not in nodes.index:
            continue
        comp_u = int(component_ids.loc[int(row.u)])
        comp_v = int(component_ids.loc[int(row.v)])
        if comp_u == bad_component and comp_v == bad_component:
            role = "bad_isolated_component"
        elif comp_u == main_component and comp_v == main_component:
            role = "main_component"
        else:
            role = "other_component"
        left = nodes.loc[int(row.u)]
        right = nodes.loc[int(row.v)]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "u": int(row.u),
                    "v": int(row.v),
                    "highway": None if pd.isna(row.highway) else str(row.highway),
                    "component_u": comp_u,
                    "component_v": comp_v,
                    "role": role,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(left.lon), float(left.lat)], [float(right.lon), float(right.lat)]],
                },
            }
        )
    path = DATA_DIR / "vietnam_tt80_pipeline_local_edges.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    return path


def make_figure() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cfg = replace(VIETNAM_CFG, base_root=RUN_ROOT)
    download_file(cfg.PBF_URL, cfg.PBF_PATH, overwrite=False, verbose=True)

    cache = CacheManager(cfg, force_recompute=False, verbose=True)
    network_cache_backend = NETWORK_BACKEND
    nodes, edges = cache.load_or_build_network_data(
        builder=lambda: load_osm_network_data(
            cfg.PBF_PATH,
            verbose=True,
            bbox=BBOX,
            backend=NETWORK_BACKEND,
            simplify=SIMPLIFY_NETWORK,
        ),
        bbox=BBOX,
        network_backend=network_cache_backend,
    )

    if "id" not in nodes.columns:
        nodes = nodes.copy()
        nodes["id"] = nodes.index.astype("int64")
    nodes = nodes.copy().set_index("id", drop=False)
    edges = unique_undirected_edges(edges)

    component_ids, component_summary = compute_network_component_labels(nodes, edges, verbose=True)
    record = source_record()
    source_points = point_gdf(record)
    source_name = str(source_points.iloc[0]["Name"])
    unfiltered = snap_points_to_nodes(
        source_points,
        nodes,
        id_col="ID",
        distance_col="dist_snap_source",
        projected_epsg=cfg.PROJECTED_EPSG,
        keep_geometry=False,
        verbose=True,
    )
    nodes_main = filter_nodes_to_components(nodes, component_ids, (0,), verbose=True)
    filtered = snap_points_to_nodes(
        source_points,
        nodes_main,
        id_col="ID",
        distance_col="dist_snap_source",
        projected_epsg=cfg.PROJECTED_EPSG,
        keep_geometry=False,
        verbose=True,
    )

    bad_node = int(unfiltered.loc[SOURCE_ID, "nearest_node"])
    bad_dist = float(unfiltered.loc[SOURCE_ID, "dist_snap_source"])
    bad_component = int(component_ids.loc[bad_node])
    good_node = int(filtered.loc[SOURCE_ID, "nearest_node"])
    good_dist = float(filtered.loc[SOURCE_ID, "dist_snap_source"])
    main_component = int(component_ids.loc[good_node])

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg.PROJECTED_EPSG}", always_xy=True)
    xy = transformer.transform(nodes["lon"].to_numpy(dtype="float64"), nodes["lat"].to_numpy(dtype="float64"))
    nodes_xy = pd.DataFrame({"x": xy[0], "y": xy[1]}, index=nodes.index)
    facility_x, facility_y = transformer.transform(
        float(source_points.iloc[0]["Longitude"]),
        float(source_points.iloc[0]["Latitude"]),
    )

    bad_xy = nodes_xy.loc[bad_node]
    good_xy = nodes_xy.loc[good_node]
    edges_main = component_edge_rows(edges, component_ids, main_component)
    edges_bad = component_edge_rows(edges, component_ids, bad_component)
    geojson_path = write_local_edge_geojson(nodes, edges, component_ids, bad_component, main_component)

    fig = plt.figure(figsize=(12.6, 8.0), dpi=220)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.48, 0.92], wspace=0.13)
    ax = fig.add_subplot(grid[0, 0])
    ax_note = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("#f6f7f3")

    plot_edges(ax, edges, nodes_xy, color="#d5d8dd", lw=1.0, alpha=0.85, zorder=1)
    plot_edges(ax, edges_main, nodes_xy, color="#2b6cb0", lw=2.0, alpha=0.94, zorder=3)
    plot_edges(ax, edges_bad, nodes_xy, color="#f28e1c", lw=3.0, alpha=0.98, zorder=5)

    main_node_ids = component_ids[component_ids == main_component].index.intersection(nodes_xy.index)
    bad_node_ids = component_ids[component_ids == bad_component].index.intersection(nodes_xy.index)
    ax.scatter(nodes_xy.loc[main_node_ids, "x"], nodes_xy.loc[main_node_ids, "y"], s=10, c="#2b6cb0", edgecolors="white", linewidths=0.25, zorder=4)
    ax.scatter(nodes_xy.loc[bad_node_ids, "x"], nodes_xy.loc[bad_node_ids, "y"], s=24, c="#101820", edgecolors="#f28e1c", linewidths=0.75, zorder=6)

    ax.plot([facility_x, bad_xy.x], [facility_y, bad_xy.y], color="#c92a2a", lw=1.5, linestyle="--", zorder=8)
    ax.plot([facility_x, good_xy.x], [facility_y, good_xy.y], color="#22863a", lw=1.5, linestyle="--", zorder=8)
    ax.scatter([facility_x], [facility_y], marker="P", s=160, c="#7c3aed", edgecolors="white", linewidths=1.1, zorder=9)
    ax.scatter([bad_xy.x], [bad_xy.y], marker="X", s=120, c="#c92a2a", edgecolors="white", linewidths=0.9, zorder=10)
    ax.scatter([good_xy.x], [good_xy.y], marker="D", s=100, c="#22863a", edgecolors="white", linewidths=0.9, zorder=10)

    ax.annotate(
        f"bad snap\ncomponent {bad_component}\n{bad_dist:.1f} m",
        xy=(bad_xy.x, bad_xy.y),
        xytext=(bad_xy.x + 125, bad_xy.y + 95),
        arrowprops={"arrowstyle": "-", "color": "#c92a2a", "lw": 1.0},
        fontsize=8.2,
        color="#7a1d1d",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#f1b0b0", "alpha": 0.94},
    )
    ax.annotate(
        f"good snap\ncomponent {main_component}\n{good_dist:.1f} m",
        xy=(good_xy.x, good_xy.y),
        xytext=(good_xy.x - 160, good_xy.y - 120),
        arrowprops={"arrowstyle": "-", "color": "#22863a", "lw": 1.0},
        fontsize=8.2,
        color="#145523",
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9bd5a8", "alpha": 0.94},
    )

    bounds = set_bbox(ax, transformer)
    add_scale_bar(ax, bounds)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pipeline-derived local road graph around TT80, Gia Lai", loc="left", fontsize=11.3, pad=10)

    legend = [
        Line2D([0], [0], color="#2b6cb0", lw=2.2, marker="o", markersize=4, label="component-aware target: main component"),
        Line2D([0], [0], color="#f28e1c", lw=3.0, marker="o", markerfacecolor="#101820", markeredgecolor="#f28e1c", markersize=5, label="bad nearest-node component"),
        Line2D([0], [0], color="#d5d8dd", lw=1.2, label="other local road edges"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor="#7c3aed", markeredgecolor="white", markersize=9, label="geocoded hospital"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#c92a2a", markeredgecolor="white", markersize=8, label="bad snap"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#22863a", markeredgecolor="white", markersize=7, label="good snap"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=True, framealpha=0.94, fontsize=7.8, borderpad=0.7)

    ax_note.axis("off")
    ax_note.set_title("Access recovered by component-aware snapping", loc="left", fontsize=11.3, pad=10)
    local_bad_size = int(component_summary.loc[component_summary["component_id"] == bad_component, "node_count"].iloc[0])
    local_main_size = int(component_summary.loc[component_summary["component_id"] == main_component, "node_count"].iloc[0])
    note_lines = [
        f"Case: TT {SOURCE_TT}, {source_name}.",
        f"The unfiltered snap uses the nearest pipeline road node, but that node is on local component {bad_component} ({local_bad_size} nodes in the plotted bbox).",
        f"Restricting snapping to component 0 moves the source to the main local component ({local_main_size} nodes in the plotted bbox).",
        f"The corrected Vietnam run assigns {SOURCE_ID} to node {DIAGNOSTIC_FACTS['corrected_snap_node']} on component {DIAGNOSTIC_FACTS['corrected_snap_component']} at {DIAGNOSTIC_FACTS['corrected_snap_distance_m']:.1f} m.",
        f"Access recovered: {DIAGNOSTIC_FACTS['retained_destination_rows_under_150km']:,} population-grid destinations are retained under the 150 km cap, instead of missing or infinite distances from the isolated component.",
        f"Original full-network diagnostic: {DIAGNOSTIC_FACTS['weak_components_original_diagnostic']:,} weak components; the largest has {DIAGNOSTIC_FACTS['giant_component_nodes_original_diagnostic']:,} nodes ({DIAGNOSTIC_FACTS['giant_component_share_percent_original_diagnostic']:.2f}%).",
    ]
    y = 0.9
    for idx, line in enumerate(note_lines):
        wrapped = textwrap.fill(line, width=57)
        is_gain = idx == 4
        ax_note.text(
            0.0,
            y,
            wrapped,
            transform=ax_note.transAxes,
            fontsize=9.0 if not is_gain else 9.3,
            weight="bold" if idx in {0, 4} else "normal",
            color="#145523" if is_gain else "#263238",
            va="top",
            linespacing=1.18,
        )
        y -= 0.040 * (wrapped.count("\n") + 1) + 0.030
    ax_note.text(
        0.0,
        0.01,
        "Fresh OSM source: Geofabrik Vietnam PBF. Snaps are verified by pipeline CLI runs; exact commands, SHA256, and package versions are in the manifest.",
        transform=ax_note.transAxes,
        fontsize=7.8,
        color="#5b6770",
        va="bottom",
        wrap=True,
    )

    png = FIG_DIR / "vietnam_tt80_pipeline_component_inset.png"
    pdf = FIG_DIR / "vietnam_tt80_pipeline_component_inset.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    markdown = OUT_DIR / "vietnam_tt80_pipeline_component_inset.md"
    latex = OUT_DIR / "vietnam_tt80_pipeline_component_inset.tex"
    markdown.write_text(
        f"""# TT80 Component-Aware Snap Diagnostic

This inset is generated from the GitHub distance pipeline, using the Vietnam
Geofabrik PBF and the pipeline's own node/edge loader, nearest-node snapping,
and weak-component labeling. It shows the Gia Lai hospital case where a close
unfiltered snap lies on a disconnected local component, while component-aware
snapping moves the source onto the main road component. The corrected run
retains {DIAGNOSTIC_FACTS['retained_destination_rows_under_150km']:,}
population-grid destinations under the 150 km cap.

A pair of pipeline CLI runs (`python -m run_pipeline vietnam_tt80_cli`) verifies
the local contrast: unrestricted snapping assigns the table source to the
15-node component, while `--snap-components 0` assigns it to component 0. The
exact commands are recorded in
`{CLI_RUN_MANIFEST.as_posix()}`.

![TT80 component-aware snap diagnostic]({png.as_posix()})
""",
        encoding="utf-8",
    )
    latex.write_text(
        rf"""\begin{{figure}}[tbp]
    \centering
    \includegraphics[width=\linewidth]{{{pdf.as_posix()}}}
    \caption{{Pipeline-derived local component diagnostic for TT {SOURCE_TT}
    in Gia Lai. The bad nearest-node snap lands on a disconnected local road
    component, while component-aware snapping moves the source to the main road
    component. The corrected run retains
    {DIAGNOSTIC_FACTS['retained_destination_rows_under_150km']:,}
    population-grid destinations under the 150 km cap. The unrestricted and
    component-restricted snaps are verified by pipeline CLI runs.}}
    \label{{fig:vietnam-tt80-pipeline-component-inset}}
\end{{figure}}
""",
        encoding="utf-8",
    )

    provenance = collect_provenance(cfg)
    provenance_path = OUT_DIR / "vietnam_tt80_pipeline_component_inset_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    manifest = {
        "source_tt": SOURCE_TT,
        "source_id": SOURCE_ID,
        "source_name": source_name,
        "bbox_lonlat": {
            "min_lon": BBOX[0],
            "min_lat": BBOX[1],
            "max_lon": BBOX[2],
            "max_lat": BBOX[3],
        },
        "pipeline_dir": str(PIPELINE_DIR),
        "run_root": str(RUN_ROOT),
        "pbf_path": str(cfg.PBF_PATH),
        "network_backend": NETWORK_BACKEND,
        "nodes_in_bbox": int(len(nodes)),
        "edges_in_bbox_undirected": int(len(edges)),
        "component_summary_top10": component_summary.head(10).to_dict(orient="records"),
        "bad_snap": {
            "node": bad_node,
            "component_id_local": bad_component,
            "component_nodes_local": local_bad_size,
            "distance_m": bad_dist,
        },
        "good_snap": {
            "node": good_node,
            "component_id_local": main_component,
            "component_nodes_local": local_main_size,
            "distance_m": good_dist,
        },
        "diagnostic_facts": DIAGNOSTIC_FACTS,
        "provenance": provenance,
        "outputs": {
            "png": str(png),
            "pdf": str(pdf),
            "markdown": str(markdown),
            "latex": str(latex),
            "local_edges_geojson": str(geojson_path),
            "provenance": str(provenance_path),
        },
    }
    manifest_path = OUT_DIR / "vietnam_tt80_pipeline_component_inset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(make_figure(), indent=2))


if __name__ == "__main__":
    main()
