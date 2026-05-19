"""Build Norway inputs for the ferry-to-road connection notebook.

The original Netherlands notebook uses Pyrosm to materialize the full national
OSM network. For Norway this proved too memory intensive. This script instead
streams the Norway Geofabrik extract with osmium and keeps only the compact
drivable road graph plus route=ferry geometries needed by connect_norway.ipynb.

Outputs are written to the local experiments data folder, which is intentionally
ignored by Git:

- data/norway-latest.osm.pbf
- data/no_nodes_and_edges_driving.pkl
- data/ferries_no.pkl
- data/no_connect_inputs_metadata.json
- data/no_drivable_components_summary.json
- C:/local/temp/HandsOnBook/generated-figures/Norway/norway_drivable_components.png
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmium
import pandas as pd
import pyrosm
from matplotlib.patches import Patch
from pyproj import Geod
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_FIGURE_DIR = Path(r"C:\local\temp\HandsOnBook\generated-figures\Norway")
PROJECTED_CRS = "EPSG:25833"

DRIVABLE_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "road",
    "track",
}
EXCLUDED_ACCESS = {"private", "no"}


class NorwayConnectExtractor(osmium.SimpleHandler):
    """Stream drivable road ways and ferry ways from the Norway OSM extract."""

    def __init__(self) -> None:
        super().__init__()
        self.node_coords: dict[int, tuple[float, float]] = {}
        self.edge_rows: list[dict[str, int | float | str]] = []
        self.ferry_rows: list[dict[str, object]] = []
        self.highway_ways_seen = 0
        self.highway_ways_kept = 0
        self.ferry_ways_seen = 0
        self.ferry_ways_kept = 0
        self.started = time.time()
        self.geod = Geod(ellps="WGS84")

    def way(self, way: osmium.osm.Way) -> None:
        tags = way.tags
        highway = tags.get("highway")
        route = tags.get("route")

        coords: list[tuple[float, float]] = []
        refs: list[int] = []
        for node in way.nodes:
            try:
                location = node.location
                if location.valid():
                    coords.append((float(location.lon), float(location.lat)))
                    refs.append(int(node.ref))
            except Exception:
                continue

        if route == "ferry":
            self._keep_ferry_way(way, coords)

        if highway:
            self._keep_drivable_way(way, highway, coords, refs)

    def _keep_ferry_way(
        self,
        way: osmium.osm.Way,
        coords: list[tuple[float, float]],
    ) -> None:
        self.ferry_ways_seen += 1
        if len(coords) < 2:
            return

        tags = way.tags
        props = {
            key: tags.get(key)
            for key in [
                "name",
                "from",
                "to",
                "ref",
                "duration",
                "ferry",
                "motor_vehicle",
                "foot",
                "bicycle",
            ]
        }
        self.ferry_rows.append(
            {
                "id": int(way.id),
                **props,
                "geometry": LineString(coords),
            }
        )
        self.ferry_ways_kept += 1

    def _keep_drivable_way(
        self,
        way: osmium.osm.Way,
        highway: str,
        coords: list[tuple[float, float]],
        refs: list[int],
    ) -> None:
        self.highway_ways_seen += 1
        tags = way.tags

        if highway not in DRIVABLE_HIGHWAYS:
            return
        if tags.get("area") == "yes":
            return
        if (
            tags.get("access") in EXCLUDED_ACCESS
            and tags.get("motor_vehicle") not in {"yes", "designated", "permissive"}
        ):
            return
        if len(coords) < 2:
            return

        self.highway_ways_kept += 1
        for ref, coord in zip(refs, coords):
            self.node_coords.setdefault(ref, coord)

        oneway = tags.get("oneway") in {"yes", "true", "1"} or tags.get("junction") == "roundabout"
        reverse_oneway = tags.get("oneway") == "-1"

        for u, v, start, end in zip(refs[:-1], refs[1:], coords[:-1], coords[1:]):
            if u == v:
                continue
            _, _, distance_m = self.geod.inv(start[0], start[1], end[0], end[1])
            if not math.isfinite(distance_m) or distance_m <= 0:
                continue
            if reverse_oneway:
                u, v = v, u
            self.edge_rows.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "length": float(distance_m),
                    "highway": highway,
                    "osm_way_id": int(way.id),
                }
            )
            if not oneway and not reverse_oneway:
                self.edge_rows.append(
                    {
                        "u": int(v),
                        "v": int(u),
                        "length": float(distance_m),
                        "highway": highway,
                        "osm_way_id": int(way.id),
                    }
                )


def acquire_norway_pbf(data_dir: Path) -> Path:
    """Download or reuse the Norway Geofabrik extract."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pbf = data_dir / "norway-latest.osm.pbf"
    if pbf.exists():
        return pbf
    return Path(pyrosm.get_data("Norway", directory=data_dir))


def build_connect_inputs(data_dir: Path, force: bool = False) -> dict[str, object]:
    """Create compact Norway road and ferry caches for connect_norway.ipynb."""
    started = time.time()
    pbf = acquire_norway_pbf(data_dir)
    road_out = data_dir / "no_nodes_and_edges_driving.pkl"
    ferry_out = data_dir / "ferries_no.pkl"
    metadata_out = data_dir / "no_connect_inputs_metadata.json"

    if force or not road_out.exists() or not ferry_out.exists():
        extractor = NorwayConnectExtractor()
        extractor.apply_file(str(pbf), locations=True)

        nodes = gpd.GeoDataFrame(
            [
                {"id": node_id, "geometry": Point(lon, lat)}
                for node_id, (lon, lat) in extractor.node_coords.items()
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        edges = pd.DataFrame(extractor.edge_rows)
        ferries = gpd.GeoDataFrame(extractor.ferry_rows, geometry="geometry", crs="EPSG:4326")

        with road_out.open("wb") as handle:
            pickle.dump((nodes, edges), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with ferry_out.open("wb") as handle:
            pickle.dump(ferries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        extractor = None
        with road_out.open("rb") as handle:
            nodes, edges = pickle.load(handle)
        with ferry_out.open("rb") as handle:
            ferries = pickle.load(handle)

    metadata = {
        "country": "Norway",
        "pbf": str(pbf),
        "method": "osmium_streaming_compact_drivable_roads_and_ferry_ways",
        "projected_crs_for_connect_notebook": PROJECTED_CRS,
        "drivable_highways": sorted(DRIVABLE_HIGHWAYS),
        "road_cache": {
            "path": str(road_out),
            "nodes": int(len(nodes)),
            "edges": int(len(edges)),
            "bytes": road_out.stat().st_size,
        },
        "ferry_cache": {
            "path": str(ferry_out),
            "features": int(len(ferries)),
            "bytes": ferry_out.stat().st_size,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if extractor is not None:
        metadata.update(
            {
                "highway_ways_seen": extractor.highway_ways_seen,
                "highway_ways_kept": extractor.highway_ways_kept,
                "ferry_ways_seen": extractor.ferry_ways_seen,
                "ferry_ways_kept": extractor.ferry_ways_kept,
            }
        )

    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def compute_drivable_components(data_dir: Path, figure_dir: Path) -> dict[str, object]:
    """Count and visualize drivable road components before ferry repair."""
    started = time.time()
    road_cache = data_dir / "no_nodes_and_edges_driving.pkl"
    summary_path = data_dir / "no_drivable_components_summary.json"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "norway_drivable_components.png"

    with road_cache.open("rb") as handle:
        nodes, edges = pickle.load(handle)

    n_nodes = len(nodes)
    node_ids = nodes["id"].to_numpy()
    id_to_idx = pd.Series(np.arange(n_nodes, dtype=np.int32), index=node_ids)
    u_idx = id_to_idx.reindex(edges["u"].to_numpy()).to_numpy(dtype=np.float64)
    v_idx = id_to_idx.reindex(edges["v"].to_numpy()).to_numpy(dtype=np.float64)
    valid = np.isfinite(u_idx) & np.isfinite(v_idx)
    missing_edges = int((~valid).sum())

    graph = sparse.coo_matrix(
        (
            np.ones(int(valid.sum()), dtype=np.uint8),
            (u_idx[valid].astype(np.int32), v_idx[valid].astype(np.int32)),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False, return_labels=True)

    component_ids, component_sizes = np.unique(labels, return_counts=True)
    order = np.argsort(component_sizes)[::-1]
    sorted_ids = component_ids[order]
    sorted_sizes = component_sizes[order]
    largest_component = int(sorted_ids[0])
    largest_size = int(sorted_sizes[0])

    _write_component_figure(
        nodes=nodes,
        labels=labels,
        sorted_ids=sorted_ids,
        sorted_sizes=sorted_sizes,
        largest_component=largest_component,
        component_count=int(component_count),
        figure_path=figure_path,
    )

    summary = {
        "country": "Norway",
        "network": "compact drivable road graph before ferry repair",
        "source_cache": str(road_cache),
        "nodes": int(n_nodes),
        "directed_edges": int(len(edges)),
        "valid_edges_used": int(valid.sum()),
        "missing_endpoint_edges": missing_edges,
        "connected_components": int(component_count),
        "largest_component_label": largest_component,
        "largest_component_nodes": largest_size,
        "largest_component_share": largest_size / n_nodes,
        "top_components": [
            {"component": int(cid), "nodes": int(size), "share": float(size / n_nodes)}
            for cid, size in zip(sorted_ids[:25], sorted_sizes[:25])
        ],
        "figure": str(figure_path),
        "elapsed_seconds_total": round(time.time() - started, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_component_figure(
    nodes: gpd.GeoDataFrame,
    labels: np.ndarray,
    sorted_ids: np.ndarray,
    sorted_sizes: np.ndarray,
    largest_component: int,
    component_count: int,
    figure_path: Path,
) -> None:
    """Rasterize component labels to a compact PNG for fast visual inspection."""
    lon = nodes.geometry.x.to_numpy(dtype=np.float64)
    lat = nodes.geometry.y.to_numpy(dtype=np.float64)
    finite = np.isfinite(lon) & np.isfinite(lat)
    lon = lon[finite]
    lat = lat[finite]
    plot_labels = labels[finite]

    xmin, xmax = float(np.nanpercentile(lon, 0.05)), float(np.nanpercentile(lon, 99.95))
    ymin, ymax = float(np.nanpercentile(lat, 0.05)), float(np.nanpercentile(lat, 99.95))
    xpad = (xmax - xmin) * 0.03
    ypad = (ymax - ymin) * 0.03
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    inside = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    lon = lon[inside]
    lat = lat[inside]
    plot_labels = plot_labels[inside]

    width = 1600
    height = 2400
    xi = ((lon - xmin) / (xmax - xmin) * (width - 1)).astype(np.int32)
    yi = ((ymax - lat) / (ymax - ymin) * (height - 1)).astype(np.int32)
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    largest_mask = plot_labels == largest_component
    img[yi[largest_mask], xi[largest_mask]] = np.array([205, 205, 205], dtype=np.uint8)
    img[yi[~largest_mask], xi[~largest_mask]] = np.array([215, 72, 58], dtype=np.uint8)

    palette = np.array(
        [
            [27, 158, 119],
            [217, 95, 2],
            [117, 112, 179],
            [231, 41, 138],
            [102, 166, 30],
            [230, 171, 2],
            [166, 118, 29],
            [102, 102, 102],
            [31, 120, 180],
            [51, 160, 44],
            [227, 26, 28],
            [255, 127, 0],
        ],
        dtype=np.uint8,
    )
    for rank, component_id in enumerate(sorted_ids[1:13], start=1):
        mask = plot_labels == component_id
        if mask.any():
            img[yi[mask], xi[mask]] = palette[(rank - 1) % len(palette)]

    fig, ax = plt.subplots(figsize=(8, 12), dpi=220)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(
        "Norway drivable road connected components before ferries\n"
        f"{component_count:,} components; largest contains {int(sorted_sizes[0]) / len(nodes):.2%} of nodes",
        fontsize=10,
    )
    legend_items = [
        Patch(color=np.array([205, 205, 205]) / 255, label="Largest drivable component"),
        Patch(color=np.array([215, 72, 58]) / 255, label="Other smaller components"),
    ]
    for rank, _component_id in enumerate(sorted_ids[1:5], start=1):
        legend_items.append(
            Patch(
                color=palette[(rank - 1) % len(palette)] / 255,
                label=f"Component {rank + 1}: {int(sorted_sizes[rank]):,} nodes",
            )
        )
    ax.legend(handles=legend_items, loc="lower left", fontsize=7, frameon=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild cached road and ferry files.")
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="Only acquire inputs; do not compute drivable connected components.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_connect_inputs(args.data_dir, force=args.force)
    print(json.dumps(metadata, indent=2))
    if not args.skip_components:
        summary = compute_drivable_components(args.data_dir, args.figure_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
