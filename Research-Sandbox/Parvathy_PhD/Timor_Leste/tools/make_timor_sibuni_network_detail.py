from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pyproj import Transformer


OUTPUTS_DIR = Path("runs/timor_network_profile_20260623/east-timor_data/outputs")
CACHE_DIR = Path("runs/timor_network_profile_20260623/east-timor_data/cache")
ARTICLE_DIR = Path("articles/seps_access_optimization")
FIGURE_DIR = ARTICLE_DIR / "figures"

SIBUNI_ID = "source_amenities_1777615156"
WEB_MERCATOR = 3857
DETAIL_RADIUS_M = 100.0

VARIANTS = [
    {
        "case_id": "timor_drive_only_simplified_1km",
        "manifest_profile": "driving",
        "simplify_network": True,
        "cache_suffix": "backend_osmium_simplified_v2",
        "panel": "A",
        "panel_label": "Drive-only / simplified",
    },
    {
        "case_id": "timor_drive_only_unsimplified_1km",
        "manifest_profile": "driving",
        "simplify_network": False,
        "cache_suffix": "backend_osmium",
        "panel": "B",
        "panel_label": "Drive-only / unsimplified",
    },
    {
        "case_id": "timor_drive_plus_walk_simplified_1km",
        "manifest_profile": "driving_walk",
        "simplify_network": True,
        "cache_suffix": "backend_osmium_driving_walk_simplified",
        "panel": "C",
        "panel_label": "Drive + walk / simplified",
    },
    {
        "case_id": "timor_drive_plus_walk_unsimplified_1km",
        "manifest_profile": "driving_walk",
        "simplify_network": False,
        "cache_suffix": "backend_osmium_driving_walk",
        "panel": "D",
        "panel_label": "Drive + walk / unsimplified",
    },
]

STYLE = {
    "basemap_alpha": 0.72,
    "other_edge": "#b7b7b7",
    "focus_road_edge": "#2166ac",
    "focus_walk_edge": "#009e73",
    "other_node": "#6f6f6f",
    "focus_node": "#111111",
    "facility": "#d7191c",
    "snap_node": "#e69f00",
    "snap_line": "#d7191c",
}

WALK_HIGHWAYS = {
    "bridleway",
    "corridor",
    "cycleway",
    "footway",
    "path",
    "pedestrian",
    "steps",
    "track",
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_manifest(profile: str, simplify: bool) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in OUTPUTS_DIR.glob("run_manifest*.yaml"):
        manifest = read_yaml(path)
        runtime = manifest["parameters"]["runtime_settings"]
        resolved = manifest["parameters"]["resolved"]
        if (
            runtime["network_profile"] == profile
            and bool(runtime["simplify_network"]) == bool(simplify)
            and int(float(resolved["candidate_grid_spacing_m"])) == 1000
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(f"expected one manifest for {profile=} {simplify=}, found {len(matches)}")
    return matches[0]


def load_facility(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["outputs"]["existing_sources"]["path"])
    frame = pd.read_parquet(
        path,
        columns=["ID", "Longitude", "Latitude", "name", "amenity", "nearest_node", "dist_snap_source", "component_id"],
    )
    match = frame.loc[frame["ID"].astype(str).eq(SIBUNI_ID)]
    if len(match) != 1:
        raise ValueError(f"expected one Sibuni row in {path}, found {len(match)}")
    return match.iloc[0].to_dict()


def transform_lonlat(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(4326, WEB_MERCATOR, always_xy=True)
    return transformer.transform(lon, lat)


def component_labels(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    node_ids = nodes["id"].astype("int64").to_numpy()
    positions = pd.Series(np.arange(len(node_ids), dtype=np.int64), index=node_ids)
    u = positions.reindex(edges["u"].astype("int64")).to_numpy()
    v = positions.reindex(edges["v"].astype("int64")).to_numpy()
    valid = np.isfinite(u) & np.isfinite(v)
    u = u[valid].astype(np.int64, copy=False)
    v = v[valid].astype(np.int64, copy=False)
    parent = np.arange(len(node_ids), dtype=np.int64)
    size = np.ones(len(node_ids), dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != x:
            nxt = int(parent[x])
            parent[x] = root
            x = nxt
        return root

    for a, b in zip(u, v):
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    roots = np.fromiter((find(i) for i in range(len(node_ids))), dtype=np.int64, count=len(node_ids))
    _, labels = np.unique(roots, return_inverse=True)
    component_sizes = np.bincount(labels)
    return pd.Series(labels, index=node_ids), component_sizes


def highway_is_walk(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).replace("[", "").replace("]", "").replace("'", "").split(",")
    normalized = {str(item).strip().lower() for item in values if str(item).strip()}
    return bool(normalized & WALK_HIGHWAYS)


def edge_segments(
    edges: pd.DataFrame,
    node_xy: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    component_by_node: pd.Series,
    focus_component: int,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list, list, list]:
    xmin, xmax, ymin, ymax = bbox
    coords = node_xy[["x", "y"]]
    u_xy = coords.reindex(edges["u"].astype("int64"))
    v_xy = coords.reindex(edges["v"].astype("int64"))
    valid = (
        u_xy["x"].notna().to_numpy()
        & v_xy["x"].notna().to_numpy()
        & (
            (
                u_xy["x"].between(xmin, xmax).to_numpy()
                & u_xy["y"].between(ymin, ymax).to_numpy()
            )
            | (
                v_xy["x"].between(xmin, xmax).to_numpy()
                & v_xy["y"].between(ymin, ymax).to_numpy()
            )
        )
    )
    edges_v = edges.loc[valid].reset_index(drop=True)
    u_xy_v = u_xy.loc[valid].reset_index(drop=True)
    v_xy_v = v_xy.loc[valid].reset_index(drop=True)
    u_comp = component_by_node.reindex(edges_v["u"].astype("int64")).to_numpy()
    v_comp = component_by_node.reindex(edges_v["v"].astype("int64")).to_numpy()

    other_segments = []
    focus_road_segments = []
    focus_walk_segments = []
    for idx, row in edges_v.iterrows():
        segment = (
            (float(u_xy_v.at[idx, "x"]), float(u_xy_v.at[idx, "y"])),
            (float(v_xy_v.at[idx, "x"]), float(v_xy_v.at[idx, "y"])),
        )
        if int(u_comp[idx]) == focus_component and int(v_comp[idx]) == focus_component:
            if highway_is_walk(row.get("highway")):
                focus_walk_segments.append(segment)
            else:
                focus_road_segments.append(segment)
        else:
            other_segments.append(segment)
    return other_segments, focus_road_segments, focus_walk_segments, edges_v


def fetch_basemap(
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = 18,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    xmin, xmax, ymin, ymax = bbox
    try:
        sources = []
        try:
            sources.append(cx.providers.CartoDB.PositronNoLabels)
        except AttributeError:
            pass
        sources.append(cx.providers.OpenStreetMap.Mapnik)
        for source in sources:
            try:
                image, extent = cx.bounds2img(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    zoom=zoom,
                    source=source,
                    ll=False,
                    wait=1,
                    max_retries=4,
                )
                return image, extent
            except Exception:
                continue
        raise RuntimeError("no basemap source returned tiles")
    except Exception as exc:
        print(f"Basemap unavailable: {exc.__class__.__name__}: {exc}", flush=True)
        return None


def draw_panel(
    ax: plt.Axes,
    variant: dict[str, Any],
    bbox: tuple[float, float, float, float],
    basemap: tuple[np.ndarray, tuple[float, float, float, float]] | None,
) -> dict[str, Any]:
    _, manifest = find_manifest(variant["manifest_profile"], bool(variant["simplify_network"]))
    facility = load_facility(manifest)
    nodes = pd.read_pickle(CACHE_DIR / f"east-timor-latest.osm_nodes_{variant['cache_suffix']}.pkl")
    edges = pd.read_pickle(CACHE_DIR / f"east-timor-latest.osm_edges_{variant['cache_suffix']}.pkl")
    nodes = nodes.copy()
    x, y = transform_lonlat(nodes["lon"].to_numpy(dtype=float), nodes["lat"].to_numpy(dtype=float))
    nodes["x"] = x
    nodes["y"] = y
    node_xy = nodes.set_index(nodes["id"].astype("int64"))[["x", "y"]]

    component_by_node, component_sizes = component_labels(nodes, edges)
    snapped_node = int(facility["nearest_node"])
    focus_component = int(component_by_node.loc[snapped_node])
    focus_component_nodes = int(component_sizes[focus_component])
    largest_component_nodes = int(component_sizes.max())
    other_segments, focus_road_segments, focus_walk_segments, visible_edges = edge_segments(
        edges,
        node_xy,
        bbox,
        component_by_node,
        focus_component,
    )

    xmin, xmax, ymin, ymax = bbox
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if basemap is not None:
        image, extent = basemap
        ax.imshow(
            image,
            extent=extent,
            interpolation="bilinear",
            alpha=STYLE["basemap_alpha"],
            zorder=1,
        )
    else:  # The figure remains useful if tile access is unavailable.
        ax.text(
            0.02,
            0.02,
            "Basemap unavailable",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
            zorder=2,
        )

    if other_segments:
        ax.add_collection(
            LineCollection(other_segments, colors=STYLE["other_edge"], linewidths=0.45, alpha=0.82, zorder=3)
        )
    if focus_road_segments:
        ax.add_collection(
            LineCollection(focus_road_segments, colors=STYLE["focus_road_edge"], linewidths=1.25, alpha=0.95, zorder=4)
        )
    if focus_walk_segments:
        ax.add_collection(
            LineCollection(focus_walk_segments, colors=STYLE["focus_walk_edge"], linewidths=1.2, alpha=0.95, zorder=5)
        )

    visible_nodes = nodes.loc[nodes["x"].between(xmin, xmax) & nodes["y"].between(ymin, ymax)].copy()
    visible_node_components = component_by_node.reindex(visible_nodes["id"].astype("int64")).to_numpy()
    focus_nodes = visible_nodes.loc[visible_node_components == focus_component]
    other_nodes = visible_nodes.loc[visible_node_components != focus_component]
    if len(other_nodes):
        ax.scatter(
            other_nodes["x"],
            other_nodes["y"],
            s=2.0,
            color=STYLE["other_node"],
            alpha=0.30,
            linewidths=0,
            zorder=6,
        )
    if len(focus_nodes):
        ax.scatter(
            focus_nodes["x"],
            focus_nodes["y"],
            s=7.0,
            color=STYLE["focus_node"],
            alpha=0.90,
            linewidths=0,
            zorder=7,
        )

    facility_x, facility_y = transform_lonlat(
        np.asarray([float(facility["Longitude"])]),
        np.asarray([float(facility["Latitude"])]),
    )
    snapped_x = float(node_xy.loc[snapped_node, "x"])
    snapped_y = float(node_xy.loc[snapped_node, "y"])
    ax.plot(
        [float(facility_x[0]), snapped_x],
        [float(facility_y[0]), snapped_y],
        color=STYLE["snap_line"],
        linestyle=(0, (4, 3)),
        linewidth=1.25,
        alpha=0.90,
        zorder=8,
    )
    ax.scatter(
        [snapped_x],
        [snapped_y],
        s=58,
        facecolors="none",
        edgecolors=STYLE["snap_node"],
        linewidths=1.45,
        zorder=9,
    )
    ax.scatter(
        facility_x,
        facility_y,
        marker="+",
        s=98,
        color=STYLE["facility"],
        linewidths=1.9,
        zorder=10,
    )
    ax.text(
        0.02,
        0.98,
        variant["panel"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        zorder=11,
    )
    ax.text(
        0.02,
        0.08,
        f"snap {float(facility['dist_snap_source']):.1f} m\ncomponent {focus_component_nodes:,} nodes",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        zorder=11,
    )
    ax.set_axis_off()
    return {
        "case_id": variant["case_id"],
        "panel": variant["panel"],
        "panel_label": variant["panel_label"],
        "facility_id": SIBUNI_ID,
        "facility_name": str(facility["name"]),
        "facility_lon": float(facility["Longitude"]),
        "facility_lat": float(facility["Latitude"]),
        "nearest_node": snapped_node,
        "snap_distance_m": float(facility["dist_snap_source"]),
        "reported_component_id": int(facility["component_id"]),
        "computed_focus_component": focus_component,
        "computed_focus_component_nodes": focus_component_nodes,
        "largest_component_nodes": largest_component_nodes,
        "visible_edges": int(len(visible_edges)),
        "visible_nodes": int(len(visible_nodes)),
        "visible_focus_nodes": int(len(focus_nodes)),
        "visible_focus_road_segments": int(len(focus_road_segments)),
        "visible_focus_walk_segments": int(len(focus_walk_segments)),
    }


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=STYLE["other_edge"], lw=1.4, label="Other local edges"),
        Line2D([0], [0], color=STYLE["focus_road_edge"], lw=2.0, label="Sibuni component roads"),
        Line2D([0], [0], color=STYLE["focus_walk_edge"], lw=2.0, label="Sibuni component walking/trail edges"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STYLE["focus_node"], markersize=4, label="Component nodes"),
        Line2D(
            [0],
            [0],
            marker="P",
            color="none",
            markerfacecolor=STYLE["facility"],
            markeredgecolor="white",
            markersize=8,
            label="Centro de Saude Sibuni",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=STYLE["snap_line"],
            markerfacecolor=STYLE["snap_node"],
            markeredgecolor="#111111",
            linestyle=(0, (4, 3)),
            markersize=6,
            label="Snapped network node",
        ),
    ]


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # Use Sibuni coordinates from the first manifest to define one common map extent.
    _, first_manifest = find_manifest("driving_walk", True)
    facility = load_facility(first_manifest)
    fx, fy = transform_lonlat(
        np.asarray([float(facility["Longitude"])]),
        np.asarray([float(facility["Latitude"])]),
    )
    bbox = (
        float(fx[0] - DETAIL_RADIUS_M),
        float(fx[0] + DETAIL_RADIUS_M),
        float(fy[0] - DETAIL_RADIUS_M),
        float(fy[0] + DETAIL_RADIUS_M),
    )
    basemap = fetch_basemap(bbox)

    metadata: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2), dpi=220, constrained_layout=False)
    for ax, variant in zip(axes.flat, VARIANTS):
        metadata.append(draw_panel(ax, variant, bbox, basemap))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02, wspace=0.03, hspace=0.03)
    combined_pdf = FIGURE_DIR / "fig_timor_sibuni_network_profiles_2x2.pdf"
    combined_png = FIGURE_DIR / "fig_timor_sibuni_network_profiles_2x2.png"
    fig.savefig(combined_pdf, bbox_inches="tight")
    fig.savefig(combined_png, bbox_inches="tight")
    plt.close(fig)

    for variant in VARIANTS:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6.1, 5.6), dpi=220)
        single_meta = draw_panel(ax_single, variant, bbox, basemap)
        fig_single.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.02)
        stem = variant["case_id"].replace("timor_", "fig_timor_sibuni_")
        fig_single.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig_single.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
        plt.close(fig_single)
        # Preserve the metadata from the combined pass but sanity-check individual rendering.
        assert single_meta["case_id"] == variant["case_id"]

    metadata_path = FIGURE_DIR / "fig_timor_sibuni_network_profiles_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bbox_web_mercator": {
                    "xmin": bbox[0],
                    "xmax": bbox[1],
                    "ymin": bbox[2],
                    "ymax": bbox[3],
                },
                "detail_radius_m": DETAIL_RADIUS_M,
                "source": "general_distances_per_country pipeline cache and outputs",
                "variants": metadata,
                "figures": {
                    "combined_pdf": str(combined_pdf),
                    "combined_png": str(combined_png),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"combined_pdf": str(combined_pdf), "combined_png": str(combined_png)}, indent=2))


if __name__ == "__main__":
    main()
