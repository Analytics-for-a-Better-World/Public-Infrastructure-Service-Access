from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


NETWORKS = {
    "driving_unsimplified": (
        "east-timor-latest.osm_nodes_backend_osmium.pkl",
        "east-timor-latest.osm_edges_backend_osmium.pkl",
    ),
    "driving_simplified": (
        "east-timor-latest.osm_nodes_backend_osmium_simplified_v2.pkl",
        "east-timor-latest.osm_edges_backend_osmium_simplified_v2.pkl",
    ),
    "driving_walk_unsimplified": (
        "east-timor-latest.osm_nodes_backend_osmium_driving_walk.pkl",
        "east-timor-latest.osm_edges_backend_osmium_driving_walk.pkl",
    ),
    "driving_walk_simplified": (
        "east-timor-latest.osm_nodes_backend_osmium_driving_walk_simplified.pkl",
        "east-timor-latest.osm_edges_backend_osmium_driving_walk_simplified.pkl",
    ),
}


def label_timor_component(row: pd.Series) -> str:
    lon_min = float(row["lon_min"])
    lon_max = float(row["lon_max"])
    lat_min = float(row["lat_min"])
    lat_max = float(row["lat_max"])
    lon_mean = float(row["lon_mean"])
    lat_mean = float(row["lat_mean"])
    count = int(row["node_count"])

    if lon_min < 124.75 and lon_max < 124.85:
        return "Oecusse-Ambeno exclave"
    if 125.45 <= lon_mean <= 125.75 and -8.45 <= lat_mean <= -8.05 and lat_max > -8.45:
        return "Atauro island"
    if lon_min > 127.20 and lat_mean > -8.70 and count >= 100:
        return "Jaco / far-east island component"
    if lon_max > 124.75 and lon_min < 127.35 and lat_min < -8.2 and count >= 1000:
        return "Timor-Leste mainland"
    return "minor detached component"


def component_table(nodes: pd.DataFrame, edges: pd.DataFrame, network_name: str) -> pd.DataFrame:
    ids = nodes["id"].to_numpy(dtype=np.int64)
    node_positions = pd.Index(ids)
    u_pos = node_positions.get_indexer(edges["u"].to_numpy(dtype=np.int64))
    v_pos = node_positions.get_indexer(edges["v"].to_numpy(dtype=np.int64))
    valid = (u_pos >= 0) & (v_pos >= 0)
    u_pos = u_pos[valid]
    v_pos = v_pos[valid]

    row = np.concatenate([u_pos, v_pos])
    col = np.concatenate([v_pos, u_pos])
    data = np.ones(row.shape[0], dtype=np.uint8)
    graph = coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes))).tocsr()
    _, raw_labels = connected_components(graph, directed=False, return_labels=True)

    counts = np.bincount(raw_labels)
    order = np.argsort(-counts)
    label_to_component = np.empty_like(order)
    label_to_component[order] = np.arange(order.shape[0])
    component_id = label_to_component[raw_labels]

    working = pd.DataFrame(
        {
            "component_id": component_id,
            "lon": nodes["lon"].to_numpy(dtype=float),
            "lat": nodes["lat"].to_numpy(dtype=float),
        }
    )
    summary = (
        working.groupby("component_id", sort=True)
        .agg(
            node_count=("lon", "size"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
            lon_mean=("lon", "mean"),
            lat_mean=("lat", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "network", network_name)
    summary["geography_label"] = summary.apply(label_timor_component, axis=1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/timor_network_profile_20260623/east-timor_data/cache"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    for network_name, (nodes_file, edges_file) in NETWORKS.items():
        nodes = pd.read_pickle(args.cache_dir / nodes_file)
        edges = pd.read_pickle(args.cache_dir / edges_file)
        table = component_table(nodes, edges, network_name)
        tables.append(table)
        table.head(args.top).to_csv(
            args.output_dir / f"{network_name}_top_components.csv",
            index=False,
        )

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(args.output_dir / "timor_component_geography.csv", index=False)
    focus = combined[
        combined["geography_label"].isin(
            [
                "Timor-Leste mainland",
                "Oecusse-Ambeno exclave",
                "Atauro island",
                "Jaco / far-east island component",
            ]
        )
    ].copy()
    focus.to_csv(args.output_dir / "timor_component_geography_focus.csv", index=False)
    (args.output_dir / "timor_component_geography_focus.json").write_text(
        json.dumps(focus.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
