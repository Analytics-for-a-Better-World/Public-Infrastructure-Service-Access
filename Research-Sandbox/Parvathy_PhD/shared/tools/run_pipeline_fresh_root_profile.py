from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter as pc
from typing import Callable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString


WALKABLE_HIGHWAYS: set[str] = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "track",
    "road",
    "pedestrian",
    "footway",
    "path",
    "steps",
    "cycleway",
    "bridleway",
}

FOOT_ALLOWED_VALUES: set[str] = {"yes", "designated", "permissive", "official"}
BLOCKING_VALUES: set[str] = {"no", "private", "agricultural", "forestry", "delivery"}
PEDESTRIAN_MOTORWAY_HIGHWAYS: set[str] = {"motorway", "motorway_link"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--fresh-base-root", type=Path, required=True)
    parser.add_argument(
        "--mobility-profile",
        choices=("driving", "driving_walk", "walking_trails"),
        default="driving",
        help="Network profile used for shortest path distances.",
    )
    return parser


def normalize_tag(value: object) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def make_walking_loaders(network_module: object) -> tuple[Callable[..., tuple[pd.DataFrame, pd.DataFrame]], Callable[..., gpd.GeoDataFrame]]:
    osmium = network_module.osmium
    if osmium is None:
        raise ImportError("The walking_trails mobility profile requires pyosmium.")

    def tag_value(tags: object, key: str) -> str | None:
        return normalize_tag(network_module._tag_value(tags, key))

    def is_walkable_way(tags: object) -> bool:
        highway = tag_value(tags, "highway")
        foot = tag_value(tags, "foot")
        access = tag_value(tags, "access")

        if foot in BLOCKING_VALUES or access in BLOCKING_VALUES:
            return False
        if highway in PEDESTRIAN_MOTORWAY_HIGHWAYS:
            return foot in FOOT_ALLOWED_VALUES
        return highway in WALKABLE_HIGHWAYS

    class WalkingNodeUseCounter(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.node_use_counts: dict[int, int] = {}

        def way(self, way: object) -> None:
            if not is_walkable_way(way.tags):
                return
            for node in way.nodes:
                node_id = int(node.ref)
                count = self.node_use_counts.get(node_id, 0)
                if count < 2:
                    self.node_use_counts[node_id] = count + 1

        def split_node_refs(self) -> set[int]:
            return {node_id for node_id, count in self.node_use_counts.items() if count > 1}

    class WalkingNetworkHandler(osmium.SimpleHandler):
        def __init__(
            self,
            bbox: object | None = None,
            *,
            collect_nodes: bool = True,
            include_geometry: bool = False,
            directed: bool = True,
            split_node_refs: set[int] | None = None,
        ) -> None:
            super().__init__()
            self.bbox = bbox
            self.collect_nodes = collect_nodes
            self.include_geometry = include_geometry
            self.directed = directed
            self.split_node_refs = split_node_refs
            self.nodes: dict[int, tuple[float, float]] = {}
            self.edge_u: list[int] = []
            self.edge_v: list[int] = []
            self.edge_length: list[float] = []
            self.edge_highway: list[str | None] = []
            self.edge_geometry: list[LineString] | None = [] if include_geometry else None

        def way(self, way: object) -> None:
            tags = way.tags
            if not is_walkable_way(tags):
                return

            highway = tag_value(tags, "highway")
            way_nodes: list[tuple[int, float, float]] = []
            for node in way.nodes:
                if not node.location.valid():
                    continue
                way_nodes.append((int(node.ref), float(node.location.lon), float(node.location.lat)))
            if len(way_nodes) < 2:
                return

            if self.split_node_refs is not None:
                self._append_simplified_way(way_nodes, highway)
                return

            for (u, lon1, lat1), (v, lon2, lat2) in zip(way_nodes, way_nodes[1:]):
                if u == v:
                    continue
                if not network_module._segment_intersects_bbox(lon1, lat1, lon2, lat2, self.bbox):
                    continue
                length = network_module._segment_length_m(lon1, lat1, lon2, lat2)
                if length <= 0:
                    continue
                self._append_edge_pair(u, v, length, highway, lon1, lat1, lon2, lat2)

        def _append_simplified_way(
            self,
            way_nodes: list[tuple[int, float, float]],
            highway: str | None,
        ) -> None:
            assert self.split_node_refs is not None
            start_id, start_lon, start_lat = way_nodes[0]
            prev_id, prev_lon, prev_lat = way_nodes[0]
            accumulated_length = 0.0
            has_included_segment = False
            last_index = len(way_nodes) - 1

            for idx, (node_id, lon, lat) in enumerate(way_nodes[1:], start=1):
                if node_id == prev_id:
                    prev_id, prev_lon, prev_lat = node_id, lon, lat
                    continue
                length = network_module._segment_length_m(prev_lon, prev_lat, lon, lat)
                if length <= 0:
                    prev_id, prev_lon, prev_lat = node_id, lon, lat
                    continue
                if not network_module._segment_intersects_bbox(prev_lon, prev_lat, lon, lat, self.bbox):
                    start_id, start_lon, start_lat = node_id, lon, lat
                    prev_id, prev_lon, prev_lat = node_id, lon, lat
                    accumulated_length = 0.0
                    has_included_segment = False
                    continue

                accumulated_length += length
                has_included_segment = True
                is_split_node = idx == last_index or node_id in self.split_node_refs
                if is_split_node:
                    if start_id != node_id and has_included_segment:
                        self._append_edge_pair(
                            start_id,
                            node_id,
                            accumulated_length,
                            highway,
                            start_lon,
                            start_lat,
                            lon,
                            lat,
                        )
                    start_id, start_lon, start_lat = node_id, lon, lat
                    accumulated_length = 0.0
                    has_included_segment = False

                prev_id, prev_lon, prev_lat = node_id, lon, lat

        def _append_edge_pair(
            self,
            u: int,
            v: int,
            length: float,
            highway: str | None,
            lon1: float,
            lat1: float,
            lon2: float,
            lat2: float,
        ) -> None:
            if self.collect_nodes:
                self.nodes[u] = (lon1, lat1)
                self.nodes[v] = (lon2, lat2)
            if self.directed:
                self._append_edge(u, v, length, highway, lon1, lat1, lon2, lat2)
                self._append_edge(v, u, length, highway, lon2, lat2, lon1, lat1)
            else:
                self._append_edge(u, v, length, highway, lon1, lat1, lon2, lat2)

        def _append_edge(
            self,
            u: int,
            v: int,
            length: float,
            highway: str | None,
            lon1: float,
            lat1: float,
            lon2: float,
            lat2: float,
        ) -> None:
            self.edge_u.append(u)
            self.edge_v.append(v)
            self.edge_length.append(float(length))
            self.edge_highway.append(highway)
            if self.edge_geometry is not None:
                self.edge_geometry.append(LineString([(lon1, lat1), (lon2, lat2)]))

        def nodes_frame(self) -> gpd.GeoDataFrame:
            node_ids: list[int] = []
            lons: list[float] = []
            lats: list[float] = []
            for node_id, (lon, lat) in self.nodes.items():
                node_ids.append(node_id)
                lons.append(lon)
                lats.append(lat)
            frame = pd.DataFrame({"id": node_ids, "lon": lons, "lat": lats})
            return gpd.GeoDataFrame(
                frame,
                geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
                crs="EPSG:4326",
            )

        def edges_frame(self) -> pd.DataFrame | gpd.GeoDataFrame:
            frame = pd.DataFrame(
                {
                    "u": self.edge_u,
                    "v": self.edge_v,
                    "length": self.edge_length,
                    "highway": pd.Categorical(self.edge_highway),
                }
            )
            if self.edge_geometry is None:
                return frame
            return gpd.GeoDataFrame(
                frame,
                geometry=gpd.GeoSeries(self.edge_geometry, crs="EPSG:4326"),
                crs="EPSG:4326",
            )

    def load_walking_network_data(
        pbf_path: str,
        verbose: bool = True,
        bbox: object | None = None,
        backend: str = "osmium_walking_trails",
        simplify: bool = False,
        **_: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del backend
        t0 = pc()
        split_node_refs = None
        if simplify:
            t_count = pc()
            counter = WalkingNodeUseCounter()
            counter.apply_file(str(pbf_path), locations=False)
            split_node_refs = counter.split_node_refs()
            referenced_node_count = len(counter.node_use_counts)
            del counter
            if verbose:
                print(
                    "Counted walkable/trail OSM node uses with osmium in "
                    f"{pc() - t_count:.2f} seconds, {referenced_node_count:,} "
                    f"referenced nodes, {len(split_node_refs):,} split nodes"
                )

        handler = WalkingNetworkHandler(
            bbox=bbox,
            collect_nodes=True,
            include_geometry=False,
            directed=simplify,
            split_node_refs=split_node_refs,
        )
        handler.apply_file(str(pbf_path), locations=True)
        nodes, edges = handler.nodes_frame(), handler.edges_frame()
        nodes, edges = network_module._prepare_network_data(nodes, edges)
        if verbose:
            area = "bbox extract" if bbox is not None else "full extract"
            mode = "simplified" if simplify else "unsimplified"
            print(
                f"Loaded {mode} walking/trail network data with osmium ({area}) "
                f"in {pc() - t0:.2f} seconds, {len(nodes):,} nodes, {len(edges):,} edges"
            )
        return nodes, edges

    def load_walking_edges(
        pbf_path: str,
        verbose: bool = True,
        bbox: object | None = None,
        backend: str = "osmium_walking_trails",
        **_: object,
    ) -> gpd.GeoDataFrame:
        del backend
        t0 = pc()
        handler = WalkingNetworkHandler(
            bbox=bbox,
            collect_nodes=False,
            include_geometry=True,
            directed=False,
        )
        handler.apply_file(str(pbf_path), locations=True)
        edges = handler.edges_frame()
        if verbose:
            area = "bbox extract" if bbox is not None else "full extract"
            print(
                f"Loaded walking/trail edges for map with osmium ({area}) "
                f"in {pc() - t0:.2f} seconds, {len(edges):,} edges"
            )
        return edges

    return load_walking_network_data, load_walking_edges


def main() -> None:
    wrapper_parser = build_parser()
    wrapper_args, remaining = wrapper_parser.parse_known_args()

    pipeline_dir = wrapper_args.pipeline_dir.resolve()
    sys.path.insert(0, str(pipeline_dir))

    from distance_pipeline.config_loader import load_cfg
    import distance_pipeline.network as network_module
    import run_pipeline as pipeline_module
    from run_pipeline import (
        build_parser as build_pipeline_parser,
        main as pipeline_main,
        resolve_destination_layers_from_args,
        resolve_input_config,
        resolve_source_layers_from_args,
        settings_from_args,
        setup_logging,
    )

    parser = build_pipeline_parser()
    args = parser.parse_args(remaining)
    setup_logging(args.log_file, verbose=not args.quiet)
    settings = settings_from_args(args)

    if wrapper_args.mobility_profile == "driving_walk":
        settings = replace(settings, network_profile="driving_walk")

    if wrapper_args.mobility_profile == "walking_trails":
        walking_network_loader, walking_edges_loader = make_walking_loaders(network_module)
        original_network_loader = pipeline_module.load_osm_network_data
        original_edges_loader = pipeline_module.load_osm_road_edges

        def load_profile_network_data(*loader_args: object, **loader_kwargs: object):
            backend = str(loader_kwargs.get("backend", ""))
            if backend.startswith("osmium_walking_trails"):
                return walking_network_loader(*loader_args, **loader_kwargs)
            return original_network_loader(*loader_args, **loader_kwargs)

        def load_profile_road_edges(*loader_args: object, **loader_kwargs: object):
            backend = str(loader_kwargs.get("backend", ""))
            if backend.startswith("osmium_walking_trails"):
                return walking_edges_loader(*loader_args, **loader_kwargs)
            return original_edges_loader(*loader_args, **loader_kwargs)

        pipeline_module.load_osm_network_data = load_profile_network_data
        pipeline_module.load_osm_road_edges = load_profile_road_edges
        network_identity = "osmium_walking_trails"
        if settings.simplify_network:
            network_identity = f"{network_identity}_simplified"
        settings = replace(settings, network_backend=network_identity)

    base_cfg = load_cfg(args.country_code)
    cfg = resolve_input_config(base_cfg, args)
    cfg = replace(cfg, base_root=wrapper_args.fresh_base_root)
    source_layers = resolve_source_layers_from_args(args)
    destination_layers = resolve_destination_layers_from_args(args)

    pipeline_main(
        cfg,
        settings,
        args.aggregate_factor,
        args.no_aggregate,
        args.build_map,
        args.map_only,
        args.amenity,
        source_layers,
        destination_layers,
        args.source_table,
        args.source_lon_column,
        args.source_lat_column,
        args.source_id_column,
        args.destination_table,
        args.destination_lon_column,
        args.destination_lat_column,
        args.destination_id_column,
        base_cfg=base_cfg,
    )


if __name__ == "__main__":
    main()
