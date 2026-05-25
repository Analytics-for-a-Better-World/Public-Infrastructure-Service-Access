from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scalable_distances.routing.base import NetworkData

DRIVABLE_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
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
}
BLOCKING_ACCESS_VALUES = {"no", "private", "agricultural", "forestry", "delivery"}
ONEWAY_FORWARD = {"yes", "true", "1"}
ONEWAY_REVERSE = {"-1", "reverse"}


def _tag(tags: Any, key: str) -> str | None:
    value = tags.get(key)
    return None if value is None else str(value)


def _is_drivable(tags: Any) -> bool:
    if _tag(tags, "highway") not in DRIVABLE_HIGHWAYS:
        return False
    return not any(_tag(tags, key) in BLOCKING_ACCESS_VALUES for key in ("access", "vehicle", "motor_vehicle", "motorcar"))


def _directions(tags: Any) -> tuple[bool, bool]:
    oneway = (_tag(tags, "oneway") or "").strip().lower()
    junction = (_tag(tags, "junction") or "").strip().lower()
    if oneway in ONEWAY_REVERSE:
        return False, True
    if oneway in ONEWAY_FORWARD or junction == "roundabout":
        return True, False
    return True, True


def _length_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    from pyproj import Geod

    _, _, distance = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    return float(distance)


def _intersects_bbox(lon1: float, lat1: float, lon2: float, lat2: float, bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return not (
        max(lon1, lon2) < min_lon
        or min(lon1, lon2) > max_lon
        or max(lat1, lat2) < min_lat
        or min(lat1, lat2) > max_lat
    )


def load_driving_network(
    pbf_path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> NetworkData:
    """Parse a drivable OSM PBF network with the npyosmium/osmium backend."""
    import osmium

    class Handler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nodes: dict[int, tuple[float, float]] = {}
            self.edges: list[dict[str, Any]] = []

        def way(self, way: Any) -> None:
            if not _is_drivable(way.tags):
                return
            forward, backward = _directions(way.tags)
            refs: list[tuple[int, float, float]] = []
            for node in way.nodes:
                if node.location.valid():
                    refs.append((int(node.ref), float(node.location.lon), float(node.location.lat)))
            for (u, lon1, lat1), (v, lon2, lat2) in zip(refs, refs[1:]):
                if u == v or not _intersects_bbox(lon1, lat1, lon2, lat2, bbox):
                    continue
                length = _length_m(lon1, lat1, lon2, lat2)
                if length <= 0:
                    continue
                self.nodes[u] = (lon1, lat1)
                self.nodes[v] = (lon2, lat2)
                if forward:
                    self.edges.append({"u": u, "v": v, "length_m": length})
                if backward:
                    self.edges.append({"u": v, "v": u, "length_m": length})

    handler = Handler()
    handler.apply_file(str(pbf_path), locations=True)
    nodes = pd.DataFrame(
        [{"node_id": node_id, "lon": lon, "lat": lat} for node_id, (lon, lat) in handler.nodes.items()]
    )
    edges = pd.DataFrame(handler.edges)
    if nodes.empty or edges.empty:
        raise ValueError(f"No drivable network extracted from {pbf_path}")
    return NetworkData(nodes=nodes, edges=edges)
