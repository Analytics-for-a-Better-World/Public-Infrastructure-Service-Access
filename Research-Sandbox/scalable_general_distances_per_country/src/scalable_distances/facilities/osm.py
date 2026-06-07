from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def extract_osm_facilities(
    pbf_path: str | Path,
    *,
    amenity_values: Iterable[str],
    source_type: str = "amenities",
    role: str = "source",
    include_ways: bool = True,
) -> pd.DataFrame:
    """Extract OSM amenity nodes and way centroids from a PBF file as points."""
    import osmium

    wanted = set(amenity_values)
    id_col = f"{role}_id"
    type_col = f"{role}_type"

    class Handler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[dict[str, Any]] = []

        def _append(self, osm_type: str, osm_id: int, tags: Any, lon: float, lat: float) -> None:
            amenity = tags.get("amenity")
            if amenity not in wanted:
                return
            self.records.append(
                {
                    id_col: f"osm_{osm_type}_{int(osm_id)}",
                    type_col: source_type,
                    "amenity": str(amenity),
                    "name": None if tags.get("name") is None else str(tags.get("name")),
                    "lon": float(lon),
                    "lat": float(lat),
                    "osm_type": osm_type,
                    "osm_id": int(osm_id),
                }
            )

        def node(self, node: Any) -> None:
            amenity = node.tags.get("amenity")
            if amenity not in wanted or not node.location.valid():
                return
            self._append("node", int(node.id), node.tags, float(node.location.lon), float(node.location.lat))

        def way(self, way: Any) -> None:
            if not include_ways:
                return
            amenity = way.tags.get("amenity")
            if amenity not in wanted:
                return
            coords = [
                (float(node.location.lon), float(node.location.lat))
                for node in way.nodes
                if node.location.valid()
            ]
            if not coords:
                return
            lon = sum(item[0] for item in coords) / len(coords)
            lat = sum(item[1] for item in coords) / len(coords)
            self._append("way", int(way.id), way.tags, lon, lat)

    handler = Handler()
    handler.apply_file(str(pbf_path), locations=True)
    df = pd.DataFrame(handler.records)
    if df.empty:
        return df
    dedupe_cols = ["amenity", "name", "lon", "lat"]
    normalized = df.assign(
        name=df["name"].fillna("").astype(str).str.strip().str.lower(),
        lon=df["lon"].round(7),
        lat=df["lat"].round(7),
    )
    return df.loc[~normalized.duplicated(dedupe_cols)].reset_index(drop=True)
