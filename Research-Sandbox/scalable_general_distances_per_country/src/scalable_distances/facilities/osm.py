from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def extract_osm_facilities(
    pbf_path: str | Path,
    *,
    amenity_values: Iterable[str],
    source_type: str = "amenities",
) -> pd.DataFrame:
    """Extract OSM node amenities from a PBF file as source points."""
    import osmium

    wanted = set(amenity_values)

    class Handler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[dict[str, Any]] = []

        def node(self, node: Any) -> None:
            amenity = node.tags.get("amenity")
            if amenity not in wanted or not node.location.valid():
                return
            self.records.append(
                {
                    "source_id": f"osm_node_{int(node.id)}",
                    "source_type": source_type,
                    "amenity": str(amenity),
                    "name": None if node.tags.get("name") is None else str(node.tags.get("name")),
                    "lon": float(node.location.lon),
                    "lat": float(node.location.lat),
                }
            )

    handler = Handler()
    handler.apply_file(str(pbf_path), locations=True)
    return pd.DataFrame(handler.records)
