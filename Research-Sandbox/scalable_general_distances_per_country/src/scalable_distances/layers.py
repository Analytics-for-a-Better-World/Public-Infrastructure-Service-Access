from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

PointRole = Literal["source", "target"]
LayerName = Literal["population", "amenities", "table", "candidates"]


def normalize_layers(values: Iterable[str], *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize user-facing layer aliases to canonical layer names."""
    aliases = {
        "amenity": "amenities",
        "amenities": "amenities",
        "candidate": "candidates",
        "candidates": "candidates",
        "grid": "candidates",
        "population": "population",
        "pop": "population",
        "table": "table",
        "file": "table",
    }
    result: list[str] = []
    for value in values or default:
        key = str(value).strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                f"Unsupported point layer {value!r}. "
                "Use population, amenities, table, or candidates."
            )
        canonical = aliases[key]
        if canonical not in result:
            result.append(canonical)
    return tuple(result or default)


def filter_bbox(
    points: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """Return only rows inside a WGS84 bbox."""
    if bbox is None or points.empty:
        return points.copy()
    min_lon, min_lat, max_lon, max_lat = bbox
    mask = (
        points["lon"].astype(float).between(min_lon, max_lon)
        & points["lat"].astype(float).between(min_lat, max_lat)
    )
    return points.loc[mask].reset_index(drop=True)


def load_point_table(
    path: str | Path,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    id_col: str | None = None,
    layer_type: str = "table",
    role: PointRole = "source",
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Load a CSV, Excel, parquet, GeoJSON, or GIS point table as source/target points."""
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(table_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path)
    elif suffix in {".geojson", ".gpkg", ".shp"}:
        import geopandas as gpd

        gdf = gpd.read_file(table_path)
        if gdf.crs is not None and str(gdf.crs).upper() not in {"EPSG:4326", "WGS84"}:
            gdf = gdf.to_crs(4326)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        df["lon"] = gdf.geometry.x
        df["lat"] = gdf.geometry.y
        lon_col = "lon"
        lat_col = "lat"
    else:
        df = pd.read_csv(table_path)

    missing = {lon_col, lat_col} - set(df.columns)
    if missing:
        raise ValueError(f"{table_path} is missing coordinate column(s): {sorted(missing)}")

    result = df.copy()
    result["lon"] = result[lon_col].astype(float)
    result["lat"] = result[lat_col].astype(float)
    type_col = f"{role}_type"
    id_name = f"{role}_id"
    result[type_col] = layer_type
    if id_col is not None and id_col in result.columns:
        result[id_name] = result[id_col].astype(str)
    elif id_name not in result.columns:
        result[id_name] = [f"{layer_type}_{i}" for i in range(len(result))]
    return filter_bbox(result, bbox)


def as_role_points(
    points: pd.DataFrame,
    *,
    role: PointRole,
    layer_type: str,
    id_prefix: str,
) -> pd.DataFrame:
    """Make source_id/source_type or target_id/target_type columns without copying geometry semantics."""
    result = points.copy()
    id_col = f"{role}_id"
    type_col = f"{role}_type"
    if id_col not in result.columns:
        result[id_col] = [f"{id_prefix}_{i}" for i in range(len(result))]
    if type_col not in result.columns:
        result[type_col] = layer_type
    return result
