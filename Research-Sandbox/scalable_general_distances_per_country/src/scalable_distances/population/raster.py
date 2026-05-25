from __future__ import annotations

from pathlib import Path

import pandas as pd


def worldpop_to_points(
    raster_path: str | Path,
    *,
    population_threshold: float = 1.0,
    aggregate_factor: int | None = None,
) -> pd.DataFrame:
    """Convert a WorldPop raster into target points."""
    import numpy as np
    import rasterio
    from affine import Affine
    from rasterio.transform import xy

    with rasterio.open(raster_path) as src:
        band = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata

    if aggregate_factor is not None and aggregate_factor > 1:
        height, width = band.shape
        new_height = height // aggregate_factor
        new_width = width // aggregate_factor
        trimmed = band[: new_height * aggregate_factor, : new_width * aggregate_factor]
        valid = np.isfinite(trimmed)
        if nodata is not None:
            valid &= trimmed != nodata
        band = np.where(valid, trimmed, 0.0).reshape(
            new_height, aggregate_factor, new_width, aggregate_factor
        ).sum(axis=(1, 3))
        transform = transform * Affine.scale(aggregate_factor, aggregate_factor)

    mask = np.isfinite(band)
    if nodata is not None:
        mask &= band != nodata
    mask &= band >= population_threshold
    rows, cols = np.where(mask)
    lon, lat = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "target_id": [f"pop_{i}" for i in range(len(rows))],
            "target_type": "population",
            "lon": lon,
            "lat": lat,
            "population": band[rows, cols],
        }
    )
