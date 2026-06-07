from __future__ import annotations

from typing import Literal

import pandas as pd

from scalable_distances.layers import as_role_points


def build_candidate_grid(
    bbox: tuple[float, float, float, float],
    *,
    spacing_m: float = 1000.0,
    role: Literal["source", "target"] = "source",
) -> pd.DataFrame:
    """Build a lightweight regular WGS84 candidate grid inside a bbox.

    The grid is intentionally dependency-light. It uses a latitude-adjusted
    degree spacing, which is sufficient for pre-snapping candidate sites; the
    road-network router computes final distances.
    """
    import math
    import numpy as np

    min_lon, min_lat, max_lon, max_lat = bbox
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive.")
    mid_lat = (min_lat + max_lat) / 2.0
    dlat = spacing_m / 111_320.0
    dlon = spacing_m / max(111_320.0 * math.cos(math.radians(mid_lat)), 1.0)
    lats = np.arange(min_lat, max_lat + dlat / 2.0, dlat)
    lons = np.arange(min_lon, max_lon + dlon / 2.0, dlon)
    records = [
        {"lon": float(lon), "lat": float(lat)}
        for lat in lats
        for lon in lons
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
    ]
    result = pd.DataFrame(records)
    return as_role_points(
        result,
        role=role,
        layer_type="candidates",
        id_prefix=f"{role}_candidate",
    )
