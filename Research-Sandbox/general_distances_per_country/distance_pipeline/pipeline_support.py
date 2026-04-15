from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from countries.base import CountryConfig
from distance_pipeline.settings import PipelineSettings


def resolve_candidate_grid_spacing(
    cfg: CountryConfig,
    settings: PipelineSettings,
) -> float | None:
    """Resolve the candidate grid spacing from runtime settings or country config."""
    if settings.candidate_grid_spacing_m is not None:
        return settings.candidate_grid_spacing_m
    return cfg.candidate_grid_spacing_m



def resolve_candidate_max_snap_dist(
    cfg: CountryConfig,
    settings: PipelineSettings,
) -> float | None:
    """Resolve the candidate maximum snap distance from runtime settings or country config."""
    if settings.candidate_max_snap_dist_m is not None:
        return settings.candidate_max_snap_dist_m
    return cfg.candidate_max_snap_dist_m



def ensure_xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the table contains ``Longitude`` and ``Latitude`` columns."""
    result = df.copy()

    has_lon = 'Longitude' in result.columns
    has_lat = 'Latitude' in result.columns

    if has_lon and has_lat:
        return result

    if 'geometry' not in result.columns:
        raise KeyError(
            "Missing coordinate columns. Expected 'Longitude' and 'Latitude', "
            "or a 'geometry' column with point geometries."
        )

    result['Longitude'] = result.geometry.x
    result['Latitude'] = result.geometry.y
    return result



def format_resolution_suffix(candidate_grid_spacing_m: float | None) -> str:
    """Format the candidate resolution suffix for filenames."""
    if candidate_grid_spacing_m is None:
        return 'no_candidates'

    if float(candidate_grid_spacing_m).is_integer():
        resolution = str(int(candidate_grid_spacing_m))
    else:
        resolution = str(candidate_grid_spacing_m).replace('.', 'p')

    return f'resolution_{resolution}m'



def build_context_map_path(
    default_path: Path,
    user_path: Path | None,
    candidate_grid_spacing_m: float | None,
) -> Path:
    """Build the output path for the context map."""
    base_path = user_path if user_path is not None else default_path
    suffix = format_resolution_suffix(candidate_grid_spacing_m)
    return base_path.with_name(f'{base_path.stem}_{suffix}{base_path.suffix}')



def build_map_facilities(
    health_centers: pd.DataFrame,
    candidate_sites: pd.DataFrame | None,
) -> gpd.GeoDataFrame:
    """Build a facilities layer for plotting existing amenities and candidates."""
    if 'geometry' not in health_centers.columns:
        raise ValueError('health_centers must contain a geometry column')

    existing = gpd.GeoDataFrame(
        health_centers.copy(),
        geometry='geometry',
        crs=getattr(health_centers, 'crs', None),
    )

    if existing.crs is None:
        raise ValueError('health_centers has no CRS')

    existing = ensure_xy_columns(existing)
    existing['source_type'] = 'existing'

    if candidate_sites is None:
        return existing

    candidates_df = ensure_xy_columns(candidate_sites)

    if 'geometry' in candidates_df.columns:
        candidates = gpd.GeoDataFrame(
            candidates_df.copy(),
            geometry='geometry',
            crs=getattr(candidate_sites, 'crs', existing.crs),
        )
        if candidates.crs is None:
            candidates = candidates.set_crs(existing.crs)
        elif candidates.crs != existing.crs:
            candidates = candidates.to_crs(existing.crs)
    else:
        candidates = gpd.GeoDataFrame(
            candidates_df.copy(),
            geometry=gpd.points_from_xy(
                candidates_df['Longitude'],
                candidates_df['Latitude'],
                crs='EPSG:4326',
            ),
        ).to_crs(existing.crs)

    candidates['source_type'] = 'candidate'

    combined = pd.concat([existing, candidates], ignore_index=True, sort=False)
    return gpd.GeoDataFrame(combined, geometry='geometry', crs=existing.crs)
