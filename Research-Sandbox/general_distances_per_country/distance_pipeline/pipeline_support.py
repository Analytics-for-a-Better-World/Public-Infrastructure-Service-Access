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


def format_output_value(value: float | int | None) -> str:
    """Format a value for output filenames."""
    if value is None:
        return 'none'
    return f'{value:g}'


def safe_filename_part(value: object) -> str:
    """Return a compact filename-safe representation of a setting value."""
    text = str(value).strip().lower()
    safe = ''.join(char if char.isalnum() else '_' for char in text).strip('_')
    return safe or 'none'


def format_amenity_suffix(amenity_values: list[str] | None) -> str:
    """Format amenity filters for output filenames."""
    if amenity_values is None:
        return 'all'
    return '-'.join(sorted(amenity_values))


def format_snap_components_suffix(snap_components: tuple[int, ...] | None) -> str:
    """Format optional component-restricted snapping for output filenames."""
    if snap_components is None:
        return ''
    return '_snap_components_' + '-'.join(str(component_id) for component_id in snap_components)


def format_filename_part(value: str) -> str:
    """Format a free-form value for output filenames."""
    text = value.strip().lower()
    if not text:
        return 'custom'
    return ''.join(ch if ch.isalnum() else '-' for ch in text).strip('-') or 'custom'


def format_pbf_suffix(pbf_filename: str | None) -> str:
    """Format an optional OSM PBF identifier for output filenames."""
    if pbf_filename is None:
        return ''

    name = Path(pbf_filename).name
    if name.lower().endswith('.osm.pbf'):
        stem = name[:-len('.osm.pbf')]
    else:
        stem = Path(name).stem
    return f'_pbf_{format_filename_part(stem)}'


def format_network_suffix(settings: PipelineSettings) -> str:
    """Format non-historical network modes for output filenames."""
    network_backend = settings.network_cache_backend()
    if network_backend in ('', 'pyrosm'):
        return ''
    return f'_network_{format_filename_part(network_backend)}'


def build_output_run_tag(
    *,
    settings: PipelineSettings,
    aggregate_factor: int | None,
    amenity_values: list[str] | None,
    candidate_grid_spacing_m: float | None,
    candidate_max_snap_dist_m: float | None,
    has_candidates: bool,
    pbf_filename: str | None = None,
) -> str:
    """Build a filename-safe tag describing the pipeline output settings."""
    candidate_part = (
        f"candidates_spacing_{format_output_value(candidate_grid_spacing_m)}_"
        f"maxsnap_{format_output_value(candidate_max_snap_dist_m)}"
        if has_candidates
        else 'no_candidates'
    )
    snap_part = format_snap_components_suffix(settings.snap_components)
    pbf_part = format_pbf_suffix(pbf_filename)
    network_part = format_network_suffix(settings)
    impedance_part = (
        ''
        if settings.network_impedance == 'length'
        else f'_impedance_{format_filename_part(settings.network_impedance)}'
    )
    chunk_part = (
        ''
        if settings.sparse_target_chunk_size is None
        else f'_sparse_chunks_{format_output_value(settings.sparse_target_chunk_size)}'
    )

    return (
        f"pop_{settings.population_threshold:g}_"
        f"sample_{settings.sample_fraction:g}_"
        f"seed_{settings.random_seed}_"
        f"max_{format_output_value(settings.max_points)}_"
        f"agg_{format_output_value(aggregate_factor)}_"
        f"maxdist_{format_output_value(settings.max_total_dist)}_"
        f"amenity_{format_amenity_suffix(amenity_values)}_"
        f"{candidate_part}{snap_part}{pbf_part}{network_part}"
        f"{impedance_part}{chunk_part}"
    )



def build_map_facilities(
    facilities: pd.DataFrame,
    candidate_sites: pd.DataFrame | None,
) -> gpd.GeoDataFrame:
    """Build a facilities layer for plotting existing amenities and candidates."""
    if 'geometry' not in facilities.columns:
        raise ValueError('facilities must contain a geometry column')

    existing = gpd.GeoDataFrame(
        facilities.copy(),
        geometry='geometry',
        crs=getattr(facilities, 'crs', None),
    )

    if existing.crs is None:
        raise ValueError('facilities has no CRS')

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
