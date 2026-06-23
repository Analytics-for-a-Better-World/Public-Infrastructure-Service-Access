from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter as pc
from urllib.parse import unquote, urlparse

import geopandas as gpd
import numpy as np
import pandas as pd

from countries.base import CountryConfig
from distance_pipeline.cache import CacheManager
from distance_pipeline.candidate_builder import build_candidate_grid, build_candidate_sites
from distance_pipeline.config_loader import load_cfg
from distance_pipeline.distance_matrix import (
    compute_dense_distance_matrices,
    compute_distances_polars,
)
from distance_pipeline.facilities import deduplicate_osm_amenities, load_facilities
from distance_pipeline.io import download_file
from distance_pipeline.manifest import build_run_manifest, write_run_manifest
from distance_pipeline.network import (
    build_pandana_network,
    load_osm_network_data,
    load_osm_road_edges,
)
from distance_pipeline.pipeline_support import (
    build_context_map_path,
    build_map_facilities,
    build_output_run_tag,
    resolve_candidate_grid_spacing,
    resolve_candidate_max_snap_dist,
)
from distance_pipeline.population import worldpop_to_points
from distance_pipeline.routing import add_edge_speeds
from distance_pipeline.settings import PipelineSettings
from distance_pipeline.snapping import snap_points_to_nodes
from distance_pipeline.source_tables import (
    ensure_id_column,
    ensure_id_index_matches,
    load_custom_points_table,
    prepare_points_as_sources,
    prepare_points_as_targets,
    set_known_categories,
)
from distance_pipeline.viz import (
    CONTEXT_BASEMAP_CHOICES,
    classify_roads,
    plot_context_map,
    to_point_geometries,
)

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars is a required runtime dependency
    pl = None


LAYER_ALIASES: dict[str, str] = {
    'amenity': 'amenities',
    'amenities': 'amenities',
    'osm': 'amenities',
    'table': 'table',
    'custom': 'table',
    'candidate': 'candidates',
    'candidates': 'candidates',
    'grid': 'candidates',
    'population': 'population',
    'pop': 'population',
}

SOURCE_LAYER_DEFAULTS: tuple[str, ...] = ('amenities', 'candidates')
DESTINATION_LAYER_DEFAULTS: tuple[str, ...] = ('population',)


def parse_bbox(values: list[float] | None) -> tuple[float, float, float, float] | None:
    """Parse and validate a lon/lat bounding box."""
    if values is None:
        return None

    min_lon, min_lat, max_lon, max_lat = values
    if min_lon >= max_lon:
        raise ValueError('--bbox min_lon must be smaller than max_lon.')
    if min_lat >= max_lat:
        raise ValueError('--bbox min_lat must be smaller than max_lat.')
    if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180:
        raise ValueError('--bbox longitudes must be between -180 and 180.')
    if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90:
        raise ValueError('--bbox latitudes must be between -90 and 90.')

    return min_lon, min_lat, max_lon, max_lat


def parse_snap_components(value: str | None) -> tuple[int, ...] | None:
    """Parse allowed snap component IDs from comma-separated values/ranges."""
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() in {'none', 'all'}:
        return None

    component_ids: list[int] = []
    seen: set[int] = set()

    for raw_part in text.split(','):
        part = raw_part.strip()
        if not part:
            raise ValueError('--snap-components contains an empty item.')

        if '-' in part:
            bounds = [bound.strip() for bound in part.split('-', maxsplit=1)]
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f'Invalid --snap-components range: {part!r}.')
            start, end = (int(bounds[0]), int(bounds[1]))
            if start < 0 or end < 0:
                raise ValueError('--snap-components cannot contain negative IDs.')
            if start > end:
                raise ValueError(f'Invalid descending --snap-components range: {part!r}.')
            values = range(start, end + 1)
        else:
            component_id = int(part)
            if component_id < 0:
                raise ValueError('--snap-components cannot contain negative IDs.')
            values = (component_id,)

        for component_id in values:
            if component_id not in seen:
                seen.add(component_id)
                component_ids.append(component_id)

    return tuple(sorted(component_ids))


def filter_to_bbox(
    gdf: gpd.GeoDataFrame | pd.DataFrame | None,
    bbox: tuple[float, float, float, float] | None,
) -> gpd.GeoDataFrame | pd.DataFrame | None:
    """Filter a geometry layer to a lon/lat bounding box."""
    if gdf is None or bbox is None:
        return gdf
    if 'geometry' not in gdf.columns:
        return gdf

    layer = gpd.GeoDataFrame(
        gdf.copy(),
        geometry='geometry',
        crs=getattr(gdf, 'crs', None),
    )
    if layer.crs is None:
        raise ValueError('Cannot apply --bbox to a geometry layer without CRS.')

    min_lon, min_lat, max_lon, max_lat = bbox
    original_crs = layer.crs
    layer_ll = layer.to_crs(4326)
    filtered = layer_ll.cx[min_lon:max_lon, min_lat:max_lat].copy()
    return filtered.to_crs(original_crs)


def empty_roads_layer() -> gpd.GeoDataFrame:
    """Return an empty road layer compatible with context-map plotting."""
    return gpd.GeoDataFrame(
        {'road_class': pd.Categorical([])},
        geometry=gpd.GeoSeries([], crs='EPSG:4326'),
        crs='EPSG:4326',
    )


def normalize_layers(
    layers: list[str] | None,
    *,
    defaults: tuple[str, ...],
    argument_name: str,
) -> list[str]:
    """Normalize layer names while preserving first occurrence order."""
    raw_layers = list(defaults if layers is None else layers)
    normalized: list[str] = []

    for raw_layer in raw_layers:
        layer_key = raw_layer.strip().lower().replace('-', '_')
        layer = LAYER_ALIASES.get(layer_key)
        if layer is None:
            valid = ', '.join(sorted(set(LAYER_ALIASES)))
            raise ValueError(
                f'Unsupported layer {raw_layer!r} in {argument_name}. '
                f'Expected one of: {valid}.'
            )
        if layer not in normalized:
            normalized.append(layer)

    if not normalized:
        raise ValueError(f'{argument_name} must contain at least one layer.')

    return normalized


def table_descriptor(path: str | Path | None) -> str:
    """Build a filename-safe descriptor for a custom table path."""
    if path is None:
        return 'table_missing'

    table_path = Path(path)
    raw_descriptor = ''.join(
        char.lower() if char.isalnum() else '_'
        for char in table_path.stem
    ).strip('_')
    descriptor = raw_descriptor or "custom"
    if len(descriptor) > 48:
        digest = hashlib.sha1(str(table_path).encode('utf-8')).hexdigest()[:10]
        descriptor = f'{descriptor[:36]}_{digest}'
    return f'table_{descriptor}'


def layer_signature(
    *,
    source_layers: list[str],
    destination_layers: list[str],
    amenity_values: list[str] | None,
    source_table: str | None,
    destination_table: str | None,
) -> list[str]:
    """Build a cache/run-tag signature that distinguishes layer combinations."""
    amenity_part = (
        'amenity_all'
        if amenity_values is None
        else 'amenity_' + '-'.join(sorted(amenity_values))
    )
    source_parts = [
        table_descriptor(source_table) if layer == 'table' else layer
        for layer in source_layers
    ]
    destination_parts = [
        table_descriptor(destination_table or source_table) if layer == 'table' else layer
        for layer in destination_layers
    ]
    return [
        f'src_{"-".join(source_parts)}',
        f'dst_{"-".join(destination_parts)}',
        amenity_part,
    ]


def with_prefixed_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix row IDs so combined source/target layers cannot collide."""
    result = ensure_id_column(df, prefix=prefix).copy()
    result['ID'] = prefix + '_' + result['ID'].astype(str)
    return result


def resolve_source_layers_from_args(args: argparse.Namespace) -> list[str]:
    """Resolve composable source layers while preserving legacy CLI behavior."""
    if args.sources is not None:
        return normalize_layers(
            args.sources,
            defaults=SOURCE_LAYER_DEFAULTS,
            argument_name='--sources',
        )

    if args.source_layer == 'table':
        return ['table']

    return list(SOURCE_LAYER_DEFAULTS)


def resolve_destination_layers_from_args(args: argparse.Namespace) -> list[str]:
    """Resolve composable destination layers from CLI arguments."""
    return normalize_layers(
        args.destinations,
        defaults=DESTINATION_LAYER_DEFAULTS,
        argument_name='--destinations',
    )


def ensure_table_arguments(
    *,
    source_layers: list[str],
    destination_layers: list[str],
    source_table: str | None,
    destination_table: str | None,
) -> None:
    """Validate table-layer arguments."""
    if 'table' in source_layers and source_table is None:
        raise ValueError('--source-table is required when sources include table.')
    if 'table' in destination_layers and destination_table is None and source_table is None:
        raise ValueError(
            '--destination-table is required when destinations include table '
            'unless --source-table should be reused.'
        )


def concat_layers(layers: list[pd.DataFrame], *, label: str) -> pd.DataFrame:
    """Concatenate layer frames with a clear error for empty selections."""
    if not layers:
        raise ValueError(f'No {label} layers were built.')
    result = pd.concat(layers, axis=0, sort=False)
    result = ensure_id_index_matches(result)
    return set_known_categories(result)


def candidate_layer_for_role(
    candidate_sites_snapped: pd.DataFrame,
    *,
    role: str,
    id_prefix: str,
) -> pd.DataFrame:
    """Convert snapped candidates into either source or target schema input."""
    distance_col = f'dist_snap_{role}'
    result = candidate_sites_snapped.rename(
        columns={'candidate_dist_road_estrada': distance_col}
    )
    return with_prefixed_ids(result, id_prefix)


def validate_pbf_filename(value: str) -> str:
    """Validate and normalize an OSM PBF cache filename."""
    filename = str(value).strip()
    if not filename:
        raise ValueError('--pbf-filename cannot be empty.')
    if Path(filename).name != filename:
        raise ValueError('--pbf-filename must be a filename, not a path.')
    if not filename.lower().endswith('.osm.pbf'):
        raise ValueError('--pbf-filename must end with .osm.pbf.')
    return filename


def pbf_filename_from_url(pbf_url: str) -> str:
    """Derive an OSM PBF filename from a download URL path."""
    filename = Path(unquote(urlparse(pbf_url).path)).name
    if not filename:
        raise ValueError('--pbf-url must include a filename, or use --pbf-filename.')
    return validate_pbf_filename(filename)


def resolve_input_config(
    cfg: CountryConfig,
    args: argparse.Namespace,
) -> CountryConfig:
    """Apply runtime input dataset/version overrides to a country config."""
    overrides: dict[str, object] = {}
    pbf_url = getattr(args, 'pbf_url', None)
    pbf_filename = getattr(args, 'pbf_filename', None)

    if args.worldpop_year is not None:
        overrides['worldpop_year'] = args.worldpop_year
    if args.worldpop_dataset is not None:
        overrides['worldpop_dataset'] = args.worldpop_dataset
        if args.worldpop_dataset == 'global2' and args.worldpop_filename is None:
            overrides['worldpop_filename'] = None
    if args.worldpop_release is not None:
        overrides['worldpop_release'] = args.worldpop_release
    if args.worldpop_version is not None:
        overrides['worldpop_version'] = args.worldpop_version
    if args.worldpop_resolution is not None:
        overrides['worldpop_resolution'] = args.worldpop_resolution
    if args.worldpop_constrained is not None:
        overrides['worldpop_constrained'] = args.worldpop_constrained == 'true'
    if args.worldpop_filename is not None:
        overrides['worldpop_filename'] = args.worldpop_filename
    if args.worldpop_url is not None:
        overrides['worldpop_url'] = args.worldpop_url
    if args.worldpop_path is not None:
        overrides['worldpop_path'] = Path(args.worldpop_path)
    if pbf_url is not None:
        overrides['pbf_url'] = pbf_url
    if pbf_filename is not None:
        overrides['pbf_filename'] = validate_pbf_filename(pbf_filename)
    elif pbf_url is not None:
        overrides['pbf_filename'] = pbf_filename_from_url(pbf_url)

    if not overrides:
        return cfg

    resolved = replace(cfg, **overrides)
    if resolved.worldpop_dataset == 'global2' and not 2015 <= resolved.worldpop_year <= 2030:
        raise ValueError('WorldPop Global2 years must be between 2015 and 2030.')
    return resolved


def resolve_worldpop_config(
    cfg: CountryConfig,
    args: argparse.Namespace,
) -> CountryConfig:
    """Apply runtime input overrides to a country config.

    Kept for compatibility with notebooks and wrappers that imported the old
    WorldPop-only helper name.
    """
    return resolve_input_config(cfg, args)


def pbf_filename_for_output_tag(
    cfg: CountryConfig,
    base_cfg: CountryConfig | None = None,
) -> str | None:
    """Return the PBF filename to include in output names, if any."""
    if base_cfg is not None:
        if (
            cfg.resolved_pbf_filename == base_cfg.resolved_pbf_filename
            and cfg.PBF_URL == base_cfg.PBF_URL
        ):
            return None
        return cfg.resolved_pbf_filename

    default_filename = f'{cfg.country_slug}-latest.osm.pbf'
    if cfg.pbf_url is None and cfg.resolved_pbf_filename == default_filename:
        return None
    return cfg.resolved_pbf_filename


# ------------------------
# Logging setup
# ------------------------
def setup_logging(log_file: str | None, verbose: bool) -> None:
    '''Configure logging to console and optionally to file, and capture stdout/stderr.'''
    import sys

    level = logging.INFO if verbose else logging.WARNING

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # ------------------------
    # Redirect stdout/stderr
    # ------------------------
    class StreamToLogger:
        def __init__(self, logger: logging.Logger, level: int) -> None:
            self.logger = logger
            self.level = level

        def write(self, message: str) -> None:
            message = message.strip()
            if message:
                self.logger.log(self.level, message)

        def flush(self) -> None:
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.WARNING)

# ------------------------
# CLI
# ------------------------
def build_parser() -> argparse.ArgumentParser:
    '''Build the command line argument parser.'''
    parser = argparse.ArgumentParser(
        description='Run the distance pipeline for a country configuration.'
    )

    parser.add_argument(
        'country_code',
        help=(
            'Country config module name, for example portugal, netherlands, '
            'timor_leste, vietnam, prt, nld, or tls.'
        ),
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Optional path to write logs to file.'
    )

    parser.add_argument(
        '--build-map',
        action='store_true',
        help='Build and plot the context map (disabled by default).'
    )

    parser.add_argument(
        '--map-only',
        action='store_true',
        help=(
            'Build the context map, save it, and stop before snapping sources, '
            'targets, or computing the distance matrix.'
        ),
    )

    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Ignore caches and rebuild all cached steps.',
    )

    parser.add_argument(
        '--save-map',
        action='store_true',
        help='Save the context map to a file.',
    )

    parser.add_argument(
        '--show-map',
        action='store_true',
        help='Display the context map interactively.',
    )

    parser.add_argument(
        '--map-path',
        type=str,
        default=None,
        help='Optional path for the saved context map.',
    )

    parser.add_argument(
        '--map-dpi',
        type=int,
        default=300,
        help='DPI used when saving the context map.',
    )

    parser.add_argument(
        '--map-basemap',
        choices=CONTEXT_BASEMAP_CHOICES,
        default='voyager-no-labels',
        help='Context map basemap tiles. Default: voyager-no-labels.',
    )

    parser.add_argument(
        '--map-basemap-alpha',
        type=float,
        default=0.52,
        help='Context map basemap opacity between 0 and 1. Default: 0.52.',
    )

    parser.add_argument(
        '--map-roads',
        choices=('true', 'false'),
        default='true',
        help='Plot the OSM road overlay on context maps. Default: true.',
    )

    parser.add_argument(
        '--map-legend-loc',
        default='center left',
        help=(
            'Matplotlib legend location for context maps. '
            'Default: center left.'
        ),
    )

    parser.add_argument(
        '--map-legend-bbox-to-anchor',
        nargs=2,
        type=float,
        metavar=('X', 'Y'),
        default=(1.02, 0.5),
        help=(
            'Optional context-map legend anchor as X Y in axes coordinates. '
            'Default: 1.02 0.5, which places the legend outside on the right.'
        ),
    )

    parser.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        default=None,
        help=(
            'Optional lon/lat bounding box used to subset geometry layers, '
            'for example 115 -12 128 -6.'
        ),
    )

    parser.add_argument(
        '--network-backend',
        choices=('pyrosm', 'osmium', 'auto'),
        default='pyrosm',
        help=(
            'Backend used to extract the OSM driving network. '
            'pyrosm preserves the historical behavior. osmium streams the PBF '
            'with pyosmium and can be more memory cautious for large extracts. '
            'auto uses osmium when the optional package is installed.'
        ),
    )

    parser.add_argument(
        '--simplify-network',
        choices=('true', 'false'),
        default='false',
        help=(
            'For the osmium backend, collapse intermediate OSM way vertices '
            'before building the routing graph. Default false preserves a '
            'denser, pyrosm-like graph.'
        ),
    )

    parser.add_argument(
        '--network-profile',
        choices=('driving', 'driving_walk'),
        default='driving',
        help=(
            'OSM network profile. driving preserves the historical drivable '
            'road network. driving_walk additionally includes pedestrian and '
            'cycling trail classes and requires --network-backend osmium or auto.'
        ),
    )

    parser.add_argument(
        '--diagnose-connectivity',
        choices=('true', 'false'),
        default='false',
        help=(
            'Compute weak connected components of the loaded road graph, log '
            'component counts, and label snapped sources/destinations with '
            'component_id.'
        ),
    )

    parser.add_argument(
        '--snap-components',
        type=str,
        default=None,
        help=(
            'Restrict snapping to road-network weak component IDs, ordered by '
            'component size. Accepts comma-separated IDs and ranges, for '
            'example 0 or 0,2,5-7. Default: snap to all components.'
        ),
    )

    parser.add_argument(
        '--population-threshold',
        type=float,
        default=1.0,
        help='Minimum population threshold used for raster to points conversion.',
    )

    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=1.0,
        help='Sampling fraction used for raster to points conversion.',
    )

    parser.add_argument(
        '--max-points',
        type=int,
        default=None,
        help='Maximum number of population points to keep.',
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed used for population sampling and max-points capping.',
    )

    parser.add_argument(
        '--max-total-dist',
        type=float,
        default=None,
        help='Optional maximum total distance to retain in the output matrix.',
    )

    parser.add_argument(
        '--worldpop-year',
        type=int,
        default=None,
        help='WorldPop population year. Overrides the country config.',
    )

    parser.add_argument(
        '--worldpop-dataset',
        choices=('global1', 'global2'),
        default=None,
        help=(
            'WorldPop dataset family. global1 uses the archived 2000-2020 tree; '
            'global2 uses the 2015-2030 release/version tree.'
        ),
    )

    parser.add_argument(
        '--worldpop-release',
        type=str,
        default=None,
        help='WorldPop Global2 release, for example R2025A.',
    )

    parser.add_argument(
        '--worldpop-version',
        type=str,
        default=None,
        help='WorldPop Global2 version, for example v1.',
    )

    parser.add_argument(
        '--worldpop-resolution',
        choices=('100m', '1km'),
        default=None,
        help='WorldPop Global2 resolution.',
    )

    parser.add_argument(
        '--worldpop-constrained',
        choices=('true', 'false'),
        default=None,
        help='Use constrained WorldPop Global2 population counts when true.',
    )

    parser.add_argument(
        '--worldpop-filename',
        type=str,
        default=None,
        help='Explicit WorldPop raster filename. Overrides generated filenames.',
    )

    parser.add_argument(
        '--worldpop-url',
        type=str,
        default=None,
        help='Explicit WorldPop raster download URL.',
    )

    parser.add_argument(
        '--worldpop-path',
        type=str,
        default=None,
        help='Explicit local WorldPop raster path. If present, download is skipped.',
    )

    parser.add_argument(
        '--pbf-filename',
        type=str,
        default=None,
        help=(
            'Explicit OSM PBF cache filename. Defaults to the country Geofabrik '
            'latest extract filename.'
        ),
    )

    parser.add_argument(
        '--pbf-url',
        type=str,
        default=None,
        help=(
            'Explicit OSM PBF download URL. If --pbf-filename is omitted, the '
            'filename is derived from the URL path.'
        ),
    )

    aggregate_group = parser.add_mutually_exclusive_group()

    aggregate_group.add_argument(
        '--aggregate-factor',
        type=int,
        default=None,
        help=(
            'Optional raster aggregation factor for population cells. '
            'Overrides the country config when provided.'
        ),
    )

    aggregate_group.add_argument(
        '--no-aggregate',
        action='store_true',
        help='Disable population raster aggregation, even if set in the country config.',
    )

    parser.add_argument(
        '--candidate-grid-spacing-m',
        type=float,
        default=None,
        help=(
            'Optional grid spacing for candidate facilities, in meters. '
            'Defaults to the country config when available.'
        ),
    )

    parser.add_argument(
        '--candidate-max-snap-dist-m',
        type=float,
        default=None,
        help='Optional maximum node snapping distance for candidate facilities, in meters.',
    )

    parser.add_argument(
        '--candidate-exclude-water',
        choices=('true', 'false'),
        default=None,
        help=(
            'Override whether generated candidate sites on OSM water bodies are '
            'removed. Defaults to the country config.'
        ),
    )

    parser.add_argument(
        '--amenity',
        nargs='+',
        default=None,
        help='Filter facilities by amenity values, for example hospital clinic.',
    )

    parser.add_argument(
        '--source-layer',
        choices=('amenity', 'table'),
        default='amenity',
        help=(
            'Legacy single-source-layer option. Prefer --sources. '
            'Use amenity for OSM amenities plus configured candidates, '
            'or table for a user-provided point table only.'
        ),
    )

    parser.add_argument(
        '--sources',
        nargs='+',
        default=None,
        help=(
            'Composable source layers. Supported values/aliases: '
            'population, amenities, table, candidates. '
            'Default: amenities candidates.'
        ),
    )

    parser.add_argument(
        '--destinations',
        nargs='+',
        default=None,
        help=(
            'Composable destination layers. Supported values/aliases: '
            'population, amenities, table, candidates. Default: population.'
        ),
    )

    parser.add_argument(
        '--source-table',
        type=str,
        default=None,
        help='CSV, Excel, parquet, or GIS point table used when sources include table.',
    )

    parser.add_argument(
        '--source-lon-column',
        type=str,
        default=None,
        help='Longitude column in --source-table. Auto-detected when omitted.',
    )

    parser.add_argument(
        '--source-lat-column',
        type=str,
        default=None,
        help='Latitude column in --source-table. Auto-detected when omitted.',
    )

    parser.add_argument(
        '--source-id-column',
        type=str,
        default=None,
        help='Optional stable ID column in --source-table.',
    )

    parser.add_argument(
        '--destination-table',
        type=str,
        default=None,
        help=(
            'CSV, Excel, parquet, or GIS point table used when destinations '
            'include table. Defaults to --source-table when omitted.'
        ),
    )

    parser.add_argument(
        '--destination-lon-column',
        type=str,
        default=None,
        help='Longitude column in --destination-table. Auto-detected when omitted.',
    )

    parser.add_argument(
        '--destination-lat-column',
        type=str,
        default=None,
        help='Latitude column in --destination-table. Auto-detected when omitted.',
    )

    parser.add_argument(
        '--destination-id-column',
        type=str,
        default=None,
        help='Optional stable ID column in --destination-table.',
    )

    parser.add_argument(
        '--deduplicate-amenities',
        choices=('true', 'false'),
        default='true',
        help=(
            'Whether to deduplicate nearby OSM amenity point/polygon duplicates. '
            'Defaults to true.'
        ),
    )

    parser.add_argument(
        '--matrix-output-mode',
        choices=('combined', 'split', 'both'),
        default='combined',
        help=(
            'How to persist the distance matrix. combined writes one matrix '
            'with source_type and target_type columns. split writes one '
            'matrix per source_type/target_type pair. both writes both forms.'
        ),
    )

    parser.add_argument(
        '--matrix-shape',
        choices=('sparse', 'dense'),
        default='sparse',
        help=(
            'Shape of the distance matrix output. sparse writes the usual '
            'long table of finite paths. dense writes a target-by-source '
            'matrix with inf for unreachable paths.'
        ),
    )

    parser.add_argument(
        '--dense-component-matrices',
        choices=('true', 'false'),
        default='false',
        help=(
            'When --matrix-shape dense is used, also write dense matrices for '
            'origin stitch distance, destination stitch distance, and road '
            'routing distance.'
        ),
    )

    parser.add_argument(
        '--network-impedance',
        default='length',
        help=(
            'Pandana edge-weight column used for routing. The default length '
            'preserves shortest-distance behavior. Use travel_time_s for '
            'fastest-route experiments with OSM-derived speeds.'
        ),
    )

    parser.add_argument(
        '--stitch-speed-kph',
        type=float,
        default=30.0,
        help=(
            'Speed used to convert straight-line source/target snap distances '
            'to seconds when --network-impedance is a time column. Default: 30.'
        ),
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce console output.',
    )

    return parser

# ------------------------
# Settings
# ------------------------
def settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    '''Build pipeline settings from parsed CLI arguments.'''
    if args.population_threshold < 0:
        raise ValueError('--population-threshold must be non negative.')
    if not 0 < args.sample_fraction <= 1:
        raise ValueError('--sample-fraction must be in (0, 1].')
    if args.max_points is not None and args.max_points <= 0:
        raise ValueError('--max-points must be positive.')
    if args.random_seed < 0:
        raise ValueError('--random-seed must be nonnegative.')
    if args.max_total_dist is not None and args.max_total_dist <= 0:
        raise ValueError('--max-total-dist must be positive.')
    if args.worldpop_year is not None and args.worldpop_year <= 0:
        raise ValueError('--worldpop-year must be positive.')
    if args.worldpop_url is not None and args.worldpop_path is not None:
        raise ValueError('--worldpop-url and --worldpop-path cannot be combined.')
    if args.pbf_filename is not None:
        validate_pbf_filename(args.pbf_filename)
    if args.pbf_url is not None and args.pbf_filename is None:
        pbf_filename_from_url(args.pbf_url)
    if args.aggregate_factor is not None and args.aggregate_factor < 2:
        raise ValueError('--aggregate-factor must be at least 2.')
    if args.candidate_grid_spacing_m is not None and args.candidate_grid_spacing_m <= 0:
        raise ValueError('--candidate-grid-spacing-m must be positive.')
    if args.candidate_max_snap_dist_m is not None and args.candidate_max_snap_dist_m <= 0:
        raise ValueError('--candidate-max-snap-dist-m must be positive.')
    if args.map_dpi <= 0:
        raise ValueError('--map-dpi must be positive.')
    if not 0 <= args.map_basemap_alpha <= 1:
        raise ValueError('--map-basemap-alpha must be between 0 and 1.')
    if not args.network_impedance or not args.network_impedance.strip():
        raise ValueError('--network-impedance must be a non-empty column name.')
    if args.stitch_speed_kph <= 0:
        raise ValueError('--stitch-speed-kph must be positive.')
    if args.network_profile != 'driving' and args.network_backend == 'pyrosm':
        raise ValueError(
            '--network-profile driving_walk requires --network-backend osmium or auto.'
        )

    source_layers = resolve_source_layers_from_args(args)
    destination_layers = resolve_destination_layers_from_args(args)
    ensure_table_arguments(
        source_layers=source_layers,
        destination_layers=destination_layers,
        source_table=args.source_table,
        destination_table=args.destination_table,
    )

    return PipelineSettings(
        population_threshold=args.population_threshold,
        sample_fraction=args.sample_fraction,
        max_points=args.max_points,
        random_seed=args.random_seed,
        max_total_dist=args.max_total_dist,
        candidate_grid_spacing_m=args.candidate_grid_spacing_m,
        candidate_max_snap_dist_m=args.candidate_max_snap_dist_m,
        candidate_exclude_water=(
            None
            if args.candidate_exclude_water is None
            else args.candidate_exclude_water == 'true'
        ),
        deduplicate_amenities=args.deduplicate_amenities == 'true',
        force_recompute=args.force_recompute,
        verbose=not args.quiet,
        save_context_map=args.save_map or args.map_only,
        show_context_map=args.show_map,
        context_map_path=None if args.map_path is None else Path(args.map_path),
        context_map_dpi=args.map_dpi,
        context_map_basemap=args.map_basemap,
        context_map_basemap_alpha=args.map_basemap_alpha,
        context_map_roads=args.map_roads == 'true',
        context_map_legend_loc=args.map_legend_loc,
        context_map_legend_bbox_to_anchor=(
            None
            if args.map_legend_bbox_to_anchor is None
            else tuple(args.map_legend_bbox_to_anchor)
        ),
        bbox=parse_bbox(args.bbox),
        matrix_output_mode=args.matrix_output_mode,
        matrix_shape=args.matrix_shape,
        dense_component_matrices=args.dense_component_matrices == 'true',
        network_impedance=args.network_impedance.strip(),
        stitch_speed_kph=args.stitch_speed_kph,
        network_backend=args.network_backend,
        simplify_network=args.simplify_network == 'true',
        network_profile=args.network_profile,
        diagnose_connectivity=args.diagnose_connectivity == 'true',
        snap_components=parse_snap_components(args.snap_components),
    )


def with_target_type(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    '''Attach a stable destination-layer label to a target table.'''
    result = df.copy()
    result['target_type'] = target_type
    return result


def safe_layer_name(value: object) -> str:
    '''Return a filename-safe layer label.'''
    text = str(value).strip().lower()
    if not text:
        return 'unknown'
    return ''.join(ch if ch.isalnum() else '_' for ch in text).strip('_') or 'unknown'


def matrix_cost_options(settings: PipelineSettings) -> dict[str, object]:
    '''Return routing impedance and output-column options for matrix builders.'''
    impedance = settings.network_impedance
    if impedance in {'length', 'length_m'}:
        return {
            'imp_name': impedance,
            'road_cost_col': 'road_distance',
            'total_cost_col': 'total_dist',
            'road_matrix_name': 'road_distance',
            'stitch_cost_factor': 1.0,
        }

    safe_impedance = safe_layer_name(impedance)
    return {
        'imp_name': impedance,
        'road_cost_col': f'road_{safe_impedance}',
        'total_cost_col': f'total_{safe_impedance}',
        'road_matrix_name': f'road_{safe_impedance}',
        'stitch_cost_factor': 3.6 / settings.stitch_speed_kph,
    }


def network_cache_key(settings: PipelineSettings) -> str:
    '''Return the cache profile key for network-derived artifacts.'''
    if settings.network_profile == 'driving':
        return settings.network_backend
    return f'{settings.network_backend}_{settings.network_profile}'


def split_matrix_output_key(source_type: object, target_type: object) -> str:
    '''Return the manifest key for a split matrix output.'''
    return (
        f'distance_matrix_src_{safe_layer_name(source_type)}_'
        f'dst_{safe_layer_name(target_type)}'
    )


def short_output_path(path: Path) -> Path:
    '''Shorten long output filenames deterministically for Windows compatibility.'''
    if len(str(path)) <= 240 and len(path.name) <= 180:
        return path

    digest = hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]
    suffix = path.suffix
    max_name_len = 120
    stem_limit = max_name_len - len(suffix) - len(digest) - 1
    stem = path.stem
    if '_pbf_' in stem:
        pbf_tail = stem[stem.rfind('_pbf_'):]
        head_limit = stem_limit - len(pbf_tail)
        if head_limit >= 24:
            readable_stem = f'{stem[:head_limit]}{pbf_tail}'
        else:
            readable_stem = stem[:max(24, stem_limit)]
    else:
        readable_stem = stem[:max(24, stem_limit)]
    return path.with_name(f'{readable_stem}_{digest}{suffix}')


def split_matrix_path(
    output_dir: Path,
    run_tag: str,
    source_type: object,
    target_type: object,
) -> Path:
    '''Return the output parquet path for one source/destination pair.'''
    return short_output_path(
        output_dir / f'{split_matrix_output_key(source_type, target_type)}_{run_tag}.parquet'
    )


def dense_matrix_output_key(component: str) -> str:
    '''Return the manifest key for a dense matrix component.'''
    return f'dense_matrix_{safe_layer_name(component)}'


def dense_matrix_path(output_dir: Path, run_tag: str, component: str) -> Path:
    '''Return the parquet path for one dense matrix component.'''
    return short_output_path(
        output_dir / f'distance_matrix_dense_{safe_layer_name(component)}_{run_tag}.parquet'
    )


def split_dense_matrix_output_key(
    component: str,
    source_type: object,
    target_type: object,
) -> str:
    '''Return the manifest key for one split dense matrix component.'''
    return (
        f'dense_matrix_{safe_layer_name(component)}_'
        f'src_{safe_layer_name(source_type)}_dst_{safe_layer_name(target_type)}'
    )


def split_dense_matrix_path(
    output_dir: Path,
    run_tag: str,
    component: str,
    source_type: object,
    target_type: object,
) -> Path:
    '''Return the parquet path for one split dense matrix component.'''
    return short_output_path(
        output_dir
        / f'{split_dense_matrix_output_key(component, source_type, target_type)}_{run_tag}.parquet'
    )


def write_parquet_table(table: object, path: Path) -> None:
    '''Write a pandas or Polars table to parquet.'''
    if pl is not None and isinstance(table, pl.DataFrame):
        table.write_parquet(path)
    else:
        table.to_parquet(path, index=False)


def table_row_count(table: object) -> int:
    '''Return row count for a pandas or Polars table.'''
    if pl is not None and isinstance(table, pl.DataFrame):
        return table.height
    return len(table)


def table_head(table: object) -> object:
    '''Return a small preview for a pandas or Polars table.'''
    return table.head()


def compute_network_component_labels(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    verbose: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    '''Compute weak connected-component labels for the loaded road graph.'''
    t0 = pc()
    node_ids = nodes.index.to_numpy(dtype='int64', copy=False)
    n_nodes = len(node_ids)
    node_pos = pd.Series(range(n_nodes), index=node_ids, dtype='int64')

    edge_nodes = edges[['u', 'v']].copy()
    edge_nodes = edge_nodes.loc[
        edge_nodes['u'].isin(node_pos.index) & edge_nodes['v'].isin(node_pos.index)
    ]
    u_pos = node_pos.reindex(edge_nodes['u']).to_numpy(dtype='int64', copy=False)
    v_pos = node_pos.reindex(edge_nodes['v']).to_numpy(dtype='int64', copy=False)

    parent = np.arange(n_nodes, dtype=np.int64)
    size = np.ones(n_nodes, dtype=np.int64)

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = int(parent[idx])
        return int(idx)

    for u, v in zip(u_pos, v_pos):
        root_u = find(int(u))
        root_v = find(int(v))
        if root_u == root_v:
            continue
        if size[root_u] < size[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        size[root_u] += size[root_v]

    roots = np.fromiter((find(i) for i in range(n_nodes)), dtype=np.int64, count=n_nodes)
    root_counts = pd.Series(roots).value_counts(sort=True)
    root_to_component = {
        root: component_id
        for component_id, root in enumerate(root_counts.index.to_list())
    }
    component_ids = pd.Series(
        [root_to_component[root] for root in roots],
        index=node_ids,
        dtype='int64',
        name='component_id',
    )
    component_summary = pd.DataFrame(
        {
            'component_id': range(len(root_counts)),
            'node_count': root_counts.to_numpy(dtype='int64'),
        }
    )

    if verbose:
        top_sizes = ', '.join(
            f'{row.component_id}:{row.node_count:,}'
            for row in component_summary.head(10).itertuples(index=False)
        )
        logging.info(
            'Road connectivity diagnostic: %s weak component(s) across %s nodes; '
            'largest component sizes %s',
            f'{len(component_summary):,}',
            f'{n_nodes:,}',
            top_sizes,
        )
        logging.info(
            'Connectivity diagnostic completed in %.2fs',
            pc() - t0,
        )

    return component_ids, component_summary


def add_component_labels(
    table: pd.DataFrame,
    component_ids: pd.Series,
    *,
    nearest_node_col: str = 'nearest_node',
) -> pd.DataFrame:
    '''Attach a road-network weak-component ID to a snapped point table.'''
    result = table.copy()
    if nearest_node_col not in result.columns:
        result['component_id'] = -1
        return result
    labels = component_ids.reindex(result[nearest_node_col].astype('int64'))
    result['component_id'] = labels.fillna(-1).to_numpy(dtype='int64')
    return result


def filter_nodes_to_components(
    nodes: gpd.GeoDataFrame,
    component_ids: pd.Series,
    allowed_components: tuple[int, ...],
    *,
    verbose: bool,
) -> gpd.GeoDataFrame:
    '''Return network nodes whose weak component ID is explicitly allowed.'''
    allowed = set(allowed_components)
    allowed_node_ids = component_ids[component_ids.isin(allowed)].index
    filtered = nodes.loc[nodes.index.intersection(allowed_node_ids)].copy()

    if filtered.empty:
        allowed_str = ', '.join(str(component_id) for component_id in allowed_components)
        raise ValueError(
            'No network nodes remain after --snap-components filtering. '
            f'Requested component IDs: {allowed_str}.'
        )

    if verbose:
        logging.info(
            'Restricting snapping to component IDs %s: %s of %s nodes retained',
            ','.join(str(component_id) for component_id in allowed_components),
            f'{len(filtered):,}',
            f'{len(nodes):,}',
        )

    return filtered


def write_matrix_outputs(
    *,
    matrix_df: object,
    output_dir: Path,
    run_tag: str,
    mode: str,
    combined_path: Path,
    verbose: bool,
) -> dict[str, Path]:
    '''Write combined and/or source-target split matrix parquet outputs.'''
    output_paths: dict[str, Path] = {}

    if mode in {'combined', 'both'}:
        write_parquet_table(matrix_df, combined_path)
        output_paths['distance_matrix'] = combined_path

    if mode in {'split', 'both'}:
        if pl is not None and isinstance(matrix_df, pl.DataFrame):
            if 'source_type' not in matrix_df.columns or 'target_type' not in matrix_df.columns:
                raise ValueError(
                    'Split matrix output requires source_type and target_type columns.'
                )
            pairs = (
                matrix_df.select(['source_type', 'target_type'])
                .unique()
                .sort(['source_type', 'target_type'])
                .iter_rows()
            )
            for source_type, target_type in pairs:
                split_path = split_matrix_path(output_dir, run_tag, source_type, target_type)
                (
                    matrix_df
                    .filter(
                        (pl.col('source_type') == source_type)
                        & (pl.col('target_type') == target_type)
                    )
                    .write_parquet(split_path)
                )
                output_paths[split_matrix_output_key(source_type, target_type)] = split_path
                if verbose:
                    logging.info(
                        'Wrote split distance matrix %s -> %s: %s',
                        source_type,
                        target_type,
                        split_path,
                    )
        else:
            if 'source_type' not in matrix_df.columns or 'target_type' not in matrix_df.columns:
                raise ValueError(
                    'Split matrix output requires source_type and target_type columns.'
                )
            for (source_type, target_type), split_df in matrix_df.groupby(
                ['source_type', 'target_type'],
                observed=True,
            ):
                split_path = split_matrix_path(output_dir, run_tag, source_type, target_type)
                write_parquet_table(split_df, split_path)
                output_paths[split_matrix_output_key(source_type, target_type)] = split_path
                if verbose:
                    logging.info(
                        'Wrote split distance matrix %s -> %s: %s',
                        source_type,
                        target_type,
                        split_path,
                    )

    return output_paths


def write_dense_matrix_outputs(
    *,
    dense_matrices: dict[str, pd.DataFrame],
    output_dir: Path,
    run_tag: str,
    include_components: bool,
    road_component: str = 'road_distance',
    verbose: bool,
) -> dict[str, Path]:
    '''Write dense total and optional component matrices.'''
    components = ['total']
    if include_components:
        components.extend(['origin_stitch', 'destination_stitch', road_component])

    output_paths: dict[str, Path] = {}
    for component in components:
        path = dense_matrix_path(output_dir, run_tag, component)
        write_parquet_table(dense_matrices[component], path)
        output_paths[dense_matrix_output_key(component)] = path
        if component == 'total':
            output_paths['distance_matrix'] = path
        if verbose:
            logging.info('Wrote dense %s matrix: %s', component, path)

    return output_paths


def write_split_dense_matrix_outputs(
    *,
    dense_matrices: dict[str, pd.DataFrame],
    output_dir: Path,
    run_tag: str,
    source_type: object,
    target_type: object,
    include_components: bool,
    road_component: str = 'road_distance',
    verbose: bool,
) -> dict[str, Path]:
    '''Write dense matrices for one source/destination layer pair.'''
    components = ['total']
    if include_components:
        components.extend(['origin_stitch', 'destination_stitch', road_component])

    output_paths: dict[str, Path] = {}
    for component in components:
        path = split_dense_matrix_path(
            output_dir,
            run_tag,
            component,
            source_type,
            target_type,
        )
        write_parquet_table(dense_matrices[component], path)
        key = split_dense_matrix_output_key(component, source_type, target_type)
        output_paths[key] = path
        if component == 'total':
            output_paths[split_matrix_output_key(source_type, target_type)] = path
        if verbose:
            logging.info(
                'Wrote split dense %s matrix %s -> %s: %s',
                component,
                source_type,
                target_type,
                path,
            )

    return output_paths


def resolve_aggregate_factor(
    *,
    cfg: CountryConfig,
    aggregate_factor: int | None,
    no_aggregate: bool,
) -> int | None:
    '''Resolve the effective population raster aggregation factor.'''
    if no_aggregate:
        return None
    if aggregate_factor is not None:
        return aggregate_factor
    return cfg.aggregate_factor


# ------------------------
# MAIN
# ------------------------
def main(
    cfg: CountryConfig,
    settings: PipelineSettings,
    aggregate_factor: int | None,
    no_aggregate: bool,
    build_map: bool,
    map_only: bool,
    amenity_values: list[str] | None,
    source_layers: list[str],
    destination_layers: list[str],
    source_table: str | None,
    source_lon_column: str | None,
    source_lat_column: str | None,
    source_id_column: str | None,
    destination_table: str | None,
    destination_lon_column: str | None,
    destination_lat_column: str | None,
    destination_id_column: str | None,
    *,
    base_cfg: CountryConfig | None = None,
) -> None:
    '''Run the pipeline for a given country.'''
    t_total = pc()

    agg = resolve_aggregate_factor(
        cfg=cfg,
        aggregate_factor=aggregate_factor,
        no_aggregate=no_aggregate,
    )

    cache = CacheManager(
        cfg=cfg,
        force_recompute=settings.force_recompute,
        verbose=settings.verbose,
    )
    network_cache_backend = settings.network_cache_backend()

    ensure_table_arguments(
        source_layers=source_layers,
        destination_layers=destination_layers,
        source_table=source_table,
        destination_table=destination_table,
    )
    source_table_path = None if source_table is None else Path(source_table)
    destination_table_path = (
        None
        if destination_table is None and source_table is None
        else Path(destination_table or source_table)
    )
    source_cache_values = layer_signature(
        source_layers=source_layers,
        destination_layers=destination_layers,
        amenity_values=amenity_values,
        source_table=source_table,
        destination_table=destination_table,
    )
    needs_context_map = (
        build_map or settings.save_context_map or settings.show_context_map or map_only
    )
    needs_context_roads = needs_context_map and settings.context_map_roads

    if settings.verbose:
        logging.info(f'Running pipeline for {cfg.COUNTRY_NAME}')
        logging.info(f'Aggregate factor: {agg}')
        logging.info(f'Source layers: {source_layers}')
        logging.info(f'Destination layers: {destination_layers}')
        logging.info(f'Amenity filter: {amenity_values}')
        logging.info(f'Network backend: {settings.network_backend}')
        logging.info(f'Network profile: {settings.network_profile}')

    cfg.BASE_DIR.mkdir(parents=True, exist_ok=True)

    download_file(cfg.PBF_URL, cfg.PBF_PATH, overwrite=False, verbose=settings.verbose)
    if cfg.worldpop_path is not None:
        if not cfg.WORLDPOP_PATH.exists():
            raise FileNotFoundError(
                f'Configured --worldpop-path does not exist: {cfg.WORLDPOP_PATH}'
            )
        if settings.verbose:
            logging.info(f'Using local WorldPop raster: {cfg.WORLDPOP_PATH}')
    else:
        download_file(
            cfg.WORLDPOP_URL,
            cfg.WORLDPOP_PATH,
            overwrite=False,
            verbose=settings.verbose,
        )

    nodes = None
    edges = None
    if map_only and not needs_context_roads:
        roads = empty_roads_layer()
    elif map_only:
        roads = cache.run(
            cache_path=cache.roads_path(
                bbox=settings.bbox,
                network_backend=network_cache_backend,
            ),
            builder=lambda: classify_roads(
                load_osm_road_edges(
                    cfg.PBF_PATH,
                    verbose=settings.verbose,
                    bbox=settings.bbox,
                    backend=settings.network_backend,
                    network_profile=settings.network_profile,
                ),
                verbose=settings.verbose,
            ),
        )
        roads = filter_to_bbox(roads, settings.bbox)
    else:
        nodes, edges = cache.load_or_build_network_data(
            builder=lambda: load_osm_network_data(
                cfg.PBF_PATH,
                verbose=settings.verbose,
                bbox=settings.bbox,
                backend=settings.network_backend,
                simplify=settings.simplify_network,
                network_profile=settings.network_profile,
            ),
            bbox=settings.bbox,
            network_backend=network_cache_backend,
        )
        if needs_context_roads:
            roads = cache.run(
                cache_path=cache.roads_path(
                    bbox=settings.bbox,
                    network_backend=network_cache_backend,
                ),
                builder=lambda: classify_roads(
                    load_osm_road_edges(
                        cfg.PBF_PATH,
                        verbose=settings.verbose,
                        bbox=settings.bbox,
                        backend=settings.network_backend,
                        network_profile=settings.network_profile,
                    ),
                    verbose=settings.verbose,
                ),
            )
            roads = filter_to_bbox(roads, settings.bbox)
        else:
            roads = empty_roads_layer()

    population_points = cache.run(
        cache_path=cache.population_points_path(
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            random_seed=settings.random_seed,
            aggregate_factor=agg,
        ),
        builder=lambda: worldpop_to_points(
            cfg.WORLDPOP_PATH,
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            random_seed=settings.random_seed,
            aggregate_factor=agg,
            verbose=settings.verbose,
        ),
    )
    population_points = filter_to_bbox(population_points, settings.bbox)

    needs_amenities = (
        'amenities' in source_layers
        or 'amenities' in destination_layers
    )
    needs_candidates = (
        'candidates' in source_layers
        or 'candidates' in destination_layers
    )

    amenity_points = None
    if needs_amenities:
        amenity_features = cache.run(
            cache_path=cache.facilities_path(
                amenity_values=amenity_values,
                bbox=settings.bbox,
                network_backend=network_cache_backend,
            ),
            builder=lambda: load_facilities(
                cfg.PBF_PATH,
                amenity_values=amenity_values,
                backend=settings.network_backend,
                bbox=settings.bbox,
                verbose=settings.verbose,
            ),
        )

        amenity_points = cache.run(
            cache_path=cache.facility_points_path(
                amenity_values=amenity_values,
                deduplicate_amenities=settings.deduplicate_amenities,
                bbox=settings.bbox,
                network_backend=network_cache_backend,
            ),
            builder=lambda: (
                deduplicate_osm_amenities(
                        to_point_geometries(
                            amenity_features,
                            projected_epsg=cfg.PROJECTED_EPSG,
                            verbose=settings.verbose,
                        ),
                    projected_epsg=cfg.PROJECTED_EPSG,
                    verbose=settings.verbose,
                )
                if settings.deduplicate_amenities
                else to_point_geometries(
                    amenity_features,
                    projected_epsg=cfg.PROJECTED_EPSG,
                    verbose=settings.verbose,
                )
            ),
        )
        amenity_points = filter_to_bbox(amenity_points, settings.bbox)

    source_table_points = None
    if 'table' in source_layers:
        source_table_points = load_custom_points_table(
            source_table_path,
            lon_column=source_lon_column,
            lat_column=source_lat_column,
            id_column=source_id_column,
        )
        source_table_points = to_point_geometries(
            source_table_points,
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        )
        source_table_points = filter_to_bbox(source_table_points, settings.bbox)

    destination_table_points = None
    if 'table' in destination_layers:
        destination_table_points = load_custom_points_table(
            destination_table_path,
            lon_column=destination_lon_column or source_lon_column,
            lat_column=destination_lat_column or source_lat_column,
            id_column=destination_id_column or source_id_column,
        )
        destination_table_points = to_point_geometries(
            destination_table_points,
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        )
        destination_table_points = filter_to_bbox(destination_table_points, settings.bbox)

    if settings.verbose:
        logging.info(f'Population points: {len(population_points):,}')
        if amenity_points is not None:
            logging.info(f'OSM amenity points: {len(amenity_points):,}')
        if source_table_points is not None:
            logging.info(f'Source table points: {len(source_table_points):,}')
        if destination_table_points is not None:
            logging.info(
                f'Destination table points: {len(destination_table_points):,}'
            )

    component_ids = None
    component_summary = None
    snap_nodes = nodes
    if not map_only and (
        settings.diagnose_connectivity or settings.snap_components is not None
    ):
        component_ids, computed_component_summary = compute_network_component_labels(
            nodes,
            edges,
            verbose=settings.verbose,
        )
        if settings.snap_components is not None:
            snap_nodes = filter_nodes_to_components(
                nodes,
                component_ids,
                settings.snap_components,
                verbose=settings.verbose,
            )
        if settings.diagnose_connectivity:
            component_summary = computed_component_summary

    if needs_candidates:
        if map_only:
            candidate_sites = build_candidate_grid(
                cfg=cfg,
                settings=settings,
                cache=cache,
            )
            candidate_sites = filter_to_bbox(candidate_sites, settings.bbox)
            candidate_sites_snapped = None
        else:
            candidate_sites, candidate_sites_snapped = build_candidate_sites(
                cfg=cfg,
                settings=settings,
                cache=cache,
                nodes=snap_nodes,
            )
            candidate_sites = filter_to_bbox(candidate_sites, settings.bbox)
            candidate_sites_snapped = filter_to_bbox(candidate_sites_snapped, settings.bbox)
        candidate_grid_spacing_m = resolve_candidate_grid_spacing(cfg, settings)
        candidate_max_snap_dist_m = resolve_candidate_max_snap_dist(cfg, settings)
    else:
        candidate_sites = None
        candidate_sites_snapped = None
        candidate_grid_spacing_m = None
        candidate_max_snap_dist_m = None

    # -------- MAP (optional) --------
    if needs_context_map:
        map_base_layers = [
            layer
            for layer in [
                amenity_points,
                source_table_points,
                destination_table_points,
            ]
            if layer is not None
        ]
        map_base = (
            pd.concat(map_base_layers, axis=0, sort=False)
            if map_base_layers
            else population_points
        )
        map_candidates = (
            candidate_sites if map_only else candidate_sites_snapped
            if 'candidates' in source_layers or 'candidates' in destination_layers
            else None
        )
        map_facilities = build_map_facilities(map_base, map_candidates)
        context_map_path = build_context_map_path(
            cache.context_map_path(),
            settings.context_map_path,
            resolve_candidate_grid_spacing(cfg, settings),
        )
        if settings.network_profile != 'driving':
            context_map_path = context_map_path.with_name(
                f'{context_map_path.stem}_network_{safe_layer_name(settings.network_profile)}'
                f'{context_map_path.suffix}'
            )
        plot_context_map(
            roads=roads,
            population_points=population_points,
            facilities=map_facilities,
            title=cfg.PLOT_TITLE,
            legend_loc=settings.context_map_legend_loc,
            legend_bbox_to_anchor=settings.context_map_legend_bbox_to_anchor,
            basemap_provider=settings.context_map_basemap,
            basemap_alpha=settings.context_map_basemap_alpha,
            output_path=context_map_path if settings.save_context_map else None,
            dpi=settings.context_map_dpi,
            show=settings.show_context_map,
            verbose=settings.verbose,
        )

        if map_only:
            if settings.verbose:
                logging.info('Map-only run completed before matrix computation.')
                logging.info(f'Total runtime: {pc() - t_total:.2f}s')
            return

    if settings.verbose:
        logging.info(
            'Building Pandana network from %s nodes and %s edges',
            f'{len(nodes):,}',
            f'{len(edges):,}',
        )
    t_pandana = pc()
    edges = add_edge_speeds(edges)
    if settings.network_impedance not in edges.columns:
        raise ValueError(
            f'--network-impedance {settings.network_impedance!r} is not available '
            f'on the edge table. Available columns include: {sorted(edges.columns)}'
        )
    network_weight_cols = [
        'length',
        'length_m',
        'travel_time_s',
        settings.network_impedance,
    ]
    network_weight_cols = list(dict.fromkeys(network_weight_cols))
    network = build_pandana_network(
        nodes,
        edges,
        weight_cols=network_weight_cols,
    )
    cost_options = matrix_cost_options(settings)
    if settings.verbose:
        logging.info(f'Built Pandana network in {pc() - t_pandana:.2f}s')

    source_parts: list[pd.DataFrame] = []
    fixed_source_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    population = None

    if 'population' in destination_layers:
        population = cache.run(
            cache_path=cache.population_snapped_path_for(
                distance_col='dist_snap_target',
                population_threshold=settings.population_threshold,
                sample_fraction=settings.sample_fraction,
                max_points=settings.max_points,
                random_seed=settings.random_seed,
                aggregate_factor=agg,
                snap_components=settings.snap_components,
                network_backend=network_cache_backend,
            ),
            builder=lambda: snap_points_to_nodes(
                population_points,
                snap_nodes,
                distance_col='dist_snap_target',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
        )
        population = with_prefixed_ids(population, 'target_population')
        target_parts.append(
            with_target_type(
                prepare_points_as_targets(population, id_prefix='target'),
                'population',
            )
        )

    if 'population' in source_layers:
        source_population = cache.run(
            cache_path=cache.population_snapped_path_for(
                distance_col='dist_snap_source',
                population_threshold=settings.population_threshold,
                sample_fraction=settings.sample_fraction,
                max_points=settings.max_points,
                random_seed=settings.random_seed,
                aggregate_factor=agg,
                snap_components=settings.snap_components,
                network_backend=network_cache_backend,
            ),
            builder=lambda: snap_points_to_nodes(
                population_points,
                snap_nodes,
                distance_col='dist_snap_source',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
        )
        source_population = with_prefixed_ids(source_population, 'source_population')
        source_population = prepare_points_as_sources(
            source_population,
            source_type='population',
            id_prefix='source_population',
        )
        source_parts.append(source_population)
        fixed_source_parts.append(source_population)

    if 'amenities' in destination_layers:
        target_amenities = cache.run(
            cache_path=cache.sources_snapped_path_for(
                distance_col='dist_snap_target',
                amenity_values=['target_amenities', *source_cache_values],
                snap_components=settings.snap_components,
                network_backend=network_cache_backend,
            ),
            builder=lambda: snap_points_to_nodes(
                amenity_points,
                snap_nodes,
                distance_col='dist_snap_target',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
        )
        target_amenities = with_prefixed_ids(target_amenities, 'target_amenities')
        target_parts.append(
            with_target_type(
                prepare_points_as_targets(target_amenities, id_prefix='target'),
                'amenities',
            )
        )

    if 'amenities' in source_layers:
        source_amenities = cache.run(
            cache_path=cache.sources_snapped_path_for(
                distance_col='dist_snap_source',
                amenity_values=['source_amenities', *source_cache_values],
                snap_components=settings.snap_components,
                network_backend=network_cache_backend,
            ),
            builder=lambda: snap_points_to_nodes(
                amenity_points,
                snap_nodes,
                distance_col='dist_snap_source',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
        )
        source_amenities = with_prefixed_ids(source_amenities, 'source_amenities')
        source_amenities = prepare_points_as_sources(
            source_amenities,
            source_type='amenities',
            id_prefix='source_amenities',
        )
        source_parts.append(source_amenities)
        fixed_source_parts.append(source_amenities)

    if 'table' in destination_layers:
        target_table = cache.run(
            cache_path=cache.sources_snapped_path_for(
                distance_col='dist_snap_target',
                amenity_values=[
                    'target_table',
                    table_descriptor(destination_table_path),
                    *source_cache_values,
                ],
                snap_components=settings.snap_components,
                network_backend=network_cache_backend,
            ),
            builder=lambda: snap_points_to_nodes(
                destination_table_points,
                snap_nodes,
                distance_col='dist_snap_target',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
        )
        target_table = with_prefixed_ids(target_table, 'target_table')
        target_parts.append(
            with_target_type(
                prepare_points_as_targets(target_table, id_prefix='target'),
                'table',
            )
        )

    if 'table' in source_layers:
        source_table_snapped = cache.run(
            cache_path=cache.sources_snapped_path_for(
                distance_col='dist_snap_source',
                amenity_values=[
                    'source_table',
                    table_descriptor(source_table_path),
                    *source_cache_values,
                ],
                snap_components=settings.snap_components,
                network_backend=network_cache_backend,
            ),
            builder=lambda: snap_points_to_nodes(
                source_table_points,
                snap_nodes,
                distance_col='dist_snap_source',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
        )
        source_table_snapped = with_prefixed_ids(
            source_table_snapped,
            'source_table',
        )
        source_table_snapped = prepare_points_as_sources(
            source_table_snapped,
            source_type='table',
            id_prefix='source_table',
        )
        source_parts.append(source_table_snapped)
        fixed_source_parts.append(source_table_snapped)

    if 'candidates' in destination_layers and candidate_sites_snapped is not None:
        target_candidates = candidate_layer_for_role(
            candidate_sites_snapped,
            role='target',
            id_prefix='target_candidates',
        )
        target_parts.append(
            with_target_type(
                prepare_points_as_targets(target_candidates, id_prefix='target'),
                'candidates',
            )
        )

    if 'candidates' in source_layers and candidate_sites_snapped is not None:
        source_candidates = candidate_layer_for_role(
            candidate_sites_snapped,
            role='source',
            id_prefix='source_candidates',
        )
        source_candidates = prepare_points_as_sources(
            source_candidates,
            source_type='candidates',
            id_prefix='source_candidates',
        )
        source_parts.append(source_candidates)

    targets = concat_layers(target_parts, label='destination')
    sources = concat_layers(source_parts, label='source')
    existing_sources = (
        concat_layers(fixed_source_parts, label='non-candidate source')
        if fixed_source_parts
        else sources.iloc[0:0].copy()
    )

    if population is None:
        population = targets

    if settings.diagnose_connectivity and component_ids is not None:
        targets = add_component_labels(targets, component_ids)
        sources = add_component_labels(sources, component_ids)
        existing_sources = add_component_labels(existing_sources, component_ids)
        population = add_component_labels(population, component_ids)
        if settings.verbose:
            target_components = targets['component_id'].value_counts().sort_index()
            source_components = sources['component_id'].value_counts().sort_index()
            logging.info(
                'Destination rows by component_id: %s',
                target_components.to_dict(),
            )
            logging.info(
                'Source rows by component_id: %s',
                source_components.to_dict(),
            )

    effective_distance_threshold_km = cfg.DISTANCE_THRESHOLD_KM
    if settings.max_total_dist is not None:
        effective_distance_threshold_km = min(
            effective_distance_threshold_km,
            settings.max_total_dist / 1000.0,
        )
    if settings.verbose and effective_distance_threshold_km != cfg.DISTANCE_THRESHOLD_KM:
        logging.info(
            'Spatial prefilter threshold tightened from %.1f km to %.1f km by max_total_dist',
            cfg.DISTANCE_THRESHOLD_KM,
            effective_distance_threshold_km,
        )

    output_dir = cfg.BASE_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = build_output_run_tag(
        settings=settings,
        aggregate_factor=agg,
        amenity_values=source_cache_values,
        candidate_grid_spacing_m=candidate_grid_spacing_m,
        candidate_max_snap_dist_m=candidate_max_snap_dist_m,
        has_candidates=candidate_sites_snapped is not None,
        pbf_filename=pbf_filename_for_output_tag(cfg, base_cfg),
    )
    if settings.diagnose_connectivity:
        run_tag = f'{run_tag}_connectivity'

    population_path = short_output_path(output_dir / f'population_{run_tag}.parquet')
    targets_path = short_output_path(output_dir / f'targets_{run_tag}.parquet')
    existing_sources_path = short_output_path(output_dir / f'existing_sources_{run_tag}.parquet')
    sources_path = short_output_path(output_dir / f'sources_{run_tag}.parquet')
    connectivity_path = short_output_path(output_dir / f'connectivity_components_{run_tag}.parquet')
    matrix_path = short_output_path(output_dir / f'distance_matrix_{run_tag}.parquet')
    manifest_path = short_output_path(output_dir / f'run_manifest_{run_tag}.yaml')

    t_dist = pc()
    matrix_df: object | None = None
    matrix_preview: object | None = None
    distance_matrix_size = 0
    matrix_output_paths: dict[str, Path] = {}
    node_pair_cache_dir = cache.node_pair_distances_dir(
        bbox=settings.bbox,
        network_backend=network_cache_backend,
        cost_profile=settings.network_impedance,
    )
    if settings.verbose and settings.matrix_shape == 'sparse':
        logging.info(f'Node-pair distance cache: {node_pair_cache_dir}')

    if settings.matrix_shape == 'sparse':
        if settings.matrix_output_mode == 'combined':
            matrix_df = cache.run(
                cache_path=cache.distance_matrix_path_for(
                    distance_threshold_largest=effective_distance_threshold_km,
                    max_total_dist=settings.max_total_dist,
                    population_threshold=settings.population_threshold,
                    sample_fraction=settings.sample_fraction,
                    max_points=settings.max_points,
                    random_seed=settings.random_seed,
                    aggregate_factor=agg,
                    amenity_values=source_cache_values,
                    candidate_grid_spacing_m=candidate_grid_spacing_m,
                    candidate_max_snap_dist_m=candidate_max_snap_dist_m,
                    has_candidates=candidate_sites_snapped is not None,
                    snap_components=settings.snap_components,
                    network_backend=network_cache_backend,
                    cost_profile=settings.network_impedance,
                ),
                builder=lambda: compute_distances_polars(
                    targets=targets,
                    sources=sources,
                    distance_threshold_largest=effective_distance_threshold_km,
                    network=network,
                    max_total_dist=settings.max_total_dist,
                    node_pair_cache_dir=node_pair_cache_dir,
                    imp_name=cost_options['imp_name'],
                    road_cost_col=cost_options['road_cost_col'],
                    total_cost_col=cost_options['total_cost_col'],
                    stitch_cost_factor=cost_options['stitch_cost_factor'],
                    verbose=settings.verbose,
                ),
            )
            if pl is None or not isinstance(matrix_df, pl.DataFrame):
                matrix_df = set_known_categories(matrix_df)
            matrix_preview = table_head(matrix_df)
            distance_matrix_size = table_row_count(matrix_df)
            matrix_output_paths = write_matrix_outputs(
                matrix_df=matrix_df,
                output_dir=output_dir,
                run_tag=run_tag,
                mode='combined',
                combined_path=matrix_path,
                verbose=settings.verbose,
            )
        else:
            combined_parts: list[object] = []
            source_types = sorted(sources['source_type'].astype(str).unique())
            target_types = sorted(targets['target_type'].astype(str).unique())
            for source_type in source_types:
                pair_sources = sources[sources['source_type'].astype(str) == source_type]
                for target_type in target_types:
                    pair_targets = targets[
                        targets['target_type'].astype(str) == target_type
                    ]
                    if pair_sources.empty or pair_targets.empty:
                        continue
                    if settings.verbose:
                        logging.info(
                            'Computing split distance matrix %s -> %s (%s x %s)',
                            source_type,
                            target_type,
                            f'{len(pair_sources):,}',
                            f'{len(pair_targets):,}',
                        )
                    pair_has_candidates = (
                        source_type == 'candidates' or target_type == 'candidates'
                    )
                    pair_cache_values = [
                        *source_cache_values,
                        f'src_{source_type}',
                        f'dst_{target_type}',
                    ]
                    split_df = cache.run(
                        cache_path=cache.distance_matrix_path_for(
                            distance_threshold_largest=effective_distance_threshold_km,
                            max_total_dist=settings.max_total_dist,
                            population_threshold=settings.population_threshold,
                            sample_fraction=settings.sample_fraction,
                            max_points=settings.max_points,
                            random_seed=settings.random_seed,
                            aggregate_factor=agg,
                            amenity_values=pair_cache_values,
                            candidate_grid_spacing_m=candidate_grid_spacing_m,
                            candidate_max_snap_dist_m=candidate_max_snap_dist_m,
                            has_candidates=pair_has_candidates,
                            snap_components=settings.snap_components,
                            network_backend=network_cache_backend,
                            cost_profile=settings.network_impedance,
                        ),
                        builder=lambda pair_targets=pair_targets, pair_sources=pair_sources: (
                            compute_distances_polars(
                                targets=pair_targets,
                                sources=pair_sources,
                                distance_threshold_largest=effective_distance_threshold_km,
                                network=network,
                                max_total_dist=settings.max_total_dist,
                                node_pair_cache_dir=node_pair_cache_dir,
                                imp_name=cost_options['imp_name'],
                                road_cost_col=cost_options['road_cost_col'],
                                total_cost_col=cost_options['total_cost_col'],
                                stitch_cost_factor=cost_options['stitch_cost_factor'],
                                verbose=settings.verbose,
                            )
                        ),
                    )
                    if pl is None or not isinstance(split_df, pl.DataFrame):
                        split_df = set_known_categories(split_df)
                    if matrix_preview is None:
                        matrix_preview = table_head(split_df)
                    distance_matrix_size += table_row_count(split_df)
                    split_path = split_matrix_path(
                        output_dir,
                        run_tag,
                        source_type,
                        target_type,
                    )
                    write_parquet_table(split_df, split_path)
                    matrix_output_paths[
                        split_matrix_output_key(source_type, target_type)
                    ] = split_path
                    if settings.verbose:
                        logging.info(
                            'Wrote split distance matrix %s -> %s: %s',
                            source_type,
                            target_type,
                            split_path,
                        )
                    if settings.matrix_output_mode == 'both':
                        combined_parts.append(split_df)

            if settings.matrix_output_mode == 'both':
                if combined_parts:
                    if pl is not None and all(
                        isinstance(part, pl.DataFrame) for part in combined_parts
                    ):
                        matrix_df = pl.concat(combined_parts, how='vertical')
                    else:
                        matrix_df = pd.concat(combined_parts, axis=0, sort=False)
                    write_parquet_table(matrix_df, matrix_path)
                    matrix_output_paths = {
                        'distance_matrix': matrix_path,
                        **matrix_output_paths,
                    }
                else:
                    matrix_df = sources.iloc[0:0].copy()
            else:
                matrix_df = matrix_preview
    else:
        if settings.matrix_output_mode == 'combined':
            dense_matrices = cache.run(
                cache_path=cache.distance_matrix_path_for(
                    distance_threshold_largest=effective_distance_threshold_km,
                    max_total_dist=settings.max_total_dist,
                    population_threshold=settings.population_threshold,
                    sample_fraction=settings.sample_fraction,
                    max_points=settings.max_points,
                    random_seed=settings.random_seed,
                    aggregate_factor=agg,
                    amenity_values=[*source_cache_values, 'matrix_dense'],
                    candidate_grid_spacing_m=candidate_grid_spacing_m,
                    candidate_max_snap_dist_m=candidate_max_snap_dist_m,
                    has_candidates=candidate_sites_snapped is not None,
                    snap_components=settings.snap_components,
                    network_backend=network_cache_backend,
                    cost_profile=settings.network_impedance,
                ),
                builder=lambda: compute_dense_distance_matrices(
                    targets=targets,
                    sources=sources,
                    network=network,
                    max_total_dist=settings.max_total_dist,
                    imp_name=cost_options['imp_name'],
                    road_matrix_name=cost_options['road_matrix_name'],
                    stitch_cost_factor=cost_options['stitch_cost_factor'],
                    verbose=settings.verbose,
                ),
            )
            matrix_df = dense_matrices['total']
            matrix_preview = table_head(matrix_df)
            distance_matrix_size = len(targets) * len(sources)
            matrix_output_paths = write_dense_matrix_outputs(
                dense_matrices=dense_matrices,
                output_dir=output_dir,
                run_tag=run_tag,
                include_components=settings.dense_component_matrices,
                road_component=cost_options['road_matrix_name'],
                verbose=settings.verbose,
            )
        else:
            source_types = sorted(sources['source_type'].astype(str).unique())
            target_types = sorted(targets['target_type'].astype(str).unique())
            dense_blocks: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
            for source_type in source_types:
                pair_sources = sources[sources['source_type'].astype(str) == source_type]
                for target_type in target_types:
                    pair_targets = targets[
                        targets['target_type'].astype(str) == target_type
                    ]
                    if pair_sources.empty or pair_targets.empty:
                        continue
                    if settings.verbose:
                        logging.info(
                            'Computing split dense matrix %s -> %s (%s x %s)',
                            source_type,
                            target_type,
                            f'{len(pair_sources):,}',
                            f'{len(pair_targets):,}',
                        )
                    pair_has_candidates = (
                        source_type == 'candidates' or target_type == 'candidates'
                    )
                    pair_cache_values = [
                        *source_cache_values,
                        'matrix_dense',
                        f'src_{source_type}',
                        f'dst_{target_type}',
                    ]
                    dense_matrices = cache.run(
                        cache_path=cache.distance_matrix_path_for(
                            distance_threshold_largest=effective_distance_threshold_km,
                            max_total_dist=settings.max_total_dist,
                            population_threshold=settings.population_threshold,
                            sample_fraction=settings.sample_fraction,
                            max_points=settings.max_points,
                            random_seed=settings.random_seed,
                            aggregate_factor=agg,
                            amenity_values=pair_cache_values,
                            candidate_grid_spacing_m=candidate_grid_spacing_m,
                            candidate_max_snap_dist_m=candidate_max_snap_dist_m,
                            has_candidates=pair_has_candidates,
                            snap_components=settings.snap_components,
                            network_backend=network_cache_backend,
                            cost_profile=settings.network_impedance,
                        ),
                        builder=lambda pair_targets=pair_targets, pair_sources=pair_sources: (
                            compute_dense_distance_matrices(
                                targets=pair_targets,
                                sources=pair_sources,
                                network=network,
                                max_total_dist=settings.max_total_dist,
                                imp_name=cost_options['imp_name'],
                                road_matrix_name=cost_options['road_matrix_name'],
                                stitch_cost_factor=cost_options['stitch_cost_factor'],
                                verbose=settings.verbose,
                            )
                        ),
                    )
                    dense_blocks[(target_type, source_type)] = dense_matrices
                    if matrix_preview is None:
                        matrix_preview = table_head(dense_matrices['total'])
                    distance_matrix_size += len(pair_targets) * len(pair_sources)
                    matrix_output_paths.update(
                        write_split_dense_matrix_outputs(
                            dense_matrices=dense_matrices,
                            output_dir=output_dir,
                            run_tag=run_tag,
                            source_type=source_type,
                            target_type=target_type,
                            include_components=settings.dense_component_matrices,
                            road_component=cost_options['road_matrix_name'],
                            verbose=settings.verbose,
                        )
                    )

            if settings.matrix_output_mode == 'both' and dense_blocks:
                components = ['total']
                if settings.dense_component_matrices:
                    components.extend([
                        'origin_stitch',
                        'destination_stitch',
                        cost_options['road_matrix_name'],
                    ])
                combined_dense: dict[str, pd.DataFrame] = {}
                for component in components:
                    target_rows: list[pd.DataFrame] = []
                    for target_type in target_types:
                        source_blocks = [
                            dense_blocks[(target_type, source_type)][component]
                            for source_type in source_types
                            if (target_type, source_type) in dense_blocks
                        ]
                        if source_blocks:
                            target_rows.append(pd.concat(source_blocks, axis=1))
                    if target_rows:
                        combined_dense[component] = pd.concat(target_rows, axis=0)
                matrix_df = combined_dense.get('total')
                matrix_output_paths = {
                    **write_dense_matrix_outputs(
                        dense_matrices=combined_dense,
                        output_dir=output_dir,
                        run_tag=run_tag,
                        include_components=settings.dense_component_matrices,
                        road_component=cost_options['road_matrix_name'],
                        verbose=settings.verbose,
                    ),
                    **matrix_output_paths,
                }
            else:
                matrix_df = matrix_preview

    population_points = filter_to_bbox(population_points, settings.bbox)

    population.to_parquet(population_path, index=False)
    targets.to_parquet(targets_path, index=False)
    existing_sources.to_parquet(existing_sources_path, index=False)
    sources.to_parquet(sources_path, index=False)
    if component_summary is not None:
        component_summary.to_parquet(connectivity_path, index=False)
    output_paths = {
        'population': population_path,
        'targets': targets_path,
        'existing_sources': existing_sources_path,
        'sources': sources_path,
        **matrix_output_paths,
    }
    if component_summary is not None:
        output_paths['connectivity_components'] = connectivity_path
    write_run_manifest(
        build_run_manifest(
            cfg=cfg,
            settings=settings,
            aggregate_factor=agg,
            amenity_values=source_cache_values,
            candidate_grid_spacing_m=candidate_grid_spacing_m,
            candidate_max_snap_dist_m=candidate_max_snap_dist_m,
            has_candidates=candidate_sites_snapped is not None,
            output_paths=output_paths,
            repo_dir=Path(__file__).resolve().parent,
        ),
        manifest_path,
    )

    preview = matrix_preview
    if preview is None and matrix_df is not None:
        preview = table_head(matrix_df)
    if pl is not None and isinstance(preview, pl.DataFrame):
        print(preview.to_pandas().to_string(index=False))
    elif preview is not None:
        print(preview)
    else:
        print('No distance matrix rows were generated.')

    if settings.verbose:
        logging.info(f'Wrote population/legacy target output: {population_path}')
        logging.info(f'Wrote targets output: {targets_path}')
        logging.info(f'Wrote existing sources output: {existing_sources_path}')
        logging.info(f'Wrote sources output: {sources_path}')
        if component_summary is not None:
            logging.info(f'Wrote connectivity components output: {connectivity_path}')
        if 'distance_matrix' in matrix_output_paths:
            logging.info(
                f'Wrote combined distance matrix output: {matrix_output_paths["distance_matrix"]}'
            )
        split_outputs = {
            name: path
            for name, path in matrix_output_paths.items()
            if name != 'distance_matrix'
        }
        if split_outputs:
            logging.info(
                f'Wrote {len(split_outputs):,} split distance matrix output(s)'
            )
        logging.info(f'Wrote run manifest: {manifest_path}')
        logging.info(f'Distance matrix size: {distance_matrix_size:,}')
        logging.info(f'Distance computation time: {pc() - t_dist:.2f}s')
        logging.info(f'Total runtime: {pc() - t_total:.2f}s')


# ------------------------
# ENTRYPOINT
# ------------------------
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_file, verbose=not args.quiet)
    settings = settings_from_args(args)
    base_cfg = load_cfg(args.country_code)
    cfg = resolve_input_config(base_cfg, args)
    source_layers = resolve_source_layers_from_args(args)
    destination_layers = resolve_destination_layers_from_args(args)

    main(
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
