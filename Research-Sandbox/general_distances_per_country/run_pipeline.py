from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter as pc

from countries.base import CountryConfig
from distance_pipeline.cache import CacheManager
from distance_pipeline.candidate_builder import build_candidate_sites
from distance_pipeline.config_loader import load_cfg
from distance_pipeline.distance_matrix import compute_distances
from distance_pipeline.facilities import load_health_facilities
from distance_pipeline.io import download_file
from distance_pipeline.manifest import build_run_manifest, write_run_manifest
from distance_pipeline.network import build_pandana_network, load_osm_network
from distance_pipeline.pipeline_support import (
    build_context_map_path,
    build_map_facilities,
    build_output_run_tag,
    resolve_candidate_grid_spacing,
    resolve_candidate_max_snap_dist,
)
from distance_pipeline.population import worldpop_to_points
from distance_pipeline.settings import PipelineSettings
from distance_pipeline.snapping import snap_points_to_nodes
from distance_pipeline.source_tables import (
    combine_existing_and_candidate_sources,
    ensure_id_column,
    ensure_id_index_matches,
    set_known_categories,
)
from distance_pipeline.viz import classify_roads, plot_context_map, to_point_geometries


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
        '--max-total-dist',
        type=float,
        default=None,
        help='Optional maximum total distance to retain in the output matrix.',
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

    # NEW facility controls
    parser.add_argument(
        '--amenity',
        nargs='+',
        default=None,
        help='Filter facilities by amenity values, for example hospital clinic.',
    )

    parser.add_argument(
        '--no-healthcare-tag',
        action='store_true',
        help='Disable use of healthcare=* tag when extracting facilities.',
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
    if args.max_total_dist is not None and args.max_total_dist <= 0:
        raise ValueError('--max-total-dist must be positive.')
    if args.aggregate_factor is not None and args.aggregate_factor < 2:
        raise ValueError('--aggregate-factor must be at least 2.')
    if args.candidate_grid_spacing_m is not None and args.candidate_grid_spacing_m <= 0:
        raise ValueError('--candidate-grid-spacing-m must be positive.')
    if args.candidate_max_snap_dist_m is not None and args.candidate_max_snap_dist_m <= 0:
        raise ValueError('--candidate-max-snap-dist-m must be positive.')
    if args.map_dpi <= 0:
        raise ValueError('--map-dpi must be positive.')

    return PipelineSettings(
        population_threshold=args.population_threshold,
        sample_fraction=args.sample_fraction,
        max_points=args.max_points,
        max_total_dist=args.max_total_dist,
        candidate_grid_spacing_m=args.candidate_grid_spacing_m,
        candidate_max_snap_dist_m=args.candidate_max_snap_dist_m,
        force_recompute=args.force_recompute,
        verbose=not args.quiet,
        save_context_map=args.save_map,
        show_context_map=args.show_map,
        context_map_path=None if args.map_path is None else Path(args.map_path),
        context_map_dpi=args.map_dpi,
    )


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
    country_code: str,
    settings: PipelineSettings,
    aggregate_factor: int | None,
    no_aggregate: bool,
    build_map: bool,
    amenity_values: list[str] | None,
    include_healthcare_tag: bool,
) -> None:
    '''Run the pipeline for a given country.'''
    t_total = pc()
    cfg: CountryConfig = load_cfg(country_code)

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

    if settings.verbose:
        logging.info(f'Running pipeline for {cfg.COUNTRY_NAME}')
        logging.info(f'Aggregate factor: {agg}')
        logging.info(f'Amenity filter: {amenity_values}')
        logging.info(f'Include healthcare tag: {include_healthcare_tag}')

    cfg.BASE_DIR.mkdir(parents=True, exist_ok=True)

    download_file(cfg.PBF_URL, cfg.PBF_PATH, overwrite=False, verbose=settings.verbose)
    download_file(cfg.WORLDPOP_URL, cfg.WORLDPOP_PATH, overwrite=False, verbose=settings.verbose)

    nodes, edges = cache.load_or_build_network_data(
        builder=lambda: load_osm_network(cfg.PBF_PATH, verbose=settings.verbose)[1:]
    )
    network = build_pandana_network(nodes, edges)

    roads = cache.run(
        cache_path=cache.roads_path(),
        builder=lambda: classify_roads(edges, verbose=settings.verbose),
    )

    population_points = cache.run(
        cache_path=cache.population_points_path(
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            aggregate_factor=agg,
        ),
        builder=lambda: worldpop_to_points(
            cfg.WORLDPOP_PATH,
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            aggregate_factor=agg,
            verbose=settings.verbose,
        ),
    )

    # -------- facilities (RESTORED + EXTENDED) --------
    facilities = cache.run(
        cache_path=cache.health_facilities_path(
            amenity_values=amenity_values,
            include_healthcare_tag=include_healthcare_tag,
        ),
        builder=lambda: load_health_facilities(
            cfg.PBF_PATH,
            amenity_values=amenity_values,
            include_healthcare_tag=include_healthcare_tag,
            verbose=settings.verbose,
        ),
    )

    facilities = cache.run(
        cache_path=cache.health_facilities_points_path(
            amenity_values=amenity_values,
            include_healthcare_tag=include_healthcare_tag,
        ),
        builder=lambda: to_point_geometries(
            facilities,
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        ),
    )

    if settings.verbose:
        logging.info(f'Population points: {len(population_points):,}')
        logging.info(f'Facilities: {len(facilities):,}')

    candidate_sites, candidate_sites_snapped = build_candidate_sites(
        cfg=cfg,
        settings=settings,
        cache=cache,
        nodes=nodes,
    )
    candidate_grid_spacing_m = resolve_candidate_grid_spacing(cfg, settings)
    candidate_max_snap_dist_m = resolve_candidate_max_snap_dist(cfg, settings)

    # -------- MAP (optional) --------
    if build_map or settings.save_context_map or settings.show_context_map:
        map_facilities = build_map_facilities(facilities, candidate_sites_snapped)
        context_map_path = build_context_map_path(
            cache.context_map_path(),
            settings.context_map_path,
            resolve_candidate_grid_spacing(cfg, settings),
        )
        plot_context_map(
            roads,
            population_points,
            map_facilities,
            cfg.PLOT_TITLE,
            context_map_path if settings.save_context_map else None,
            settings.context_map_dpi,
            settings.show_context_map,
            settings.verbose,
        )

    population = cache.run(
        cache_path=cache.population_snapped_path_for(
            distance_col='dist_snap_target',
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            aggregate_factor=agg,
        ),
        builder=lambda: snap_points_to_nodes(
            population_points,
            nodes,
            distance_col='dist_snap_target',
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        ),
    )

    population = ensure_id_column(population, prefix='target')
    population = ensure_id_index_matches(population)

    existing_sources = cache.run(
        cache_path=cache.hospitals_snapped_path_for(
            distance_col='dist_snap_source',
            amenity_values=amenity_values,
            include_healthcare_tag=include_healthcare_tag,
        ),
        builder=lambda: snap_points_to_nodes(
            facilities,
            nodes,
            distance_col='dist_snap_source',
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        ),
    )

    existing_sources = existing_sources.copy()
    existing_sources['source_type'] = 'existing'
    existing_sources = ensure_id_column(existing_sources, prefix='existing')
    existing_sources = ensure_id_index_matches(existing_sources)

    sources = combine_existing_and_candidate_sources(
        existing_sources,
        candidate_sites_snapped,
    )

    t_dist = pc()
    matrix_df = cache.run(
        cache_path=cache.distance_matrix_path_for(
            distance_threshold_largest=cfg.DISTANCE_THRESHOLD_KM,
            max_total_dist=settings.max_total_dist,
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            aggregate_factor=agg,
            amenity_values=amenity_values,
            include_healthcare_tag=include_healthcare_tag,
            candidate_grid_spacing_m=candidate_grid_spacing_m,
            candidate_max_snap_dist_m=candidate_max_snap_dist_m,
            has_candidates=candidate_sites_snapped is not None,
        ),
        builder=lambda: compute_distances(
            targets=population,
            sources=sources,
            distance_threshold_largest=cfg.DISTANCE_THRESHOLD_KM,
            network=network,
            max_total_dist=settings.max_total_dist,
            verbose=settings.verbose,
        ),
    )

    matrix_df = set_known_categories(matrix_df)

    output_dir = cfg.BASE_DIR / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = build_output_run_tag(
        settings=settings,
        aggregate_factor=agg,
        amenity_values=amenity_values,
        include_healthcare_tag=include_healthcare_tag,
        candidate_grid_spacing_m=candidate_grid_spacing_m,
        candidate_max_snap_dist_m=candidate_max_snap_dist_m,
        has_candidates=candidate_sites_snapped is not None,
    )

    population_path = output_dir / f'population_{run_tag}.parquet'
    existing_sources_path = output_dir / f'existing_sources_{run_tag}.parquet'
    matrix_path = output_dir / f'distance_matrix_{run_tag}.parquet'
    manifest_path = output_dir / f'run_manifest_{run_tag}.json'

    population.to_parquet(population_path, index=False)
    existing_sources.to_parquet(existing_sources_path, index=False)
    matrix_df.to_parquet(matrix_path, index=False)
    write_run_manifest(
        build_run_manifest(
            cfg=cfg,
            settings=settings,
            aggregate_factor=agg,
            amenity_values=amenity_values,
            include_healthcare_tag=include_healthcare_tag,
            candidate_grid_spacing_m=candidate_grid_spacing_m,
            candidate_max_snap_dist_m=candidate_max_snap_dist_m,
            has_candidates=candidate_sites_snapped is not None,
            output_paths={
                'population': population_path,
                'existing_sources': existing_sources_path,
                'distance_matrix': matrix_path,
            },
            repo_dir=Path(__file__).resolve().parent,
        ),
        manifest_path,
    )

    print(matrix_df.head())

    if settings.verbose:
        logging.info(f'Wrote population output: {population_path}')
        logging.info(f'Wrote existing sources output: {existing_sources_path}')
        logging.info(f'Wrote distance matrix output: {matrix_path}')
        logging.info(f'Wrote run manifest: {manifest_path}')
        logging.info(f'Distance matrix size: {len(matrix_df):,}')
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

    main(
        args.country_code,
        settings,
        args.aggregate_factor,
        args.no_aggregate,
        args.build_map,
        args.amenity,
        not args.no_healthcare_tag,
    )
