from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter as pc

from countries.base import CountryConfig
from distance_pipeline.cache import CacheManager
from distance_pipeline.candidate_builder import build_candidate_sites
from distance_pipeline.config_loader import load_cfg
from distance_pipeline.distance_matrix import compute_distances
from distance_pipeline.facilities import load_health_facilities
from distance_pipeline.io import download_file
from distance_pipeline.network import build_pandana_network, load_osm_network
from distance_pipeline.pipeline_support import (
    build_context_map_path,
    build_map_facilities,
    resolve_candidate_grid_spacing,
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



def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run the distance pipeline for a country configuration.'
    )
    parser.add_argument(
        'country_code',
        help=(
            'Country config module name, for example portugal, netherlands, '
            'timor_leste, prt, nld, or tls.'
        ),
    )
    parser.add_argument('--force-recompute', action='store_true', help='Ignore caches and rebuild all cached steps.')
    parser.add_argument('--save-map', action='store_true', help='Save the context map to a file.')
    parser.add_argument('--show-map', action='store_true', help='Display the context map interactively.')
    parser.add_argument('--map-path', type=str, default=None, help='Optional path for the saved context map.')
    parser.add_argument('--map-dpi', type=int, default=300, help='DPI used when saving the context map.')
    parser.add_argument('--population-threshold', type=float, default=1.0, help='Minimum population threshold used for raster to points conversion.')
    parser.add_argument('--sample-fraction', type=float, default=1.0, help='Sampling fraction used for raster to points conversion.')
    parser.add_argument('--max-points', type=int, default=None, help='Maximum number of population points to keep.')
    parser.add_argument('--max-total-dist', type=float, default=None, help='Optional maximum total distance to retain in the output matrix.')
    parser.add_argument('--candidate-grid-spacing-m', type=float, default=None, help='Optional grid spacing for candidate facilities, in meters. Defaults to the country config when available.')
    parser.add_argument('--candidate-max-snap-dist-m', type=float, default=None, help='Optional maximum node snapping distance for candidate facilities, in meters.')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output.')
    return parser



def settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    """Build pipeline settings from parsed CLI arguments."""
    if args.population_threshold < 0:
        raise ValueError('--population-threshold must be non negative.')
    if not 0 < args.sample_fraction <= 1:
        raise ValueError('--sample-fraction must be in the interval (0, 1].')
    if args.max_points is not None and args.max_points <= 0:
        raise ValueError('--max-points must be positive when provided.')
    if args.max_total_dist is not None and args.max_total_dist <= 0:
        raise ValueError('--max-total-dist must be positive when provided.')
    if args.candidate_grid_spacing_m is not None and args.candidate_grid_spacing_m <= 0:
        raise ValueError('--candidate-grid-spacing-m must be positive when provided.')
    if args.candidate_max_snap_dist_m is not None and args.candidate_max_snap_dist_m <= 0:
        raise ValueError('--candidate-max-snap-dist-m must be positive when provided.')
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



def main(country_code: str, settings: PipelineSettings) -> None:
    """Run the pipeline for a given country."""
    t_total = pc()
    cfg: CountryConfig = load_cfg(country_code)
    cache = CacheManager(
        cfg=cfg,
        force_recompute=settings.force_recompute,
        verbose=settings.verbose,
    )

    if settings.verbose:
        print(f'Running pipeline for {cfg.COUNTRY_NAME}')

    cfg.BASE_DIR.mkdir(parents=True, exist_ok=True)
    download_file(cfg.PBF_URL, cfg.PBF_PATH, overwrite=False, verbose=settings.verbose)
    download_file(cfg.WORLDPOP_URL, cfg.WORLDPOP_PATH, overwrite=False, verbose=settings.verbose)

    t0 = pc()
    nodes, edges = cache.load_or_build_network_data(
        builder=lambda: load_osm_network(cfg.PBF_PATH, verbose=settings.verbose)[1:],
    )
    network = build_pandana_network(nodes=nodes, edges=edges)
    if settings.verbose:
        print(f'Prepared network objects in {pc() - t0:.2f} seconds')

    roads = cache.run(
        cache_path=cache.roads_path(),
        builder=lambda: classify_roads(edges, verbose=settings.verbose),
    )

    population_points = cache.run(
        cache_path=cache.population_points_path(
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
        ),
        builder=lambda: worldpop_to_points(
            cfg.WORLDPOP_PATH,
            population_threshold=settings.population_threshold,
            sample_fraction=settings.sample_fraction,
            max_points=settings.max_points,
            verbose=settings.verbose,
        ),
    )

    amenity_values = None
    include_healthcare_tag = True
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

    candidate_grid_spacing_m = resolve_candidate_grid_spacing(cfg, settings)
    candidate_sites, candidate_sites_snapped = build_candidate_sites(
        cfg=cfg,
        settings=settings,
        cache=cache,
        nodes=nodes,
    )

    print(f'Candidates full: {len(candidate_sites):,}')
    print(f'Candidates snapped: {len(candidate_sites_snapped):,}')

    map_facilities = build_map_facilities(
        health_centers=facilities,
        candidate_sites=candidate_sites,
    )

    context_map_path = build_context_map_path(
        default_path=cache.context_map_path(),
        user_path=settings.context_map_path,
        candidate_grid_spacing_m=candidate_grid_spacing_m,
    )

    plot_context_map(
        roads=roads,
        population_points=population_points,
        health_centers=map_facilities,
        title=cfg.PLOT_TITLE,
        output_path=context_map_path if settings.save_context_map else None,
        dpi=settings.context_map_dpi,
        show=settings.show_context_map,
        verbose=settings.verbose,
    )

    population = cache.run(
        cache_path=cache.population_snapped_path(distance_col='dist_snap_target'),
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
        cache_path=cache.hospitals_snapped_path(distance_col='dist_snap_source'),
        builder=lambda: snap_points_to_nodes(
            facilities,
            nodes,
            distance_col='dist_snap_source',
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        ),
    )

    sources_for_matrix = combine_existing_and_candidate_sources(
        facilities=existing_sources,
        candidate_sites_snapped=candidate_sites_snapped,
    )

    matrix_cache_path = cache.distance_matrix_path(
        distance_threshold_largest=cfg.DISTANCE_THRESHOLD_KM,
        max_total_dist=settings.max_total_dist,
    )
    if candidate_sites is not None:
        matrix_cache_path = matrix_cache_path.with_stem(f'{matrix_cache_path.stem}_with_candidates')

    matrix_df = cache.run(
        cache_path=matrix_cache_path,
        builder=lambda: compute_distances(
            targets=population,
            sources=sources_for_matrix,
            distance_threshold_largest=cfg.DISTANCE_THRESHOLD_KM,
            network=network,
            max_total_dist=settings.max_total_dist,
            verbose=settings.verbose,
        ),
    )

    matrix_df = set_known_categories(matrix_df)
    print(matrix_df.head())

    if settings.verbose:
        n_existing = int((sources_for_matrix['source_type'] == 'existing').sum())
        n_candidate = int((sources_for_matrix['source_type'] == 'candidate').sum())
        print(f'Existing sources used: {n_existing:,}')
        print(f'Candidate sources used: {n_candidate:,}')
        print(f'Context map path: {context_map_path}')
        print(f'Total pipeline runtime: {pc() - t_total:.2f} seconds')


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    settings = settings_from_args(args)
    main(args.country_code, settings)
