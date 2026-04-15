from __future__ import annotations

from time import perf_counter as pc

import pandas as pd

from countries.base import CountryConfig
from distance_pipeline.boundaries import load_country_geometry
from distance_pipeline.candidate_sites import (
    build_regular_grid_within_polygon,
    exclude_points_on_water,
    filter_snapped_candidates_by_distance,
)
from distance_pipeline.snapping import snap_points_to_nodes
from distance_pipeline.water import load_water_bodies
from distance_pipeline.settings import PipelineSettings
from distance_pipeline.cache import CacheManager


def build_candidate_sites(
    *,
    cfg: CountryConfig,
    settings: PipelineSettings,
    cache: CacheManager,
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    '''
    Build and snap candidate facility locations.

    Parameters
    ----------
    cfg
        Country configuration.
    settings
        Runtime pipeline settings.
    cache
        Cache manager instance.
    nodes
        Network nodes used for snapping.

    Returns
    -------
    tuple[pd.DataFrame | None, pd.DataFrame | None]
        candidate_sites, candidate_sites_snapped
    '''
    t0 = pc()

    candidate_grid_spacing_m = (
        settings.candidate_grid_spacing_m
        if settings.candidate_grid_spacing_m is not None
        else cfg.candidate_grid_spacing_m
    )

    candidate_max_snap_dist_m = (
        settings.candidate_max_snap_dist_m
        if settings.candidate_max_snap_dist_m is not None
        else cfg.candidate_max_snap_dist_m
    )

    if settings.verbose:
        print(f'candidate_grid_spacing_m = {candidate_grid_spacing_m}')
        print(f'candidate_max_snap_dist_m = {candidate_max_snap_dist_m}')

    if candidate_grid_spacing_m is None:
        if settings.verbose:
            print('Candidate generation is disabled.')
        return None, None

    if cfg.boundary_source != 'natural_earth':
        raise ValueError(
            f'Unsupported boundary_source {cfg.boundary_source!r}. '
            'Only natural_earth is supported.'
        )

    country_boundary = cache.run(
        cache_path=cache.country_boundary_path(),
        builder=lambda: load_country_geometry(
            iso3=cfg.iso3,
            cache_dir=cache.boundaries_dir(),
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        ),
    )

    water_bodies = cache.run(
        cache_path=cache.water_bodies_path(),
        builder=lambda: load_water_bodies(
            cfg.PBF_PATH,
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=settings.verbose,
        ),
    )

    def build_candidates() -> pd.DataFrame:
        candidates = build_regular_grid_within_polygon(
            polygon_gdf=country_boundary,
            spacing_m=candidate_grid_spacing_m,
            include_boundary=cfg.candidate_include_boundary,
            verbose=settings.verbose,
        )

        if cfg.candidate_exclude_water:
            candidates = exclude_points_on_water(
                candidates=candidates,
                water_bodies=water_bodies,
                verbose=settings.verbose,
            )

        return candidates

    candidate_sites = cache.run(
        cache_path=cache.candidate_sites_path(
            grid_spacing_m=candidate_grid_spacing_m,
            exclude_water=cfg.candidate_exclude_water,
            include_boundary=cfg.candidate_include_boundary,
        ),
        builder=build_candidates,
    )

    candidate_sites_snapped = cache.run(
        cache_path=cache.candidate_sites_snapped_path(
            grid_spacing_m=candidate_grid_spacing_m,
            exclude_water=cfg.candidate_exclude_water,
            include_boundary=cfg.candidate_include_boundary,
            distance_col='candidate_dist_road_estrada',
            max_snap_dist_m=candidate_max_snap_dist_m,
        ),
        builder=lambda: filter_snapped_candidates_by_distance(
            snap_points_to_nodes(
                candidate_sites,
                nodes,
                distance_col='candidate_dist_road_estrada',
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=settings.verbose,
            ),
            distance_col='candidate_dist_road_estrada',
            max_snap_dist_m=candidate_max_snap_dist_m,
            verbose=settings.verbose,
        ),
    )

    if settings.verbose:
        print(f'Candidate pipeline completed in {pc() - t0:.2f} seconds')
        print(candidate_sites.shape)
        print(candidate_sites_snapped.shape)

    return candidate_sites, candidate_sites_snapped