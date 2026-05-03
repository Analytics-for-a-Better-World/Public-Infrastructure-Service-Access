from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter as pc

import geopandas as gpd
import gurobipy as gp
import numpy as np
import pandas as pd
import polars as pl
from gurobipy import GRB
from scipy.spatial import cKDTree

from distance_pipeline.cache import CacheManager
from distance_pipeline.config_loader import load_cfg
from distance_pipeline.distance_matrix import compute_distances_polars
from distance_pipeline.network import build_pandana_network, load_osm_network
from distance_pipeline.population import worldpop_to_points
from distance_pipeline.snapping import snap_points_to_nodes
from distance_pipeline.source_tables import ensure_id_column, ensure_id_index_matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Locate defibrillators at Luxembourg road-network nodes.'
    )
    parser.add_argument(
        'country_code',
        nargs='?',
        default='luxembourg',
        help='Country config name or alias. Defaults to luxembourg.',
    )
    parser.add_argument(
        '--coverage-radius-m',
        type=float,
        default=500.0,
        help='Maximum total road-network distance from population point to AED node.',
    )
    parser.add_argument(
        '--aggregate-factor',
        type=int,
        default=None,
        help='Optional WorldPop aggregation factor. Defaults to unaggregated.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Optional output directory. Defaults to <country_data>/defibrillator_outputs.',
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Rebuild cached pipeline stages and AED coverage matrix.',
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=None,
        help='Optional Gurobi time limit in seconds per solve.',
    )
    parser.add_argument(
        '--dispersion-iterations',
        type=int,
        default=18,
        help='Binary-search iterations for the max-min separation stage.',
    )
    parser.add_argument(
        '--trace',
        action='store_true',
        help='Show Gurobi output.',
    )
    parser.add_argument(
        '--include-target-stitch',
        choices=('true', 'false'),
        default='false',
        help=(
            'Whether the 500 m coverage radius includes the off-network snap '
            'from population point to road node. Defaults to false because '
            'strict total access can be infeasible when population raster cells '
            'are already farther than the radius from the road network.'
        ),
    )
    parser.add_argument(
        '--candidate-mode',
        choices=('population_nodes', 'all_nodes'),
        default='population_nodes',
        help=(
            'Network nodes eligible for AED placement. population_nodes uses '
            'only road nodes that at least one population point snaps to; '
            'all_nodes allows every OSM road node and is much larger.'
        ),
    )
    parser.add_argument(
        '--min-candidate-degree',
        type=int,
        default=1,
        help=(
            'Optional pruning threshold for candidate nodes. Candidates covering '
            'at least this many population points are kept, and the nearest '
            'candidate for every population point is always kept to preserve '
            'feasibility. Use 2 to make all_nodes experiments smaller.'
        ),
    )
    return parser


def _population_cache_path(cache: CacheManager, radius_m: float, aggregate_factor: int | None) -> Path:
    return cache.population_snapped_path_for(
        distance_col='dist_snap_target',
        population_threshold=1.0,
        sample_fraction=1.0,
        max_points=None,
        aggregate_factor=aggregate_factor,
    )


def _load_or_build_population(
    *,
    cfg,
    cache: CacheManager,
    aggregate_factor: int | None,
) -> pd.DataFrame:
    points = cache.run(
        cache_path=cache.population_points_path(
            population_threshold=1.0,
            sample_fraction=1.0,
            max_points=None,
            aggregate_factor=aggregate_factor,
        ),
        builder=lambda: worldpop_to_points(
            cfg.WORLDPOP_PATH,
            population_threshold=1.0,
            sample_fraction=1.0,
            max_points=None,
            aggregate_factor=aggregate_factor,
            verbose=True,
        ),
    )

    nodes, _ = cache.load_or_build_network_data(
        builder=lambda: load_osm_network(cfg.PBF_PATH, verbose=True)[1:],
    )

    population = cache.run(
        cache_path=_population_cache_path(cache, 500.0, aggregate_factor),
        builder=lambda: snap_points_to_nodes(
            points,
            nodes,
            distance_col='dist_snap_target',
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=True,
        ),
    )
    population = ensure_id_column(population, prefix='target')
    return ensure_id_index_matches(population)


def _node_sources(
    nodes: gpd.GeoDataFrame,
    *,
    candidate_node_ids: set[int] | None = None,
) -> pd.DataFrame:
    if candidate_node_ids is not None:
        nodes = nodes.loc[nodes['id'].astype('int64').isin(candidate_node_ids)].copy()

    sources = pd.DataFrame(
        {
            'ID': nodes['id'].astype('int64').to_numpy(),
            'Longitude': nodes['lon'].astype('float64').to_numpy(),
            'Latitude': nodes['lat'].astype('float64').to_numpy(),
            'nearest_node': nodes['id'].astype('int64').to_numpy(),
            'dist_snap_source': np.zeros(len(nodes), dtype='float64'),
        }
    )
    return ensure_id_index_matches(sources)


def _build_coverage_matrix(
    *,
    cfg,
    cache: CacheManager,
    population: pd.DataFrame,
    radius_m: float,
    candidate_mode: str,
    output_path: Path,
) -> pl.DataFrame:
    if output_path.exists() and not cache.force_recompute:
        print(f'Loading AED coverage matrix: {output_path}')
        return pl.read_parquet(output_path)

    nodes, edges = cache.load_or_build_network_data(
        builder=lambda: load_osm_network(cfg.PBF_PATH, verbose=True)[1:],
    )
    network = build_pandana_network(nodes=nodes, edges=edges)
    candidate_node_ids = None
    if candidate_mode == 'population_nodes':
        candidate_node_ids = set(population['nearest_node'].astype('int64'))
    sources = _node_sources(nodes, candidate_node_ids=candidate_node_ids)

    matrix = compute_distances_polars(
        targets=population,
        sources=sources,
        distance_threshold_largest=radius_m / 1000.0,
        network=network,
        max_total_dist=radius_m,
        verbose=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_parquet(output_path)
    return matrix


def _coverage_targets(
    population: pd.DataFrame,
    *,
    nodes: gpd.GeoDataFrame,
    include_target_stitch: bool,
) -> pd.DataFrame:
    targets = population.copy()
    if not include_target_stitch:
        targets['dist_snap_target'] = 0.0
        node_coords = nodes[['id', 'lon', 'lat']].reset_index(drop=True).copy()
        node_coords['id'] = node_coords['id'].astype('int64')
        targets = targets.merge(
            node_coords,
            left_on='nearest_node',
            right_on='id',
            how='left',
            validate='many_to_one',
        )
        if targets['lon'].isna().any() or targets['lat'].isna().any():
            raise ValueError('Some snapped population nodes are missing from the node table.')
        targets['Longitude'] = targets['lon'].astype('float64')
        targets['Latitude'] = targets['lat'].astype('float64')
        targets = targets.drop(columns=['id', 'lon', 'lat'])
        targets = ensure_id_index_matches(targets)
    return targets


def _coverage_sets(matrix: pl.DataFrame) -> tuple[list[int], list[int], dict[int, list[int]]]:
    df = matrix.select(['target_id', 'source_id']).to_pandas()
    grouped = df.groupby('target_id', sort=True)['source_id'].agg(list)
    targets = [int(target) for target in grouped.index]
    candidates = sorted({int(source) for values in grouped for source in values})
    cover = {int(target): [int(source) for source in values] for target, values in grouped.items()}
    return targets, candidates, cover


def _prune_candidates(matrix: pl.DataFrame, min_candidate_degree: int) -> pl.DataFrame:
    if min_candidate_degree <= 1:
        return matrix

    degree_keep = (
        matrix.group_by('source_id')
        .len()
        .filter(pl.col('len') >= min_candidate_degree)
        .select('source_id')
    )
    nearest_keep = (
        matrix.sort(['target_id', 'total_dist'])
        .group_by('target_id', maintain_order=True)
        .first()
        .select('source_id')
        .unique()
    )
    keep = pl.concat([degree_keep, nearest_keep]).unique()
    return matrix.join(keep, on='source_id', how='inner')


def _solve_set_cover(
    *,
    targets: list[int],
    candidates: list[int],
    cover: dict[int, list[int]],
    trace: bool,
    time_limit: float | None,
) -> tuple[list[int], float, str]:
    t0 = pc()
    model = gp.Model('luxembourg_aed_set_cover')
    model.Params.OutputFlag = int(trace)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    y = model.addVars(candidates, vtype=GRB.BINARY, name='y')
    for target in targets:
        model.addConstr(gp.quicksum(y[j] for j in cover[target]) >= 1, name=f'cover_{target}')
    model.setObjective(gp.quicksum(y[j] for j in candidates), GRB.MINIMIZE)
    model.optimize()

    selected = [j for j in candidates if y[j].X > 0.5]
    return selected, pc() - t0, str(model.Status)


def _close_pairs(nodes_proj: gpd.GeoDataFrame, candidate_ids: list[int], min_dist_m: float) -> list[tuple[int, int]]:
    if min_dist_m <= 0 or len(candidate_ids) < 2:
        return []

    subset = nodes_proj.loc[candidate_ids]
    coords = np.column_stack([subset.geometry.x.to_numpy(), subset.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    pairs_idx = tree.query_pairs(r=min_dist_m)
    ids = subset.index.to_numpy(dtype='int64')
    return [(int(ids[i]), int(ids[j])) for i, j in pairs_idx]


def _solve_dispersion_feasibility(
    *,
    targets: list[int],
    candidates: list[int],
    cover: dict[int, list[int]],
    nodes_proj: gpd.GeoDataFrame,
    n_selected: int,
    min_dist_m: float,
    trace: bool,
    time_limit: float | None,
) -> list[int] | None:
    close_pairs = _close_pairs(nodes_proj, candidates, min_dist_m)

    model = gp.Model('luxembourg_aed_dispersion')
    model.Params.OutputFlag = int(trace)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    y = model.addVars(candidates, vtype=GRB.BINARY, name='y')
    model.addConstr(gp.quicksum(y[j] for j in candidates) == n_selected)
    for target in targets:
        model.addConstr(gp.quicksum(y[j] for j in cover[target]) >= 1)
    for i, j in close_pairs:
        model.addConstr(y[i] + y[j] <= 1)
    model.setObjective(0.0, GRB.MAXIMIZE)
    model.optimize()

    if model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL} and model.SolCount > 0:
        return [j for j in candidates if y[j].X > 0.5]
    return None


def _maximize_min_separation(
    *,
    selected_min_cover: list[int],
    targets: list[int],
    candidates: list[int],
    cover: dict[int, list[int]],
    nodes: gpd.GeoDataFrame,
    projected_epsg: int,
    coverage_radius_m: float,
    iterations: int,
    trace: bool,
    time_limit: float | None,
) -> tuple[list[int], float, float]:
    t0 = pc()
    nodes_proj = nodes.to_crs(epsg=projected_epsg).copy()
    nodes_proj.index = nodes_proj['id'].astype('int64')

    candidate_gdf = nodes_proj.loc[candidates]
    minx, miny, maxx, maxy = candidate_gdf.total_bounds
    low = 0.0
    high = min(float(np.hypot(maxx - minx, maxy - miny)), 2.0 * coverage_radius_m)
    best = selected_min_cover

    for _ in range(iterations):
        mid = (low + high) / 2.0
        print(f'Testing AED minimum separation {mid:.1f} m...')
        solution = _solve_dispersion_feasibility(
            targets=targets,
            candidates=candidates,
            cover=cover,
            nodes_proj=nodes_proj,
            n_selected=len(selected_min_cover),
            min_dist_m=mid,
            trace=trace,
            time_limit=time_limit,
        )
        if solution is None:
            high = mid
        else:
            low = mid
            best = solution

    return best, low, pc() - t0


def _write_solution(
    *,
    selected: list[int],
    nodes: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    solution = nodes.loc[nodes['id'].isin(selected)].copy()
    solution['selected'] = True
    solution = solution.reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solution.to_file(output_path, driver='GeoJSON')


def run(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.country_code)
    output_dir = args.output_dir or (cfg.BASE_DIR / 'defibrillator_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = CacheManager(
        cfg=cfg,
        force_recompute=args.force_recompute,
        verbose=True,
    )

    population = _load_or_build_population(
        cfg=cfg,
        cache=cache,
        aggregate_factor=args.aggregate_factor,
    )
    include_target_stitch = args.include_target_stitch == 'true'
    uncovered_by_stitch = population.loc[
        population['dist_snap_target'] > args.coverage_radius_m
    ]
    if include_target_stitch and len(uncovered_by_stitch):
        raise ValueError(
            f'{len(uncovered_by_stitch):,} population points are farther than '
            f'{args.coverage_radius_m:g} m from the road network.'
        )
    nodes, _ = cache.load_or_build_network_data(
        builder=lambda: load_osm_network(cfg.PBF_PATH, verbose=True)[1:],
    )
    coverage_population = _coverage_targets(
        population,
        nodes=nodes,
        include_target_stitch=include_target_stitch,
    )

    tag = (
        f"{cfg.iso3.lower()}_aed_radius_{args.coverage_radius_m:g}m_"
        f"agg_{args.aggregate_factor if args.aggregate_factor is not None else 'none'}_"
        f"{'total_access' if include_target_stitch else 'snapped_node_v2'}_"
        f"candidates_{args.candidate_mode}"
    )
    coverage_path = output_dir / f'coverage_matrix_{tag}.parquet'
    summary_path = output_dir / f'summary_{tag}.json'
    solution_path = output_dir / f'solution_nodes_{tag}.geojson'

    matrix = _build_coverage_matrix(
        cfg=cfg,
        cache=cache,
        population=coverage_population,
        radius_m=args.coverage_radius_m,
        candidate_mode=args.candidate_mode,
        output_path=coverage_path,
    )

    pruned_matrix = _prune_candidates(matrix, args.min_candidate_degree)
    targets, candidates, cover = _coverage_sets(pruned_matrix)
    missing_targets = set(population['ID'].astype(int)).difference(targets)
    if missing_targets:
        raise ValueError(f'{len(missing_targets):,} population points have no AED candidate within radius.')

    selected_min, set_cover_time_s, set_cover_status = _solve_set_cover(
        targets=targets,
        candidates=candidates,
        cover=cover,
        trace=args.trace,
        time_limit=args.time_limit,
    )
    print(
        f'Minimum cover selected {len(selected_min):,} AED nodes '
        f'in {set_cover_time_s:.2f} seconds.'
    )

    selected, min_sep_m, dispersion_time_s = _maximize_min_separation(
        selected_min_cover=selected_min,
        targets=targets,
        candidates=candidates,
        cover=cover,
        nodes=nodes,
        projected_epsg=cfg.PROJECTED_EPSG,
        coverage_radius_m=args.coverage_radius_m,
        iterations=args.dispersion_iterations,
        trace=args.trace,
        time_limit=args.time_limit,
    )
    _write_solution(selected=selected, nodes=nodes, output_path=solution_path)

    total_population = float(population['population'].sum())
    summary = {
        'country': cfg.country_name,
        'coverage_radius_m': float(args.coverage_radius_m),
        'coverage_mode': 'total_access' if include_target_stitch else 'snapped_road_node',
        'candidate_mode': args.candidate_mode,
        'aggregate_factor': args.aggregate_factor,
        'population_points': int(len(population)),
        'represented_population': total_population,
        'population_points_with_snap_dist_over_radius': int(len(uncovered_by_stitch)),
        'population_with_snap_dist_over_radius': float(uncovered_by_stitch['population'].sum()),
        'coverage_rows': int(matrix.height),
        'coverage_rows_after_candidate_pruning': int(pruned_matrix.height),
        'min_candidate_degree': int(args.min_candidate_degree),
        'candidate_network_nodes': int(len(candidates)),
        'selected_defibrillator_nodes': int(len(selected)),
        'minimum_pairwise_projected_distance_m': float(min_sep_m),
        'set_cover_status': set_cover_status,
        'set_cover_time_s': float(set_cover_time_s),
        'dispersion_time_s': float(dispersion_time_s),
        'outputs': {
            'coverage_matrix': coverage_path.as_posix(),
            'solution_nodes': solution_path.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
