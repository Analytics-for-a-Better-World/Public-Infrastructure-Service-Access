from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from distance_pipeline.cache import CacheManager
from distance_pipeline.config_loader import load_cfg
from distance_pipeline.distance_matrix import compute_distances_polars
from distance_pipeline.facilities import deduplicate_osm_amenities, load_facilities
from distance_pipeline.io import download_file
from distance_pipeline.network import build_pandana_network, load_osm_network
from distance_pipeline.routing import (
    add_edge_speeds,
    build_networkx_graph,
    directed_costs_from_matrix,
    directed_tsp_via_gurobi_sparse,
    route_between_nodes,
    sparse_costs_from_matrix,
    symmetric_tsp_via_gurobi_sparse,
)
from distance_pipeline.snapping import snap_points_to_nodes
from distance_pipeline.source_tables import ensure_id_column, ensure_id_index_matches
from distance_pipeline.viz import to_point_geometries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run routing experiments on pipeline-generated OSM layers.'
    )
    parser.add_argument(
        'country_code',
        nargs='?',
        default='netherlands',
        help='Country config name or alias. Defaults to netherlands.',
    )
    parser.add_argument(
        '--amenity',
        nargs='+',
        default=['university', 'college'],
        help='OSM amenity values used as routing stops.',
    )
    parser.add_argument(
        '--max-total-dist',
        type=float,
        default=100_000.0,
        help='Maximum retained road distance in meters for sparse routing edges.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Optional output directory. Defaults to <country_data>/routing_outputs.',
    )
    parser.add_argument(
        '--deduplicate-amenities',
        choices=('true', 'false'),
        default='true',
        help='Whether to deduplicate OSM amenity point/polygon duplicates.',
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Ignore matching pipeline caches.',
    )
    parser.add_argument(
        '--trace',
        action='store_true',
        help='Show Gurobi TSP output.',
    )
    return parser


def _prepare_targets_and_sources(snapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build source and target tables from the same snapped institution layer."""
    base = ensure_id_column(snapped, prefix='institution')
    base = ensure_id_index_matches(base)

    sources = base.copy()
    if 'dist_snap_source' not in sources.columns:
        raise KeyError("snapped institutions must contain 'dist_snap_source'")
    sources['source_type'] = 'institution'

    targets = base.copy()
    targets['dist_snap_target'] = targets['dist_snap_source']

    return targets, sources


def _write_tour_routes(
    *,
    tour: list[int],
    stops: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    output_path: Path,
    weight_col: str = 'length',
    bidirectional: bool = True,
) -> None:
    graph = build_networkx_graph(
        nodes,
        edges,
        weight_col=weight_col,
        bidirectional=bidirectional,
    )

    rows = []
    for order, (origin_id, destination_id) in enumerate(zip(tour[:-1], tour[1:]), start=1):
        origin = stops.loc[origin_id]
        destination = stops.loc[destination_id]
        path_nodes, geometry = route_between_nodes(
            graph,
            int(origin['nearest_node']),
            int(destination['nearest_node']),
            weight='weight',
        )
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        path_weight = sum(float(graph[u][v]['weight']) for u, v in path_edges)
        path_length_m = sum(float(graph[u][v].get('length', 0.0)) for u, v in path_edges)
        rows.append(
            {
                'order': order,
                'origin_id': origin_id,
                'destination_id': destination_id,
                'origin_name': origin.get('Name', origin.get('name', origin_id)),
                'destination_name': destination.get('Name', destination.get('name', destination_id)),
                'n_nodes': len(path_nodes),
                'weight_col': weight_col,
                'path_weight': path_weight,
                'path_length_m': path_length_m,
                'geometry': geometry,
            }
        )

    routes = gpd.GeoDataFrame(rows, geometry='geometry', crs=edges.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    routes.to_file(output_path, driver='GeoJSON')


def run_institution_tsp(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.country_code)
    output_dir = args.output_dir or (cfg.BASE_DIR / 'routing_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.PBF_PATH.exists():
        download_file(cfg.PBF_URL, cfg.PBF_PATH)

    amenity_values = args.amenity
    deduplicate = args.deduplicate_amenities == 'true'

    cache = CacheManager(
        cfg=cfg,
        force_recompute=args.force_recompute,
        verbose=True,
    )

    nodes, edges = cache.load_or_build_network_data(
        builder=lambda: load_osm_network(cfg.PBF_PATH, verbose=True)[1:],
    )
    network = build_pandana_network(nodes=nodes, edges=edges)

    facilities = cache.run(
        cache_path=cache.facilities_path(amenity_values=amenity_values),
        builder=lambda: load_facilities(
            cfg.PBF_PATH,
            amenity_values=amenity_values,
            verbose=True,
        ),
    )

    facilities = cache.run(
        cache_path=cache.facility_points_path(
            amenity_values=amenity_values,
            deduplicate_amenities=deduplicate,
        ),
        builder=lambda: (
            deduplicate_osm_amenities(
                to_point_geometries(
                    facilities,
                    projected_epsg=cfg.PROJECTED_EPSG,
                    verbose=True,
                ),
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=True,
            )
            if deduplicate
            else to_point_geometries(
                facilities,
                projected_epsg=cfg.PROJECTED_EPSG,
                verbose=True,
            )
        ),
    )

    snapped = snap_points_to_nodes(
        facilities,
        nodes,
        distance_col='dist_snap_source',
        projected_epsg=cfg.PROJECTED_EPSG,
        keep_geometry=True,
        verbose=True,
    )

    targets, sources = _prepare_targets_and_sources(snapped)
    matrix = compute_distances_polars(
        targets=targets,
        sources=sources,
        distance_threshold_largest=args.max_total_dist / 1000.0,
        network=network,
        max_total_dist=args.max_total_dist,
        verbose=True,
    )

    tag = (
        f"{cfg.iso3.lower()}_amenity_{'-'.join(sorted(amenity_values))}_"
        f"maxdist_{args.max_total_dist:g}_"
        f"{'dedup' if deduplicate else 'raw'}"
    )
    stops_path = output_dir / f'institution_stops_{tag}.parquet'
    matrix_path = output_dir / f'institution_distance_matrix_{tag}.parquet'
    undirected_tour_path = output_dir / f'institution_tsp_tour_undirected_{tag}.csv'
    directed_tour_path = output_dir / f'institution_tsp_tour_directed_{tag}.csv'
    undirected_route_path = output_dir / f'institution_tsp_route_undirected_shortest_{tag}.geojson'
    directed_route_path = output_dir / f'institution_tsp_route_directed_shortest_{tag}.geojson'
    fastest_route_path = output_dir / f'institution_tsp_route_directed_fastest_{tag}.geojson'
    summary_path = output_dir / f'institution_tsp_summary_{tag}.json'

    gpd.GeoDataFrame(snapped, geometry='geometry', crs=facilities.crs).to_parquet(
        stops_path,
        index=False,
    )
    matrix.write_parquet(matrix_path)

    matrix_pd = matrix.select(['target_id', 'source_id', 'total_dist']).to_pandas()
    undirected_costs = sparse_costs_from_matrix(
        matrix_pd,
        origin_col='target_id',
        destination_col='source_id',
        cost_col='total_dist',
    )
    directed_costs = directed_costs_from_matrix(
        matrix_pd,
        origin_col='target_id',
        destination_col='source_id',
        cost_col='total_dist',
    )

    undirected_result = symmetric_tsp_via_gurobi_sparse(
        undirected_costs,
        nodes=snapped['ID'].astype(int).tolist(),
        trace=args.trace,
    )
    directed_result = directed_tsp_via_gurobi_sparse(
        directed_costs,
        nodes=snapped['ID'].astype(int).tolist(),
        trace=args.trace,
    )

    pd.DataFrame(
        {'order': range(len(undirected_result.tour)), 'ID': undirected_result.tour}
    ).to_csv(
        undirected_tour_path,
        index=False,
    )
    pd.DataFrame(
        {'order': range(len(directed_result.tour)), 'ID': directed_result.tour}
    ).to_csv(
        directed_tour_path,
        index=False,
    )

    stops = snapped.copy()
    stops.index = stops['ID'].astype(int)
    if undirected_result.tour:
        _write_tour_routes(
            tour=[int(node) for node in undirected_result.tour],
            stops=stops,
            nodes=nodes,
            edges=edges,
            output_path=undirected_route_path,
            weight_col='length',
            bidirectional=True,
        )
    if directed_result.tour:
        _write_tour_routes(
            tour=[int(node) for node in directed_result.tour],
            stops=stops,
            nodes=nodes,
            edges=edges,
            output_path=directed_route_path,
            weight_col='length',
            bidirectional=False,
        )
        _write_tour_routes(
            tour=[int(node) for node in directed_result.tour],
            stops=stops,
            nodes=nodes,
            edges=add_edge_speeds(edges),
            output_path=fastest_route_path,
            weight_col='travel_time_s',
            bidirectional=False,
        )

    summary = {
        'country': cfg.country_name,
        'amenity_values': amenity_values,
        'deduplicate_amenities': deduplicate,
        'n_stops': int(len(snapped)),
        'n_sparse_edges_undirected': int(len(undirected_costs)),
        'n_sparse_arcs_directed': int(len(directed_costs)),
        'max_total_dist_m': float(args.max_total_dist),
        'undirected_tsp_status': undirected_result.status,
        'undirected_tsp_objective_m': undirected_result.objective,
        'undirected_tsp_runtime_s': undirected_result.runtime_s,
        'directed_tsp_status': directed_result.status,
        'directed_tsp_objective_m': directed_result.objective,
        'directed_tsp_runtime_s': directed_result.runtime_s,
        'outputs': {
            'stops': stops_path.as_posix(),
            'matrix': matrix_path.as_posix(),
            'tour_undirected': undirected_tour_path.as_posix(),
            'tour_directed': directed_tour_path.as_posix(),
            'route_undirected_shortest': undirected_route_path.as_posix()
            if undirected_result.tour
            else None,
            'route_directed_shortest': directed_route_path.as_posix()
            if directed_result.tour
            else None,
            'route_directed_fastest': fastest_route_path.as_posix()
            if directed_result.tour
            else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    run_institution_tsp(args)


if __name__ == '__main__':
    main()
