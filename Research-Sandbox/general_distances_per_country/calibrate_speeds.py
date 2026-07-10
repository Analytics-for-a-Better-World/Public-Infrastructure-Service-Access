"""Sample OD pairs and calibrate OSM edge speeds for Pandana travel times.

This script is deliberately separate from ``run_pipeline.py``. It prepares a
speed profile that can later be used as a travel-time impedance layer without
changing the distance-matrix pipeline defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter as pc, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from distance_pipeline.config_loader import load_cfg
from distance_pipeline.facilities import deduplicate_osm_amenities, load_facilities
from distance_pipeline.network import build_pandana_network, load_osm_network_data
from distance_pipeline.population import worldpop_to_points
from distance_pipeline.routing import (
    add_edge_speeds,
    build_networkx_graph,
    normalize_highway,
    shortest_path_nodes,
)
from distance_pipeline.snapping import snap_points_to_nodes
from distance_pipeline.viz import to_point_geometries


DEFAULT_AMENITIES = ['hospital', 'clinic', 'doctors']


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Benchmark travel time for one sampled OD pair."""

    duration_s: float
    distance_m: float | None
    provider: str
    status: str


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except ImportError:
        text = json.dumps(data, indent=2)
    path.write_text(text, encoding='utf-8')


def _read_user_env_var(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    if sys.platform != 'win32':
        return None

    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value) if value else None


def _sample_trace_coordinates(
    graph: Any,
    path_nodes: list[int],
    *,
    max_coordinates: int,
) -> list[tuple[float, float]]:
    if len(path_nodes) <= max_coordinates:
        selected = path_nodes
    else:
        idx = np.linspace(0, len(path_nodes) - 1, max_coordinates)
        selected = [path_nodes[int(round(i))] for i in idx]

    coords: list[tuple[float, float]] = []
    previous: tuple[float, float] | None = None
    for node in selected:
        item = graph.nodes[int(node)]
        coord = (float(item['x']), float(item['y']))
        if coord != previous:
            coords.append(coord)
            previous = coord

    if len(coords) < 2:
        raise ValueError('Mapbox trace needs at least two distinct coordinates')
    return coords


def _mapbox_trace_duration(
    coords: list[tuple[float, float]],
    *,
    token: str,
    profile: str,
    timeout_s: float,
) -> BenchmarkResult:
    coordinate_text = ';'.join(f'{lon:.6f},{lat:.6f}' for lon, lat in coords)
    query = urlencode(
        {
            'access_token': token,
            'overview': 'false',
            'geometries': 'geojson',
            'annotations': 'duration,distance,speed',
        }
    )
    url = (
        f'https://api.mapbox.com/directions/v5/mapbox/{profile}/'
        f'{coordinate_text}?{query}'
    )

    try:
        with urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except (HTTPError, URLError, TimeoutError) as exc:
        return BenchmarkResult(
            duration_s=float('nan'),
            distance_m=None,
            provider='mapbox',
            status=f'error: {exc}',
        )

    routes = payload.get('routes') or []
    if not routes:
        return BenchmarkResult(
            duration_s=float('nan'),
            distance_m=None,
            provider='mapbox',
            status=f"no route: {payload.get('code')}",
        )

    route = routes[0]
    return BenchmarkResult(
        duration_s=float(route['duration']),
        distance_m=float(route.get('distance', float('nan'))),
        provider='mapbox',
        status='ok',
    )


def _path_highway_times(
    graph: Any,
    path_nodes: list[int],
    *,
    base_time_col: str = 'base_time_s',
) -> dict[str, float]:
    values: dict[str, float] = {}
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge = graph[int(u)][int(v)]
        highway = normalize_highway(edge.get('highway'))
        values[highway] = values.get(highway, 0.0) + float(edge[base_time_col])
    return values


def _nearest_source_pairs(
    origins: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    *,
    projected_epsg: int,
    nearest_sources: int,
) -> pd.DataFrame:
    origins_proj = origins.to_crs(epsg=projected_epsg)
    sources_proj = sources.to_crs(epsg=projected_epsg)

    origin_xy = np.column_stack(
        [origins_proj.geometry.x.to_numpy(), origins_proj.geometry.y.to_numpy()]
    )
    source_xy = np.column_stack(
        [sources_proj.geometry.x.to_numpy(), sources_proj.geometry.y.to_numpy()]
    )
    tree = cKDTree(source_xy)
    distances, idx = tree.query(
        origin_xy,
        k=min(nearest_sources, len(sources_proj)),
    )
    distances = np.atleast_2d(distances)
    idx = np.atleast_2d(idx)
    if distances.shape[0] != len(origins):
        distances = distances.T
        idx = idx.T

    rows = []
    source_ids = sources['ID'].astype(str).to_numpy()
    source_nodes = sources['nearest_node'].astype('int64').to_numpy()
    for origin_pos, origin in enumerate(origins.itertuples(index=False)):
        for rank in range(idx.shape[1]):
            source_pos = int(idx[origin_pos, rank])
            rows.append(
                {
                    'origin_id': str(getattr(origin, 'ID')),
                    'source_id': source_ids[source_pos],
                    'origin_nearest_node': int(getattr(origin, 'nearest_node')),
                    'source_nearest_node': int(source_nodes[source_pos]),
                    'straight_distance_m': float(distances[origin_pos, rank]),
                    'source_rank': rank + 1,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        subset=['origin_nearest_node', 'source_nearest_node']
    )


def _prepare_population(
    *,
    args: argparse.Namespace,
    cfg: Any,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if args.population_path:
        population = gpd.read_parquet(args.population_path)
        if population.crs is None:
            population = population.set_crs('EPSG:4326')
    else:
        population = worldpop_to_points(
            cfg.WORLDPOP_PATH,
            population_threshold=args.population_threshold,
            sample_fraction=1.0,
            max_points=None,
            random_seed=args.random_seed,
            aggregate_factor=args.aggregate_factor,
            verbose=args.verbose,
        )

    if 'nearest_node' not in population.columns:
        population = snap_points_to_nodes(
            population,
            nodes,
            id_col='ID',
            distance_col='dist_snap_target',
            projected_epsg=cfg.PROJECTED_EPSG,
            keep_geometry=True,
            verbose=args.verbose,
        )
    elif not isinstance(population, gpd.GeoDataFrame):
        population = gpd.GeoDataFrame(
            population,
            geometry=gpd.points_from_xy(population['Longitude'], population['Latitude']),
            crs='EPSG:4326',
        )

    return population.reset_index(drop=True)


def _prepare_sources(
    *,
    args: argparse.Namespace,
    cfg: Any,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if args.sources_path:
        sources = gpd.read_parquet(args.sources_path)
        if sources.crs is None:
            sources = sources.set_crs('EPSG:4326')
    else:
        amenities = args.amenity or DEFAULT_AMENITIES
        facilities = load_facilities(
            cfg.PBF_PATH,
            amenity_values=amenities,
            verbose=args.verbose,
            bbox=args.bbox,
            backend=args.network_backend,
        )
        facilities = to_point_geometries(
            facilities,
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=args.verbose,
        )
        facilities = deduplicate_osm_amenities(
            facilities,
            projected_epsg=cfg.PROJECTED_EPSG,
            verbose=args.verbose,
        )
        sources = facilities.rename(columns={'id': 'osm_row_id'}).copy()
        sources['ID'] = sources['ID'].astype(str)

    if 'nearest_node' not in sources.columns:
        sources = snap_points_to_nodes(
            sources,
            nodes,
            id_col='ID',
            distance_col='dist_snap_source',
            projected_epsg=cfg.PROJECTED_EPSG,
            keep_geometry=True,
            verbose=args.verbose,
        )
    elif not isinstance(sources, gpd.GeoDataFrame):
        sources = gpd.GeoDataFrame(
            sources,
            geometry=gpd.points_from_xy(sources['Longitude'], sources['Latitude']),
            crs='EPSG:4326',
        )

    return sources.reset_index(drop=True)


def _sample_origins(
    population: gpd.GeoDataFrame,
    *,
    sample_size: int,
    random_seed: int,
) -> gpd.GeoDataFrame:
    n = min(sample_size, len(population))
    weights = None
    if 'population' in population.columns:
        raw = pd.to_numeric(population['population'], errors='coerce').fillna(0.0)
        if raw.sum() > 0:
            weights = raw
    return population.sample(n=n, weights=weights, random_state=random_seed).copy()


def _fit_speed_multipliers(
    sample_rows: pd.DataFrame,
    highway_columns: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    if not highway_columns:
        return {}, {}

    x = sample_rows[highway_columns].to_numpy(dtype='float64')
    y = sample_rows['observed_time_s'].to_numpy(dtype='float64')
    mask = np.isfinite(y) & (y > 0) & np.isfinite(x).all(axis=1) & (x.sum(axis=1) > 0)
    if int(mask.sum()) < 2:
        return {}, {}

    # observed_time = sum(base_time_by_class * beta_class),
    # beta_class = 1 / speed_multiplier_class.
    result = lsq_linear(x[mask], y[mask], bounds=(0.2, 5.0))
    beta = dict(zip(highway_columns, result.x, strict=True))
    multipliers = {key: float(1.0 / value) for key, value in beta.items()}
    return multipliers, beta


def _metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mask = observed.notna() & predicted.notna() & np.isfinite(observed) & np.isfinite(predicted)
    if int(mask.sum()) == 0:
        return {'n': 0, 'mae_s': float('nan'), 'rmse_s': float('nan'), 'mape': float('nan')}
    err = predicted[mask].astype(float) - observed[mask].astype(float)
    denom = observed[mask].astype(float).replace(0, np.nan)
    return {
        'n': int(mask.sum()),
        'mae_s': float(np.abs(err).mean()),
        'rmse_s': float(math.sqrt(np.square(err).mean())),
        'mape': float((np.abs(err) / denom).mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Calibrate OSM road-type speeds from sampled OD benchmark times.'
    )
    parser.add_argument('country_code')
    parser.add_argument('--provider', choices=['dry-run', 'mapbox'], default='dry-run')
    parser.add_argument('--mapbox-profile', default='driving')
    parser.add_argument('--mapbox-token-env', default='MAPBOX_ACCESS_TOKEN')
    parser.add_argument('--mapbox-timeout-s', type=float, default=30.0)
    parser.add_argument('--request-sleep-s', type=float, default=0.2)
    parser.add_argument('--sample-size', type=int, default=30)
    parser.add_argument('--nearest-sources', type=int, default=2)
    parser.add_argument('--max-pairs', type=int, default=40)
    parser.add_argument('--trace-coordinate-limit', type=int, default=25)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--aggregate-factor', type=int, default=20)
    parser.add_argument('--population-threshold', type=float, default=1.0)
    parser.add_argument('--amenity', nargs='+', default=DEFAULT_AMENITIES)
    parser.add_argument('--network-backend', choices=['pyrosm', 'osmium', 'auto'], default='osmium')
    parser.add_argument('--population-path')
    parser.add_argument('--sources-path')
    parser.add_argument('--bbox', nargs=4, type=float)
    parser.add_argument('--output-dir')
    parser.add_argument('--quiet', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.verbose = not args.quiet

    cfg = load_cfg(args.country_code)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.BASE_DIR / 'speed_calibration'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = pc()
    if not cfg.PBF_PATH.exists():
        raise FileNotFoundError(
            f'OSM PBF not found: {cfg.PBF_PATH}. '
            'Run run_pipeline.py for this country once, or copy the PBF into the country data folder.'
        )
    if not args.population_path and not cfg.WORLDPOP_PATH.exists():
        raise FileNotFoundError(
            f'WorldPop raster not found: {cfg.WORLDPOP_PATH}. '
            'Run run_pipeline.py for this country once, or pass --population-path with snapped points.'
        )

    if args.verbose:
        print(f'Loading {cfg.COUNTRY_NAME} network from {cfg.PBF_PATH}')
    nodes, edges = load_osm_network_data(
        cfg.PBF_PATH,
        verbose=args.verbose,
        bbox=args.bbox,
        backend=args.network_backend,
    )

    population = _prepare_population(args=args, cfg=cfg, nodes=nodes)
    edges = add_edge_speeds(
        edges,
        default_speeds_kph=cfg.legal_speeds_kph,
        surface_multipliers=cfg.surface_speed_multipliers,
        fallback_speed_kph=25.0,
        general_speed_factor=cfg.speed_general_factor,
        population_points=population,
        projected_epsg=cfg.PROJECTED_EPSG,
        urban_density_threshold_pop_per_km2=cfg.urban_density_threshold_pop_per_km2,
        urban_density_speed_factor=cfg.urban_density_speed_factor,
        urban_density_radius_m=cfg.urban_density_radius_m,
        time_col='travel_time_s',
    )
    edges['base_time_s'] = edges['travel_time_s']

    if args.verbose:
        print('Building NetworkX graph for sampled route composition')
    graph = build_networkx_graph(
        nodes,
        edges,
        weight_col='length',
        bidirectional=False,
    )

    sources = _prepare_sources(args=args, cfg=cfg, nodes=nodes)

    sampled_origins = _sample_origins(
        population,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )
    pairs = _nearest_source_pairs(
        sampled_origins,
        sources,
        projected_epsg=cfg.PROJECTED_EPSG,
        nearest_sources=args.nearest_sources,
    ).head(args.max_pairs)

    token = None
    if args.provider == 'mapbox':
        token = _read_user_env_var(args.mapbox_token_env)
        if not token:
            raise RuntimeError(
                f'{args.mapbox_token_env} is not set in this process or the Windows user environment.'
            )

    rows: list[dict[str, Any]] = []
    for pos, pair in enumerate(pairs.itertuples(index=False), start=1):
        if args.verbose:
            print(f'Pair {pos:,}/{len(pairs):,}: {pair.origin_id} -> {pair.source_id}')

        try:
            path_nodes = shortest_path_nodes(
                graph,
                int(pair.origin_nearest_node),
                int(pair.source_nearest_node),
                weight='weight',
            )
        except Exception as exc:
            rows.append({**pair._asdict(), 'status': f'path error: {exc}'})
            continue

        highway_times = _path_highway_times(graph, path_nodes)
        model_time_s = float(sum(highway_times.values()))
        path_length_m = float(
            sum(
                graph[int(u)][int(v)]['length']
                for u, v in zip(path_nodes[:-1], path_nodes[1:])
            )
        )

        if args.provider == 'dry-run':
            benchmark = BenchmarkResult(
                duration_s=model_time_s,
                distance_m=path_length_m,
                provider='dry-run',
                status='ok',
            )
        else:
            coords = _sample_trace_coordinates(
                graph,
                path_nodes,
                max_coordinates=args.trace_coordinate_limit,
            )
            benchmark = _mapbox_trace_duration(
                coords,
                token=token or '',
                profile=args.mapbox_profile,
                timeout_s=args.mapbox_timeout_s,
            )
            sleep(args.request_sleep_s)

        row = {
            **pair._asdict(),
            'path_node_count': len(path_nodes),
            'path_length_m': path_length_m,
            'model_time_s': model_time_s,
            'observed_time_s': benchmark.duration_s,
            'benchmark_distance_m': benchmark.distance_m,
            'provider': benchmark.provider,
            'status': benchmark.status,
        }
        row.update({f'time_s__{key}': value for key, value in highway_times.items()})
        rows.append(row)

    sample = pd.DataFrame(rows)
    sample_path = output_dir / 'timor_leste_speed_calibration_sample_pairs.csv'
    if cfg.iso3 != 'TLS':
        sample_path = output_dir / f'{cfg.country_slug}_speed_calibration_sample_pairs.csv'
    sample.to_csv(sample_path, index=False)

    highway_columns = sorted(col for col in sample.columns if col.startswith('time_s__'))
    multipliers, beta = _fit_speed_multipliers(sample, highway_columns)
    class_multipliers = {
        col.replace('time_s__', '', 1): value
        for col, value in multipliers.items()
    }
    beta_by_class = {
        col.replace('time_s__', '', 1): value
        for col, value in beta.items()
    }

    sample['calibrated_time_s'] = sample['model_time_s']
    for col, multiplier in multipliers.items():
        sample['calibrated_time_s'] += sample[col].fillna(0.0) * (1.0 / multiplier - 1.0)
    sample.to_csv(sample_path, index=False)

    profile = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'country': cfg.COUNTRY_NAME,
        'country_code': args.country_code,
        'provider': args.provider,
        'profile': args.mapbox_profile if args.provider == 'mapbox' else None,
        'base_speeds_kph': cfg.legal_speeds_kph,
        'surface_speed_multipliers': cfg.surface_speed_multipliers,
        'speed_general_factor': cfg.speed_general_factor,
        'urban_density_threshold_pop_per_km2': cfg.urban_density_threshold_pop_per_km2,
        'urban_density_speed_factor': cfg.urban_density_speed_factor,
        'urban_density_radius_m': cfg.urban_density_radius_m,
        'speed_multipliers_by_highway': class_multipliers,
        'inverse_speed_multipliers_beta_by_highway': beta_by_class,
        'sample_size_requested': args.sample_size,
        'nearest_sources': args.nearest_sources,
        'max_pairs': args.max_pairs,
        'sample_pairs': sample_path.as_posix(),
        'metrics_base': _metrics(sample['observed_time_s'], sample['model_time_s']),
        'metrics_calibrated': _metrics(sample['observed_time_s'], sample['calibrated_time_s']),
        'runtime_s': pc() - t0,
    }
    profile_path = output_dir / f'{cfg.country_slug}_speed_profile_{args.provider}.yaml'
    _write_yaml(profile_path, profile)

    # Materialize calibrated edge weights for immediate Pandana POC use.
    calibrated_edges = edges.copy()
    calibrated_edges['calibrated_time_s'] = calibrated_edges['base_time_s']
    for highway, multiplier in class_multipliers.items():
        mask = calibrated_edges['highway'].map(normalize_highway) == highway
        calibrated_edges.loc[mask, 'calibrated_time_s'] = (
            calibrated_edges.loc[mask, 'base_time_s'] / multiplier
        )
    edge_path = output_dir / f'{cfg.country_slug}_calibrated_edges_{args.provider}.parquet'
    calibrated_edges.to_parquet(edge_path)

    # Smoke-build the Pandana time network. This is the propagation step.
    if args.verbose:
        print('Building Pandana network with distance and time weights')
    _ = build_pandana_network(
        nodes=nodes,
        edges=calibrated_edges,
        weight_cols=('length', 'length_m', 'travel_time_s', 'calibrated_time_s'),
    )

    print(f'Wrote sampled pairs: {sample_path}')
    print(f'Wrote speed profile: {profile_path}')
    print(f'Wrote calibrated edges: {edge_path}')
    print(json.dumps(profile['metrics_calibrated'], indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
