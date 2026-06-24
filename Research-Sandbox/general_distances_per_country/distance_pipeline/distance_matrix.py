# Standard library imports
import json
import shutil
from pathlib import Path
import warnings
from time import perf_counter as pc, time_ns

# Third-party library imports
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree

warnings.filterwarnings(
    'ignore',
    message=r'Unsigned integer: shortest path distance is trying to be calculated.*',
    category=UserWarning,
    module=r'pandana\.network',
)

NO_PATH_SENTINEL = 4294967.295
EARTH_RADIUS_KM = 6367.0
NODE_PAIR_KEY_COLUMNS = ['target_nearest_node', 'source_nearest_node']
NODE_PAIR_CACHE_BUCKET_COUNT = 256
DEFAULT_MAX_SPATIAL_PAIRS_PER_CHUNK = 25_000_000


def _safe_impedance_name(imp_name: str) -> str:
    '''Return a filesystem-friendly impedance name for cache subfolders.'''
    return ''.join(char if char.isalnum() else '_' for char in imp_name).strip('_')


def _node_pair_cache_dir_for_impedance(
    cache_dir: Path,
    imp_name: str | None,
) -> Path:
    '''
    Keep the historical distance cache path for the default impedance, and use
    subfolders for named non-default impedances such as travel_time_s.
    '''
    if imp_name in (None, '', 'length'):
        return cache_dir

    safe_name = _safe_impedance_name(imp_name)
    if not safe_name:
        raise ValueError(f'Invalid Pandana impedance name: {imp_name!r}')
    return cache_dir / f'impedance_{safe_name}'


def _shortest_path_lengths(
    network: object,
    target_nodes: np.ndarray,
    source_nodes: np.ndarray,
    *,
    imp_name: str | None,
) -> np.ndarray:
    '''Call Pandana shortest paths with an optional named impedance.'''
    if imp_name is None:
        impedance_names = getattr(network, 'impedance_names', None)
        if impedance_names is not None and len(impedance_names) > 1:
            if 'length' in impedance_names:
                return network.shortest_path_lengths(
                    target_nodes,
                    source_nodes,
                    imp_name='length',
                )
        return network.shortest_path_lengths(target_nodes, source_nodes)
    return network.shortest_path_lengths(
        target_nodes,
        source_nodes,
        imp_name=imp_name,
    )


def _get_target_coordinate_columns(targets: pd.DataFrame) -> tuple[str, str]:
    '''
    Return the target longitude and latitude column names.

    Parameters
    ----------
    targets
        Target table.

    Returns
    -------
    tuple[str, str]
        Pair ``(x_col, y_col)``.

    Raises
    ------
    KeyError
        If no supported target coordinate columns are found.
    '''
    if {'xcoord', 'ycoord'}.issubset(targets.columns):
        return 'xcoord', 'ycoord'

    if {'Longitude', 'Latitude'}.issubset(targets.columns):
        return 'Longitude', 'Latitude'

    raise KeyError(
        'Missing target coordinate columns. '
        "Expected either ['xcoord', 'ycoord'] or ['Longitude', 'Latitude']"
    )


def _validate_inputs(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Validate that the input tables contain the expected columns and index layout.

    Parameters
    ----------
    targets
        Target table indexed by target ID.
    sources
        Source table indexed by source ID.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Possibly renamed target and source tables, with backward compatible
        snap distance column names normalized to the current API.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If the index does not match the corresponding ID column.
    '''
    _get_target_coordinate_columns(targets)

    if 'dist_snap_target' not in targets.columns:
        if 'pop_dist_road_estrada' in targets.columns:
            targets = targets.rename(
                columns={'pop_dist_road_estrada': 'dist_snap_target'}
            )
        else:
            raise KeyError("Missing target columns: ['dist_snap_target']")

    if 'dist_snap_source' not in sources.columns:
        if 'hosp_dist_road_estrada' in sources.columns:
            sources = sources.rename(
                columns={'hosp_dist_road_estrada': 'dist_snap_source'}
            )
        else:
            raise KeyError("Missing source columns: ['dist_snap_source']")

    required_target_cols = {
        'ID',
        'nearest_node',
        'dist_snap_target',
    }
    required_source_cols = {
        'ID',
        'Longitude',
        'Latitude',
        'nearest_node',
        'dist_snap_source',
    }

    missing_targets = required_target_cols.difference(targets.columns)
    missing_sources = required_source_cols.difference(sources.columns)

    if missing_targets:
        raise KeyError(f'Missing target columns: {sorted(missing_targets)}')
    if missing_sources:
        raise KeyError(f'Missing source columns: {sorted(missing_sources)}')

    target_index_values = np.asarray(targets.index)
    target_id_values = targets['ID'].to_numpy(copy=False)

    source_index_values = np.asarray(sources.index)
    source_id_values = sources['ID'].to_numpy(copy=False)

    if not np.array_equal(target_index_values, target_id_values):
        raise ValueError("targets.index must match targets['ID']")

    if not np.array_equal(source_index_values, source_id_values):
        raise ValueError("sources.index must match sources['ID']")

    return targets, sources


def _validate_node_columns(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
) -> None:
    '''
    Validate that nearest node columns are present and contain no missing values.

    Parameters
    ----------
    targets
        Target table.
    sources
        Source table.

    Raises
    ------
    ValueError
        If either nearest node column contains missing values.
    '''
    if targets['nearest_node'].isna().any():
        n_missing = int(targets['nearest_node'].isna().sum())
        raise ValueError(
            f"targets['nearest_node'] contains {n_missing:,} missing values"
        )

    if sources['nearest_node'].isna().any():
        n_missing = int(sources['nearest_node'].isna().sum())
        raise ValueError(
            f"sources['nearest_node'] contains {n_missing:,} missing values"
        )


def _dense_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Map arbitrary identifiers to dense zero based codes.

    Parameters
    ----------
    values
        One dimensional array of identifiers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair ``(codes, unique_values)`` such that
        ``unique_values[codes] == values``.
    '''
    unique_values, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int64, copy=False), unique_values


def _unique_node_pairs(
    target_nodes: np.ndarray,
    source_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Deduplicate aligned node pairs safely for large sparse identifiers such as
    OSM node IDs.

    Parameters
    ----------
    target_nodes
        Target nearest node IDs.
    source_nodes
        Source nearest node IDs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Unique aligned target and source nearest node IDs.
    '''
    if target_nodes.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    source_nodes = np.asarray(source_nodes, dtype=np.int64)

    target_codes, target_unique = _dense_codes(target_nodes)
    source_codes, source_unique = _dense_codes(source_nodes)

    base = np.uint64(len(source_unique))
    keys = target_codes.astype(np.uint64) * base + source_codes.astype(np.uint64)
    unique_keys = np.unique(keys)

    unique_target_codes = (unique_keys // base).astype(np.int64, copy=False)
    unique_source_codes = (unique_keys % base).astype(np.int64, copy=False)

    unique_target_nodes = np.asarray(
        target_unique[unique_target_codes],
        dtype=np.int64,
    )
    unique_source_nodes = np.asarray(
        source_unique[unique_source_codes],
        dtype=np.int64,
    )

    return unique_target_nodes, unique_source_nodes


def _candidate_row_pairs(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    distance_threshold_largest: float,
    *,
    max_total_dist: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute candidate target source row pairs using a spatial prefilter.

    Parameters
    ----------
    targets
        Target table.
    sources
        Source table.
    distance_threshold_largest
        Maximum crow flies prefilter distance in kilometers.
    max_total_dist
        Optional maximum total distance in meters. Candidate pairs whose
        source and target stitch distances already exceed this value are
        discarded before road routing.
    verbose
        Whether to print progress messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(target_indices, source_indices)`` containing aligned
        candidate row pairs.
    '''
    t = pc()

    target_x_col, target_y_col = _get_target_coordinate_columns(targets)

    target_coords = targets[[target_x_col, target_y_col]].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    source_coords = sources[['Longitude', 'Latitude']].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    target_coords_rad = np.radians(target_coords)
    source_coords_rad = np.radians(source_coords)

    if verbose:
        print(
            f'preparing {len(target_coords_rad):,} x {len(source_coords_rad):,} '
            f'for spatial nearest neighbors bounded by {distance_threshold_largest} km '
            f'in {pc() - t:.2f} seconds'
        )

    t = pc()

    tree = cKDTree(source_coords_rad)
    radius = distance_threshold_largest / EARTH_RADIUS_KM
    indices = tree.query_ball_point(target_coords_rad, r=radius)

    lengths = np.fromiter((len(row) for row in indices), dtype=np.int64)
    n_candidate_pairs = int(lengths.sum())

    target_indices = np.repeat(np.arange(len(indices), dtype=np.int64), lengths)
    source_indices = np.fromiter(
        (source_idx for row in indices for source_idx in row),
        dtype=np.int64,
        count=n_candidate_pairs,
    )

    if verbose:
        print(
            f'finding {n_candidate_pairs:,} pairs of spatial nearest neighbors '
            f'in {pc() - t:.2f} seconds'
        )

    if n_candidate_pairs == 0:
        return target_indices, source_indices

    keep = np.ones(n_candidate_pairs, dtype=bool)

    if max_total_dist is not None:
        target_snap = targets['dist_snap_target'].to_numpy(
            dtype=np.float64,
            copy=False,
        )[target_indices]
        source_snap = sources['dist_snap_source'].to_numpy(
            dtype=np.float64,
            copy=False,
        )[source_indices]
        keep &= (target_snap + source_snap) <= max_total_dist

    if 'component_id' in targets.columns and 'component_id' in sources.columns:
        target_components = targets['component_id'].to_numpy(
            dtype=np.int64,
            copy=False,
        )[target_indices]
        source_components = sources['component_id'].to_numpy(
            dtype=np.int64,
            copy=False,
        )[source_indices]
        keep &= (target_components >= 0) & (target_components == source_components)

    if not keep.all():
        kept = int(keep.sum())
        if verbose:
            print(
                f'pruned {n_candidate_pairs - kept:,} spatial candidate pairs '
                f'before routing; {kept:,} remain'
            )
        target_indices = target_indices[keep]
        source_indices = source_indices[keep]

    return target_indices, source_indices


def _source_spatial_tree(
    sources: pd.DataFrame,
    distance_threshold_largest: float,
) -> tuple[cKDTree, float]:
    '''Return a KD-tree and angular radius for sparse prefilter estimates.'''
    source_coords = sources[['Longitude', 'Latitude']].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    source_coords_rad = np.radians(source_coords)
    return cKDTree(source_coords_rad), distance_threshold_largest / EARTH_RADIUS_KM


def _estimate_spatial_candidate_pairs(
    targets: pd.DataFrame,
    *,
    source_tree: cKDTree,
    radius: float,
) -> int:
    '''Count spatial prefilter pairs without materializing their indices.'''
    if targets.empty:
        return 0

    target_x_col, target_y_col = _get_target_coordinate_columns(targets)
    target_coords = targets[[target_x_col, target_y_col]].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    target_coords_rad = np.radians(target_coords)
    lengths = source_tree.query_ball_point(
        target_coords_rad,
        r=radius,
        return_length=True,
    )
    return int(np.asarray(lengths, dtype=np.int64).sum())


def _adaptive_target_chunk_stop(
    targets: pd.DataFrame,
    *,
    chunk_start: int,
    requested_chunk_stop: int,
    source_tree: cKDTree,
    radius: float,
    max_spatial_pairs_per_chunk: int | None,
    verbose: bool = False,
) -> tuple[int, int | None]:
    '''Shrink a target chunk until spatial pair materialization is bounded.'''
    if max_spatial_pairs_per_chunk is None:
        return requested_chunk_stop, None

    chunk_stop = requested_chunk_stop
    while True:
        estimate = _estimate_spatial_candidate_pairs(
            targets.iloc[chunk_start:chunk_stop],
            source_tree=source_tree,
            radius=radius,
        )
        chunk_len = chunk_stop - chunk_start
        if estimate <= max_spatial_pairs_per_chunk or chunk_len <= 1:
            if verbose and chunk_stop < requested_chunk_stop:
                print(
                    'adjusted sparse matrix target chunk '
                    f'{chunk_start:,}:{requested_chunk_stop:,} to '
                    f'{chunk_start:,}:{chunk_stop:,}; estimated '
                    f'{estimate:,} spatial candidate pairs within cap '
                    f'{max_spatial_pairs_per_chunk:,}'
                )
            return chunk_stop, estimate

        suggested_len = int(
            chunk_len * (max_spatial_pairs_per_chunk / max(estimate, 1)) * 0.9
        )
        new_len = max(1, min(chunk_len - 1, suggested_len))
        if new_len >= chunk_len:
            new_len = max(1, chunk_len // 2)
        chunk_stop = chunk_start + new_len


def _empty_node_pair_distances() -> pl.DataFrame:
    '''Return an empty node-pair distance table with the canonical schema.'''
    return pl.DataFrame(
        schema={
            'target_nearest_node': pl.Int64,
            'source_nearest_node': pl.Int64,
            'road_distance': pl.Float64,
        }
    )


def _node_pair_cache_bucket_id(target_nearest_node: int) -> int:
    '''Return the cache bucket ID for a target road-node ID.'''
    return int(target_nearest_node) % NODE_PAIR_CACHE_BUCKET_COUNT


def _node_pair_cache_files(cache_dir: Path) -> list[Path]:
    '''Return legacy flat parquet chunks in a node-pair distance cache directory.'''
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob('node_pairs_*.parquet'))


def _node_pair_cache_bucket_dir(cache_dir: Path, bucket_id: int) -> Path:
    '''Return the directory for one node-pair cache bucket.'''
    return cache_dir / f'bucket={bucket_id:03d}'


def _node_pair_cache_bucket_files(cache_dir: Path, bucket_id: int) -> list[Path]:
    '''Return parquet chunks for one node-pair distance cache bucket.'''
    bucket_dir = _node_pair_cache_bucket_dir(cache_dir, bucket_id)
    if not bucket_dir.exists():
        return []
    return sorted(bucket_dir.glob('node_pairs_*.parquet'))


def _load_cached_node_pair_distances(
    cache_dir: Path,
    required_pairs: pl.DataFrame,
    *,
    verbose: bool = False,
) -> pl.DataFrame:
    '''Load cached distances for the required road-node pairs.'''
    if required_pairs.height == 0:
        return _empty_node_pair_distances()

    t = pc()

    cached_parts: list[pl.DataFrame] = []

    legacy_cache_files = _node_pair_cache_files(cache_dir)
    if legacy_cache_files:
        cached_parts.append(
            required_pairs.lazy()
            .join(
                pl.scan_parquet([str(path) for path in legacy_cache_files])
                .select([*NODE_PAIR_KEY_COLUMNS, 'road_distance'])
                .unique(subset=NODE_PAIR_KEY_COLUMNS, keep='last'),
                on=NODE_PAIR_KEY_COLUMNS,
                how='inner',
            )
            .collect()
        )

    required_with_bucket = required_pairs.with_columns(
        (
            pl.col('target_nearest_node') % NODE_PAIR_CACHE_BUCKET_COUNT
        ).cast(pl.Int64).alias('_cache_bucket')
    )
    bucket_ids = required_with_bucket['_cache_bucket'].unique().sort().to_list()
    bucket_file_count = 0

    for bucket_id in bucket_ids:
        bucket_files = _node_pair_cache_bucket_files(cache_dir, int(bucket_id))
        if not bucket_files:
            continue
        bucket_file_count += len(bucket_files)
        bucket_required = required_with_bucket.filter(
            pl.col('_cache_bucket') == bucket_id
        ).select(NODE_PAIR_KEY_COLUMNS)
        cached_parts.append(
            bucket_required.lazy()
            .join(
                pl.scan_parquet([str(path) for path in bucket_files])
                .select([*NODE_PAIR_KEY_COLUMNS, 'road_distance'])
                .unique(subset=NODE_PAIR_KEY_COLUMNS, keep='last'),
                on=NODE_PAIR_KEY_COLUMNS,
                how='inner',
            )
            .collect()
        )

    if cached_parts:
        cached = (
            pl.concat(cached_parts, how='vertical')
            .unique(subset=NODE_PAIR_KEY_COLUMNS, keep='last')
        )
    else:
        cached = _empty_node_pair_distances()

    if verbose:
        print(
            f'loaded {cached.height:,} cached road-node pair distances '
            f'from {len(legacy_cache_files) + bucket_file_count:,} chunk(s) '
            f'in {pc() - t:.2f} seconds'
        )

    return cached


def _write_node_pair_cache_chunk(
    cache_dir: Path,
    distances: pl.DataFrame,
    *,
    verbose: bool = False,
) -> list[Path]:
    '''Append parquet chunks to the reusable bucketed node-pair distance cache.'''
    if distances.height == 0:
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)

    t = pc()
    written_paths: list[Path] = []
    distances = distances.with_columns(
        (
            pl.col('target_nearest_node') % NODE_PAIR_CACHE_BUCKET_COUNT
        ).cast(pl.Int64).alias('_cache_bucket')
    )
    for bucket_id in distances['_cache_bucket'].unique().sort().to_list():
        bucket_distances = distances.filter(
            pl.col('_cache_bucket') == bucket_id
        ).drop('_cache_bucket')
        bucket_dir = _node_pair_cache_bucket_dir(cache_dir, int(bucket_id))
        bucket_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = bucket_dir / (
            f'node_pairs_{time_ns()}_{bucket_distances.height}.parquet'
        )
        tmp_path = chunk_path.with_suffix('.tmp.parquet')
        bucket_distances.write_parquet(tmp_path)
        tmp_path.replace(chunk_path)
        written_paths.append(chunk_path)

    if verbose:
        print(
            f'wrote {distances.height:,} road-node pair distances to '
            f'{len(written_paths):,} bucketed chunk(s) in {pc() - t:.2f} seconds'
        )

    return written_paths


def _compute_node_pair_distances_from_arrays(
    target_nodes: np.ndarray,
    source_nodes: np.ndarray,
    network: object,
    *,
    keep_unreachable: bool,
    imp_name: str | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    '''Compute Pandana impedances for aligned unique road-node arrays.'''
    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    source_nodes = np.asarray(source_nodes, dtype=np.int64)

    if target_nodes.dtype.kind != 'i' or source_nodes.dtype.kind != 'i':
        raise TypeError(
            'Pandana node ids must be integer dtype, got '
            f'{target_nodes.dtype} and {source_nodes.dtype}'
        )

    if target_nodes.size == 0:
        return _empty_node_pair_distances()

    t = pc()
    road_distance = np.asarray(
        _shortest_path_lengths(
            network,
            target_nodes,
            source_nodes,
            imp_name=imp_name,
        ),
        dtype=np.float64,
    )
    elapsed = pc() - t

    valid = road_distance < NO_PATH_SENTINEL

    if verbose:
        print(
            f'{len(road_distance):,} shortest paths of which {int(valid.sum()):,} exist '
            f'found in {elapsed:.2f} seconds'
        )

    if keep_unreachable:
        stored_distance = road_distance.copy()
        stored_distance[~valid] = np.inf
        return pl.DataFrame(
            {
                'target_nearest_node': target_nodes,
                'source_nearest_node': source_nodes,
                'road_distance': stored_distance,
            }
        )

    return pl.DataFrame(
        {
            'target_nearest_node': target_nodes[valid],
            'source_nearest_node': source_nodes[valid],
            'road_distance': road_distance[valid],
        }
    )


def _compute_unique_node_pair_distances(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    network: object,
    *,
    node_pair_cache_dir: Path | None = None,
    imp_name: str | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    '''
    Compute shortest path distances for unique nearest node pairs.

    Parameters
    ----------
    targets
        Target table.
    sources
        Source table.
    target_indices
        Candidate target row indices.
    source_indices
        Candidate source row indices.
    network
        Pandana like network object exposing ``shortest_path_lengths``.
    verbose
        Whether to print progress messages.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        ``'target_nearest_node'``, ``'source_nearest_node'``,
        ``'road_distance'``.
    '''
    t = pc()

    target_nodes = targets['nearest_node'].to_numpy(dtype=np.int64, copy=False)[
        target_indices
    ]
    source_nodes = sources['nearest_node'].to_numpy(dtype=np.int64, copy=False)[
        source_indices
    ]

    unique_target_nodes, unique_source_nodes = _unique_node_pairs(
        target_nodes,
        source_nodes,
    )

    unique_target_nodes = np.asarray(unique_target_nodes, dtype=np.int64)
    unique_source_nodes = np.asarray(unique_source_nodes, dtype=np.int64)

    if verbose:
        print(
            f'creating {len(unique_target_nodes):,} unique target source '
            f'node pairs in {pc() - t:.2f} seconds'
        )

    if unique_target_nodes.size == 0:
        return _empty_node_pair_distances()

    required_pairs = pl.DataFrame(
        {
            'target_nearest_node': unique_target_nodes,
            'source_nearest_node': unique_source_nodes,
        }
    )

    if node_pair_cache_dir is None:
        return _compute_node_pair_distances_from_arrays(
            unique_target_nodes,
            unique_source_nodes,
            network,
            keep_unreachable=False,
            imp_name=imp_name,
            verbose=verbose,
        )

    node_pair_cache_dir = _node_pair_cache_dir_for_impedance(
        node_pair_cache_dir,
        imp_name,
    )

    cached_distances = _load_cached_node_pair_distances(
        node_pair_cache_dir,
        required_pairs,
        verbose=verbose,
    )

    missing_pairs = required_pairs.join(
        cached_distances.select(NODE_PAIR_KEY_COLUMNS),
        on=NODE_PAIR_KEY_COLUMNS,
        how='anti',
    )

    if verbose:
        print(
            f'road-node pair cache hit {cached_distances.height:,} / '
            f'{required_pairs.height:,}; computing {missing_pairs.height:,} missing'
        )

    if missing_pairs.height:
        missing_distances = _compute_node_pair_distances_from_arrays(
            missing_pairs['target_nearest_node'].to_numpy(),
            missing_pairs['source_nearest_node'].to_numpy(),
            network,
            keep_unreachable=True,
            imp_name=imp_name,
            verbose=verbose,
        )
        _write_node_pair_cache_chunk(
            node_pair_cache_dir,
            missing_distances,
            verbose=verbose,
        )
        all_distances = (
            missing_distances
            if cached_distances.height == 0
            else pl.concat([cached_distances, missing_distances], how='vertical')
        )
    else:
        all_distances = cached_distances

    return all_distances.filter(pl.col('road_distance').is_finite())


def _targets_to_polars(targets: pd.DataFrame) -> pl.DataFrame:
    '''
    Build the target side join table as a Polars DataFrame.

    Parameters
    ----------
    targets
        Target table.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        ``'target_id'``, ``'target_nearest_node'``,
        ``'target_to_road_dist'``.
    '''
    data = {
        'target_id': targets['ID'].to_numpy(copy=False),
        'target_nearest_node': targets['nearest_node'].to_numpy(
            dtype=np.int64,
            copy=False,
        ),
        'target_to_road_dist': targets['dist_snap_target'].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
    }
    if 'target_type' in targets.columns:
        data['target_type'] = targets['target_type'].astype(str).to_numpy(copy=False)
    return pl.DataFrame(data)


def _sources_to_polars(sources: pd.DataFrame) -> pl.DataFrame:
    '''
    Build the source side join table as a Polars DataFrame.

    Parameters
    ----------
    sources
        Source table.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        ``'source_id'``, ``'source_nearest_node'``,
        ``'source_to_road_dist'``.
    '''
    data = {
        'source_id': sources['ID'].to_numpy(copy=False),
        'source_nearest_node': sources['nearest_node'].to_numpy(
            dtype=np.int64,
            copy=False,
        ),
        'source_to_road_dist': sources['dist_snap_source'].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
    }
    if 'source_type' in sources.columns:
        data['source_type'] = sources['source_type'].astype(str).to_numpy(copy=False)
    return pl.DataFrame(data)


def _empty_distance_result(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    road_cost_col: str,
    total_cost_col: str,
) -> pl.DataFrame:
    '''Return an empty sparse distance table with the canonical schema.'''
    schema = {
        'target_id': pl.Int64,
        'source_id': pl.Int64,
        'source_nearest_node': pl.Int64,
        'target_nearest_node': pl.Int64,
        'target_to_road_dist': pl.Float64,
        road_cost_col: pl.Float64,
        'source_to_road_dist': pl.Float64,
        total_cost_col: pl.Float64,
    }
    if 'target_type' in targets.columns:
        schema['target_type'] = pl.Utf8
    if 'source_type' in sources.columns:
        schema['source_type'] = pl.Utf8
    return pl.DataFrame(schema=schema)


def _assemble_distance_result(
    distances_pl: pl.DataFrame,
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    max_total_dist: float | None,
    road_cost_col: str,
    total_cost_col: str,
    stitch_cost_factor: float,
    verbose: bool,
) -> pl.DataFrame:
    '''Join road-node distances back to source/target rows and filter totals.'''
    if distances_pl.height == 0:
        return _empty_distance_result(
            targets,
            sources,
            road_cost_col=road_cost_col,
            total_cost_col=total_cost_col,
        )

    targets_pl = _targets_to_polars(targets)
    sources_pl = _sources_to_polars(sources)

    t = pc()

    result = (
        distances_pl.lazy()
        .join(targets_pl.lazy(), on='target_nearest_node', how='inner')
        .join(sources_pl.lazy(), on='source_nearest_node', how='inner')
        .with_columns(
            (
                pl.col('target_to_road_dist') * stitch_cost_factor
                + pl.col('road_distance')
                + pl.col('source_to_road_dist') * stitch_cost_factor
            ).alias(total_cost_col)
        )
    )

    if road_cost_col != 'road_distance':
        result = result.with_columns(pl.col('road_distance').alias(road_cost_col))

    if max_total_dist is not None:
        result = result.filter(pl.col(total_cost_col) <= max_total_dist)

    output_columns = [
        'target_id',
        'source_id',
        'source_nearest_node',
        'target_nearest_node',
        'target_to_road_dist',
        road_cost_col,
        'source_to_road_dist',
        total_cost_col,
    ]
    if 'target_type' in targets.columns:
        output_columns.append('target_type')
    if 'source_type' in sources.columns:
        output_columns.append('source_type')

    result = result.select(output_columns).collect()

    if verbose:
        print(
            f'assembling {result.height:,} distances of interest '
            f'in {pc() - t:.2f} seconds'
        )

    return result


def _read_partitioned_distance_summary(output_dir: Path) -> dict[str, object] | None:
    '''Load metadata for an existing partitioned sparse matrix, if complete.'''
    success_path = output_dir / '_SUCCESS.json'
    if not success_path.exists():
        return None

    with success_path.open('r', encoding='utf-8') as handle:
        metadata = json.load(handle)

    part_files = sorted(output_dir.glob('part-*.parquet'))
    preview = (
        pl.read_parquet(part_files[0]).head()
        if part_files
        else None
    )
    metadata['preview'] = preview
    metadata['path'] = output_dir
    return metadata


def write_distances_polars_partitioned(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    output_dir: str | Path,
    *,
    max_total_dist: float | None = None,
    node_pair_cache_dir: str | Path | None = None,
    imp_name: str | None = None,
    road_cost_col: str = 'road_distance',
    total_cost_col: str = 'total_dist',
    stitch_cost_factor: float = 1.0,
    target_chunk_size: int = 10_000,
    max_spatial_pairs_per_chunk: int | None = DEFAULT_MAX_SPATIAL_PAIRS_PER_CHUNK,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    '''
    Stream sparse distances to a partitioned Parquet dataset by target chunks.

    This avoids materializing the full source-target candidate universe and the
    global road-node-pair deduplication arrays in memory. Existing behavior is
    preserved by ``compute_distances_polars``; this function is intended for
    large runs that need chunked on-disk output.
    '''
    targets, sources = _validate_inputs(targets, sources)
    _validate_node_columns(targets, sources)

    if stitch_cost_factor <= 0:
        raise ValueError('stitch_cost_factor must be positive')
    if target_chunk_size <= 0:
        raise ValueError('target_chunk_size must be positive')
    if (
        max_spatial_pairs_per_chunk is not None
        and max_spatial_pairs_per_chunk <= 0
    ):
        raise ValueError('max_spatial_pairs_per_chunk must be positive')
    if road_cost_col == total_cost_col:
        raise ValueError('road_cost_col and total_cost_col must be different')

    output_dir = Path(output_dir)
    existing = _read_partitioned_distance_summary(output_dir)
    if existing is not None and not overwrite:
        if verbose:
            print(f'using existing partitioned sparse matrix: {output_dir}')
        return existing
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f'Partitioned sparse matrix output exists but is incomplete: {output_dir}'
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_pair_cache_path = (
        None if node_pair_cache_dir is None else Path(node_pair_cache_dir)
    )

    row_count = 0
    part_count = 0
    empty_chunk_count = 0
    chunk_records: list[dict[str, object]] = []
    preview: pl.DataFrame | None = None
    t_total = pc()
    source_tree, spatial_radius = _source_spatial_tree(
        sources,
        distance_threshold_largest,
    )

    chunk_start = 0
    while chunk_start < len(targets):
        requested_chunk_stop = min(chunk_start + target_chunk_size, len(targets))
        chunk_stop, estimated_spatial_pairs = _adaptive_target_chunk_stop(
            targets,
            chunk_start=chunk_start,
            requested_chunk_stop=requested_chunk_stop,
            source_tree=source_tree,
            radius=spatial_radius,
            max_spatial_pairs_per_chunk=max_spatial_pairs_per_chunk,
            verbose=verbose,
        )
        target_chunk = targets.iloc[chunk_start:chunk_stop]

        if verbose:
            print(
                f'computing sparse matrix target chunk '
                f'{chunk_start:,}:{chunk_stop:,} of {len(targets):,}'
            )

        target_indices, source_indices = _candidate_row_pairs(
            targets=target_chunk,
            sources=sources,
            distance_threshold_largest=distance_threshold_largest,
            max_total_dist=max_total_dist,
            verbose=verbose,
        )

        distances_pl = _compute_unique_node_pair_distances(
            targets=target_chunk,
            sources=sources,
            target_indices=target_indices,
            source_indices=source_indices,
            network=network,
            node_pair_cache_dir=node_pair_cache_path,
            imp_name=imp_name,
            verbose=verbose,
        )

        result = _assemble_distance_result(
            distances_pl=distances_pl,
            targets=target_chunk,
            sources=sources,
            max_total_dist=max_total_dist,
            road_cost_col=road_cost_col,
            total_cost_col=total_cost_col,
            stitch_cost_factor=stitch_cost_factor,
            verbose=verbose,
        )

        if result.height == 0:
            empty_chunk_count += 1
            chunk_records.append(
                {
                    'part_name': None,
                    'target_start': int(chunk_start),
                    'target_stop': int(chunk_stop),
                    'target_count': int(chunk_stop - chunk_start),
                    'requested_target_stop': int(requested_chunk_stop),
                    'adjusted': bool(chunk_stop < requested_chunk_stop),
                    'estimated_spatial_candidate_pairs': estimated_spatial_pairs,
                    'materialized_spatial_candidate_pairs': int(len(target_indices)),
                    'unique_node_pair_count': int(distances_pl.height),
                    'sparse_row_count': 0,
                }
            )
            chunk_start = chunk_stop
            continue

        if preview is None:
            preview = result.head()

        part_path = output_dir / f'part-{part_count:06d}.parquet'
        result.write_parquet(part_path, compression='zstd')
        row_count += result.height
        chunk_records.append(
            {
                'part_name': part_path.name,
                'target_start': int(chunk_start),
                'target_stop': int(chunk_stop),
                'target_count': int(chunk_stop - chunk_start),
                'requested_target_stop': int(requested_chunk_stop),
                'adjusted': bool(chunk_stop < requested_chunk_stop),
                'estimated_spatial_candidate_pairs': estimated_spatial_pairs,
                'materialized_spatial_candidate_pairs': int(len(target_indices)),
                'unique_node_pair_count': int(distances_pl.height),
                'sparse_row_count': int(result.height),
            }
        )
        part_count += 1

        if verbose:
            print(
                f'wrote {result.height:,} sparse distance rows to '
                f'{part_path.name}'
            )

        chunk_start = chunk_stop

    nonempty_records = [record for record in chunk_records if record['part_name']]
    metadata: dict[str, object] = {
        'path': str(output_dir),
        'row_count': row_count,
        'part_count': part_count,
        'empty_chunk_count': empty_chunk_count,
        'target_chunk_size': target_chunk_size,
        'max_spatial_pairs_per_chunk': max_spatial_pairs_per_chunk,
        'target_count': len(targets),
        'source_count': len(sources),
        'distance_threshold_km': distance_threshold_largest,
        'max_total_dist': max_total_dist,
        'elapsed_seconds': pc() - t_total,
        'chunking': {
            'requested_target_chunk_size': int(target_chunk_size),
            'max_spatial_pairs_per_chunk': max_spatial_pairs_per_chunk,
            'chunk_count': len(chunk_records),
            'adjusted_chunk_count': sum(
                1 for record in chunk_records if record['adjusted']
            ),
            'min_target_count': (
                min(record['target_count'] for record in chunk_records)
                if chunk_records else 0
            ),
            'max_target_count': (
                max(record['target_count'] for record in chunk_records)
                if chunk_records else 0
            ),
            'max_estimated_spatial_candidate_pairs': (
                max(
                    record['estimated_spatial_candidate_pairs'] or 0
                    for record in chunk_records
                ) if chunk_records else 0
            ),
            'max_materialized_spatial_candidate_pairs': (
                max(
                    record['materialized_spatial_candidate_pairs']
                    for record in chunk_records
                ) if chunk_records else 0
            ),
            'max_unique_node_pair_count': (
                max(record['unique_node_pair_count'] for record in chunk_records)
                if chunk_records else 0
            ),
            'max_sparse_row_count': (
                max(record['sparse_row_count'] for record in nonempty_records)
                if nonempty_records else 0
            ),
        },
        'chunks': chunk_records,
    }

    success_path = output_dir / '_SUCCESS.json'
    with success_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    metadata['path'] = output_dir
    metadata['preview'] = (
        preview
        if preview is not None
        else _empty_distance_result(
            targets,
            sources,
            road_cost_col=road_cost_col,
            total_cost_col=total_cost_col,
        )
    )

    if verbose:
        print(
            f'wrote partitioned sparse matrix with {row_count:,} row(s) '
            f'in {part_count:,} part(s) to {output_dir} '
            f'in {metadata["elapsed_seconds"]:.2f} seconds'
        )

    return metadata


def _dense_distance_frame(
    values: np.ndarray,
    *,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
) -> pd.DataFrame:
    '''Return a target-by-source dense matrix with stable axis names.'''
    frame = pd.DataFrame(values, index=target_ids, columns=source_ids)
    frame.index.name = 'target_id'
    frame.columns.name = 'source_id'
    return frame


def compute_dense_distance_matrices(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    network: object,
    *,
    max_total_dist: float | None = None,
    imp_name: str | None = None,
    road_matrix_name: str = 'road_distance',
    stitch_cost_factor: float = 1.0,
    chunk_size: int = 1_000_000,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    '''
    Compute dense target-by-source distance matrices.

    Unreachable road paths are represented by ``np.inf``. ``imp_name`` can be
    used to request a named Pandana impedance, for example ``travel_time_s``.
    When
    ``max_total_dist`` is provided, entries above that total-distance cap are
    also set to ``np.inf`` in the total matrix.
    '''
    targets, sources = _validate_inputs(targets, sources)
    _validate_node_columns(targets, sources)

    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if stitch_cost_factor <= 0:
        raise ValueError('stitch_cost_factor must be positive')
    if road_matrix_name in {'total', 'origin_stitch', 'destination_stitch'}:
        raise ValueError(
            'road_matrix_name must not be one of '
            "'total', 'origin_stitch', or 'destination_stitch'"
        )

    target_ids = targets['ID'].to_numpy(copy=False)
    source_ids = sources['ID'].to_numpy(copy=False)
    n_targets = len(targets)
    n_sources = len(sources)
    n_pairs = n_targets * n_sources

    if verbose:
        print(
            f'computing dense {n_targets:,} x {n_sources:,} matrix '
            f'({n_pairs:,} paths requested)'
        )

    target_nodes = targets['nearest_node'].to_numpy(dtype=np.int64, copy=False)
    source_nodes = sources['nearest_node'].to_numpy(dtype=np.int64, copy=False)
    road_flat = np.empty(n_pairs, dtype=np.float64)

    t = pc()
    for start in range(0, n_pairs, chunk_size):
        stop = min(start + chunk_size, n_pairs)
        pair_indices = np.arange(start, stop, dtype=np.int64)
        target_idx = pair_indices // n_sources
        source_idx = pair_indices % n_sources
        road_flat[start:stop] = np.asarray(
            _shortest_path_lengths(
                network,
                target_nodes[target_idx],
                source_nodes[source_idx],
                imp_name=imp_name,
            ),
            dtype=np.float64,
        )

    road_flat[road_flat >= NO_PATH_SENTINEL] = np.inf
    road = road_flat.reshape((n_targets, n_sources))

    target_to_road = targets['dist_snap_target'].to_numpy(
        dtype=np.float64,
        copy=False,
    )[:, None]
    source_to_road = sources['dist_snap_source'].to_numpy(
        dtype=np.float64,
        copy=False,
    )[None, :]
    total = target_to_road * stitch_cost_factor + road + source_to_road * stitch_cost_factor

    if max_total_dist is not None:
        total = total.copy()
        total[total > max_total_dist] = np.inf

    if verbose:
        finite_paths = int(np.isfinite(road).sum())
        print(
            f'dense routing found {finite_paths:,} finite road paths '
            f'in {pc() - t:.2f} seconds'
        )

    return {
        'total': _dense_distance_frame(
            total,
            target_ids=target_ids,
            source_ids=source_ids,
        ),
        'origin_stitch': _dense_distance_frame(
            np.broadcast_to(source_to_road, (n_targets, n_sources)).copy(),
            target_ids=target_ids,
            source_ids=source_ids,
        ),
        'destination_stitch': _dense_distance_frame(
            np.broadcast_to(target_to_road, (n_targets, n_sources)).copy(),
            target_ids=target_ids,
            source_ids=source_ids,
        ),
        road_matrix_name: _dense_distance_frame(
            road,
            target_ids=target_ids,
            source_ids=source_ids,
        ),
    }


def compute_distances_polars(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    *,
    max_total_dist: float | None = None,
    node_pair_cache_dir: str | Path | None = None,
    imp_name: str | None = None,
    road_cost_col: str = 'road_distance',
    total_cost_col: str = 'total_dist',
    stitch_cost_factor: float = 1.0,
    verbose: bool = False,
) -> pl.DataFrame:
    '''
    Compute sparse target to source distances and return a Polars DataFrame.

    Parameters
    ----------
    targets
        DataFrame indexed by target ID, with columns:
        ``'ID'``, coordinate columns either ``('xcoord', 'ycoord')`` or
        ``('Longitude', 'Latitude')``, ``'nearest_node'``,
        ``'dist_snap_target'``.
    sources
        DataFrame indexed by source ID, with columns:
        ``'ID'``, ``'Longitude'``, ``'Latitude'``, ``'nearest_node'``,
        ``'dist_snap_source'``.
    distance_threshold_largest
        Maximum crow flies distance in kilometers used to prefilter candidate
        target source pairs.
    network
        Pandana like network object exposing ``shortest_path_lengths``.
    max_total_dist
        Optional upper bound on total distance. If provided, only rows with
        ``total_dist <= max_total_dist`` are kept.
    node_pair_cache_dir
        Optional directory containing reusable road-node-pair distance parquet
        chunks. Missing node pairs are computed and appended to this cache.
    imp_name
        Optional Pandana impedance name. ``None`` preserves the historical
        shortest-distance behavior. Use values such as ``travel_time_s`` or
        ``calibrated_time_s`` with a network built with those edge weights.
    road_cost_col
        Output column name for the road-network impedance value. The default
        preserves the historical ``road_distance`` schema.
    total_cost_col
        Output column name for the total impedance plus source and target
        stitch distances. The default preserves the historical ``total_dist``
        schema.
    stitch_cost_factor
        Multiplier applied to source and target stitch distances before adding
        them to the road-network impedance. The default ``1.0`` preserves the
        historical meter-based total. For travel-time impedances, use seconds
        per meter, for example ``3.6 / speed_kph``.
    verbose
        Whether to print progress messages.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        ``'target_id'``, ``'source_id'``, ``'source_nearest_node'``,
        ``'target_nearest_node'``, ``'target_to_road_dist'``,
        ``'road_distance'``, ``'source_to_road_dist'``, ``'total_dist'``.
    '''
    targets, sources = _validate_inputs(targets, sources)
    _validate_node_columns(targets, sources)

    if stitch_cost_factor <= 0:
        raise ValueError('stitch_cost_factor must be positive')
    if road_cost_col == total_cost_col:
        raise ValueError('road_cost_col and total_cost_col must be different')

    if road_cost_col in {
        'target_id',
        'source_id',
        'source_nearest_node',
        'target_nearest_node',
        'target_to_road_dist',
        'source_to_road_dist',
    }:
        raise ValueError(f'road_cost_col conflicts with a required column: {road_cost_col}')
    if total_cost_col in {
        'target_id',
        'source_id',
        'source_nearest_node',
        'target_nearest_node',
        'target_to_road_dist',
        'road_distance',
        'source_to_road_dist',
    }:
        raise ValueError(f'total_cost_col conflicts with a required column: {total_cost_col}')

    target_indices, source_indices = _candidate_row_pairs(
        targets=targets,
        sources=sources,
        distance_threshold_largest=distance_threshold_largest,
        max_total_dist=max_total_dist,
        verbose=verbose,
    )

    distances_pl = _compute_unique_node_pair_distances(
        targets=targets,
        sources=sources,
        target_indices=target_indices,
        source_indices=source_indices,
        network=network,
        node_pair_cache_dir=(
            None if node_pair_cache_dir is None else Path(node_pair_cache_dir)
        ),
        imp_name=imp_name,
        verbose=verbose,
    )

    return _assemble_distance_result(
        distances_pl=distances_pl,
        targets=targets,
        sources=sources,
        max_total_dist=max_total_dist,
        road_cost_col=road_cost_col,
        total_cost_col=total_cost_col,
        stitch_cost_factor=stitch_cost_factor,
        verbose=verbose,
    )


def compute_distances(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    *,
    max_total_dist: float | None = None,
    node_pair_cache_dir: str | Path | None = None,
    imp_name: str | None = None,
    road_cost_col: str = 'road_distance',
    total_cost_col: str = 'total_dist',
    stitch_cost_factor: float = 1.0,
    verbose: bool = False,
) -> pd.DataFrame:
    '''
    Compute sparse target to source distances and return a pandas DataFrame.

    This is a thin wrapper around ``compute_distances_polars``.

    Parameters
    ----------
    targets
        DataFrame indexed by target ID, with columns:
        ``'ID'``, coordinate columns either ``('xcoord', 'ycoord')`` or
        ``('Longitude', 'Latitude')``, ``'nearest_node'``,
        ``'dist_snap_target'``.
    sources
        DataFrame indexed by source ID, with columns:
        ``'ID'``, ``'Longitude'``, ``'Latitude'``, ``'nearest_node'``,
        ``'dist_snap_source'``.
    distance_threshold_largest
        Maximum crow flies distance in kilometers used to prefilter candidate
        target source pairs.
    network
        Pandana like network object exposing ``shortest_path_lengths``.
    max_total_dist
        Optional upper bound on total distance. If provided, only rows with
        ``total_dist <= max_total_dist`` are kept.
    node_pair_cache_dir
        Optional directory containing reusable road-node-pair distance parquet
        chunks.
    imp_name
        Optional Pandana impedance name. ``None`` preserves historical
        shortest-distance behavior.
    road_cost_col
        Output column name for the road-network impedance value.
    total_cost_col
        Output column name for the total impedance plus source and target
        stitch distances.
    stitch_cost_factor
        Multiplier applied to source and target stitch distances before adding
        them to the road-network impedance.
    verbose
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns:
        ``'target_id'``, ``'source_id'``, ``'source_nearest_node'``,
        ``'target_nearest_node'``, ``'target_to_road_dist'``,
        ``'road_distance'``, ``'source_to_road_dist'``, ``'total_dist'``.
    '''
    return compute_distances_polars(
        targets=targets,
        sources=sources,
        distance_threshold_largest=distance_threshold_largest,
        network=network,
        max_total_dist=max_total_dist,
        node_pair_cache_dir=node_pair_cache_dir,
        imp_name=imp_name,
        road_cost_col=road_cost_col,
        total_cost_col=total_cost_col,
        stitch_cost_factor=stitch_cost_factor,
        verbose=verbose,
    ).to_pandas()
