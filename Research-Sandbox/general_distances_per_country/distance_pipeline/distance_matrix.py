# Standard library imports
import warnings
from time import perf_counter as pc

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

    return target_indices, source_indices


def _compute_unique_node_pair_distances(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    network: object,
    *,
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

    if unique_target_nodes.dtype.kind != 'i' or unique_source_nodes.dtype.kind != 'i':
        raise TypeError(
            'Pandana node ids must be integer dtype, got '
            f'{unique_target_nodes.dtype} and {unique_source_nodes.dtype}'
        )

    if verbose:
        print(
            f'creating {len(unique_target_nodes):,} unique target source '
            f'node pairs in {pc() - t:.2f} seconds'
        )

    if unique_target_nodes.size == 0:
        return pl.DataFrame(
            {
                'target_nearest_node': np.empty(0, dtype=np.int64),
                'source_nearest_node': np.empty(0, dtype=np.int64),
                'road_distance': np.empty(0, dtype=np.float64),
            }
        )

    t = pc()

    road_distance = np.asarray(
        network.shortest_path_lengths(unique_target_nodes, unique_source_nodes),
        dtype=np.float64,
    )
    elapsed = pc() - t

    valid = road_distance < NO_PATH_SENTINEL

    if verbose:
        print(
            f'{len(road_distance):,} shortest paths of which {int(valid.sum()):,} exist '
            f'found in {elapsed:.2f} seconds'
        )

    return pl.DataFrame(
        {
            'target_nearest_node': unique_target_nodes[valid],
            'source_nearest_node': unique_source_nodes[valid],
            'road_distance': road_distance[valid],
        }
    )


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
    return pl.DataFrame(
        {
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
    )


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
    return pl.DataFrame(
        {
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
    )


def compute_distances_polars(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    *,
    max_total_dist: float | None = None,
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

    target_indices, source_indices = _candidate_row_pairs(
        targets=targets,
        sources=sources,
        distance_threshold_largest=distance_threshold_largest,
        verbose=verbose,
    )

    distances_pl = _compute_unique_node_pair_distances(
        targets=targets,
        sources=sources,
        target_indices=target_indices,
        source_indices=source_indices,
        network=network,
        verbose=verbose,
    )

    if distances_pl.height == 0:
        return pl.DataFrame(
            schema={
                'target_id': pl.Int64,
                'source_id': pl.Int64,
                'source_nearest_node': pl.Int64,
                'target_nearest_node': pl.Int64,
                'target_to_road_dist': pl.Float64,
                'road_distance': pl.Float64,
                'source_to_road_dist': pl.Float64,
                'total_dist': pl.Float64,
            }
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
                pl.col('target_to_road_dist')
                + pl.col('road_distance')
                + pl.col('source_to_road_dist')
            ).alias('total_dist')
        )
    )

    if max_total_dist is not None:
        result = result.filter(pl.col('total_dist') <= max_total_dist)

    result = result.select(
        [
            'target_id',
            'source_id',
            'source_nearest_node',
            'target_nearest_node',
            'target_to_road_dist',
            'road_distance',
            'source_to_road_dist',
            'total_dist',
        ]
    ).collect()

    if verbose:
        print(
            f'assembling {result.height:,} distances of interest '
            f'in {pc() - t:.2f} seconds'
        )

    return result


def compute_distances(
    targets: pd.DataFrame,
    sources: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    *,
    max_total_dist: float | None = None,
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
        verbose=verbose,
    ).to_pandas()