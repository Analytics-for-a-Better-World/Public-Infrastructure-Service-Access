# Standard library imports
from time import perf_counter as pc
import warnings

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


def _validate_inputs(
    population: pd.DataFrame,
    all_hospitals: pd.DataFrame,
) -> None:
    """
    Validate that the input tables contain the expected columns and index layout.

    Parameters
    ----------
    population
        Population table indexed by population ID.
    all_hospitals
        Hospital table indexed by hospital ID.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If the index does not match the corresponding ID column.
    """
    required_pop_cols = {
        'ID',
        'xcoord',
        'ycoord',
        'nearest_node',
        'pop_dist_road_estrada',
    }
    required_hosp_cols = {
        'ID',
        'Longitude',
        'Latitude',
        'nearest_node',
        'hosp_dist_road_estrada',
    }

    missing_pop = required_pop_cols.difference(population.columns)
    missing_hosp = required_hosp_cols.difference(all_hospitals.columns)

    if missing_pop:
        raise KeyError(f'Missing population columns: {sorted(missing_pop)}')
    if missing_hosp:
        raise KeyError(f'Missing hospital columns: {sorted(missing_hosp)}')

    if not (population.index == population['ID']).all():
        raise ValueError("population.index must match population['ID']")

    if not (all_hospitals.index == all_hospitals['ID']).all():
        raise ValueError("all_hospitals.index must match all_hospitals['ID']")


def _dense_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map arbitrary identifiers to dense zero based codes.

    Parameters
    ----------
    values
        One dimensional array of identifiers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair `(codes, unique_values)` such that
        `unique_values[codes] == values`.
    """
    unique_values, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int64, copy=False), unique_values


def _unique_node_pairs(
    pop_nodes: np.ndarray,
    hosp_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate aligned node pairs safely for large sparse identifiers such as
    OSM node IDs.

    Parameters
    ----------
    pop_nodes
        Population nearest node IDs.
    hosp_nodes
        Hospital nearest node IDs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Unique aligned population and hospital nearest node IDs.
    """
    if pop_nodes.size == 0:
        return (
            np.empty(0, dtype=pop_nodes.dtype),
            np.empty(0, dtype=hosp_nodes.dtype),
        )

    pop_codes, pop_unique = _dense_codes(pop_nodes)
    hosp_codes, hosp_unique = _dense_codes(hosp_nodes)

    base = np.uint64(len(hosp_unique))
    keys = pop_codes.astype(np.uint64) * base + hosp_codes.astype(np.uint64)
    unique_keys = np.unique(keys)

    unique_pop_codes = (unique_keys // base).astype(np.int64, copy=False)
    unique_hosp_codes = (unique_keys % base).astype(np.int64, copy=False)

    unique_pop_nodes = pop_unique[unique_pop_codes]
    unique_hosp_nodes = hosp_unique[unique_hosp_codes]

    return unique_pop_nodes, unique_hosp_nodes


def _candidate_row_pairs(
    population: pd.DataFrame,
    all_hospitals: pd.DataFrame,
    distance_threshold_largest: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute candidate population hospital row pairs using a spatial prefilter.

    Parameters
    ----------
    population
        Population table.
    all_hospitals
        Hospital table.
    distance_threshold_largest
        Maximum crow flies prefilter distance in kilometers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays `(pop_indices, hosp_indices)` containing aligned candidate row
        pairs.
    """
    t = pc()

    pop_coords = population[['xcoord', 'ycoord']].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    hosp_coords = all_hospitals[['Longitude', 'Latitude']].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    pop_coords_rad = np.radians(pop_coords)
    hosp_coords_rad = np.radians(hosp_coords)

    print(
        f'preparing {len(pop_coords_rad):,} x {len(hosp_coords_rad):,} '
        f'for spatial nearest neighbors bounded by {distance_threshold_largest} km '
        f'in {pc() - t:.2f} seconds'
    )

    t = pc()

    tree = cKDTree(hosp_coords_rad)
    radius = np.radians(distance_threshold_largest / EARTH_RADIUS_KM)
    indices = tree.query_ball_point(pop_coords_rad, r=radius)

    lengths = np.fromiter((len(row) for row in indices), dtype=np.int64)
    n_candidate_pairs = int(lengths.sum())

    pop_indices = np.repeat(np.arange(len(indices), dtype=np.int64), lengths)
    hosp_indices = np.fromiter(
        (h for row in indices for h in row),
        dtype=np.int64,
        count=n_candidate_pairs,
    )

    print(
        f'finding {n_candidate_pairs:,} pairs of spatial nearest neighbors '
        f'in {pc() - t:.2f} seconds'
    )

    return pop_indices, hosp_indices


def _compute_unique_node_pair_distances(
    population: pd.DataFrame,
    all_hospitals: pd.DataFrame,
    pop_indices: np.ndarray,
    hosp_indices: np.ndarray,
    network: object,
) -> pl.DataFrame:
    """
    Compute shortest path distances for unique nearest node pairs.

    Parameters
    ----------
    population
        Population table.
    all_hospitals
        Hospital table.
    pop_indices
        Candidate population row indices.
    hosp_indices
        Candidate hospital row indices.
    network
        Pandana like network object exposing `shortest_path_lengths`.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        'pop_nearest_node', 'hosp_nearest_node', 'road_distance'.
    """
    t = pc()

    pop_nodes = population['nearest_node'].to_numpy(copy=False)[pop_indices]
    hosp_nodes = all_hospitals['nearest_node'].to_numpy(copy=False)[hosp_indices]

    unique_pop_nodes, unique_hosp_nodes = _unique_node_pairs(pop_nodes, hosp_nodes)

    print(
        f'creating {len(unique_pop_nodes):,} unique origin destination node pairs '
        f'in {pc() - t:.2f} seconds'
    )

    t = pc()

    road_distance = np.asarray(
        network.shortest_path_lengths(unique_pop_nodes, unique_hosp_nodes),
        dtype=np.float64,
    )
    elapsed = pc() - t

    valid = road_distance < NO_PATH_SENTINEL

    print(
        f'{len(road_distance):,} shortest paths of which {valid.sum():,} exist '
        f'found in {elapsed:.2f} seconds'
    )

    return pl.DataFrame(
        {
            'pop_nearest_node': unique_pop_nodes[valid],
            'hosp_nearest_node': unique_hosp_nodes[valid],
            'road_distance': road_distance[valid],
        }
    )


def _population_polars(population: pd.DataFrame) -> pl.DataFrame:
    """
    Build the population side join table as a Polars DataFrame.

    Parameters
    ----------
    population
        Population table.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        'pop_id', 'pop_nearest_node', 'pop_to_road_dist'.
    """
    return pl.DataFrame(
        {
            'pop_id': population['ID'].to_numpy(copy=False),
            'pop_nearest_node': population['nearest_node'].to_numpy(copy=False),
            'pop_to_road_dist': population['pop_dist_road_estrada'].to_numpy(
                dtype=np.float64,
                copy=False,
            ),
        }
    )


def _hospital_polars(all_hospitals: pd.DataFrame) -> pl.DataFrame:
    """
    Build the hospital side join table as a Polars DataFrame.

    Parameters
    ----------
    all_hospitals
        Hospital table.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        'hosp_id', 'hosp_nearest_node', 'hosp_to_road_dist'.
    """
    return pl.DataFrame(
        {
            'hosp_id': all_hospitals['ID'].to_numpy(copy=False),
            'hosp_nearest_node': all_hospitals['nearest_node'].to_numpy(copy=False),
            'hosp_to_road_dist': all_hospitals['hosp_dist_road_estrada'].to_numpy(
                dtype=np.float64,
                copy=False,
            ),
        }
    )


def compute_distances_polars(
    population: pd.DataFrame,
    all_hospitals: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    *,
    max_total_dist: float | None = None,
) -> pl.DataFrame:
    """
    Compute sparse population to hospital distances and return a Polars DataFrame.

    Parameters
    ----------
    population
        DataFrame indexed by population ID, with columns:
        'ID', 'xcoord', 'ycoord', 'nearest_node', 'pop_dist_road_estrada'.
    all_hospitals
        DataFrame indexed by hospital ID, with columns:
        'ID', 'Longitude', 'Latitude', 'nearest_node', 'hosp_dist_road_estrada'.
    distance_threshold_largest
        Maximum crow flies distance in kilometers used to prefilter candidate
        population hospital pairs.
    network
        Pandana like network object exposing `shortest_path_lengths`.
    max_total_dist
        Optional upper bound on total distance. If provided, only triplets with
        `total_dist <= max_total_dist` are kept.

    Returns
    -------
    pl.DataFrame
        Polars table with columns:
        'pop_id', 'hosp_id', 'hosp_nearest_node', 'pop_nearest_node',
        'pop_to_road_dist', 'road_distance', 'hosp_to_road_dist', 'total_dist'.
    """
    _validate_inputs(population, all_hospitals)

    pop_indices, hosp_indices = _candidate_row_pairs(
        population=population,
        all_hospitals=all_hospitals,
        distance_threshold_largest=distance_threshold_largest,
    )

    dists_pl = _compute_unique_node_pair_distances(
        population=population,
        all_hospitals=all_hospitals,
        pop_indices=pop_indices,
        hosp_indices=hosp_indices,
        network=network,
    )

    pop_pl = _population_polars(population)
    hosp_pl = _hospital_polars(all_hospitals)

    t = pc()

    result = (
        dists_pl.lazy()
        .join(pop_pl.lazy(), on='pop_nearest_node', how='inner')
        .join(hosp_pl.lazy(), on='hosp_nearest_node', how='inner')
        .with_columns(
            (
                pl.col('pop_to_road_dist')
                + pl.col('road_distance')
                + pl.col('hosp_to_road_dist')
            ).alias('total_dist')
        )
    )

    if max_total_dist is not None:
        result = result.filter(pl.col('total_dist') <= max_total_dist)

    result = result.select(
        [
            'pop_id',
            'hosp_id',
            'hosp_nearest_node',
            'pop_nearest_node',
            'pop_to_road_dist',
            'road_distance',
            'hosp_to_road_dist',
            'total_dist',
        ]
    ).collect()

    print(
        f'assembling {result.height:,} distances of interest '
        f'in {pc() - t:.2f} seconds'
    )

    return result


def compute_distances(
    population: pd.DataFrame,
    all_hospitals: pd.DataFrame,
    distance_threshold_largest: float,
    network: object,
    *,
    max_total_dist: float | None = None,
) -> pd.DataFrame:
    """
    Compute sparse population to hospital distances and return a pandas DataFrame.

    This is a thin wrapper around `compute_distances_polars`.

    Parameters
    ----------
    population
        DataFrame indexed by population ID, with columns:
        'ID', 'xcoord', 'ycoord', 'nearest_node', 'pop_dist_road_estrada'.
    all_hospitals
        DataFrame indexed by hospital ID, with columns:
        'ID', 'Longitude', 'Latitude', 'nearest_node', 'hosp_dist_road_estrada'.
    distance_threshold_largest
        Maximum crow flies distance in kilometers used to prefilter candidate
        population hospital pairs.
    network
        Pandana like network object exposing `shortest_path_lengths`.
    max_total_dist
        Optional upper bound on total distance. If provided, only triplets with
        `total_dist <= max_total_dist` are kept.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns:
        'pop_id', 'hosp_id', 'hosp_nearest_node', 'pop_nearest_node',
        'pop_to_road_dist', 'road_distance', 'hosp_to_road_dist', 'total_dist'.
    """
    return compute_distances_polars(
        population=population,
        all_hospitals=all_hospitals,
        distance_threshold_largest=distance_threshold_largest,
        network=network,
        max_total_dist=max_total_dist,
    ).to_pandas()