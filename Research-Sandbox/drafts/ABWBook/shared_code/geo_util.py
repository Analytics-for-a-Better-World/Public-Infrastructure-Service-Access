# -*- coding: utf-8 -*-

# ─── Standard Library ───────────────────────────────────────────────────────────
import pickle
import itertools
import logging
import requests
import tracemalloc
import warnings
from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import perf_counter as pc

# ─── Scientific Computing ───────────────────────────────────────────────────────
import fast_histogram
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

# ─── Geospatial Processing ──────────────────────────────────────────────────────
import geopandas as gpd
import pyproj
from pyproj import Transformer

# ─── Geometry and Spatial Operations (Shapely) ──────────────────────────────────
from shapely import geometry as shpg
from shapely import points  # vectorized operations in Shapely 2.x
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import transform

# ─── Network Analysis ───────────────────────────────────────────────────────────
import networkx as nx
import pandana

# ─── Geocoding (Geopy) ──────────────────────────────────────────────────────────
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

# ─── Mapping and Visualization ──────────────────────────────────────────────────
import colorcet as cc
import contextily as ctx
import folium
import matplotlib.pyplot as plt
from colorcet import rgb_to_hex

# ─── Progress Monitoring ────────────────────────────────────────────────────────
from tqdm.notebook import tqdm

# ─── Global Definitions ─────────────────────────────────────────────────────────
projectedCrsNL = 'EPSG:28992'  # Amersfoort / RD New


import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


def optimize_dataframe(
    df: pd.DataFrame,
    category_threshold: int = 50,
    category_density: float = 0.1
) -> pd.DataFrame:
    """
    Optimize a DataFrame by:
    - Converting object columns to categorical if low cardinality or high repetition
    - Downcasting float and int columns to smaller dtypes
    - Replacing monotonic integer columns with RangeIndex if applicable

    Args:
        df: Input DataFrame.
        category_threshold: Max unique values to convert to category directly.
        category_density: Max (unique / total) ratio to allow category conversion.

    Returns:
        Optimized copy of the DataFrame.
    """
    df = df.copy()

    for col in df.columns:
        col_data = df[col]

        # ── 1. Convert object columns to category ──────────────────────────────
        if col_data.dtype == 'object':
            num_unique = col_data.nunique(dropna=False)
            total = len(col_data)
            repetition_ratio = num_unique / total if total > 0 else 1.0

            if (num_unique <= category_threshold) or (
                repetition_ratio <= category_density
            ):
                df[col] = col_data.astype('category')

        # ── 2. Downcast numerics ───────────────────────────────────────────────
        elif pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast='float')

        elif pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast='integer')

    # ── 3. Convert monotonic int columns to RangeIndex ─────────────────────────
    for col in df.select_dtypes(include='int').columns:
        values = df[col]
        if (
            values.is_monotonic_increasing
            and values.diff().dropna().eq(1).all()
        ):
            df = df.set_index(col, drop=True)
            df.index = pd.RangeIndex(
                start=values.iloc[0], stop=values.iloc[-1] + 1, step=1
            )
            break  # only one index column allowed

    return df


def seconds2clock(nofSeconds: float) -> str:
    """Convert number of seconds (possibly float) to HH:MM:SS.sss clock format."""
    hours, remainder = divmod(nofSeconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02}:{int(minutes):02}:{seconds:06.3f}'


def get_logger(name: str = 'geo_util', log_dir: Path = None) -> logging.Logger:
    '''
    Create or retrieve a named logger that writes to both the console and
    optionally a log file (if `log_dir` is provided).

    Parameters:
    - name: The name of the logger (used for filtering/config).
    - log_dir: Optional Path to a directory where logs will be stored.

    Returns:
    - A configured `logging.Logger` instance.
    '''
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # Optional file handler
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f'{name}.log'
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

    return logger


def get_remote_osm_pbf_timestamp(region: str) -> datetime | None:
    '''
    Fetch the Last-Modified timestamp of the remote .pbf file for the given region,
    typically used with Pyrosm or Geofabrik downloads.

    Parameters:
    - region: Name of the region (e.g., "netherlands")

    Returns:
    - A timezone-aware datetime object representing the remote file's last modified time,
      or None if it cannot be retrieved.
    '''
    base_url = 'https://download.geofabrik.de/europe'
    file_url = f'{base_url}/{region.lower()}-latest.osm.pbf'

    try:
        response = requests.head(file_url, allow_redirects=True, timeout=10)
        if 'Last-Modified' in response.headers:
            return parsedate_to_datetime(response.headers['Last-Modified'])
        else:
            print(f'Warning: No Last-Modified header found at {file_url}')
    except Exception as e:
        print(f'Warning: Failed to fetch remote metadata from {file_url}: {e}')

    return None


def load_or_acquire(
    data_path: Path,
    file_name: str,
    acquire_func: callable,
    *args,
    force_refresh: bool = False,
    logger=None,
    verbose_args: bool = False,
    **kwargs
) -> object:
    '''
    Load data from a pickle file located in `data_path` if it exists
    (unless `force_refresh` is True). Otherwise, acquire the data using
    `acquire_func`, save it as a pickle, and return it.

    Parameters:
    - data_path: Path where the cache file is located or saved.
    - file_name: Base name of the file to store/load (without extension).
    - acquire_func: Callable that returns the data when invoked.
    - *args, **kwargs: Arguments passed to `acquire_func`.
    - force_refresh: If True, bypass cache and re-acquire.
    - logger: Optional logger with `.info()` method.
    - verbose_args: If True, include full function arguments in acquisition log.

    Returns:
    - The data object, either loaded from disk or freshly acquired.
    '''
    path = data_path.joinpath(file_name).with_suffix('.pkl')

    if path.exists() and not force_refresh:
        if logger:
            logger.info(f'[{file_name}] Loading cached data from {path}')
        start = pc()
        with path.open('rb') as f:
            data = pickle.load(f)
        duration = pc() - start
        if logger:
            logger.info(f'[{file_name}] Data loaded in {seconds2clock(duration)} from {path}')
        return data

    if logger:
        reason = 'forced refresh' if force_refresh else 'cache miss'
        if verbose_args:
            arg_str = ', '.join(map(repr, args))
            kwarg_str = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
            full_args = f'{arg_str}{", " if args and kwargs else ""}{kwarg_str}'
            logger.info(f'[{file_name}] Acquiring new data due to {reason}, using {acquire_func.__name__}({full_args})')
        else:
            logger.info(f'[{file_name}] Acquiring new data due to {reason}, using {acquire_func.__name__}()')

    path.parent.mkdir(parents=True, exist_ok=True)

    start = pc()
    data = acquire_func(*args, **kwargs)
    duration = pc() - start

    with path.open('wb') as f:
        pickle.dump(data, f)

    if logger:
        logger.info(f'[{file_name}] Data acquired in {seconds2clock(duration)} and saved to {path}')

    return data


def plotFastHistogram(
    data: np.ndarray,
    bins: int = 20,
    title: str | None = None,
    ax=None
) -> plt.Axes:
    """
    Plots a histogram using fast_histogram. Creates a new Axes if none is provided.

    Args:
        data: 1D NumPy array of numeric values to plot.
        bins: Number of histogram bins (default: 20).
        title: Optional title for the plot.
        ax: Optional matplotlib Axes. If None, a new figure and axes are created.

    Returns:
        The matplotlib Axes object used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if data.size == 0:
        ax.set_title(f'{title or "Empty data"} (no data)')
        return ax

    vmin, vmax = data.min(), data.max()
    if vmin == vmax:
        ax.set_title(f'{title or "Constant data"} (single value)')
        ax.text(0.5, 0.5, f'{vmin:.2f}', ha='center', va='center', transform=ax.transAxes)
        return ax

    counts = fast_histogram.histogram1d(data, bins=bins, range=(vmin, vmax))
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(centers, counts, width=(vmax - vmin) / bins, edgecolor='black')
    if title:
        ax.set_title(title)

    return ax


def buildRoadKDTree(roadNodes: gpd.GeoDataFrame) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    """
    Builds a cKDTree for fast spatial queries on road node geometries.

    Args:
        roadNodes: A GeoDataFrame with Point geometries and an 'id' column.

    Returns:
        A tuple (tree, coords, nodeIds):
            - tree: cKDTree built from the coordinates of the points.
            - coords: Nx2 array of (x, y) coordinates used to build the tree.
            - nodeIds: 1D array of node IDs from the 'id' column.
    """
    if roadNodes.empty:
        raise ValueError('roadNodes GeoDataFrame is empty')

    if not all(roadNodes.geometry.geom_type == 'Point'):
        raise ValueError('All geometries in roadNodes must be Points')

    if 'id' not in roadNodes.columns:
        raise ValueError("GeoDataFrame must contain an 'id' column.")

    print('⏳ Building KDTree from road nodes...')

    t0 = pc()
    coords = np.column_stack((roadNodes.geometry.x, roadNodes.geometry.y))
    nodeIds = roadNodes['id'].to_numpy()
    t1 = pc()
    print(f'✅ Extracted {len(coords)} coordinates and node IDs in {t1 - t0:.2f}s.')

    tree = cKDTree(coords)
    t2 = pc()
    print(f'✅ Built KDTree in {t2 - t1:.2f}s.')

    print(f'🏁 Total time for buildRoadKDTree: {t2 - t0:.2f}s.\n')

    return tree, coords, nodeIds


def createComponentMapping(edges: pd.DataFrame) -> dict[int, int]:
    """
    Creates a mapping from node ID to its connected component index.

    Args:
        edges: DataFrame with 'u' and 'v' columns representing undirected edges.

    Returns:
        A dictionary mapping each node ID to a connected component index.
    """
    G = nx.from_pandas_edgelist(edges, source='u', target='v', create_using=nx.Graph)
    return {
        node: idx
        for idx, component in enumerate(nx.connected_components(G))
        for node in component
    }


def assignComponentsToNodes(nodes: gpd.GeoDataFrame, edges: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Assigns a connected component label to each node based on the provided edge list.

    Args:
        nodes: GeoDataFrame with an 'id' column containing node IDs.
        edges: DataFrame with 'u' and 'v' columns representing undirected edges.

    Returns:
        GeoDataFrame with an added 'component' column indicating component membership.
    """
    mapping = createComponentMapping(edges)
    nodes['component'] = nodes['id'].map(mapping).astype('category')
    return nodes


def showNodesColoredPerComponentWithBasemap(
    nodes: gpd.GeoDataFrame,
    width: int = 800,
    height: int = 600,
    file_name: str | None = None
) -> None:
    """
    Plots nodes colored by component on a background basemap.
    The largest component is shown in light gray; others use distinct colors.

    Args:
        nodes: GeoDataFrame with geometry and 'component' column.
        width: Width of the figure in pixels.
        height: Height of the figure in pixels.
        file_name: Optional path to save the figure.
    """
    t0 = pc()
    print('⏳ Starting component coloring with basemap...')

    # ── Reproject if necessary ─────────────────────────────────────────────
    if nodes.crs != 'EPSG:3857':
        t_reproj = pc()
        nodes = nodes.to_crs('EPSG:3857')
        print(f'📐 Reprojected to EPSG:3857 in {pc() - t_reproj:.2f}s')

    # ── Component and color assignment ─────────────────────────────────────
    t_comp = pc()
    component_sizes = nodes['component'].value_counts()
    n_components = len(component_sizes)
    largest_group = component_sizes.idxmax()
    largest_size = component_sizes.max()
    print(f'✔️ Found {n_components} components (largest at index {largest_group} with size {largest_size:,} nodes) in {pc() - t_comp:.2f}s')

    t_color = pc()
    color_key = {largest_group: '#E5E5E5'}  # light gray background
    color_idx = 0
    for group_id in component_sizes.index:
        if group_id == largest_group:
            continue
        r, g, b = cc.glasbey_hv[color_idx % len(cc.glasbey_hv)]
        color_key[group_id] = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
        color_idx += 1
    nodes['color'] = nodes['component'].map(color_key)
    print(f'🎨 Assigned colors in {pc() - t_color:.2f}s')

    # ── Scatter plot drawing (fast, single call) ─────────────────────────────
    t_plot = pc()
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Set marker size and alpha based on whether it's the largest component
    is_largest = nodes['component'] == largest_group
    marker_size = is_largest.map({True: 0.1, False: 1.0})
    alpha = is_largest.map({True: 0.3, False: 1.0})

    ax.scatter(
        nodes.geometry.x,
        nodes.geometry.y,
        c=nodes['color'],
        s=marker_size,
        alpha=alpha,
        marker='.',
        linewidths=0
    )
    print(f'🖌️ Plotted {len(nodes):,} nodes in {pc() - t_plot:.2f}s')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    # ── Add basemap ────────────────────────────────────────────────────────
    t_base = pc()
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, attribution_size=6)
    print(f'🗺️ Added basemap in {pc() - t_base:.2f}s')

    # ── Save to file if requested ──────────────────────────────────────────
    if file_name:
        t_save = pc()
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f'💾 Saved plot to {file_name} in {pc() - t_save:.2f}s')

    print(f'✅ Total visualization completed in {pc() - t0:.2f}s')
    plt.show()


def extractFerryCoordinatesAndIds(ferries: gpd.GeoDataFrame) -> tuple[list[tuple[float, float]], list[int]]:
    """
    Extracts all coordinate tuples from LineString or MultiLineString geometries in a GeoDataFrame,
    and associates each coordinate with its corresponding ferry ID.

    Args:
        ferries: A GeoDataFrame with a 'geometry' column containing LineString or MultiLineString geometries
                 and an 'id' attribute per row (as either a column or an attribute).

    Returns:
        A tuple (coordinates, ferry_ids):
            - coordinates: List of (x, y) coordinate tuples.
            - ferry_ids: List of ferry IDs corresponding to each coordinate.
    """
    coordinates_ids = [
        (coord, row.id)
        for row in ferries.itertuples()
        for geom in [row.geometry]
        for coord in (
            geom.coords if geom.geom_type == 'LineString'
            else [pt for line in geom.geoms for pt in line.coords]
        )
    ]

    if not coordinates_ids:
        return [], []

    coordinates, ferry_ids = zip(*coordinates_ids)
    return list(coordinates), list(ferry_ids)


def groupFerryIdsByCoordinate(
    ferryCoords: list[tuple[float, float]],
    ferryIds: list[int],
) -> dict[tuple[float, float], list[int]]:
    """
    Groups ferry IDs by their coordinate location.

    Args:
        ferryCoords: List of (x, y) coordinate tuples.
        ferryIds: List of ferry IDs corresponding to each coordinate.

    Returns:
        A dictionary mapping each coordinate to a list of ferry IDs that pass through it.
    """
    coordToIds = defaultdict(list)
    for coord, fid in zip(ferryCoords, ferryIds):
        coordToIds[coord].append(fid)
    return dict(coordToIds)


def generateZeroDistanceEdges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates all possible edges between roadNodeIds with distance == 0, grouped by ferryId.

    Args:
        df: Input DataFrame with at least ['ferryId', 'roadNodeId', 'distance'] columns.

    Returns:
        DataFrame with columns ['ferryId', 'fromNodeId', 'toNodeId'] representing edges.
    """
    groups = df[df['distance'] == 0.0].groupby('ferryId')

    edge_rows = []

    for ferryId, group in groups:
        node_ids = group['roadNodeId'].unique()
        if len(node_ids) >= 2:
            for u, v in itertools.combinations(sorted(node_ids), 2):
                edge_rows.append({'ferryId': ferryId, 'fromNodeId': u, 'toNodeId': v})

    return pd.DataFrame(edge_rows)


def connectFerryToRoad(
    ferryCoords: list[tuple[float, float]],
    ferryIds: list[int],
    roadTree: cKDTree,
    roadCoords: np.ndarray,
    roadNodeIds: np.ndarray,
    maxDistance: float,
    targetCRS: str,
    sourceCRS: str = projectedCrsNL
) -> gpd.GeoDataFrame:
    """
    Connects ferry coordinates to their nearest road nodes using a precomputed cKDTree, 
    transforming all geometries to the target CRS.

    Args:
        ferryCoords: List of (x, y) coordinate tuples in source CRS.
        ferryIds: List of ferry IDs corresponding to each coordinate.
        roadTree: cKDTree built from road node coordinates (in source CRS).
        roadCoords: Nx2 array of (x, y) coordinates corresponding to roadTree (source CRS).
        roadNodeIds: 1D array of node IDs corresponding to roadCoords.
        maxDistance: Maximum snapping distance (in source CRS units).
        targetCRS: CRS to use for the output GeoDataFrame (e.g., 'EPSG:4326').
        sourceCRS: CRS of input coordinates (default: 'EPSG:28992').

    Returns:
        GeoDataFrame with columns:
            - 'ferryId'
            - 'ferryPoint' (in target CRS)
            - 'roadPoint' (in target CRS)
            - 'roadNodeId'
            - 'distance' (in source CRS units)
            - 'geometry' (LineString in target CRS)
    """
    start_total = pc()
    print('⏳ Starting ferry-to-road connection...')

    # Step 1: Nearest-neighbor search (in source CRS)
    t1 = pc()
    ferryArray = np.array(ferryCoords)
    distances, indices = roadTree.query(ferryArray, k=1)
    print(f'✅ Nearest-neighbor query completed in {pc() - t1:.2f}s.')

    # Step 2: Transformer for coordinates
    to_target_crs = pyproj.Transformer.from_crs(sourceCRS, targetCRS, always_xy=True).transform

    # Step 3: Build connector features (with transformed coordinates)
    t2 = pc()
    connectors = []
    for i, dist in enumerate(distances):
        if dist <= maxDistance:
            ferryPt_raw = Point(ferryCoords[i])
            roadPt_raw = Point(roadCoords[indices[i]])

            ferryPt = transform(to_target_crs, ferryPt_raw)
            roadPt = transform(to_target_crs, roadPt_raw)

            connectors.append({
                'ferryId': ferryIds[i],
                'ferryPoint': ferryPt,
                'roadPoint': roadPt,
                'roadNodeId': roadNodeIds[indices[i]],
                'distance': dist,  # remains in source CRS units
                'geometry': LineString([ferryPt, roadPt])
            })

    print(f'✅ Constructed {len(connectors)} connectors (≤ {maxDistance} units) in {pc() - t2:.2f}s.')

    if not connectors:
        print('⚠️ No connectors found within the specified maxDistance.')

    t_total = pc()
    print(f'🏁 Total runtime: {t_total - start_total:.2f}s.\n')

    return gpd.GeoDataFrame(connectors, geometry='geometry', crs=targetCRS)


def computeAccessMatrix(
    population: gpd.GeoDataFrame,
    points_of_interest: gpd.GeoDataFrame,
    network: pandana.Network,
    max_distance_threshold: float,
    batch_size = 10_000_000,
    num_fallback_poi = 0
) -> pd.DataFrame:
    """
    Computes total access distances between population points and POIs using Pandana
    with cKDTree-based prefiltering. Logs progress with timing and memory usage.

    Args:
        population: GeoDataFrame with 'pop_idx', 'nearest_node_id', 'nearest_node_distance'.
        points_of_interest: GeoDataFrame with 'poi_idx', 'nearest_node_id', 'nearest_node_distance'.
        network: Pandana Network object.
        max_distance_threshold: Euclidean filter radius in meters.
        batch_size: maximum number of shortest paths to compute in one call
        num_fallback_poi: add OD paris from closest num_fallback_poi POI to all missing POP points
    Returns:
        DataFrame with distances between population and POIs.
    """
    # ─── Setup logging ───────────────────────────────────────────────────────
    log_filename = f'access_matrix_{int(max_distance_threshold):,}_m_{batch_size:,}.log'.replace(',', '_')

    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)

    warnings.filterwarnings('ignore', message='Unsigned integer: shortest path distance is trying to be calculated')
    t0 = pc()
    t_prev = t0
    tracemalloc.start()
    _, mem_prev = tracemalloc.get_traced_memory()

    def log(msg: str):
        nonlocal t_prev, mem_prev
        now = pc()
        current, _ = tracemalloc.get_traced_memory()
        elapsed = now - t0
        step_time = now - t_prev
        mem_now = current
        mem_delta = (mem_now - mem_prev) / 1024**2

        elapsed_str = f'{int(elapsed//3600):02}:{int((elapsed%3600)//60):02}:{int(elapsed%60):02}'
        step_str = f'{step_time:,.2f}'
        mem_str = f'{mem_now/1024**2:,.1f}'
        delta_str = f'{mem_delta:+,.1f}'
        logger.info(f"{elapsed_str:>11}  {step_str:>10}  {mem_str:>12}  {delta_str:>12}  {msg}")

        t_prev = now
        mem_prev = mem_now

    logger.info(f"{'Elapsed':>11}  {'ΔTime (s)':>10}  {'Memory (MB)':>12}  {'ΔMemory (MB)':>12}  {'Step'}")
    logger.info(f"{'-'*11}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*40}")
    log('Start computeAccessMatrix')

    # ─── Assert data is projected ────────────────────────────────────────────
    assert population.crs.is_projected
    assert population.crs == points_of_interest.crs
    log('CRS check passed')

    # ─── Check required columns present ──────────────────────────────────────
    required_cols = ['nearest_node_id', 'nearest_node_distance']
    if not all(col in population.columns for col in ['pop_idx'] + required_cols):
        raise KeyError('population missing required columns')
    if not all(col in points_of_interest.columns for col in ['poi_idx'] + required_cols):
        raise KeyError('POIs missing required columns')
    log('Required columns present')

    # ─── Construct mappings ──────────────────────────────────────────────────
    pop_node_map = population.nearest_node_id.to_dict()
    pop_dist_map = population.nearest_node_distance.to_dict()

    poi_node_map = points_of_interest.nearest_node_id.to_dict()
    poi_dist_map = points_of_interest.nearest_node_distance.to_dict()
    
    log('Constructed pop/POI mappings: nearest_node_id and nearest_node_distance')

    # ─── KDTree Spatial Filter ───────────────────────────────────────────────
    log('Building KDTree on population...')
    tree = cKDTree(np.column_stack((population.geometry.x, population.geometry.y)))
    log(f'KDTree built for {len(population):,} population points')

    log(f'Running KDTree spatial query (query_ball_point) for {len(points_of_interest):,} POIs within {max_distance_threshold:,} meters...')
    matches = tree.query_ball_point(np.column_stack((points_of_interest.geometry.x, points_of_interest.geometry.y)), r=max_distance_threshold)
    log('Spatial query completed')

    # ─── Create lists of origins and destinations ────────────────────────────
    log('Expanding OD candidate pairs...')
    lens = np.fromiter((len(m) for m in matches), dtype=int)
    df = pd.DataFrame({   
        'pop_idx': np.concatenate(matches).astype(int), 
        'poi_idx': np.repeat(points_of_interest.poi_idx.to_numpy(), lens).astype(int)
    })
    log(f'Expanded to {len(df):,} OD pairs')

    # ─── Add missing pop points ─────────────────────────────────────────────
    if num_fallback_poi > 0:
        found_pop = set(df['pop_idx'].unique())
        all_pop = set(population['pop_idx'])
        missing_pop = sorted(all_pop - found_pop)
        log(f'Found {len(missing_pop):,} population points missing after KDTree filtering')

        if missing_pop:
            log(f'Finding {num_fallback_poi} closest POIs for each missing population point...')
            missing_gdf = population.set_index('pop_idx').loc[missing_pop]
            poi_coords = np.column_stack((points_of_interest.geometry.x, points_of_interest.geometry.y))
            poi_tree = cKDTree(poi_coords)

            query_coords = np.column_stack((missing_gdf.geometry.x, missing_gdf.geometry.y))
            distances, indices = poi_tree.query(query_coords, k=min(num_fallback_poi, len(points_of_interest)))

            if len(indices.shape) == 1:
                # Only one POI case
                poi_idx_array = points_of_interest.poi_idx.to_numpy()
                fallback_df = pd.DataFrame({
                    'pop_idx': missing_gdf.index,
                    'poi_idx': poi_idx_array[indices]
                })
            else:
                poi_idx_array = points_of_interest.poi_idx.to_numpy()
                fallback_df = pd.DataFrame({
                    'pop_idx': np.repeat(missing_gdf.index.to_numpy(), indices.shape[1]),
                    'poi_idx': poi_idx_array[indices.flatten()]
                })

            df = pd.concat([df, fallback_df], ignore_index=True)
            log(f'Added {len(fallback_df):,} fallback OD pairs from each missing pop to {num_fallback_poi} closest POIs')

    # ─── Merge Nearest Node IDs ─────────────────────────────────────────────
    log('Merging nearest_node_id for pop_idx...')
    df['pop_node_id'] = df.pop_idx.map(pop_node_map)
    log('Merged pop_node_id')

    log('Merging nearest_node_id for poi_idx...')
    df['poi_node_id'] = df.poi_idx.map(poi_node_map)
    log('Merged poi_node_id')

    # ─── Shortest Path Computation in Batches ────────────────────────────────
    log(f'Running shortest path computation in batches of at most {batch_size:,} OD pairs...')
    _tsp_total = pc()

    numPairs = len(df)
    allDistances = []

    popNodes = df.pop_node_id.to_numpy()
    poiNodes = df.poi_node_id.to_numpy()

    log('Preparation for batches done.')
    
    for i in range(0, numPairs, batch_size):
        batchStart = i
        batchEnd = min(i + batch_size, numPairs)
        
        log(f'Processing batch {batchStart:,} to {batchEnd-1:,}...')
        _tsp_batch = pc()

        batchDistances = network.shortest_path_lengths(popNodes[batchStart:batchEnd], poiNodes[batchStart:batchEnd])
        allDistances.extend(batchDistances)

        _tsp_batch = pc() - _tsp_batch
        log(f'Batch done in {_tsp_batch:,.1f} seconds ({_tsp_batch / (batchEnd - batchStart):.1e} seconds per path)')

    df['road_distance'] = np.array(allDistances)

    _tsp_total = pc() - _tsp_total
    log(f'All {numPairs:,} shortest paths computed in {_tsp_total:,.1f} seconds, {_tsp_total / numPairs:.1e} seconds per path')

    # ─── Path QA and Filtering ──────────────────────────────────────────────
    finite = np.isfinite(df['road_distance'].to_numpy())
    df = df[finite & (df['road_distance'] < 4_294_967.295)]
    log(f'Filtered to {len(df):,} valid paths')

    # ─── Merge Snap Distances and Compute Total ─────────────────────────────
    log('Merging node snap distances...')
    df['pop_to_node_dist'] = df.pop_idx.map(pop_dist_map)
    df['poi_to_node_dist'] = df.poi_idx.map(poi_dist_map)
    log('Snap distances merged')

    df['total_dist'] = df['road_distance'] + df['pop_to_node_dist'] + df['poi_to_node_dist']
    log(f'Final total_dist computed for {len(df):,} OD pairs')

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"\n📈 Peak memory usage: {peak / 1024**2:,.1f} MB")

    log('Finished computeAccessMatrix')

    return df


class SparseDistanceMatrix:
    """
    Sparse (pop_idx, poi_idx) distance matrix for fast lookups and nearest-POI queries.

    Missing entries are treated as np.inf. True zeros are preserved.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pop_col: str = 'pop_idx',
        poi_col: str = 'poi_idx',
        dist_col: str = 'total_dist'
    ):
        """
        Args:
            df: DataFrame with columns for population, POI, and distance.
            pop_col: Name of the column for population identifiers.
            poi_col: Name of the column for POI identifiers.
            dist_col: Name of the column with distance values.
        """
        row = df[pop_col].values
        col = df[poi_col].values
        data = df[dist_col].values

        self.rows = row
        self.cols = col
        self.data = data
        self.shape = (row.max() + 1, col.max() + 1)

        self.pop_col = pop_col
        self.poi_col = poi_col
        self.dist_col = dist_col

        self.df = df[[pop_col, poi_col, dist_col]]
        self.matrix = coo_matrix((data, (row, col)), shape=self.shape).tocsr()

    def get(self, pop_idx: int, poi_idx: int) -> float:
        """Returns distance if exists, else np.inf."""
        row = self.matrix.getrow(pop_idx)
        try:
            pos = row.indices.tolist().index(poi_idx)
            return row.data[pos]
        except ValueError:
            return np.inf

    def toDense(self) -> np.ndarray:
        """Returns dense matrix with np.inf for missing entries."""
        dense = np.full(self.shape, np.inf)
        dense[self.rows, self.cols] = self.data
        return dense

    def nearestPOIs(self, pop_idx: int, k: int = 5) -> pd.DataFrame:
        """
        Returns top-k closest POIs for a given population point.

        Args:
            pop_idx: Index of population location.
            k: Number of nearest POIs to return.

        Returns:
            DataFrame with 'poi_idx' and 'distance'.
        """
        row = self.matrix.getrow(pop_idx)
        if row.nnz == 0:
            return pd.DataFrame(columns=['poi_idx', 'distance'])

        poi_indices = row.indices
        distances = row.data
        order = np.argsort(distances)[:k]

        return pd.DataFrame({
            'poi_idx': poi_indices[order],
            'distance': distances[order]
        })

    def popToPOIsWithin(self, threshold: float) -> dict[int, list[int]]:
        """
        Returns a dictionary mapping each population node ID to a list of POI node IDs
        whose access distance is less than or equal to the given threshold.

        This is computed by filtering the internal distance DataFrame on the specified
        distance threshold and grouping by population node ID.

        Args:
            threshold: Maximum allowed total access distance (inclusive).

        Returns:
            Dictionary of the form {pop_idx: [poi_idx1, poi_idx2, ...]}.
        """
        return (
            self.df[self.df[self.dist_col] <= threshold]
            .groupby(self.pop_col)[self.poi_col]
            .apply(list)
            .to_dict()
        )

    def poiToPopsWithin(self, threshold: float) -> dict[int, list[int]]:
        """
        Returns a dictionary mapping each POI node ID to a list of population node IDs
        whose access distance is less than or equal to the given threshold.

        This is computed by filtering the internal distance DataFrame on the specified
        distance threshold and grouping by POI node ID.

        Args:
            threshold: Maximum allowed total access distance (inclusive).

        Returns:
            Dictionary of the form {poi_idx: [pop_idx1, pop_idx2, ...]}.
        """
        return (
            self.df[self.df[self.dist_col] <= threshold]
            .groupby(self.poi_col)[self.pop_col]
            .apply(list)
            .to_dict()
        )

def assignNearestPandanaNodesWithGeometry(
    gdf: gpd.GeoDataFrame,
    network: pandana.Network,
    geometry_column: str = 'geometry',
    columnPrefix: str = 'nearest_node'
) -> gpd.GeoDataFrame:
    """
    Assigns the nearest Pandana network node to each point in a GeoDataFrame.

    Adds:
        - '{prefix}_id':        ID of the nearest node
        - '{prefix}_geom':      geometry of the nearest node (Point)
        - '{prefix}_distance':  snapping distance in meters

    Assumes all coordinates are in a projected CRS (e.g. EPSG:3857 or EPSG:28992).

    Args:
        gdf: GeoDataFrame of input points.
        network: A Pandana Network object.
        geometry_column: Name of the point geometry column.
        columnPrefix: Prefix used for new columns.

    Returns:
        A new GeoDataFrame with three additional columns.
    """
    t0 = pc()
    print('⏳ Starting nearest-node assignment...')

    # ── Extract coordinates ──────────────────────────────────────────────
    x_coords = gdf[geometry_column].x.values
    y_coords = gdf[geometry_column].y.values
    print(f'📌 Extracted {len(x_coords):,} coordinates in {pc() - t0:.2f}s')

    # ── Snap to nearest nodes ────────────────────────────────────────────
    t_snap = pc()
    nearest_node_ids = network.get_node_ids(x_coords, y_coords)
    print(f'🔗 Snapped to nodes in {pc() - t_snap:.2f}s')

    # ── Retrieve node coordinates efficiently ────────────────────────────
    t_nodes = pc()
    nodes_df = network.nodes_df
    if isinstance(nodes_df.index, pd.RangeIndex):
        x_arr = nodes_df['x'].values
        y_arr = nodes_df['y'].values
        node_coords = np.column_stack((x_arr[nearest_node_ids], y_arr[nearest_node_ids]))
    else:
        node_coords = nodes_df.loc[nearest_node_ids, ['x', 'y']].to_numpy()
    print(f'📦 Retrieved node coordinates in {pc() - t_nodes:.2f}s')

    # ── Build snapped node geometries ────────────────────────────────────
    t_geom = pc()
    snapped_points = gpd.GeoSeries(points(node_coords), crs=gdf.crs)
    print(f'🗺️ Built snapped node geometries in {pc() - t_geom:.2f}s')

    # ── Compute distances ────────────────────────────────────────────────
    t_dist = pc()
    distances = gdf[geometry_column].distance(snapped_points)
    print(f'📏 Computed distances in {pc() - t_dist:.2f}s')

    # ── Combine results ──────────────────────────────────────────────────
    result = gdf.copy()
    result[f'{columnPrefix}_id'] = nearest_node_ids
    result[f'{columnPrefix}_geom'] = snapped_points
    result[f'{columnPrefix}_distance'] = distances
    print(f'✅ Assignment completed in {pc() - t0:.2f}s total')

    return result

def visualizeOrAddPopPoiConnection(
    row_idx,
    all_distances,
    population,
    points_of_interest,
    network,
    m: folium.Map = None,
    metric_crs='EPSG:28992',
    pop_id_col=None,
    poi_id_col=None,
    color='red',
    weight=3,
    index_label=None
) -> folium.Map:
    """
    Visualizes or adds a population–POI shortest path (and snapping lines) to a folium map.

    Args:
        row_idx: Index in `all_distances`.
        all_distances: DataFrame with 'pop_node_id' and 'poi_node_id' columns.
        population: GeoDataFrame of population locations.
        points_of_interest: GeoDataFrame of POIs.
        network: Pandana network object.
        m: Existing folium.Map to update (if None, a new one is created centered on this route).
        metric_crs: Projected CRS used for distance computations.
        pop_id_col: Population ID column.
        poi_id_col: POI ID column.
        color: Line color for the shortest path.
        weight: Line weight (thickness).
        index_label: Optional label to show in tooltips.

    Returns:
        folium.Map with the route added.
    """
    row = all_distances.loc[row_idx]

    if pop_id_col is None:
        pop_id_col = next((c for c in population.columns if c in row and 'pop' in c), 'idx')
    if poi_id_col is None:
        poi_id_col = next((c for c in points_of_interest.columns if c in row and 'poi' in c), 'idx')

    pop_idx = row.get(pop_id_col)
    poi_idx = row.get(poi_id_col)
    from_node = row['pop_node_id']
    to_node = row['poi_node_id']

    path_nodes = network.shortest_path(from_node, to_node)
    if path_nodes is None or len(path_nodes) < 2:
        return m  # skip trivial path

    node_coords = network.nodes_df.loc[path_nodes, ['x', 'y']].to_numpy()
    path_line_proj = LineString(node_coords)
    path_length_m = path_line_proj.length

    transformer = Transformer.from_crs(metric_crs, 'EPSG:4326', always_xy=True)
    lon_lat = [transformer.transform(x, y) for x, y in node_coords]
    path_line = LineString(lon_lat)

    pop_row = population.loc[population[pop_id_col] == pop_idx].iloc[0]
    poi_row = points_of_interest.loc[points_of_interest[poi_id_col] == poi_idx].iloc[0]
    pop_geom = pop_row.geometry
    poi_geom = poi_row.geometry
    pop_lonlat = transformer.transform(pop_geom.x, pop_geom.y)
    poi_lonlat = transformer.transform(poi_geom.x, poi_geom.y)

    snap_pop_xy = network.nodes_df.loc[from_node, ['x', 'y']]
    snap_poi_xy = network.nodes_df.loc[to_node, ['x', 'y']]
    snap_pop_lonlat = transformer.transform(snap_pop_xy['x'], snap_pop_xy['y'])
    snap_poi_lonlat = transformer.transform(snap_poi_xy['x'], snap_poi_xy['y'])

    snap_line_pop = LineString([pop_geom.coords[0], (snap_pop_xy['x'], snap_pop_xy['y'])])
    snap_line_poi = LineString([poi_geom.coords[0], (snap_poi_xy['x'], snap_poi_xy['y'])])

    # Create new map if needed
    if m is None:
        m = folium.Map(location=[(pop_lonlat[1] + poi_lonlat[1]) / 2, (pop_lonlat[0] + poi_lonlat[0]) / 2], zoom_start=13)

    # Add markers
    folium.Marker(
        location=(pop_lonlat[1], pop_lonlat[0]),
        tooltip=f"Population: {pop_row.get('Population', 'N/A')}",
        icon=folium.Icon(color='blue', icon='user')
    ).add_to(m)

    folium.Marker(
        location=(poi_lonlat[1], poi_lonlat[0]),
        tooltip=f"POI: {poi_row.get('FullAddress', 'N/A')}",
        icon=folium.Icon(color='green', icon='flag')
    ).add_to(m)

    folium.CircleMarker(
        location=(snap_pop_lonlat[1], snap_pop_lonlat[0]),
        radius=4,
        color='blue',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

    folium.CircleMarker(
        location=(snap_poi_lonlat[1], snap_poi_lonlat[0]),
        radius=4,
        color='green',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

    folium.PolyLine(
        locations=[(pop_lonlat[1], pop_lonlat[0]), (snap_pop_lonlat[1], snap_pop_lonlat[0])],
        color='black',
        dash_array='3',
        tooltip=f'Snap line (pop): {snap_line_pop.length:.1f} m'
    ).add_to(m)

    folium.PolyLine(
        locations=[(poi_lonlat[1], poi_lonlat[0]), (snap_poi_lonlat[1], snap_poi_lonlat[0])],
        color='black',
        dash_array='3',
        tooltip=f'Snap line (POI): {snap_line_poi.length:.1f} m'
    ).add_to(m)

    # Add path line
    tooltip = f"Shortest path: {path_length_m:.1f} m"
    if index_label is not None:
        tooltip = f"#{index_label}: {path_length_m:.1f} m"

    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in lon_lat],
        color=color,
        weight=weight,
        tooltip=tooltip
    ).add_to(m)

    return m


def compareGeocodedAndNetworkLocations(
    banks: gpd.GeoDataFrame,
    furthest_index: int,
    geocoded_xy: tuple[float, float],  # (lon, lat)
    crs_latlon: str = 'EPSG:4326',
    crs_projected: str = 'EPSG:28992'
) -> tuple[folium.Map, float]:
    """
    Compares a bank's original (network-snapped) location to a geocoded one,
    computes distance, and displays a Folium map and comparison table.

    Args:
        banks: GeoDataFrame with bank locations (must include 'Latitude' and 'Longitude').
        furthest_index: Index of the record to compare.
        geocoded_xy: Tuple (lon, lat) from external geocoding service.
        crs_latlon: CRS for latitude/longitude (default WGS84).
        crs_projected: Projected CRS for distance computation in meters.

    Returns:
        A tuple of:
        - folium.Map showing both points and distance line
        - distance in meters
    """
    # 1. Build comparison table
    top_record = banks.loc[[furthest_index], ['Latitude', 'Longitude']]
    comparison = pd.concat([
        top_record.rename(index={furthest_index: 'furthest'}),
        pd.DataFrame([{'Latitude': geocoded_xy[1], 'Longitude': geocoded_xy[0]}], index=['geocoded'])
    ])
    display(comparison)

    # 2. Create GeoDataFrame for both points
    gdf_points = gpd.GeoDataFrame({
        'name': ['furthest', 'geocoded'],
        'geometry': [
            Point(top_record['Longitude'].values[0], top_record['Latitude'].values[0]),
            Point(*geocoded_xy)
        ]
    }, crs=crs_latlon).to_crs(crs_projected)

    # 3. Compute distance in meters
    distance_m = gdf_points.distance(gdf_points.iloc[0].geometry)[1]
    print(f"📏 Distance from network-snapped to geocoded location: {distance_m:,.1f} meters")

    # 4. Create Folium map
    m = banks.loc[[furthest_index]].explore(
    tooltip=['Bank', 'FullAddress', 'nearest_node_distance'],
        style_kwds={'color': 'red', 'weight': 8}
    )

    red_latlon = top_record.iloc[0]['Latitude'], top_record.iloc[0]['Longitude']
    blue_latlon = geocoded_xy[1], geocoded_xy[0]

    # Add geocoded point (blue)
    folium.CircleMarker(
        location=blue_latlon,
        popup='Geocoded location',
        radius=8,
        color='blue',
        fill_color='blue',
        fill_opacity=1.0,
    ).add_to(m)

    # Add line connecting both
    folium.PolyLine(
        locations=[red_latlon, blue_latlon],
        color='black',
        weight=2,
        dash_array='5,5',
        tooltip=f'{distance_m:.1f} meters'
    ).add_to(m)

    # Adjust bounds
    bounds = [
        [min(red_latlon[0], blue_latlon[0]), min(red_latlon[1], blue_latlon[1])],
        [max(red_latlon[0], blue_latlon[0]), max(red_latlon[1], blue_latlon[1])]
    ]
    m.fit_bounds(bounds)

    return m, distance_m


def geocodeTopNIntoGeometryColumn(
    gdf: gpd.GeoDataFrame,
    address_column: str = 'FullAddress',
    sort_column: str = 'nearest_node_distance',
    new_geometry_column: str = 'geocoded_geometry',
    top_n: int = 10
) -> gpd.GeoDataFrame:
    """
    Adds a new geometry column to a GeoDataFrame with geocoded values for the top N records
    (based on sort_column). Other rows keep the original geometry.

    Args:
        gdf: GeoDataFrame with address and geometry info.
        address_column: Column name containing full address strings.
        sort_column: Column to sort by for selecting top N rows to geocode.
        new_geometry_column: Name of the new column to create.
        top_n: Number of top rows to geocode.

    Returns:
        Updated GeoDataFrame with an added Point column.
    """
    geolocator = Nominatim(user_agent='Course notes on Analytics for a Better World (j.a.s.gromicho@uva.nl)')
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=8,
        error_wait_seconds=8
    )

    gdf = gdf.copy()
    gdf[new_geometry_column] = gdf.geometry

    # Clean non-breaking spaces
    cleaned_addresses = gdf[address_column].str.replace('\xa0', ' ', regex=False)

    top_idx = gdf.nlargest(top_n, sort_column).index
    new_points = {}

    for idx in top_idx:
        address = cleaned_addresses.loc[idx]
        if pd.notna(address) and address.strip():
            location = geocode(address)
            if location:
                new_points[idx] = Point(location.longitude, location.latitude)

    geocoded_series = gpd.GeoSeries(new_points, crs='EPSG:4326')
    if gdf.crs is not None:
        geocoded_series = geocoded_series.to_crs(gdf.crs)

    for idx, geom in geocoded_series.items():
        gdf.at[idx, new_geometry_column] = geom

    return gdf


def visualizePopPoiConnection(
    row_idx,
    all_distances,
    population,
    points_of_interest,
    network,
    metric_crs='EPSG:28992',
    pop_id_col=None,
    poi_id_col=None,
):
    """
    Visualizes the connection between a population location and a POI via shortest path,
    including snapping lines and detailed tooltips.

    Args:
        row_idx: Index of the row in `all_distances` to visualize.
        all_distances: DataFrame with columns like 'pop_node_id', 'poi_node_id', etc.
        population: GeoDataFrame with at least geometry and ID column.
        points_of_interest: GeoDataFrame with at least geometry and ID column.
        network: Pandana network object.
        metric_crs: Projected CRS of the network (default EPSG:28992).
        pop_id_col: Name of population ID column (optional; inferred if None).
        poi_id_col: Name of POI ID column (optional; inferred if None).

    Returns:
        folium.Map showing the route between the two points.
    """

    row = all_distances.loc[row_idx]

    if pop_id_col is None:
        pop_id_col = next((c for c in population.columns if c in row and 'pop' in c), 'idx')
    if poi_id_col is None:
        poi_id_col = next((c for c in points_of_interest.columns if c in row and 'poi' in c), 'idx')

    pop_idx = row.get(pop_id_col)
    poi_idx = row.get(poi_id_col)
    from_node = row['pop_node_id']
    to_node = row['poi_node_id']

    path_nodes = network.shortest_path(from_node, to_node)
    if path_nodes is None or len(path_nodes) < 2:
        raise ValueError('Shortest path is empty or trivial.')

    if not {'x', 'y'}.issubset(network.nodes_df.columns):
        raise ValueError('network nodes must contain "x" and "y" columns with projected coordinates.')

    node_coords = network.nodes_df.loc[path_nodes, ['x', 'y']].to_numpy()
    path_line_proj = LineString(node_coords)
    path_length_m = path_line_proj.length

    transformer = Transformer.from_crs(metric_crs, 'EPSG:4326', always_xy=True)
    lon_lat = [transformer.transform(x, y) for x, y in node_coords]
    path_line = LineString(lon_lat)

    pop_row = population.loc[population[pop_id_col] == pop_idx].iloc[0]
    poi_row = points_of_interest.loc[points_of_interest[poi_id_col] == poi_idx].iloc[0]
    pop_geom = pop_row.geometry
    poi_geom = poi_row.geometry
    pop_lonlat = transformer.transform(pop_geom.x, pop_geom.y)
    poi_lonlat = transformer.transform(poi_geom.x, poi_geom.y)

    snap_pop_xy = network.nodes_df.loc[from_node, ['x', 'y']]
    snap_poi_xy = network.nodes_df.loc[to_node, ['x', 'y']]
    snap_pop_lonlat = transformer.transform(snap_pop_xy['x'], snap_pop_xy['y'])
    snap_poi_lonlat = transformer.transform(snap_poi_xy['x'], snap_poi_xy['y'])

    snap_line_pop = LineString([pop_geom.coords[0], (snap_pop_xy['x'], snap_pop_xy['y'])])
    snap_line_poi = LineString([poi_geom.coords[0], (snap_poi_xy['x'], snap_poi_xy['y'])])

    m = folium.Map(location=[(pop_lonlat[1] + poi_lonlat[1]) / 2, (pop_lonlat[0] + poi_lonlat[0]) / 2], zoom_start=13)

    folium.Marker(
        location=(pop_lonlat[1], pop_lonlat[0]),
        tooltip=f"Population: {pop_row.get('Population', 'N/A')}",
        icon=folium.Icon(color='blue', icon='user')
    ).add_to(m)

    folium.Marker(
        location=(poi_lonlat[1], poi_lonlat[0]),
        tooltip=f"POI: {poi_row.get('FullAddress', 'N/A')}",
        icon=folium.Icon(color='green', icon='flag')
    ).add_to(m)

    folium.CircleMarker(
        location=(snap_pop_lonlat[1], snap_pop_lonlat[0]),
        radius=4,
        color='blue',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

    folium.CircleMarker(
        location=(snap_poi_lonlat[1], snap_poi_lonlat[0]),
        radius=4,
        color='green',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

    folium.PolyLine(
        locations=[(pop_lonlat[1], pop_lonlat[0]), (snap_pop_lonlat[1], snap_pop_lonlat[0])],
        color='black',
        dash_array='3',
        tooltip=f'Snap line (pop): {snap_line_pop.length:.1f} m'
    ).add_to(m)

    folium.PolyLine(
        locations=[(poi_lonlat[1], poi_lonlat[0]), (snap_poi_lonlat[1], snap_poi_lonlat[0])],
        color='black',
        dash_array='3',
        tooltip=f'Snap line (POI): {snap_line_poi.length:.1f} m'
    ).add_to(m)

    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in lon_lat],
        color='red',
        weight=3,
        tooltip=f'Shortest path: {path_length_m:.1f} m'
    ).add_to(m)

    return m