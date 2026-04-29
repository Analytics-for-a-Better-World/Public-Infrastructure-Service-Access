import pandas as pd
import geopandas as gpd

# access_matrix.py

def compute_access_matrix(
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
    log('Start compute_access_matrix')

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

    log('Finished compute_access_matrix')

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

    def to_dense(self) -> np.ndarray:
        """Returns dense matrix with np.inf for missing entries."""
        dense = np.full(self.shape, np.inf)
        dense[self.rows, self.cols] = self.data
        return dense

    def nearest_p_o_is(self, pop_idx: int, k: int = 5) -> pd.DataFrame:
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

    def pop_to_p_o_is_within(self, threshold: float) -> dict[int, list[int]]:
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

    def poi_to_pops_within(self, threshold: float) -> dict[int, list[int]]:
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


import pandas as pd
import geopandas as gpd

# access_matrix.py

def compute_access_matrix(
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
    log('Start compute_access_matrix')

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

    log('Finished compute_access_matrix')

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

    def to_dense(self) -> np.ndarray:
        """Returns dense matrix with np.inf for missing entries."""
        dense = np.full(self.shape, np.inf)
        dense[self.rows, self.cols] = self.data
        return dense

    def nearest_p_o_is(self, pop_idx: int, k: int = 5) -> pd.DataFrame:
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

    def pop_to_p_o_is_within(self, threshold: float) -> dict[int, list[int]]:
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

    def poi_to_pops_within(self, threshold: float) -> dict[int, list[int]]:
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
