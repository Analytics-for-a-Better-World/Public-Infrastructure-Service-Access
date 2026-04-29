import pandas as pd
import geopandas as gpd

# pandana_utils.py

def assign_nearest_pandana_nodes_with_geometry(
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


def build_road_k_d_tree(roadNodes: gpd.GeoDataFrame) -> tuple[cKDTree, np.ndarray, np.ndarray]:
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

    print(f'🏁 Total time for build_road_k_d_tree: {t2 - t0:.2f}s.\n')

    return tree, coords, nodeIds

def generate_zero_distance_edges(df: pd.DataFrame) -> pd.DataFrame:
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

import pandas as pd
import geopandas as gpd

# pandana_utils.py

def assign_nearest_pandana_nodes_with_geometry(
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


def build_road_k_d_tree(roadNodes: gpd.GeoDataFrame) -> tuple[cKDTree, np.ndarray, np.ndarray]:
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

    print(f'🏁 Total time for build_road_k_d_tree: {t2 - t0:.2f}s.\n')

    return tree, coords, nodeIds

def generate_zero_distance_edges(df: pd.DataFrame) -> pd.DataFrame:
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