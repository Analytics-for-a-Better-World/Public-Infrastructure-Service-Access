'''
Snapping utilities.
'''

from time import perf_counter as pc

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def snap_points_to_nodes(
    points: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    id_col: str = 'ID',
    distance_col: str = 'dist_to_node',
    projected_epsg: int = 32751,
    keep_geometry: bool = False,
    verbose: bool = True,
) -> pd.DataFrame | gpd.GeoDataFrame:
    '''
    Snap points to nearest network nodes.

    Parameters
    ----------
    points
        Input point GeoDataFrame.
    nodes
        Network node GeoDataFrame.
    id_col
        Identifier column in points.
    distance_col
        Name of output snapping distance column.
    projected_epsg
        EPSG used for metric snapping distances.
    keep_geometry
        Whether to return a GeoDataFrame preserving geometry.
    verbose
        Whether to print timing information.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        Table with original attributes plus Longitude, Latitude,
        nearest_node, and the snapping distance column.
    '''
    t0 = pc()

    if points.crs is None:
        raise ValueError('points has no CRS')
    if nodes.crs is None:
        raise ValueError('nodes has no CRS')
    if id_col not in points.columns:
        raise ValueError(f'Missing column: {id_col}')

    result = points.copy()

    if 'Longitude' not in result.columns:
        result['Longitude'] = result.geometry.x.astype('float64')
    else:
        result['Longitude'] = pd.to_numeric(result['Longitude'], errors='raise').astype('float64')

    if 'Latitude' not in result.columns:
        result['Latitude'] = result.geometry.y.astype('float64')
    else:
        result['Latitude'] = pd.to_numeric(result['Latitude'], errors='raise').astype('float64')

    result[id_col] = pd.to_numeric(result[id_col], errors='raise').astype('int64')

    points_proj = result.to_crs(epsg=projected_epsg)
    nodes_proj = nodes.to_crs(epsg=projected_epsg)

    point_xy = np.column_stack(
        [points_proj.geometry.x.to_numpy(), points_proj.geometry.y.to_numpy()]
    )
    node_xy = np.column_stack(
        [nodes_proj.geometry.x.to_numpy(), nodes_proj.geometry.y.to_numpy()]
    )

    tree = cKDTree(node_xy)
    distances, idx = tree.query(point_xy, k=1)

    result['nearest_node'] = nodes.iloc[idx]['id'].to_numpy(dtype='int64')
    result[distance_col] = distances.astype('float64')

    result = result.drop_duplicates(subset=id_col).set_index(id_col, drop=False)
    result.index = result.index.astype('int64')

    if not keep_geometry:
        result = pd.DataFrame(result.drop(columns='geometry')).copy()

    if verbose:
        print(
            f'Snapped {len(result):,} points in {pc() - t0:.2f} seconds, '
            f'mean {distance_col} {np.mean(distances):.2f} m'
        )

    return result