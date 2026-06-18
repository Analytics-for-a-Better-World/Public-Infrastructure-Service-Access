'''
Snapping utilities.
'''

from time import perf_counter as pc

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


def _crs_epsg(gdf: gpd.GeoDataFrame) -> int | None:
    """Return an EPSG code for a GeoDataFrame CRS when available."""
    if gdf.crs is None:
        return None
    return gdf.crs.to_epsg()


def _project_lon_lat(
    lon: pd.Series,
    lat: pd.Series,
    projected_epsg: int,
) -> np.ndarray:
    """Project EPSG:4326 lon/lat columns into a metric coordinate array."""
    transformer = Transformer.from_crs(4326, projected_epsg, always_xy=True)
    x, y = transformer.transform(
        lon.to_numpy(dtype='float64', copy=False),
        lat.to_numpy(dtype='float64', copy=False),
    )
    return np.column_stack(
        [
            np.asarray(x, dtype='float64'),
            np.asarray(y, dtype='float64'),
        ]
    )


def _point_xy(
    points: gpd.GeoDataFrame,
    result: gpd.GeoDataFrame,
    projected_epsg: int,
) -> np.ndarray:
    """Return projected point coordinates, preferring numeric lon/lat columns."""
    if _crs_epsg(points) == 4326:
        return _project_lon_lat(
            pd.to_numeric(result['Longitude'], errors='raise'),
            pd.to_numeric(result['Latitude'], errors='raise'),
            projected_epsg,
        )

    points_proj = result.to_crs(epsg=projected_epsg)
    return np.column_stack(
        [points_proj.geometry.x.to_numpy(), points_proj.geometry.y.to_numpy()]
    )


def _node_xy(nodes: gpd.GeoDataFrame, projected_epsg: int) -> np.ndarray:
    """Return projected node coordinates without reprojecting Shapely geometries."""
    if {'lon', 'lat'}.issubset(nodes.columns):
        return _project_lon_lat(
            pd.to_numeric(nodes['lon'], errors='raise'),
            pd.to_numeric(nodes['lat'], errors='raise'),
            projected_epsg,
        )

    nodes_proj = nodes.to_crs(epsg=projected_epsg)
    return np.column_stack(
        [nodes_proj.geometry.x.to_numpy(), nodes_proj.geometry.y.to_numpy()]
    )


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

    # Custom point-table identifiers may be strings, for example OSM-style
    # facility IDs. Keep identifiers stable instead of forcing integer IDs.
    result[id_col] = result[id_col].astype(str)

    point_xy = _point_xy(points, result, projected_epsg)
    node_xy = _node_xy(nodes, projected_epsg)

    tree = cKDTree(node_xy)
    distances, idx = tree.query(point_xy, k=1)

    result['nearest_node'] = nodes.iloc[idx]['id'].to_numpy(dtype='int64')
    result[distance_col] = distances.astype('float64')

    result = result.drop_duplicates(subset=id_col).set_index(id_col, drop=False)

    if not keep_geometry:
        result = pd.DataFrame(result.drop(columns='geometry')).copy()

    if verbose:
        print(
            f'Snapped {len(result):,} points in {pc() - t0:.2f} seconds, '
            f'mean {distance_col} {np.mean(distances):.2f} m'
        )

    return result
