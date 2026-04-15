from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def build_regular_grid_within_polygon(
    polygon_gdf: gpd.GeoDataFrame,
    spacing_m: float,
    include_boundary: bool = True,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''
    Create a regular grid of candidate points within a polygon geometry.

    Parameters
    ----------
    polygon_gdf
        GeoDataFrame containing polygon geometry in a projected CRS with meter units.
    spacing_m
        Grid spacing in meters.
    include_boundary
        Whether to keep points on the boundary.
    verbose
        Whether to print progress information.

    Returns
    -------
    gpd.GeoDataFrame
        Candidate points with ID, Longitude, Latitude, and geometry.
    '''
    if polygon_gdf.empty:
        raise ValueError('polygon_gdf is empty')

    if polygon_gdf.crs is None:
        raise ValueError('polygon_gdf must have a CRS')

    if spacing_m <= 0:
        raise ValueError('spacing_m must be positive')

    polygon = polygon_gdf.union_all()
    minx, miny, maxx, maxy = polygon.bounds

    xs = np.arange(minx, maxx + spacing_m, spacing_m, dtype='float64')
    ys = np.arange(miny, maxy + spacing_m, spacing_m, dtype='float64')

    if verbose:
        print(
            f'Creating regular grid with spacing {spacing_m:,.0f} m, '
            f'{len(xs):,} columns, {len(ys):,} rows'
        )

    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    coords = np.column_stack((xx.ravel(), yy.ravel()))

    points = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(
            [Point(x, y) for x, y in coords],
            crs=polygon_gdf.crs,
        )
    )

    if include_boundary:
        mask = points.within(polygon) | points.touches(polygon)
    else:
        mask = points.within(polygon)

    points = points.loc[mask].copy()
    points.reset_index(drop=True, inplace=True)
    points['ID'] = np.arange(len(points), dtype='int64')

    points_wgs84 = points.to_crs(4326)
    points['Longitude'] = points_wgs84.geometry.x.to_numpy()
    points['Latitude'] = points_wgs84.geometry.y.to_numpy()

    if verbose:
        print(
            f'Created regular grid, {len(points):,} points kept inside '
            f'the country geometry'
        )

    return points


def exclude_points_on_water(
    candidates: gpd.GeoDataFrame,
    water_bodies: gpd.GeoDataFrame,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''
    Remove candidate points that intersect water body polygons.

    Parameters
    ----------
    candidates
        Candidate point GeoDataFrame.
    water_bodies
        Water body polygon GeoDataFrame.
    verbose
        Whether to print progress information.

    Returns
    -------
    gpd.GeoDataFrame
        Candidate points not intersecting water.
    '''
    if candidates.empty:
        return candidates.copy()

    if water_bodies.empty:
        if verbose:
            print('No water bodies found, no candidate points removed')
        return candidates.copy()

    if candidates.crs != water_bodies.crs:
        water_bodies = water_bodies.to_crs(candidates.crs)

    joined = gpd.sjoin(
        candidates[['geometry']].copy(),
        water_bodies[['geometry']].copy(),
        how='left',
        predicate='intersects',
    )

    matched_idx = joined.loc[joined['index_right'].notna()].index.unique()
    keep_mask = ~candidates.index.isin(matched_idx)

    result = candidates.loc[keep_mask].copy()
    result.reset_index(drop=True, inplace=True)
    result['ID'] = np.arange(len(result), dtype='int64')

    if verbose:
        removed = len(candidates) - len(result)
        print(
            f'Removed {removed:,} candidate points on water, '
            f'{len(result):,} remain'
        )

    return result


def filter_snapped_candidates_by_distance(
    candidates: pd.DataFrame,
    max_snap_dist_m: float | None,
    distance_col: str = 'candidate_dist_road',
    verbose: bool = True,
) -> pd.DataFrame:
    '''
    Filter snapped candidate points by maximum snapping distance.

    Parameters
    ----------
    candidates
        DataFrame of snapped candidate points.
    max_snap_dist_m
        Maximum allowed snapping distance in meters. If None, no filtering is applied.
    distance_col
        Name of the snapping distance column.
    verbose
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        Filtered candidate points.
    '''
    if max_snap_dist_m is None:
        return candidates.copy()

    if distance_col not in candidates.columns:
        raise ValueError(f'Missing required distance column: {distance_col}')

    result = candidates.loc[candidates[distance_col] <= max_snap_dist_m].copy()
    result.reset_index(drop=True, inplace=True)

    if 'ID' in result.columns:
        result['ID'] = np.arange(len(result), dtype='int64')

    if verbose:
        removed = len(candidates) - len(result)
        print(
            f'Removed {removed:,} candidate points with snap distance above '
            f'{max_snap_dist_m:,.2f} m, {len(result):,} remain'
        )

    return result