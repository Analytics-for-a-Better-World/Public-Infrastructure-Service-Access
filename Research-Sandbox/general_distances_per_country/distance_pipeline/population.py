'''
Population data utilities.
'''

from pathlib import Path
from time import perf_counter as pc

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy


def worldpop_to_points(
    tif_path: str | Path,
    population_threshold: float = 1.0,
    sample_fraction: float = 1.0,
    max_points: int | None = None,
    random_seed: int = 42,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''
    Convert a population raster into point centroids.

    Parameters
    ----------
    tif_path
        Path to the raster.
    population_threshold
        Minimum raster value to keep.
    sample_fraction
        Fraction of retained cells to sample.
    max_points
        Optional hard cap on points kept.
    random_seed
        Random seed used in sampling.
    verbose
        Whether to print timing information.

    Returns
    -------
    gpd.GeoDataFrame
        Population points.
    '''
    t0 = pc()

    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f'File not found: {tif_path}')

    if not 0 < sample_fraction <= 1:
        raise ValueError('sample_fraction must be in the interval (0, 1]')

    with rasterio.open(tif_path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    mask = np.isfinite(band)
    if nodata is not None:
        mask &= band != nodata
    mask &= band >= population_threshold

    rows, cols = np.where(mask)
    values = band[rows, cols].astype('float64')

    if len(values) == 0:
        raise ValueError('No raster cells found above the threshold')

    rng = np.random.default_rng(random_seed)

    if sample_fraction < 1.0:
        n_sample = max(1, int(len(values) * sample_fraction))
        idx = rng.choice(len(values), size=n_sample, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        values = values[idx]

    if max_points is not None and len(values) > max_points:
        idx = rng.choice(len(values), size=max_points, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        values = values[idx]

    xs, ys = xy(transform, rows, cols, offset='center')

    gdf = gpd.GeoDataFrame(
        {
            'ID': np.arange(len(values), dtype='int64'),
            'Longitude': np.asarray(xs, dtype='float64'),
            'Latitude': np.asarray(ys, dtype='float64'),
            'population': values,
            'row': rows.astype('int32'),
            'col': cols.astype('int32'),
        },
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )

    if verbose:
        print(
            f'Created {len(gdf):,} population points in {pc() - t0:.2f} seconds'
        )

    return gdf
