'''
Population data utilities.
'''

from pathlib import Path
from time import perf_counter as pc

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import xy

from distance_pipeline.source_tables import normalize_custom_points


def _compute_total_population(
    *,
    band: np.ndarray,
    nodata: float | int | None,
) -> float:
    '''
    Compute total population in a raster band, ignoring nodata.
    '''
    mask = np.isfinite(band)
    if nodata is not None:
        mask &= band != nodata
    return float(band[mask].sum())


def _aggregate_raster_by_sum(
    *,
    band: np.ndarray,
    transform: Affine,
    nodata: float | int | None,
    factor: int,
) -> tuple[np.ndarray, Affine]:
    '''
    Aggregate a raster band by summing non-overlapping square blocks.
    '''
    if factor < 1:
        raise ValueError('factor must be at least 1')

    if factor == 1:
        return band.astype('float64'), transform

    height, width = band.shape
    new_height = height // factor
    new_width = width // factor

    if new_height == 0 or new_width == 0:
        raise ValueError('factor is larger than the raster dimensions')

    trimmed = band[: new_height * factor, : new_width * factor].astype('float64')

    valid = np.isfinite(trimmed)
    if nodata is not None:
        valid &= trimmed != nodata

    values = np.where(valid, trimmed, 0.0)

    aggregated = values.reshape(
        new_height,
        factor,
        new_width,
        factor,
    ).sum(axis=(1, 3))

    new_transform = transform * Affine.scale(factor, factor)

    return aggregated, new_transform


def worldpop_to_points(
    tif_path: str | Path,
    population_threshold: float = 1.0,
    sample_fraction: float = 1.0,
    max_points: int | None = None,
    random_seed: int = 42,
    aggregate_factor: int | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''
    Convert a population raster into point centroids, with optional aggregation.

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

    if aggregate_factor is not None and aggregate_factor < 1:
        raise ValueError('aggregate_factor must be at least 1')

    with rasterio.open(tif_path) as src:
        band = src.read(1).astype('float64')
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # --- totals before aggregation ---
    total_before = _compute_total_population(band=band, nodata=nodata)

    if verbose:
        print(f'Raster shape: {band.shape}')
        print(f'Total population before aggregation: {total_before:,.0f}')

    # --- aggregation ---
    if aggregate_factor is not None:
        height, width = band.shape

        if verbose:
            if height % aggregate_factor != 0 or width % aggregate_factor != 0:
                print(
                    'Warning, raster not divisible by aggregate_factor, '
                    'edge cells will be trimmed'
                )

        band, transform = _aggregate_raster_by_sum(
            band=band,
            transform=transform,
            nodata=nodata,
            factor=aggregate_factor,
        )

        total_after = float(band.sum())

        if verbose:
            print(f'Raster shape after aggregation: {band.shape}')
            print(f'Total population after aggregation: {total_after:,.0f}')
            print(
                f'Difference (after - before): {total_after - total_before:,.2f}'
            )

    # --- masking ---
    mask = np.isfinite(band)
    if nodata is not None and aggregate_factor is None:
        mask &= band != nodata
    mask &= band >= population_threshold

    rows, cols = np.where(mask)
    values = band[rows, cols].astype('float64')

    if len(values) == 0:
        raise ValueError('No raster cells found above the threshold')

    total_retained = float(values.sum())

    if verbose:
        print(f'Cells retained after threshold: {len(values):,}')
        print(f'Total population retained: {total_retained:,.0f}')

    rng = np.random.default_rng(random_seed)

    # --- sampling ---
    if sample_fraction < 1.0:
        n_sample = max(1, int(len(values) * sample_fraction))
        idx = rng.choice(len(values), size=n_sample, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        values = values[idx]

        if verbose:
            print(f'Sampled down to {len(values):,} points')

    if max_points is not None and len(values) > max_points:
        idx = rng.choice(len(values), size=max_points, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        values = values[idx]

        if verbose:
            print(f'Capped to {len(values):,} points')

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


def table_to_population_points(
    table_path: str | Path,
    population_threshold: float = 1.0,
    sample_fraction: float = 1.0,
    max_points: int | None = None,
    random_seed: int = 42,
    aggregate_factor: int | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''
    Convert a point population table into the population-point schema.

    This supports Meta Data for Good / Facebook HRSL-style point exports and
    other already-disaggregated population tables. Coordinate columns are
    normalized using the same logic as custom source/destination tables.
    '''
    t0 = pc()

    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f'File not found: {table_path}')

    if aggregate_factor is not None:
        raise ValueError(
            'aggregate_factor is only supported for raster population data. '
            'Use a pre-aggregated point table or omit --aggregate-factor.'
        )

    if not 0 < sample_fraction <= 1:
        raise ValueError('sample_fraction must be in the interval (0, 1]')

    suffix = table_path.suffix.lower()
    if suffix == '.parquet':
        raw = pd.read_parquet(table_path)
    elif suffix in {'.geojson', '.gpkg', '.shp'}:
        raw = gpd.read_file(table_path)
    else:
        raw = pd.read_csv(table_path)

    lower_to_actual = {str(col).lower(): col for col in raw.columns}
    for candidate in (
        'population',
        'pop',
        'population_count',
        'population_estimate',
        'population_2020',
        'value',
    ):
        actual = lower_to_actual.get(candidate)
        if actual is not None:
            if actual != 'population':
                raw = raw.rename(columns={actual: 'population'})
            break

    gdf = normalize_custom_points(raw, prefix=table_path.stem)
    gdf['population'] = pd.to_numeric(gdf['population'], errors='coerce')
    gdf = gdf[gdf['population'].notna()].copy()
    gdf = gdf[gdf['population'] >= population_threshold].copy()

    if gdf.empty:
        raise ValueError('No table rows found above the population threshold')

    rng = np.random.default_rng(random_seed)

    if sample_fraction < 1.0:
        n_sample = max(1, int(len(gdf) * sample_fraction))
        idx = rng.choice(len(gdf), size=n_sample, replace=False)
        gdf = gdf.iloc[idx].copy()

        if verbose:
            print(f'Sampled down to {len(gdf):,} points')

    if max_points is not None and len(gdf) > max_points:
        idx = rng.choice(len(gdf), size=max_points, replace=False)
        gdf = gdf.iloc[idx].copy()

        if verbose:
            print(f'Capped to {len(gdf):,} points')

    gdf = gdf.reset_index(drop=True)
    gdf['ID'] = np.arange(len(gdf), dtype='int64')

    keep_cols = [
        'ID',
        'Longitude',
        'Latitude',
        'population',
        'geometry',
    ]
    gdf = gdf[keep_cols].copy()

    if verbose:
        print(
            f'Created {len(gdf):,} population points from table '
            f'in {pc() - t0:.2f} seconds'
        )

    return gdf


def population_to_points(
    population_path: str | Path,
    population_threshold: float = 1.0,
    sample_fraction: float = 1.0,
    max_points: int | None = None,
    random_seed: int = 42,
    aggregate_factor: int | None = None,
    data_format: str = 'auto',
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''
    Convert raster or point-table population data into population points.

    ``data_format='auto'`` treats common raster extensions as rasters and all
    other files as point tables.
    '''
    population_path = Path(population_path)
    format_key = data_format.lower().strip()
    if format_key not in {'auto', 'raster', 'table'}:
        raise ValueError("data_format must be 'auto', 'raster', or 'table'.")

    raster_suffixes = {'.tif', '.tiff', '.vrt'}
    if format_key == 'auto':
        format_key = (
            'raster'
            if population_path.suffix.lower() in raster_suffixes
            else 'table'
        )

    if format_key == 'raster':
        return worldpop_to_points(
            population_path,
            population_threshold=population_threshold,
            sample_fraction=sample_fraction,
            max_points=max_points,
            random_seed=random_seed,
            aggregate_factor=aggregate_factor,
            verbose=verbose,
        )

    return table_to_population_points(
        population_path,
        population_threshold=population_threshold,
        sample_fraction=sample_fraction,
        max_points=max_points,
        random_seed=random_seed,
        aggregate_factor=aggregate_factor,
        verbose=verbose,
    )
