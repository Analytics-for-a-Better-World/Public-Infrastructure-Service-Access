'''
Facility loading utilities.
'''

from time import perf_counter as pc

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM


DEFAULT_AMENITY_VALUES: list[str] = [
    'hospital',
    'clinic',
    'doctors',
    'dentist',
    'pharmacy',
    'health_post',
    'nursing_home',
    'social_facility',
]


def load_facilities(
    pbf_path: str,
    amenity_values: list[str] | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Extract facilities from OSM using amenity values.'''

    t0 = pc()

    if amenity_values is None:
        amenity_values = DEFAULT_AMENITY_VALUES

    osm = OSM(str(pbf_path))

    custom_filter: dict[str, list[str] | bool] = {
        'amenity': amenity_values,
    }

    gdf = osm.get_pois(custom_filter=custom_filter)

    if gdf is None or len(gdf) == 0:
        raise ValueError('No matching OSM POIs found')

    gdf = gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf['osm_geometry_type'] = gdf.geometry.geom_type.astype(str)

    if 'name' not in gdf.columns:
        gdf['name'] = None

    gdf['Name'] = gdf['name'].fillna('Unnamed facility')

    if 'id' in gdf.columns:
        stable_ids = gdf['id']
    elif 'osm_id' in gdf.columns:
        stable_ids = gdf['osm_id']
    else:
        stable_ids = np.arange(len(gdf), dtype='int64')

    gdf['ID'] = np.asarray(stable_ids, dtype='int64')

    if verbose:
        print(
            f'Loaded facilities in {pc() - t0:.2f} seconds, '
            f'{len(gdf):,} features'
        )

    return gdf


def _normalized_text_key(values: pd.Series) -> pd.Series:
    """Return a stable, low-noise text key for duplicate matching."""
    return (
        values.fillna('')
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r'\s+', ' ', regex=True)
    )


def deduplicate_osm_amenities(
    facilities: gpd.GeoDataFrame,
    *,
    projected_epsg: int,
    tolerance_m: float = 25.0,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Remove duplicate OSM amenity features, preferring point/node features.

    OSM often contains the same facility twice: once as a point POI and once as
    a polygon/building footprint. The pipeline routes from a single facility
    point, so this helper keeps one representative feature per nearby
    name/amenity pair and ranks original point geometries ahead of centroids.
    """
    t0 = pc()

    if facilities.crs is None:
        raise ValueError('facilities has no CRS')
    if facilities.empty:
        return facilities.copy()

    result = facilities.copy()
    if 'osm_geometry_type' in result.columns:
        result['_dedup_geometry_type'] = result['osm_geometry_type'].astype(str)
    else:
        result['_dedup_geometry_type'] = result.geometry.geom_type.astype(str)
    result['_dedup_is_point'] = result['_dedup_geometry_type'].isin(
        ['Point', 'MultiPoint']
    )

    if 'amenity' in result.columns:
        result['_dedup_amenity'] = _normalized_text_key(result['amenity'])
    else:
        result['_dedup_amenity'] = ''

    if 'name' in result.columns:
        result['_dedup_name'] = _normalized_text_key(result['name'])
    elif 'Name' in result.columns:
        result['_dedup_name'] = _normalized_text_key(result['Name'])
    else:
        result['_dedup_name'] = ''

    projected = result.to_crs(epsg=projected_epsg)
    cell_size = float(tolerance_m)
    projected['_dedup_cell_x'] = np.floor(projected.geometry.x / cell_size).astype(
        'int64'
    )
    projected['_dedup_cell_y'] = np.floor(projected.geometry.y / cell_size).astype(
        'int64'
    )

    result['_dedup_cell_x'] = projected['_dedup_cell_x'].to_numpy()
    result['_dedup_cell_y'] = projected['_dedup_cell_y'].to_numpy()
    result['_dedup_original_order'] = np.arange(len(result), dtype='int64')
    result['_dedup_rank'] = np.where(result['_dedup_is_point'], 0, 1)

    # Named duplicates are matched by name and amenity. Unnamed facilities fall
    # back to amenity plus spatial cell so distinct unnamed amenities are kept.
    named = result['_dedup_name'] != ''
    result['_dedup_group_name'] = np.where(
        named,
        result['_dedup_name'],
        '__unnamed__',
    )

    sort_cols = [
        '_dedup_amenity',
        '_dedup_group_name',
        '_dedup_cell_x',
        '_dedup_cell_y',
        '_dedup_rank',
        '_dedup_original_order',
    ]
    subset_cols = [
        '_dedup_amenity',
        '_dedup_group_name',
        '_dedup_cell_x',
        '_dedup_cell_y',
    ]

    result = (
        result.sort_values(sort_cols)
        .drop_duplicates(subset=subset_cols, keep='first')
        .sort_values('_dedup_original_order')
    )

    helper_cols = [col for col in result.columns if col.startswith('_dedup_')]
    result = result.drop(columns=helper_cols)
    result = result.reset_index(drop=True)

    if 'ID' not in result.columns or result['ID'].duplicated().any():
        result['ID'] = np.arange(len(result), dtype='int64')

    if verbose:
        removed = len(facilities) - len(result)
        print(
            f'Deduplicated OSM amenities in {pc() - t0:.2f} seconds, '
            f'{removed:,} duplicate features removed'
        )

    return gpd.GeoDataFrame(result, geometry='geometry', crs=facilities.crs)


def load_health_facilities(
    pbf_path: str,
    amenity_values: list[str] | None = None,
    include_healthcare_tag: bool = True,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Backward-compatible alias for the generic facility loader.'''
    return load_facilities(
        pbf_path=pbf_path,
        amenity_values=amenity_values,
        verbose=verbose,
    )
