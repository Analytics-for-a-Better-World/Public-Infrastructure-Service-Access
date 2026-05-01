'''
Facility loading utilities.
'''

from time import perf_counter as pc

import geopandas as gpd
import numpy as np
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
