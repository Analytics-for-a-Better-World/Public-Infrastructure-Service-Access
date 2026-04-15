'''
Facility loading utilities.
'''

from time import perf_counter as pc

import geopandas as gpd
import numpy as np
from pyrosm import OSM


def load_health_facilities(
    pbf_path: str,
    amenity_values: list[str] | None = None,
    include_healthcare_tag: bool = True,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Extract health related facilities from OSM.'''

    t0 = pc()

    if amenity_values is None:
        amenity_values = [
            'hospital',
            'clinic',
            'doctors',
            'dentist',
            'pharmacy',
            'health_post',
            'nursing_home',
            'social_facility',
        ]

    osm = OSM(str(pbf_path))

    custom_filter: dict[str, list[str] | bool] = {
        'amenity': amenity_values,
    }
    if include_healthcare_tag:
        custom_filter['healthcare'] = True

    gdf = osm.get_pois(custom_filter=custom_filter)

    if gdf is None or len(gdf) == 0:
        raise ValueError('No health related OSM POIs found')

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
            f'Loaded health facilities in {pc() - t0:.2f} seconds, '
            f'{len(gdf):,} features'
        )

    return gdf
