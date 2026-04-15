from time import perf_counter as pc

import geopandas as gpd
import pandas as pd
from pyrosm import OSM


WATER_NATURAL_VALUES: tuple[str, ...] = (
    'water',
    'wetland',
    'bay',
    'strait',
)
WATER_LANDUSE_VALUES: tuple[str, ...] = (
    'reservoir',
    'basin',
    'salt_pond',
)
WATERWAY_POLYGON_VALUES: tuple[str, ...] = (
    'riverbank',
)


def _clean_polygon_layer(layer: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    '''Return a clean polygon layer with WGS84 CRS when available.'''
    if layer is None or len(layer) == 0:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    result = layer.copy()
    if result.crs is None:
        result = result.set_crs(epsg=4326)

    result = result.loc[~result.geometry.isna()].copy()
    polygon_mask = result.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    result = result.loc[polygon_mask].copy()
    return result


def _subset_values(
    layer: gpd.GeoDataFrame,
    column: str,
    values: tuple[str, ...],
) -> gpd.GeoDataFrame:
    '''Filter a layer to the provided tag values.'''
    if layer.empty or column not in layer.columns:
        return gpd.GeoDataFrame(geometry=[], crs=layer.crs)

    mask = pd.Series(layer[column]).isin(values).to_numpy()
    result = layer.loc[mask, ['geometry']].copy()
    return result


def _load_waterway_polygons(osm: OSM) -> gpd.GeoDataFrame:
    '''Load riverbank polygons when the pyrosm helper is available.'''
    try:
        layer = osm.get_data_by_custom_criteria(
            custom_filter={'waterway': list(WATERWAY_POLYGON_VALUES)}
        )
    except AttributeError:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return _subset_values(_clean_polygon_layer(layer), 'waterway', WATERWAY_POLYGON_VALUES)


def load_water_bodies(
    pbf_path: str,
    projected_epsg: int,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Load water body polygons from OSM and project them.'''
    t0 = pc()
    osm = OSM(str(pbf_path))

    natural = _subset_values(_clean_polygon_layer(osm.get_natural()), 'natural', WATER_NATURAL_VALUES)
    landuse = _subset_values(_clean_polygon_layer(osm.get_landuse()), 'landuse', WATER_LANDUSE_VALUES)
    waterways = _load_waterway_polygons(osm)

    layers: list[gpd.GeoDataFrame] = [layer for layer in (natural, landuse, waterways) if not layer.empty]
    if not layers:
        result = gpd.GeoDataFrame(geometry=[], crs=f'EPSG:{projected_epsg}')
        if verbose:
            print(f'Loaded water bodies in {pc() - t0:.2f} seconds, 0 polygons')
        return result

    result = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(
            [geom for layer in layers for geom in layer.geometry],
            crs=layers[0].crs,
        )
    )
    result = result.to_crs(epsg=projected_epsg)
    result = result.loc[~result.geometry.isna()].copy()
    result.reset_index(drop=True, inplace=True)

    if verbose:
        print(
            f'Loaded water bodies in {pc() - t0:.2f} seconds, '
            f'{len(result):,} polygons'
        )

    return result
