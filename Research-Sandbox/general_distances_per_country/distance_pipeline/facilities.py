'''
Facility loading utilities.
'''

from time import perf_counter as pc
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM
from shapely.geometry import LineString, Point, Polygon

try:
    import osmium
except ImportError:  # pragma: no cover - optional backend
    osmium = None


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

FacilityBackend = Literal['pyrosm', 'osmium', 'auto']
BBox = tuple[float, float, float, float] | list[float]


def _resolve_facility_backend(backend: FacilityBackend) -> Literal['pyrosm', 'osmium']:
    if backend == 'auto':
        return 'osmium' if osmium is not None else 'pyrosm'
    return backend


def _tag_value(tags: object, key: str) -> str | None:
    value = tags.get(key)
    if value is None:
        return None
    return str(value)


def _encoded_osm_id(osm_type: str, osm_id: int) -> int:
    if osm_type == 'node':
        return int(osm_id)
    if osm_type == 'way':
        return -int(osm_id)
    return -(1_000_000_000_000_000 + int(osm_id))


def _point_in_bbox(lon: float, lat: float, bbox: BBox | None) -> bool:
    if bbox is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _geometry_intersects_bbox(geometry: object, bbox: BBox | None) -> bool:
    if bbox is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    geom_min_lon, geom_min_lat, geom_max_lon, geom_max_lat = geometry.bounds
    return not (
        geom_max_lon < min_lon
        or geom_min_lon > max_lon
        or geom_max_lat < min_lat
        or geom_min_lat > max_lat
    )


def _geometry_from_coords(
    coords: list[tuple[float, float]],
) -> Point | LineString | Polygon | None:
    if not coords:
        return None
    if len(coords) == 1:
        return Point(coords[0])
    if len(coords) >= 4 and coords[0] == coords[-1]:
        polygon = Polygon(coords)
        if polygon.is_valid and not polygon.is_empty:
            return polygon
    return LineString(coords)


def _copy_tags(tags: object) -> dict[str, str]:
    return {str(tag.k): str(tag.v) for tag in tags}


def _facility_record(
    *,
    osm_type: str,
    osm_id: int,
    tags: object,
    geometry: Point | LineString | Polygon,
    amenity_values: set[str],
    bbox: BBox | None,
) -> dict[str, object] | None:
    amenity = _tag_value(tags, 'amenity')
    if amenity not in amenity_values:
        return None
    if not _geometry_intersects_bbox(geometry, bbox):
        return None

    name = _tag_value(tags, 'name')
    return {
        'id': int(osm_id),
        'osm_id': int(osm_id),
        'osm_type': osm_type,
        'amenity': amenity,
        'healthcare': _tag_value(tags, 'healthcare'),
        'name': name,
        'Name': name if name else 'Unnamed facility',
        'addr:street': _tag_value(tags, 'addr:street'),
        'addr:housenumber': _tag_value(tags, 'addr:housenumber'),
        'addr:city': _tag_value(tags, 'addr:city'),
        'osm_geometry_type': geometry.geom_type,
        'ID': _encoded_osm_id(osm_type, int(osm_id)),
        'geometry': geometry,
    }


class OsmiumAmenityScanner(osmium.SimpleHandler if osmium is not None else object):
    """First pass over OSM amenities without a full node-location index."""

    def __init__(
        self,
        amenity_values: list[str],
        bbox: BBox | None = None,
    ) -> None:
        if osmium is None:
            raise ImportError(
                "The optional 'osmium' package is required for the osmium backend."
            )
        super().__init__()
        self.amenity_values = set(amenity_values)
        self.bbox = bbox
        self.records: list[dict[str, object]] = []
        self.way_specs: list[dict[str, object]] = []
        self.needed_node_ids: set[int] = set()

    def node(self, node: object) -> None:
        if not node.location.valid():
            return
        lon = float(node.location.lon)
        lat = float(node.location.lat)
        if not _point_in_bbox(lon, lat, self.bbox):
            return
        record = _facility_record(
            osm_type='node',
            osm_id=int(node.id),
            tags=node.tags,
            geometry=Point(lon, lat),
            amenity_values=self.amenity_values,
            bbox=self.bbox,
        )
        if record is not None:
            self.records.append(record)

    def way(self, way: object) -> None:
        amenity = _tag_value(way.tags, 'amenity')
        if amenity not in self.amenity_values:
            return

        node_refs = [int(node.ref) for node in way.nodes]
        if not node_refs:
            return

        self.needed_node_ids.update(node_refs)
        self.way_specs.append(
            {
                'osm_id': int(way.id),
                'tags': _copy_tags(way.tags),
                'node_refs': node_refs,
            }
        )


class OsmiumWayNodeCollector(osmium.SimpleHandler if osmium is not None else object):
    """Second pass collecting coordinates for nodes used by matched ways."""

    def __init__(self, needed_node_ids: set[int]) -> None:
        if osmium is None:
            raise ImportError(
                "The optional 'osmium' package is required for the osmium backend."
            )
        super().__init__()
        self.needed_node_ids = needed_node_ids
        self.node_coords: dict[int, tuple[float, float]] = {}

    def node(self, node: object) -> None:
        node_id = int(node.id)
        if node_id not in self.needed_node_ids:
            return
        if not node.location.valid():
            return
        self.node_coords[node_id] = (
            float(node.location.lon),
            float(node.location.lat),
        )


def _load_facilities_osmium(
    pbf_path: str,
    amenity_values: list[str],
    bbox: BBox | None,
    verbose: bool,
) -> gpd.GeoDataFrame:
    t0 = pc()

    if osmium is None:
        raise ImportError(
            "The optional 'osmium' package is required for the osmium backend."
        )

    if verbose:
        print('Scanning OSM amenities with osmium without a full node-location index')

    amenity_filter = osmium.filter.TagFilter(
        *((('amenity', value) for value in amenity_values))
    )

    scanner = OsmiumAmenityScanner(amenity_values=amenity_values, bbox=bbox)
    scanner.apply_file(str(pbf_path), locations=False, filters=[amenity_filter])
    records = list(scanner.records)

    if verbose:
        print(
            f'First osmium amenity pass found {len(records):,} node features '
            f'and {len(scanner.way_specs):,} way features'
        )

    if scanner.way_specs:
        if verbose:
            print(
                f'Collecting coordinates for {len(scanner.needed_node_ids):,} '
                'nodes referenced by matching amenity ways'
            )

        node_filter = osmium.filter.IdFilter(scanner.needed_node_ids)
        collector = OsmiumWayNodeCollector(scanner.needed_node_ids)
        collector.apply_file(str(pbf_path), locations=False, filters=[node_filter])

        amenity_set = set(amenity_values)
        for spec in scanner.way_specs:
            coords = [
                collector.node_coords[node_ref]
                for node_ref in spec['node_refs']
                if node_ref in collector.node_coords
            ]
            geometry = _geometry_from_coords(coords)
            if geometry is None:
                continue
            record = _facility_record(
                osm_type='way',
                osm_id=int(spec['osm_id']),
                tags=spec['tags'],
                geometry=geometry,
                amenity_values=amenity_set,
                bbox=bbox,
            )
            if record is not None:
                records.append(record)

        if verbose:
            print(
                f'Second osmium amenity pass collected '
                f'{len(collector.node_coords):,} referenced node coordinates'
            )

    if not records:
        raise ValueError('No matching OSM POIs found')

    gdf = gpd.GeoDataFrame(
        records,
        geometry='geometry',
        crs='EPSG:4326',
    )

    if gdf['ID'].duplicated().any():
        gdf['ID'] = np.arange(len(gdf), dtype='int64')

    if verbose:
        area = 'bbox extract' if bbox is not None else 'full extract'
        print(
            f'Loaded facilities with osmium ({area}) in {pc() - t0:.2f} seconds, '
            f'{len(gdf):,} features'
        )

    return gdf


def load_facilities(
    pbf_path: str,
    amenity_values: list[str] | None = None,
    backend: FacilityBackend = 'pyrosm',
    bbox: BBox | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Extract facilities from OSM using amenity values.'''

    t0 = pc()

    if amenity_values is None:
        amenity_values = DEFAULT_AMENITY_VALUES

    resolved_backend = _resolve_facility_backend(backend)
    if resolved_backend == 'osmium':
        return _load_facilities_osmium(
            pbf_path=pbf_path,
            amenity_values=amenity_values,
            bbox=bbox,
            verbose=verbose,
        )

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
    backend: FacilityBackend = 'pyrosm',
    bbox: BBox | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Backward-compatible alias for the generic facility loader.'''
    return load_facilities(
        pbf_path=pbf_path,
        amenity_values=amenity_values,
        backend=backend,
        bbox=bbox,
        verbose=verbose,
    )
