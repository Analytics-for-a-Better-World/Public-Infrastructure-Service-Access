'''
Network loading utilities.
'''

from time import perf_counter as pc
from typing import Literal
from importlib.metadata import PackageNotFoundError, version
import sys
import warnings

import geopandas as gpd
import pandas as pd
from pyproj import Geod
from pyrosm import OSM
from shapely.geometry import LineString, Point


def _version_numbers(value: str) -> tuple[int, ...]:
    """Return numeric version components from a package version string."""
    cleaned = value.split('+', 1)[0].split('-', 1)[0]
    numbers: list[int] = []
    for part in cleaned.split('.'):
        digits = ''.join(ch for ch in part if ch.isdigit())
        if digits == '':
            break
        numbers.append(int(digits))
    return tuple(numbers)


def _version_at_least(value: str, minimum: tuple[int, ...]) -> bool:
    numbers = _version_numbers(value)
    width = max(len(numbers), len(minimum))
    return numbers + (0,) * (width - len(numbers)) >= minimum + (0,) * (width - len(minimum))


def _version_at_most(value: str, maximum: tuple[int, ...]) -> bool:
    numbers = _version_numbers(value)
    width = max(len(numbers), len(maximum))
    return numbers + (0,) * (width - len(numbers)) <= maximum + (0,) * (width - len(maximum))


def warn_if_pandana_numpy_incompatible() -> None:
    """Warn before importing Pandana when the installed NumPy is likely incompatible."""
    try:
        pandana_version = version('pandana')
        numpy_version = version('numpy')
    except PackageNotFoundError:
        return

    if not (
        _version_at_most(pandana_version, (0, 7))
        and _version_at_least(numpy_version, (2, 0))
    ):
        return

    message = (
        'WARNING: Pandana compatibility risk detected. '
        f'Installed pandana=={pandana_version} is likely to fail with '
        f'numpy=={numpy_version}. Pandana 0.7 wheels were built against the '
        'NumPy 1.x C API, so importing or using Pandana under NumPy 2 can raise '
        'binary-compatibility errors. Use numpy<2, or install a Pandana build '
        'that explicitly supports NumPy 2, before running this distance pipeline.'
    )
    print(message, file=sys.stderr)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


warn_if_pandana_numpy_incompatible()

import pandana as pdna

try:
    import osmium
except ImportError:  # pragma: no cover - optional backend
    osmium = None


BBox = tuple[float, float, float, float] | list[float]
NetworkBackend = Literal['pyrosm', 'osmium', 'auto']

DRIVABLE_HIGHWAYS: set[str] = {
    'motorway',
    'motorway_link',
    'trunk',
    'trunk_link',
    'primary',
    'primary_link',
    'secondary',
    'secondary_link',
    'tertiary',
    'tertiary_link',
    'unclassified',
    'residential',
    'living_street',
    'service',
    'track',
    'road',
}

BLOCKING_ACCESS_VALUES: set[str] = {
    'no',
    'private',
    'agricultural',
    'forestry',
    'delivery',
}

ONEWAY_FORWARD: set[str] = {'yes', 'true', '1'}
ONEWAY_REVERSE: set[str] = {'-1', 'reverse'}
ONEWAY_NO: set[str] = {'no', 'false', '0'}
GEOD = Geod(ellps='WGS84')


def _pyrosm_bbox(bbox: BBox | None) -> list[float] | None:
    '''Return a pyrosm-compatible bbox list, or None for the full extract.'''
    if bbox is None:
        return None
    min_lon, min_lat, max_lon, max_lat = bbox
    return [min_lon, min_lat, max_lon, max_lat]


def _resolve_network_backend(backend: NetworkBackend) -> Literal['pyrosm', 'osmium']:
    '''Resolve auto network backend selection.'''
    if backend == 'auto':
        return 'osmium' if osmium is not None else 'pyrosm'
    return backend


def _segment_intersects_bbox(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    bbox: BBox | None,
) -> bool:
    '''Cheap bbox test for a road segment.'''
    if bbox is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return not (
        max(lon1, lon2) < min_lon
        or min(lon1, lon2) > max_lon
        or max(lat1, lat2) < min_lat
        or min(lat1, lat2) > max_lat
    )


def _tag_value(tags: object, key: str) -> str | None:
    '''Return a tag value as a plain string, if present.'''
    value = tags.get(key)
    if value is None:
        return None
    return str(value)


def _is_drivable_way(tags: object) -> bool:
    '''Return whether OSM tags describe a drivable road segment.'''
    highway = _tag_value(tags, 'highway')
    if highway not in DRIVABLE_HIGHWAYS:
        return False

    for key in ('access', 'vehicle', 'motor_vehicle', 'motorcar'):
        value = _tag_value(tags, key)
        if value in BLOCKING_ACCESS_VALUES:
            return False

    return True


def _way_directions(tags: object) -> tuple[bool, bool]:
    '''Return whether a way should be emitted forward and backward.'''
    oneway = (_tag_value(tags, 'oneway') or '').strip().lower()
    junction = (_tag_value(tags, 'junction') or '').strip().lower()

    if oneway in ONEWAY_REVERSE:
        return False, True
    if oneway in ONEWAY_FORWARD or junction == 'roundabout':
        return True, False
    if oneway in ONEWAY_NO:
        return True, True
    return True, True


def _segment_length_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    '''Return geodesic segment length in meters.'''
    _, _, distance_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance_m)


class OsmiumDrivingNetworkHandler(osmium.SimpleHandler if osmium is not None else object):
    '''Streaming OSM PBF handler for a pyrosm-like driving graph.'''

    def __init__(self, bbox: BBox | None = None) -> None:
        if osmium is None:
            raise ImportError(
                "The optional 'osmium' package is required for the osmium backend."
            )
        super().__init__()
        self.bbox = bbox
        self.nodes: dict[int, tuple[float, float]] = {}
        self.edge_records: list[dict[str, object]] = []

    def way(self, way: object) -> None:
        tags = way.tags
        if not _is_drivable_way(tags):
            return

        highway = _tag_value(tags, 'highway')
        name = _tag_value(tags, 'name')
        ref = _tag_value(tags, 'ref')
        forward, backward = _way_directions(tags)

        way_nodes: list[tuple[int, float, float]] = []
        for node in way.nodes:
            if not node.location.valid():
                continue
            lon = float(node.location.lon)
            lat = float(node.location.lat)
            way_nodes.append((int(node.ref), lon, lat))

        if len(way_nodes) < 2:
            return

        for (u, lon1, lat1), (v, lon2, lat2) in zip(way_nodes, way_nodes[1:]):
            if u == v:
                continue
            if not _segment_intersects_bbox(lon1, lat1, lon2, lat2, self.bbox):
                continue

            length = _segment_length_m(lon1, lat1, lon2, lat2)
            if length <= 0:
                continue

            self.nodes[u] = (lon1, lat1)
            self.nodes[v] = (lon2, lat2)
            geometry = LineString([(lon1, lat1), (lon2, lat2)])
            base_record = {
                'length': length,
                'highway': highway,
                'name': name,
                'ref': ref,
                'oneway': _tag_value(tags, 'oneway'),
                'geometry': geometry,
            }

            if forward:
                self.edge_records.append({'u': u, 'v': v, **base_record})
            if backward:
                self.edge_records.append({'u': v, 'v': u, **base_record})


def build_pandana_network(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> pdna.Network:
    '''Build a Pandana network from prepared nodes and edges.'''

    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        module='pandana.network',
        message='Unsigned integer: shortest path distance is trying to be calculated.*',
    )

    return pdna.Network(
        node_x=nodes['lon'],
        node_y=nodes['lat'],
        edge_from=edges['u'],
        edge_to=edges['v'],
        edge_weights=edges[['length']],
    )


def _prepare_network_data(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''Normalize OSM network nodes and edges for routing.'''
    nodes = nodes.copy()
    edges = edges.copy()

    nodes['id'] = pd.to_numeric(nodes['id'], errors='raise').astype('int64')
    nodes = nodes.drop_duplicates(subset='id').set_index('id', drop=False)

    edges['u'] = pd.to_numeric(edges['u'], errors='coerce')
    edges['v'] = pd.to_numeric(edges['v'], errors='coerce')
    edges['length'] = pd.to_numeric(edges['length'], errors='coerce')
    edges = edges.dropna(subset=['u', 'v', 'length']).copy()
    edges['u'] = edges['u'].astype('int64')
    edges['v'] = edges['v'].astype('int64')
    edges['length'] = edges['length'].astype('float64')

    valid_node_ids = set(nodes.index)
    edges = edges.loc[
        edges['u'].isin(valid_node_ids) & edges['v'].isin(valid_node_ids)
    ].copy()

    return nodes, edges


def _load_osm_network_data_osmium(
    pbf_path: str,
    verbose: bool = True,
    bbox: BBox | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''Load a driving network by streaming an OSM PBF with pyosmium.'''
    if osmium is None:
        raise ImportError(
            "The optional 'osmium' package is required for --network-backend osmium."
        )

    t0 = pc()
    handler = OsmiumDrivingNetworkHandler(bbox=bbox)
    handler.apply_file(str(pbf_path), locations=True)

    nodes = gpd.GeoDataFrame(
        [
            {
                'id': node_id,
                'lon': lon,
                'lat': lat,
                'geometry': Point(lon, lat),
            }
            for node_id, (lon, lat) in handler.nodes.items()
        ],
        columns=['id', 'lon', 'lat', 'geometry'],
        geometry='geometry',
        crs='EPSG:4326',
    )
    edges = gpd.GeoDataFrame(
        handler.edge_records,
        columns=['u', 'v', 'length', 'highway', 'name', 'ref', 'oneway', 'geometry'],
        geometry='geometry',
        crs='EPSG:4326',
    )

    nodes, edges = _prepare_network_data(nodes, edges)

    if verbose:
        area = 'bbox extract' if bbox is not None else 'full extract'
        print(
            f'Loaded network data with osmium ({area}) in {pc() - t0:.2f} seconds, '
            f'{len(nodes):,} nodes, {len(edges):,} edges'
        )

    return nodes, edges


def load_osm_network_data(
    pbf_path: str,
    verbose: bool = True,
    bbox: BBox | None = None,
    backend: NetworkBackend = 'pyrosm',
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''
    Load OSM driving network nodes and edges without building Pandana.
    '''
    t0 = pc()
    resolved_backend = _resolve_network_backend(backend)

    if resolved_backend == 'osmium':
        return _load_osm_network_data_osmium(
            pbf_path,
            verbose=verbose,
            bbox=bbox,
        )

    osm = OSM(str(pbf_path), bounding_box=_pyrosm_bbox(bbox))
    nodes, edges = osm.get_network(network_type='driving', nodes=True)
    nodes, edges = _prepare_network_data(nodes, edges)

    if verbose:
        area = 'bbox extract' if bbox is not None else 'full extract'
        print(
            f'Loaded network data with pyrosm ({area}) in {pc() - t0:.2f} seconds, '
            f'{len(nodes):,} nodes, {len(edges):,} edges'
        )

    return nodes, edges


def load_osm_road_edges(
    pbf_path: str,
    verbose: bool = True,
    bbox: BBox | None = None,
    backend: NetworkBackend = 'pyrosm',
) -> gpd.GeoDataFrame:
    '''
    Load OSM driving road edges for map rendering only.

    This avoids building Pandana, which is useful for large country context maps.
    '''
    t0 = pc()
    resolved_backend = _resolve_network_backend(backend)

    if resolved_backend == 'osmium':
        _, edges = _load_osm_network_data_osmium(
            pbf_path,
            verbose=verbose,
            bbox=bbox,
        )
        return edges

    osm = OSM(str(pbf_path), bounding_box=_pyrosm_bbox(bbox))
    edges = osm.get_network(network_type='driving', nodes=False).copy()

    if verbose:
        area = 'bbox extract' if bbox is not None else 'full extract'
        print(
            f'Loaded road edges for map with pyrosm ({area}) in {pc() - t0:.2f} seconds, '
            f'{len(edges):,} edges'
        )

    return edges


def load_osm_network(
    pbf_path: str,
    verbose: bool = True,
    bbox: BBox | None = None,
    backend: NetworkBackend = 'pyrosm',
) -> tuple[pdna.Network, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''
    Load OSM driving network and build Pandana network.

    Parameters
    ----------
    pbf_path
        Path to OSM PBF.
    verbose
        Whether to print timing information.

    Returns
    -------
    tuple[pdna.Network, gpd.GeoDataFrame, gpd.GeoDataFrame]
        Pandana network, nodes, edges.
    '''
    t0 = pc()
    nodes, edges = load_osm_network_data(
        pbf_path,
        verbose=False,
        bbox=bbox,
        backend=backend,
    )
    network = build_pandana_network(nodes=nodes, edges=edges)

    if verbose:
        print(
            f'Loaded network in {pc() - t0:.2f} seconds, '
            f'{len(nodes):,} nodes, {len(edges):,} edges'
        )

    return network, nodes, edges
