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
from shapely.geometry import LineString


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

    try:
        from importlib import import_module

        cyaccess_module = import_module('pandana.cyaccess')
        getattr(cyaccess_module, 'cyaccess')
    except Exception as exc:  # pragma: no cover - depends on binary wheel build
        import_error = f' Local import check failed with {type(exc).__name__}: {exc}'
    else:
        return

    message = (
        'WARNING: Pandana compatibility risk detected. '
        f'Installed pandana=={pandana_version} is likely to fail with '
        f'numpy=={numpy_version}. Pandana 0.7 wheels were built against the '
        'NumPy 1.x C API, so importing or using Pandana under NumPy 2 can raise '
        'binary-compatibility errors. Use numpy<2, or install a Pandana build '
        'that explicitly supports NumPy 2, before running this distance pipeline.'
        f'{import_error}'
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


class OsmiumDrivingNodeUseCounter(osmium.SimpleHandler if osmium is not None else object):
    """First-pass counter for identifying drivable-way split nodes."""

    def __init__(self) -> None:
        if osmium is None:
            raise ImportError(
                "The optional 'osmium' package is required for the osmium backend."
            )
        super().__init__()
        self.node_use_counts: dict[int, int] = {}

    def way(self, way: object) -> None:
        if not _is_drivable_way(way.tags):
            return

        for node in way.nodes:
            node_id = int(node.ref)
            count = self.node_use_counts.get(node_id, 0)
            if count < 2:
                self.node_use_counts[node_id] = count + 1

    def split_node_refs(self) -> set[int]:
        return {
            node_id
            for node_id, count in self.node_use_counts.items()
            if count > 1
        }


class OsmiumDrivingNetworkHandler(osmium.SimpleHandler if osmium is not None else object):
    '''Streaming OSM PBF handler for a pyrosm-like driving graph.'''

    def __init__(
        self,
        bbox: BBox | None = None,
        *,
        collect_nodes: bool = True,
        include_geometry: bool = False,
        directed: bool = True,
        split_node_refs: set[int] | None = None,
    ) -> None:
        if osmium is None:
            raise ImportError(
                "The optional 'osmium' package is required for the osmium backend."
            )
        super().__init__()
        self.bbox = bbox
        self.collect_nodes = collect_nodes
        self.include_geometry = include_geometry
        self.directed = directed
        self.split_node_refs = split_node_refs
        self.nodes: dict[int, tuple[float, float]] = {}
        self.edge_u: list[int] = []
        self.edge_v: list[int] = []
        self.edge_length: list[float] = []
        self.edge_highway: list[str | None] = []
        self.edge_geometry: list[LineString] | None = [] if include_geometry else None

    def way(self, way: object) -> None:
        tags = way.tags
        if not _is_drivable_way(tags):
            return

        highway = _tag_value(tags, 'highway')
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

        if self.split_node_refs is not None:
            self._append_simplified_way(way_nodes, highway, forward, backward)
            return

        for (u, lon1, lat1), (v, lon2, lat2) in zip(way_nodes, way_nodes[1:]):
            if u == v:
                continue
            if not _segment_intersects_bbox(lon1, lat1, lon2, lat2, self.bbox):
                continue

            length = _segment_length_m(lon1, lat1, lon2, lat2)
            if length <= 0:
                continue

            self._append_edge_pair(
                u,
                v,
                length,
                highway,
                lon1,
                lat1,
                lon2,
                lat2,
                forward,
                backward,
            )

    def _append_simplified_way(
        self,
        way_nodes: list[tuple[int, float, float]],
        highway: str | None,
        forward: bool,
        backward: bool,
    ) -> None:
        """Collapse non-intersection shape nodes into longer routing edges."""
        assert self.split_node_refs is not None

        start_id, start_lon, start_lat = way_nodes[0]
        prev_id, prev_lon, prev_lat = way_nodes[0]
        accumulated_length = 0.0
        has_included_segment = False
        last_index = len(way_nodes) - 1

        for idx, (node_id, lon, lat) in enumerate(way_nodes[1:], start=1):
            if node_id == prev_id:
                prev_id, prev_lon, prev_lat = node_id, lon, lat
                continue

            length = _segment_length_m(prev_lon, prev_lat, lon, lat)
            if length <= 0:
                prev_id, prev_lon, prev_lat = node_id, lon, lat
                continue

            if not _segment_intersects_bbox(prev_lon, prev_lat, lon, lat, self.bbox):
                start_id, start_lon, start_lat = node_id, lon, lat
                prev_id, prev_lon, prev_lat = node_id, lon, lat
                accumulated_length = 0.0
                has_included_segment = False
                continue

            accumulated_length += length
            has_included_segment = True

            is_split_node = idx == last_index or node_id in self.split_node_refs
            if is_split_node:
                if start_id != node_id and has_included_segment:
                    self._append_edge_pair(
                        start_id,
                        node_id,
                        accumulated_length,
                        highway,
                        start_lon,
                        start_lat,
                        lon,
                        lat,
                        forward,
                        backward,
                    )
                start_id, start_lon, start_lat = node_id, lon, lat
                accumulated_length = 0.0
                has_included_segment = False

            prev_id, prev_lon, prev_lat = node_id, lon, lat

    def _append_edge_pair(
        self,
        u: int,
        v: int,
        length: float,
        highway: str | None,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
        forward: bool,
        backward: bool,
    ) -> None:
        if self.collect_nodes:
            self.nodes[u] = (lon1, lat1)
            self.nodes[v] = (lon2, lat2)

        if self.directed:
            if forward:
                self._append_edge(u, v, length, highway, lon1, lat1, lon2, lat2)
            if backward:
                self._append_edge(v, u, length, highway, lon2, lat2, lon1, lat1)
        else:
            self._append_edge(u, v, length, highway, lon1, lat1, lon2, lat2)

    def _append_edge(
        self,
        u: int,
        v: int,
        length: float,
        highway: str | None,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> None:
        self.edge_u.append(u)
        self.edge_v.append(v)
        self.edge_length.append(length)
        self.edge_highway.append(highway)
        if self.edge_geometry is not None:
            self.edge_geometry.append(LineString([(lon1, lat1), (lon2, lat2)]))

    def nodes_frame(self) -> gpd.GeoDataFrame:
        node_ids: list[int] = []
        lons: list[float] = []
        lats: list[float] = []
        for node_id, (lon, lat) in self.nodes.items():
            node_ids.append(node_id)
            lons.append(lon)
            lats.append(lat)

        frame = pd.DataFrame({'id': node_ids, 'lon': lons, 'lat': lats})
        return gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame['lon'], frame['lat']),
            crs='EPSG:4326',
        )

    def edges_frame(self) -> pd.DataFrame | gpd.GeoDataFrame:
        frame = pd.DataFrame(
            {
                'u': self.edge_u,
                'v': self.edge_v,
                'length': self.edge_length,
                'highway': pd.Categorical(self.edge_highway),
            }
        )
        if self.edge_geometry is None:
            return frame
        return gpd.GeoDataFrame(
            frame,
            geometry=gpd.GeoSeries(self.edge_geometry, crs='EPSG:4326'),
            crs='EPSG:4326',
        )


def build_pandana_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
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
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    simplify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Load a driving network by streaming an OSM PBF with pyosmium.'''
    if osmium is None:
        raise ImportError(
            "The optional 'osmium' package is required for --network-backend osmium."
        )

    t0 = pc()
    split_node_refs = None
    if simplify:
        t_count = pc()
        counter = OsmiumDrivingNodeUseCounter()
        counter.apply_file(str(pbf_path), locations=False)
        split_node_refs = counter.split_node_refs()
        referenced_node_count = len(counter.node_use_counts)
        del counter

        if verbose:
            print(
                f'Counted drivable OSM node uses with osmium in {pc() - t_count:.2f} seconds, '
                f'{referenced_node_count:,} referenced nodes, {len(split_node_refs):,} split nodes'
            )

    handler = OsmiumDrivingNetworkHandler(
        bbox=bbox,
        collect_nodes=True,
        include_geometry=False,
        directed=simplify,
        split_node_refs=split_node_refs,
    )
    handler.apply_file(str(pbf_path), locations=True)
    if split_node_refs is not None:
        del split_node_refs

    nodes = handler.nodes_frame()
    edges = handler.edges_frame()

    nodes, edges = _prepare_network_data(nodes, edges)

    if verbose:
        area = 'bbox extract' if bbox is not None else 'full extract'
        mode = 'simplified' if simplify else 'unsimplified'
        print(
            f'Loaded {mode} network data with osmium ({area}) in {pc() - t0:.2f} seconds, '
            f'{len(nodes):,} nodes, {len(edges):,} edges'
        )

    return nodes, edges


def _load_osm_road_edges_osmium(
    pbf_path: str,
    verbose: bool = True,
    bbox: BBox | None = None,
) -> gpd.GeoDataFrame:
    '''Load one geometry-bearing road segment per OSM segment for map rendering.'''
    if osmium is None:
        raise ImportError(
            "The optional 'osmium' package is required for --network-backend osmium."
        )

    t0 = pc()
    handler = OsmiumDrivingNetworkHandler(
        bbox=bbox,
        collect_nodes=False,
        include_geometry=True,
        directed=False,
    )
    handler.apply_file(str(pbf_path), locations=True)
    edges = handler.edges_frame()

    if verbose:
        area = 'bbox extract' if bbox is not None else 'full extract'
        print(
            f'Loaded road edges for map with osmium ({area}) in {pc() - t0:.2f} seconds, '
            f'{len(edges):,} edges'
        )

    return edges


def load_osm_network_data(
    pbf_path: str,
    verbose: bool = True,
    bbox: BBox | None = None,
    backend: NetworkBackend = 'pyrosm',
    simplify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            simplify=simplify,
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
        return _load_osm_road_edges_osmium(
            pbf_path,
            verbose=verbose,
            bbox=bbox,
        )

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
    simplify: bool = False,
) -> tuple[pdna.Network, pd.DataFrame, pd.DataFrame]:
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
        simplify=simplify,
    )
    network = build_pandana_network(nodes=nodes, edges=edges)

    if verbose:
        print(
            f'Loaded network in {pc() - t0:.2f} seconds, '
            f'{len(nodes):,} nodes, {len(edges):,} edges'
        )

    return network, nodes, edges
