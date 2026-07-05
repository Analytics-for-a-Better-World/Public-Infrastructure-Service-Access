import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from distance_pipeline.cache import _backend_part
from distance_pipeline.network import _prepare_network_data
from distance_pipeline.pipeline_support import build_output_run_tag
from distance_pipeline.routing import build_networkx_graph, route_geometry_from_nodes
from distance_pipeline.settings import PipelineSettings
from distance_pipeline.snapping import snap_points_to_nodes
from run_pipeline import filter_edges_to_nodes, pandana_weight_columns


def test_prepare_network_data_accepts_geometry_free_edges():
    nodes = gpd.GeoDataFrame(
        {
            'id': [1, 2, 3],
            'lon': [0.0, 1.0, 2.0],
            'lat': [0.0, 1.0, 1.0],
            'geometry': [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 1.0)],
        },
        geometry='geometry',
        crs='EPSG:4326',
    )
    edges = pd.DataFrame(
        {
            'u': [1, 2, 99],
            'v': [2, 3, 1],
            'length': [10.0, 12.5, 8.0],
            'highway': pd.Categorical(['residential', 'primary', 'service']),
        }
    )

    prepared_nodes, prepared_edges = _prepare_network_data(nodes, edges)

    assert list(prepared_edges.columns) == ['u', 'v', 'length', 'highway']
    assert prepared_edges[['u', 'v', 'length']].to_dict('records') == [
        {'u': 1, 'v': 2, 'length': 10.0},
        {'u': 2, 'v': 3, 'length': 12.5},
    ]
    assert prepared_nodes.index.to_list() == [1, 2, 3]
    assert prepared_edges['u'].dtype == 'int64'
    assert prepared_edges['v'].dtype == 'int64'


def test_networkx_route_geometry_falls_back_to_node_coordinates():
    nodes = gpd.GeoDataFrame(
        {
            'id': [1, 2],
            'lon': [10.0, 11.0],
            'lat': [20.0, 21.0],
            'geometry': [Point(10.0, 20.0), Point(11.0, 21.0)],
        },
        geometry='geometry',
        crs='EPSG:4326',
    )
    edges = pd.DataFrame({'u': [1], 'v': [2], 'length': [100.0]})

    graph = build_networkx_graph(nodes, edges)
    route = route_geometry_from_nodes(graph, [1, 2])

    assert list(route.coords) == [(10.0, 20.0), (11.0, 21.0)]


def test_snap_points_to_nodes_uses_lon_lat_node_columns():
    points = gpd.GeoDataFrame(
        {
            'ID': ['a'],
            'geometry': [Point(10.01, 20.01)],
        },
        geometry='geometry',
        crs='EPSG:4326',
    )
    nodes = gpd.GeoDataFrame(
        {
            'id': [1, 2],
            'lon': [10.0, 12.0],
            'lat': [20.0, 22.0],
            'geometry': [Point(10.0, 20.0), Point(12.0, 22.0)],
        },
        geometry='geometry',
        crs='EPSG:4326',
    )

    snapped = snap_points_to_nodes(
        points,
        nodes,
        projected_epsg=3857,
        verbose=False,
    )

    assert snapped.loc['a', 'nearest_node'] == 1
    assert snapped.loc['a', 'dist_to_node'] > 0


def test_osmium_simplification_has_distinct_cache_and_output_keys():
    dense = PipelineSettings(network_backend='osmium')
    simplified = PipelineSettings(network_backend='osmium', simplify_network=True)

    assert dense.network_cache_backend() == 'osmium'
    assert simplified.network_cache_backend() == 'osmium_simplified'
    assert _backend_part('osmium') == '_backend_osmium'
    assert _backend_part('osmium_simplified') == '_backend_osmium_simplified_v2'

    dense_tag = build_output_run_tag(
        settings=dense,
        aggregate_factor=10,
        amenity_values=None,
        candidate_grid_spacing_m=None,
        candidate_max_snap_dist_m=None,
        has_candidates=False,
    )
    simplified_tag = build_output_run_tag(
        settings=simplified,
        aggregate_factor=10,
        amenity_values=None,
        candidate_grid_spacing_m=None,
        candidate_max_snap_dist_m=None,
        has_candidates=False,
    )

    assert dense_tag.endswith('_network_osmium')
    assert simplified_tag.endswith('_network_osmium-simplified')


def test_pandana_weight_columns_only_include_requested_impedance():
    assert pandana_weight_columns(PipelineSettings()) == ['length']
    assert pandana_weight_columns(
        PipelineSettings(network_impedance='travel_time_s')
    ) == ['travel_time_s']
    assert pandana_weight_columns(
        PipelineSettings(network_impedance='calibrated_time_s')
    ) == ['calibrated_time_s']


def test_filter_edges_to_nodes_keeps_only_selected_component_edges():
    nodes = pd.DataFrame(
        {
            'lon': [0.0, 1.0],
            'lat': [0.0, 1.0],
        },
        index=pd.Index([10, 11], name='id'),
    )
    edges = pd.DataFrame(
        {
            'u': [10, 11, 12, 10],
            'v': [11, 10, 13, 99],
            'length': [1.0, 1.0, 2.0, 3.0],
        }
    )

    filtered = filter_edges_to_nodes(edges, nodes, verbose=False)

    assert filtered[['u', 'v', 'length']].to_dict('records') == [
        {'u': 10, 'v': 11, 'length': 1.0},
        {'u': 11, 'v': 10, 'length': 1.0},
    ]

