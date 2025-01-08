import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from gpbp.distance import _get_poly_nx


@pytest.mark.parametrize(
    "load_graphml_file",
    ["tests/test_data/walk_network_4_nodes_6_edges.graphml"],
    indirect=True,
)
def test_get_poly_nx_nodes(load_graphml_file):

    G = load_graphml_file

    assert isinstance(G, nx.MultiDiGraph)

    (actual_nodes_gdf, _) = _get_poly_nx(
        G, road_node=5909483619, dist_value=50, distance_type="length"
    )

    expected_nodes_gdf_data = {
        "id": [5909483619, 5909483625, 5909483636],
        "geometry": [
            Point(-122.2314069, 37.7687054),
            Point(-122.231243, 37.7687576),
            Point(-122.2317839, 37.7689584),
        ],
    }

    expected_nodes_gdf = gpd.GeoDataFrame(
        expected_nodes_gdf_data,
    ).set_index("id")

    # Geopandas 1.0.1 has a function to compare dataframes,
    # gpd.testing.assert_geodataframe_equal(actual_nodes_gdf, expected_nodes_gdf)
    # Here we use a workaround as we're behind on geopandas version

    # assert that nodes (index) are equal (though not necessarily in the same order)
    assert set(actual_nodes_gdf.index) == set(expected_nodes_gdf.index)

    # assert geometry almost equal
    assert actual_nodes_gdf.geom_almost_equals(expected_nodes_gdf, decimal=5).all()


@pytest.mark.parametrize(
    "load_graphml_file",
    ["tests/test_data/walk_network_4_nodes_6_edges.graphml"],
    indirect=True,
)
def test_get_poly_nx_edges(load_graphml_file):

    G = load_graphml_file

    # we expect this edge to be discarded as its lenght is larger than 50
    assert G.edges[(5909483619, 5909483569, 0)]["length"] > 50

    (_, actual_edges_gdf) = _get_poly_nx(
        G, road_node=5909483619, dist_value=50, distance_type="length"
    )

    coordinates_25_to_19 = [(-122.23124, 37.76876), (-122.23141, 37.76871)]

    coordinates_19_to_36 = [
        (-122.2314069, 37.7687054),
        (-122.2314797, 37.7687656),
        (-122.2315618, 37.7688239),
        (-122.2316698, 37.7688952),
        (-122.2317839, 37.7689584),
    ]

    expected_edges_gdf = gpd.GeoSeries(
        [
            LineString(coordinates_25_to_19),  # edge 5909483625 -> 5909483619
            LineString(coordinates_25_to_19[::-1]),  # edge 5909483619 -> 5909483625
            LineString(coordinates_19_to_36),  # edge 5909483619 -> 5909483636
            LineString(coordinates_19_to_36[::-1]),  # edge 5909483636 -> 5909483619
        ]
    )

    # Use assert_geoseries_equal after update to geopandas 1.0.1

    assert actual_edges_gdf.geom_almost_equals(expected_edges_gdf, decimal=2).all()
