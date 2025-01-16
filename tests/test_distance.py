import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from gpbp.distance import _get_poly_nx

@pytest.fixture
def nodes_gdf():
    nodes_gdf_data = {
        "id": [5909483619, 5909483625, 5909483636],
        "geometry": [
            Point(-122.2314069, 37.7687054),
            Point(-122.231243, 37.7687576),
            Point(-122.2317839, 37.7689584),
        ],
    }

    return gpd.GeoDataFrame(
        nodes_gdf_data,
    ).set_index("id")

@pytest.fixture
def edges_gdf():
    coordinates_25_to_19 = [(-122.23124, 37.76876), (-122.23141, 37.76871)]

    coordinates_19_to_36 = [
        (-122.2314069, 37.7687054),
        (-122.2314797, 37.7687656),
        (-122.2315618, 37.7688239),
        (-122.2316698, 37.7688952),
        (-122.2317839, 37.7689584),
    ]

    return gpd.GeoSeries(
        [
            LineString(coordinates_25_to_19),  # edge 5909483625 -> 5909483619
            LineString(coordinates_25_to_19[::-1]),  # edge 5909483619 -> 5909483625
            LineString(coordinates_19_to_36),  # edge 5909483619 -> 5909483636
            LineString(coordinates_19_to_36[::-1]),  # edge 5909483636 -> 5909483619
        ]
    )


class TestGetPolyNx:

    @pytest.fixture(autouse=True)
    def setup(self, load_graphml_file):
        """Both tests use the same graph"""
        self.G = load_graphml_file


    @pytest.mark.parametrize(
        "load_graphml_file",
        ["tests/test_data/walk_network_4_nodes_6_edges.graphml"],
        indirect=True,
    )
    def test_get_poly_nx_nodes(self, nodes_gdf):
        """Tests that the nodes are correct"""

        (actual_nodes_gdf, _) = _get_poly_nx(
            self.G, center_node=5909483619, dist_value=50, distance_type="length"
        )

        # TODO: gpd.testing.assert_geodataframe_equal(actual_nodes_gdf, expected_nodes_gdf)
        # after updating to Geopandas 1.0.1

        # assert that nodes (index) are equal (though not necessarily in the same order)
        assert set(actual_nodes_gdf.index) == set(nodes_gdf.index)

        assert actual_nodes_gdf.geom_almost_equals(nodes_gdf, decimal=4).all()


    @pytest.mark.parametrize(
        "load_graphml_file",
        ["tests/test_data/walk_network_4_nodes_6_edges.graphml"],
        indirect=True,
    )
    def test_get_poly_nx_edges(self, edges_gdf):
        """Tests that the geometry of the edges is correct"""

        # we expect this edge to be discarded as its lenght is larger than 50
        assert self.G.edges[(5909483619, 5909483569, 0)]["length"] > 50

        (_, actual_edges_gdf) = _get_poly_nx(
            self.G, center_node=5909483619, dist_value=50, distance_type="length"
        )

        # TODO: use assert_geoseries_equal after update to geopandas 1.0.1
        assert actual_edges_gdf.geom_almost_equals(edges_gdf, decimal=4).all()
