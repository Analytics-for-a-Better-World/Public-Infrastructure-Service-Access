import geopandas as gpd
import osmnx as ox
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas import testing as tm
from shapely.geometry import LineString, Point

from pisa.isopolygons import OsmIsopolygonCalculator


@pytest.fixture
def nodes_gdf() -> gpd.GeoSeries:

    data = {
        "osmid": [5909483625, 5909483619, 5909483636],
        "geometry": [
            Point(-122.231243, 37.7687576),
            Point(-122.2314069, 37.7687054),
            Point(-122.2317839, 37.7689584),
        ],
    }
    return gpd.GeoSeries(data["geometry"], index=data["osmid"], crs="EPSG:4326")


@pytest.fixture
def nodes_geodataframe() -> gpd.GeoDataFrame:
    data = {
        "id": [5909483625, 5909483619, 5909483636],
        "geometry": [
            Point(-122.231243, 37.7687576),
            Point(-122.2314069, 37.7687054),
            Point(-122.2317839, 37.7689584),
        ],
    }
    return gpd.GeoDataFrame(data).set_index("id")


@pytest.fixture
def edges_gdf() -> gpd.GeoSeries:
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
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def excluded_node() -> Point:
    """Geometry of the node 5909483569"""
    return Point(-122.2315948, 37.768278)


@pytest.fixture
def dataframe_with_lat_and_lon() -> pd.DataFrame:

    points = [
        (-122.2314069, 37.7687054),  # closest node 19
        (-122.23124, 37.76876),  # closest node 25
    ]

    return pd.DataFrame(points, columns=["longitude", "latitude"])


# class TestOsmCalculateIsopolygons:

#     @pytest.fixture(autouse=True)
#     def setup(
#         self,
#         dataframe_with_lat_and_lon,
#     ):
#         """
#         Here I'm deliberately choosing to override the defaults for node_buff and edge_buff
#         so the tests pass, otherwise the tests would fail because the buffers are too large
#         """

#         self.isopolygons = OsmIsopolygonCalculator(
#             facilities_lat_lon=dataframe_with_lat_and_lon,
#             distance_type="length",
#             distance_values=[5, 20, 50],
#             road_network=ox.load_graphml(
#                 "tests/test_data/walk_network_4_nodes_6_edges.graphml"
#             ),
#             node_buff=0.00005,
#             edge_buff=0.00005,
#         ).calculate_isopolygons()

#     def test_format(self):
#         assert self.isopolygons.shape == (
#             2,
#             3,
#         ), "The output should have two rows (one per point in road_nodes) and three columns (one per distance)"

#         assert list(self.isopolygons.columns) == ["ID_5", "ID_20", "ID_50"]

#         # Node 5909483636 was kicked out in .nearest_nodes()
#         assert list(self.isopolygons.index) == [5909483619, 5909483625]


class TestCreatePolygonFromNodesAndEdges:

    def test_excluded_node_is_left_out(self, nodes_gdf, edges_gdf, excluded_node):
        """Desired behavior: the node previously excluded because it was
        too far away is excluded from the resulting polygon"""

        poly = OsmIsopolygonCalculator.create_polygon_from_nodes_and_edges(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            node_buff=0.00005,
            edge_buff=0.00005,
        )

        assert not poly.contains(excluded_node)

    def test_excluded_node_is_back_in(self, nodes_gdf, edges_gdf, excluded_node):
        """Undesired behavior: the node previously excluded because it was
        too far away is included in the resulting polygon. The problem is
        that buffers are too large.

        These buffers are the default in gpbp/distance.py"""

        poly = OsmIsopolygonCalculator.create_polygon_from_nodes_and_edges(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            node_buff=0.001,
            edge_buff=0.0005,
        )

        assert poly.contains(excluded_node)

    def test_with_0_node_buffer(self, nodes_gdf, edges_gdf):
        """This should not be a problem because all nodes are connected"""

        poly = OsmIsopolygonCalculator.create_polygon_from_nodes_and_edges(
            nodes_gdf=nodes_gdf, edges_gdf=edges_gdf, node_buff=0, edge_buff=0.00005
        )

        assert poly.area > 0


class TestGetPolyNxNew:

    @pytest.fixture(autouse=True)
    def setup(self):
        """All tests use the same graph"""
        self.road_network = ox.load_graphml(
            "tests/test_data/walk_network_4_nodes_6_edges.graphml"
        )

    def test_get_poly_nx_nodes(self, nodes_gdf):
        """Tests that the nodes are correct"""

        (actual_nodes_gdf, _) = OsmIsopolygonCalculator._get_poly_nx_new(
            self.road_network,
            center_node=5909483619,
            dist_value=50,
            distance_type="length",
        )

        assert_geoseries_equal(actual_nodes_gdf, nodes_gdf)

    def test_get_poly_nx_edges(self, edges_gdf):
        """Tests that the geometry of the edges is correct"""

        # we expect this edge to be discarded as its lenght is larger than 50
        assert self.road_network.edges[(5909483619, 5909483569, 0)]["length"] > 50

        (_, actual_edges_gdf) = OsmIsopolygonCalculator._get_poly_nx_new(
            self.road_network,
            center_node=5909483619,
            dist_value=50,
            distance_type="length",
        )

        assert actual_edges_gdf.geom_equals_exact(edges_gdf, tolerance=4).all()

    def test_raises_value_error_if_all_nodes_are_too_far(self):

        # TODO: this should actually just return the node itself

        """All nodes are farther than 15 meters away from node 5909483619"""
        with pytest.raises(ValueError, match="Graph contains no edges"):
            OsmIsopolygonCalculator._get_poly_nx_new(
                self.road_network,
                center_node=5909483619,
                dist_value=15,
                distance_type="length",
            )


class TestGetPolyNxOld:

    @pytest.fixture(autouse=True)
    def setup(self):
        """All tests use the same graph"""
        self.road_network = ox.load_graphml(
            "tests/test_data/walk_network_4_nodes_6_edges.graphml"
        )

    def test_get_poly_nx_nodes(self, nodes_geodataframe):
        """Tests that the nodes are correct"""

        (actual_nodes_gdf, _) = OsmIsopolygonCalculator._get_poly_nx(
            self.road_network,
            center_node=5909483619,
            dist_value=50,
            distance_type="length",
        )

        assert_geodataframe_equal(actual_nodes_gdf, nodes_geodataframe)

    def test_get_poly_nx_edges(self, edges_gdf):
        """Tests that the geometry of the edges is correct"""

        # we expect this edge to be discarded as its lenght is larger than 50
        assert self.road_network.edges[(5909483619, 5909483569, 0)]["length"] > 50

        (_, actual_edges_gdf) = OsmIsopolygonCalculator._get_poly_nx(
            self.road_network,
            center_node=5909483619,
            dist_value=50,
            distance_type="length",
        )

        assert actual_edges_gdf.geom_equals_exact(edges_gdf, tolerance=4).all()

    def test_returns_center_node_if_all_others_are_too_far(self):
        """All nodes are farther than 15 meters away from node 5909483619,
        so the function returns the center node itself and no edges"""

        (actual_nodes_gdf, actual_edges_gdf) = OsmIsopolygonCalculator._get_poly_nx(
            self.road_network,
            center_node=5909483619,
            dist_value=15,
            distance_type="length",
        )

        assert actual_nodes_gdf.shape[0] == 1  # only the center node
        assert actual_edges_gdf.empty
