import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pytest
from geopandas.testing import assert_geoseries_equal
from shapely.geometry import LineString, Point

from pisa.isopolygons import OsmIsopolygonCalculator


@pytest.fixture
def nodes_gdf() -> gpd.GeoSeries:
    """The road network used for tests in this file has 4 nodes. These are three of them:"""

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
def excluded_node() -> Point:
    """Geometry of the node 5909483569, the 4th node in the road network"""
    return Point(-122.2315948, 37.768278)


@pytest.fixture
def edges_gdf() -> gpd.GeoSeries:
    """Some of the edges in the road network used for the tests in this file"""

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
def dataframe_with_lon_and_lat() -> pd.DataFrame:
    """Location (longitude and latitude) of two ficticious facilities"""

    points = [
        (-122.2314069, 37.7687054),  # closest node 19
        (-122.23124, 37.76876),  # closest node 25
    ]

    return pd.DataFrame(points, columns=["longitude", "latitude"])


class TestOsmCalculateIsopolygons:

    @pytest.fixture(autouse=True)
    def setup(self, dataframe_with_lon_and_lat):

        self.graph = ox.load_graphml(
            "tests/test_data/walk_network_4_nodes_6_edges.graphml"
        )

        self.graph_nodes, self.graph_edges = ox.graph_to_gdfs(self.graph)

        self.isopolygon_calculator = OsmIsopolygonCalculator(
            facilities_lon_lat=dataframe_with_lon_and_lat,
            distance_type="length",
            distance_values=[5, 20, 50],
            road_network=self.graph,
            node_buffer=0.00005,
            edge_buffer=0.00005,
        )

        self.isopolygons = self.isopolygon_calculator.calculate_isopolygons()

    def test_format(self):

        np.array_equal(
            self.isopolygon_calculator.nearest_nodes, [5909483625, 5909483619]
        )

        assert self.isopolygons.shape == (
            2,
            3,
        ), "The output should have two rows (one per node in nearest_nodes) and three columns (one per distance)"

        assert list(self.isopolygons.columns) == ["ID_5", "ID_20", "ID_50"]

        # Node 5909483636 is in the road_network but was kicked out in .nearest_nodes()
        assert list(self.isopolygons.index) == [5909483619, 5909483625]

    def test_nodes_in_isopolygon_5909483619_5(self):

        # The only node less than 5m away from 5909483619 is itself
        nodes_within = self.graph_nodes[
            self.graph_nodes.within(self.isopolygons.loc[5909483619, "ID_5"])
        ]
        assert set(nodes_within.index) == {
            5909483619
        }, "The only node in this isopolygon should be 5909483619"

    def test_no_edges_in_isopolygon_5909483619_5(self):

        # Since the only node less than 5m away from 5909483619 is itself, there
        # are no edges from the road network in this isopolygon

        assert not any(
            self.graph_edges.within(self.isopolygons.loc[5909483619, "ID_5"])
        ), "There should be no edges in this isopolygon"

    def test_nodes_in_isopolygon_5909483619_50(self):

        nodes_within = self.graph_nodes[
            self.graph_nodes.within(self.isopolygons.loc[5909483619, "ID_50"])
        ]

        # Node 5909483569 is farther than 50m from node 5909483619
        assert set(nodes_within.index) == {
            5909483619,
            5909483625,
            5909483636,
        }, "Nodes 5909483619, 5909483625 and 5909483636 should be in this isopolygon, but 5909483569 should not"

    def test_nodes_in_isopolygon_5909483625_50(self):

        nodes_within = self.graph_nodes[
            self.graph_nodes.within(self.isopolygons.loc[5909483625, "ID_50"])
        ]

        # Nodes 5909483636 and 5909483569 are farther than 50m away from node 5909483625
        assert set(nodes_within.index) == {
            5909483619,
            5909483625,
        }, "Nodes 5909483619 and 5909483625 should be in this isopolygon, but 5909483636 and 5909483569 should not"


class TestInflateSkeletonToIsopolygon:

    def test_excluded_node_is_left_out(self, nodes_gdf, edges_gdf, excluded_node):
        """Desired behavior: the node previously excluded because it was
        too far away is excluded from the resulting polygon"""

        poly = OsmIsopolygonCalculator._add_buffer_to_isopolygon_skeleton(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            node_buffer=0.00005,
            edge_buffer=0.00005,
        )

        assert not poly.contains(excluded_node)

    def test_excluded_node_is_back_in(self, nodes_gdf, edges_gdf, excluded_node):
        """Undesired behavior: the node previously excluded because it was
        too far away is included in the resulting polygon. The problem is
        that buffers are too large.

        These buffers are the default in gpbp/distance.py"""

        poly = OsmIsopolygonCalculator._add_buffer_to_isopolygon_skeleton(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            node_buffer=0.001,
            edge_buffer=0.0005,
        )

        assert poly.contains(excluded_node)

    def test_with_0_node_buffer(self, nodes_gdf, edges_gdf):
        """This should not be a problem because all nodes are connected"""

        poly = OsmIsopolygonCalculator._add_buffer_to_isopolygon_skeleton(
            nodes_gdf=nodes_gdf, edges_gdf=edges_gdf, node_buffer=0, edge_buffer=0.00005
        )

        assert poly.area > 0


class TestGetSkeletonNodesAndEdges:

    @pytest.fixture(autouse=True)
    def setup(self):
        """All tests use the same graph"""
        self.road_network = ox.load_graphml(
            "tests/test_data/walk_network_4_nodes_6_edges.graphml"
        )

    def test_get_correct_nodes(self, nodes_gdf):
        """Tests that the nodes are correct"""

        (actual_nodes_gdf, _) = OsmIsopolygonCalculator._get_skeleton_nodes_and_edges(
            self.road_network,
            center_node=5909483619,
            distance_value=50,
            distance_type="length",
        )

        assert_geoseries_equal(actual_nodes_gdf, nodes_gdf)

    def test_get_correct_edges(self, edges_gdf):
        """Tests that the geometry of the edges is correct"""

        # we expect this edge to be discarded as its lenght is larger than 50
        assert self.road_network.edges[(5909483619, 5909483569, 0)]["length"] > 50

        (_, actual_edges_gdf) = OsmIsopolygonCalculator._get_skeleton_nodes_and_edges(
            self.road_network,
            center_node=5909483619,
            distance_value=50,
            distance_type="length",
        )

        assert actual_edges_gdf.geom_equals_exact(edges_gdf, tolerance=4).all()

    def test_returns_center_node_if_all_others_are_too_far(self):
        """All nodes are farther than 15 meters away from node 5909483619,
        so the function returns the center node itself and no edges"""

        (actual_nodes_gdf, actual_edges_gdf) = (
            OsmIsopolygonCalculator._get_skeleton_nodes_and_edges(
                self.road_network,
                center_node=5909483619,
                distance_value=15,
                distance_type="length",
            )
        )

        assert actual_nodes_gdf.shape[0] == 1  # only the center node
        assert actual_edges_gdf.empty
