import numpy as np
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import Point

from pisa_abw.isopolygons import OsmIsopolygonCalculatorAlternative


@pytest.fixture
def dataframe_with_lon_and_lat() -> pd.DataFrame:
    """Location (longitude and latitude) of two ficticious facilities"""

    points = [
        (-122.2314069, 37.7687054),  # closest node 5909483619
        (-122.23124, 37.76876),  # closest node 5909483625
    ]
    osmids = [5909483619, 5909483625]

    return pd.DataFrame(points, columns=["longitude", "latitude"], index=osmids)


class TestOsmCalculateIsopolygonsAlternative:
    @pytest.fixture(autouse=True)
    def setup(self, dataframe_with_lon_and_lat):
        self.graph = ox.load_graphml("tests/test_data/walk_network_4_nodes_6_edges.graphml")

        self.graph_nodes, self.graph_edges = ox.graph_to_gdfs(self.graph)

        self.isopolygon_calculator = OsmIsopolygonCalculatorAlternative(
            facilities_df=dataframe_with_lon_and_lat,
            distance_type="length",
            distance_values=[5, 20, 50],
            road_network=self.graph,
            buffer=15,
        )

        self.isopolygons = self.isopolygon_calculator.calculate_isopolygons()

    def test_format(self):
        np.array_equal(
            self.isopolygon_calculator.nearest_nodes_dict.keys(),
            [5909483625, 5909483619],
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
        nodes_within = self.graph_nodes[self.graph_nodes.within(self.isopolygons.loc[5909483619, "ID_5"])]
        assert set(nodes_within.index) == {5909483619}, "The only node in this isopolygon should be 5909483619"

    def test_no_edges_in_isopolygon_5909483619_5(self):
        # Since the only node less than 5m away from 5909483619 is itself, there
        # are no edges from the road network in this isopolygon

        assert not any(self.graph_edges.within(self.isopolygons.loc[5909483619, "ID_5"])), (
            "There should be no edges in this isopolygon"
        )

    def test_nodes_in_isopolygon_5909483619_50(self):
        nodes_within = self.graph_nodes[self.graph_nodes.within(self.isopolygons.loc[5909483619, "ID_50"])]

        # Node 5909483569 is farther than 50m from node 5909483619
        assert set(nodes_within.index) == {
            5909483619,
            5909483625,
            5909483636,
        }, "Nodes 5909483619, 5909483625 and 5909483636 should be in this isopolygon, but 5909483569 should not"

    def test_nodes_in_isopolygon_5909483625_50(self):
        nodes_within = self.graph_nodes[self.graph_nodes.within(self.isopolygons.loc[5909483625, "ID_50"])]

        # Nodes 5909483636 and 5909483569 are farther than 50m away from node 5909483625
        assert set(nodes_within.index) == {
            5909483619,
            5909483625,
        }, "Nodes 5909483619 and 5909483625 should be in this isopolygon, but 5909483636 and 5909483569 should not"


class TestGetSkeletonNodesAndEdgesAlternative:
    @pytest.fixture(autouse=True)
    def setup(self):
        """All tests use the same graph"""
        self.road_network = ox.load_graphml("tests/test_data/walk_network_4_nodes_6_edges.graphml")

        self.road_nodes, self.road_edges = ox.graph_to_gdfs(self.road_network)

    def test_single_node(self):
        """Tests that the skeleton is a single shapely Point corresponding
        to the geometry of the node 5909483619"""

        skeleton = OsmIsopolygonCalculatorAlternative._get_skeleton_nodes_and_edges(
            self.road_network,
            center_node=5909483619,
            distance_value=5,
            distance_type="length",
        )

        assert isinstance(skeleton, Point), "The skeleton should be a single point"
        assert skeleton.within(self.road_nodes.loc[5909483619, "geometry"])

    def test_nodes_and_edges(self):
        skeleton = OsmIsopolygonCalculatorAlternative._get_skeleton_nodes_and_edges(
            self.road_network,
            center_node=5909483619,
            distance_value=50,
            distance_type="length",
        )

        assert self.road_edges.loc[(5909483625, 5909483619, 0)].geometry.within(skeleton), (
            "Edge (5909483625, 5909483619) should be within the skeleton"
        )

        assert self.road_edges.loc[(5909483636, 5909483619, 0)].geometry.within(skeleton), (
            "Edge (5909483636, 5909483619) should be within the skeleton"
        )

        assert not self.road_edges.loc[(5909483619, 5909483569, 0)].geometry.within(
            skeleton
        ), """Edge (5909483619, 5909483569) should not be within the skeleton because 
        node 5909483569 is farther than 50m from node 5909483619"""
