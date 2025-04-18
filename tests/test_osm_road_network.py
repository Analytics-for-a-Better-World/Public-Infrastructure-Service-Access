import random

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import Polygon

from pisa.osm_road_network import OsmRoadNetwork


@pytest.fixture
def mock_geometry():
    """Mock geometry for testing purposes."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def mock_graph():
    # Load network
    G = ox.load_graphml("tests/test_data/drive_network_MAIN.graphml")
    # Choose a random node and create subgraph
    random.seed(43)
    ego_node = random.choice(list(G.nodes))
    subgraph = nx.ego_graph(G, ego_node, radius=2)
    return subgraph


@pytest.mark.parametrize(
    ["mode_of_transport", "distance_type", "fallback_speed"],
    [
        ["driving", "length", None],
        ["walking", "length", None],
        ["cycling", "length", None],
        ["driving", "travel_time", 15],
        ["walking", "travel_time", 15],
        ["cycling", "travel_time", 15],
    ],
)
def test_osm_road_network(mock_geometry, mode_of_transport, distance_type, fallback_speed, mocker, mock_graph):
    mocker.patch("pisa.osm_road_network.ox.graph_from_polygon", return_value=mock_graph)

    road_network = OsmRoadNetwork(
        admin_area_boundaries=mock_geometry,
        mode_of_transport=mode_of_transport,
        distance_type=distance_type,
        fallback_speed=fallback_speed,
    ).get_osm_road_network()

    if distance_type == "travel_time":
        for _, _, data in road_network.edges(data=True):
            assert data["speed_kph"] == fallback_speed
            expected_travel_time = data["length"] / (data["speed_kph"] * 1000 / 60)  # length in meters, speed in kph
            assert round(data["travel_time"], 2) == round(expected_travel_time, 2)
            assert round(data["travel_time"], 2) == round(expected_travel_time, 2)
            assert round(data["travel_time"], 2) == round(expected_travel_time, 2)
