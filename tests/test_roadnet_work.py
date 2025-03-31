import osmnx as ox
import pytest
from pisa.roadnetwork import get_road_network
import networkx as nx
import random
from pisa.administrative_area import AdministrativeArea

@pytest.fixture
def mock_graph():
    # Load network
    G = ox.load_graphml('tests/test_data/drive_network_MAIN.graphml')

    # Fix a seed to get reproducible results
    random.seed(43)

    # choose a random node
    ego_node = random.choice(list(G.nodes))

    # Get a subgraph
    subgraph = nx.ego_graph(G, ego_node, radius=2)

    return subgraph

@pytest.fixture
def define_admin_area():
    adm_area = AdministrativeArea("Timor-Leste", admin_level=1)
    return adm_area

@pytest.mark.parametrize(["network_type", "default_speed"], [["driving", 50], ["walking", 4], ["cycling", 15]])
def test_get_road_network(mocker, adm_area, mock_graph, network_type, default_speed):
    mocker.patch("gpbp.layers.ox.graph_from_polygon", return_value=mock_graph)
    
    road_network = get_road_network(adm_area, network_type=network_type)

    for _, _, data in adm_area.road_network.edges(data=True):
        assert data["speed_kph"] == default_speed
        expected_travel_time = data["length"] / (data["speed_kph"] * 1000 / 60)  # length in meters, speed in kph
        assert round(data["travel_time"], 2) == round(expected_travel_time, 2)