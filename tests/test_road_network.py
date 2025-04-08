import random

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import Polygon

from pisa.administrative_area import AdministrativeArea
from pisa.road_network import get_road_network


def get_shape_data_by_country(admin_level: int) -> gpd.GeoDataFrame:
    if admin_level == 0:
        data = {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}
    else:
        data = {
            f"NAME_{admin_level}": ["AreaA", "AreaB"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
        }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# Automatically patch _download_admin_areas to use the dummy downloader.
@pytest.fixture(autouse=True)
def patch_download(mocker):
    mocker.patch.object(
        AdministrativeArea,
        "_download_admin_areas",
        lambda self, country, admin_level: get_shape_data_by_country(admin_level),
    )


@pytest.fixture
def mock_graph():
    # Load network
    G = ox.load_graphml("tests/test_data/drive_network_MAIN.graphml")

    # Fix a seed to get reproducible results
    random.seed(43)

    # choose a random node
    ego_node = random.choice(list(G.nodes))

    # Get a subgraph
    subgraph = nx.ego_graph(G, ego_node, radius=2)

    return subgraph


@pytest.fixture
def adm_area():
    adm_area = AdministrativeArea("Timor-Leste", admin_level=1)
    adm_area.geometry = adm_area.get_admin_area_boundaries("AreaA")
    return adm_area


@pytest.mark.parametrize(
    ["network_type", "default_speed"], [["driving", 50], ["walking", 4], ["cycling", 15]]
)
def test_get_road_network(mocker, adm_area, mock_graph, network_type, default_speed):
    mocker.patch("gpbp.layers.ox.graph_from_polygon", return_value=mock_graph)

    road_network = get_road_network(adm_area, network_type=network_type)

    for _, _, data in road_network.edges(data=True):
        assert data["speed_kph"] == default_speed
        expected_travel_time = data["length"] / (
            data["speed_kph"] * 1000 / 60
        )  # length in meters, speed in kph
        assert round(data["travel_time"], 2) == round(expected_travel_time, 2)
