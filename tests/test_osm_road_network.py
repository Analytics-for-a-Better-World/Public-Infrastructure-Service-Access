import random

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import Polygon

from pisa.administrative_area import AdministrativeArea
from pisa.constants import (
    DEFAULT_FALLBACK_CYCLING_SPEED,
    DEFAULT_FALLBACK_DRIVING_SPEED,
    DEFAULT_FALLBACK_WALKING_SPEED,
)
from pisa.osm_road_network import OsmRoadNetwork


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
def test_osm_road_network(adm_area, mode_of_transport, distance_type, fallback_speed, mocker, mock_graph):
    mocker.patch("gpbp.layers.ox.graph_from_polygon", return_value=mock_graph)

    road_network = OsmRoadNetwork(
        admin_area_boundaries=adm_area.geometry,
        mode_of_transport=mode_of_transport,
        distance_type=distance_type,
        fallback_speed=fallback_speed,
    ).get_osm_road_network()

    if fallback_speed is not None:
        default_speed = fallback_speed
    else:
        if mode_of_transport == "driving":
            default_speed = DEFAULT_FALLBACK_DRIVING_SPEED
        elif mode_of_transport == "walking":
            default_speed = DEFAULT_FALLBACK_WALKING_SPEED
        elif mode_of_transport == "cycling":
            default_speed = DEFAULT_FALLBACK_CYCLING_SPEED

    if distance_type == "travel_time":
        for _, _, data in road_network.edges(data=True):
            assert data["speed_kph"] == default_speed
            expected_travel_time = data["length"] / (data["speed_kph"] * 1000 / 60)  # length in meters, speed in kph
            assert round(data["travel_time"], 2) == round(expected_travel_time, 2)
            assert data["speed_kph"] == default_speed
            expected_travel_time = data["length"] / (data["speed_kph"] * 1000 / 60)  # length in meters, speed in kph
            assert round(data["travel_time"], 2) == round(expected_travel_time, 2)
