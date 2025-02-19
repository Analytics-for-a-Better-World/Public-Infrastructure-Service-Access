import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import pytest

from pisa.isopolygons import OsmIsopolygonCalculator

# I need a facilities_lat_lon gdf and a road_network


@pytest.fixture
def dataframe_with_lat_and_lon() -> pd.DataFrame:

    points = [
        (-122.2314069, 37.7687054),  # closest node 19
        (-122.23124, 37.76876),  # closest node 25
    ]

    return pd.DataFrame(points, columns=["longitude", "latitude"])


def test_calculate_isopolygons():
    
    isopolygon_calculator = OsmIsopolygonCalculator(
        facilities_lat_lon=dataframe_with_lat_and_lon,
        distance_type="length",
        distance_values=[10, 20, 50],
        road_network=ox.load_graphml(
            "tests/test_data/walk_network_4_nodes_6_edges.graphml"
        ),
    )


