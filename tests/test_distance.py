import geopandas as gpd
import osmnx as ox
import pandas as pd
import pytest
from geopandas.testing import assert_geoseries_equal
from shapely.geometry import LineString, Point, Polygon

from gpbp.distance import _get_poly_nx, calculate_isopolygons_graph


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
def dataframe_with_lat_and_lon() -> pd.DataFrame:

    points = [
        (-122.2314069, 37.7687054),  # closest node 19
        (-122.23124, 37.76876),  # closest node 25
    ]

    return pd.DataFrame(points, columns=["longitude", "latitude"])


class TestGetPolyNx:

    @pytest.fixture(autouse=True)
    def setup(self):
        """All tests use the same graph"""
        self.road_network = ox.load_graphml(
            "tests/test_data/walk_network_4_nodes_6_edges.graphml"
        )

    def test_get_poly_nx_nodes(self, nodes_gdf):
        """Tests that the nodes are correct"""

        (actual_nodes_gdf, _) = _get_poly_nx(
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

        (_, actual_edges_gdf) = _get_poly_nx(
            self.road_network,
            center_node=5909483619,
            dist_value=50,
            distance_type="length",
        )

        assert actual_edges_gdf.geom_almost_equals(edges_gdf, decimal=4).all()

    def test_raises_value_error_if_all_nodes_are_too_far(self):
        """All nodes are farther than 15 meters away from node 5909483619"""
        with pytest.raises(ValueError, match="Graph contains no edges"):
            _get_poly_nx(
                self.road_network,
                center_node=5909483619,
                dist_value=15,
                distance_type="length",
            )


class TestCalculateIsopolygonsGraph:

    def test_format(self, dataframe_with_lat_and_lon):

        isopolygons = calculate_isopolygons_graph(
            X=dataframe_with_lat_and_lon.longitude.values,
            Y=dataframe_with_lat_and_lon.latitude.values,
            distance_type="length",
            distance_values=[20, 50],
            road_network=ox.load_graphml(
                "tests/test_data/walk_network_4_nodes_6_edges.graphml"
            ),
        )

        assert isinstance(isopolygons, dict)

        assert isopolygons.keys() == {"ID_20", "ID_50"}

        assert isinstance(isopolygons["ID_20"], list)

        assert len(isopolygons["ID_50"]) == 2

        isopolygons_ID_50 = isopolygons["ID_50"]

        assert isinstance(isopolygons_ID_50[0], Polygon)
