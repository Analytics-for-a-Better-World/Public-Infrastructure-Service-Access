import random

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from shapely.geometry import MultiPolygon, Polygon, Point

from gpbp.layers import AdmArea


class TestAdmAreaInit:
    def test_valid_country_name(self, mocker):
        # Mock _get_country_data to avoid data fetching with GADMDownloader
        mocker.patch("gpbp.layers.AdmArea._get_country_data")

        adm_area = AdmArea(country="Timor-Leste", level=0)
        assert adm_area.country.name == "Timor-Leste"
        assert adm_area.level == 0

    def test_invalid_country_name_with_suggestions(self, mocker):
        # Mock _get_country_data to avoid data fetching with GADMDownloader
        mocker.patch("gpbp.layers.AdmArea._get_country_data")

        with pytest.raises(Exception) as exception_message:
            AdmArea(country="Timor", level=0)

        assert "Country not found. Possible matches: ['Timor-Leste']" in str(exception_message.value)

    def test_completely_invalid_country_name(self, mocker):
        # Mock _get_country_data to avoid data fetching with GADMDownloader
        mocker.patch("gpbp.layers.AdmArea._get_country_data")

        with pytest.raises(Exception) as exc_info:
            AdmArea(country="XXXXX", level=0)

        assert "Invalid form of country name" in str(exc_info.value)


@pytest.fixture
def multipolygon():
    # Create two simple square polygons
    polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    polygon2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

    # Combine them into a MultiPolygon
    return MultiPolygon([polygon1, polygon2])


@pytest.fixture
def mock_gdf(multipolygon):
    data = {
        'id': [0, 1],
        'COUNTRY': ["Mock Country", "Mock Country"],
        'NAME_1': ["Mock Region 1", "Mock Region 2"],
        'geometry': [multipolygon, multipolygon],
    }
    return gpd.GeoDataFrame(data, crs='EPSG:4326')

class TestAdmAreaGetCountryData:
    def test_get_country_data_level_0(self, mocker, multipolygon):
        data = {
            'id': [0],
            'COUNTRY': ["Mock Country"],
            'geometry': [multipolygon],
        }
        mock_gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')

        mocker.patch("gpbp.layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=0)

        assert isinstance(adm_area.geometry, MultiPolygon)
        assert adm_area.geometry == multipolygon
        assert adm_area.adm_name == "Timor-Leste"

    def test_get_country_data_level_1(self, mocker, capsys, mock_gdf):
        mocker.patch("gpbp.layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=1)

        printed_output = capsys.readouterr().out.strip().split('\n')
        for line_nr, line in enumerate(printed_output):
            if line.startswith("Administrative areas for level "):
                assert printed_output[line_nr + 1] == "['Mock Region 1' 'Mock Region 2']"
                break

        assert getattr(adm_area, "geometry", None) is None
        assert getattr(adm_area, "adm_name", None) is None


class TestAdmAreaRetrieveAdmAreaNames:
    def test_retrieve_adm_area_names_level_0(self, mocker):
        # Mock _get_country_data to avoid data fetching with GADMDownloader
        mocker.patch("gpbp.layers.AdmArea._get_country_data")

        adm_area = AdmArea(country="Timor-Leste", level=0)
        assert adm_area.retrieve_adm_area_names() == ["Timor-Leste"]

    def test_retrieve_adm_area_names_level_1(self, mocker, mock_gdf):
        mocker.patch("gpbp.layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=1)

        assert np.array_equal(adm_area.retrieve_adm_area_names(), np.array(["Mock Region 1", "Mock Region 2"]))


class TestAdmAreaGetAdmArea:
    @pytest.mark.xfail(reason="adm_name and geometry not set for level 0. Refactor", strict=True)
    def test_get_adm_area_level_0(self, mocker):
        mocker.patch("gpbp.layers.AdmArea._get_country_data")
        adm_area = AdmArea(country="Timor-Leste", level=0)
        adm_area.get_adm_area("Timor-Leste")

        assert adm_area.adm_name == "Timor-Leste"
        assert adm_area.geometry is not None

    def test_get_adm_area_valid_name(self, mocker, mock_gdf):
        mocker.patch("gpbp.layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=1)
        adm_area.get_adm_area("Mock Region 1")

        assert isinstance(adm_area.geometry, MultiPolygon)
        assert adm_area.geometry == mock_gdf.geometry[0]
        assert adm_area.adm_name == "Mock Region 1"

    def test_get_adm_area_invalid_name(self, mocker, capsys, mock_gdf):
        mocker.patch("gpbp.layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=1)
        adm_area.get_adm_area("Invalid Region")

        captured = capsys.readouterr()
        assert "No data found for Invalid Region" in captured.out


@pytest.fixture
def adm_area(mocker, multipolygon):
    mocker.patch("gpbp.layers.AdmArea._get_country_data")
    adm_area = AdmArea(country="Timor-Leste", level=0)
    adm_area.geometry = multipolygon
    adm_area.adm_name = "Timor-Leste"
    return adm_area

@pytest.fixture
def osm_hospital_tags():
    return {"building": "hospital"}

class TestAdmAreaGetFacilities:
    def test_get_facilities_valid_method(self, mocker, adm_area, osm_hospital_tags):
        mock_facilities_src = mocker.patch("gpbp.layers.FACILITIES_SRC", {"osm": mocker.Mock()})
        adm_area.get_facilities(method="osm", tags=osm_hospital_tags)
        mock_facilities_src["osm"].assert_called_once_with(adm_area.adm_name, adm_area.geometry, osm_hospital_tags)

    def test_get_facilities_invalid_method(self, adm_area, osm_hospital_tags):
        with pytest.raises(Exception) as exc_info:
            adm_area.get_facilities(method="invalid_method", tags=osm_hospital_tags)
        assert "Invalid method" in str(exc_info.value)

    def test_get_facilities_no_geometry(self, mocker, osm_hospital_tags):
        mocker.patch("gpbp.layers.AdmArea._get_country_data")
        adm_area = AdmArea(country="Timor-Leste", level=0)
        with pytest.raises(Exception) as exc_info:
            adm_area.get_facilities(method="osm", tags=osm_hospital_tags)
        assert "Geometry is not defined. Call get_adm_area()" in str(exc_info.value)


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


@pytest.mark.parametrize(["network_type", "default_speed"], [["driving", 50], ["walking", 4], ["cycling", 15]])
def test_get_road_network(mocker, adm_area, mock_graph, network_type, default_speed):
    mocker.patch("gpbp.layers.ox.graph_from_polygon", return_value=mock_graph)
    
    adm_area.get_road_network(network_type=network_type)

    for _, _, data in adm_area.road_network.edges(data=True):
        assert data["speed_kph"] == default_speed
        expected_travel_time = data["length"] / (data["speed_kph"] * 1000 / 60)  # length in meters, speed in kph
        assert round(data["travel_time"], 2) == round(expected_travel_time, 2)


class TestAdmAreaPrepareOptimizationData:
    @pytest.fixture
    def adm_area_with_population_and_facilities(self, adm_area, population_dataframe):
        adm_area.pop_df = population_dataframe
        adm_area.fac_gdf = gpd.GeoDataFrame({"ID": [0], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
        adm_area.pot_fac_gdf = gpd.GeoDataFrame({"ID": [1], "geometry": [Point(1, 1)]}, crs="EPSG:4326")

        return adm_area

    def test_prepare_optimization_data_pop_count(self, mocker, adm_area_with_population_and_facilities, population_dataframe):
        # Mock population_served, we're not testing it in this unit test
        dummy_current = {"length": "dummy_current_df"}
        dummy_potential = {"length": "dummy_potential_df"}
        mocker.patch(
            "gpbp.layers.population_served",
            side_effect=[dummy_current["length"], dummy_potential["length"]]
        )

        pop_count, _, _ = adm_area_with_population_and_facilities.prepare_optimization_data(
            distance_type="length",
            distance_values=[1000],
            mode_of_transport="driving",
            strategy="osm",
            population_resolution=1,  # Testing the minimum resolution. The result should be the total population.
        )

        assert pop_count == population_dataframe["population"].sum()

    def test_prepare_optimization_data_current_and_potential(self, mocker, adm_area_with_population_and_facilities):
        dummy_current = {"length": "dummy_current_df"}
        dummy_potential = {"length": "dummy_potential_df"}
        mocker.patch(
            "gpbp.layers.population_served",
            side_effect=[dummy_current["length"], dummy_potential["length"]]
        )

        # We don't care about pop_count here; just check current and potential outputs
        _, current, potential = adm_area_with_population_and_facilities.prepare_optimization_data(
            distance_type="length",
            distance_values=[1000],
            mode_of_transport="driving",
            strategy="osm"
        )

        assert current["length"] == dummy_current["length"]
        assert potential["length"] == dummy_potential["length"]
