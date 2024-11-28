import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from gpbp.layers import AdmArea


@pytest.fixture
def multipolygon():
    # Create two simple square polygons
    polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    polygon2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

    # Combine them into a MultiPolygon
    return MultiPolygon([polygon1, polygon2])

class TestAdmAreaGetCountryData:
    def test_get_country_data_level_0(self, mocker, multipolygon):
        data = {
            'id': [0],
            'COUNTRY': ["Mock Country"],
            'geometry': [multipolygon],
        }
        mock_gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')

        mocker.patch("layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=0)

        assert isinstance(adm_area.geometry, MultiPolygon)
        assert adm_area.geometry == multipolygon
        assert adm_area.adm_name == "Timor-Leste"

    def test_get_country_data_level_1(self, mocker, capsys, multipolygon):
        data = {
            'id': [0, 1],
            'COUNTRY': ["Mock Country", "Mock Country"],
            'NAME_1': ["Mock Region 1", "Mock Region 2"],
            'geometry': [multipolygon, multipolygon],
        }
        mock_gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')

        mocker.patch("layers.GADMDownloader.get_shape_data_by_country_name", return_value=mock_gdf)
        adm_area = AdmArea(country="Timor-Leste", level=1)

        printed_output = capsys.readouterr().out.strip().split('\n')
        for line_nr, line in enumerate(printed_output):
            if line.startswith("Administrative areas for level "):
                assert printed_output[line_nr + 1] == "['Mock Region 1' 'Mock Region 2']"
                break

        assert getattr(adm_area, "geometry", None) is None
        assert getattr(adm_area, "adm_name", None) is None
