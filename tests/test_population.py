import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import Polygon, MultiPolygon
from pisa.population import Population


@pytest.fixture
def multipolygon():
    polygon1 = Polygon([(67.36, 36.29), (67.11, 36.00), (67.14, 35.96)])
    polygon2 = Polygon([(66.50, 36.90), (66.49, 36.91), (66.48, 36.92)])
    return MultiPolygon([polygon1, polygon2])


@pytest.fixture
def fake_raster_dataset():
    fake_raster_data = np.zeros((1, 100, 100), dtype=np.uint8)
    fake_raster_data[0, 50:60, 50:60] = 1  # Simulate non-zero values for the mask
    return fake_raster_data


@pytest.fixture
def mock_raster_datasetreader(mocker):
    mock_raster_dataset = mocker.MagicMock()
    mock_raster_dataset.crs = "EPSG:4326"
    mock_raster_dataset.width = 100
    mock_raster_dataset.height = 100
    mock_raster_dataset.xy = MagicMock(side_effect=lambda row, col: (
        66.48 + col * (67.36 - 66.48) / (100 - 1),
        36.92 - row * (36.92 - 35.96) / (100 - 1),
    ))
    mock_raster_dataset.read = MagicMock(side_effect=lambda band: np.random.rand(100, 100), dtype=np.uint8)
    return mock_raster_dataset


@pytest.fixture
def population_instance_facebook(multipolygon):
    iso3 = "ABC"
    admin_area = multipolygon
    return Population("facebook", iso3, admin_area)


@pytest.fixture
def population_instance_worldpop(multipolygon):
    iso3 = "ABC"
    admin_area = multipolygon
    return Population("world_pop", iso3, admin_area)


@patch("pisa.population.Resource.search_in_hdx")
@patch("pisa.population.urllib.request.urlretrieve")
def test_download_population_facebook(mock_urlretrieve, mock_search, population_instance_facebook):
    mock_search.return_value = [{"download_url": "http://example.com/data.zip"}]
    mock_urlretrieve.return_value = ("/tmp/data.zip", None)

    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        df = population_instance_facebook.download_population_facebook(population_instance_facebook.iso3_country_code)

    assert isinstance(df, pd.DataFrame)
    mock_search.assert_called()
    mock_urlretrieve.assert_called()


class TestProcessPopulationFacebook:
    def test_process_population_facebook_mask_multipolygon_and_dataset_same(self, population_instance_facebook):
        data = pd.DataFrame(
            {
                "longitude": [67.36, 67.11, 67.14, 66.50, 66.49, 66.48],
                "latitude": [36.29, 36.00, 35.96, 36.90, 36.91, 36.92],
                "population": [1, 2, 3, 4, 5, 6]
            }
        )
        result = population_instance_facebook.process_population_facebook(data,
                                                                          population_instance_facebook.iso3_country_code,
                                                                          population_instance_facebook.admin_area_boundaries)

        assert isinstance(result, pd.DataFrame)
        assert "population" in result.columns
        assert result['population'].sum() == 21

    def test_process_population_facebook_mask_multipolygon_and_dataset_partial_overlap(self, population_instance_facebook):
        data = pd.DataFrame(
            {
                "longitude": [67.36, 67.11, 67.14, 54.13, 54.89, 53.45],
                "latitude": [36.29, 36.00, 35.96, 22.13, 23.54, 22.63],
                "population": [1, 2, 3, 4, 5, 6]
            }
        )

        result = population_instance_facebook.process_population_facebook(data,
                                                                          population_instance_facebook.iso3_country_code,
                                                                          population_instance_facebook.admin_area_boundaries)

        assert result['population'].sum() == 6

    def test_process_population_facebook_mask_multipolygon_and_dataset_no_overlap(self, population_instance_facebook):
        population_df = pd.DataFrame(
            {
                "longitude": [54.13, 54.89, 53.45, 53.78, 54.09, 53.94],
                "latitude": [22.13, 23.54, 22.63, 22.87, 23.02, 22.98],
                "population": [1, 2, 3, 4, 5, 6]
            }
        )

        result = population_instance_facebook.process_population_facebook(population_df,
                                                                          population_instance_facebook.iso3_country_code,
                                                                          population_instance_facebook.admin_area_boundaries)

        assert result['population'].sum() == 0


@patch("pisa.population.requests.get")
@patch("pisa.population.urllib.request.urlretrieve")
def test_download_population_worldpop(mock_urlretrieve, mock_requests, population_instance_worldpop):
    mock_requests.return_value.json.return_value = {"data": [{"files": ["http://example.com/data.tif"]}]}
    mock_urlretrieve.return_value = ("/tmp/data.tif", None)

    file_path = population_instance_worldpop.download_population_worldpop(population_instance_worldpop.iso3_country_code)

    assert file_path == "/tmp/data.tif"
    mock_requests.assert_called()
    mock_urlretrieve.assert_called()


class TestRastertoDF:
    def test_raster_to_df(self, mocker, mock_raster_datasetreader, fake_raster_dataset, population_instance_worldpop, multipolygon):
        mock_src = mocker.patch('data_src.rasterio.open', return_value=mock_raster_datasetreader)
        mock_mask = mocker.patch('pisa.population.riomask.mask', return_value=(fake_raster_dataset, None))

        df = population_instance_worldpop.raster_to_df("fake_path.tif", population_instance_worldpop.admin_area_boundaries)

        assert isinstance(df, pd.DataFrame)
        assert "population" in df.columns
        assert df.shape[1] == 3
        assert mock_src.call_count == 1
        assert mock_mask.call_count == 1
        mock_src.assert_called_once_with('fake_path.tif')
        mock_mask.assert_called_once_with(mock_raster_datasetreader, [multipolygon], all_touched=True, crop=False)

    def test_raster_to_df_false_input_path(self, population_instance_worldpop):
        with pytest.raises(rasterio.errors.RasterioIOError):
            population_instance_worldpop.raster_to_df(file_path='fake_path', mask_polygon=None)

    def test_raster_to_df_empty_polygon(self, mocker, mock_raster_datasetreader, population_instance_worldpop):
        mock_src = mocker.patch('data_src.rasterio.open', return_value=mock_raster_datasetreader)

        with pytest.raises(IndexError):
            population_instance_worldpop.raster_to_df(file_path='fake_path.tif', mask_polygon=MultiPolygon([]))


class TestAdmArea:
    def test_get_admarea_mask_correct(self, mocker, population_instance_worldpop, multipolygon, mock_raster_datasetreader, fake_raster_dataset):
        mock_mask = mocker.patch('pisa.population.riomask.mask', return_value=(fake_raster_dataset, None))

        adm_mask = population_instance_worldpop.get_admarea_mask(multipolygon, mock_raster_datasetreader)

        # Assert that the result adm_mask is correct
        expected_mask = fake_raster_dataset[0] > 0
        assert adm_mask.shape == expected_mask.shape
        assert np.array_equal(adm_mask, expected_mask)

        # Verify that riomask.mask was called with correct parameters
        mock_mask.assert_called_once_with(
            mock_raster_datasetreader, [multipolygon], all_touched=True, crop=False
        )

    def test_get_admarea_mask_empty(self, population_instance_worldpop):
        with pytest.raises(AttributeError):
            population_instance_worldpop.get_admarea_mask(MultiPolygon([]), None)

class TestGroupPopulation:
    @pytest.mark.parametrize("nof_digits, count_of_areas_included", [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_group_pop_length(self, population_instance_worldpop, population_dataframe, nof_digits, count_of_areas_included):
        group_pop = population_instance_worldpop.group_population(population_dataframe, population_resolution=nof_digits)
        assert group_pop.shape[0] == count_of_areas_included

    @pytest.mark.parametrize("nof_digits, longitude, latitude, population_sum", [(1, 6.9, 53.1, 14), (2, 6.88, 53.06, 12), (3, 6.876, 53.062, 9), (4, 6.8796, 53.0600, 3)])
    def test_group_pop_values(self, population_instance_worldpop, population_dataframe, nof_digits, longitude, latitude, population_sum):
        group_pop = population_instance_worldpop.group_population(population_dataframe, population_resolution=nof_digits)
        assert group_pop.loc[(group_pop['longitude'] == longitude) & (group_pop['latitude'] == latitude)]['population'].values[0] == population_sum