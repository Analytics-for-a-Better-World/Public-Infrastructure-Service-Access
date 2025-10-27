from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import MultiPolygon, Polygon

from pisa.population import FacebookPopulation, Population, WorldpopPopulation


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
def mock_raster_dataset(mocker):
    mock_dataset = mocker.MagicMock(spec=rasterio.io.DatasetReader)
    mock_dataset.crs = "EPSG:4326"
    mock_dataset.width = 100
    mock_dataset.height = 100

    def mock_xy(row, col):
        return (
            66.48 + col * (67.36 - 66.48) / (100 - 1),
            36.92 - row * (36.92 - 35.96) / (100 - 1),
        )

    mock_dataset.xy = MagicMock(side_effect=mock_xy)
    mock_dataset.read = MagicMock(return_value=np.random.rand(100, 100))
    # Mock context manager behavior for `rasterio.open()`
    mock_dataset.__enter__.return_value = mock_dataset
    mock_dataset.__exit__.return_value = None  # or whatever value you expect for exit

    return mock_dataset


@pytest.fixture
def population_instance_facebook(multipolygon):
    return FacebookPopulation(admin_area_boundaries=multipolygon, iso3_country_code="XYZ")


@pytest.fixture
def population_instance_worldpop(multipolygon):
    return WorldpopPopulation(multipolygon, iso3_country_code="XYZ")


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
                "population": [1, 2, 3, 4, 5, 6],
            }
        )
        result = population_instance_facebook.process_population_facebook(
            data,
            population_instance_facebook.iso3_country_code,
            population_instance_facebook.admin_area_boundaries,
        )

        assert isinstance(result, pd.DataFrame)
        assert "population" in result.columns
        assert result["population"].sum() == 21

    def test_process_population_facebook_mask_multipolygon_and_dataset_partial_overlap(
        self, population_instance_facebook
    ):
        data = pd.DataFrame(
            {
                "longitude": [67.36, 67.11, 67.14, 54.13, 54.89, 53.45],
                "latitude": [36.29, 36.00, 35.96, 22.13, 23.54, 22.63],
                "population": [1, 2, 3, 4, 5, 6],
            }
        )

        result = population_instance_facebook.process_population_facebook(
            data,
            population_instance_facebook.iso3_country_code,
            population_instance_facebook.admin_area_boundaries,
        )

        assert result["population"].sum() == 6

    def test_process_population_facebook_mask_multipolygon_and_dataset_no_overlap(self, population_instance_facebook):
        population_df = pd.DataFrame(
            {
                "longitude": [54.13, 54.89, 53.45, 53.78, 54.09, 53.94],
                "latitude": [22.13, 23.54, 22.63, 22.87, 23.02, 22.98],
                "population": [1, 2, 3, 4, 5, 6],
            }
        )

        result = population_instance_facebook.process_population_facebook(
            population_df,
            population_instance_facebook.iso3_country_code,
            population_instance_facebook.admin_area_boundaries,
        )

        assert result["population"].sum() == 0


@patch("pisa.population.requests.get")
@patch("pisa.population.urllib.request.urlretrieve")
def test_download_population_worldpop(mock_urlretrieve, mock_requests, population_instance_worldpop):
    mock_requests.return_value.json.return_value = {"data": [{"files": ["http://example.com/data.tif"]}]}
    mock_urlretrieve.return_value = ("/tmp/data.tif", None)

    file_path = population_instance_worldpop.download_population_worldpop(
        population_instance_worldpop.iso3_country_code
    )

    assert file_path == "/tmp/data.tif"
    mock_requests.assert_called()
    mock_urlretrieve.assert_called()


class TestProcessPopulationWorldpop:
    def test_process_population_worldpop(
        self,
        mocker,
        mock_raster_dataset,
        fake_raster_dataset,
        population_instance_worldpop,
    ):
        mock_open = mocker.patch("pisa.population.rasterio.open", return_value=mock_raster_dataset)
        mock_mask = mocker.patch("pisa.population.mask", return_value=(fake_raster_dataset, None))

        df = population_instance_worldpop.process_population_worldpop(
            "fake_path.tif", population_instance_worldpop.admin_area_boundaries
        )

        assert df.shape[1] == 3
        assert mock_open.call_count == 1
        assert mock_mask.call_count == 1
        mock_open.assert_called_once_with("fake_path.tif")
        mock_mask.assert_called_once_with(
            mock_raster_dataset,
            [population_instance_worldpop.admin_area_boundaries],
            all_touched=True,
            crop=False,
        )

    def test_process_population_worldpop_false_input_path(self, population_instance_worldpop):
        with pytest.raises(rasterio.errors.RasterioIOError):
            population_instance_worldpop.process_population_worldpop(
                "fake_path", population_instance_worldpop.admin_area_boundaries
            )

    def test_process_population_worldpop_empty_polygon(self, mocker, mock_raster_dataset, population_instance_worldpop):
        mocker.patch("rasterio.open", return_value=mock_raster_dataset)

        with pytest.raises(IndexError):
            population_instance_worldpop.process_population_worldpop("fake_path.tif", MultiPolygon([]))


class TestAdmArea:
    def test_get_admarea_mask_correct(
        self,
        mocker,
        population_instance_worldpop,
        multipolygon,
        mock_raster_dataset,
        fake_raster_dataset,
    ):
        mock_mask = mocker.patch("pisa.population.mask", return_value=(fake_raster_dataset, None))

        adm_mask = population_instance_worldpop.get_admarea_mask(multipolygon, mock_raster_dataset)

        # Assert that the result adm_mask is correct
        expected_mask = fake_raster_dataset[0] > 0
        assert adm_mask.shape == expected_mask.shape
        assert np.array_equal(adm_mask, expected_mask)

        # Verify that mask was called with correct parameters
        mock_mask.assert_called_once_with(mock_raster_dataset, [multipolygon], all_touched=True, crop=False)

    def test_get_admarea_mask_empty(self, population_instance_worldpop):
        with pytest.raises(AttributeError):
            population_instance_worldpop.get_admarea_mask(MultiPolygon([]), None)


class TestGroupPopulation:
    @pytest.mark.parametrize("nof_digits, count_of_areas_included", [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_group_pop_length(self, population_dataframe, nof_digits, count_of_areas_included):
        group_pop = Population._group_population(population_dataframe, population_resolution=nof_digits)
        assert group_pop.shape[0] == count_of_areas_included

    @pytest.mark.parametrize(
        "nof_digits, longitude, latitude, population_sum",
        [
            (1, 6.9, 53.1, 14),
            (2, 6.88, 53.06, 12),
            (3, 6.876, 53.062, 9),
            (4, 6.8796, 53.0600, 3),
        ],
    )
    def test_group_pop_values(self, population_dataframe, nof_digits, longitude, latitude, population_sum):
        group_pop = Population._group_population(population_dataframe, population_resolution=nof_digits)
        assert (
            group_pop.loc[(group_pop["longitude"] == longitude) & (group_pop["latitude"] == latitude)][
                "population"
            ].values[0]
            == population_sum
        )
