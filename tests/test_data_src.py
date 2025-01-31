from unittest.mock import MagicMock
import pandas as pd
import geopandas as gpd

import pytest
import numpy as np
import rasterio

from shapely.geometry import Polygon, MultiPolygon, Point
from gpbp.data_src import raster_to_df, get_admarea_mask, osm_facilities, fb_pop_data


@pytest.fixture
def multipolygon():

    # Create two square polygons
    polygon1 = Polygon([(67.36, 36.29), (67.11, 36.00), (67.14, 35.96)])
    polygon2 = Polygon([(66.50, 36.90), (66.49, 36.91), (66.48, 36.92)])
    # Combine them into a MultiPolygon

    return MultiPolygon([polygon1, polygon2])


@pytest.fixture
def fake_raster_dataset():

    # Create a fake raster array with some non-zero values
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


@pytest.fixture()
def mock_hospital_osm_facilities_data():

    hospital1 = [
        "way",
        585261448,
        [5593434709, 5593434710, 5593434711, 5593434712, 5593434709],
        "hospital",
        Polygon(
            (
                (126.66558, -8.60363),
                (126.66574, -8.60379),
                (126.66581, -8.60371),
                (126.66565, -8.60356),
                (126.66558, -8.60363),
            )
        ),
    ]

    hospital2 = [
        "node",
        1131971312,
        [10554642882],
        "hospital",
        Point((26.21212, -8.51170)),
    ]

    gdf = gpd.GeoDataFrame(
        [hospital1, hospital2],
        columns=["element_type", "osmid", "nodes", "building", "geometry"],
    ).set_index(["element_type", "osmid"])

    return gdf


class TestGetAdmareaMask():
    def test_get_admarea_mask_correct(self, mocker, multipolygon, mock_raster_datasetreader, fake_raster_dataset):
        mock_mask = mocker.patch('data_src.riomask.mask', return_value=(fake_raster_dataset, None))

        adm_mask = get_admarea_mask(multipolygon, mock_raster_datasetreader)

        # Assert that the result adm_mask is correct
        expected_mask = fake_raster_dataset[0] > 0
        assert adm_mask.shape == expected_mask.shape
        assert np.array_equal(adm_mask, expected_mask)

        # Verify that riomask.mask was called with correct parameters
        mock_mask.assert_called_once_with(
            mock_raster_datasetreader, [multipolygon], all_touched=True, crop=False
        )
    def test_get_admarea_mask_empty(self):
        with pytest.raises(AttributeError):
            get_admarea_mask(MultiPolygon([]), None)


class TestRastertoDF:
    def test_raster_to_df_correct(self, mocker, fake_raster_dataset, mock_raster_datasetreader, multipolygon):

        mock_src = mocker.patch('data_src.rasterio.open', return_value=mock_raster_datasetreader)
        mock_mask = mocker.patch('data_src.riomask.mask', return_value=(fake_raster_dataset, None))

        df = raster_to_df(raster_fpath='fake_path.tif', mask_polygon=multipolygon)

        assert df.shape[1] == 3
        assert mock_src.call_count == 1
        assert mock_mask.call_count == 1
        mock_src.assert_called_once_with('fake_path.tif')
        mock_mask.assert_called_once_with(mock_raster_datasetreader, [multipolygon], all_touched=True, crop=False)

    def test_raster_to_df_false_input_path(self):
        with pytest.raises(rasterio.errors.RasterioIOError):
            raster_to_df(raster_fpath='fake_path', mask_polygon=None)

    def test_raster_to_df_empty_polygon(self, mocker, mock_raster_datasetreader):
        mock_src = mocker.patch('data_src.rasterio.open', return_value=mock_raster_datasetreader)

        with pytest.raises(IndexError):
            raster_to_df(raster_fpath='fake_path.tif', mask_polygon=MultiPolygon([]))


class TestFBdata:
    def test_fb_data_mask_multipolygon_and_dataset_same(self, mocker, multipolygon):
        population_df = pd.DataFrame(
            {
        "longitude": [67.36, 67.11, 67.14, 66.50, 66.49, 66.48],
        "latitude": [36.29, 36.00, 35.96, 36.90, 36.91, 36.92],
        "population": [1, 2, 3, 4, 5, 6]
        }
        )

        mocker.patch("data_src.Resource.search_in_hdx", return_value=[{'id': 'fake_id', 'download_url': 'fake_url'}])
        mocker.patch("data_src.urllib.request.urlretrieve", return_value= ('fakehandle.csv', None))
        mocker.patch("data_src.pd.read_csv", return_value=population_df)

        fb_data = fb_pop_data('ABC', multipolygon)

        assert fb_data['population'].sum() == 21

    def test_fb_data_mask_multipolygon_and_dataset_partial_overlap(self, mocker, multipolygon):
        population_df = pd.DataFrame(
            {
        "longitude": [67.36, 67.11, 67.14, 54.13, 54.89, 53.45],
        "latitude": [36.29, 36.00, 35.96, 22.13, 23.54, 22.63],
        "population": [1, 2, 3, 4, 5, 6]
        }
        )

        mocker.patch("data_src.Resource.search_in_hdx", return_value=[{'id': 'fake_id', 'download_url': 'fake_url'}])
        mocker.patch("data_src.urllib.request.urlretrieve", return_value= ('fakehandle.csv', None))
        mocker.patch("data_src.pd.read_csv", return_value=population_df)

        fb_data = fb_pop_data('ABC', multipolygon)

        assert fb_data['population'].sum() == 6

    def test_fb_data_mask_multipolygon_and_dataset_no_overlap(self, mocker, multipolygon):
        population_df = pd.DataFrame(
            {
        "longitude": [54.13, 54.89, 53.45, 53.78, 54.09, 53.94],
        "latitude": [22.13, 23.54, 22.63, 22.87, 23.02, 22.98],
        "population": [1, 2, 3, 4, 5, 6]
        }
        )

        mocker.patch("data_src.Resource.search_in_hdx", return_value=[{'id': 'fake_id', 'download_url': 'fake_url'}])
        mocker.patch("data_src.urllib.request.urlretrieve", return_value= ('fakehandle.csv', None))
        mocker.patch("data_src.pd.read_csv", return_value=population_df)

        fb_data = fb_pop_data('ABC', multipolygon)

        assert fb_data['population'].sum() == 0


class TestOSMFacilities:
    def test_osm_facilities(self, mocker, multipolygon, mock_hospital_osm_facilities_data):
        mocker.patch("data_src.ox.features_from_polygon", return_value=mock_hospital_osm_facilities_data)

        gdf = osm_facilities('test_country', multipolygon, {"amenity": ["hospital", "clinic"]})

        # Check polygon
        assert gdf.loc[0, 'longitude'] == gdf.loc[0, 'geometry'].centroid.x
        assert gdf.loc[0, 'latitude'] == gdf.loc[0, 'geometry'].centroid.y

        # Check Point
        assert gdf.loc[1, 'longitude'] == 26.21212
        assert gdf.loc[1, 'latitude'] == -8.51170

    def test_osm_faciltiies_empty_multipolygon(self):
        with pytest.raises(ValueError):
            osm_facilities('test_country', MultiPolygon([]), {"amenity": ["hospital", "clinic"]})