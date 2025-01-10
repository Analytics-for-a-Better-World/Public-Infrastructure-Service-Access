from unittest.mock import MagicMock
import geopandas as gpd

import pytest
import numpy as np

from shapely.geometry import Polygon, MultiPolygon
from gpbp import data_src, layers
from gpbp.data_src import raster_to_df, get_admarea_mask, osm_facilities


def test_init_admarea():
    adm_area = layers.AdmArea("GREECE", level=0)
    assert adm_area is not None


@pytest.mark.skip(reason="legacy test")
def test_world_pop():
    adm_area = layers.AdmArea("GRC", level=0)
    adm_area.get_adm_area("Greece")
    df = data_src.world_pop_data(adm_area=adm_area)
    est_pop = round(df["population"].sum() / 1000000, 1)
    print(est_pop)
    assert est_pop == 10.4


@pytest.mark.skip(reason="legacy test")
def test_fb_pop():
    adm_area = layers.AdmArea("GRC", level=0)
    adm_area.get_adm_area("Greece")
    df = data_src.fb_pop_data(adm_area=adm_area)
    est_pop = round(df["population"].sum() / 1000000, 1)
    print(est_pop)
    assert est_pop == 10.4


@pytest.fixture
def multipolygon():
    # Create two square polygons
    polygon1 = Polygon([(67.36, 36.29), (67.11, 36.00), (67.14, 35.96)])
    polygon2 = Polygon([(66.50, 36.90), (66.49, 36.91), (66.48, 36.92)])
    # Combine them into a MultiPolygon
    return MultiPolygon([polygon1, polygon2])


@pytest.fixture
def polygon():
    return Polygon([(67.36, 36.29), (67.11, 36.00), (67.14, 35.96)])


@pytest.fixture
def mock_raster_datasetreader(mocker):
    mock_raster_dataset = mocker.MagicMock()
    mock_raster_dataset.crs = "EPSG:4326"
    mock_raster_dataset.width = 100
    mock_raster_dataset.height = 100
    mock_raster_dataset.x = 66.48
    mock_raster_dataset.y = 36.92
    mock_raster_dataset.xy = MagicMock(side_effect=lambda row, col: (
        66.48 + col * (67.36 - 66.48) / (100 - 1),
        36.92 - row * (36.92 - 35.96) / (100 - 1),
    ))
    mock_raster_dataset.read = MagicMock(side_effect=lambda band: np.random.rand(100, 100), dtype=np.uint8)
    return mock_raster_dataset

@pytest.fixture
def fake_raster_dataset():
    # Create a fake raster array with some non-zero values
    fake_raster_data = np.zeros((1, 100, 100), dtype=np.uint8)
    fake_raster_data[0, 50:60, 50:60] = 1  # Simulate non-zero values for the mask
    return fake_raster_data

class TestGetAdmareaMask():
    def test_get_admarea_mask(self, mocker, multipolygon, mock_raster_datasetreader, fake_raster_dataset):
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


class TestRastertoDF:
    def test_raster_to_df(self, mocker, fake_raster_dataset, mock_raster_datasetreader, multipolygon):

        mock_src = mocker.patch('data_src.rasterio.open', return_value=mock_raster_datasetreader)
        mock_mask = mocker.patch('data_src.riomask.mask', return_value=(fake_raster_dataset, None))

        df = raster_to_df(raster_fpath='fake_path.tif', mask_polygon=multipolygon)

        assert df is not None
        assert df.shape[1] == 3
        assert mock_src.call_count == 1
        assert mock_mask.call_count == 1
        mock_src.assert_called_once_with('fake_path.tif')
        mock_mask.assert_called_once_with(mock_raster_datasetreader, [multipolygon], all_touched=True, crop=False)


@pytest.fixture
def mock_gdf(mocker, polygon):
    mock_gdf = mocker.MagicMock()
    mock_gdf.crs = "EPSG:4326"
    mock_gdf.nodes = 1
    mock_gdf.amenity = "hospital"
    mock_gdf.building = "yes"
    mock_gdf.name = "Hospital"
    mock_gdf.geometry = polygon
    mock_gdf.geometry.values = [polygon]
    mock_gdf.get_level_values = MagicMock(return_value=["1"])
    return mock_gdf

class TestOSMFacilities:
    def test_osm_facilities(self, mocker, mock_gdf, multipolygon):
        mocker.patch('data_src.ox.geometries_from_polygon', return_value=mock_gdf)

        gdf = osm_facilities('test_country', multipolygon, {"amenity": ["hospital", "clinic"]})

        assert gdf is not None
        assert gdf.shape[1] == 4
