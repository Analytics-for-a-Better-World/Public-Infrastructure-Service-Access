from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon

from pisa.facilities import Facilities


@pytest.fixture
def simple_polygon():
    """Create a simple square polygon for testing"""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def multi_polygon():
    """Create a MultiPolygon with two simple polygons"""
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    return MultiPolygon([poly1, poly2])


@pytest.fixture()
def fake_facilities_gdf() -> GeoDataFrame:
    """Create a mock GeoDataFrame as would be returned by ox.features_from_polygon"""
    idx = pd.MultiIndex.from_tuples(
        [("node", 1777614876), ("node", 1777614896), ("way", 527394448)],
        names=["element", "id"],
    )

    hospital_geometries = [
        Point(126.60048, -8.54733),
        Point(126.65333, -8.62843),
        Polygon(
            [
                (126.449046, -8.479301),
                (126.446577, -8.478061),
                (126.445365, -8.480475),
                (126.447833, -8.481715),
                (126.449046, -8.479301),
            ]
        ),
    ]

    gdf = gpd.GeoDataFrame(
        index=idx,
        data={"geometry": hospital_geometries, "amenity": "hospital"},
        crs="EPSG:4326",
    )

    return gdf


class TestFacilities:
    def test_init(self, simple_polygon):
        """Test proper initialization of Facilities class"""
        facilities = Facilities(administrative_area=simple_polygon)

        assert facilities.administrative_area == simple_polygon
        assert facilities.data_src == "osm"
        assert facilities.osm_tags == {"amenity": "hospital"}

    def test_init_with_custom_tags(self, simple_polygon):
        """Test initialization with custom location tags"""
        custom_tags = {"amenity": "school"}
        facilities = Facilities(
            administrative_area=simple_polygon, data_src="osm", osm_tags=custom_tags
        )

        assert facilities.administrative_area == simple_polygon
        assert facilities.data_src == "osm"
        assert facilities.osm_tags == custom_tags

    def test_initialization_with_multipolygon(self, multi_polygon):
        """Test initialization with MultiPolygon"""
        facilities = Facilities(administrative_area=multi_polygon)

        assert facilities.administrative_area == multi_polygon
        assert facilities.data_src == "osm"

    def test_get_existing_facilities_invalid_data_src(self, simple_polygon):
        """Test error when using invalid data source"""
        facilities = Facilities(administrative_area=simple_polygon, data_src="invalid")

        with pytest.raises(
            NotImplementedError, match="Data source 'invalid' not implemented"
        ):
            facilities.get_existing_facilities()

    @patch("pisa.facilities.ox.features_from_polygon")
    def test_get_existing_facilities_osm(
        self, mocked_osm_response, fake_facilities_gdf, multi_polygon
    ):
        mocked_osm_response.return_value = fake_facilities_gdf

        facilities_df = Facilities._get_existing_facilities_osm(
            osm_tags={"amenity": "hospital"}, administrative_area=multi_polygon
        )

        mocked_osm_response.assert_called_once()

        assert isinstance(facilities_df, pd.DataFrame)
        assert list(facilities_df.columns) == ["longitude", "latitude"]

        # Should have 3 entries (2 nodes + 1 way)
        assert len(facilities_df) == 3

        # Create fake centroids with re-projecting the geometries to mimik behaviour in get_existing_facilities_osm
        fake_centroids = fake_facilities_gdf.to_crs("EPSG:4087").centroid.to_crs(
            fake_facilities_gdf.crs
        )

        # Check elements "node", should have coordinates directly from Point geometry
        assert (
            facilities_df.loc[1777614876, "longitude"]
            == fake_centroids.loc["node", 1777614876].x
        )  # 126.60048
        assert (
            facilities_df.loc[1777614876, "latitude"]
            == fake_centroids.loc["node", 1777614876].y
        )  # -8.54733
        assert (
            facilities_df.loc[1777614896, "longitude"]
            == fake_centroids.loc["node", 1777614896].x
        )  # 126.65333
        assert (
            facilities_df.loc[1777614896, "latitude"]
            == fake_centroids.loc["node", 1777614896].y
        )  # -8.62843

        # Check element "way", should have coordinates as the centroid of the Polygon geometry
        assert (
            facilities_df.loc[527394448, "longitude"]
            == fake_centroids.loc["way", 527394448].x
        )
        assert (
            facilities_df.loc[527394448, "latitude"]
            == fake_centroids.loc["way", 527394448].y
        )

    def test_estimate_potential_facilities_format(self, simple_polygon):
        """Test the format of the potential facilities GeoDataFrame"""
        facilities = Facilities(administrative_area=simple_polygon)
        result = facilities.estimate_potential_facilities(spacing=0.5)

        assert isinstance(result, gpd.GeoDataFrame)
        assert "ID" in result.columns
        assert "longitude" in result.columns
        assert "latitude" in result.columns
        assert "geometry" in result.columns

    def test_estimate_potential_facilities_grid(self, simple_polygon):
        """Test the creation of the grid to estimate potential facilities"""
        test_spacing = 0.5
        facilities = Facilities(administrative_area=simple_polygon)
        result = facilities.estimate_potential_facilities(spacing=test_spacing)

        # Check grid points are within boundaries
        assert result["longitude"].min() == 0.0
        assert result["longitude"].max() == 1.0
        assert result["latitude"].min() == 0.0
        assert result["latitude"].max() == 1.0

        # With 0.5 spacing in a 1x1 square, should have 9 points (3x3 grid)
        assert len(result) == 9

        # Check grid spacing
        for i in range(len(result) - 1):
            if result.iloc[i].longitude == result.iloc[i + 1].longitude:
                assert (
                    result.iloc[i + 1].latitude - result.iloc[i].latitude == test_spacing
                )
            if result.iloc[i].latitude == result.iloc[i + 1].latitude:
                assert (
                    result.iloc[i + 1].longitude - result.iloc[i].longitude
                    == test_spacing
                )

        # Should have more points with smaller spacing
        result_smaller = facilities.estimate_potential_facilities(
            spacing=test_spacing / 2
        )
        assert len(result_smaller) > len(result)

    def test_estimate_potential_facilities_grid_multipolygon(self, multi_polygon):
        """Test the creation of the grid to estimate potential facilities (MultiPolygon)"""
        test_spacing = 0.5
        facilities = Facilities(administrative_area=multi_polygon)
        result = facilities.estimate_potential_facilities(spacing=test_spacing)

        # Check grid points are within boundaries
        assert result["longitude"].min() == 0.0
        assert result["longitude"].max() == 2.0
        assert result["latitude"].min() == 0.0
        assert result["latitude"].max() == 2.0

        # With 0.5 spacing in two 1x1 square where one point overlaps, should have 17 points (2x3x3 grid - 1)
        assert len(result) == 17

        # Should have more points with smaller spacing
        result_smaller = facilities.estimate_potential_facilities(
            spacing=test_spacing / 2
        )
        assert len(result_smaller) > len(result)

        point_in_grid = (test_spacing, test_spacing)
        point_not_in_grid = (0, 2.0)
        point_not_in_grid2 = (test_spacing, test_spacing / 2)
        assert (
            result.loc[
                (result["longitude"] == point_in_grid[0])
                & (result["latitude"] == point_in_grid[1])
            ].shape[0]
            == 1
        )
        assert (
            result.loc[
                (result["longitude"] == point_not_in_grid[0])
                & (result["latitude"] == point_not_in_grid[1])
            ].shape[0]
            == 0
        )
        assert (
            result.loc[
                (result["longitude"] == point_not_in_grid2[0])
                & (result["latitude"] == point_not_in_grid2[1])
            ].shape[0]
            == 0
        )
