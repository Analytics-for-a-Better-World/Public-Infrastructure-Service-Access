import pytest
from blosc2.info import info_text_report

from gpbp.utils import generate_grid_in_polygon
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd


polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
polygon2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
multi_polygon = MultiPolygon([polygon1, polygon2])


class TestGenerateGridInPolygon:
    def test_bounds_in_polygon(self):
        grid = generate_grid_in_polygon(1, multi_polygon)
        assert multi_polygon.bounds == (0.0, 0.0, 3.0, 3.0)
        assert grid['longitude'].min() == 0.0
        assert grid['longitude'].max() == 2.0
        assert grid['latitude'].min() == 0.0
        assert grid['latitude'].max() == 2.0

    def test_number_of_points_in_polygon(self):
        grid = generate_grid_in_polygon(0.5, multi_polygon)
        assert grid.shape[0] == 13

    def test_points_in_polygon(self):
        grid = generate_grid_in_polygon(0.5, multi_polygon)
        point_in_grid = (0.5, 0.5)
        point_not_in_grid = (0.5, 1.5)
        assert grid.loc[(grid['longitude'] == point_in_grid[0]) & (grid['latitude'] == point_in_grid[1])].shape[0] == 1
        assert grid.loc[(grid['longitude'] == point_not_in_grid[0]) & (grid['latitude'] == point_not_in_grid[1])].shape[0] == 0

    def test_spacing_in_grid(self):
        grid = generate_grid_in_polygon(1.5, multi_polygon)
        for i in range(len(grid) - 1):
            if grid.iloc[i].longitude == grid.iloc[i + 1].longitude:
                assert grid.iloc[i + 1].latitude - grid.iloc[i].latitude == 1.5
            if grid.iloc[i].latitude == grid.iloc[i + 1].latitude:
                assert grid.iloc[i + 1].longitude - grid.iloc[i].longitude == 1.5

    def test_empty_multipolygon(self):
        empty_multipolygon = MultiPolygon([])
        with pytest.raises(ValueError):
            generate_grid_in_polygon(0.5, empty_multipolygon)

    def test_single_polygon(self):
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        with pytest.raises(ValueError):
            generate_grid_in_polygon(0.5, polygon)

