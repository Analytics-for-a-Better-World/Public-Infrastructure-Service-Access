import pytest
import pandas as pd
from more_itertools.recipes import grouper

from gpbp.utils import generate_grid_in_polygon, group_population
from shapely.geometry import Polygon, MultiPolygon


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


df = pd.DataFrame({"longitude": [6.87641, 6.87644, 6.87964, 6.88710], "latitude": [53.06167, 53.06180, 53.06000, 53.08787],
                   "population": [5, 4, 3, 2]})

class TestGroupPopulation:
    def test_group_pop(self):
        grouped_pop_1 = group_population(df, 1)
        grouped_pop_2 = group_population(df, 2)
        grouped_pop_3 = group_population(df, 3)
        grouped_pop_4 = group_population(df, 4)
        assert grouped_pop_1.shape[0] == 1
        assert grouped_pop_2.shape[0] == 2
        assert grouped_pop_3.shape[0] == 3
        assert grouped_pop_4.shape[0] == 4
        assert grouped_pop_1.loc[(grouped_pop_1['longitude'] == 6.9) & (grouped_pop_1['latitude'] == 53.1)]['population'].values[0] == 14
        assert grouped_pop_2.loc[(grouped_pop_2['longitude'] == 6.88) & (grouped_pop_2['latitude'] == 53.06)][
                   'population'].values[0] == 12
        assert grouped_pop_3.loc[(grouped_pop_3['longitude'] == 6.876) & (grouped_pop_3['latitude'] == 53.062)]['population'].values[0] == 9



