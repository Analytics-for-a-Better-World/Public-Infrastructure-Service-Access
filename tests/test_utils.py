import pytest
import pandas as pd

from gpbp.utils import generate_grid_in_polygon, group_population
from shapely.geometry import Polygon, MultiPolygon

@pytest.fixture
def multipolygon():
    # Create two simple square polygons
    polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    polygon2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

    # Combine them into a MultiPolygon
    return MultiPolygon([polygon1, polygon2])


class TestGenerateGridInPolygon:
    def test_bounds_in_polygon(self, multipolygon):
        grid = generate_grid_in_polygon(1, multipolygon)
        assert grid['longitude'].min() == 0.0
        assert grid['longitude'].max() == 2.0
        assert grid['latitude'].min() == 0.0
        assert grid['latitude'].max() == 2.0

    def test_number_of_points_in_polygon(self, multipolygon):
        grid = generate_grid_in_polygon(0.5, multipolygon)
        assert grid.shape[0] == 13

    def test_points_in_polygon(self, multipolygon):
        grid = generate_grid_in_polygon(0.5, multipolygon)
        point_in_grid = (0.5, 0.5)
        point_not_in_grid = (0.5, 1.5)
        assert grid.loc[(grid['longitude'] == point_in_grid[0]) & (grid['latitude'] == point_in_grid[1])].shape[0] == 1
        assert grid.loc[(grid['longitude'] == point_not_in_grid[0]) & (grid['latitude'] == point_not_in_grid[1])].shape[0] == 0

    def test_spacing_in_grid(self, multipolygon):
        grid = generate_grid_in_polygon(1.5, multipolygon)
        for i in range(len(grid) - 1):
            if grid.iloc[i].longitude == grid.iloc[i + 1].longitude:
                assert grid.iloc[i + 1].latitude - grid.iloc[i].latitude == 1.5
            if grid.iloc[i].latitude == grid.iloc[i + 1].latitude:
                assert grid.iloc[i + 1].longitude - grid.iloc[i].longitude == 1.5

    def test_empty_multipolygon(self):
        empty_multipolygon = MultiPolygon([])
        with pytest.raises(ValueError):
            generate_grid_in_polygon(0.5, empty_multipolygon)



@pytest.fixture()
def df():
    return pd.DataFrame({"longitude": [6.87641, 6.87644, 6.87964, 6.88710], "latitude": [53.06167, 53.06180, 53.06000, 53.08787],
                         "population": [5, 4, 3, 2]})


class TestGroupPopulation:
    @pytest.mark.parametrize("nof_digits, count_of_areas_included", [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_group_pop_length(self, df, nof_digits, count_of_areas_included):
        group_pop = group_population(df, nof_digits=nof_digits)
        assert group_pop.shape[0] == count_of_areas_included

    @pytest.mark.parametrize("nof_digits, longitude, latitude, population_sum", [(1, 6.9, 53.1, 14), (2, 6.88, 53.06, 12), (3, 6.876, 53.062, 9), (4, 6.8796, 53.0600, 3)])
    def test_group_pop_values(self, df, nof_digits, longitude, latitude, population_sum):
        group_pop = group_population(df, nof_digits=nof_digits)
        assert group_pop.loc[(group_pop['longitude'] == longitude) & (group_pop['latitude'] == latitude)]['population'].values[0] == population_sum