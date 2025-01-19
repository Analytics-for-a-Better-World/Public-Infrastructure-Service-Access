import osmnx as ox
import pytest
import pandas as pd


@pytest.fixture
def load_graphml_file(request):
    """
    Fixture to load the GraphML file from a given filepath.
    """
    return ox.load_graphml(request.param)

@pytest.fixture()
def population_dataframe():
    return pd.DataFrame({"longitude": [6.87641, 6.87644, 6.87964, 6.88710], "latitude": [53.06167, 53.06180, 53.06000, 53.08787],
                         "population": [5, 4, 3, 2]})
