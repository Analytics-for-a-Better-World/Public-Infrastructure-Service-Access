import osmnx as ox
import pandas as pd
import pytest


@pytest.fixture()
def population_dataframe():
    return pd.DataFrame(
        {
            "longitude": [6.87641, 6.87644, 6.87964, 6.88710],
            "latitude": [53.06167, 53.06180, 53.06000, 53.08787],
            "population": [5, 4, 3, 2],
        }
    )
