"""
Catalina, Feb 25:

Temporary test. Useful for refactoring stage
to make sure that the code for calculating isopolygons
with mapbox is the same. When gpbp folder is gone, 
delete.

In the meantime, here's how to use it:

- clear the mapbox_cache folder
- insert your token 
- remove skips
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gpbp.distance import calculate_isopolygons_Mapbox
from pisa.isopolygons_mapbox import MapboxIsopolygonCalculator

MAPBOX_TOKEN = "INSERT YOUR MAPBOX TOKEN HERE"


@pytest.fixture
def dataframe_with_lon_and_lat() -> pd.DataFrame:
    """Location (longitude and latitude) of an actual facility in Baucau, Timor-Leste"""

    points = [
        (126.60048, -8.54733),
    ]

    return pd.DataFrame(points, columns=["longitude", "latitude"])


class TestMapboxIsopolygonCalculator:

    @pytest.fixture(autouse=True)
    def setup(self, dataframe_with_lon_and_lat):

        self.distance_values = [1000, 2000]

        self.isopolygon_calculator = MapboxIsopolygonCalculator(
            facilities_df=dataframe_with_lon_and_lat,
            distance_type="length",
            distance_values=self.distance_values,
            route_profile="driving",
            mapbox_api_token=MAPBOX_TOKEN,
        )

        self.isopolygons = self.isopolygon_calculator.calculate_isopolygons()

    @pytest.mark.skip(reason="automatically run without valid API token")
    def test_old_code_is_same_as_new(self, dataframe_with_lon_and_lat):

        old_isopolygons = calculate_isopolygons_Mapbox(
            X=dataframe_with_lon_and_lat.longitude.to_list(),
            Y=dataframe_with_lon_and_lat.latitude.to_list(),
            route_profile="driving",
            distance_type="length",
            distance_values=self.distance_values,
            access_token=MAPBOX_TOKEN,
        )

        assert_frame_equal(pd.DataFrame(old_isopolygons), self.isopolygons)
