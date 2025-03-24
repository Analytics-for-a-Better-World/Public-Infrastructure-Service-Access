"""
Testing error handling that forces request to fulfill constraints of MapboxAPI
"""

import pandas as pd
import pytest

from pisa.isopolygons_mapbox import MapboxIsopolygonCalculator


@pytest.fixture
def valid_mapbox_api_token() -> str:
    """Fixture providing a mock Mapbox API token for testing"""
    return "a_valid_api_token"


@pytest.fixture
def valid_facilities_df() -> pd.DataFrame:
    """Location (longitude and latitude) of two ficticious facilities"""
    return pd.DataFrame(
        [(0, 1), (1, 2)],
        columns=["longitude", "latitude"],
    )


@pytest.mark.parametrize(
    "facilities_df,error_message",
    [
        (
            pd.DataFrame([(-1, 1)], columns=["x", "y"]),
            "facilities_df must have columns 'longitude' and 'latitude'",
        ),
        (
            pd.DataFrame([], columns=["longitude", "latitude"]),
            "facilities_df must have at least one row",
        ),
    ],
)
def test_invalid_facilities_df(facilities_df, error_message, valid_mapbox_api_token):
    with pytest.raises(ValueError, match=error_message):
        MapboxIsopolygonCalculator(
            facilities_df=facilities_df,
            distance_type="length",
            distance_values=[1000],
            route_profile="driving",
            mapbox_api_token=valid_mapbox_api_token,
        )


@pytest.mark.parametrize(
    "param_name,invalid_value",
    [
        ("distance_type", "blah"),
        ("route_profile", "flying"),
    ],
)
def test_invalid_distance_type_or_route_profile(
    param_name, invalid_value, valid_facilities_df, valid_mapbox_api_token
):
    params = {
        "facilities_df": valid_facilities_df,
        "distance_type": "length",
        "distance_values": [1000],
        "route_profile": "driving",
        "mapbox_api_token": valid_mapbox_api_token,
    }
    params[param_name] = invalid_value

    with pytest.raises(ValueError):
        MapboxIsopolygonCalculator(**params)


def test_too_many_distance_values(valid_facilities_df, valid_mapbox_api_token):
    with pytest.raises(
        ValueError, match="Mapbox API accepts a maximum of 4 distance_values"
    ):
        MapboxIsopolygonCalculator(
            facilities_df=valid_facilities_df,
            distance_type="length",
            distance_values=[20, 15, 40, 60, 10],
            route_profile="driving",
            mapbox_api_token=valid_mapbox_api_token,
        )


@pytest.mark.parametrize(
    "distance_values",
    [
        [10.5, 12],
        10.5,
        [10, "oops"],
    ],
)
def test_wrong_format_distance_values(
    distance_values, valid_facilities_df, valid_mapbox_api_token
):
    with pytest.raises(TypeError, match="distance_values must be a list of integers"):
        MapboxIsopolygonCalculator(
            facilities_df=valid_facilities_df,
            distance_type="length",
            distance_values=distance_values,
            route_profile="driving",
            mapbox_api_token=valid_mapbox_api_token,
        )


def test_validate_mapbox_distance_values():

    assert MapboxIsopolygonCalculator._validate_mapbox_distance_values([3, 1, 4]) == [
        1,
        3,
        4,
    ]

    with pytest.raises(
        ValueError, match="Mapbox API accepts a maximum of 4 distance_values"
    ):
        MapboxIsopolygonCalculator._validate_mapbox_distance_values([1, 2, 3, 4, 5])


def test_empty_string_mapbox_api_token(valid_facilities_df):
    with pytest.raises(ValueError, match="Mapbox API token is required"):
        MapboxIsopolygonCalculator(
            facilities_df=valid_facilities_df,
            distance_type="length",
            distance_values=[1000],
            route_profile="driving",
            mapbox_api_token="",
        )


def test_scalar_distance_value_is_handled(valid_facilities_df, valid_mapbox_api_token):
    calculator = MapboxIsopolygonCalculator(
        facilities_df=valid_facilities_df,
        distance_type="length",
        distance_values=10,
        route_profile="driving",
        mapbox_api_token=valid_mapbox_api_token,
    )
    assert calculator.distance_values == [10]
