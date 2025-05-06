import pytest

from pisa.utils import (
    validate_distance_type,
    validate_fallback_speed,
    validate_mode_of_transport,
)


@pytest.mark.parametrize(
    "speed, network_type, expected",
    [
        (None, "bike", None),
        (10, "drive", 10),
        (10, "walk", 10),
        (10, "bike", 10),
        (60, "drive", 60),
        (30.5, "drive", 30.5),
    ],
)
def test_validate_fallback_speed_input_valid(speed, network_type, expected):
    assert validate_fallback_speed(speed, network_type) == expected


@pytest.mark.parametrize(
    "speed, network_type",
    [
        (-4, "drive"),
        ("walk", "walk"),
        (1000, "bike"),
        (60, "walk"),
        (60, "bike"),
    ],
)
def test_validate_fallback_speed_input_invalid(speed, network_type):
    with pytest.raises(ValueError):
        validate_fallback_speed(speed, network_type)


class TestValidateDistanceType:
    @pytest.mark.parametrize(
        "input_distance, expected",
        [
            ("length", "length"),
            ("LeNgTh ", "length"),
        ],
    )
    def test_validate_distance_type(self, input_distance, expected):
        assert validate_distance_type(input_distance) == expected

    def test_validate_light_years(self):
        with pytest.raises(ValueError):
            validate_distance_type("lightyears")


class TestValidateModeOfTransport:
    @pytest.mark.parametrize(
        "input_mode, expected",
        [
            ("Driving", "driving"),
            ("WaLkInG ", "walking"),
        ],
    )
    def test_validate_mode_of_transport(self, input_mode, expected):
        assert validate_mode_of_transport(input_mode) == expected

    def test_validate_horseback(self):
        with pytest.raises(ValueError):
            validate_mode_of_transport("horseback")
