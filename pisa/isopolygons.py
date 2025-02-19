from abc import ABC, abstractmethod

from networkx import MultiDiGraph
from pandas import DataFrame


class IsopolygonCalculator(ABC):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,  # e.g. travel_time or length
        distance_values: list[int],
    ):
        # error if distance values are too high
        # convert lon and lat to np arrays
        ...

    # TODO: is dict[DataFrame] the return type we want?

    @abstractmethod
    def calculate_isopolygons(self) -> dict[DataFrame]:
        """must be implemented in subclasses"""
        pass


class OsmIsopolygonCalculator(IsopolygonCalculator):

    def __init__(
        self,
        facilities_lat_lon: DataFrame,
        distance_type: str,
        distance_values: list[int],
        road_network: MultiDiGraph,
    ):
        super().__init__(facilities_lat_lon, distance_type, distance_values)
        self.road_network = road_network

    def calculate_isopolygons(self) -> dict[DataFrame]: ...


class MapboxIsopolygonCalculator(IsopolygonCalculator):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,
        distance_values: list[int],
        route_profile: str,  # ?
        mapbox_api_token: str,
    ):
        super().__init__(facilities_lon_lat, distance_type, distance_values)
        self.route_profile = route_profile
        self.mapbox_api_token = mapbox_api_token

    def calculate_isopolygons(self) -> dict[DataFrame]: ...


# If you want to implement a new way of calculating isopolygons (e.g. GoogleMaps),
# create a class that inherits from IsopolygonCalculator and implements calculate_isopolygons
