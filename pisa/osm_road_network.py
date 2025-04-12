import logging

import networkx as nx
import osmnx as ox
from shapely import MultiPolygon, Polygon

from pisa.constants import (
    DEFAULT_FALLBACK_CYCLING_SPEED,
    DEFAULT_FALLBACK_DRIVING_SPEED,
    DEFAULT_FALLBACK_WALKING_SPEED,
)
from pisa.utils import _validate_distance_type, _validate_mode_of_transport

logger = logging.getLogger(__name__)


class OsmRoadNetwork:
    """
    Class to retrieve and process OpenStreetMap road network data.

    TODO: adjust docstring, explain what fallback speed means (check
    docstring of ox.add_edge_speeds)

    Parameters
    ----------
    admin_area_boundaries: Polygon | MultiPolygon
        The geography of the administrative area object.

    mode_of_transport: str
        The mode of transport for which the road network is required.

    distance_type: str
        The type of distance to be calculated.

    """

    def __init__(
        self,
        admin_area_boundaries: Polygon | MultiPolygon,
        mode_of_transport: str,  # must be an element of VALID_MODES_OF_TRANSPORT
        distance_type: str,  # must be an element of VALID_DISTANCE_TYPES
        fallback_speed: str = None,
    ):
        # validate distance type
        self.distance_type = _validate_distance_type(distance_type)

        # validate mode of transport
        mode_of_transport = _validate_mode_of_transport(mode_of_transport)

        self.network_type = self._set_network_type(mode_of_transport)

        self.admin_area_boundaries = admin_area_boundaries

        if self.distance_type == "travel_time":
            self.fallback_speed = self._set_fallback_speed(fallback_speed)

        logger.info(
            """OSM road network set with parameters 
                    
                    network_type f"{self.network_type}",
                    distance_type f"{self.distance_type}",                    
            """
        )

    def get_osm_road_network(self) -> nx.MultiDiGraph:
        """Returns the processed OSM road network."""

        road_network = self._download_osm_road_network()

        if self.distance_type == "travel_time":
            return self._add_time_to_edges(road_network, self.fallback_speed)

        return road_network

    def _download_osm_road_network(self) -> nx.MultiDiGraph:
        """Download the OSM road network from OpenStreetMap for the specified administrative area."""

        return ox.graph_from_polygon(
            polygon=self.admin_area_boundaries, network_type=self.network_type
        )

    @staticmethod
    def _set_network_type(mode_of_transport: str) -> str:
        """TODO: modify only the strings (e.g. driving -> drive)"""

        ...

    ## FYI: next functions are only used when calculating isochrones, not distances

    @staticmethod
    def _add_time_to_edges(
        road_network: nx.MultiDiGraph, road_speed: int
    ) -> nx.MultiDiGraph:
        """Add travel time edge attribute and change unit to minutes

        TODO:
        - add tests!!! Important to understand what fallback speed means in practice

        - specify units (kph?)

        - add docstring

        """
        road_network = ox.add_edge_speeds(road_network, fallback=road_speed)
        road_network = ox.add_edge_travel_times(road_network)

        time = nx.get_edge_attributes(road_network, "travel_time")
        time_in_minutes = {k: round(v / 60, 2) for k, v in time.items()}
        nx.set_edge_attributes(road_network, time_in_minutes, "travel_time")

        return road_network

    def _set_fallback_speed(self, fallback_speed: int | None) -> int:
        """TODO:

        If user wrote in a fallback speed, make sure it sort of makes sense
        (see function _validate_speed_input)

        Otherwise, return default fallback speed for the network type

        """

        if fallback_speed is not None:
            return self._validate_fallback_speed_input(
                fallback_speed, self.network_type
            )

        ### TODO: return the default for the network_type

        ...

    @staticmethod
    def _validate_fallback_speed_input(fallback_speed: int, network_type: str) -> int:
        """TODO: do some "common-sense" checks if the user wants to override the
        default fallback speeds

        User should not input "very" high or low speeds proportionately to the default_fallback_speed for the network_type.
        Examples:
        - 40 might be ok for a driving network, but not for a walking network
        - 500 is inadmissible always

        If the road_speed makes sense, return it. Else, raise errors

        Important: clarify units (I think it's km/h, but please double check)

        TODO: add tests for corner cases

        """
        # raise errors if values don't make sense

        # else:

        logger.info("""Setting fallback speed to f"{fallback_speed}" """)

        return fallback_speed
        ...
