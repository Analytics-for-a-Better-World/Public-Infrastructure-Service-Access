import logging

import networkx as nx
import osmnx as ox
from shapely import MultiPolygon, Polygon

from pisa.utils import (
    validate_distance_type,
    validate_fallback_speed,
    validate_mode_of_transport,
)

logger = logging.getLogger(__name__)


class OsmRoadNetwork:
    """
    Class to retrieve and process OpenStreetMap road network data.

    Parameters
    ----------
    admin_area_boundaries: Polygon | MultiPolygon
        The geography of the administrative area object.

    mode_of_transport: str
        The mode of transport for which the road network is required.

    distance_type: str
        The type of distance to be calculated.

    fallback_speed: int | float | None
        The speed to be used for road types where OSM does not provide a speed attribute.
        If not provided, osmnx will do the imputation (recommended).
    """

    def __init__(
        self,
        admin_area_boundaries: Polygon | MultiPolygon,
        mode_of_transport: str,  # must be an element of VALID_MODES_OF_TRANSPORT
        distance_type: str,  # must be an element of VALID_DISTANCE_TYPES
        fallback_speed: int | float | None = None,
    ):
        self.admin_area_boundaries = admin_area_boundaries

        self.distance_type = validate_distance_type(distance_type)

        self.network_type = self._set_network_type(mode_of_transport)

        if self.distance_type == "travel_time":
            self.fallback_speed = validate_fallback_speed(
                fallback_speed, network_type=self.network_type
            )

        logger.info(
            f"OSM road network set with parameters network_type '{self.network_type}' and distance_type '{self.distance_type}'"
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
        """Set valid network_type based on valid values for mode_of_transport"""

        # validate mode of transport
        mode_of_transport = validate_mode_of_transport(mode_of_transport)

        if mode_of_transport == "driving":
            return "drive"
        if mode_of_transport == "walking":
            return "walk"
        # Info: only other option in current implementation is cycling.
        # If more modes of transport are added, adapt this function
        return "bike"

    @staticmethod
    def _add_time_to_edges(
        road_network: nx.MultiDiGraph, fallback_speed: int | float | None
    ) -> nx.MultiDiGraph:
        """Add travel time edge attribute and change unit to minutes"""
        road_network = ox.add_edge_speeds(road_network, fallback=fallback_speed)
        road_network = ox.add_edge_travel_times(road_network)

        time = nx.get_edge_attributes(road_network, "travel_time")
        time_in_minutes = {k: round(v / 60, 2) for k, v in time.items()}
        nx.set_edge_attributes(road_network, time_in_minutes, "travel_time")

        return road_network
