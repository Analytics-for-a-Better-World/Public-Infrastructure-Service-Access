"""OpenStreetMap road network data retrieval and processing module.

This module provides functionality for accessing and processing road network data from OpenStreetMap (OSM)
for various transportation modes (driving, walking, cycling). It serves as the foundation for service area
calculations and travel time/distance analyses in the PISA package.

Examples
--------
Retrieve and process a road network for walking travel time analysis:

>>> from pisa.administrative_area import AdministrativeArea
>>> from pisa.osm_road_network import OsmRoadNetwork
>>>
>>> # Get administrative area boundaries
>>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
>>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
>>>
>>> # Create a road network for walking travel time analysis
>>> road_network = OsmRoadNetwork(
>>>     admin_area_boundaries=boundaries,
>>>     mode_of_transport="walking",
>>>     distance_type="travel_time"
>>> )
>>>
>>> # Get the processed network with travel time attributes
>>> graph = road_network.get_osm_road_network()
>>>
>>> # Check some network statistics
>>> print(f"Number of nodes: {len(graph.nodes)}")
>>> print(f"Number of edges: {len(graph.edges)}")

See Also
--------
isopolygons : Module for calculating service areas using road networks
administrative_area : Module for defining the geographic scope of the road network
"""

import logging

import networkx as nx
import osmnx as ox
from shapely import MultiPolygon, Polygon

from pisa.utils import (validate_distance_type, validate_fallback_speed,
                        validate_mode_of_transport)

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
            self.fallback_speed = validate_fallback_speed(fallback_speed, network_type=self.network_type)

        logger.info(
            f"OSM road network set with parameters network_type '{self.network_type}' and distance_type '{self.distance_type}'"
        )

    def get_osm_road_network(self) -> nx.MultiDiGraph:
        """Get the processed OpenStreetMap road network for the administrative area.
        
        This method retrieves the OSM road network for the specified administrative area and processes it according to 
        the configured distance type. If the distance type is ``travel_time``, travel times are added to the network edges.
        
        Returns
        -------
        nx.MultiDiGraph
            NetworkX MultiDiGraph representing the road network with appropriate attributes:

                - If distance_type is ``length``, the graph has ``length`` attributes on edges (in meters)
                - If distance_type is ``travel_time``, the graph has ``travel_time`` attributes on edges (in minutes)
        """
        road_network = self._download_osm_road_network()

        if self.distance_type == "travel_time":
            return self._add_time_to_edges(road_network, self.fallback_speed)

        return road_network

    def _download_osm_road_network(self) -> nx.MultiDiGraph:
        """Download the OSM road network from OpenStreetMap for the administrative area.
        
        Returns
        -------
        nx.MultiDiGraph
            NetworkX MultiDiGraph representing the road network within the administrative area
        """
        return ox.graph_from_polygon(polygon=self.admin_area_boundaries, network_type=self.network_type)

    @staticmethod
    def _set_network_type(mode_of_transport: str) -> str:
        """Convert mode of transport to OSMnx network type.
    
        This method maps PISA's mode of transport terminology to the terminology used by the OSMnx library:
        - 'driving' -> 'drive'
        - 'walking' -> 'walk'
        - 'cycling' -> 'bike'

        Parameters
        ----------
        mode_of_transport : str
            The mode of transport (must be one of 'driving', 'walking', or 'cycling')
            
        Returns
        -------
        str
            The corresponding OSMnx network type ('drive', 'walk', or 'bike')
        """
        # validate mode of transport
        mode_of_transport = validate_mode_of_transport(mode_of_transport)

        transform_dct = {
            "driving": "drive",
            "walking": "walk",
            "cycling": "bike",
        }

        return transform_dct[mode_of_transport]

    @staticmethod
    def _add_time_to_edges(road_network: nx.MultiDiGraph, fallback_speed: int | float | None) -> nx.MultiDiGraph:
        """Add travel time attributes to the road network edges and convert to minutes.

        Parameters
        ----------
        road_network : nx.MultiDiGraph
            The road network graph to which travel times will be added
        fallback_speed : int, float, or None
            The default speed (in km/h) to use when speed limits are not available. If None, OSMnx's default values will
             be used
            
        Returns
        -------
        nx.MultiDiGraph
            The road network with travel times added to edges:
            - Edge attribute 'speed_kph' contains the speed in kilometers per hour
            - Edge attribute 'travel_time' contains the travel time in minutes
        """
        road_network = ox.add_edge_speeds(road_network, fallback=fallback_speed)
        road_network = ox.add_edge_travel_times(road_network)

        time = nx.get_edge_attributes(road_network, "travel_time")
        time_in_minutes = {k: round(v / 60, 2) for k, v in time.items()}
        nx.set_edge_attributes(road_network, time_in_minutes, "travel_time")

        return road_network
