import networkx as nx
import osmnx as ox
from shapely import MultiPolygon, Polygon

from pisa.utils import _validate_distance_type, _validate_mode_of_transport


class OsmRoadNetwork:
    """
    Class to retrieve and process OpenStreetMap road network data.

    Parameters
    ----------
    admin_area_boundaries: Polygon | MultiPolygon
        The geography of the administrative area object.

    mode_of_transport: str
        The mode of transport for which the road network is required.
        Valid inputs : 'driving', 'walking', 'cycling'

    distance_type: str
        The type of distance to be calculated.
        Valid inputs : 'length', 'travel_time'
    """

    def __init__(
        self,
        admin_area_boundaries: Polygon | MultiPolygon,
        mode_of_transport: str,  # must be an element of VALID_MODES_OF_TRANSPORT
        distance_type: str,  # must be an element of VALID_DISTANCE_TYPES
    ):
        # validate distance type
        self.distance_type = _validate_distance_type(distance_type)

        # validate mode of transport
        mode_of_transport = _validate_mode_of_transport(mode_of_transport)

        # process mode of transport
        self.network_type, self.default_speed = self._get_network_type(mode_of_transport)

        self.admin_area_boundaries = admin_area_boundaries

    def get_osm_road_network(self) -> nx.MultiDiGraph:
        """Process the OSM road network."""

        road_network = self._download_osm_road_network()

        if self.distance_type == "travel_time":
            return self._add_time_to_edges(road_network, self.default_speed)

        return road_network

    def _download_osm_road_network(self) -> nx.MultiDiGraph:
        """Download the OSM road network from OpenStreetMap for the specified administrative area."""

        return ox.graph_from_polygon(
            polygon=self.admin_area_boundaries, network_type=self.network_type
        )

    @staticmethod
    def _get_network_type(mode_of_transport: str) -> tuple[str, int]:
        """Set valid network type and default speed based on input"""
        network_mapping = {
            "driving": ("drive", 50),
            "walking": ("walk", 4),
            "cycling": ("bike", 15),
        }
        return network_mapping[mode_of_transport]

    @staticmethod
    def _add_time_to_edges(
        road_network: nx.MultiDiGraph, default_speed: int
    ) -> nx.MultiDiGraph:
        """Add travel time edge attribute and change unit to minutes"""
        road_network = ox.add_edge_speeds(road_network, fallback=default_speed)
        road_network = ox.add_edge_travel_times(road_network)

        time = nx.get_edge_attributes(road_network, "travel_time")
        time_in_minutes = {k: round(v / 60, 2) for k, v in time.items()}
        nx.set_edge_attributes(road_network, time_in_minutes, "travel_time")

        return road_network
