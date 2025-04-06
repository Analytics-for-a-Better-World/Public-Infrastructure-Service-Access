import networkx as nx
import osmnx as ox
from shapely import MultiPolygon, Polygon

from pisa.utils import _validate_distance_type, _validate_mode_of_transport


class OsmRoadNetwork:
    """TODO:
    - add docstring
    - make sure the road_network part of pisa_showcase works
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
        self.network_type = self._get_network_type(mode_of_transport)

        self.admin_area_boundaries = admin_area_boundaries

    def get_osm_road_network(self) -> nx.MultiDiGraph:
        """TODO: add docstring"""

        road_network = self._download_osm_road_network()

        if self.distance_type == "travel_time":
            return self._add_time_to_edges(road_network)

        return road_network

    def _download_osm_road_network(self) -> nx.MultiDiGraph:
        """TODO: add docstring"""

        return ox.graph_from_polygon(
            polygon=self.admin_area_boundaries, network_type=self.network_type
        )

    @staticmethod
    def _get_network_type(mode_of_transport: str) -> str:
        """

        TODO: implement, add docstring and tests

        What it should do: takes in a mode_of_transport in VALID_MODE_OF_TRANSPORT and
        converts it to a valid argument for network_type in osmnx.graph_from_polygon

        For example: driving -> drive
        """

        ...

    @staticmethod
    def _add_time_to_edges(road_network: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """TODO: implement, add docstring and tests"""

        ...
