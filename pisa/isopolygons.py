"""
This module provides functionality for calculating isopolygons around facilities using different methods and services.

An isopolygon represents the area that can be reached within a specific distance (isodistance) or time (isochrone) from a facility.

The module contains an abstract base class IsopolygonCalculator and its implementations.

Note:
    To implement a new way of calculating isopolygons (e.g., using Google Maps),
    create a class that inherits from IsopolygonCalculator and implements calculate_isopolygons.

"""

import json
import logging
import time
from abc import ABC, abstractmethod

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from geopandas import GeoSeries
from networkx import MultiDiGraph
from pandas import DataFrame
from shapely import Polygon
from shapely.geometry import shape

from pisa.utils import disk_cache, validate_distance_type, validate_mode_of_transport

logger = logging.getLogger(__name__)


class IsopolygonCalculator(ABC):
    """Abstract base class for isopolygon calculation."""

    def __init__(
        self,
        facilities_df: DataFrame,  # must have columns "longitude" and "latitude"
        distance_type: str,  # must be an element of VALID_DISTANCE_TYPES
        distance_values: int | list[int],  # in meters or minutes
    ):
        self.facilities_df = self._validate_facilities_df_format(facilities_df)

        self.distance_type = validate_distance_type(distance_type)

        self.distance_values = self._validate_distance_values_are_ints(distance_values)

        self._validate_distance_upper_limits()

    @staticmethod
    def _validate_facilities_df_format(facilities_df: DataFrame) -> DataFrame:
        """Checks that facilities_df has columns "longitude and latitude, and
        has one or more rows"""

        if (
            "longitude" not in facilities_df.columns
            or "latitude" not in facilities_df.columns
        ):
            raise ValueError(
                "facilities_df must have columns 'longitude' and 'latitude'"
            )

        if len(facilities_df) == 0:
            raise ValueError("facilities_df must have at least one row")

        return facilities_df

    @staticmethod
    def _validate_distance_values_are_ints(distance_values) -> list[int]:
        """
        Ensures that distance_values are in the correct format for calculating isopolygons.
        It converts single integer inputs into a list format.

        The requirement that all distance_values be integers comes from the Mapbox Isochrone API.

        Args:
            distance_values (int | list[int]): Either a single integer or a list of integers representing
                distances for isopolygon calculations.
        Returns:
            list[int]: A list containing the validated distance values. If input was a single integer,
                returns a single-element list.
        Raises:
            TypeError: If distance_values is neither an integer nor a list of integers.
            TypeError: If any element in the distance_values list is not an integer.

        """

        if isinstance(distance_values, int):
            return [distance_values]

        if isinstance(distance_values, list):
            if not all(isinstance(x, int) for x in distance_values):
                raise TypeError("distance_values must be a list of integers")
            return distance_values

        raise TypeError("distance_values must be a list of integers")

    def _validate_distance_upper_limits(self) -> None:
        """Checks that distance_values are within the permitted limits:
        100.000 meters for length and 60 minutes for time. This is requested by
        the Mapbox Isochrone API.
        """

        if self.distance_type == "length" and max(self.distance_values) > 100000:
            raise ValueError(
                "One or more distance values are larger than the permitted 100.000 meters limit."
            )

        if self.distance_type == "travel_time" and max(self.distance_values) > 60:
            raise ValueError(
                "One or more distance values are larger than the permitted 60 minutes limit."
            )

    @abstractmethod
    def calculate_isopolygons(self) -> DataFrame:
        """Must be implemented in subclasses"""
        pass


class OsmIsopolygonCalculator(IsopolygonCalculator):
    """This implementation of IsopolygonCalculator uses OpenStreetMap data to calculate isopolygons."""

    def __init__(
        self,
        facilities_df: DataFrame,
        distance_type: str,
        distance_values: list[int],
        road_network: MultiDiGraph,
        node_buffer: float = 0.001,
        edge_buffer: float = 0.0005,
    ):
        super().__init__(facilities_df, distance_type, distance_values)
        self.road_network = road_network
        self.node_buff = node_buffer
        self.edge_buff = edge_buffer

        # Find the nearest node in the road network for each facility
        self.nearest_nodes = ox.distance.nearest_nodes(
            G=self.road_network,
            X=self.facilities_df.longitude.values,
            Y=self.facilities_df.latitude.values,
        )

    def calculate_isopolygons(self) -> DataFrame:
        """Calculates isopolygons for each facility at different distances (distance_values).

        An isopolygon represents the area that can be reached within a specific distance/time from a facility
        using the road network.

        N.B.: distances will be computed with respect to the nearest node to the facility,
        not the facility itself."""

        index = pd.Index(self.nearest_nodes, name="nearest_node")
        columns = [f"ID_{d}" for d in self.distance_values]
        isopolygons = DataFrame(index=index, columns=columns)

        # Construct isopolygon for each distance value
        for distance_value in self.distance_values:

            for road_node in self.nearest_nodes:

                nodes_gdf, edges_gdf = self._get_skeleton_nodes_and_edges(
                    self.road_network, road_node, distance_value, self.distance_type
                )

                try:

                    new_isopolygon = self._add_buffer_to_isopolygon_skeleton(
                        nodes_gdf=nodes_gdf,
                        edges_gdf=edges_gdf,
                        node_buffer=self.node_buff,
                        edge_buffer=self.edge_buff,
                    )
                    isopolygons.loc[road_node, "ID_" + str(distance_value)] = (
                        new_isopolygon
                    )

                except (
                    AttributeError
                ):  # Probably trying to catch 'MultiPolygon' object has no attribute 'exterior' from _inflate_skeleton, but it wasn't specified in the code before

                    logger.info(f"problem with node {road_node}")  # stops execution

        return isopolygons

    @staticmethod
    def _add_buffer_to_isopolygon_skeleton(
        nodes_gdf: GeoSeries,
        edges_gdf: GeoSeries,
        node_buffer: float,
        edge_buffer: float,
    ) -> Polygon:
        """The nodes and edges form the "skeleton" of the isopolygon. This function turns that skeleton into
        a polygon by buffering and merging the geometries of the nodes and edges.

        Returns
        -------
        shapely.geometry.Polygon
            A polygon representing the merged buffer zones
        Raises
        ------
        ValueError
            If edge_buffer is less than or equal to 0
        AttributeError
            If the unary_union results in two (or more) disconnected polygons
        Notes
        -----
        - The function is sensitive to buffer size values. If they are too large relative to the distances between
        nodes, the buffer could be so large that unintended areas are included by mistake (like nodes that had been
        previously excluded).
        - Input geometries should be in a projected CRS, not geographic CRS, to ensure
          accurate buffer calculations."""

        if edge_buffer <= 0:
            raise ValueError("The parameter edge_buffer must be greater than 0.")

        # creates a circle with radius node_buffer around each node
        disks = nodes_gdf.buffer(node_buffer).geometry

        # creates a buffer around each edge. For example, if the edge is a line between
        # two points, it creates a sort of 2-d "cylinder" of "radius" edge_buffer
        # and semicircles of radius edge_buff at the end points
        cylinders = edges_gdf.buffer(edge_buffer).geometry

        all_geometries = list(disks) + list(cylinders)

        # Merges all_geometries into a single unified geometry, removing overlaps and simplifying the resulting shape
        geometric_union = gpd.GeoSeries(all_geometries).unary_union

        # Removes any interior boundaries (holes) that geometric_union might have had
        buffer_polygon = Polygon(geometric_union.exterior)

        return buffer_polygon

    @staticmethod
    def _get_skeleton_nodes_and_edges(
        road_network: nx.MultiDiGraph,
        center_node: int,
        distance_value: int,
        distance_type: str,
    ) -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
        """Get nodes and edges within a specified distance from a certain node in a road network.
        This will be the "skeleton" of the isopolygon.

        Parameters:
            road_network (nx.MultiDiGraph): The road network.
            center_node (int): The node from which to measure the distance.
            dist_value (int): The distance value.
            distance_type (str): The type of distance (e.g., 'length').

        Returns:
            nodes_gdf: a GeoSeries of the nodes with their osmid and geometry.
            edges_gdf: a GeoSeries of the geometry of the edges.


        Notes
        -----
        If an edge (u,v) doesn't have geometry data in the road_network, edges_gdf contains
        a straight line from u to v.

        If no edges are found (for example, if all other nodes are too far away from center_node),
        edges_gdf is an empty dataframe."""

        subgraph = nx.ego_graph(
            road_network, center_node, radius=distance_value, distance=distance_type
        )

        try:
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
            return (
                nodes_gdf.loc[:, "geometry"],
                edges_gdf.loc[:, "geometry"].reset_index(),
            )

        except ValueError:  # if no edges are found
            nodes_gdf = ox.graph_to_gdfs(subgraph, edges=False)
            return (
                nodes_gdf.loc[:, "geometry"],
                gpd.GeoSeries([], name="geometry", crs=nodes_gdf.crs),
            )


class OsmIsopolygonCalculatorAlternative(IsopolygonCalculator):
    """This implementation of IsopolygonCalculator uses OpenStreetMap data to calculate isopolygons."""

    def __init__(
        self,
        facilities_df: DataFrame,
        distance_type: str,
        distance_values: list[int],
        road_network: MultiDiGraph,
        buffer: float = 50,  # in meters
    ):
        super().__init__(facilities_df, distance_type, distance_values)
        self.road_network = road_network
        self.buffer = buffer

        # Find the nearest node in the road network for each facility
        self.nearest_nodes = ox.distance.nearest_nodes(
            G=self.road_network,
            X=self.facilities_df.longitude.values,
            Y=self.facilities_df.latitude.values,
        )

    def calculate_isopolygons(self) -> DataFrame:
        """
        Calculate isopolygons for each facility at different distances (distance_values).

        An isopolygon represents the area that can be reached within a specific distance/time from a facility
        using the road network.

        N.B.: distances will be computed with respect to the nearest node to the facility,
        not the facility itself.

        """

        index = pd.Index(self.nearest_nodes, name="nearest_node")
        columns = [f"ID_{d}" for d in self.distance_values]
        isopolygons = pd.DataFrame(index=index, columns=columns)

        for distance_value in self.distance_values:

            # Get skeletons for all nodes at this distance value
            skeletons = [
                self._get_skeleton_nodes_and_edges(
                    self.road_network, node, distance_value, self.distance_type
                )
                for node in self.nearest_nodes
            ]

            # "Inflate" skeletons to isopolygons at this distance value
            isopolygon_series = pd.Series(skeletons).apply(
                lambda x: ox.utils_geo.buffer_geometry(x, self.buffer)
            )

            # Assign column with isopolygons to the dataframe
            isopolygons[f"ID_{distance_value}"] = isopolygon_series.values

        return isopolygons

    @staticmethod
    def _get_skeleton_nodes_and_edges(
        road_network: nx.MultiDiGraph,
        center_node: int,
        distance_value: int,
        distance_type: str,
    ):
        """
        Get nodes and edges within distance_value from a node in the road network, and return
        the union of their geometries. This will be the "skeleton" of the isopolygon.

        Parameters:
            road_network (nx.MultiDiGraph): The road network.
            center_node (int): The node from which to measure the distance.
            distance_value (int): The distance value.
            distance_type (str): The type of distance (e.g., 'length').

        Returns:
            The union of the geometries of the nodes and edges within the specified distance from center_node.
        """
        subgraph = nx.ego_graph(
            road_network, center_node, radius=distance_value, distance=distance_type
        )

        try:
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)

            skeleton = (edges_gdf.geometry.union_all()).union(
                nodes_gdf.geometry.union_all()
            )

        except ValueError:  # if no edges are found
            nodes_gdf = ox.graph_to_gdfs(subgraph, edges=False)

            skeleton = nodes_gdf.geometry.union_all()

        return skeleton


class MapboxIsopolygonCalculator(IsopolygonCalculator):
    """From Mapbox docs: When you provide geographic coordinates to a Mapbox API,
    they should be formatted in the order longitude, latitude and specified as decimal degrees
    in the WGS84 coordinate system. This pattern matches existing standards, including GeoJSON and KML.
    Mapbox APIs use GeoJSON formatting wherever possible to represent geospatial data.
    """

    def __init__(
        self,
        facilities_df: DataFrame,
        distance_type: str,
        distance_values: list[int],  # in minutes or meters
        mode_of_transport: str,  # must be an element of VALID_TRANSPORT_MODES
        mapbox_api_token: str,
        base_url: str = "https://api.mapbox.com/isochrone/v1/",
    ):

        self.mapbox_api_token = self._validate_mapbox_token_not_empty(mapbox_api_token)

        self.route_profile = validate_mode_of_transport(mode_of_transport)

        super().__init__(facilities_df, distance_type, distance_values)

        self.distance_values = self._validate_mapbox_distance_values(
            self.distance_values
        )

        self.base_url = base_url

        self.contour_type = (
            "contours_meters" if self.distance_type == "length" else "contours_minutes"
        )

    def calculate_isopolygons(self) -> DataFrame:
        """Calculates isopolygons for all facilities using Mapbox API.

        This method generates isopolygons (polygons of equal distance/time) for each facility
        using the Mapbox Isochrone API.

        Returns:
            Dataframe: A pandas DataFrame where:
                - Each row represents a facility
                - Each column represents a distance value prefixed with "ID_"
                - Each cell contains the corresponding isopolygon geometry
        Note:
            - Requires valid Mapbox API credentials
            - Subject to Mapbox API rate limits (300 requests per minute)
            - Uses the distance values specified in self.distance_values
        """

        columns = [f"ID_{d}" for d in self.distance_values]

        # DataFrame with each row per facility and one column per distance
        isopolygons = DataFrame(index=self.facilities_df.index, columns=columns)

        # The Isochrone API supports 1 coordinate per request
        for i, (idx, facility) in enumerate(self.facilities_df.iterrows()):
            self._handle_rate_limit(request_count=i)

            request_url = self._build_request_url(facility.longitude, facility.latitude)
            features = self._fetch_isopolygons(request_url)

            for feature in features:
                isopolygon = shape(feature["geometry"])

                # countour is a distance value (e.g. 1000 (for distance) or 60 (for time))
                contour = feature["properties"]["contour"]

                # TODO: can we remove this "ID_" prefix?
                isopolygons.at[idx, f"ID_{contour}"] = isopolygon

        return isopolygons

    def _build_request_url(self, longitude: float, latitude: float) -> str:
        """Builds the Mapbox API request URL for isopolygon calculation."""
        return (
            f"{self.base_url}mapbox/{self.route_profile}/{longitude},"
            f"{latitude}?{self.contour_type}={','.join(map(str, self.distance_values))}"
            f"&polygons=true&denoise=1&access_token={self.mapbox_api_token}"
        )

    def _handle_rate_limit(self, request_count: int) -> None:
        """Handles Mapbox API rate limiting, a maximum of 300 requests per minute."""

        if (request_count + 1) % 300 == 0:
            logger.info("Reached Mapbox API request limit. Waiting for 1 minute...")
            time.sleep(60)
            logger.info("Resuming requests")

    @disk_cache("mapbox_cache")
    def _fetch_isopolygons(self, request_url: str) -> list:
        """
        Makes a GET request to the Mapbox Isochrone API endpoint and handles various potential errors.

        Args:
            request_url (str): The complete URL for the Mapbox Isochrone API request.

        Returns:
            list: GeoJSON Feature object.

        Raises:
            ValueError: If the Mapbox access token is invalid (401 error).
            PermissionError: If the token lacks permission to access the resource (403 error).
            requests.exceptions.HTTPError: For other HTTP-related errors.
            TimeoutError: If the request times out (>60 seconds).
            RuntimeError: For unexpected errors during the API request.
        """

        try:
            # Make the request
            response = requests.get(request_url, timeout=60)

            # Check for HTTP errors
            response.raise_for_status()

            # Try to parse the JSON
            request_pack = json.loads(response.content)

            # Check if features exist in the response
            if "features" not in request_pack:
                raise KeyError(
                    "Response does not contain 'features' key. API may have changed or returned unexpected format."
                )

            return request_pack["features"]

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP status codes
            if response.status_code == 401:
                raise ValueError("Unauthorized: Invalid Mapbox access token") from e
            elif response.status_code == 403:
                raise PermissionError(
                    "Forbidden: The Mapbox token doesn't have access to this resource"
                ) from e
            else:
                raise requests.exceptions.HTTPError(
                    f"HTTP Error {response.status_code}: {e}"
                ) from e

        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                "Request timed out: Mapbox servers took too long to respond."
            ) from e

        except Exception as e:
            # Last resort for unexpected errors
            logger.error(f"Unexpected error in Mapbox API request: {e}", exc_info=True)
            raise RuntimeError(
                f"Unexpected error when connecting to Mapbox: {str(e)}"
            ) from e

    @staticmethod
    def _validate_mapbox_distance_values(distance_values: list[int]) -> list[int]:
        """Checks if distance_values meet Mapbox API requirements:
        a maximum of 4 values in increasing order.

        Args:
            distance_values (list[int]): List of integer distances to validate

        Returns:
            list[int]: Sorted list of distance values

        Raises:
            ValueError: If more than 4 distance values are provided

        """

        if len(distance_values) > 4:
            raise ValueError("Mapbox API accepts a maximum of 4 distance_values")

        distance_values.sort()

        return distance_values

    @staticmethod
    def _validate_mapbox_token_not_empty(mapbox_api_token: str) -> str:
        if not mapbox_api_token:
            raise ValueError("Mapbox API token is required")
        return mapbox_api_token
