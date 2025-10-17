"""Isopolygon calculation module for service area analysis.

This module provides functionality for calculating isopolygons around facilities using different methods and services.
An isopolygon represents the area that can be reached within a specific distance (isodistance) or time (isochrone) from a
facility.

The module contains an abstract base class IsopolygonCalculator and its implementations for different calculation
methods.

Examples
--------
Calculate isochrones around facilities using OpenStreetMap:

>>> from pisa.administrative_area import AdministrativeArea
>>> from pisa.facilities import Facilities
>>> from pisa.osm_road_network import OsmRoadNetwork
>>> from pisa.isopolygons import OsmIsopolygonCalculator
>>>
>>> # Get administrative area and facilities
>>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
>>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
>>> facilities = Facilities(admin_area_boundaries=boundaries)
>>> existing_facilities = facilities.get_existing_facilities()
>>>
>>> # Create a road network for travel time calculations
>>> road_network = OsmRoadNetwork(
>>>     admin_area_boundaries=boundaries,
>>>     mode_of_transport="walking",
>>>     distance_type="travel_time"
>>> )
>>> graph = road_network.get_osm_road_network()
>>>
>>> # Calculate isochrones (5, 10, 15 minutes walking)
>>> isopolygon_calculator = OsmIsopolygonCalculator(
>>>     facilities_df=existing_facilities,
>>>     distance_type="travel_time",
>>>     distance_values=[5, 10, 15],
>>>     road_network=graph
>>> )
>>> isopolygons = isopolygon_calculator.calculate_isopolygons()

Note:
    To implement a new way of calculating isopolygons (e.g., using Google Maps), create a class that inherits from
    IsopolygonCalculator and implements calculate_isopolygons.

See Also
--------
facilities : Module for retrieving facility locations
osm_road_network : Module for retrieving and processing road networks
population_served_by_isopolygons : Module for analyzing population coverage
"""

import json
import logging
import time
from abc import ABC, abstractmethod

import geopandas as gpd
import networkx as nx
import numpy as np
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
        """Validate that facilities DataFrame has the required format.

        This validation ensures that the facilities DataFrame contains the minimum required information for isopolygon
        calculation: longitude and latitude coordinates for at least one facility.

        Parameters
        ----------
        facilities_df : pandas.DataFrame
            DataFrame containing facility locations to validate

        Returns
        -------
        pandas.DataFrame
            The validated DataFrame (unchanged)

        Raises
        ------
        ValueError
            If facilities_df is missing required columns 'longitude' and 'latitude'
        ValueError
            If facilities_df has no rows (empty DataFrame)
        """
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
        """Ensure that distance values are integers and properly formatted as a list.

        Parameters
        ----------
        distance_values : int or list[int]
            Either a single integer or a list of integers representing distances for isopolygon calculations

        Returns
        -------
        list[int]
            A list containing the validated distance values. If input was a single integer, returns a single-element
            list.

        Raises
        ------
        TypeError
            If distance_values is neither an integer nor a list of integers
        TypeError
            If any element in the distance_values list is not an integer

        Notes
        -----
        The requirement that all distance values be integers comes from the Mapbox Isochrone API,
        but is applied to all implementations for consistency.
        """
        if isinstance(distance_values, int):
            return [distance_values]

        if isinstance(distance_values, list):
            if not all(isinstance(x, int) for x in distance_values):
                raise TypeError("distance_values must be a list of integers")
            return distance_values

        raise TypeError("distance_values must be a list of integers")

    def _validate_distance_upper_limits(self) -> None:
        """Validate that distance values are within permitted limits.

        This method checks that all distance values are within the API-imposed limits:
        - For 'length' type: maximum of 100,000 meters
        - For 'travel_time' type: maximum of 60 minutes

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any distance value exceeds the maximum limit for its distance type

        Notes
        -----
        These limits are imposed by the Mapbox Isochrone API specifications. Even when using other providers (e.g., OSM),
         the same limits are applied for consistency across implementations.
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
        """Calculate isopolygons for the specified facilities.

        This abstract method must be implemented by subclasses to provide the actual implementation of isopolygon
        calculation using specific data sources and algorithms.

        Specific implementations should provide detailed error handling and logging appropriate to their data sources and
        algorithms.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing isopolygons with the following structure:

                - Each row represents a facility from facilities_df
                - One column named ``ID_{distance}`` for each distance value in distance_values, where {distance} is the
                  distance value in meters or minutes
                - Each cell contains a Shapely ``Polygon`` or ``MultiPolygon`` representing the area that can be reached
                  within the corresponding distance

        Raises
        ------
        NotImplementedError
            If this method is not implemented by a subclass
        """
        pass


class OsmIsopolygonCalculator(IsopolygonCalculator):
    """OpenStreetMap-based implementation of isopolygon calculation.

    This implementation uses OpenStreetMap road network data to calculate isopolygons (isochrones or isodistances)
    around facilities. It leverages the NetworkX and OSMnx libraries for network analysis.

    This implementation performs network-based calculations on an OSM road network, which provides accurate results but
    may be computationally intensive for large areas or many facilities.

    Parameters
    ----------
    facilities_df : pandas.DataFrame
        DataFrame containing facility locations with ``longitude`` and ``latitude`` columns
    distance_type : str
        Type of distance to calculate (``length`` or ``travel_time``)
    distance_values : list[int]
        List of distance values in meters (for ``length``) or minutes (for ``travel_time``)
    road_network : networkx.MultiDiGraph
        Road network graph to use for calculations
    node_buffer : float, optional
        Buffer distance to apply around network nodes. (default: ``0.001``)
    edge_buffer : float, optional
        Buffer distance to apply around network edges. (default: ``0.0005``)

    See Also
    --------
    IsopolygonCalculator : Abstract base class
    MapboxIsopolygonCalculator : Mapbox API-based implementation
    """

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

        # Find the nearest node in the road network and create a dictionary to store the match with the respective facility
        nearest_nodes, distance_to_nearest_nodes = ox.distance.nearest_nodes(
            G=self.road_network,
            X=self.facilities_df.longitude.values,
            Y=self.facilities_df.latitude.values,
            return_dist=True,
        )
        self._warn_facilities_too_far_away(nearest_nodes, distance_to_nearest_nodes)

        self.nearest_nodes_dict = {
            facility_id: node
            for facility_id, node in zip(self.facilities_df.index, nearest_nodes)
        }

    def calculate_isopolygons(self) -> DataFrame:
        """Calculate isopolygons for each facility at different distances.

        This method generates isopolygons (areas reachable within specific travel times or distances)
        for each facility using the OpenStreetMap road network data.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing isopolygons with the following structure:

                - Each row represents a facility from facilities_df
                - One column named ``ID_{distance}`` for each distance value in distance_values,
                  where {distance} is the distance value in meters or minutes
                - Each cell contains a Shapely ``Polygon`` or ``MultiPolygon`` representing
                  the area that can be reached within the corresponding distance

        Notes
        -----
        - Distances are computed with respect to the nearest node to the facility,
          not the facility itself
        - The method creates network "skeletons" and then buffers nodes and edges
          separately with different buffer sizes to create accurate isopolygons
        """
        # Initialize an empty DataFrame to store isopolygons
        index = self.nearest_nodes_dict.keys()
        columns = [f"ID_{d}" for d in self.distance_values]
        isopolygons = DataFrame(index=index, columns=columns)
        isopolygons.index.name = self.facilities_df.index.name

        # Construct isopolygon for each distance value
        for distance_value in self.distance_values:
            for facility_id, road_node in self.nearest_nodes_dict.items():
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
                    isopolygons.loc[facility_id, "ID_" + str(distance_value)] = (
                        new_isopolygon
                    )

                except (
                    AttributeError
                ):  # Probably trying to catch 'MultiPolygon' object has no attribute 'exterior' from _inflate_skeleton, but it wasn't specified in the code before
                    logger.info(
                        f"problem with node {road_node} belonging to facility {facility_id}"
                    )  # stops execution

        return isopolygons

    @staticmethod
    def _add_buffer_to_isopolygon_skeleton(
        nodes_gdf: GeoSeries,
        edges_gdf: GeoSeries,
        node_buffer: float,
        edge_buffer: float,
    ) -> Polygon:
        """Convert an isopolygon skeleton into a proper polygon by buffering.

        This method takes the "skeleton" of an isopolygon (nodes and edges from the road network) and turns it into a
        proper polygon by buffering and merging the geometries.

        Parameters
        ----------
        nodes_gdf : geopandas.GeoSeries
            GeoSeries containing node geometries (points)
        edges_gdf : geopandas.GeoSeries
            GeoSeries containing edge geometries (lines)
        node_buffer : float
            Buffer distance to apply around nodes
        edge_buffer : float
            Buffer distance to apply around edges

        Returns
        -------
        shapely.geometry.Polygon
            A polygon representing the merged buffer zones

        Raises
        ------
        ValueError
            If edge_buffer is less than or equal to 0
        AttributeError
            If the union_all results in two (or more) disconnected polygons

        Notes
        -----
        - The function is sensitive to buffer size values. If they are too large relative to the distances between
          nodes, the buffer could be so large that unintended areas are included by mistake.
        - Input geometries should be in a projected CRS, not geographic CRS, to ensure accurate buffer calculations.
        - Node buffers create circles, while edge buffers create rounded rectangles along the edges
        """

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
        geometric_union = gpd.GeoSeries(all_geometries).union_all()

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
        """Get nodes and edges within a specified distance from a center node.

        This method extracts all nodes and edges within a specified distance from a center node
        in a road network, returning them as separate GeoSeries to form the "skeleton" of an isopolygon.

        Parameters
        ----------
        road_network : nx.MultiDiGraph
            The road network graph
        center_node : int
            The node ID from which to measure distance
        distance_value : int
            The maximum distance value (in meters for 'length', minutes for 'travel_time')
        distance_type : str
            The type of distance to use ('length' or 'travel_time')

        Returns
        -------
        tuple[gpd.GeoSeries, gpd.GeoSeries]
            A tuple containing:
            - nodes_gdf: GeoSeries of node geometries
            - edges_gdf: GeoSeries of edge geometries

        Notes
        -----
        - If an edge doesn't have geometry data in the road_network, a straight line
          from the source to target node is used instead
        - If no edges are found (e.g., if all other nodes are beyond the distance threshold),
          edges_gdf will be an empty GeoSeries
        - The method uses NetworkX's ego_graph to extract the subgraph within the specified distance
        """
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

    def _warn_facilities_too_far_away(self, nearest_nodes, distance_to_nearest_nodes):
        """Logs a warning if any facilities are found more than 10km from nearest road node,
        suggesting manual inspection may be needed to verify results.
        """
        if any(distance_to_nearest_nodes > 10000):

            high_distance_indices = np.where(distance_to_nearest_nodes > 10000)
            far_from_road_facilities = self.facilities_df.iloc[high_distance_indices]
            far_from_road_facilities["nearest_road_node"] = nearest_nodes[
                high_distance_indices
            ]

            logger.warning(
                f"""Some facilities are more than 10 km away from the nearest node on the OSM road network. 
                The facilities and their nearest nodes are: {far_from_road_facilities[['nearest_road_node']]} \n 
                It makes sense to visually inspect these in a notebook or compare your results with the Mapbox API."""
            )


class MapboxIsopolygonCalculator(IsopolygonCalculator):
    """Mapbox-based implementation of isopolygon calculation.

    This implementation uses the Mapbox Isochrone API to calculate isopolygons
    (isochrones or isodistances) around facilities.

    Parameters
    ----------
    facilities_df : pandas.DataFrame
        DataFrame containing facility locations with ``longitude`` and ``latitude`` columns
    distance_type : str
        Type of distance to calculate (``length`` or ``travel_time``)
    distance_values : list[int]
        List of distance values in meters (for ``length``) or minutes (for ``travel_time``)
        Maximum of 4 values allowed by the Mapbox API
    mode_of_transport : str
        The mode of transport to use (must be one of ``driving``, ``walking``, ``cycling``)
    mapbox_api_token : str
        A valid Mapbox API access token with Isochrone API permissions
    base_url : str, optional
        The base URL for the Mapbox Isochrone API, default is 'https://api.mapbox.com/isochrone/v1/'

    See Also
    --------
    IsopolygonCalculator : Abstract base class
    OsmIsopolygonCalculator : OSM-based implementation with precise node/edge buffering

    Notes
    -----
    From Mapbox docs: When providing geographic coordinates to a Mapbox API,
    they should be formatted in the order longitude, latitude and specified as decimal degrees
    in the WGS84 coordinate system. This pattern matches existing standards, including GeoJSON and KML.

    This implementation is subject to Mapbox API rate limits and requires a valid Mapbox account
    and access token.
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
        """Calculate isopolygons for all facilities using the Mapbox API.

        This method generates isopolygons (areas reachable within specific travel times or distances)
        for each facility using the Mapbox Isochrone API.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing isopolygons with the following structure:

                - Each row represents a facility from facilities_df
                - One column named ``ID_{distance}`` for each distance value in distance_values,
                  where {distance} is the distance value in meters or minutes
                - Each cell contains a Shapely ``Polygon`` or ``MultiPolygon`` representing
                  the area that can be reached within the corresponding distance

        Notes
        -----
        - Requires a valid Mapbox API token with appropriate permissions
        - Subject to Mapbox API rate limits (300 requests per minute)
        - Makes one API request per facility (not per distance value)
        - The API returns GeoJSON features that are converted to Shapely geometries
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
        """Build the Mapbox API request URL for isopolygon calculation.

        Parameters
        ----------
        longitude : float
            The longitude coordinate of the facility
        latitude : float
            The latitude coordinate of the facility

        Returns
        -------
        str
            The complete URL for the Mapbox Isochrone API request
        """
        return (
            f"{self.base_url}mapbox/{self.route_profile}/{longitude},"
            f"{latitude}?{self.contour_type}={','.join(map(str, self.distance_values))}"
            f"&polygons=true&denoise=1&access_token={self.mapbox_api_token}"
        )

    def _handle_rate_limit(self, request_count: int) -> None:
        """Handle Mapbox API rate limiting.

        The Mapbox API has a rate limit of 300 requests per minute. This method pauses execution for 60 seconds when
        reaching this limit to avoid rate limit errors.

        Parameters
        ----------
        request_count : int
            The current count of API requests that have been made

        Returns
        -------
        None
        """
        if (request_count + 1) % 300 == 0:
            logger.info("Reached Mapbox API request limit. Waiting for 1 minute...")
            time.sleep(60)
            logger.info("Resuming requests")

    @disk_cache("mapbox_cache")
    def _fetch_isopolygons(self, request_url: str) -> list:
        """Fetch isopolygon data from the Mapbox Isochrone API.

        This method makes a GET request to the Mapbox Isochrone API and handles various potential errors that might
        occur during the request.

        Parameters
        ----------
        request_url : str
            The complete URL for the Mapbox Isochrone API request

        Returns
        -------
        list
            List of GeoJSON Feature objects representing isopolygons

        Raises
        ------
        ValueError
            If the Mapbox access token is invalid (HTTP 401 error)
        PermissionError
            If the token lacks permission to access the resource (HTTP 403 error)
        requests.exceptions.HTTPError
            For other HTTP-related errors
        TimeoutError
            If the request times out (>60 seconds)
        RuntimeError
            For unexpected errors during the API request

        Notes
        -----
        This method uses the disk_cache decorator to avoid making repeated requests for the same URL, which helps reduce
         API usage and improves performance for repeated calculations.
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
        """Validate distance values against Mapbox API requirements.

        The Mapbox Isochrone API accepts a maximum of 4 distinct contour values, which must be provided in ascending
        order. This method validates the input against these constraints and sorts the values.

        Parameters
        ----------
        distance_values : list[int]
            List of integer distances to validate

        Returns
        -------
        list[int]
            Sorted list of distance values in ascending order

        Raises
        ------
        ValueError
            If more than 4 distance values are provided (Mapbox API limitation)
        """
        if len(distance_values) > 4:
            raise ValueError("Mapbox API accepts a maximum of 4 distance_values")

        distance_values.sort()

        return distance_values

    @staticmethod
    def _validate_mapbox_token_not_empty(mapbox_api_token: str) -> str:
        """Validate that the Mapbox API token is not empty.

        Parameters
        ----------
        mapbox_api_token : str
            The Mapbox API token to validate

        Returns
        -------
        str
            The validated Mapbox API token (unchanged)

        Raises
        ------
        ValueError
            If the Mapbox API token is empty

        Notes
        -----
        This simple validation ensures that an API token has been provided.
        It does not verify that the token is valid or has the necessary permissions.
        """
        if not mapbox_api_token:
            raise ValueError("Mapbox API token is required")
        return mapbox_api_token
