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

import requests
from isopolygons import IsopolygonCalculator
from pandas import DataFrame
from shapely.geometry import shape

from pisa.utils import disk_cache

logger = logging.getLogger(__name__)


class MapboxIsopolygonCalculator(IsopolygonCalculator):
    """From Mapbox docs: When you provide geographic coordinates to a Mapbox API,
    they should be formatted in the order longitude, latitude and specified as decimal degrees
    in the WGS84 coordinate system. This pattern matches existing standards, including GeoJSON and KML.
    Mapbox APIs use GeoJSON formatting wherever possible to represent geospatial data.
    """

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,  # either travel_time or length (meters). TODO: Force?
        distance_values: list[
            int
        ],  # TODO: minutes or meters. Max 4, in ascending order
        route_profile: str,  # ? driving, walking or cycling. TODO: Force?
        mapbox_api_token: str,
        base_url: str = "https://api.mapbox.com/isochrone/v1/",
    ):
        super().__init__(facilities_lon_lat, distance_type, distance_values)
        self.route_profile = route_profile
        self.mapbox_api_token = mapbox_api_token
        self.base_url = base_url

        if not self.mapbox_api_token:
            raise ValueError("Mapbox API token is required.")

        # TODO: implement (from mapbox docs)
        # You can specify up to four contours. Times must be in increasing order. The maximum time that can be specified is 60 minutes.
        # You can specify up to four contours. Distances must be in increasing order. The maximum distance that can be specified is 100000 meters (100km).

        self.contour_type = self._set_countour_type(self.distance_type)

        self.facilities_lon_lat = facilities_lon_lat

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
        number_of_facilities = len(self.facilities_lon_lat)

        # DataFrame with each row per facility and one column per distance
        isopolygons = DataFrame(index=range(number_of_facilities), columns=columns)

        # The Isochrone API supports 1 coordinate per request
        for idx, facility in self.facilities_lon_lat.iterrows():

            # The Isochrone API is limited to 300 requests per minute
            self._handle_rate_limit(request_count=idx)

            request_url = self._build_request_url(facility.longitude, facility.latitude)
            features = self._fetch_isopolygons(request_url)

            for feature in features:
                isopolygon = shape(feature["geometry"])

                # countour is a distance value (e.g. 1000 (for distance) or 60 (for time))
                contour = feature["properties"]["contour"]

                # TODO: can we remove this "ID_" prefix?
                isopolygons.at[idx, f"ID_{contour}"] = isopolygon

        return isopolygons

    @staticmethod
    def _set_countour_type(distance_type: str) -> str:

        # todo: force distance_type to be either travel_time or length

        """Determines countour_type (Mapbox readable) according to distance_type (given by user)"""
        if distance_type == "travel_time":
            return "contours_minutes"
        if distance_type == "length":
            return "contours_meters"
        raise ValueError("Distance type must be either 'travel_time' or 'length'.")

    def _build_request_url(self, longitude: float, latitude: float) -> str:
        """Builds the Mapbox API request URL for isopolygon calculation."""
        return (
            f"{self.base_url}mapbox/{self.route_profile}/{longitude},"
            f"{latitude}?{self.contour_type}={','.join(map(str, self.distance_values))}"
            f"&polygons=true&denoise=1&access_token={self.mapbox_api_token}"
        )

    def _handle_rate_limit(self, request_count: int) -> None:
        """Handles Mapbox API rate limiting."""
        if (request_count + 1) % 300 == 0:
            logger.info("Reached Mapbox API request limit. Waiting for 5 minutes...")
            time.sleep(300)
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
            logging.error(f"Unexpected error in Mapbox API request: {e}", exc_info=True)
            raise RuntimeError(
                f"Unexpected error when connecting to Mapbox: {str(e)}"
            ) from e
