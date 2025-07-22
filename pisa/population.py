"""Population data retrieval and processing module for geographic analysis.

This module provides classes and functions for retrieving, processing, and analyzing population data from various 
sources such as Facebook's Data for Good and WorldPop. It includes an abstract base class that defines the common 
interface, and concrete implementations for specific data sources.

The module supports retrieving population data within specified administrative boundaries, aggregating the data at
different resolutions, and preparing it for accessibility analysis with facilities.

Examples
--------
Retrieve and process population data from WorldPop:

>>> from pisa.administrative_area import AdministrativeArea
>>> from pisa.population import WorldpopPopulation
>>>
>>> # Get administrative area boundaries
>>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
>>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
>>> country_code = admin_area.get_iso3_country_code()
>>>
>>> # Create a population object and retrieve data
>>> population = WorldpopPopulation(
>>>     admin_area_boundaries=boundaries,
>>>     iso3_country_code=country_code
>>> )
>>>
>>> # Get processed population data as a GeoDataFrame
>>> population_gdf = population.get_population_gdf()
>>> print(f"Total population: {population_gdf['population'].sum()}")

See Also
--------
administrative_area : Module for retrieving administrative area boundaries
facilities : Module for working with facility location data
"""

import urllib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import requests
from geopandas import GeoDataFrame, clip, points_from_xy
from hdx.api.configuration import Configuration
from hdx.data.resource import Resource
from rasterio.mask import mask
from shapely import MultiPolygon, Polygon


@dataclass
class Population(ABC):
    """Abstract base class for Population data retrieval and processing.
    
    This class provides the core functionality for retrieving and processing population data for a specified 
    administrative area. Subclasses must implement the method get_population_data() to support different data sources.
    
    Parameters
    ----------
    admin_area_boundaries : Polygon or MultiPolygon
        The geographical boundaries of the administrative area for which to retrieve population data
    iso3_country_code : str
        The ISO3 country code for the administrative area (e.g., ``TLS`` for Timor-Leste)
    population_resolution : int, optional
        The decimal precision to which latitude and longitude coordinates are rounded for aggregation purposes 
        (default: ``5``)
    """

    admin_area_boundaries: Polygon | MultiPolygon
    iso3_country_code: str
    population_resolution: int = 5

    def get_population_gdf(self) -> GeoDataFrame:
        """Get aggregated population data for the administrative area as a GeoDataFrame.

        This method integrates the population data retrieval workflow by:

            1. Retrieving raw population data using the specific implementation of the get_population_data() method
            2. Aggregating the population data based on the specified resolution
        
        Returns
        -------
        geopandas.GeoDataFrame
            Aggregated population data with columns:

                - ``longitude``: Rounded longitude coordinate
                - ``latitude``: Rounded latitude coordinate
                - ``population``: Total population at the coordinate
                - ``geometry``: Point geometry representing the coordinate
                - ``ID``: Unique identifier for each point
        """
        population_df = self._get_population_df()
        return self._group_population(population_df, self.population_resolution)

    @staticmethod
    def _group_population(
        population_df: pd.DataFrame, population_resolution: int
    ) -> GeoDataFrame:
        """Group population data by coordinates based on a specified resolution.
        
        This method aggregates population data by rounding longitude and latitude coordinates to a specified decimal 
        precision (population_resolution), then grouping by these rounded coordinates. For each unique coordinate pair,
        the population values are summed.
        
        Parameters
        ----------
        population_df : pandas.DataFrame
            DataFrame containing population data with at least the columns:
            - longitude: Longitude coordinate
            - latitude: Latitude coordinate
            - population: Population count
            
        population_resolution : int
            Number of decimal places to round the coordinates to
            
        Returns
        -------
        geopandas.GeoDataFrame
            Aggregated population data with columns:
            - longitude: Rounded longitude coordinate
            - latitude: Rounded latitude coordinate
            - population: Total population for the coordinate
            - geometry: Point geometry created from the coordinates
        
        Notes
        -----
        This method creates point geometries using the rounded coordinates and assigns a unique ID to each point based 
        on its position in the dataframe.
        """
        population_df.loc[:, ["longitude", "latitude"]] = population_df[
            ["longitude", "latitude"]
        ].round(population_resolution)

        population = (
            population_df.groupby(["longitude", "latitude"], as_index=False)[
                "population"
            ]
            .sum()
        )
        population["population"] = population["population"].round(2)
        population = GeoDataFrame(
            population,
            geometry=points_from_xy(population.longitude, population.latitude),
        )
        return population

    @abstractmethod
    def _get_population_df(self) -> pd.DataFrame:
        """Get population data from a specific data source.
        
        This abstract method must be implemented by subclasses to provide population data retrieval from specific sources
         (e.g., Facebook, WorldPop).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing population data with columns:
            - longitude: Longitude coordinate
            - latitude: Latitude coordinate
            - population: Population count at the coordinate
            
        Raises
        ------
        NotImplementedError
            If this method is not implemented by a subclass
        """
        pass


class FacebookPopulation(Population):
    """Population data from Facebook's High Resolution Population Density Maps.
    
    This class retrieves and processes population data from Facebook's Data for Good program, which provides 
    high-resolution population density maps. The data is accessed via the Humanitarian Data Exchange (HDX) platform.
    
    The class follows the same initialization pattern as its parent class Population.
    
    See Also
    --------
    Population : Parent abstract class
    WorldPopulation : Alternative population data source implementation
    """

    def _get_population_df(self) -> pd.DataFrame:
        """Download and process population data from Facebook.
        
        Implements the abstract method from the Population class to retrieve population data specifically from Facebook's
         High Resolution Population Density Maps via the HDX platform.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with population data containing columns:
            - longitude: Longitude coordinate
            - latitude: Latitude coordinate
            - population: Population count at the coordinate
        """
        downloaded_data = self.download_population_facebook(
            iso3_country_code=self.iso3_country_code,
        )

        processed_data = self.process_population_facebook(
            downloaded_data,
            iso3_country_code=self.iso3_country_code,
            admin_area_boundaries=self.admin_area_boundaries,
        )

        return processed_data

    @staticmethod
    def download_population_facebook(iso3_country_code: str) -> pd.DataFrame:
        """Download Facebook population data for a specific country.
        
        This method retrieves population data from Facebook's High Resolution Population Density Maps via the 
        Humanitarian Data Exchange (HDX) platform for a country specified by its ISO3 code.
        
        Parameters
        ----------
        iso3_country_code : str
            The ISO3 country code for which to download population data (e.g., ``TLS`` for Timor-Leste)
        
        Returns
        -------
        pandas.DataFrame
            Raw population data from Facebook for the specified country
            
        Raises
        ------
        Exception
            If there are issues with the HDX configuration or data download
            
        Notes
        -----
        The method accesses the Facebook population dataset from the year 2020.
        """
        try:
            Configuration.create(
                hdx_site="prod", user_agent="Get_Population_Data", hdx_read_only=True
            )
        except Exception as e:
            print(f"Warning: Unable to configure HDX. Error: {e}")

        resource = Resource.search_in_hdx(
            f"name:{iso3_country_code.lower()}_general_2020_csv.zip"
        )
        if not resource:
            raise ValueError(f"No resource found for country code: {iso3_country_code}")

        url = resource[0]["download_url"]
        filehandle, _ = urllib.request.urlretrieve(url)

        df = pd.read_csv(filehandle, compression="zip")
        return df

    @staticmethod
    def process_population_facebook(
        downloaded_data: pd.DataFrame,
        iso3_country_code: str,
        admin_area_boundaries: Polygon | MultiPolygon,
    ) -> pd.DataFrame:
        """Process Facebook population data for a specific administrative area.
        
        This method transforms raw Facebook population data into a filtered dataset that only includes points within the 
        specified administrative area boundaries.
        
        Parameters
        ----------
        downloaded_data : pandas.DataFrame
            Raw population data downloaded from Facebook/HDX
        iso3_country_code : str
            ISO3 country code used to identify the population column in the data
        admin_area_boundaries : Polygon or MultiPolygon
            The geographical boundaries used to clip the population data
            
        Returns
        -------
        pandas.DataFrame
            Processed population data containing only points within the administrative area, with columns:

                - ``longitude``: Longitude coordinate
                - ``latitude``: Latitude coordinate
                - ``population``: Population count (renamed from country-specific column)
        """
        gdf = GeoDataFrame(
            downloaded_data,
            geometry=points_from_xy(
                downloaded_data["longitude"], downloaded_data["latitude"]
            ),
        )
        gdf = clip(gdf, admin_area_boundaries)
        df = gdf.drop(columns=["geometry"]).rename(
            columns={f"{iso3_country_code.lower()}_general_2020": "population"}
        )
        return df


class WorldpopPopulation(Population):
    """Population data from WorldPop global population data.
    
    This class retrieves and processes population data from the WorldPop project, which provides high-resolution 
    population density estimates globally. The data is accessed via the WorldPop REST API.
    
    The class follows the same initialization pattern as its parent class Population.
    
    See Also
    --------
    Population : Parent abstract class
    FacebookPopulation : Alternative population data source implementation
    """

    def _get_population_df(self) -> pd.DataFrame:
        """Download and process population data from WorldPop.
        
        Implements the abstract method from the Population class to retrieve population data specifically from WorldPop's
         REST API.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with population data containing columns:
            - longitude: Longitude coordinate
            - latitude: Latitude coordinate
            - population: Population count at the coordinate
        """
        downloaded_data = self.download_population_worldpop(
            iso3_country_code=self.iso3_country_code,
        )

        processed_data = self.process_population_worldpop(
            downloaded_data,
            admin_area_boundaries=self.admin_area_boundaries,
        )

        return processed_data

    @staticmethod
    def download_population_worldpop(iso3_country_code: str) -> str:
        """Download population data from WorldPop for a specific country.
        
        This method retrieves population data from the WorldPop REST API for a country specified by its ISO3 code, using 
        the most recently available dataset.
        
        Parameters
        ----------
        iso3_country_code : str
            The ISO3 country code for which to download population data (e.g., ``TLS`` for Timor-Leste)
        
        Returns
        -------
        str
            File path to the downloaded raster data file
            
        Raises
        ------
        requests.exceptions.RequestException
            If there are issues with the WorldPop API request
        """
        worldpop_url = f"https://www.worldpop.org/rest/data/pop/wpgpunadj/?iso3={iso3_country_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }  # User-Agent is required to get API response

        with requests.get(worldpop_url, headers=headers) as response:
            response.raise_for_status()
            data = response.json()["data"][-1]  # Extract latest year data

        # Extract url for data for the latest year
        url = data["files"][0]
        filehandle, _ = urllib.request.urlretrieve(url)

        return filehandle

    @staticmethod
    def process_population_worldpop(
        file_path: str, admin_area_boundaries: Polygon | MultiPolygon
    ) -> pd.DataFrame:
        """Process WorldPop raster data for a specific administrative area.
        
        This method converts the downloaded WorldPop raster data into a DataFrame of point-based population values, 
        filtered to include only points within the specified administrative area boundaries.
        
        Parameters
        ----------
        file_path : str
            Path to the downloaded WorldPop raster file
        admin_area_boundaries : Polygon or MultiPolygon
            The geographical boundaries used to filter the population data
            
        Returns
        -------
        pandas.DataFrame
            Processed population data containing only points within the administrative area, with columns:

                - ``longitude``: Longitude coordinate
                - ``latitude``: Latitude coordinate
                - ``population``: Population count
        """
        # Convert raster file to dataframe
        with rasterio.open(file_path) as src:
            # Create 2D arrays
            xmin, ymax = np.around(src.xy(0.00, 0.00), 9)
            xmax, ymin = np.around(src.xy(src.height - 1, src.width - 1), 9)
            x = np.linspace(xmin, xmax, src.width)
            y = np.linspace(ymax, ymin, src.height)
            xs, ys = np.meshgrid(x, y)
            zs = src.read(1)
            # Adm area mask
            adm_mask = WorldpopPopulation.get_admarea_mask(admin_area_boundaries, src)
            xs, ys, zs = xs[adm_mask], ys[adm_mask], zs[adm_mask]
            data = {
                "longitude": pd.Series(xs),
                "latitude": pd.Series(ys),
                "population": pd.Series(zs),
            }
            # Create X,Y,Z DataFrame
            df = pd.DataFrame(data=data)
        return df

    @staticmethod
    def get_admarea_mask(
        vector_polygon: Polygon | MultiPolygon, raster_layer: rasterio.DatasetReader
    ) -> np.ndarray:
        """Create a boolean mask identifying raster pixels within a vector polygon.
        
        This method creates a mask that can be used to filter raster data to include only points that fall within a 
        specified vector polygon.
        
        Parameters
        ----------
        vector_polygon : Polygon or MultiPolygon
            The vector geometry used to create the mask
        raster_layer : rasterio.DatasetReader
            The open raster dataset to be masked
            
        Returns
        -------
        np.ndarray
            A boolean mask with the same dimensions as the raster, where:

                - ``True``: pixel is within the polygon
                - ``False``: pixel is outside the polygon
            
        Notes
        -----
        The method uses rasterio's mask function with the all_touched=True parameter,
        which includes all pixels that are touched by the polygon, not just those
        whose centers are within it. It then creates a boolean mask by checking
        which pixels have values greater than zero.
        """
        gtraster, bound = mask(
            raster_layer, [vector_polygon], all_touched=True, crop=False
        )

        # Keep only non zero values
        adm_mask = gtraster[0] > 0
        return adm_mask
