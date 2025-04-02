import urllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.mask import mask
from geopandas import GeoDataFrame, clip, points_from_xy
from shapely import MultiPolygon, Polygon

from hdx.api.configuration import Configuration
from hdx.data.resource import Resource

from pisa.administrative_area import AdministrativeArea


class Population(ABC):
    """Abstract base class for Population. Subclasses must implement the method get_population_data()
    If you want to add new data source (e.g. geojson): create new subclass with method get_population_data().
    """
    def __init__(
        self,
        admin_area: AdministrativeArea,
        admin_boundaries: Polygon | MultiPolygon,
        population_resolution: int=5,
    ):
        self.admin_area = admin_area
        self.iso3_country_code = admin_area.get_iso3_country_code()
        self.admin_boundaries = admin_boundaries
        self.population_resolution = population_resolution

    def get_population_gdf(self) -> tuple[GeoDataFrame, pd.DataFrame]:
        """Integrates the methods to get the population numbers for the selected area into one flow and
        returns grouped population data for the admin area as a GeoDataFrame."""
        population_df = self.get_population_data()
        return self.group_population(population_df, self.population_resolution), population_df

    @staticmethod
    def group_population(population_df: pd.DataFrame, population_resolution: int) -> GeoDataFrame:
        """ Group population data by longitude and latitude based on the population resolution. The population resolution
        is an integer that indicates the number of digits after the decimal point to which latitude and longitude get rounded
        and then grouped. The population of all rows with the same unique combination of latitude and longitude after
        rounding by population resolution is summed. The resulting dataframe is returned as a GeoDataFrame."""
        population_df.loc[:, ["longitude", "latitude"]] = population_df[["longitude", "latitude"]].round(
            population_resolution)

        population = (
            population_df.groupby(["longitude", "latitude"], as_index=False)["population"]
            .sum()
            .reset_index(names="ID")
        )
        population["population"] = population["population"].round(2)
        population = GeoDataFrame(population, geometry=points_from_xy(population.longitude, population.latitude))
        return population

    @abstractmethod
    def get_population_data(self) -> pd.DataFrame:
        """Must be implemented in subclasses"""
        pass


class FacebookPopulation(Population):
    def __init__(
            self,
            admin_area: AdministrativeArea,
            admin_boundaries: Polygon | MultiPolygon,
            population_resolution: int=5):
        super().__init__(admin_area, admin_boundaries, population_resolution)

    def get_population_data(self) -> pd.DataFrame:
        """Download & process data from the chosen datasource 'facebook'. Returns a DataFrame with population data."""

        downloaded_data = self.download_population_facebook(
            iso3_country_code=self.iso3_country_code,
        )

        processed_data = self.process_population_facebook(
            downloaded_data,
            iso3_country_code=self.iso3_country_code,
            admin_boundaries=self.admin_boundaries,
        )

        return processed_data

    @staticmethod
    def download_population_facebook(
            iso3_country_code: str
    ) -> pd.DataFrame:
        """Download population data from 2020 from facebook for a country defined by the iso3_country_code."""
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
    def process_population_facebook(downloaded_data: pd.DataFrame, iso3_country_code: str,
                                    admin_boundaries: Polygon | MultiPolygon) -> pd.DataFrame:
        """ Create geodataframe, clip with admin area boundaries to keep only those areas inside the admin area boundaries
         and convert back to pandas dataframe"""
        gdf = GeoDataFrame(downloaded_data,
                           geometry=points_from_xy(downloaded_data["longitude"], downloaded_data["latitude"]))
        gdf = clip(gdf, admin_boundaries)
        df = gdf.drop(columns=["geometry"]).rename(
            columns={f"{iso3_country_code.lower()}_general_2020": "population"})
        return df


class WorldpopPopulation(Population):
    def __init__(
            self,
            admin_area: AdministrativeArea,
            admin_boundaries: Polygon | MultiPolygon,
            population_resolution: int=5):
        super().__init__(admin_area, admin_boundaries, population_resolution)

    def get_population_data(self) -> pd.DataFrame:
        """Download & process data from the chosen datasource 'worldpop'. Returns a DataFrame with population data."""

        downloaded_data = self.download_population_worldpop(
            iso3_country_code=self.iso3_country_code,
        )

        processed_data = self.process_population_worldpop(
            downloaded_data,
            admin_boundaries=self.admin_boundaries,
        )

        return processed_data

    @staticmethod
    def download_population_worldpop(
        iso3_country_code: str
    ) -> str:
        """Download population numbers from worldpop from last year for a country defined by the iso3_country_code."""

        worldpop_url = (
            f"https://www.worldpop.org/rest/data/pop/wpgpunadj/?iso3={iso3_country_code}"
        )
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
    def process_population_worldpop(file_path: str, admin_boundaries: Polygon | MultiPolygon) -> pd.DataFrame:
        """
        Processes the downloaded worldpop data raster file into the required format of a dataframe of longitude, latitude
        and statistical population count

        Function takes the bounds of the raster file in the raster_fpath, draws an evenly spaced sequence of points between
        the xmin & xmax, and between ymin & ymax, and then generates a grid to cover the complete square area inside the
        boundaries. The population count for each point in the grid that falls within the given MultiPolygon area
        (identified by the mask) is extracted from the raster file, and a dataframe with latitude, longitude & population
        count for each point in the raster is returned.
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
            adm_mask = WorldpopPopulation.get_admarea_mask(admin_boundaries, src)
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
    def get_admarea_mask(vector_polygon: Polygon | MultiPolygon, raster_layer: rasterio.DatasetReader) -> np.ndarray:
        """
        Extract mask from raster for a given MultiPolygon

        Return a boolean mask for the raster layer which is True where the (multi)polygon is located and false for all
        points outside the given (Multi)Polygon
        """
        gtraster, bound = mask(
            raster_layer, [vector_polygon], all_touched = True, crop = False)

        # Keep only non zero values
        adm_mask = gtraster[0] > 0
        return adm_mask

