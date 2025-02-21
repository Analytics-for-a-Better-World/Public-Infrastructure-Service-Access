import urllib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from geopandas import GeoDataFrame, clip, points_from_xy
from numpy import ndarray
from pandas import DataFrame
from rasterio import DatasetReader
from shapely import MultiPolygon, Polygon, Point
import requests
from hdx.api.configuration import Configuration
from hdx.data.resource import Resource


@dataclass
class Population:

    # remember to use "with" statement when getting resources

    data_source: str
    iso3_country_code: str
    admin_area_boundaries: Polygon | MultiPolygon
    population_resolution: int = 5

    def get_population_gdf(self) -> GeoDataFrame:
        """Integrates all the methods into one flow"""

        # define population_df according to the data_source given

        if self.data_source == "world_pop":
            population_df = self.get_population_worldpop()

        elif self.data_source == "facebook":
            population_df = self.get_population_facebook()

        # geojson or any other source can be added here in the future


        else:
            return  # handle error no valid data_source (consider Enum??)

        # check recency

        if not self.is_recent_data(population_df):
            # warn?
            ...

        # return grouped population

        return self.group_population(population_df, self.population_resolution)

    @staticmethod
    def group_population(population_df: DataFrame, population_resolution: int) -> GeoDataFrame:
        population = population_df.copy()
        population["longitude"] = population["longitude"].round(population_resolution)
        population["latitude"] = population["latitude"].round(population_resolution)

        population = (
            population.groupby(["longitude", "latitude"])["population"]
            .sum()
            .reset_index()
        )
        population["population"] = population["population"].round(2)
        population.columns = ["ID", "longitude", "latitude", "population"]
        population = population.set_geometry(
            points_from_xy(population.longitude, population.latitude)
        )
        return population

    @staticmethod
    def is_recent_data(population_df: DataFrame) -> bool:

        # should this be implemented? Unsure how. What is a valid check? No longer than one year ago? Two? Five?
        ...

    ###### methods for dealing w facebook population start here #####
    def get_population_facebook(self) -> DataFrame:
        """
        - downloads data
        - processes data
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
    def download_population_facebook(
        iso3_country_code: str
    ) -> DataFrame:
        """
        Get 2020 facebook data for an area defined by the MultiPolygon geometry
        """
        try:
            Configuration.create(
                hdx_site="prod", user_agent="Get_Population_Data", hdx_read_only=True
            )
        except:
            pass
        resource = Resource.search_in_hdx(
            f"name:{iso3_country_code.lower()}_general_2020_csv.zip"
        )
        url = resource[0]["download_url"]
        filehandle, _ = urllib.request.urlretrieve(url)

        df = pd.read_csv(filehandle, compression="zip")
        return df


    @staticmethod
    def process_population_facebook(downloaded_data: DataFrame, iso3_country_code: str, admin_area_boundaries: Polygon | MultiPolygon) -> DataFrame:
        """ Create GDF, clip with geometry and convert back to regular DF"""
        downloaded_data["geometry"] = downloaded_data.apply(lambda x: Point(x["longitude"], x["latitude"]), axis=1)
        gdf = GeoDataFrame(downloaded_data, geometry="geometry")
        gdf = clip(gdf, admin_area_boundaries)
        df = pd.DataFrame(gdf.drop(columns=["geometry"]))

        print("Loading data to dataframe")
        df = df.rename(columns={f"{iso3_country_code.lower()}_general_2020": "population"})
        return df

    ##### methods for dealing w worldpop population start here #####

    def get_population_worldpop(self) -> DataFrame:
        """
        - downloads data
        - processes data
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
    def download_population_worldpop(
        iso3_country_code: str
    ) -> str:

        worldpop_url = (
            f"https://www.worldpop.org/rest/data/pop/wpgpunadj/?iso3={iso3_country_code}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }  # User-Agent is required to get API response

        response = requests.get(worldpop_url, headers=headers)

        # Extract url for data for the latest year
        data = response.json()["data"][-1]
        url = data["files"][0]
        filehandle, _ = urllib.request.urlretrieve(url)

        return filehandle


    @staticmethod
    def process_population_worldpop(downloaded_data_file: str, admin_area_boundaries: Polygon | MultiPolygon) -> DataFrame:
        df = Population.raster_to_df(downloaded_data_file, admin_area_boundaries)

        return df


    @staticmethod
    def raster_to_df(downloaded_data_file: str, mask_polygon: Polygon | MultiPolygon) -> DataFrame:
        """
        Convert raster file to a dataframe of longitude, latitude
        and statistical population count

        Function takes the bounds of the raster file in the raster_fpath, draws an evenly spaced sequence of points between
        the xmin & xmax, and between ymin & ymax, and then generates a grid to cover the complete square area inside the
        boundaries. The population count for each point in the grid that falls within the given MultiPolygon area
        (identified by the mask) is extracted from the raster file, and a dataframe with latitude, longitude & population
        count for each point in the raster is returned.
        """

        # Convert raster file to dataframe
        src = rasterio.open(downloaded_data_file)
        # Create 2D arrays
        xmin, ymax = np.around(src.xy(0.00, 0.00), 9)
        xmax, ymin = np.around(src.xy(src.height - 1, src.width - 1), 9)
        x = np.linspace(xmin, xmax, src.width)
        y = np.linspace(ymax, ymin, src.height)
        xs, ys = np.meshgrid(x, y)
        zs = src.read(1)
        # Adm area mask
        mask = Population.get_admarea_mask(mask_polygon, src)
        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        data = {
            "longitude": pd.Series(xs),
            "latitude": pd.Series(ys),
            "population": pd.Series(zs),
        }
        # Create X,Y,Z DataFrame
        df = pd.DataFrame(data=data)
        src.close()
        return df


    @staticmethod
    def get_admarea_mask(
        vector_polygon: Polygon | MultiPolygon, raster_layer: DatasetReader, riomask=None) -> ndarray:
        """
        Extract mask from raster for a given MultiPolygon

        Return a boolean mask for the raster layer which is True where the polygon is located and false for all points outside
        the given MultiPolygon
        """
        gtraster, bound = riomask.mask(
        raster_layer, [vector_polygon], all_touched = True, crop = False

        )
        # Keep only non zero values
        adm_mask = gtraster[0] > 0
        return adm_mask

    # if you want to add new data source (e.g. geojson):
    # create method get_population_gson(), and add a call to it in get_population_gdf