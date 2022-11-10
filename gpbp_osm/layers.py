import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

import gadm
from .constants import FACILITIES_SRC, POPULATION_SRC
from .utils import generate_grid_in_polygon


class AdmArea:
    def __init__(self, country: str, level: int) -> None:
        self.country = country
        self.level = level
        self._get_country_data()
        self.geometry = None
        self.fac_gdf = None
        self.pop_df = None
        self.grid_gdf = None

    def _get_country_data(self) -> None:
        print(f"Retrieving data for {self.country} of granularity level {self.level}")
        self.country_fc = gadm.get_data(code=self.country, level=self.level)
        print(f"Administrative areas for level {self.level}:")
        print([feat["properties"][f"NAME_{self.level}"] for feat in self.country_fc])

    def get_adm_area(self, adm_name: str) -> None:
        self.adm_name = adm_name

        args = {f"NAME_{self.level}": self.adm_name}
        adm_fc = self.country_fc.get(**args)
        if adm_fc:
            geometry = MultiPolygon(
                Polygon(shape[0]) for shape in adm_fc["geometry"]["coordinates"]
            )
            self.geometry = geometry
        else:
            print(f"No data found for {self.adm_name}")

    def get_facilities(self, method: str, tags: dict = None) -> None:
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        self.fac_gdf = FACILITIES_SRC[method](self.adm_name, self.geometry, tags)

    def get_population(self, method: str) -> None:
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        self.pop_df = POPULATION_SRC[method](self.country, self.geometry)

    def compute_potential_fac(self, spacing: float) -> None:
        self.grid_gdf = generate_grid_in_polygon(spacing, self.geometry)
