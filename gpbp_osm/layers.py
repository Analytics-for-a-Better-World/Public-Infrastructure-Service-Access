from gadm import GADMDownloader
import osmnx as ox
from .constants import FACILITIES_SRC, POPULATION_SRC, RWI_SRC
from .utils import generate_grid_in_polygon, group_population
from .distance import population_served
from .road_network import get_road_network_overpass
import pycountry
import pandas as pd


class AdmArea:
    def __init__(self, country: str, level: int) -> None:
        self.country = pycountry.countries.get(name=country)
        self.level = level
        self.geometry = None
        self.fac_gdf = None
        self.pop_df = None
        self.grid_gdf = None
        self.rwi_df = None
        self.iso_gdf = None
        self.road_network = None
        self._get_country_data()

    def _get_country_data(self) -> None:
        print(
            f"Retrieving data for {self.country.name} of granularity level {self.level}"
        )
        downloader = GADMDownloader(version="4.0")
        self.country_gdf = downloader.get_shape_data_by_country_name(
            country_name=self.country.name, ad_level=self.level
        )
        if self.level > 0:
            print(f"Administrative areas for level {self.level}:")
            print(self.country_gdf[f"NAME_{self.level}"].values)
        else:
            print(f"Extracting geometry for {self.country.name}")
            self.geometry = self.country_gdf.geometry.values[0]
            self.adm_name = self.country.name

    def get_adm_area(self, adm_name: str) -> None:
        self.adm_name = adm_name
        try:
            self.geometry = self.country_gdf[
                self.country_gdf[f"NAME_{self.level}"] == adm_name
            ].geometry.values[0]
        except:
            print(f"No data found for {self.adm_name}")

    def get_facilities(self, method: str, tags: dict = None) -> None:
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        self.fac_gdf = FACILITIES_SRC[method](self.adm_name, self.geometry, tags)

    def get_population(self, method: str) -> None:
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        self.pop_df = POPULATION_SRC[method](self.country.alpha_3, self.geometry)

    def get_road_network(self, network_type: str) -> None:
        self.road_network = ox.graph_from_polygon(
            self.geometry, network_type=network_type
        )

    # self.road_network = get_road_network_overpass(
    #   self.geometry, network_type=network_type
    # )

    def get_rwi(self, method: str) -> None:
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        self.rwi_df = RWI_SRC[method](self.country.name, self.geometry)

    def compute_potential_fac(self, spacing: float) -> None:
        self.grid_gdf = generate_grid_in_polygon(spacing, self.geometry)

    def prepare_optimization_data(
        self,
        distance_type,
        distance_values,
        mode_of_transport,
        strategy,
        population_resolution: int = 5,
    ):
        if self.pop_df is None:
            raise Exception("Population data not available")
        if self.fac_gdf is None:
            raise Exception("Facility data not available")
        if self.grid_gdf is None:
            raise Exception("Potential locations not computed")
        pop_gdf = group_population(self.pop_df, population_resolution)
        pop_count = pop_gdf.population.values
        total_fac = pd.concat([self.fac_gdf, self.grid_gdf], ignore_index=True)
        total_fac = (
            total_fac.drop(columns=["ID"]).reset_index().rename(columns={"index": "ID"})
        )
        cutoff_idx = int(self.fac_gdf["ID"].max()) + 1
        current = {}
        current[distance_type] = population_served(
            pop_gdf,
            total_fac[0:cutoff_idx],
            "facilities",
            distance_type,
            distance_values,
            mode_of_transport,
            strategy,
            self.road_network,
        )
        potential = {}
        potential[distance_type] = population_served(
            pop_gdf,
            total_fac[cutoff_idx:],
            "facilities",
            distance_type,
            distance_values,
            mode_of_transport,
            strategy,
            self.road_network,
        )
        return pop_count, current, potential
