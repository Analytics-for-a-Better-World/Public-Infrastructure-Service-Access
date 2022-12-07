from gadm import GADMDownloader
import osmnx as ox
import networkx as nx
from gpbp.constants import FACILITIES_SRC, POPULATION_SRC, RWI_SRC
from gpbp.utils import generate_grid_in_polygon, group_population
from gpbp.distance import population_served

import pycountry
import pandas as pd
import numpy as np


class AdmArea:
    def __init__(self, country: str, level: int) -> None:
        """
        Object representing an administrative area of a country.
        Parameters
        ----------
        country : string
            Name of the country
        level : int
            Level of subdivision
        """

        self.country = pycountry.countries.get(name=country)
        if self.country is None:
            try:
                possible_matches = pycountry.countries.search_fuzzy(country)
            except:
                raise Exception("Invalid form of country name")
            raise Exception(
                f"Country not found. Possible matches: {[match.name for match in possible_matches]}"
            )
        self.level = level
        self.geometry = None
        self.fac_gdf = None
        self.pop_df = None
        self.pot_fac_gdf = None
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

    def retrieve_adm_area_names(self) -> list[str]:
        """
        Return all administrative areas

        Returns
        ----------
        list of strings
        Names of administrative areas
        """
        if self.level == 0:
            return [self.country.name]
        return self.country_gdf[f"NAME_{self.level}"].values

    def get_adm_area(self, adm_name: str) -> None:
        """
        Extract the geometry for a specific administrative area.
        Parameters
        ----------
        adm_name : string
            Administrative area name
        """
        if self.level == 0:
            return
        self.adm_name = adm_name
        try:
            self.geometry = self.country_gdf[
                self.country_gdf[f"NAME_{self.level}"] == adm_name
            ].geometry.values[0]
            print("Extracting geometry for administrative area")
        except:
            print(f"No data found for {self.adm_name}")

    def get_facilities(self, method: str, tags: dict) -> None:
        """
        Retrieve facility locations specified by tags using a strategy
        defined by method

        Parameters
        ----------
        method : string
            Strategy alias. Currently supported options: 'osm'
        tags : dictionary
            OSM tag dictionary
            e.g. {'building':'hospital'} or {'amenity':['school', 'kindergarden']}
        """
        if self.geometry is None:
            raise Exception("Geometry is not defined. Call get_adm_area()")
        if method not in FACILITIES_SRC.keys():
            raise Exception("Invalid method")
        self.fac_gdf = FACILITIES_SRC[method](self.adm_name, self.geometry, tags)

    def get_population(self, method: str) -> None:
        """
        Retrieve geolocated statistical population count

        Parameters
        ----------
        method : string
            Strategy alias. Currently supported options: 'world_pop', 'fb_pop'
        """
        if self.geometry is None:
            raise Exception("Geometry is not defined. Call get_adm_area()")
        if method not in POPULATION_SRC.keys():
            raise Exception("Invalid method")
        self.pop_df = POPULATION_SRC[method](self.country.alpha_3, self.geometry)

    def get_road_network(self, network_type: str) -> None:
        """
        Retrieve open street map road network for a network_type
        and calculate road travel time

        Parameters
        ----------
        network_type : string
            The network type in terms of mode of transportation.
            Valid inputs : 'driving', 'walking', 'cycling'
        """
        if network_type == "driving":
            network_type = "drive"
            default_speed = 50

        elif network_type == "walking":
            network_type = "walk"
            default_speed = 4

        elif network_type == "cycling":
            network_type = "bike"
            default_speed = 15
        else:
            raise Exception("Invalid network type")
        # Get network
        self.road_network = ox.graph_from_polygon(
            self.geometry, network_type=network_type
        )
        # Add travel time edge attribute in minutes
        self.road_network = ox.add_edge_speeds(
            self.road_network, fallback=default_speed
        )
        self.road_network = ox.add_edge_travel_times(self.road_network)
        time = nx.get_edge_attributes(self.road_network, "travel_time")
        time_min = dict(
            zip(list(time.keys()), list(map(lambda x: round(x / 60, 2), time.values())))
        )
        nx.set_edge_attributes(self.road_network, time_min, "travel_time")

    def get_rwi(self, method: str) -> None:
        """
        Retrieve geolocated relative wealth index

        Parameters
        ----------
        method : string
            Strategy alias. Currently supported options: 'fb_rwi'
        """
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        if method not in RWI_SRC.keys():
            raise Exception("Invalid method")
        self.rwi_df = RWI_SRC[method](self.country.name, self.geometry)

    def compute_potential_fac(self, spacing: float) -> None:
        """
        Compute potential facilities locations by generating evenly spaced points
        within the administrative area.

        Parameters
        ----------
        spacing : float
            Defines the distance between the points in coordinate units.
        """
        self.pot_fac_gdf = generate_grid_in_polygon(spacing, self.geometry)

    def prepare_optimization_data(
        self,
        distance_type: str,
        distance_values: list[int],
        mode_of_transport: str,
        strategy: str,
        mapbox_access_token: str = None,
        population_resolution: int = 5,
    ) -> tuple[np.ndarray[float], dict[pd.DataFrame], dict[pd.DataFrame]]:
        """
        Prepare input for the optimization model.
        Computes which households are served from existing and potential facility locations
        within the distance_values of the distance_type measure.

        Parameters
        ----------
        distance_type : string
            The measure of distance between points of interest.
            Supported measures: 'length' (as in road length) and 'travel_time'.
        distance_values : list-like of ints
            Values of the distance measure. For length input is expected to be in meters,
            for travel_time in minutes. For performance reasons values greater than 60 minutes
            and 100000 meters are not accepted.
        mode_of_transport : string
            The mode of transportation assumed.
        strategy: string
            Strategy alias. Currently supported options: 'osm', 'mapbox'.
        mapbox_access_token: string
            If mapbox strategy selected the access token to be used for the mapbox api.
        population_resoltion: int
            The resolution of the geolocation coordinates of population households
            in terms of number of decimal digits. Value should be in the range of (1,6).
            The higher the value the more fine-grained the resolution.

        Returns
        -------
        pop_count : array of floats
            The population count of households.
        current : dictionary of DataFrames
            They key of the dictionary is the distance_type and the value is a DataFrame
            using as index the current facilities ids and containing columns of the form ID_{distance_value}
            for each of the distance values where each element contains a list of household ids
            which are served from the index facility within the distance_value.
        potential : dictionary of DataFrames
            Same as current but for the potential location of facilities.

        """
        if self.pop_df is None:
            raise Exception("Population data not available")
        if self.fac_gdf is None:
            raise Exception("Facility data not available")
        if self.pot_fac_gdf is None:
            raise Exception("Potential locations not computed")
        if strategy == "mapbox" and mapbox_access_token is None:
            raise Exception("Mapbox strategy requires an access token")
        if distance_type == "length" and max(distance_values) > 100000:
            raise Exception(
                "One or more distance values are larger than the permitted 100000 meters limit."
            )
        if distance_type == "minutes" and max(distance_values) > 60:
            raise Exception(
                "One or more distance values are larger than the permitted 60 minutes limit."
            )
        pop_gdf = group_population(self.pop_df, population_resolution)
        pop_count = pop_gdf.population.values
        total_fac = pd.concat([self.fac_gdf, self.pot_fac_gdf], ignore_index=True)
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
            mapbox_access_token,
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
            mapbox_access_token,
            self.road_network,
        )
        return pop_count, current, potential
