import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon

from gadm import GADMDownloader

from .constants import FACILITIES_SRC, POPULATION_SRC, RWI_SRC
from .utils import generate_grid_in_polygon
import pycountry
import requests
import json


class AdmArea:
    def __init__(self, country: str, level: int) -> None:
        self.country = pycountry.countries.get(name=country)
        self.level = level
        self.geometry = None
        self.fac_gdf = None
        self.pop_df = None
        self.grid_gdf = None
        self.rwi_df = None
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

    def get_rwi(self, method: str) -> None:
        if self.geometry is None:
            raise Exception("Geometry is not defined")
        self.rwi_df = RWI_SRC[method](self.country.name, self.geometry)

    def compute_potential_fac(self, spacing: float) -> None:
        self.grid_gdf = generate_grid_in_polygon(spacing, self.geometry)


class RoadNetworkLayer:
    def __init__(self, geometry, network_type):
        self.geometry = geometry
        self.network_type = network_type
        self.road_network = ox.graph_from_polygon(self.geometry, self.network_type, retain_all=False)

    def calculate_isochrone_isodistance_OSM(
        self, coord_pair, trip_length, distance_type
    ) -> Polygon:

        G = self.road_network

        hwy_speeds = {'residential': 35,
                      'secondary': 50,
                      'tertiary': 60}
        G = ox.add_edge_speeds(G, hwy_speeds)

        G_speed = ox.add_edge_travel_times(G)

        road_node = ox.distance.nearest_nodes(G, coord_pair[0], coord_pair[1])

        # For isochrones distance="time", for isodistances distance = "length"
        subgraph = nx.ego_graph(
            G, road_node, radius=trip_length, distance=distance_type
        )
        node_points = [
            Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
        ]
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull

        return bounding_poly

    def calculate_isochrone_isodistance_Mapbox(self, coord_pair : tuple[float, float],
                                               route_profile : str, distance_type: str,
                                               distance_values : list[int], access_token: str):
        base_url = "https://api.mapbox.com/isochrone/v1/mapbox/"
        if distance_type == "time":
            contour_type = "contours_minutes"
        elif distance_type == "distance":
            contour_type = "contours_meters"
        request = f"{base_url}{route_profile}{coord_pair[0]}," \
                  f"{coord_pair[1]}?{contour_type}={','.join(list(map(str, distance_values)))}" \
                  f"&polygons=true&denoise=1&access_token={access_token}"
        try:
            request_pack = json.loads(requests.get(request).content)
        except:
            print("Something went wrong")
        features = request_pack["features"]
        bounding_polys = [Polygon(feature['geometry']['coordinates']) for feature in features]
        return bounding_polys


