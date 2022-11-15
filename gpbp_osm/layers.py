import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import Point

from gadm import GADMDownloader

from .constants import FACILITIES_SRC, POPULATION_SRC, RWI_SRC
from .utils import generate_grid_in_polygon
import pycountry


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
    def __init__(self):
        pass

    def calculate_isochrone_isodistance_OSM(
        self, trip_length, country, adm_area_name, level, network_type, distance_type
    ) -> None:

        adm_area = AdmArea(country, level)
        adm_area.get_adm_area(adm_area_name)
        G = ox.graph_from_polygon(adm_area.geometry, network_type, retain_all=False)

        gdf_nodes = ox.graph_to_gdfs(G, edges=False)
        x, y = gdf_nodes["geometry"].unary_union.centroid.xy
        center_node = ox.distance.nearest_nodes(G, x[0], y[0])

        # For isochrones distance="time", for isodistances distance = "distance"
        # TODO: convert distance to time if distance_type is time, depends on network_type

        subgraph = nx.ego_graph(
            G, center_node, radius=trip_length, distance=distance_type
        )
        node_points = [
            Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
        ]
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull

        # Plot
        fig, ax = ox.plot_graph(
            G,
            show=False,
            close=False,
            edge_color="#999999",
            edge_alpha=0.2,
            node_size=0,
        )
        patch = PolygonPatch(
            bounding_poly, fc="#59b9f2", ec="none", alpha=0.6, zorder=-1
        )
        ax.add_patch(patch)
        #plt.show()

    def calculate_isochrone_isodistance_Mapbox(self):
        pass
