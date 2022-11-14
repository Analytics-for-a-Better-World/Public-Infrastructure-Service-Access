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
        
class RoadNetworkLayer():
    def __init__(self):
        pass

    def calculate_isochrone_isodistance_OSM(self, trip_length, country, adm_area_name, level, network_type,
                                            distance_type) -> None:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import networkx as nx
        import osmnx as ox
        from descartes import PolygonPatch
        from shapely.geometry import Point

        adm_area = AdmArea(country, level)
        adm_area.get_adm_area(adm_area_name)
        G = ox.graph_from_polygon(adm_area.geometry, network_type, retain_all=False)

        gdf_nodes = ox.graph_to_gdfs(G, edges=False)
        x, y = gdf_nodes["geometry"].unary_union.centroid.xy
        center_node = ox.distance.nearest_nodes(G, x[0], y[0])

        # For isochrones distance="time", for isodistances distance = "distance"
        #TODO: convert distance to time if distance_type is time, depends on network_type

        subgraph = nx.ego_graph(G, center_node, radius=trip_length, distance=distance_type)
        node_points = [Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)]
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull

        # Plot
        fig, ax = ox.plot_graph(
            G, show=False, close=False, edge_color="#999999", edge_alpha=0.2, node_size=0
        )
        patch = PolygonPatch(bounding_poly, fc="#59b9f2", ec="none", alpha=0.6, zorder=-1)
        ax.add_patch(patch)
        #plt.show()

    def calculate_isochrone_isodistance_Mapbox(self):
        pass
