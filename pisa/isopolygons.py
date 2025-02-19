from abc import ABC, abstractmethod

import geopandas as gpd
import networkx as nx
import osmnx as ox
from networkx import MultiDiGraph
from pandas import DataFrame
from shapely import LineString, Point, Polygon


class IsopolygonCalculator(ABC):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,  # e.g. travel_time or length
        distance_values: list[int],
    ):
        # error if distance values are too high
        # convert lon and lat to np arrays
        ...

    # TODO: is dict[DataFrame] the return type we want?

    @abstractmethod
    def calculate_isopolygons(self) -> dict[DataFrame]:
        """must be implemented in subclasses"""
        pass


class OsmIsopolygonCalculator(IsopolygonCalculator):

    def __init__(
        self,
        facilities_lat_lon: DataFrame,
        distance_type: str,
        distance_values: list[int],
        road_network: MultiDiGraph,
        node_buff: float = 0.001,  # TODO: replace with buffer = 50 m
        edge_buff: float = 0.0005,  # TODO: remove
    ):
        super().__init__(facilities_lat_lon, distance_type, distance_values)
        self.road_network = road_network

    def calculate_isopolygons(self) -> dict[DataFrame]: ...

    @staticmethod
    def create_polygon_from_nodes_and_edges(nodes_gdf, edges_gdf, node_buff, edge_buff):
        """This method should be replaced with use osmnx.utils_geo.buffer_geometry(geom, dist)
        where dist is 50m, as directed by Joaquim"""

        if edge_buff <= 0:
            raise ValueError("The parameter edge_buff must be greater than 0.")

        # creates a circle with radius node_buff around each node
        disks = nodes_gdf.buffer(node_buff).geometry

        # creates a buffer around each edge. For example, if the edge is a line between
        # two points, it creates a sort of 2-d "cylinder" of "radius" edge_buff
        # and semicircles of radius edge_buff at the end points
        cylinders = edges_gdf.buffer(edge_buff).geometry

        all_geometries = list(disks) + list(cylinders)

        # Merges all_geometries into a single unified geometry, removing overlaps and simplifying the resulting shape
        geometric_union = gpd.GeoSeries(all_geometries).unary_union

        # Removes any interior boundaries (holes) that geometric_union might have had
        buffer_polygon = Polygon(geometric_union.exterior)

        return buffer_polygon

    @staticmethod
    def _get_poly_nx_new(
        road_network: nx.MultiDiGraph,
        center_node: int,
        dist_value: int,
        distance_type: str,
    ) -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
        """
        Get nodes and edges within a specified distance from a certain node in a road network.

        Parameters:
            road_network (nx.MultiDiGraph): The road network.
            center_node (int): The node from which to measure the distance.
            dist_value (int): The distance value.
            distance_type (str): The type of distance (e.g., 'length').

        Returns:
            nodes_gdf: a GeoSeries of the nodes with their osmid and geometry.
            edges_gdf: a GeoSeries of the geometry of the edges.

        If an edge (u,v) doesn't have geometry data in G, edges_gdf contains
        a straight line from u to v.

        Raises:
            ValueError if all other nodes are farther than dist_value from center_node

        """
        subgraph = nx.ego_graph(
            road_network, center_node, radius=dist_value, distance=distance_type
        )

        nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)

        return nodes_gdf.loc[:, "geometry"], edges_gdf.loc[:, "geometry"].reset_index()

    @staticmethod
    def _get_poly_nx(
        road_network: nx.MultiDiGraph, center_node, dist_value, distance_type
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoSeries]:
        """
        todo: change to native function if possible
        todo: Rename
        """

        subgraph = nx.ego_graph(
            road_network, center_node, radius=dist_value, distance=distance_type
        )

        node_points = [
            Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
        ]
        nodes_gdf = gpd.GeoDataFrame({"id": list(subgraph.nodes)}, geometry=node_points)
        nodes_gdf = nodes_gdf.set_index("id")

        edge_lines = []
        for n_fr, n_to in subgraph.edges():
            f = nodes_gdf.loc[n_fr].geometry
            t = nodes_gdf.loc[n_to].geometry
            edge_lookup = road_network.get_edge_data(n_fr, n_to)[0].get(
                "geometry", LineString([f, t])
            )
            edge_lines.append(edge_lookup)
        edges_gdf = gpd.GeoSeries(edge_lines)
        return nodes_gdf, edges_gdf


class MapboxIsopolygonCalculator(IsopolygonCalculator):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,
        distance_values: list[int],
        route_profile: str,  # ?
        mapbox_api_token: str,
    ):
        super().__init__(facilities_lon_lat, distance_type, distance_values)
        self.route_profile = route_profile
        self.mapbox_api_token = mapbox_api_token

    def calculate_isopolygons(self) -> dict[DataFrame]: ...


# If you want to implement a new way of calculating isopolygons (e.g. GoogleMaps),
# create a class that inherits from IsopolygonCalculator and implements calculate_isopolygons
