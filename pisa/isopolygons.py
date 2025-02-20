import logging
from abc import ABC, abstractmethod

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from networkx import MultiDiGraph
from pandas import DataFrame
from shapely import MultiPolygon, Polygon


class IsopolygonCalculator(ABC):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,  # e.g. travel_time or length
        distance_values: list[int],
    ):

        self.facilities_longitude_array = facilities_lon_lat.longitude.values
        self.facilities_latitude_array = facilities_lon_lat.latitude.values
        self.distance_type = distance_type
        self.distance_values = distance_values

        self._validate_input()

    def _validate_input(self):
        """Checks that distance values are within the permitted limits"""

        if self.distance_type == "length" and max(self.distance_values) > 100000:
            raise ValueError(
                "One or more distance values are larger than the permitted 100.000 meters limit."
            )

        if self.distance_type == "minutes" and max(self.distance_values) > 60:
            raise ValueError(
                "One or more distance values are larger than the permitted 60 minutes limit."
            )

    @abstractmethod
    def calculate_isopolygons(self) -> DataFrame:
        """must be implemented in subclasses"""
        pass


class OsmIsopolygonCalculator(IsopolygonCalculator):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,
        distance_values: list[int],
        road_network: MultiDiGraph,
        node_buff: float = 0.001,  # TODO: replace with buffer = 50 m
        edge_buff: float = 0.0005,  # TODO: remove
    ):
        super().__init__(facilities_lon_lat, distance_type, distance_values)
        self.road_network = road_network
        self.node_buff = node_buff
        self.edge_buff = edge_buff

        # Find the nearest node in the road network for each facility
        self.nearest_nodes = ox.distance.nearest_nodes(
            G=self.road_network,
            X=self.facilities_longitude_array,
            Y=self.facilities_latitude_array,
        )

    def calculate_isopolygons(self) -> DataFrame:

        isopolygons = pd.DataFrame()

        # Construct isopolygon for each distance value
        for distance_value in self.distance_values:

            for road_node in self.nearest_nodes:

                nodes_gdf, edges_gdf = self._get_skeleton_nodes_and_edges(
                    self.road_network, road_node, distance_value, self.distance_type
                )

                try:

                    new_isopolygon = self._inflate_skeleton_to_isopolygon(
                        nodes_gdf=nodes_gdf,
                        edges_gdf=edges_gdf,
                        node_buff=self.node_buff,
                        edge_buff=self.edge_buff,
                    )
                    isopolygons.loc[road_node, "ID_" + str(distance_value)] = (
                        new_isopolygon
                    )

                except (
                    AttributeError
                ):  # Probably trying to catch 'MultiPolygon' object has no attribute 'exterior' from _inflate_skeleton, but it wasn't specified in the code before

                    logging.info(f"problem with node {road_node}")  # stops execution

        return isopolygons

    @staticmethod
    def _inflate_skeleton_to_isopolygon(nodes_gdf, edges_gdf, node_buff, edge_buff):
        """
        Catalina, Feb 2025:
        This method should be replaced with use osmnx.utils_geo.buffer_geometry(geom, dist)
        where dist is 50m, as directed by Joaquim


        Catalina, Jan 2025:

        This method is sensitive to the values of node_buff and edge_buff.
        If they are too large relative to the distances between nodes (for
        example in a walking road network), the buffer could be so large
        that you end up including areas by mistake (like nodes you had
        previously excluded).

        TODO:
        - find appropriate default values for node_buff and edge_buff
        relative to network distances?
        - why different buff for nodes and edges?
        - this method throws
        AttributeError: 'MultiPolygon' object has no attribute 'exterior'
        if the result of unary_union is two (or more) disconnected Polygons.
        Figure out when this could happen (could a node be disconnected?) and
        make strategy to catch it
        - Tests throw warning:
        "UserWarning: Geometry is in a geographic CRS. Results from 'buffer' are likely incorrect.
        Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before
        this operation."
        TODO: use osmnx.utils_geo.buffer_geometry(geom, dist) after migration to ox 2
        """

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
    def _get_skeleton_nodes_and_edges(
        road_network: nx.MultiDiGraph,
        center_node: int,
        distance_value: int,
        distance_type: str,
    ) -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
        """
        Get nodes and edges within a specified distance from a certain node in a road network.
        This will be the "skeleton" of the isopolygon.

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

        If no edges are found (for example, if all other nodes are too far away from center_node),
        edges_gdf is an empty dataframe.

        """
        subgraph = nx.ego_graph(
            road_network, center_node, radius=distance_value, distance=distance_type
        )

        try:
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
            return (
                nodes_gdf.loc[:, "geometry"],
                edges_gdf.loc[:, "geometry"].reset_index(),
            )

        except ValueError:  # if no edges are found
            nodes_gdf = ox.graph_to_gdfs(subgraph, edges=False)
            return (
                nodes_gdf.loc[:, "geometry"],
                gpd.GeoSeries([], name="geometry", crs=nodes_gdf.crs),
            )


class OsmIsopolygonCalculatorAlternative(IsopolygonCalculator):

    def __init__(
        self,
        facilities_lon_lat: DataFrame,
        distance_type: str,
        distance_values: list[int],
        road_network: MultiDiGraph,
        buffer: float = 50,  # in meters
    ):
        super().__init__(facilities_lon_lat, distance_type, distance_values)
        self.road_network = road_network
        self.buffer = buffer

        # Find the nearest node in the road network for each facility
        self.nearest_nodes = ox.distance.nearest_nodes(
            G=self.road_network,
            X=self.facilities_longitude_array,
            Y=self.facilities_latitude_array,
        )

        # nearest_nodes has type ndarray, converting to list

    def calculate_isopolygons(self) -> DataFrame:

        # Initialize DataFrame with explicit index
        isopolygons = pd.DataFrame()

        # Construct isopolygon for each distance value
        for distance_value in self.distance_values:

            for road_node in self.nearest_nodes:

                skeleton = self._get_skeleton_nodes_and_edges(
                    self.road_network, road_node, distance_value, self.distance_type
                )

                new_isopolygon = ox.utils_geo.buffer_geometry(
                    geom=skeleton, dist=self.buffer
                )

                isopolygons.at[road_node, "ID_" + str(distance_value)] = new_isopolygon

        return isopolygons

    @staticmethod
    def _get_skeleton_nodes_and_edges(
        road_network: nx.MultiDiGraph,
        center_node: int,
        distance_value: int,
        distance_type: str,
    ):  # hard to specify return type. It should be type "Geometry", but I don't know how to write that correctly
        """
        Get nodes and edges within a specified distance from a certain node in a road network.
        This will be the "skeleton" of the isopolygon.

        Parameters:
            road_network (nx.MultiDiGraph): The road network.
            center_node (int): The node from which to measure the distance.
            distance_value (int): The distance value.
            distance_type (str): The type of distance (e.g., 'length').

        Returns:
            The union of the geometries of the nodes and edges within the specified distance from center_node.


        """
        subgraph = nx.ego_graph(
            road_network, center_node, radius=distance_value, distance=distance_type
        )

        try:
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)

            skeleton = (edges_gdf.geometry.union_all()).union(
                nodes_gdf.geometry.union_all()
            )

        except ValueError:  # if no edges are found
            nodes_gdf = ox.graph_to_gdfs(subgraph, edges=False)

            skeleton = nodes_gdf.geometry.union_all()

        return skeleton


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

    def calculate_isopolygons(self) -> DataFrame: ...


# If you want to implement a new way of calculating isopolygons (e.g. GoogleMaps),
# create a class that inherits from IsopolygonCalculator and implements calculate_isopolygons
