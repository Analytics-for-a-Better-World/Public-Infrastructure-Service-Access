from .config import MAPBOX_API_ACCESS_TOKEN
import json
import requests
from typing import Any, Union
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import geopandas as gpd
import pandas as pd
from .utils import group_population
from .layers import AdmArea

import networkx as nx
import osmnx as ox


def calculate_isopolygons_graph(
    X: Union[float, list[float]],
    Y: Union[float, list[float]],
    distance_type: str,
    distance_values: list[int],
    road_network: Any,
    road_speeds: dict = None,
    default_speed: int = None,
    edge_buff: float = 0.0005,
    node_buff: float = 0.001,
    infill=True,
) -> dict:

    # make coordinates arrays if user passed non-iterable values
    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        is_scalar = True
        X = [X]
        Y = [Y]

    G = road_network
    isochrone_polys = {}
    road_nodes = ox.distance.nearest_nodes(G, X, Y)

    # Add travel time in seconds edge attribute to network
    if distance_type == "time":
        G = ox.add_edge_speeds(G, hwy_speeds=road_speeds, fallback=default_speed)
        G = ox.add_edge_travel_times(G)
        distance_type = "travel_time"

    # Construct isopolygon for each distance value
    for dist_value in distance_values:
        isochrone_polys[str(dist_value)] = []
        for road_node in road_nodes:
            subgraph = nx.ego_graph(
                G, road_node, radius=dist_value, distance=distance_type
            )

            node_points = [
                Point((data["x"], data["y"]))
                for node, data in subgraph.nodes(data=True)
            ]
            nodes_gdf = gpd.GeoDataFrame(
                {"id": list(subgraph.nodes)}, geometry=node_points
            )
            nodes_gdf = nodes_gdf.set_index("id")

            edge_lines = []
            for n_fr, n_to in subgraph.edges():
                f = nodes_gdf.loc[n_fr].geometry
                t = nodes_gdf.loc[n_to].geometry
                edge_lookup = G.get_edge_data(n_fr, n_to)[0].get(
                    "geometry", LineString([f, t])
                )
                edge_lines.append(edge_lookup)

            n = nodes_gdf.buffer(node_buff).geometry
            e = gpd.GeoSeries(edge_lines).buffer(edge_buff).geometry
            all_gs = list(n) + list(e)
            new_iso = gpd.GeoSeries(all_gs).unary_union

            # try to fill in surrounded areas so shapes will appear solid and
            # blocks without white space inside them
            if infill:
                new_iso = Polygon(new_iso.exterior)
            isochrone_polys[str(dist_value)].append(new_iso)
        if is_scalar:
            isochrone_polys[str(dist_value)] = isochrone_polys[str(dist_value)][0]

    return isochrone_polys


def calculate_isopolygons_Mapbox(
    X: Union[float, list[float]],
    Y: Union[float, list[float]],
    route_profile: str,
    distance_type: str,
    distance_values: list[int],
):
    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        is_scalar = True
        X = [X]
        Y = [Y]
    iso_dict = {str(dist_value): [] for dist_value in distance_values}
    base_url = "https://api.mapbox.com/isochrone/v1/"
    if distance_type == "time":
        contour_type = "contours_minutes"
    elif distance_type == "length":
        contour_type = "contours_meters"
    for coord_pair in list(zip(X, Y)):
        request = (
            f"{base_url}mapbox/{route_profile}/{coord_pair[0]},"
            f"{coord_pair[1]}?{contour_type}={','.join(list(map(str, distance_values)))}"
            f"&polygons=true&denoise=1&access_token={MAPBOX_API_ACCESS_TOKEN}"
        )
        try:
            request_pack = json.loads(requests.get(request).content)
        except:
            print("Something went wrong")
        features = request_pack["features"]
        for feature in features:
            iso_dict[str(feature["properties"]["contour"])].append(
                MultiPolygon(list(map(Polygon, feature["geometry"]["coordinates"])))
            )
            if is_scalar:
                iso_dict[str(feature["properties"]["contour"])] = iso_dict[
                    str(feature["properties"]["contour"])
                ][0]

    return iso_dict


def population_served(
    pop_df: pd.DataFrame,
    fac_gdf: gpd.GeoDataFrame,
    data_as_key: str,
    distance_type: str,
    distance_value: int,
    route_mode: str,
    strategy: str,
    road_network: Any = None,
    road_speeds: dict = None,
    default_speed: int = None,
    pop_resolution: int = 3,
) -> dict:
    # TODO: route mode is useful only for mapbox if we take network as variable
    pop_df = pop_df.copy()
    iso_gdf = fac_gdf.copy()
    # Get isopolygons geodataframe
    if strategy == "mapbox":
        iso_gdf["geometry"] = calculate_isopolygons_Mapbox(
            iso_gdf.longitude.to_list(),
            iso_gdf.latitude.to_list(),
            route_mode,
            distance_type,
            [distance_value],
        )[str(distance_value)]
    elif strategy == "osm":
        if road_network == None:
            raise Exception("OSM strategy needs a road network")
        # OSM accepts time in seconds
        if distance_type == "time":
            distance_value = distance_value * 60
        iso_gdf["geometry"] = calculate_isopolygons_graph(
            iso_gdf.longitude.to_list(),
            iso_gdf.latitude.to_list(),
            distance_type,
            [distance_value],
            road_network,
            road_speeds,
            default_speed,
        )[str(distance_value)]

    iso_gdf = gpd.GeoDataFrame(iso_gdf, geometry="geometry")
    # Group population
    pop_gdf = group_population(pop_df=pop_df, nof_digits=pop_resolution)
    pop_gdf = pop_gdf.set_crs(iso_gdf.crs)
    # Find households within isopolygons
    serve_gdf = pop_gdf.sjoin(iso_gdf, how="right", predicate="within")
    serve_gdf = serve_gdf.dropna()
    if data_as_key == "population":
        serve_dict = (
            serve_gdf.groupby(pop_gdf.index.name)["index_right"].apply(list).to_dict()
        )
    elif data_as_key == "facilities":
        serve_dict = (
            serve_gdf.groupby(fac_gdf.index.name)["index_left"].apply(list).to_dict()
        )
    return serve_dict


def calculate_isopolygons_graph_convex(
    road_network: Any,
    coord_pair: tuple[float, float],
    distance_type: str,
    distance_values: list[int],
    road_speeds: dict = None,
) -> Polygon:
    # Add travel time attribute to network

    G = road_network
    G = ox.add_edge_speeds(G, road_speeds)
    G = ox.add_edge_travel_times(G)

    road_node = ox.distance.nearest_nodes(G, coord_pair[0], coord_pair[1])

    # For isochrones distance="time", for isodistances distance = "length"
    iso_dict = {}
    for distance_value in distance_values:
        subgraph = nx.ego_graph(
            G, road_node, radius=distance_value, distance=distance_type
        )
        node_points = [
            Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
        ]
        bounding_poly = gpd.GeoSeries(node_points).unary_union
        # bounding_poly = Polygon([i for i in node_points])
        iso_dict[str(distance_value)] = bounding_poly

    return iso_dict
