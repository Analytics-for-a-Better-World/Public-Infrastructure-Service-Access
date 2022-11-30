from .config import MAPBOX_API_ACCESS_TOKEN
import json
import requests
from typing import Any, Union
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import geopandas as gpd
import pandas as pd

import networkx as nx
import pandana
import osmnx as ox


def _get_poly_nx(G: nx.MultiDiGraph, road_node, dist_value, distance_type):
    subgraph = nx.ego_graph(G, road_node, radius=dist_value, distance=distance_type)

    node_points = [
        Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
    ]
    nodes_gdf = gpd.GeoDataFrame({"id": list(subgraph.nodes)}, geometry=node_points)
    nodes_gdf = nodes_gdf.set_index("id")

    edge_lines = []
    for n_fr, n_to in subgraph.edges():
        f = nodes_gdf.loc[n_fr].geometry
        t = nodes_gdf.loc[n_to].geometry
        edge_lookup = G.get_edge_data(n_fr, n_to)[0].get("geometry", LineString([f, t]))
        edge_lines.append(edge_lookup)
    edges_gdf = gpd.GeoSeries(edge_lines)
    return nodes_gdf, edges_gdf


# TODO :
def _get_poly_pandana(G: pandana.Network, road_node, dist_value, distance_type):
    nodes_gdf = G.nodes_in_range(road_node, dist_value, distance_type)


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
) -> dict:

    # make coordinates arrays if user passed non-iterable values
    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        is_scalar = True
        X = [X]
        Y = [Y]

    G = road_network
    isochrone_polys = {}
    is_networkx = False
    if isinstance(G, nx.MultiDiGraph):
        road_nodes = ox.distance.nearest_nodes(G, X, Y)
        is_networkx = True
    elif isinstance(G, pandana.Network):
        road_nodes = G.get_node_ids(X, Y, mapping_distance=None)
    else:
        raise Exception("Invalid network type")

    # Add travel time in seconds edge attribute to network
    # TODO: add travel time attribute for pandana
    if distance_type == "travel_time":
        G = ox.add_edge_speeds(G, hwy_speeds=road_speeds, fallback=default_speed)
        G = ox.add_edge_travel_times(G)

    # Construct isopolygon for each distance value
    for dist_value in distance_values:
        isochrone_polys["ID_" + str(dist_value)] = []
        if is_networkx:
            get_poly_func = _get_poly_nx
        else:
            get_poly_func = _get_poly_pandana
        for road_node in road_nodes:
            nodes_gdf, edges_gdf = get_poly_func(
                G, road_node, dist_value, distance_type
            )
            n = nodes_gdf.buffer(node_buff).geometry
            e = edges_gdf.buffer(edge_buff).geometry
            all_gs = list(n) + list(e)
            new_iso = gpd.GeoSeries(all_gs).unary_union
            new_iso = Polygon(new_iso.exterior)
            isochrone_polys["ID_" + str(dist_value)].append(new_iso)
        if is_scalar:
            isochrone_polys["ID_" + str(dist_value)] = isochrone_polys[str(dist_value)][
                0
            ]

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
    iso_dict = {"ID_" + str(dist_value): [] for dist_value in distance_values}
    base_url = "https://api.mapbox.com/isochrone/v1/"
    if distance_type == "travel_time":
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
            features = request_pack["features"]
        except:
            print("Something went wrong")
            print(request)

        for feature in features:
            iso_dict["ID_" + str(feature["properties"]["contour"])].append(
                MultiPolygon(list(map(Polygon, feature["geometry"]["coordinates"])))
            )
            if is_scalar:
                iso_dict[str(feature["properties"]["contour"])] = iso_dict[
                    str(feature["properties"]["contour"])
                ][0]

    return iso_dict


# TODO:


def population_served(
    pop_gdf: pd.DataFrame,
    fac_gdf: gpd.GeoDataFrame,
    data_as_key: str,
    distance_type: str,
    distance_values: list[int],
    route_mode: str,
    strategy: str,
    road_network: Any = None,
    road_speeds: dict = None,
    default_speed: int = None,
) -> dict:
    # TODO: route mode is useful only for mapbox if we take network as variable
    pop_gdf = pop_gdf.copy()
    iso_gdf = fac_gdf.copy().drop(columns="geometry")
    # Get isopolygons geodataframe
    if strategy == "mapbox":
        dist_dict = calculate_isopolygons_Mapbox(
            iso_gdf.longitude.to_list(),
            iso_gdf.latitude.to_list(),
            route_mode,
            distance_type,
            distance_values,
        )
        dist_df = pd.DataFrame.from_dict(dist_dict)
        iso_gdf = pd.concat(
            [iso_gdf.reset_index(drop=True), dist_df.reset_index(drop=True)], axis=1
        )
    elif strategy == "osm":
        if road_network == None:
            raise Exception("OSM strategy needs a road network")
        # OSM accepts time in seconds
        if distance_type == "travel_time":
            distance_value = distance_value * 60
        dist_dict = calculate_isopolygons_graph(
            iso_gdf.longitude.to_list(),
            iso_gdf.latitude.to_list(),
            distance_type,
            distance_values,
            road_network,
            road_speeds,
            default_speed,
        )
        dist_df = pd.DataFrame.from_dict(dist_dict)
        iso_gdf = pd.concat(
            [iso_gdf.reset_index(drop=True), dist_df.reset_index(drop=True)], axis=1
        )
    serve_dict = {}
    for value in distance_values:
        column_name = "ID_" + str(value)
        temp_iso_gdf = gpd.GeoDataFrame(
            iso_gdf[["ID", column_name]], geometry=column_name, crs="EPSG:4326"
        )
        pop_gdf = pop_gdf.set_crs(temp_iso_gdf.crs)
        temp_iso_gdf = temp_iso_gdf.dropna()
        # Find households within isopolygons
        serve_gdf = pop_gdf.sjoin(temp_iso_gdf, how="right", predicate="within")
        serve_gdf = serve_gdf.dropna()
        if data_as_key == "population":
            serve_dict[column_name] = (
                serve_gdf.groupby("ID_left", group_keys=True)["index_right"]
                .apply(list)
                .to_dict()
            )
        elif data_as_key == "facilities":
            serve_dict[column_name] = (
                serve_gdf.groupby("ID_right", group_keys=True)["index_left"]
                .apply(list)
                .to_dict()
            )
    return serve_dict
