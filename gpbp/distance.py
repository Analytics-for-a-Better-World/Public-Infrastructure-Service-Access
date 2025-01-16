import hashlib
import json
import os
import pickle
import time
from functools import wraps
from typing import Any, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandana
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


def disk_cache(cache_dir="cache"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Create a hash key from the function name and arguments
            hash_key = hashlib.sha256()
            hash_key.update(func.__name__.encode())
            hash_key.update(pickle.dumps(args))
            hash_key.update(pickle.dumps(kwargs))
            filename = f"{cache_dir}/{hash_key.hexdigest()}.pkl"

            # Check if the cache file exists
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    return pickle.load(f)
            else:
                # Call the function and cache its result
                result = func(*args, **kwargs)
                with open(filename, "wb") as f:
                    pickle.dump(result, f)
                return result

        return wrapper

    return decorator


def _get_poly_nx(
    road_network: nx.MultiDiGraph, center_node: int, dist_value: int, distance_type: str
) -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
    """
    Get nodes and edges within a specified distance from a certain node in a road network.

    Parameters:
    road_network (nx.MultiDiGraph): The road network.
    center_node (int): The node from which to measure the distance.
    dist_value (int): The distance value.
    distance_type (str): The type of distance (e.g., 'length').

    Returns:
    - nodes_gdf: a GeoSeries of the nodes with their osmid and geometry.
    - edges_gdf: a GeoSeries of the geometry of the edges.

    If an edge (u,v) doesn't have geometry data in G, edges_gdf contains
    a straight line from u to v.

    """
    subgraph = nx.ego_graph(road_network, center_node, radius=dist_value, distance=distance_type)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
    
    return nodes_gdf.loc[:, "geometry"], edges_gdf.loc[:, "geometry"].reset_index()


# TODO : complains about input type
def _get_poly_pandana(G: pandana.Network, road_node, dist_value, distance_type):
    array = np.array([road_node], dtype=np.int_)
    nodes_gdf = G.nodes_in_range(array, dist_value, distance_type)
    nodes_gdf["geometry"] = nodes_gdf.apply(
        lambda row: Point(row["x"], row["y"], axis=1)
    )
    edge_lines = []
    for _, row in G.edges.iterrows():
        f = nodes_gdf.loc[row["from"]].geometry
        t = nodes_gdf.loc[row["to"]].geometry
        edge_lines.append(LineString([f, t]))
    edges_gdf = gpd.GeoSeries(edge_lines)
    return nodes_gdf, edges_gdf


def calculate_isopolygons_graph(
    X: Any,
    Y: Any,
    distance_type: str,
    distance_values: list[int],
    road_network: Any,
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
        raise Exception("Not implemented yet")
        road_nodes = G.get_node_ids(X, Y, mapping_distance=None)
        road_nodes = road_nodes.astype(np.intc)
    else:
        raise Exception("Invalid network type")

    # Construct isopolygon for each distance value
    for dist_value in distance_values:
        isochrone_polys["ID_" + str(dist_value)] = []
        if is_networkx:
            get_poly_func = _get_poly_nx
        #        else:
        #            get_poly_func = _get_poly_pandana
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
            edges_gdf = gpd.GeoSeries(edge_lines)
            try:
                n = nodes_gdf.buffer(node_buff).geometry
                e = edges_gdf.buffer(edge_buff).geometry
                all_gs = list(n) + list(e)
                new_iso = gpd.GeoSeries(all_gs).unary_union
                new_iso = Polygon(new_iso.exterior)
                isochrone_polys["ID_" + str(dist_value)].append(new_iso)
                if is_scalar:
                    isochrone_polys["ID_" + str(dist_value)] = isochrone_polys[
                        "ID_" + str(dist_value)
                    ][0]
            except:
                print(road_node)

    return isochrone_polys


@disk_cache("mapbox_cache")
def calculate_isopolygons_Mapbox(
    X: Any,
    Y: Any,
    route_profile: str,
    distance_type: str,
    distance_values: list[int],
    access_token: str = None,
):
    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        is_scalar = True
        X = [X]
        Y = [Y]
    iso_dict = {"ID_" + str(dist_value): [] for dist_value in distance_values}

    if access_token is None:
        raise Exception("Access token not provided")

    base_url = "https://api.mapbox.com/isochrone/v1/"
    if distance_type == "travel_time":
        contour_type = "contours_minutes"
    elif distance_type == "length":
        contour_type = "contours_meters"
    else:
        raise Exception("Invalid distance type")
    for idx, coord_pair in enumerate(list(zip(X, Y))):
        request = (
            f"{base_url}mapbox/{route_profile}/{coord_pair[0]},"
            f"{coord_pair[1]}?{contour_type}={','.join(list(map(str, distance_values)))}"
            f"&polygons=true&denoise=1&access_token={access_token}"
        )
        # Check if reached 300 api calls
        if (idx + 1) % 300 == 0:
            # Sleep for 1 minute
            print("Reached mapbox api request limit. Waiting for one minute...")
            time.sleep(300)
            print("Starting requests again")
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


def population_served(
    pop_gdf: pd.DataFrame,
    fac_gdf: gpd.GeoDataFrame,
    data_as_key: str,
    distance_type: str,
    distance_values: list[int],
    route_mode: str,
    strategy: str,
    access_token: str = None,
    road_network: Any = None,
) -> dict:
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
            access_token=access_token,
        )
        dist_df = pd.DataFrame.from_dict(dist_dict)
        iso_gdf = pd.concat(
            [iso_gdf.reset_index(drop=True), dist_df.reset_index(drop=True)], axis=1
        )
    elif strategy == "osm":
        if road_network == None:
            raise Exception("OSM strategy needs a road network")
        dist_dict = calculate_isopolygons_graph(
            iso_gdf.longitude.to_list(),
            iso_gdf.latitude.to_list(),
            distance_type,
            distance_values,
            road_network,
        )
        dist_df = pd.DataFrame.from_dict(dist_dict)
        iso_gdf = pd.concat(
            [iso_gdf.reset_index(drop=True), dist_df.reset_index(drop=True)], axis=1
        )
    else:
        raise Exception("Invalid strategy")
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
    serve_df = pd.DataFrame(index=fac_gdf["ID"].values, data=serve_dict).applymap(
        lambda d: list(map(int, d)) if isinstance(d, list) else []
    )
    serve_df = serve_df.reset_index().rename(columns={"index": "Cluster_ID"})
    return serve_df
