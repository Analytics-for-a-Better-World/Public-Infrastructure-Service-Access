from .config import MAPBOX_API_ACCESS_TOKEN
import json
import requests
from typing import Any
from shapely.geometry import Polygon, MultiPolygon, Point
import geopandas as gpd

import networkx as nx
import osmnx as ox


def calculate_isopolygons_graph(
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
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
        iso_dict[str(distance_value)] = bounding_poly

    return iso_dict


def calculate_isopolygons_Mapbox(
    coord_pair: tuple[float, float],
    route_profile: str,
    distance_type: str,
    distance_values: list[int],
):
    base_url = "https://api.mapbox.com/isochrone/v1/"
    if distance_type == "time":
        contour_type = "contours_minutes"
    elif distance_type == "distance":
        contour_type = "contours_meters"
    request = (
        f"{base_url}mapbox/{route_profile}/{coord_pair[0]},"
        f"{coord_pair[1]}?{contour_type}={','.join(list(map(str, distance_values)))}"
        f"&polygons=true&denoise=1&access_token={MAPBOX_API_ACCESS_TOKEN}"
    )
    print(request)
    try:
        request_pack = json.loads(requests.get(request).content)
    except:
        print("Something went wrong")
    features = request_pack["features"]
    iso_dict = {
        str(feature["properties"]["contour"]): MultiPolygon(
            list(map(Polygon, feature["geometry"]["coordinates"]))
        )
        for feature in features
    }
    return iso_dict


def population_served(
    iso_gdf: gpd.GeoDataFrame, pop_gdf: gpd.GeoDataFrame, resolution: float = 0.01
) -> dict:
    pass
