from osmxtract import overpass
import geopy.distance
import pandana
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon


def get_length_edge(x):
    lon_x = float(x["from_x"])
    lat_x = float(x["from_y"])
    lon_y = float(x["to_x"])
    lat_y = float(x["to_y"])
    dist = geopy.distance.geodesic((lat_x, lon_x), (lat_y, lon_y))
    return dist.meters


def get_road_geometries_overpass(
    geometry: MultiPolygon, network_type: str = "drive", timeout: int = 2000
):
    bounds = geometry.bounds
    # Query needs latitude first
    bounds = (bounds[1], bounds[0], bounds[3], bounds[2])
    main_road_types = [
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "road",
    ]
    if network_type == "driving":
        values = main_road_types + ["motorway", "trunk"]
    elif network_type == "walking":
        values = main_road_types + ["path", "footway", "pedestrian"]
    elif network_type == "cycling":
        values = main_road_types + ["path", "cycleway"]
    else:
        raise Exception("Invalid network type")

    query = overpass.ql_query(bounds, tag="highway", values=values, timeout=timeout)
    response = overpass.request(query)
    geofeatures = overpass.as_geojson(response, "linestring")
    gdf = gpd.GeoDataFrame.from_features(geofeatures)
    return gdf


def get_road_network_overpass(
    geometry: MultiPolygon,
    network_type: str = "driving",
    timeout: int = 2000,
    rounding: int = 5,
    graph_type: str = "pandana",
):
    """Use geopandas to read line shapefile and compile all paths and nodes in a line file based on a rounding tolerance.
    geometry: geometry of the area to get road network
    rounding: tolerance parameter for coordinate precision"""
    print("Building network")
    edges = get_road_geometries_overpass(geometry, network_type, timeout)
    edges["from_x"] = edges["geometry"].apply(lambda x: round(x.coords[0][0], rounding))
    edges["from_y"] = edges["geometry"].apply(lambda x: round(x.coords[0][1], rounding))
    edges["to_x"] = edges["geometry"].apply(lambda x: round(x.coords[-1][0], rounding))
    edges["to_y"] = edges["geometry"].apply(lambda x: round(x.coords[-1][1], rounding))
    nodes_from = edges[["from_x", "from_y"]].rename(
        index=str, columns={"from_x": "x", "from_y": "y"}
    )
    nodes_to = edges[["to_x", "to_y"]].rename(
        index=str, columns={"to_x": "x", "to_y": "y"}
    )
    nodes = pd.concat([nodes_from, nodes_to], axis=0)
    nodes["xy"] = list(zip(nodes["x"], nodes["y"]))
    nodes = pd.DataFrame(nodes["xy"].unique(), columns=["xy"])
    nodes["x"] = nodes["xy"].apply(lambda x: x[0])
    nodes["y"] = nodes["xy"].apply(lambda x: x[1])
    nodes = nodes[["x", "y"]].copy()
    nodes = nodes.reset_index()
    nodes.columns = ["nodeID", "lon", "lat"]
    edges_attr = pd.merge(
        edges, nodes, left_on=["from_x", "from_y"], right_on=["lon", "lat"]
    )
    edges_attr = pd.merge(
        edges_attr, nodes, left_on=["to_x", "to_y"], right_on=["lon", "lat"]
    )
    edges_attr.rename(
        columns={"nodeID_x": "node_start", "nodeID_y": "node_end"}, inplace=True
    )
    edges_attr["length"] = edges_attr[["from_x", "from_y", "to_x", "to_y"]].apply(
        get_length_edge, axis=1
    )

    edges_attr = edges_attr.reset_index()

    # Road Network Data in Nodes and Edges nodes as a Pandana Network
    if graph_type == "pandana":
        network = pandana.Network(
            nodes["lon"],
            nodes["lat"],
            edges_attr["node_start"],
            edges_attr["node_end"],
            edges_attr[["length"]],
            twoway=True,
        )
    elif graph_type == "networkx":
        network = nx.from_pandas_edgelist(
            df=edges_attr,
            source="node_start",
            target="node_end",
            edge_attr=["length", "maxspeed", "geometry"],
            create_using=nx.MultiDiGraph,
            edge_key="index",
        )
        network.graph["crs"] = "EPSG:4326"
    return nodes, edges_attr, network
