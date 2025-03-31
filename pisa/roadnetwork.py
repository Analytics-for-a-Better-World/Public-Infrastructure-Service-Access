import osmnx as ox
import networkx as nx
from osmxtract import overpass
import geopy.distance
import pandana
import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon

def get_road_network(AdmArea, network_type: str) -> None:
        """
        Retrieve open street map road network for a network_type
        and calculate road travel time

        Parameters
        ----------
        network_type : string
            The network type in terms of mode of transportation.
            Valid inputs : 'driving', 'walking', 'cycling'
        """
        if network_type == "driving":
            network_type = "drive"
            default_speed = 50

        elif network_type == "walking":
            network_type = "walk"
            default_speed = 4

        elif network_type == "cycling":
            network_type = "bike"
            default_speed = 15
        else:
            raise Exception("Invalid network type")
        # Get network
        road_network = ox.graph_from_polygon(
            AdmArea.geometry, network_type=network_type
        )
        # Add travel time edge attribute in minutes
        road_network = ox.add_edge_speeds(road_network, fallback=default_speed)

        road_network = ox.add_edge_travel_times(road_network)
        time = nx.get_edge_attributes(road_network, "travel_time")
        time_in_minutes = dict(
            zip(list(time.keys()), list(map(lambda x: round(x / 60, 2), time.values())))
        )
        nx.set_edge_attributes(road_network, time_in_minutes, "travel_time")
        
        return road_network
