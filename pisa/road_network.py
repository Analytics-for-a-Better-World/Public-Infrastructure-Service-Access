import networkx as nx
import osmnx as ox


def get_road_network(AdmArea, network_type: str) -> None:
    """
    Retrieve open street map road network for a network_type
    and calculate road travel time

    Parameters
    ----------
    AdmArea: AdministrativeArea
        The administrative area object, needs to contain a geometry attribute.

    network_type : string
        The network type in terms of mode of transportation.
        Valid inputs : 'driving', 'walking', 'cycling'
    """
    # Set network type and default speed based on input
    network_mapping = {
        "driving": ("drive", 50),
        "walking": ("walk", 4),
        "cycling": ("bike", 15),
    }
    if network_type not in network_mapping:
        raise ValueError("Invalid network type")
    network_type, default_speed = network_mapping[network_type]

    # Get network from osmnx
    if AdmArea.geometry is None:
        raise ValueError("AdmArea must have a geometry attribute.")
    road_network = ox.graph_from_polygon(AdmArea.geometry, network_type=network_type)

    # Add travel time edge attribute in minutes
    road_network = ox.add_edge_speeds(road_network, fallback=default_speed)
    road_network = ox.add_edge_travel_times(road_network)

    # Recast travel time to minutes
    time = nx.get_edge_attributes(road_network, "travel_time")
    time_in_minutes = {k: round(v / 60, 2) for k, v in time.items()}
    nx.set_edge_attributes(road_network, time_in_minutes, "travel_time")

    return road_network
