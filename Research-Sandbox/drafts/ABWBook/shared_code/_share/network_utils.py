import pandas as pd
import geopandas as gpd

# network_utils.py

def assign_components_to_nodes(nodes: gpd.GeoDataFrame, edges: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Assigns a connected component label to each node based on the provided edge list.

    Args:
        nodes: GeoDataFrame with an 'id' column containing node IDs.
        edges: DataFrame with 'u' and 'v' columns representing undirected edges.

    Returns:
        GeoDataFrame with an added 'component' column indicating component membership.
    """
    mapping = create_component_mapping(edges)
    nodes['component'] = nodes['id'].map(mapping).astype('category')
    return nodes


def create_component_mapping(edges: pd.DataFrame) -> dict[int, int]:
    """
    Creates a mapping from node ID to its connected component index.

    Args:
        edges: DataFrame with 'u' and 'v' columns representing undirected edges.

    Returns:
        A dictionary mapping each node ID to a connected component index.
    """
    G = nx.from_pandas_edgelist(edges, source='u', target='v', create_using=nx.Graph)
    return {
        node: idx
        for idx, component in enumerate(nx.connected_components(G))
        for node in component
    }