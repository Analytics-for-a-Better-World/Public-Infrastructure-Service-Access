'''
Network loading utilities.
'''

from time import perf_counter as pc
import warnings

import geopandas as gpd
import pandas as pd
import pandana as pdna
from pyrosm import OSM


def build_pandana_network(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> pdna.Network:
    '''Build a Pandana network from prepared nodes and edges.'''

    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        module='pandana.network',
        message='Unsigned integer: shortest path distance is trying to be calculated.*',
    )

    return pdna.Network(
        node_x=nodes['lon'],
        node_y=nodes['lat'],
        edge_from=edges['u'],
        edge_to=edges['v'],
        edge_weights=edges[['length']],
    )


def load_osm_network(
    pbf_path: str,
    verbose: bool = True,
) -> tuple[pdna.Network, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''
    Load OSM driving network and build Pandana network.

    Parameters
    ----------
    pbf_path
        Path to OSM PBF.
    verbose
        Whether to print timing information.

    Returns
    -------
    tuple[pdna.Network, gpd.GeoDataFrame, gpd.GeoDataFrame]
        Pandana network, nodes, edges.
    '''
    t0 = pc()

    osm = OSM(str(pbf_path))
    nodes, edges = osm.get_network(network_type='driving', nodes=True)

    nodes = nodes.copy()
    edges = edges.copy()

    nodes['id'] = pd.to_numeric(nodes['id'], errors='raise').astype('int64')
    nodes = nodes.drop_duplicates(subset='id').set_index('id', drop=False)

    edges['u'] = pd.to_numeric(edges['u'], errors='coerce')
    edges['v'] = pd.to_numeric(edges['v'], errors='coerce')
    edges['length'] = pd.to_numeric(edges['length'], errors='coerce')
    edges = edges.dropna(subset=['u', 'v', 'length']).copy()
    edges['u'] = edges['u'].astype('int64')
    edges['v'] = edges['v'].astype('int64')
    edges['length'] = edges['length'].astype('float64')

    valid_node_ids = set(nodes.index)
    edges = edges.loc[
        edges['u'].isin(valid_node_ids) & edges['v'].isin(valid_node_ids)
    ].copy()

    network = build_pandana_network(nodes=nodes, edges=edges)

    if verbose:
        print(
            f'Loaded network in {pc() - t0:.2f} seconds, '
            f'{len(nodes):,} nodes, {len(edges):,} edges'
        )

    return network, nodes, edges
