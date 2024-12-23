import networkx as nx
import pytest


@pytest.mark.parametrize(
    "load_graphml_file",
    ["tests/test_data/walk_network_4_nodes_6_edges.graphml"],
    indirect=True,
)
def test_graph_loads_correctly(load_graphml_file):
    """
    This is not so much a test as it is a way to understand the graph
    that is being loaded as input for the tests.
    """

    G = load_graphml_file

    assert isinstance(G, nx.MultiDiGraph)

    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 6
