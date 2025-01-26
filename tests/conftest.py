import osmnx as ox
import pytest


@pytest.fixture
def load_graphml_file(request):
    """
    Fixture to load the GraphML file from a given filepath.
    """
    return ox.load_graphml(request.param)
