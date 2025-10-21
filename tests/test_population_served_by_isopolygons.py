import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from pisa_abw.population_served_by_isopolygons import get_population_served_by_isopolygons


@pytest.fixture
def sample_population():
    """Create sample population points with known locations.

    Points:
    - Point(0.5, 0.5) at index 0: inside poly1 and poly3
    - Point(1.5, 1.5) at index 1: inside poly2 and poly3
    - Point(2, 2) at index 2: outside all polygons
    - Point(0.5, 1.5) at index 3: inside only poly3
    - Point(5, 5) at index 4: outside all polygons (far outside)
    """
    points = [
        Point(0.5, 0.5),  # Inside poly1 and poly3
        Point(1.5, 1.5),  # Inside poly2 and poly3
        Point(2, 2),  # Outside all polygons (just outside)
        Point(0.5, 1.5),  # Inside only poly3
        Point(5, 5),  # Outside all polygons (far outside)
    ]
    data = gpd.GeoDataFrame(geometry=points, index=[0, 1, 2, 3, 4])
    return data


@pytest.fixture
def sample_isopolygons():
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    poly3 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    data = {
        "ID_10": [poly1, poly2],
        "ID_20": [poly3, None],
    }
    return pd.DataFrame(data, index=["i0", "i1"])


def test_population_served_basic(sample_population, sample_isopolygons):
    result = get_population_served_by_isopolygons(sample_population, sample_isopolygons)

    assert isinstance(result, pd.DataFrame)
    assert "Cluster_ID" in result.columns
    assert [col for col in result.columns if col.startswith("ID_")] == sample_isopolygons.columns.tolist()

    assert len(result) == len(sample_isopolygons)
    for col in result.columns:
        if col.startswith("ID_"):
            assert all(isinstance(x, list) for x in result[col])


def test_empty_inputs(sample_population, sample_isopolygons):
    empty_population = gpd.GeoDataFrame(geometry=[])
    empty_isopolygons = pd.DataFrame(columns=["ID_10", "ID_20"])

    with pytest.raises(ValueError, match="Input dataframes cannot be empty."):
        get_population_served_by_isopolygons(empty_population, sample_isopolygons)

    with pytest.raises(ValueError, match="Input dataframes cannot be empty."):
        get_population_served_by_isopolygons(sample_population, empty_isopolygons)

    with pytest.raises(ValueError, match="Input dataframes cannot be empty."):
        get_population_served_by_isopolygons(empty_population, empty_isopolygons)


def test_no_population_in_isopolygons(sample_isopolygons):
    pop = gpd.GeoDataFrame(geometry=[Point(10, 10), Point(11, 11)], index=["p3", "p4"])

    result = get_population_served_by_isopolygons(pop, sample_isopolygons)
    assert all(len(x) == 0 for x in result["ID_10"])
    assert all(len(x) == 0 for x in result["ID_20"])


def test_population_within_isopolygons(sample_population, sample_isopolygons):
    result = get_population_served_by_isopolygons(sample_population, sample_isopolygons)

    # Check ID_10 column (contains poly1 and poly2)
    # Index 'i0' should contain only point 0
    assert set(result.loc[result["Cluster_ID"] == "i0", "ID_10"].iloc[0]) == {0}
    # Index 'i1' should contain only point 1
    assert set(result.loc[result["Cluster_ID"] == "i1", "ID_10"].iloc[0]) == {1}

    # Check ID_20 column (contains poly3 and None)
    # Index 'i0' should contain points 0, 1, and 3
    assert set(result.loc[result["Cluster_ID"] == "i0", "ID_20"].iloc[0]) == {0, 1, 3}
    # Index 'i1' should be empty (as it has None in ID_20)
    assert result.loc[result["Cluster_ID"] == "i1", "ID_20"].iloc[0] == []

    # Ensure points 2 and 4 are not in any list as they're outside all relevant polygons
    for col in ["ID_10", "ID_20"]:
        for idx in result.index:
            assert 2 not in result.loc[idx, col]
            assert 4 not in result.loc[idx, col]
