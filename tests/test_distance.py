import pickle
import time
import pandas as pd
import networkx as nx
import numpy as np
import pytest
import geopandas as gpd
from gpbp.distance import population_served
from gpbp.layers import AdmArea

@pytest.fixture
def population_data():
    # This population data is made up but located around Dili, Timor-Leste
    pop_gdf = pd.DataFrame(
        {
            "ID": [0, 1, 2],
            "longitude" : [125.5, 125.7, 124.8],
            "latitude" : [-8.6, -8.7, -9.1],
            "population": [0.1, 0.2, 0.05],
        }
    )
    pop_gdf = gpd.GeoDataFrame(pop_gdf, geometry=gpd.points_from_xy(pop_gdf.longitude, pop_gdf.latitude))
    return pop_gdf

@pytest.fixture
def facilities_data():
    # These are the Timor-Leste Hospitals with IDs 8 and 19
    fac_gdf = pd.DataFrame(
        {
            "ID": [0, 1],
            "osmid": [1258236683.0, 1150722282.0],
            "longitude" : [125.52661711024034, 126.99755338864472],
            "latitude" : [-8.557962051100692, -8.5238558567869],
            "geometry": [None, None],
            # Geometry is dropped in the first step of population_served anyways and never used again (except ID column).
        }
    )
    return fac_gdf

@pytest.fixture
def road_network():
    # I need the road network and cannot just hand-make that data.
    return pickle.load(open("tests/test_assets/road_network_TLS.pkl", "rb"))

def test_population_served_invalid_strategy(population_data, facilities_data):
    with pytest.raises(Exception) as exc_info:
        population_served(
            pop_gdf=population_data,
            fac_gdf=facilities_data,
            data_as_key="facilities",
            distance_type="length",
            distance_values=[1000, 2000],
            route_mode="drive",
            strategy="XXX",
            access_token=None,
            road_network=None,
        )

    assert "Invalid strategy" in str(exc_info.value)

# This parameterization works, but would need to provide a valid access token for mapbox.
# Instead, I guess we should mock the API call to mapbox and return some pre-saved data.
# Also: the number of different parameters means the tests take now 50 seconds to run (on my machine), about 10x longer than with only one case.
@pytest.mark.parametrize(
        "distance_type, distance_values, strategy, access_token",
        [
            ("length", [0.1, 1000], "osm", None),
            ("travel_time", [10], "osm", None),
            # ("mapbox", None),
        ]
)
def test_population_served(population_data, facilities_data, road_network, distance_type, distance_values, strategy, access_token):
    serve_df = population_served(
        pop_gdf=population_data,
        fac_gdf=facilities_data,
        data_as_key="facilities",
        distance_type=distance_type,
        distance_values=distance_values,
        route_mode="drive", # route_mode is irrelevant for strategy osm
        strategy=strategy,
        access_token=access_token,
        road_network=road_network,
    )

    # Add different assert statements here
    id_names = [f"ID_{value}" for value in distance_values]
    first_actual_value = serve_df[id_names[0]][0]
    assert all(id in serve_df.columns for id in id_names), "The columns in serve_df are not named correctly"
    assert serve_df.shape == (facilities_data.shape[0], len(id_names)+1), "The shape of serve_df is not correct"
    assert serve_df["Cluster_ID"].equals(facilities_data["ID"]), "The ID column in serve_df is not correct"
    assert type(first_actual_value) == list and len(first_actual_value) <= len(population_data), "The values in serve_df should be lists no longer than the population data"
    assert all(household_id in population_data["ID"] for household_id in first_actual_value), "Some household IDs in serve_df are not in the population data"
    if "ID_0.1" in serve_df.columns:
        assert all(len(household_ids) == 0 for household_ids in serve_df["ID_0.1"]), "The 0.1 distance should not have any households"


@pytest.mark.skip(reason="legacy test")
def test_isopolygons_length():
    road_network = pickle.load(
        open(
            r"C:\Users\EiriniK\Documents\repos\Public-Infrastructure-Location-Optimiser\examples\crete_network.pickle",
            "rb",
        )
    )
    adm_area = AdmArea(country="Greece", level=2)
    adm_area.get_adm_area(adm_name="Crete")
    adm_area.get_facilities(method="osm", tags={"building": "hospital"})
    adm_area.get_population(method="world_pop")
    mapbox_dict = population_served(
        adm_area.pop_df,
        adm_area.fac_gdf,
        "facilities",
        "length",
        2000,
        "driving",
        strategy="mapbox",
    )
    osm_dict = population_served(
        adm_area.pop_df,
        adm_area.fac_gdf,
        "facilities",
        "length",
        2000,
        "drive",
        strategy="osm",
        road_network=road_network,
    )
    differences = []
    for i in mapbox_dict:
        differences.append(
            len(mapbox_dict[i])
            - len(set(mapbox_dict[i]).intersection(set(osm_dict[i])))
        )

    diff = np.mean(differences)
    print(diff)
    assert mapbox_dict == osm_dict


@pytest.mark.skip(reason="legacy test")
def test_polygons_time():
    road_network = pickle.load(
        open(
            r"C:\Users\EiriniK\Documents\repos\Public-Infrastructure-Location-Optimiser\examples\crete_network.pickle",
            "rb",
        )
    )
    adm_area = AdmArea(country="Greece", level=2)
    adm_area.get_adm_area(adm_name="Crete")
    adm_area.get_facilities(method="osm", tags={"building": "hospital"})
    adm_area.get_population(method="world_pop")
    mapbox_dict = population_served(
        adm_area.pop_df,
        adm_area.fac_gdf,
        "facilities",
        "time",
        15,
        "driving",
        strategy="mapbox",
    )
    t0 = time.time()
    osm_dict = population_served(
        adm_area.pop_df,
        adm_area.fac_gdf,
        "facilities",
        "time",
        15,
        "drive",
        strategy="osm",
        road_network=road_network,
        default_speed=50,
    )
    t1 = time.time()
    differences = []
    for i in mapbox_dict:
        differences.append(
            len(mapbox_dict[i])
            - len(set(mapbox_dict[i]).intersection(set(osm_dict[i])))
        )

    diff = np.mean(differences)
    osm_time = t1 - t0
    print(diff)
    print(osm_time)
    assert mapbox_dict == osm_dict
