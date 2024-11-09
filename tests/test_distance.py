import pickle
import time

import networkx as nx
import numpy as np
import pytest

from gpbp.distance import population_served
from gpbp.layers import AdmArea


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
