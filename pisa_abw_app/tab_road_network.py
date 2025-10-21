import streamlit as st
from streamlit_folium import st_folium

from pisa_abw.constants import VALID_DISTANCE_TYPES, VALID_MODES_OF_TRANSPORT
from pisa_abw.isopolygons import OsmIsopolygonCalculator
from pisa_abw.osm_road_network import OsmRoadNetwork
from pisa_abw.visualisation import plot_isochrones


def road_network(ss):
    st.subheader("Road Network")
    st.radio(
        "Mode of transport",
        options=sorted(VALID_MODES_OF_TRANSPORT),
        horizontal=True,
        key="network_type",
    )
    st.radio(
        "Distance measure",
        options=sorted(VALID_DISTANCE_TYPES),
        horizontal=True,
        key="distance_type",
    )
    st.multiselect(
        "Distance values",
        options=[
            "2000 meters",
            "5000 meters",
            "8000 meters",
            "10000 meters",
            "12000 meters",
            "15000 meters",
        ]
        if ss.distance_type == "length"
        else ["10 min", "20 min", "30 min", "40 min", "50 min", "60 min"],
        max_selections=4,
        key="distance_values",
    )
    road_button = st.button("Get OSM data", key="road_osm_button")

    if road_button:
        ss.road_network = OsmRoadNetwork(
            ss.admin_area_boundaries, ss.network_type, ss.distance_type
        ).get_osm_road_network()
        st.success("OSM road network retrieved.")

        isopolygons = OsmIsopolygonCalculator(
            ss.existing_facilities_df,
            ss.distance_type,
            [int(x.split()[0]) for x in ss.distance_values],  # distance values must be a list of integers
            ss.road_network,
        ).calculate_isopolygons()

        ss.road_network_map_obj = plot_isochrones(isopolygons, ss.admin_area_boundaries)

    st_folium(
        ss.road_network_map_obj,
        width=500,
        height=500,
        key="road_network_map_obj",
    )
