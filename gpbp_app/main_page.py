import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from gpbp_osm.layers import AdmArea
import gpbp_osm.visualisation
import pycountry
import folium
import pandas as pd

st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("Public Infrastructure Location Optimiser")
# st.sidebar.markdown("# Country Data")

# adm_names = st.empty()
if "adm_area" not in st.session_state:
    st.session_state["adm_area"] = None
if "adm_areas_str" not in st.session_state:
    st.session_state["adm_areas_str"] = []
if "fac_map_obj" not in st.session_state:
    st.session_state["fac_map_obj"] = folium.Map(
        location=(51.509865, -0.118092),
        zoom_start=1,
    )
if "pop_map_obj" not in st.session_state:
    st.session_state["pop_map_obj"] = folium.Map(
        location=(51.509865, -0.118092),
        zoom_start=1,
    )


countries = sorted([country.name for country in list(pycountry.countries)])
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Country Data", "Facility Data", "Population Data", "Road Network", "Optimization"]
)

with tab1:
    st.subheader("Country Data")
    country_input = st.selectbox("Country:", countries, key="country")
    adm_level = st.selectbox("Adminstrative Level Granularity:", [0, 1, 2], key="level")
    submitted_country = st.button("Submit Country")

    if submitted_country:
        st.session_state.adm_area = AdmArea(
            st.session_state.country, st.session_state.level
        )

        st.write("Choose administrative area")
        st.session_state[
            "adm_areas_str"
        ] = st.session_state.adm_area.retrieve_adm_area_names()

    # st.write(st.session_state.adm_names_str)
    st.selectbox(
        "Adminstrative Areas:",
        st.session_state["adm_areas_str"],
        key="adm_names",
    )
    submitted_admarea = st.button("Submit Administrative Area")

    if submitted_admarea:
        st.session_state.adm_area.get_adm_area(st.session_state.adm_names)
        if st.session_state.adm_area.geometry is not None:
            st.success(
                "Administrative area is set. Continue with Facility and Population data."
            )

with tab2:
    st.subheader("Facility data")
    osm_button = st.button("Get OSM data")
    st.write("OR")
    facility_data = st.file_uploader("Upload GeoJSON", key="fac_user_data")

    if osm_button:
        st.session_state.adm_area.get_facilities("osm", {"building": "hospital"})
        st.session_state.fac_map_obj = gpbp_osm.visualisation.plot_facilities(
            st.session_state.adm_area.fac_gdf
        )
    fac_map = st_folium(
        st.session_state.fac_map_obj,
        width=500,
        height=500,
        key="fac_map",
    )


with tab3:
    st.subheader("Population data")

    worldpop_button = st.button("Get WorldPop data", key="worldpop_button")
    st.write("OR")
    population_data = st.file_uploader("Upload GeoJSON", key="pop_user_data")

    if worldpop_button:
        st.session_state.adm_area.get_population("world_pop")
        st.session_state.pop_map_obj = gpbp_osm.visualisation.plot_population_heatmap(
            st.session_state.adm_area.pop_df
        )
        if (
            st.session_state.adm_area.pop_df is not None
            and st.session_state.adm_area.fac_gdf is not None
        ):
            st.success(
                "Facilities and population data retrieved. Proceed with calculation of potential location facilitites."
            )
    pop_map = st_folium(
        st.session_state.pop_map_obj,
        width=500,
        height=500,
        key="pop_map",
    )

with tab4:
    st.subheader("Road Network")
    worldpop_button = st.button("Get OSM data", key="road_osm_button")
    st.write("OR")
    population_data = st.file_uploader("Upload GeoJSON", key="road_user_data")

with tab5:
    st.subheader("Optimization")
