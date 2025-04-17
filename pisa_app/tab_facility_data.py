import folium
import streamlit as st
from streamlit_folium import st_folium

import gpbp.visualisation
from pisa.facilities import Facilities
from pisa_app.utils import fit_to_bounding_box


def facility_data(ss):
    st.header("Facility data")
    col1, col2 = st.columns([2, 1])

    with col1:
        existing_facilities(ss)

    with col2:
        potential_facilities(ss)


def existing_facilities(ss):
    st.subheader("Existing Facilities")
    osm_button = st.button("Get OSM data")

    if osm_button:
        ss.facilities = Facilities(admin_area_boundaries=ss.admin_area_boundaries)
        ss.existing_facilities_df = ss.facilities.get_existing_facilities()

        ss.fac_map_obj = gpbp.visualisation.plot_facilities(
            ss.existing_facilities_df
        )
        ss.fac_map_obj = fit_to_bounding_box(
            ss.fac_map_obj,
            *ss.admin_area_boundaries.bounds
        )

    if ss.adm_area and ss.existing_facilities_df is not None:
        st.metric("Number of existing facilities", ss.existing_facilities_df.shape[0])

    st_folium(
        ss.fac_map_obj,
        width=500,
        height=500,
    )


def potential_facilities(ss):
    st.subheader("Potential Facilities")
    st.slider(
        "**Pick the resolution (larger values mean fewer locations)**",
        min_value=0.01,
        max_value=0.5,
        step=0.01,
        key="spacing",
    )

    pot_fac_button = st.button("Compute potential locations")
    if pot_fac_button:
        ss.potential_facilities_gdf = ss.facilities.estimate_potential_facilities(ss.spacing)
        for i in range(0, len(ss.potential_facilities_gdf)):
            folium.CircleMarker(
                [ss.potential_facilities_gdf.iloc[i]["latitude"], ss.potential_facilities_gdf.iloc[i]["longitude"]],
                color="red",
                fill=True,
                radius=2,
            ).add_to(ss.fac_map_obj)

        st_folium(
            ss.fac_map_obj,
            width=500,
            height=500,
        )

    st.metric(
        "Number of potential locations",
        ss.potential_facilities_gdf.shape[0]
        if ss.adm_area is not None and ss.potential_facilities_gdf is not None
        else 0,
    )