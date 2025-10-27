import streamlit as st
from streamlit_folium import st_folium

from pisa.facilities import Facilities
from pisa.visualisation import plot_facilities


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

        ss.fac_map_obj = plot_facilities(ss.existing_facilities_df, ss.admin_area_boundaries)

    if ss.adm_area and ss.existing_facilities_df is not None:
        st.metric("Number of existing facilities", ss.existing_facilities_df.shape[0])

    st_folium(
        ss.fac_map_obj,
        width=500,
        height=500,
        key="existing_facilities_map",
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

        ss.fac_map_obj = plot_facilities(
            ss.existing_facilities_df, ss.admin_area_boundaries, ss.potential_facilities_gdf
        )

        st_folium(
            ss.fac_map_obj,
            width=500,
            height=500,
            key="potential_facilities_map",
        )

    st.metric(
        "Number of potential locations",
        ss.potential_facilities_gdf.shape[0]
        if ss.adm_area is not None and ss.potential_facilities_gdf is not None
        else 0,
    )
