import os
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from gpbp_osm.layers import AdmArea
import gpbp_osm.visualisation
from optimization import jg_opt
from functools import partial
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
    st.header("Facility data")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Existing Facilities")
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
    with col2:
        st.subheader("Potential Facilities")
        st.slider(
            "**Pick the resolution (larger values mean more locations)**",
            min_value=0.05,
            max_value=0.5,
            step=0.05,
            key="spacing",
        )
        pot_fac_button = st.button("Compute potential locations")
        if pot_fac_button:
            st.session_state.adm_area.compute_potential_fac(st.session_state.spacing)
        st.metric(
            "Number of potential locations",
            st.session_state.adm_area.pot_fac_gdf.shape[0]
            if st.session_state.adm_area is not None
            and st.session_state.adm_area.pot_fac_gdf is not None
            else 0,
        )


with tab3:
    st.subheader("Population data")

    worldpop_button = st.button("Get WorldPop data", key="worldpop_button")
    st.write("OR")
    population_data = st.file_uploader("Upload GeoJSON", key="pop_user_data")

    if worldpop_button:
        st.session_state.adm_area.get_population("world_pop")
        # st.session_state.pop_map_obj = gpbp_osm.visualisation.plot_population_heatmap(
        #    st.session_state.adm_area.pop_df
        # )
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
    st.radio(
        "Mode of transport",
        options=["driving", "walking", "cycling"],
        horizontal=True,
        key="network_type",
    )
    road_button = st.button("Get OSM data", key="road_osm_button")
    st.write("OR")
    road_data = st.file_uploader("Upload GeoJSON", key="road_user_data")

    if road_button:
        st.session_state.adm_area.get_road_network(st.session_state.network_type)

with tab5:
    st.subheader("Optimization")
    with st.container():
        strategy = st.radio(
            "Tool for calculating distances",
            options=["osm", "mapbox"],
            horizontal=True,
            key="strategy",
        )
        if (
            st.session_state.strategy == "osm"
            and st.session_state.adm_area is not None
            and st.session_state.adm_area.road_network is None
        ):
            st.warning(
                "Please set a road network from the road network tab before continuing with the osm strategy."
            )
        st.text_input(
            "Mapbox access token",
            disabled=not (st.session_state.strategy == "mapbox"),
            key="mapbox_access",
        )
        st.radio(
            "Mode of transport",
            options=["driving", "walking", "cycling"],
            horizontal=True,
            key="route_profile",
        )
        st.radio(
            "Distance measure",
            options=["length", "travel time"],
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
            if st.session_state.distance_type == "length"
            else ["10 min", "20 min", "30 min", "40 min", "50 min", "60 min"],
            max_selections=4,
            key="distance_values",
        )
        st.slider(
            "Pick the resolution of households (larger values mean more households)",
            min_value=1,
            max_value=5,
            step=1,
            key="population_resolution",
        )
        max_value_pot = 250
        if st.session_state.adm_area is not None:
            if st.session_state.adm_area.pot_fac_gdf is not None:
                max_value_pot = st.session_state.adm_area.pot_fac_gdf.shape[0]
        options = np.array([5, 10, 20, 50, 100, 150, 200, 250])
        st.multiselect(
            "Budget (number of potential locations to be built)",
            options=options[options <= max_value_pot],
            max_selections=4,
            key="budget",
        )
        st.text_input("Path to cbc optimization software", key="opt_solver_path")
        opt_ready = st.button("Start optimization", key="opt_ready")
        if opt_ready:
            with st.spinner(text="Preparing data for optimization..."):
                (
                    pop_count,
                    current,
                    potential,
                ) = st.session_state.adm_area.prepare_optimization_data(
                    distance_type=st.session_state.distance_type.replace(" ", "_"),
                    distance_values=list(
                        map(
                            lambda x: int(x.split(" ")[0]),
                            st.session_state.distance_values,
                        )
                    ),
                    mode_of_transport=st.session_state.route_profile,
                    strategy=st.session_state.strategy,
                    access_token=st.session_state.mapbox_access,
                    population_resolution=st.session_state.population_resolution,
                )
            if os.path.exists(st.session_state.opt_solver_path):
                opt_func = partial(
                    jg_opt.OpenOptimize, solver_path=st.session_state.opt_solver_path
                )
            else:
                raise Exception("Path does not exist")
            with st.spinner(text="Running optimization..."):
                values, solutions = jg_opt.Solve(
                    pop_count,
                    current,
                    potential,
                    st.session_state.distance_type.replace(" ", "_"),
                    list(map(int, st.session_state.budget)),
                    optimize=opt_func,
                    type="ID",
                )
            st.write(values)
            st.write(solutions)
