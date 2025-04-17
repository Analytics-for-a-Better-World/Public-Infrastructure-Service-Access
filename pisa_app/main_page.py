import folium
import numpy as np
import pandas as pd
import plotly.express as px
import pycountry
import streamlit as st
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events
import sys

import gpbp.visualisation
from optimization import maxcovering as mc
from pisa.administrative_area import AdministrativeArea
from pisa.constants import VALID_MODES_OF_TRANSPORT, VALID_DISTANCE_TYPES
from pisa.facilities import Facilities
from pisa.population import WorldpopPopulation, FacebookPopulation

sys.path.insert(0, "..")
sys.path.insert(0, "../optimization")


def fit_to_bounding_box(
        folium_map: folium.Map,
        lon_min: float, lat_min: float,
        lon_max: float, lat_max: float
) -> folium.Map:
    folium_map.fit_bounds(((lat_min, lon_min), (lat_max, lon_max)))
    return folium_map


@st.cache_data(show_spinner=False)
def get_set_of_available_pyomo_solvers():
    print('scanning pyomo solvers...', end='', flush=True)
    import subprocess
    shell_command = "pyomo help --solvers"
    output = subprocess.check_output(shell_command, shell=True).decode()
    print(' done.')
    return {
        line.strip()[1:]
        for line in output.split()
        if line.strip().startswith("+") and not line.strip().endswith(")")
    }


def get_available_solvers():
    candidate_solvers = {
        'appsi_gurobi',
        'appsi_highs',
        'cbc',
        'cplex',
        'cplex_direct',
        'glpk',
        'gurobi',
        'gurobi_direct',
    }
    scanned = get_set_of_available_pyomo_solvers()
    return sorted(candidate_solvers & scanned)


def init_session_state():
    solvers = get_available_solvers()
    defaults = {
        'available_solvers': solvers,
        'adm_area': None,
        'adm_areas_str': [],
        'fac_map_obj': folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        'pop_map_obj': folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        'existing_facilities_df': None,
        'potential_facilities_gdf': None,
        'population_gdf': None,
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value


st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("Public Infrastructure Location Optimiser")

ss = st.session_state
init_session_state()

# Hardcoded strategy while OSM strategy is not working
ss.strategy = "mapbox"

countries = sorted([country.name for country in list(pycountry.countries)])
tab1, tab2, tab3, tab4 = st.tabs(
    ["Country Data", "Facility Data", "Population Data", "Optimization"]
)

with tab1:
    st.subheader("Country Data")
    st.selectbox("Country:", countries, key="country")
    st.selectbox("Administrative Level Granularity:", [0, 1, 2], key="level")
    submitted_country = st.button("Submit Country")

    if submitted_country:
        ss.adm_area = AdministrativeArea(
            ss.country, ss.level
        )
        st.write("Choose administrative area")
        ss.adm_areas_str = ss.adm_area.get_admin_area_names()

    st.selectbox(
        "Administrative Areas:",
        ss.adm_areas_str,
        key="adm_names",
    )

    submitted_admarea = st.button("Submit Administrative Area")

    if submitted_admarea:
        ss.admin_area_boundaries = ss.adm_area.get_admin_area_boundaries(ss.adm_names)
        if ss.admin_area_boundaries is not None:
            st.success(
                "Administrative area is set. Continue with Facility and Population data."
            )

with tab2:
    st.header("Facility data")
    col1, col2 = st.columns([2, 1])

    with col1:
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

        fac_map = st_folium(
            ss.fac_map_obj,
            width=500,
            height=500,
        )

    with col2:
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

with tab3:
    st.subheader("Population data")

    ss.population_resolution = st.slider(
        "Pick the population resolution (larger values increase accuracy)",
        min_value=1,
        max_value=5,
        value=5,
        step=1,
    )

    worldpop_button = st.button("Get WorldPop data", key="worldpop_button")
    st.write("OR")
    fb_pop_button = st.button("Get FB data", key="fb_pop_button")

    if worldpop_button or fb_pop_button:
        country_code = ss.adm_area.get_iso3_country_code()

        if worldpop_button:
            ss.population_gdf = WorldpopPopulation(ss.admin_area_boundaries, country_code, ss.population_resolution
                                                   ).get_population_gdf()

        elif fb_pop_button:
            ss.population_gdf = FacebookPopulation(ss.admin_area_boundaries, country_code, ss.population_resolution
                                                   ).get_population_gdf()

        ss.pop_map_obj = gpbp.visualisation.plot_population_heatmap(ss.population_gdf)

        ss.pop_map_obj = fit_to_bounding_box(
            ss.pop_map_obj,
            *ss.admin_area_boundaries.bounds
        )

        if (
                ss.population_gdf is not None
                and ss.existing_facilities_df is not None
        ):
            st.success(
                "Facilities and population data retrieved. Proceed with calculation of potential location facilities."
            )

    if ss.adm_area is not None and ss.population_gdf is not None:
        total_population = round(ss.population_gdf.population.sum())
        st.metric("Population", f"{total_population:,}")

    pop_map = st_folium(
        st.session_state.pop_map_obj,
        width=500,
        height=500,
        key="pop_map",
    )

with tab4:
    st.subheader("Optimization")
    with st.container():
        # strategy = st.radio(
        #     "Tool for calculating distances",
        #     options=["osm", "mapbox"],
        #     horizontal=True,
        #     key="strategy",
        # )
        if (
                ss.strategy == "osm"
                and ss.adm_area is not None
                and ss.road_network is None
        ):
            st.warning(
                "Please set a road network from the road network tab before continuing with the osm strategy."
            )
        st.text_input(
            "Mapbox access token",
            disabled=not (ss.strategy == "mapbox"),
            key="mapbox_access",
        )
        st.radio(
            "Mode of transport",
            options=VALID_MODES_OF_TRANSPORT,
            horizontal=True,
            key="route_profile",
        )
        st.radio(
            "Distance measure",
            options=VALID_DISTANCE_TYPES,
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

        max_value_pot = 250
        if ss.adm_area is not None:
            if ss.potential_facilities_gdf is not None:
                max_value_pot = ss.potential_facilities_gdf.shape[0]
        options = np.array([1, 5, 10, 20, 50, 100, 150, 200, 250])
        st.selectbox(
            "Budget (max number of potential locations to be built)",
            options=options[options <= max_value_pot],
            key="budget",
        )
        solver = st.selectbox("Solver:",
                              ss.available_solvers,
                              key="solver"
                              )
        opt_ready = st.button("Start optimization", key="opt_ready")
        if opt_ready:
            with st.spinner(text="Preparing data for optimization..."):
                (
                    pop_count,
                    current,
                    potential,
                ) = ss.adm_area.prepare_optimization_data(
                    distance_type=ss.distance_type.replace(" ", "_"),
                    distance_values=list(
                        map(
                            lambda x: int(x.split(" ")[0]),
                            ss.distance_values,
                        )
                    ),
                    mode_of_transport=ss.route_profile,
                    strategy=ss.strategy,
                    mapbox_access_token=ss.mapbox_access,
                    population_resolution=ss.population_resolution,
                )

            assert set(current.keys()) == set(potential.keys())

            with st.spinner(text="Running optimization..."):

                results = dict()
                for key in current.keys():
                    results[key] = dict()
                    already_open = list(current[key].Cluster_ID)
                    assert all(current[key].columns == potential[key].columns)
                    facs = pd.concat([current[key], potential[key]]).set_index('Cluster_ID')
                    assert len(set(facs.index)) == len(facs.index)
                    mappings = facs.to_dict()
                    for col in facs.columns:
                        IJ = mappings[col]
                        I = np.unique(np.concatenate(list(IJ.values())).astype(int))
                        J = np.unique(list(IJ.keys()))
                        # Transpose IJ
                        IJ = {i: [j for j in J if i in IJ[j]] for i in I}
                        results[key][col] = mc.OptimizeWithPyomo(
                            pop_count, I, J, IJ,
                            already_open=already_open,
                            budget_list=range(int(ss.budget)), solver=solver
                        )

            pdf = pd.DataFrame()
            pdf.index.name = "budget"
            sdf = pd.DataFrame()
            sdf.index.name = "budget"
            for key in results.keys():
                for col in results[key].keys():
                    df = pd.DataFrame.from_dict(results[key][col], orient='index')
                    pdf["covered"] = df.value / pop_count.sum()
                    sdf["solution"] = df.solution

            ss.results = {
                "pdf": pdf,
                "sdf": sdf,
            }

        if "results" in ss:
            st.subheader("Results")

            results = ss.results
            pdf = results["pdf"]
            sdf = results["sdf"]

            fig = px.line(pdf, title='Budget vs Population Covered')
            fig.update_layout(
                yaxis_title="population covered",
                legend_title_text="",
            )
            clicked_points = plotly_events(fig, click_event=True)

            if len(clicked_points) > 0:
                point = clicked_points[0]
                results["selected_budget"] = point["x"]

            if "selected_budget" in results:
                st.subheader("Details for selected budget")

                selected_budget = results["selected_budget"]
                fac_gdf = ss.existing_facilities_df
                pot_fac_gdf = ss.potential_facilities_gdf

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Existing facilities", fac_gdf.shape[0])
                with col2:
                    st.metric("Selected budget", selected_budget)
                with col3:
                    percentage = round(pdf.loc[selected_budget, 'covered'] * 100, 2)
                    st.metric("Population covered", f"{percentage}%")

                open_locations = sdf.loc[selected_budget].values[0]

                map = folium.Map(
                    location=(0, 0),
                    zoom_start=1,
                )

                for location in open_locations:
                    existing = location < len(fac_gdf)
                    if existing:
                        location_data = fac_gdf.loc[location]
                    else:
                        location_data = pot_fac_gdf.loc[location - len(fac_gdf)]

                    folium.Marker(
                        [location_data.latitude, location_data.longitude],
                        icon=folium.Icon(
                            color="blue" if existing else "darkpurple",
                            icon="hospital-o" if existing else "question",
                            prefix="fa",
                        ),
                    ).add_to(map)

                map = fit_to_bounding_box(
                    map,
                    *ss.admin_area_boundaries.bounds
                )

                st_folium(map, use_container_width=True, height=500, key="result_map")
