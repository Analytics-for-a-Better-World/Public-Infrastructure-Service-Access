import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../optimization")

from optimization import optdata as od
from optimization import maxcovering as mc

import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from gpbp.layers import AdmArea
import gpbp.visualisation
from functools import partial
import pycountry
import folium
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_plotly_events import plotly_events

def fit_to_bounding_box( 
        folium_map : folium.Map,
        lon_min: float, lat_min: float, 
        lon_max: float, lat_max: float
    ) -> folium.Map:
    folium_map.fit_bounds(((lat_min, lon_min), (lat_max, lon_max)))
    return folium_map

def GetSetOfAvailablePyomoSolvers():
    print('scanning pyomo solvers...',end='',flush=True)
    import subprocess
    shell_command = "pyomo help --solvers"
    output = subprocess.check_output(shell_command, shell=True).decode()
    print(' done.')
    return {
        line.strip()[1:]
        for line in output.split()
        if line.strip().startswith("+") and not line.strip().endswith(")")
    }

def GetAvailableSolvers():
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
    scanned=GetSetOfAvailablePyomoSolvers() 
    return sorted( candidate_solvers & scanned )
    
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
if "available_solvers" not in st.session_state:
    st.session_state["available_solvers"] = GetAvailableSolvers()
    
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

# Hardcoded strategy while OSM strategy is not working
st.session_state.strategy = "mapbox"

countries = sorted([country.name for country in list(pycountry.countries)])
tab1, tab2, tab3, tab4 = st.tabs(
    ["Country Data", "Facility Data", "Population Data", "Optimization"]
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
            st.session_state.adm_area.get_facilities("osm", {"amenity": ["hospital", "clinic"]})
            st.session_state.fac_map_obj = gpbp.visualisation.plot_facilities(
                st.session_state.adm_area.fac_gdf
            )
            st.session_state.fac_map_obj = fit_to_bounding_box( 
                st.session_state.fac_map_obj, 
                *st.session_state.adm_area.geometry.bounds 
            )
        
        if st.session_state.adm_area and st.session_state.adm_area.fac_gdf is not None:
            st.metric("Number of existing facilities", st.session_state.adm_area.fac_gdf.shape[0])
            
        fac_map = st_folium(
            st.session_state.fac_map_obj,
            width=500,
            height=500,
            key="fac_map",
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
            st.session_state.adm_area.compute_potential_fac(st.session_state.spacing)
            pot_fac_gdf = st.session_state.adm_area.pot_fac_gdf
            for i in range(0, len(pot_fac_gdf)):
                folium.CircleMarker(
                    [pot_fac_gdf.iloc[i]["latitude"], pot_fac_gdf.iloc[i]["longitude"]],
                    color="red",
                    fill=True,
                    radius=2,
                ).add_to(st.session_state.fac_map_obj)
                
            fac_map = st_folium(
                st.session_state.fac_map_obj,
                width=500,
                height=500,
                key="fac_map",
            )

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
    fb_pop_button = st.button("Get FB data", key="fb_pop_button")
    st.write("OR")
    population_data = st.file_uploader("Upload GeoJSON", key="pop_user_data")

    if worldpop_button or fb_pop_button:
        source = "world_pop" if worldpop_button else "fb_pop"
        st.session_state.adm_area.get_population(source)
        # st.session_state.pop_map_obj = gpbp_osm.visualisation.plot_population_heatmap(
        #    st.session_state.adm_area.pop_df
        # )
        
        st.session_state.pop_map_obj = gpbp.visualisation.plot_population_heatmap(st.session_state.adm_area.pop_df)    
        
        st.session_state.pop_map_obj = fit_to_bounding_box( 
                st.session_state.pop_map_obj, 
                *st.session_state.adm_area.geometry.bounds 
            )
        
        if (
            st.session_state.adm_area.pop_df is not None
            and st.session_state.adm_area.fac_gdf is not None
        ):
            st.success(
                "Facilities and population data retrieved. Proceed with calculation of potential location facilitites."
            )
    
    if st.session_state.adm_area is not None and st.session_state.adm_area.pop_df is not None:
        total_population = round(st.session_state.adm_area.pop_df.population.sum())
        st.metric("Population", f"{total_population:,}")

    pop_map = st_folium(
        st.session_state.pop_map_obj,
        width=500,
        height=500,
        key="pop_map",
    )

# Road network tab disabled as the functionality is currently not working
# with tab4:
#     st.subheader("Road Network")
#     st.radio(
#         "Mode of transport",
#         options=["driving", "walking", "cycling"],
#         horizontal=True,
#         key="network_type",
#     )
#     road_button = st.button("Get OSM data", key="road_osm_button")
#     st.write("OR")
#     road_data = st.file_uploader("Upload GeoJSON", key="road_user_data")

#     if road_button:
#         st.session_state.adm_area.get_road_network(st.session_state.network_type)

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
            "Pick the population resolution (larger values increase accuracy)",
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
        st.selectbox(
            "Budget (max number of potential locations to be built)",
            options=options[options <= max_value_pot],
            key="budget",
        )
        solver = st.selectbox( "Solver:", 
                              st.session_state["available_solvers"],
                              key="solver"
                              )
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
                    mapbox_access_token=st.session_state.mapbox_access,
                    population_resolution=st.session_state.population_resolution,
                )
            
            assert set(current.keys()) == set(potential.keys())
            
            with st.spinner(text="Running optimization..."):
                
                results = dict()
                for key in current.keys():
                    results[key] = dict()
                    already_open = list(current[key].Cluster_ID)
                    assert all( current[key].columns == potential[key].columns )
                    facs = pd.concat( [current[key], potential[key]] ).set_index('Cluster_ID')
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
                            already_open = already_open,
                            budget_list = range(int(st.session_state.budget)), solver = solver 
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

            st.session_state["results"] = {
                "pdf": pdf,
                "sdf": sdf,
                "pop_count": pop_count,
                "current": current,
                "potential": potential
            }

        if "results" in st.session_state:
            st.subheader("Results")

            results = st.session_state["results"]
            pdf = results["pdf"]
            sdf = results["sdf"]
            pop_count = results["pop_count"]
            current = results["current"]
            potential = results["potential"]

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
                fac_gdf = st.session_state.adm_area.fac_gdf
                pot_fac_gdf = st.session_state.adm_area.pot_fac_gdf

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Existing facilities", fac_gdf.shape[0])
                with col2:
                    st.metric("Selected budget", selected_budget)
                with col3:
                    percentage = round(pdf.loc[selected_budget].sum().sum() * 100, 2)
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
                    *st.session_state.adm_area.geometry.bounds 
                )

                st_folium(map, use_container_width=True, height=500, key="result_map")
