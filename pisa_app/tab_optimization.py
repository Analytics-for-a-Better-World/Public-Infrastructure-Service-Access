import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events
import plotly.express as px

from optimization import maxcovering as mc
from pisa.constants import VALID_MODES_OF_TRANSPORT, VALID_DISTANCE_TYPES
from pisa_app.utils import fit_to_bounding_box


def pisa_optimization(ss):
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
