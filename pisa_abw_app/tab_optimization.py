import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events

from optimization import maxcovering as mc
from pisa_abw.population_served_by_isopolygons import get_population_served_by_isopolygons
from pisa_abw.visualisation import plot_results
from pisa_abw_app.utils import get_isopolygon_calculator


def pisa_optimization(ss):
    st.subheader("Optimization")
    with st.container():
        st.radio(
            "Tool for calculating distances",
            options=["osm", "mapbox"],
            horizontal=True,
            key="strategy",
        )
        if ss.strategy == "osm" and ss.adm_area is not None and ss.road_network is None:
            st.warning("Please set a road network in the road network tab before continuing with the osm strategy.")
        st.text_input(
            "Mapbox access token",
            disabled=not (ss.strategy == "mapbox"),
            key="mapbox_api_token",
        )
        if ss.network_type is None or ss.distance_type is None:
            st.warning("Please set a mode of transport and distance type in the road network tab before continuing.")

        max_value_pot = 250
        if ss.adm_area is not None:
            if ss.potential_facilities_gdf is not None:
                max_value_pot = ss.potential_facilities_gdf.shape[0]
        options = np.array([5, 10, 20, 50, 100, 150, 200, 250])
        if "max_value_pot" in locals() and max_value_pot < 5 and max_value_pot > 0:
            options = np.append(options, max_value_pot)
        st.selectbox(
            "Budget (max number of potential locations to be built)",
            options=options[options <= max_value_pot],
            key="budget",
        )
        solver = st.selectbox("Solver:", ss.available_solvers, key="solver")
        opt_ready = st.button("Start optimization", key="opt_ready")
        if opt_ready:
            with st.spinner(text="Preparing data for optimization..."):
                isopolygon_calculator, kwargs = get_isopolygon_calculator(ss.strategy, ss)

                ss.total_fac = pd.concat([ss.existing_facilities_df, ss.potential_facilities_gdf])
                cutoff_idx = len(ss.existing_facilities_df)

                ss.isopolygon_facilities = isopolygon_calculator(ss.total_fac, **kwargs).calculate_isopolygons()

                current = get_population_served_by_isopolygons(
                    ss.population_gdf, ss.isopolygon_facilities[0:cutoff_idx]
                )
                potential = get_population_served_by_isopolygons(
                    ss.population_gdf, ss.isopolygon_facilities[cutoff_idx:]
                )
                pop_count = ss.population_gdf.population.values

            assert set(current.keys()) == set(potential.keys())

            with st.spinner(text="Running optimization..."):
                results = dict()
                already_open = list(current.Cluster_ID)
                facs = pd.concat([current, potential]).set_index("Cluster_ID")
                mappings = facs.to_dict()
                for col in facs.columns:
                    IJ = mappings[col]
                    I = np.unique(np.concatenate(list(IJ.values())).astype(int))
                    J = np.unique(list(IJ.keys()))
                    # Transpose IJ
                    IJ = {i: [j for j in J if i in IJ[j]] for i in I}
                    results[col] = mc.OptimizeWithPyomo(
                        pop_count, I, J, IJ, already_open=already_open, budget_list=range(int(ss.budget)), solver=solver
                    )

            pdf = pd.DataFrame()
            pdf.index.name = "budget"
            sdf = pd.DataFrame()
            sdf.index.name = "budget"
            for col in results.keys():
                df = pd.DataFrame.from_dict(results[col], orient="index")
                pdf["covered"] = df.value / pop_count.sum()
                sdf["solution"] = df.solution

            ss.results = {
                "pdf": pdf,
                "sdf": sdf,
                "pop_count": pop_count,
                "current": current,
                "potential": potential,
            }

        if "results" in ss:
            st.subheader("Results")

            pdf = ss.results["pdf"]
            sdf = ss.results["sdf"]
            current = ss.results["current"]

            fig = px.line(pdf, title="Budget vs Population Covered")
            fig.update_layout(
                yaxis_title="population covered",
                legend_title_text="",
            )
            clicked_points = plotly_events(fig, click_event=True)

            if len(clicked_points) > 0:
                point = clicked_points[0]
                ss.results["selected_budget"] = point["x"]

            if "selected_budget" in ss.results:
                st.subheader("Details for selected budget")

                selected_budget = ss.results["selected_budget"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Existing facilities", ss.existing_facilities_df.shape[0])
                with col2:
                    st.metric("Selected budget", selected_budget)
                with col3:
                    percentage = round(pdf.loc[selected_budget, "covered"] * 100, 2)
                    st.metric("Population covered", f"{percentage}%")

                open_locations = sdf.loc[selected_budget].values[0]

                ss.results_map = plot_results(open_locations, current, ss.total_fac, ss.admin_area_boundaries)

                st_folium(
                    ss.results_map,
                    use_container_width=True,
                    height=500,
                    key="result_map",
                )
