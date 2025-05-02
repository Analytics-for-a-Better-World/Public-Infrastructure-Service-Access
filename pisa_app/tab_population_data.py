import streamlit as st
from streamlit_folium import st_folium

from pisa.visualisation import plot_population_heatmap
from pisa.population import WorldpopPopulation, FacebookPopulation


def population_data(ss):
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
            try:
                ss.population_gdf = FacebookPopulation(ss.admin_area_boundaries, country_code, ss.population_resolution
                                                       ).get_population_gdf()
            except ValueError:
                st.warning(
                    (
                        f"No facebook population data available for the selected country {ss.country}. "
                    ),
                    icon="⚠️",
                )
                return

        ss.pop_map_obj = plot_population_heatmap(ss.population_gdf, ss.admin_area_boundaries)

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

    st_folium(
        ss.pop_map_obj,
        width=500,
        height=500,
        key="pop_map",
    )
