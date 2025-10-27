import pycountry
import streamlit as st

from pisa.administrative_area import AdministrativeArea

countries = sorted([country.name for country in list(pycountry.countries)])


def country_data(ss):
    st.subheader("Country Data")
    st.selectbox("Country:", countries, key="country")
    st.selectbox("Administrative Level Granularity:", [0, 1, 2], key="level")
    ss.submitted_country = st.button("Submit Country")

    if ss.submitted_country:
        ss.adm_area = AdministrativeArea(ss.country, ss.level)
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
            st.success("Administrative area is set. Continue with Facility and Population data.")
