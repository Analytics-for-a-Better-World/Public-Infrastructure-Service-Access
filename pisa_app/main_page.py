import streamlit as st
import sys

from pisa_app.tab_road_network import road_network
from pisa_app.utils import init_session_state
from pisa_app.tab_country_data import country_data
from pisa_app.tab_facility_data import facility_data
from pisa_app.tab_optimization import pisa_optimization
from pisa_app.tab_population_data import population_data

sys.path.insert(0, "..")
sys.path.insert(0, "../optimization")


st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("Public Infrastructure Location Optimiser")

ss = st.session_state
init_session_state(ss)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Country Data", "Facility Data", "Population Data", "OSM Road Network", "Optimization"]
)

with tab1:
    country_data(ss)

with tab2:
    facility_data(ss)

with tab3:
    population_data(ss)

with tab4:
    road_network(ss)

with tab5:
    pisa_optimization(ss)
