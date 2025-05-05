import folium
import streamlit as st
from typing import Type

from pisa.isopolygons import MapboxIsopolygonCalculator, OsmIsopolygonCalculatorAlternative, IsopolygonCalculator


@st.cache_data(show_spinner=False)
def get_set_of_available_pyomo_solvers():
    print('scanning pyomo solvers...', end='', flush=True)
    import subprocess
    shell_command = "pyomo help --solvers"
    output = subprocess.check_output(shell_command, shell=True).decode()
    print(' done.')

    def is_valid_solver_line(line):
        stripped_line = line.strip()
        return stripped_line.startswith("+") and not stripped_line.endswith(")")

    return {
        line.strip()[1:]
        for line in output.split()
        if is_valid_solver_line(line)
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


def init_session_state(ss):
    defaults = {
        'adm_area': None,
        'adm_areas_str': [],
        'available_solvers': get_available_solvers(),
        'existing_facilities_df': None,
        'fac_map_obj': folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        'pop_map_obj': folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        'potential_facilities_gdf': None,
        'population_gdf': None,
        'road_network': None,
        'road_network_map_obj': folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        'strategy': None,
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value


def get_isopolygon_calculator(strategy: str, ss) -> tuple[Type[IsopolygonCalculator], dict]:
    if strategy == "mapbox":
        return MapboxIsopolygonCalculator, {
            "distance_type": ss.distance_type,
            "distance_values": [int(x.split()[0]) for x in ss.distance_values],
            "mode_of_transport": ss.network_type,
            "mapbox_api_token": ss.mapbox_api_token,
        }
    elif strategy == "osm":
        return OsmIsopolygonCalculatorAlternative, {
            "distance_type": ss.distance_type,
            "distance_values": [int(x.split()[0]) for x in ss.distance_values],
            "road_network": ss.road_network,
        }
