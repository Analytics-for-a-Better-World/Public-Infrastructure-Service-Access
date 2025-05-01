import folium
import streamlit as st


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
        'available_solvers': get_available_solvers(),
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
