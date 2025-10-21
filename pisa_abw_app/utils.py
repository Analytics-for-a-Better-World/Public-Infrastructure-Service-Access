from typing import Type

import folium
import streamlit as st

from pisa_abw.isopolygons import IsopolygonCalculator, MapboxIsopolygonCalculator, OsmIsopolygonCalculator


@st.cache_data(show_spinner=False)
def get_set_of_available_pyomo_solvers():
    """Get the set of all available Pyomo solvers on the system.

    This function executes a shell command to query Pyomo for available solvers and parses the output to extract valid
    solver names.

    Returns
    -------
    set
        Set of available Pyomo solver names as strings

    Notes
    -----
    - Uses Streamlit's cache_data decorator to avoid repeated system calls
    - Prints status messages to indicate when scanning starts and completes
    - Uses subprocess to execute the 'pyomo help --solvers' command
    - Filters the output to include only valid solver entries
    """
    print("scanning pyomo solvers...", end="", flush=True)
    import subprocess

    shell_command = "pyomo help --solvers"
    output = subprocess.check_output(shell_command, shell=True).decode()
    print(" done.")

    def is_valid_solver_line(line):
        stripped_line = line.strip()
        return stripped_line.startswith("+") and not stripped_line.endswith(")")

    return {line.strip()[1:] for line in output.split() if is_valid_solver_line(line)}


def get_available_solvers():
    """Get a filtered list of available optimization solvers.

    This function provides a list of available solvers that are both:
    1. In the predefined set of candidate solvers (known to work well with the app)
    2. Currently installed and available on the system

    Returns
    -------
    list
        Alphabetically sorted list of available solver names as strings

    Notes
    -----
    The candidate solvers are a predefined set of commonly used and compatible optimization solvers for facility location
     problems:
    - appsi_gurobi: Gurobi solver using the APPSI interface
    - appsi_highs: HiGHS solver using the APPSI interface
    - cbc: COIN-OR CBC solver
    - cplex: IBM CPLEX solver
    - cplex_direct: Direct interface to IBM CPLEX
    - glpk: GNU Linear Programming Kit
    - gurobi: Gurobi solver
    - gurobi_direct: Direct interface to Gurobi

    The function filters this set to include only solvers that are actually available on the current system.
    """
    candidate_solvers = {
        "appsi_gurobi",
        "appsi_highs",
        "cbc",
        "cplex",
        "cplex_direct",
        "glpk",
        "gurobi",
        "gurobi_direct",
    }
    scanned = get_set_of_available_pyomo_solvers()
    return sorted(candidate_solvers & scanned)


def init_session_state(ss):
    """Initialize the Streamlit session state with default values.

    This function sets up the default values for the Streamlit session state variables used throughout the application.
    If a variable already exists in the session state, it is not overwritten.

    Parameters
    ----------
    ss : streamlit.SessionState
        The Streamlit session state object

    Returns
    -------
    None
        This function modifies the session state in-place

    Notes
    -----
    The initialized session state variables include:
    - adm_area: Selected administrative area (None by default)
    - adm_areas_str: List of administrative area names (empty by default)
    - available_solvers: List of available optimization solvers
    - existing_facilities_df: DataFrame for existing facilities (None by default)
    - fac_map_obj: Folium map for displaying facilities (centered on London by default)
    - pop_map_obj: Folium map for displaying population (centered on London by default)
    - potential_facilities_gdf: GeoDataFrame for potential facilities (None by default)
    - population_gdf: GeoDataFrame for population data (None by default)

    Default map center is set to London (51.509865, -0.118092) with zoom level 1.
    """
    defaults = {
        "adm_area": None,
        "adm_areas_str": [],
        "available_solvers": get_available_solvers(),
        "existing_facilities_df": None,
        "fac_map_obj": folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        "pop_map_obj": folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        "potential_facilities_gdf": None,
        "population_gdf": None,
        "road_network": None,
        "road_network_map_obj": folium.Map(location=(51.509865, -0.118092), zoom_start=1),
        "strategy": None,
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value


def get_isopolygon_calculator(strategy: str, ss) -> tuple[Type[IsopolygonCalculator], dict]:
    """Get the appropriate isopolygon calculator class and configuration based on strategy.

    This function returns an appropriate isopolygon calculator class and its configuration parameters based on the
    selected calculation strategy (e.g., Mapbox API or OpenStreetMap).

    Parameters
    ----------
    strategy : str
        The isopolygon calculation strategy, either "mapbox" or "osm"
    ss : streamlit.SessionState
        The Streamlit session state object containing configuration parameters:
        - distance_type: Type of distance ("length" or "travel_time")
        - distance_values: List of distance thresholds with units (e.g., "10 minutes")
        - network_type: Mode of transport ("driving", "walking", "cycling")
        - mapbox_api_token: API token (only for "mapbox" strategy)
        - road_network: Road network graph (only for "osm" strategy)

    Returns
    -------
    tuple[Type[IsopolygonCalculator], dict]
        A tuple containing:
        - The isopolygon calculator class to instantiate
        - A dictionary of parameters to pass to the calculator constructor
    """
    if strategy == "mapbox":
        return MapboxIsopolygonCalculator, {
            "distance_type": ss.distance_type,
            "distance_values": [int(x.split()[0]) for x in ss.distance_values],
            "mode_of_transport": ss.network_type,
            "mapbox_api_token": ss.mapbox_api_token,
        }
    elif strategy == "osm":
        return OsmIsopolygonCalculator, {
            "distance_type": ss.distance_type,
            "distance_values": [int(x.split()[0]) for x in ss.distance_values],
            "road_network": ss.road_network,
        }
