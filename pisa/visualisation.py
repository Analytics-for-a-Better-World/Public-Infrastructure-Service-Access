"""Data visualization module for spatial accessibility analysis.

This module provides functions for creating interactive maps and visualizations of facilities, population data,
and isopolygons (service areas). It helps in visualizing spatial relationships between facilities and population,
service coverage areas, and optimization scenarios.

The visualizations use Folium (based on Leaflet.js) to create interactive web maps that can be
displayed in notebooks, web applications, or exported as HTML files.

Examples
--------
Create interactive maps for facilities and population:

>>> from pisa.administrative_area import AdministrativeArea
>>> from pisa.facilities import Facilities
>>> from pisa.population import WorldpopPopulation
>>> from pisa.visualisation import plot_facilities, plot_population
>>>
>>> # Get administrative area and data
>>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
>>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
>>> country_code = admin_area.get_iso3_country_code()
>>>
>>> # Get facilities data
>>> facilities = Facilities(admin_area_boundaries=boundaries)
>>> existing_facilities = facilities.get_existing_facilities()
>>>
>>> # Get population data
>>> population = WorldpopPopulation(
>>>     admin_area_boundaries=boundaries,
>>>     iso3_country_code=country_code
>>> )
>>> population_gdf = population.get_population_gdf()
>>>
>>> # Create interactive maps
>>> facility_map = plot_facilities(existing_facilities, boundaries)
>>> population_map = plot_population(population_gdf, boundaries)

See Also
--------
facilities : Module for facility data processing
population : Module for population data processing
isopolygons : Module for generating service area polygons
"""

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from matplotlib import cm
from matplotlib.colors import to_hex
from shapely.geometry import MultiPolygon, Polygon


def plot_facilities(
    df_facilities: pd.DataFrame,
    admin_area_boundaries: MultiPolygon | Polygon,
    df_potential_facilities: pd.DataFrame | None = None,
    tiles="OpenStreetMap",
) -> folium.Map:
    """Plot facilities on an interactive map with administrative area boundaries.
    
    This function creates a Folium map showing existing facilities as blue circle markers, and optionally potential 
    facilities as orange circle markers, within the context of administrative area boundaries.
    
    Parameters
    ----------
    df_facilities : pandas.DataFrame
        DataFrame containing information about existing facilities. Must have columns:

            - ``latitude``: Latitude coordinate of the facility
            - ``longitude``: Longitude coordinate of the facility
    
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
        
    df_potential_facilities : pandas.DataFrame, optional
        DataFrame containing information about potential facility locations. Must have columns:

            - ``latitude``: Latitude coordinate of the potential facility
            - ``longitude``: Longitude coordinate of the potential facility
        
    tiles : str, optional
        The tile provider for the base map. Any valid Folium tile provider name can be used.
        See folium.Map documentation for available options. (default: ``OpenStreetMap``)
        
    Returns
    -------
    folium.Map
        Interactive Folium map with administrative boundaries and facility markers
    """
    # Initialize the map
    start_coords = _start_coordinates_from_admin_area(admin_area_boundaries)
    folium_map = folium.Map(location=start_coords, tiles=tiles)

    # Add a polygon layer for the administrative area boundaries
    def style_function(x) -> dict:
        return {
            "fillColor": "green",
            "color": "green",
            "weight": 2,
            "fillOpacity": 0.05,
        }

    folium.GeoJson(
        admin_area_boundaries,
        style_function=style_function,
    ).add_to(folium_map)

    # Fit bounding box around the administrative area
    bounds = _bounding_box_from_admin_area(admin_area_boundaries)
    folium_map.fit_bounds(bounds)

    # Add a marker for each facility
    try:
        for i in range(0, len(df_facilities)):
            folium.CircleMarker(
                [df_facilities.iloc[i]["latitude"], df_facilities.iloc[i]["longitude"]],
                color="blue",
                fill=True,
                radius=2,
            ).add_to(folium_map)
    except (KeyError, IndexError) as e:
        print(f"Error plotting facilities: {e}")

    # Add a marker for each potential facility
    if df_potential_facilities is not None:
        for i in range(0, len(df_potential_facilities)):
            folium.CircleMarker(
                [df_potential_facilities.iloc[i]["latitude"], df_potential_facilities.iloc[i]["longitude"]],
                color="red",
                fill=True,
                radius=2,
            ).add_to(folium_map)
    return folium_map


def plot_population_heatmap(
    df_population: pd.DataFrame, admin_area_boundaries: MultiPolygon | Polygon, tiles="OpenStreetMap"
) -> folium.Map:
    """Create a heatmap visualization of population density.
    
    This function generates an interactive Folium map displaying population density as a heatmap within the specified 
    administrative area boundaries.
    
    Parameters
    ----------
    df_population : pandas.DataFrame
        DataFrame containing population data. Must have columns:
        
            - ``latitude``: Latitude coordinate
            - ``longitude``: Longitude coordinate
            - ``population``: Population value (intensity for the heatmap)
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
    tiles : str, optional
        The tile provider for the base map. Any valid Folium tile provider name can be used.
        See folium.Map documentation for available options. (default: ``OpenStreetMap``)
        
    Returns
    -------
    folium.Map
        Interactive Folium map with population heatmap visualization
    """
    start_coords = _start_coordinates_from_admin_area(admin_area_boundaries)
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=6,
        tiles=tiles,
    )

    # Fit bounding box around the administrative area
    bounds = _bounding_box_from_admin_area(admin_area_boundaries)
    folium_map.fit_bounds(bounds)

    HeatMap(
        df_population.reindex(["latitude", "longitude", "population"], axis="columns").values,
        min_opacity=0.1,
    ).add_to(folium.FeatureGroup(name="Heat Map").add_to(folium_map))
    folium.LayerControl().add_to(folium_map)
    return folium_map


def plot_population(
    df_population: pd.DataFrame,
    admin_area_boundaries: MultiPolygon | Polygon,
    random_sample_n: int | None = None,
    tiles="OpenStreetMap",
) -> folium.Map:
    """Plot population points on an interactive map.
    
    This function creates a Folium map showing population points as circle markers, with size and opacity reflecting the 
    relative population values.
    
    Parameters
    ----------
    df_population : pandas.DataFrame
        DataFrame containing population data. Must have columns:

            - ``latitude``: Latitude coordinate
            - ``longitude``: Longitude coordinate
            - ``population``: Population value at that point
    
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
        
    random_sample_n : int, optional
        Number of population points to randomly sample and display.
        If None, all points will be displayed (can be performance-intensive for large datasets) (default: ``None``)
        
    tiles : str, optional
        The tile provider for the base map. Any valid Folium tile provider name can be used.
        See folium.Map documentation for available options. (default: ``OpenStreetMap``)
        
    Returns
    -------
    folium.Map
        Interactive Folium map with population points displayed
    """
    start_coords = _start_coordinates_from_admin_area(admin_area_boundaries)
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=6,
        tiles=tiles,
    )

    # Fit bounding box around the administrative area
    bounds = _bounding_box_from_admin_area(admin_area_boundaries)
    folium_map.fit_bounds(bounds)

    df_population["percent_rank"] = df_population["population"].rank(pct=True)
    N = len(df_population) if random_sample_n is None else random_sample_n
    for _, row in df_population.sample(N).iterrows():
        folium.Circle(
            [row["latitude"], row["longitude"]],
            radius=0.5,
            color="red",
            fill=True,
            opacity=row["percent_rank"],
        ).add_to(folium_map)
    return folium_map


def plot_isochrones(
    df_isopolygons: pd.DataFrame,
    admin_area_boundaries: MultiPolygon | Polygon,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """Plot isochrones/isopolygons for multiple facilities on an interactive map.
    
    This function creates a Folium map displaying isopolygons (areas reachable within specific travel times or distances)
    around facilities, with different colors for each facility and varying opacity for different time/distance 
    thresholds.
    
    Parameters
    ----------
    df_isopolygons : pandas.DataFrame
        DataFrame containing isopolygon geometries. Should have:

            - Index representing unique facility identifiers
            - Columns named ``ID_X`` where X is the distance/time threshold (e.g., ``ID_10`` for 10-minute isochrone)
            - Each cell contains a Shapely ``Polygon`` or ``MultiPolygon``
    
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
        
    tiles : str, optional
        The tile provider for the base map. Any valid Folium tile provider name can be used.
        See folium.Map documentation for available options. (default: ``OpenStreetMap``)
        
    Returns
    -------
    folium.Map
        Interactive Folium map with isopolygons visualized
    """
    start_coords = _start_coordinates_from_admin_area(admin_area_boundaries)
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=10,
        tiles=tiles,
    )

    col_values = [int(col.replace("ID_", "")) for col in df_isopolygons.columns]
    sorted_cols = [df_isopolygons.columns[i] for i in np.argsort(col_values)[::-1]]
    df_isopolygons = df_isopolygons.loc[:, sorted_cols]
    df_isopolygons = df_isopolygons.drop_duplicates()

    # Create colors for each FACILITY (row index)
    facility_indices = df_isopolygons.index.unique()
    colors = cm.rainbow(np.linspace(0, 1, len(facility_indices)))
    colors = list(map(to_hex, list(colors)))

    # Create a color dictionary to maintain consistent colors per facility
    facility_colors = {idx: colors[i] for i, idx in enumerate(facility_indices)}

    # Vary opacity by isochrone time
    weights = np.linspace(0.4, 0.7, len(df_isopolygons.columns))

    # Add each isopolygon column to the map separately
    for i, col in enumerate(df_isopolygons.columns):
        weight = weights[i]
        for idx, poly in df_isopolygons[col].items():
            if poly is not None:  # Skip None values
                color = facility_colors[idx]
                folium.GeoJson(
                    data=poly,
                    style_function=lambda x, color=color, weight=weight: {
                        "fillColor": color,
                        "color": color,
                        "weight": 1,
                        "fillOpacity": weight,
                    },
                ).add_to(folium_map)

    # Add marker for reference point
    folium.Marker(location=start_coords).add_to(folium_map)

    return folium_map


def plot_results(
        open_locations: list,
        current: pd.DataFrame,
        total_fac: pd.DataFrame,
        admin_area_boundaries: MultiPolygon | Polygon) -> folium.Map:
    """Plot optimization results showing existing and proposed facility locations.
    
    This function creates a Folium map displaying the results of a facility location optimization model, showing both 
    existing facilities and proposed new facilities.
    
    Parameters
    ----------
    open_locations : list
        List of facility identifiers (matching indices in total_fac) that are selected as part of the optimization 
        solution (both existing and new)
    
    current : pandas.DataFrame
        DataFrame containing information about existing facilities, with a column ``Cluster_ID`` that identifies each
        facility
        
    total_fac : pandas.DataFrame
        DataFrame containing information about all facility locations (both existing and potential). Must have:

            - Index matching the identifiers in open_locations
            - Columns ``latitude`` and ``longitude`` for facility coordinates
        
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
        
    Returns
    -------
    folium.Map
        Interactive Folium map showing optimization results with markers
        
    Notes
    -----
    - Existing facilities are shown with blue hospital icons
    - Proposed new facilities are shown with purple question mark icons
    """
    folium_map = folium.Map(
        location=(0, 0),
        zoom_start=1,
    )

    for location in open_locations:
        existing = location in current['Cluster_ID'].values
        location_data = total_fac.loc[location]

        folium.Marker(
            [location_data.latitude, location_data.longitude],
            icon=folium.Icon(
                color="blue" if existing else "darkpurple",
                icon="hospital-o" if existing else "question",
                prefix="fa",
            ),
        ).add_to(folium_map)

    # Fit bounding box around the administrative area
    bounds = _bounding_box_from_admin_area(admin_area_boundaries)
    folium_map.fit_bounds(bounds)

    return folium_map


def _start_coordinates_from_admin_area(admin_area_boundaries: MultiPolygon | Polygon) -> list:
    """Calculate the center coordinates for initializing a map from administrative area boundaries.
    
    Parameters
    ----------
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
        
    Returns
    -------
    list
        A list of [latitude, longitude] coordinates representing the centroid of the administrative area, suitable for 
        initializing a Folium map
        
    Notes
    -----
    This function calculates the centroid of the administrative area boundaries and returns the coordinates in the 
    format [latitude, longitude] required by Folium, which is the reverse of the standard [longitude, latitude] format 
    used by Shapely.
    """
    return list(admin_area_boundaries.centroid.coords)[0][::-1]


def _bounding_box_from_admin_area(admin_area_boundaries: MultiPolygon | Polygon) -> list:
    """Calculate the bounding box coordinates for a map from administrative area boundaries.
    
    Parameters
    ----------
    admin_area_boundaries : MultiPolygon or Polygon
        Shapely geometry representing the boundaries of the administrative area
        
    Returns
    -------
    list
        A list containing two coordinate pairs in the format expected by Folium's fit_bounds method:
        [[southwest_latitude, southwest_longitude], [northeast_latitude, northeast_longitude]]
        
    Notes
    -----
    This function extracts the bounding box coordinates from the administrative area boundaries
    and formats them for use with Folium's fit_bounds method. The coordinates are transformed
    from Shapely's [minx, miny, maxx, maxy] format to Folium's [[sw_lat, sw_lon], [ne_lat, ne_lon]] format.
    """
    bounds = admin_area_boundaries.bounds
    return [
        [bounds[1], bounds[0]],  # southwest [latitude, longitude]
        [bounds[3], bounds[2]],  # northeast [latitude, longitude]
    ]
