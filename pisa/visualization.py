from typing import Optional

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from matplotlib import cm
from matplotlib.colors import to_hex
from shapely.geometry import MultiPolygon, Polygon


def plot_facilities(
    df_facilities: pd.DataFrame, admin_area_boundaries: MultiPolygon | Polygon, tiles="OpenStreetMap"
) -> folium.Map:
    """Plot facilities on a map with administrative area boundaries."""

    # Initialize the map
    start_coords = _start_coordinates_from_admin_area(admin_area_boundaries)
    folium_map = folium.Map(location=start_coords, tiles=tiles)

    # Add a polygon layer for the administrative area boundaries
    def style_function(x):
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
    except:
        print("No facilities found")
    return folium_map


def plot_population_heatmap(
    df_population: pd.DataFrame, admin_area_boundaries: MultiPolygon | Polygon, tiles="OpenStreetMap"
) -> folium.Map:
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
    random_sample_n: Optional[int] = None,
    tiles="OpenStreetMap",
) -> folium.Map:
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


def plot_isochrones(df_isopolygons: pd.DataFrame, admin_area_boundaries: MultiPolygon | Polygon, tiles="OpenStreetMap"):
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
    folium.Marker(location=start_coords)

    return folium_map


def _start_coordinates_from_admin_area(admin_area_boundaries: MultiPolygon | Polygon) -> list:
    """Identify the start coordinates for the map."""
    return list(admin_area_boundaries.centroid.coords)[0][::-1]


def _bounding_box_from_admin_area(admin_area_boundaries: MultiPolygon | Polygon) -> list:
    """Identify the bounding box for the map."""
    bounds = admin_area_boundaries.bounds
    return [
        [bounds[1], bounds[0]],  # southwest
        [bounds[3], bounds[2]],  # northeast
    ]
