import folium
import geopandas as gpd
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
    start_coords = list(admin_area_boundaries.centroid.coords)[0][::-1]
    folium_map = folium.Map(location=start_coords, tiles=tiles)

    # Add a polygon layer for the administrative area boundaries
    def style_function(x):
        return {
            "fillColor": "green",
            "color": "green",
            "weight": 2,
            "fillOpacity": 0.1,
        }

    folium.GeoJson(
        admin_area_boundaries,
        style_function=style_function,
    ).add_to(folium_map)

    # Fit bounding box around the administrative area
    bounds = admin_area_boundaries.bounds
    folium_map.fit_bounds(
        [
            [bounds[1], bounds[0]],  # southwest corner
            [bounds[3], bounds[2]],  # northeast corner
        ],
    )

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


def plot_population_heatmap(pop_df: pd.DataFrame, tiles="OpenStreetMap") -> folium.Map:
    start_coords = (pop_df.latitude.mean(), pop_df.longitude.mean())
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=6,
        tiles=tiles,
    )
    HeatMap(
        pop_df.reindex(["latitude", "longitude", "population"], axis="columns").values,
        min_opacity=0.1,
    ).add_to(folium.FeatureGroup(name="Heat Map").add_to(folium_map))
    folium.LayerControl().add_to(folium_map)
    return folium_map


def plot_population(pop_df: pd.DataFrame, tiles="OpenStreetMap") -> folium.Map:
    start_coords = (pop_df.latitude.mean(), pop_df.longitude.mean())
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=6,
        tiles=tiles,
    )
    pop_df["percent_rank"] = pop_df["population"].rank(pct=True)
    for _, row in pop_df.iterrows():
        folium.Circle(
            [row["latitude"], row["longitude"]],
            radius=0.5,
            color="red",
            fill=True,
            opacity=row["percent_rank"],
        ).add_to(folium_map)
    return folium_map


def plot_isochrones(isochrones: list[MultiPolygon], tiles="OpenStreetMap"):
    start_coords = list(isochrones[0].centroid.coords)[0][::-1]
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=10,
        tiles=tiles,
    )
    colors = cm.rainbow(np.linspace(0, 1, len(isochrones)))
    colors = list(map(to_hex, list(colors)))
    geo_j = gpd.GeoSeries(isochrones).to_json()

    def style_function(x):
        return {
            "fillColor": colors[int(x["id"])],
            "line_color": colors[int(x["id"])],
        }

    folium.GeoJson(data=geo_j, style_function=style_function).add_to(folium_map)
    folium.Marker(location=start_coords)
    return folium_map
