import folium
from folium.plugins import HeatMap
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon

from matplotlib import cm
from matplotlib.colors import to_hex

from pisa.administrative_area import AdministrativeArea


def plot_facilities(admin_area: AdministrativeArea, loc_gdf: gpd.GeoDataFrame, tiles="OpenStreetMap") -> folium.Map:
    start_coords = list(admin_area.centroid.coords)[0][::-1]
    folium_map = folium.Map(
        location=start_coords,
        zoom_start=6,
        tiles=tiles
    )
    try:
        for i in range(0, len(loc_gdf)):
            folium.CircleMarker(
                [loc_gdf.iloc[i]["latitude"], loc_gdf.iloc[i]["longitude"]],
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
    style_function = lambda x: {
        "fillColor": colors[int(x["id"])],
        "line_color": colors[int(x["id"])],
    }
    folium.GeoJson(data=geo_j, style_function=style_function).add_to(folium_map)
    folium.Marker(location=start_coords)
    return folium_map
