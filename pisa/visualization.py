import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import HeatMap


def plot_facilities(loc_gdf: gpd.GeoDataFrame, tiles="OpenStreetMap") -> folium.Map:
    start_coords = (loc_gdf.latitude.mean(), loc_gdf.longitude.mean())
    folium_map = folium.Map(location=start_coords, zoom_start=6, tiles=tiles)
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
