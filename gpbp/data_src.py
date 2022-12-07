import os
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask as riomask
import requests
import osmnx as ox
from hdx.api.configuration import Configuration
from hdx.data.resource import Resource

# from layers import AdmArea
from shapely.geometry import Polygon, MultiPolygon

# Population data sources


def get_admarea_mask(
    vector_polygon: MultiPolygon, raster_layer: rasterio.DatasetReader
) -> np.ndarray:
    """
    Extract mask from raster for a given polygon
    """
    gtraster, bound = riomask.mask(
        raster_layer, [vector_polygon], all_touched=True, crop=False
    )
    # Keep only non zero values
    adm_mask = gtraster[0] > 0
    return adm_mask


def raster_to_df(raster_fpath: str, mask_polygon: MultiPolygon) -> pd.DataFrame:
    """
    Convert raster file to a dataframe of longitude, latitude
    and statistical population count
    """
    # Convert raster file to dataframe
    src = rasterio.open(raster_fpath)
    # Create 2D arrays
    xmin, ymax = np.around(src.xy(0.00, 0.00), 9)
    xmax, ymin = np.around(src.xy(src.height - 1, src.width - 1), 9)
    x = np.linspace(xmin, xmax, src.width)
    y = np.linspace(ymax, ymin, src.height)
    xs, ys = np.meshgrid(x, y)
    zs = src.read(1)
    # Adm area mask
    mask = get_admarea_mask(mask_polygon, src)
    xs, ys, zs = xs[mask], ys[mask], zs[mask]
    data = {
        "longitude": pd.Series(xs),
        "latitude": pd.Series(ys),
        "population": pd.Series(zs),
    }
    # Create X,Y,Z DataFrame
    df = pd.DataFrame(data=data)
    src.close()
    return df


def world_pop_data(country_iso3: str, geometry: MultiPolygon) -> pd.DataFrame:
    """
    Get latest worldpop data for an area defined by the MultiPolygon geometry
    """
    worldpop_url = (
        f"https://www.worldpop.org/rest/data/pop/wpgpunadj/?iso3={country_iso3}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }  # User-Agent is required to get API response
    response = requests.get(worldpop_url, headers=headers)

    # Extract url for data for the latest year
    data = response.json()["data"][-1]
    url = data["files"][0]
    filehandle, _ = urllib.request.urlretrieve(url)
    print(f"Data downloaded")
    # Convert raster file to dataframe

    print(f"Converting raster file to dataframe")
    df = raster_to_df(filehandle, geometry)
    return df


def fb_pop_data(country_iso3: str, geometry: MultiPolygon) -> pd.DataFrame:
    """
    Get 2020 facebook data for an area defined by the MultiPolygon geometry
    """
    try:
        Configuration.create(
            hdx_site="prod", user_agent="Get_Population_Data", hdx_read_only=True
        )
    except:
        pass
    resource = Resource.search_in_hdx(
        f"name:{country_iso3.lower()}_general_2020_csv.zip"
    )
    url = resource[0]["download_url"]
    filehandle, _ = urllib.request.urlretrieve(url)
    print("Data downloaded")
    df = pd.read_csv(filehandle, compression="zip")
    bounds = list(map(lambda x: round(x, 6), geometry.bounds))
    df = df.query(f"{bounds[0]} <= longitude <= {bounds[2]}")
    df = df.query(f"{bounds[1]} <= latitude <= {bounds[3]}")
    print("Loading data to dataframe")
    df = df.rename(columns={f"{country_iso3.lower()}_general_2020": "population"})
    return df


def rwi_data(country_name: str, geometry: MultiPolygon) -> pd.DataFrame:
    """
    Get Facebook Relative Wealth index data defined by the MultiPolygon geometry
    """
    try:
        Configuration.create(
            hdx_site="prod", user_agent="Get_RWI_Data", hdx_read_only=True
        )
    except:
        pass
    resource = Resource.search_in_hdx(
        f"name:{country_name.lower()}_relative_wealth_index.csv"
    )
    url = resource[0]["download_url"]
    filehandle, _ = urllib.request.urlretrieve(url)
    print("Data downloaded")
    df = pd.read_csv(filehandle)
    bounds = list(map(lambda x: round(x, 6), geometry.bounds))
    df = df.query(f"{bounds[0]} <= longitude <= {bounds[2]}")
    df = df.query(f"{bounds[1]} <= latitude <= {bounds[3]}")
    print("Loading data to dataframe")
    return df


# Facilities data sources


def osm_facilities(
    adm_name: str, geometry: MultiPolygon, tags: dict
) -> gpd.GeoDataFrame:
    """
    Retrieve facilities specified by the tags parameter
    for an area defined by the MultiPolygon geometry
    """
    print(f"Retrieving {tags} for {adm_name} area")
    gdf = ox.geometries_from_polygon(polygon=geometry, tags=tags)
    osmids = gdf.index.get_level_values("osmid")
    lon, lat = [], []
    for index, data in gdf.iterrows():
        if index[0] == "node":
            lon.append(data["geometry"].x)
            lat.append(data["geometry"].y)
        else:
            lon.append(data["geometry"].centroid.x)
            lat.append(data["geometry"].centroid.y)
    gdf = gpd.GeoDataFrame(
        data={"osmid": osmids, "longitude": lon, "latitude": lat},
        geometry=gdf.geometry.values,
    )
    gdf = gdf.reset_index().rename(columns={"index": "ID"})
    return gdf
