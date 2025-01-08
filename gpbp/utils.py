import numpy as np
from shapely.geometry import MultiPolygon
import geopandas as gpd
import pandas as pd


def generate_grid_in_polygon(
    spacing: float, geometry: MultiPolygon
) -> gpd.GeoDataFrame:
    """
    This Function generates evenly spaced points within the given GeoDataFrame.
    The parameter 'spacing' defines the distance between the points in coordinate units.

    AnoukB dec '24:
    Function takes outer boundaries of MultiPolygon, creates a grid of points with spacing between them that
    includes the lower boundary and excludes the upper boundary.
    The generated grid is then clipped together with the original Multipolygon and only those points that are on the
    grid and within the MultiPolygon are kept and returned.
    """

    # Get the bounds of the polygon
    minx, miny, maxx, maxy = geometry.bounds

    # Square around the country with the min, max polygon bounds
    # Now generate the entire grid
    x_coords = list(np.arange(np.floor(minx), int(np.ceil(maxx)), spacing))
    y_coords = list(np.arange(np.floor(miny), int(np.ceil(maxy)), spacing))
    mesh = np.meshgrid(x_coords, y_coords)
    grid = gpd.GeoDataFrame(
        data={"longitude": mesh[0].flatten(), "latitude": mesh[1].flatten()},
        geometry=gpd.points_from_xy(mesh[0].flatten(), mesh[1].flatten()),
        crs="EPSG:4326",
    )
    grid = gpd.clip(grid, geometry)
    grid = grid.reset_index(drop=True).reset_index().rename(columns={"index": "ID"})

    return grid


def group_population(pop_df: pd.DataFrame, nof_digits: int) -> gpd.GeoDataFrame:
    population = pop_df.copy()
    population["longitude"] = population["longitude"].round(nof_digits)
    population["latitude"] = population["latitude"].round(nof_digits)

    population = (
        population.groupby(["longitude", "latitude"])["population"]
        .sum()
        .reset_index()
        .reset_index()
    )
    population["population"] = population["population"].round(2)
    population.columns = ["ID", "longitude", "latitude", "population"]
    population = population.set_geometry(
        gpd.points_from_xy(population.longitude, population.latitude)
    )
    return population
