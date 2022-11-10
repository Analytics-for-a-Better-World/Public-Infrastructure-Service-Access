import numpy as np
from shapely.geometry import MultiPolygon
import geopandas as gpd
from geopandas.tools import sjoin


def generate_grid_in_polygon(
    spacing: float, geometry: MultiPolygon
) -> gpd.GeoDataFrame:
    """This Function generates evenly spaced points within the given GeoDataFrame.
    The parameter 'spacing' defines the distance between the points in coordinate units."""

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
    )
    grid = gpd.clip(grid, geometry)

    return grid
