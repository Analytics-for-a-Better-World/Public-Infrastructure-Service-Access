import logging
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely import MultiPolygon, Polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Facilities:
    """Get existing and potential facility locations for given administrative area

    Facility locations are retrieved from OpenStreetMap (OSM) data. Consult the OSM wiki for more information on the tags used to identify facilities.

    Example usage:

    ```python
    print("test")
    ```
    """

    admin_area_boundaries: Polygon | MultiPolygon
    data_src: str = "osm"
    tags: dict = field(default_factory=lambda: {"building": "hospital"})  # we think this default should change, awaiting Joaquim's response

    def get_existing_facilities(self) -> gpd.GeoDataFrame:
        """Get facilities from specified data source"""
        if self.data_src == "osm":
            return self.__get_existing_facilities_osm()
        else:
            raise NotImplementedError(f"Data source '{self.data_src}' not implemented")

    def __get_existing_facilities_osm(self) -> gpd.GeoDataFrame:
        """Get facilities from OSM

        Retrieve facilities from OSM data using the specified tags. Facilities are returned as a GeoDataFrame with columns for osmid, longitude, latitude and geometry.

        Returns:
            GeoDataFrame containing existing facilities

        """
        logger.info(f"Retrieving existing facilities with tags {self.tags} using OSM.")
        gdf = ox.features_from_polygon(polygon=self.admin_area_boundaries, tags=self.tags)
        osmids = gdf.index.get_level_values("id")
        lon, lat = [], []
        for index, data in gdf.iterrows():
            if index[0] == "node":
                # For node geometries, use the x and y coordinates directly
                lon.append(data["geometry"].x)
                lat.append(data["geometry"].y)
            else:
                # For non-node geometries (ways), use the centroid of the geometry
                lon.append(data["geometry"].centroid.x)
                lat.append(data["geometry"].centroid.y)
        gdf = gpd.GeoDataFrame(
            data={"ID": osmids, "longitude": lon, "latitude": lat},
            geometry=gdf.geometry.values,
        )
        gdf = gdf.reset_index()

        logger.info("Successfully retrieved existing facilities.")
        return gdf

    def estimate_potential_facilities(self, spacing: float) -> gpd.GeoDataFrame:
        """Create grid of potential facility locations

        Create a grid of evenly spaced points within the given GeoDataFrame.

        Arguments:
            spacing: The distance between the points in coordinate units.

        Returns:
            GeoDataFrame containing potential facility locations

        """
        # Get the bounds of the polygon
        minx, miny, maxx, maxy = self.admin_area_boundaries.bounds

        # Square around the polygon with the min, max polygon bounds
        x_coords = list(np.arange(np.floor(minx), np.ceil(maxx + spacing), spacing))
        y_coords = list(np.arange(np.floor(miny), np.ceil(maxy + spacing), spacing))

        # Now generate the entire grid
        mesh = np.meshgrid(x_coords, y_coords)
        grid = gpd.GeoDataFrame(
            data={"longitude": mesh[0].flatten(), "latitude": mesh[1].flatten()},
            geometry=gpd.points_from_xy(mesh[0].flatten(), mesh[1].flatten()),
            crs="EPSG:4326",
        )

        # Clip the grid to the admin area boundaries
        grid = gpd.clip(grid, self.admin_area_boundaries)
        grid = grid.reset_index(drop=True).reset_index().rename(columns={"index": "ID"})

        return grid
