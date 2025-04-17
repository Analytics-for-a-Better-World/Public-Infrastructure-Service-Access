import logging
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import osmnx as ox
from pandas import DataFrame
from shapely import MultiPolygon, Polygon

from pisa.constants import OSM_TAGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Facilities:
    """Get existing and potential facility locations for given administrative area"""

    admin_area_boundaries: Polygon | MultiPolygon
    data_src: str = "osm"
    osm_tags: dict = field(
        default_factory=lambda: OSM_TAGS
    )

    def get_existing_facilities(self) -> DataFrame:
        """Get facilities from specified data source"""
        if self.data_src == "osm":
            return self._get_existing_facilities_osm(
                admin_area_boundaries=self.admin_area_boundaries, osm_tags=self.osm_tags
            )
        raise NotImplementedError(f"Data source '{self.data_src}' not implemented")

    @staticmethod
    def _get_existing_facilities_osm(
        osm_tags: dict, admin_area_boundaries: Polygon | MultiPolygon
    ) -> DataFrame:
        """
        Fetches existing facilities from OpenStreetMap (OSM) within a specified administrative area.
        Parameters
        ----------
        osm_tags : dict
            Dictionary of OSM tags to filter facilities (e.g., {'amenity': 'school'})
        administrative_area : Polygon | MultiPolygon
            Geographic area within which to search for facilities, defined as a shapely Polygon
            or MultiPolygon object
        Returns
        -------
        pandas.DataFrame
            DataFrame containing facilities information with columns:
            - osmid (index): OSM ID of the facility
            - longitude: Longitude of facility
            - latitude: Latitude of facility
        """

        logger.info(f"Retrieving existing facilities with tags {osm_tags} using OSM.")

        # retrieves facilities GeodataFrame from osm
        facilities_gdf = ox.features_from_polygon(
            polygon=admin_area_boundaries, tags=osm_tags
        )

        # from the geometry column create longitude and latitude columns,
        # independently on whether the element is node or way
        facilities_gdf["longitude"] = facilities_gdf.geometry.centroid.x
        facilities_gdf["latitude"] = facilities_gdf.geometry.centroid.y

        # reset multiindex and drop some columns. It becomes a DataFrame
        facilities_df = facilities_gdf.reset_index().drop(
            columns=["element", "amenity", "geometry"]
        )

        logger.info("Successfully retrieved existing facilities from OSM.")

        # index facilities_df is the OSMID of the facility.
        # We don't strictly need it, but it could be useful for debugging.
        return facilities_df.set_index("id").rename_axis("osmid")[
            ["longitude", "latitude"]
        ]

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
            data={
                "longitude": mesh[0].flatten(),
                "latitude": mesh[1].flatten(),
            },
            geometry=gpd.points_from_xy(mesh[0].flatten(), mesh[1].flatten()),
        )

        # Clip the grid to the admin area boundaries
        grid = gpd.clip(grid, self.admin_area_boundaries)
        grid = grid.drop(columns=["geometry"])
        grid = grid.reset_index(drop=True).reset_index().rename(columns={"index": "ID"})

        return grid
