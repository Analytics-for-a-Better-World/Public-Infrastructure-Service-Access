"""Facility data retrieval and management module for accessibility analysis.

This module provides functionality for retrieving, managing, and analyzing facility location data
within specified administrative areas. It supports extracting facility data from OpenStreetMap (OSM),
generating potential facility locations through grid sampling, and preparing data for accessibility
and optimization analyses.

Examples
--------
Retrieve existing facilities and generate potential facility locations:

>>> from pisa.administrative_area import AdministrativeArea
>>> from pisa.facilities import Facilities
>>>
>>> # Get administrative area boundaries
>>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
>>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
>>>
>>> # Create a facilities object
>>> facilities = Facilities(admin_area_boundaries=boundaries)
>>>
>>> # Get existing facilities from OpenStreetMap
>>> existing = facilities.get_existing_facilities()
>>> print(f"Found {len(existing)} existing facilities")
>>>
>>> # Generate potential facility locations (grid points)
>>> potential = facilities.estimate_potential_facilities(spacing=0.05)
>>> print(f"Generated {len(potential)} potential facility locations")

See Also
--------
administrative_area : Module for retrieving administrative area boundaries
population : Module for population data processing
isopolygons : Module for calculating service areas around facilities
"""

import logging
import warnings
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx._errors import InsufficientResponseError
from pandas import DataFrame
from shapely import MultiPolygon, Polygon

from pisa.constants import OSM_TAGS

# Suppress user warning about geometry in geographic CRS. Centroid is calculated
# over a single facility (e.g. a hospital), so distances are very small and
# projection isn't necessary
warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect",
    category=UserWarning,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Facilities:
    """Retrieve and manage facility location data for a given administrative area.
    
    This class provides functionality to retrieve existing facilities within a specified administrative area from 
    various data sources (currently supporting OpenStreetMap).
    
    Parameters
    ----------
    admin_area_boundaries : Polygon or MultiPolygon
        The geographical boundaries of the administrative area for which to retrieve facilities
    data_src : str, optional
        The data source from which to retrieve facility data. Currently supported:
        - "osm": OpenStreetMap (default: ``osm``)
    osm_tags : dict, optional
        Dictionary of OpenStreetMap tags to identify facilities of interest (e.g., ``{'amenity': 'hospital'}``). (default: ``OSM_TAGS``)
        
    Notes
    -----
    The default OSM_TAGS are defined in the constants module and typically target health facilities. To use 
    different facility types, provide custom osm_tags when creating the Facilities object.
    
    See Also
    --------
    AdministrativeArea : Class for retrieving administrative area boundaries
    IsopolygonCalculator : Class for calculating service areas around facilities
    
    Examples
    --------
    >>> from pisa.administrative_area import AdministrativeArea
    >>> from pisa.facilities import Facilities
    >>> 
    >>> # Get an administrative area for a specific country and region
    >>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
    >>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
    >>> 
    >>> # Get hospital facilities within the administrative area
    >>> facilities = Facilities(admin_area_boundaries=boundaries)
    >>> existing_facilities = facilities.get_existing_facilities()
    """
    admin_area_boundaries: Polygon | MultiPolygon
    data_src: str = "osm"
    osm_tags: dict = field(default_factory=lambda: OSM_TAGS)

    def get_existing_facilities(self) -> DataFrame:
        """Retrieve existing facilities from the specified data source.
        
        This method acts as a dispatcher that calls the appropriate data source-specific method based on the `data_src` 
        attribute of the Facilities instance.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing facilities information with columns:

                - ``osmid`` (index): Facility identifier (e.g., OSM ID for OpenStreetMap data)
                - ``longitude``: Longitude coordinate of the facility
                - ``latitude``: Latitude coordinate of the facility
            
        Raises
        ------
        NotImplementedError
            If the specified data source is not implemented
            
        Notes
        -----
        Currently, only the ``osm`` (OpenStreetMap) data source is implemented. To support additional data sources,
        implement new methods and update this dispatcher method accordingly, or refactor to use an abstract base class.
        """
        if self.data_src == "osm":
            return self._get_existing_facilities_osm(
                admin_area_boundaries=self.admin_area_boundaries, osm_tags=self.osm_tags
            )
        raise NotImplementedError(f"Data source '{self.data_src}' not implemented")

    @staticmethod
    def _get_existing_facilities_osm(
        osm_tags: dict, admin_area_boundaries: Polygon | MultiPolygon
    ) -> DataFrame:
        """Fetch existing facilities from OpenStreetMap (OSM) within a specified administrative area.
        
        Parameters
        ----------
        osm_tags : dict
            Dictionary of OSM tags to filter facilities (e.g., {'amenity': 'school'})
        admin_area_boundaries : Polygon or MultiPolygon
            Geographic area within which to search for facilities, defined as a shapely Polygon or MultiPolygon object
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing facilities information with columns:
            - longitude: Longitude coordinate of the facility
            - latitude: Latitude coordinate of the facility
            
        Notes
        -----
        This method uses the OSMnx library to retrieve facilities matching the specified tags from OpenStreetMap. If no 
        facilities are found, it returns an empty DataFrame with the expected structure.
        
        In case of an InsufficientResponseError from the OSM API, an empty GeoDataFrame is created as a fallback.
        """
        logger.info(f"Retrieving existing facilities with tags {osm_tags} using OSM.")

        # retrieves facilities GeodataFrame from osm
        try:
            facilities_gdf = ox.features_from_polygon(
                polygon=admin_area_boundaries, tags=osm_tags
            )
        except InsufficientResponseError:
            facilities_gdf = gpd.GeoDataFrame(
                pd.DataFrame(columns=["id", "element", "amenity", "geometry"]),
                geometry=[],
                crs="EPSG:4326",
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
        facilities_df = facilities_df.set_index("id")[["longitude", "latitude"]]
        facilities_df.index.name = None

        return facilities_df

    def estimate_potential_facilities(self, spacing: float = 0.05) -> gpd.GeoDataFrame:
        """Create a grid of potential facility locations within the administrative area.
        
        This method generates a regular grid of points within the administrative area boundaries that can be used as 
        potential locations for new facilities in optimization scenarios.
        
        Parameters
        ----------
        spacing : float
            The distance between adjacent points in the grid, in coordinate units (degrees). Smaller values create a 
            denser grid with more potential facility locations.
            
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing potential facility locations with columns:

                - ``longitude``: Longitude coordinate of the potential facility
                - ``latitude``: Latitude coordinate of the potential facility
            
        Notes
        -----
        The grid is created by:

            1. Finding the bounding box of the administrative area
            2. Creating a regular grid covering the entire bounding box
            3. Clipping the grid to include only points within the actual administrative area boundaries
        
        The resulting grid points are spaced at regular intervals determined by the spacing parameter.
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

        # Clip the grid to the admin area boundaries - note that this is the reason why the row-index does not start at 0
        grid = gpd.clip(grid, self.admin_area_boundaries)
        grid = grid.drop(columns=["geometry"])

        # create index column numbering the grid points
        grid.index = range(len(grid))

        return grid
