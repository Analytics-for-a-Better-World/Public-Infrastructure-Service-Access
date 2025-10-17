"""Administrative boundaries retrieval and management module.

This module provides functionality to access and work with administrative area boundaries at various levels 
(countries, provinces, districts, etc.) using the GADM (Global Administrative Areas) database. 
It enables the retrieval of geographic boundaries for specific administrative areas within countries, 
which is a fundamental component for spatial analysis in public infrastructure planning.

Examples
--------
Retrieve administrative boundaries for a country and its subdivisions:

>>> from pisa.administrative_area import AdministrativeArea
>>>
>>> # Get country-level boundaries
>>> country = AdministrativeArea("Timor-Leste", admin_level=0)
>>> country_boundary = country.get_admin_area_boundaries("Timor-Leste")
>>> 
>>> # Get province-level boundaries
>>> provinces = AdministrativeArea("Timor-Leste", admin_level=1)
>>> province_names = provinces.get_admin_area_names()
>>> print(f"Provinces in Timor-Leste: {', '.join(province_names)}")
>>> 
>>> # Get boundary for a specific province
>>> baucau_boundary = provinces.get_admin_area_boundaries("Baucau")

See Also
--------
facilities : Module for working with facility locations
population : Module for population data within administrative areas
"""

import logging

import pycountry
from gadm import GADMDownloader
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdministrativeArea:
    """Get the boundaries of administrative areas for a specified country.
    
    This class provides functionality to retrieve administrative area boundaries
    from the GADM (Global Administrative Areas) database for a specified country.
    
    The administrative area level is specified by an integer, where 0 is the country level, 
    1 is the next broadest level (e.g., states or provinces), and so on.
    
    Parameters
    ----------
    country_name : str
        The name of the country for which to retrieve administrative areas
    admin_level : int
        The administrative area level to retrieve (``0`` for country, ``1`` for first-level divisions, etc.)
        
    See Also
    --------
    Facilities : Class for retrieving facilities within administrative areas
    Population : Class for retrieving population data within administrative areas
    
    Examples
    --------
    >>> # Create an administrative area object for Timor-Leste at province level
    >>> timor_leste = AdministrativeArea("Timor-Leste", admin_level=1)
    >>> 
    >>> # Get a list of all administrative areas at this level
    >>> print(timor_leste.get_admin_area_names())
    >>> 
    >>> # Get the boundaries for a specific administrative area
    >>> baucau_boundaries = timor_leste.get_admin_area_boundaries("Baucau")
    >>> 
    >>> # Get the ISO3 country code for the country
    >>> timor_leste_country_code = timor_leste.get_iso3_country_code()
    """

    def __init__(
        self,
        country_name: str,
        admin_level: int,
    ):
        self.country = self._get_pycountry_from_country_name(country_name)
        self.admin_level = admin_level
        self.all_admin_areas_gdf = self._download_admin_areas(country=self.country, admin_level=self.admin_level)

    @staticmethod
    def _get_pycountry_from_country_name(country_name: str) -> object:
        """Validate country name using fuzzy matching and return pycountry object.
        
        Parameters
        ----------
        country_name : str
            Name of the country to validate
        
        Returns
        -------
        pycountry.db.Country
            Country object from the pycountry library
            
        Raises
        ------
        ValueError
            If the country name is invalid or not found
        
        Notes
        -----
        This method performs an exact match first, and if unsuccessful,
        attempts a fuzzy search to suggest possible matches.
        """
        logger.info(f"Validating country name: {country_name}")
        country = pycountry.countries.get(name=country_name)

        if country is None:
            logger.warning(f"Country name '{country_name}' not found, attempting fuzzy search")
            try:
                possible_matches = pycountry.countries.search_fuzzy(country_name)
            except LookupError as e:
                raise ValueError("Invalid form of country name") from e
            raise ValueError(f"Country not found. Possible matches: {[match.name for match in possible_matches]}")
        
        logger.info(f"Country name '{country_name}' validated successfully")
        return country
    
    @staticmethod
    def _download_admin_areas(country, admin_level: int) -> GeoDataFrame:
        """Download and return all administrative areas of specified level for a country.
        
        Parameters
        ----------
        country : pycountry.db.Country
            Country object from the pycountry library
        admin_level : int
            Administrative area level (0 for country, 1 for first-level divisions, etc.)
        
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing all administrative areas of the specified level with their
            geometries and associated attributes
            
        Notes
        -----
        This method uses the GADM (Global Administrative Areas) database version 4.0
        to retrieve administrative boundaries.
        """
        logger.info(f"Retrieving boundaries of all administrative areas of level {admin_level} for country {country.name}")
        downloader = GADMDownloader(version="4.0")
        return downloader.get_shape_data_by_country(
            country=country,
            ad_level=admin_level
        )
    
    def get_admin_area_names(self) -> list[str]:
        """Retrieve the names of all administrative areas for the specified level.
        
        Returns
        -------
        list of str
            List of administrative area names at the specified level.
            For admin_level=0, returns just the country name.
        """
        if self.admin_level == 0:
            return [self.country.name]
        
        return self.all_admin_areas_gdf[f"NAME_{self.admin_level}"].tolist()

    def get_admin_area_boundaries(self, admin_area_name: str) -> Polygon | MultiPolygon:
        """Return the boundary geometry for the specified administrative area.
        
        Parameters
        ----------
        admin_area_name : str
            Name of the administrative area to retrieve boundaries for.
            For admin_level=0, this parameter is ignored and the country boundary is returned.
        
        Returns
        -------
        shapely.geometry.Polygon or shapely.geometry.MultiPolygon
            Geometry representing the administrative area boundaries
            
        Raises
        ------
        ValueError
            If admin_area_name is not found in the available administrative areas
        """
        if self.admin_level == 0:
            return self.all_admin_areas_gdf.geometry.iloc[0]
            
        filtered = self.all_admin_areas_gdf[
            self.all_admin_areas_gdf[f"NAME_{self.admin_level}"] == admin_area_name
        ]
        
        if filtered.empty:
            raise ValueError(
                f"Administrative area '{admin_area_name}' not found. "
                f"Available areas: {self.get_admin_area_names()}"
            )
            
        return filtered.geometry.iloc[0]

    def get_iso3_country_code(self) -> str:
        """Retrieve the ISO 3166-1 alpha-3 country code for the country.
        
        Returns
        -------
        str
            The ISO 3166-1 alpha-3 country code in lowercase (e.g., ``tls`` for Timor-Leste)
        """
        return self.country.alpha_3.lower()
