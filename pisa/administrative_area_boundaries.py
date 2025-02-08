import logging

import pycountry
from gadm import GADMDownloader
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdministrativeAreaBoundaries:
    """Get the boundaries of administrative areas for a specified country.

    The administrative area level is specified by an integer, where 0 is the country level, 
    1 is the next broadest level (e.g. states or provinces), and so on.

    It is possible to get the geometry of a specific administrative area by name.

    Example usage:
    
    ```python
    admin_area_boundaries = AdministrativeAreaBoundaries("Timor-Leste", admin_level=1)
    print(boundaries.get_admin_area_names())
    target_admin_area_boundary = boundaries.get_admin_area_boundaries("Baucau")
    country_code = boundaries.get_iso3_country_code()
    ```
    """

    def __init__(
        self,
        country_name: str,
        admin_level: int,
    ):
        self.country = self._validate_country_name(country_name)
        self.admin_level = admin_level
        self.all_admin_areas_gdf = self._download_admin_areas()

    @staticmethod
    def _validate_country_name(country_name: str):
        """Validates country name using fuzzy matching and returns pycountry object.
        
        Args:
            country_name: Name of the country to validate
        
        Returns:
            pycountry.db.Country object
            
        Raises:
            Exception: If country name is invalid or not found
        """
        logger.info(f"Validating country name: {country_name}")
        country = pycountry.countries.get(name=country_name)

        if country is None:
            logger.warning(f"Country name '{country_name}' not found, attempting fuzzy search")
            try:
                possible_matches = pycountry.countries.search_fuzzy(country_name)
            except Exception as e:
                raise Exception("Invalid form of country name")
            raise Exception(f"Country not found. Possible matches: {[match.name for match in possible_matches]}")
        
        logger.info(f"Country name '{country_name}' validated successfully")
        return country
    
    @staticmethod
    def _download_admin_areas(country_name: str, admin_level: int) -> GeoDataFrame:
        """Downloads and returns all administrative areas of specified level for a country.
        
        Args:
            country_name: Name of the country
            admin_level: Administrative area level (0 for country, 1 for first level divisions, etc.)
        
        Returns:
            GeoDataFrame containing all administrative areas of the specified level
        """
        logger.info(f"Retrieving boundaries of all administrative areas of level {admin_level} for country {country_name}")
        downloader = GADMDownloader(version="4.0")
        return downloader.get_shape_data_by_country_name(
            country_name=country_name,
            ad_level=admin_level
        )
    
    def get_admin_area_names(self) -> list[str]:
        """Retrieves the names of all administrative areas for the specified level.
        
        Returns:
            List of administrative area names at the specified level.
            For admin_level=0, returns just the country name.
        """
        if self.admin_level == 0:
            return [self.country.name]
        
        return self.all_admin_areas_gdf[f"NAME_{self.admin_level}"].tolist()

    def get_admin_area_boundaries(self, admin_area_name: str) -> Polygon | MultiPolygon:
        """Returns the boundary geometry for the specified administrative area.
        
        Args:
            admin_area_name: Name of the administrative area to retrieve boundaries for
        
        Returns:
            Polygon or MultiPolygon representing the administrative area boundaries
            
        Raises:
            Exception: If admin_area_name is not set or not found in the data
        """
        if self.admin_level == 0:
            return self.all_admin_areas_gdf.geometry.iloc[0]
            
        filtered = self.all_admin_areas_gdf[
            self.all_admin_areas_gdf[f"NAME_{self.admin_level}"] == admin_area_name
        ]
        
        if filtered.empty:
            raise Exception(
            f"Administrative area '{admin_area_name}' not found. "
            f"Available areas: {self.get_admin_area_names()}"
            )
            
        return filtered.geometry.iloc[0]

    def get_iso3_country_code(self) -> str:
        """
        Retrieve the ISO 3166-1 alpha-3 country code for the country associated with this instance.

        Returns:
            str: The ISO 3166-1 alpha-3 country code in lowercase.
        """
        return self.country.alpha_3.lower()
