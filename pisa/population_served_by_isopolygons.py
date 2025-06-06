"""Compute population coverage by facility service areas (isopolygons).

This module provides functions to determine which population points are covered by service areas
(isopolygons) around facilities at different distance thresholds. It helps answer questions such as
"How many people can reach a facility within X minutes?" or "Which populations are served by which
facilities at different distance thresholds?"

Examples
--------
Retrieve population coverage for example points and polygons::

>>> from shapely.geometry import Point, Polygon
>>> import geopandas as gpd
>>> import pandas as pd
>>> from pisa.population_served_by_isopolygons import get_population_served_by_isopolygons
>>>
>>> # Get administrative area and facilities
>>> admin_area = AdministrativeArea("Timor-Leste", admin_level=1)
>>> boundaries = admin_area.get_admin_area_boundaries("Baucau")
>>>
>>> # Get grouped population
>>> population = WorldpopPopulation(
>>>     admin_area_boundaries=boundaries,
>>>     iso3_country_code=country_code
>>> )
>>> grouped_population = population.get_population_gdf()
>>>
>>> facilities = Facilities(admin_area_boundaries=boundaries)
>>> existing_facilities = facilities.get_existing_facilities()
>>> 
>>> # Create a road network for travel time calculations
>>> road_network = OsmRoadNetwork(
>>>     admin_area_boundaries=boundaries,
>>>     mode_of_transport="walking",
>>>     distance_type="travel_time"
>>> )
>>> graph = road_network.get_osm_road_network()
>>> 
>>> # Calculate isochrones (5, 10, 15 minutes walking)
>>> isopolygon_calculator = OsmIsopolygonCalculator(
>>>     facilities_df=existing_facilities,
>>>     distance_type="travel_time",
>>>     distance_values=[5, 10, 15],
>>>     road_network=graph
>>> )
>>> isopolygons = isopolygon_calculator.calculate_isopolygons()
>>>
>>> # Find which population points are served by each isopolygon
>>> result = get_population_served_by_isopolygons(grouped_population, isopolygons)

See Also
--------
isopolygons : Module for generating service area polygons
facilities : Module for facility location and clustering
get_population_served_by_isopolygons : Main function for population coverage analysis
"""
import geopandas as gpd
import pandas as pd


def get_population_served_by_isopolygons(
    grouped_population: gpd.GeoDataFrame,
    isopolygons: pd.DataFrame,
) -> pd.DataFrame:
    """Identify population points that fall within each facility's isopolygons.

    This function performs a spatial join between population points and isopolygons
    representing service areas at various distances from facilities. It returns a DataFrame
    showing, for each facility (cluster) and distance threshold, which population points
    are contained within the corresponding service area.

    Parameters
    ----------
    grouped_population : geopandas.GeoDataFrame
        GeoDataFrame containing population points with geometry column.
        The index values are used as identifiers in the result.
        Must include a valid geometry column with point geometries.
    isopolygons : pandas.DataFrame
        DataFrame where each column starting with ``ID_`` contains Shapely Polygon objects
        representing service areas at different distances. Each row represents a facility
        isopolygon. The index values are used as ``Cluster_ID`` in the result.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

            - ``Cluster_ID``: The index from the isopolygons input
            - One column for each ``ID_`` column in isopolygons, containing lists of
              population indices that fall within the corresponding polygon
    
    Raises
    ------
    ValueError
        If either input DataFrame is empty

    Notes
    -----
    This function:

        1. Ensures both inputs have proper CRS (Coordinate Reference System)
        2. Performs a spatial join to find population points within each isopolygon
        3. Groups results by facility and distance threshold
        4. Returns population indices served by each facility at each distance threshold

    Example
    --------
    Basic usage with sample data::

        grouped_population:
            index   geometry
            p0      POINT (...)
            p1      POINT (...)
            p2      POINT (...)

        isopolygons:
            index   ID_10           ID_20
            i0      POLYGON (...)   POLYGON (...)
            i1      POLYGON (...)   POLYGON (...)
            i2      POLYGON (...)   POLYGON (...)

        result = get_population_served_by_isopolygons(grouped_population, isopolygons):
            index   Cluster_ID   ID_10           ID_20
            0      i0          [p0, p1]       [p0, p1]
            1      i1          [p2]           []
            2      i2          []             [p1, p2]
    """
    crs = "EPSG:4326"

    if grouped_population.empty or isopolygons.empty:
        raise ValueError("Input dataframes cannot be empty.")

    grouped_population = grouped_population.copy()
    grouped_population.index.name = "population_idx"
    grouped_population = (
        grouped_population.to_crs(crs)
        if grouped_population.crs
        else grouped_population.set_crs(crs)
    )

    isopolygons = isopolygons.copy()
    isopolygons.index.name = "isopolygon_idx"

    # Melt isopolygons to long format for all ID_ columns
    distance_cols = [col for col in isopolygons.columns if col.startswith("ID_")]
    melted_isopolygons = (
        isopolygons[distance_cols]
        .reset_index()
        .melt(id_vars=["isopolygon_idx"], var_name="distance_id", value_name="geometry")
        .dropna(subset=["geometry"])
    )
    melted_isopolygons = gpd.GeoDataFrame(
        melted_isopolygons, geometry="geometry", crs=crs
    )

    # Find spatial overlap between grouped population points and isopolygons
    population_isopolygon_overlap = grouped_population.sjoin(
        melted_isopolygons, how="right", predicate="within"
    )

    # Collect population points per isopolygon and distance, unstack ID_ columns into wide format
    population_isopolygon_overlap = (
        population_isopolygon_overlap.groupby(["isopolygon_idx", "distance_id"])[
            "population_idx"
        ]
        .apply(lambda x: list(x) if len(x) > 0 else [])
        .unstack("distance_id")
    )
    population_isopolygon_overlap.columns.name = None
    population_isopolygon_overlap = population_isopolygon_overlap.reset_index().rename(
        columns={"isopolygon_idx": "Cluster_ID"}
    )

    # Clean up lists: replace [nan] and nan with [] and convert float lists to int lists
    def sanitize_lists(x: list | float | int) -> list:
        """Convert list of floats to list of integers, or handle NaN values.
        
        Parameters
        ----------
        x : list, float or int
            Input data that may be a list of population indices (as floats) or a scalar value
            
        Returns
        -------
        list
            If input is a list without NaN values: list of integers
            If input is a list with NaN values or is a scalar: empty list
            
        Notes
        -----
        This helper function ensures consistent types and handles edge cases in the 
        population indices from the spatial join operation.
        """
        if isinstance(x, list):
            return [] if pd.isna(x).any() else [int(i) for i in x]
        else:
            return []

    population_isopolygon_overlap[distance_cols] = population_isopolygon_overlap[
        distance_cols
    ].map(sanitize_lists)

    return population_isopolygon_overlap
