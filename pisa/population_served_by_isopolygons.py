import geopandas as gpd
import pandas as pd


def get_population_served_by_isopolygons(
    grouped_population: gpd.GeoDataFrame,
    isopolygons: pd.DataFrame,
) -> pd.DataFrame:
    """Get the list of population IDs that fall within the isopolygons of each facility.

    Each column of isopolygons is expected to represent the isopolygons of a facility at a specific distance.

    Example usage:
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

        get_population_served_by_isopolygons(grouped_population, isopolygons):
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
    grouped_population = grouped_population.to_crs(crs) if grouped_population.crs else grouped_population.set_crs(crs)

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
    melted_isopolygons = gpd.GeoDataFrame(melted_isopolygons, geometry="geometry", crs=crs)

    # Find spatial overlap between grouped population points and isopolygons
    population_isopolygon_overlap = (
        grouped_population
        .sjoin(melted_isopolygons, how="right", predicate="within")
    )

    # Collect population points per isopolygon and distance, unstack ID_ columns into wide format
    population_isopolygon_overlap = (
        population_isopolygon_overlap.groupby(["isopolygon_idx", "distance_id"])["population_idx"]
        .apply(lambda x: list(x) if len(x) > 0 else [])
        .unstack("distance_id")
    )
    population_isopolygon_overlap.columns.name = None
    population_isopolygon_overlap = (
        population_isopolygon_overlap.reset_index()
        .rename(columns={"isopolygon_idx": "Cluster_ID"})
    )

    # Clean up lists: replace [nan] and nan with [] and convert float lists to int lists
    def sanitize_lists(x):
        """Convert list of floats to list of ints, or return empty list if NaN."""
        if isinstance(x, list):
            return [] if pd.isna(x).any() else [int(i) for i in x]
        else:
            return []
    population_isopolygon_overlap[distance_cols] = population_isopolygon_overlap[distance_cols].applymap(sanitize_lists)
    
    return population_isopolygon_overlap
