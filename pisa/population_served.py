import geopandas as gpd
import pandas as pd


def population_served(
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

        population_served(grouped_population, isopolygons):
            index   Cluster_ID   ID_10           ID_20
            0      i0          [p0, p1]       [p0, p1]
            1      i1          [p2]           []
            2      i2          []             [p1, p2]
    """
    crs = "EPSG:4326"

    grouped_population = grouped_population.copy()
    grouped_population.index.name = "population_idx"
    grouped_population = grouped_population.to_crs(crs) if grouped_population.crs else grouped_population.set_crs(crs)

    isopolygons = isopolygons.copy()
    isopolygons.index.name = "isopolygon_idx"

    served_dict = {}

    # For each distance, find the population that falls within the facility isopolygons
    for col in isopolygons.columns:
        # Get the isopolygon geometries and project the population to the same CRS as the isopolygon
        temp_isopolygons = gpd.GeoDataFrame(isopolygons[col], geometry=col, crs=crs).dropna()

        # Get the population IDs that fall within each isopolygon
        served_gdf = grouped_population.sjoin(temp_isopolygons, how="right", predicate="within").dropna()
        served_dict[col] = (
            served_gdf.groupby("isopolygon_idx", group_keys=True)["population_idx"]
            .apply(list)
            .to_dict()
        )

    served_df = pd.DataFrame(index=isopolygons.index, data=served_dict).map(
        lambda d: list(map(int, d)) if isinstance(d, list) else []
    )
    served_df = served_df.reset_index().rename(columns={"isopolygon_idx": "Cluster_ID"})
    return served_df
