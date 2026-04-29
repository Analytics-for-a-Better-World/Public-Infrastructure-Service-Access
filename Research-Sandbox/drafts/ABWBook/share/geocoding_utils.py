import pandas as pd
import geopandas as gpd

# geocoding_utils.py

def geocode_top_n_into_geometry_column(
    gdf: gpd.GeoDataFrame,
    address_column: str = 'FullAddress',
    sort_column: str = 'nearest_node_distance',
    new_geometry_column: str = 'geocoded_geometry',
    top_n: int = 10
) -> gpd.GeoDataFrame:
    """
    Adds a new geometry column to a GeoDataFrame with geocoded values for the top N records
    (based on sort_column). Other rows keep the original geometry.

    Args:
        gdf: GeoDataFrame with address and geometry info.
        address_column: Column name containing full address strings.
        sort_column: Column to sort by for selecting top N rows to geocode.
        new_geometry_column: Name of the new column to create.
        top_n: Number of top rows to geocode.

    Returns:
        Updated GeoDataFrame with an added Point column.
    """
    geolocator = Nominatim(user_agent='Course notes on Analytics for a Better World (j.a.s.gromicho@uva.nl)')
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=8,
        error_wait_seconds=8
    )

    gdf = gdf.copy()
    gdf[new_geometry_column] = gdf.geometry

    # Clean non-breaking spaces
    cleaned_addresses = gdf[address_column].str.replace('\xa0', ' ', regex=False)

    top_idx = gdf.nlargest(top_n, sort_column).index
    new_points = {}

    for idx in top_idx:
        address = cleaned_addresses.loc[idx]
        if pd.notna(address) and address.strip():
            location = geocode(address)
            if location:
                new_points[idx] = Point(location.longitude, location.latitude)

    geocoded_series = gpd.GeoSeries(new_points, crs='EPSG:4326')
    if gdf.crs is not None:
        geocoded_series = geocoded_series.to_crs(gdf.crs)

    for idx, geom in geocoded_series.items():
        gdf.at[idx, new_geometry_column] = geom

    return gdf


import pandas as pd
import geopandas as gpd

# geocoding_utils.py

def geocode_top_n_into_geometry_column(
    gdf: gpd.GeoDataFrame,
    address_column: str = 'FullAddress',
    sort_column: str = 'nearest_node_distance',
    new_geometry_column: str = 'geocoded_geometry',
    top_n: int = 10
) -> gpd.GeoDataFrame:
    """
    Adds a new geometry column to a GeoDataFrame with geocoded values for the top N records
    (based on sort_column). Other rows keep the original geometry.

    Args:
        gdf: GeoDataFrame with address and geometry info.
        address_column: Column name containing full address strings.
        sort_column: Column to sort by for selecting top N rows to geocode.
        new_geometry_column: Name of the new column to create.
        top_n: Number of top rows to geocode.

    Returns:
        Updated GeoDataFrame with an added Point column.
    """
    geolocator = Nominatim(user_agent='Course notes on Analytics for a Better World (j.a.s.gromicho@uva.nl)')
    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=8,
        error_wait_seconds=8
    )

    gdf = gdf.copy()
    gdf[new_geometry_column] = gdf.geometry

    # Clean non-breaking spaces
    cleaned_addresses = gdf[address_column].str.replace('\xa0', ' ', regex=False)

    top_idx = gdf.nlargest(top_n, sort_column).index
    new_points = {}

    for idx in top_idx:
        address = cleaned_addresses.loc[idx]
        if pd.notna(address) and address.strip():
            location = geocode(address)
            if location:
                new_points[idx] = Point(location.longitude, location.latitude)

    geocoded_series = gpd.GeoSeries(new_points, crs='EPSG:4326')
    if gdf.crs is not None:
        geocoded_series = geocoded_series.to_crs(gdf.crs)

    for idx, geom in geocoded_series.items():
        gdf.at[idx, new_geometry_column] = geom

    return gdf
