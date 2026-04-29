import pandas as pd
import geopandas as gpd

# diagnostic_utils.py

def compare_geocoded_and_network_locations(
    banks: gpd.GeoDataFrame,
    furthest_index: int,
    geocoded_xy: tuple[float, float],  # (lon, lat)
    crs_latlon: str = 'EPSG:4326',
    crs_projected: str = 'EPSG:28992'
) -> tuple[folium.Map, float]:
    """
    Compares a bank's original (network-snapped) location to a geocoded one,
    computes distance, and displays a Folium map and comparison table.

    Args:
        banks: GeoDataFrame with bank locations (must include 'Latitude' and 'Longitude').
        furthest_index: Index of the record to compare.
        geocoded_xy: Tuple (lon, lat) from external geocoding service.
        crs_latlon: CRS for latitude/longitude (default WGS84).
        crs_projected: Projected CRS for distance computation in meters.

    Returns:
        A tuple of:
        - folium.Map showing both points and distance line
        - distance in meters
    """
    # 1. Build comparison table
    top_record = banks.loc[[furthest_index], ['Latitude', 'Longitude']]
    comparison = pd.concat([
        top_record.rename(index={furthest_index: 'furthest'}),
        pd.DataFrame([{'Latitude': geocoded_xy[1], 'Longitude': geocoded_xy[0]}], index=['geocoded'])
    ])
    display(comparison)

    # 2. Create GeoDataFrame for both points
    gdf_points = gpd.GeoDataFrame({
        'name': ['furthest', 'geocoded'],
        'geometry': [
            Point(top_record['Longitude'].values[0], top_record['Latitude'].values[0]),
            Point(*geocoded_xy)
        ]
    }, crs=crs_latlon).to_crs(crs_projected)

    # 3. Compute distance in meters
    distance_m = gdf_points.distance(gdf_points.iloc[0].geometry)[1]
    print(f"📏 Distance from network-snapped to geocoded location: {distance_m:,.1f} meters")

    # 4. Create Folium map
    m = banks.loc[[furthest_index]].explore(
    tooltip=['Bank', 'FullAddress', 'nearest_node_distance'],
        style_kwds={'color': 'red', 'weight': 8}
    )

    red_latlon = top_record.iloc[0]['Latitude'], top_record.iloc[0]['Longitude']
    blue_latlon = geocoded_xy[1], geocoded_xy[0]

    # Add geocoded point (blue)
    folium.CircleMarker(
        location=blue_latlon,
        popup='Geocoded location',
        radius=8,
        color='blue',
        fill_color='blue',
        fill_opacity=1.0,
    ).add_to(m)

    # Add line connecting both
    folium.PolyLine(
        locations=[red_latlon, blue_latlon],
        color='black',
        weight=2,
        dash_array='5,5',
        tooltip=f'{distance_m:.1f} meters'
    ).add_to(m)

    # Adjust bounds
    bounds = [
        [min(red_latlon[0], blue_latlon[0]), min(red_latlon[1], blue_latlon[1])],
        [max(red_latlon[0], blue_latlon[0]), max(red_latlon[1], blue_latlon[1])]
    ]
    m.fit_bounds(bounds)

    return m, distance_m
