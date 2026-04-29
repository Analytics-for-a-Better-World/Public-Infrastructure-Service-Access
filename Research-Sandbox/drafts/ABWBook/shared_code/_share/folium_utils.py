import folium
from folium.plugins import BeautifyIcon
from shapely.geometry import LineString
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from PIL import Image
import io


def get_points(data: pd.DataFrame) -> pd.DataFrame:
    """Extracts unique, non-null lat/lon points from columns 'FINISH_LATITUDE' and 'FINISH_LONGITUDE'."""
    return (
        data[['FINISH_LATITUDE', 'FINISH_LONGITUDE']]
        .dropna()
        .drop_duplicates()
        .copy()
        .rename(columns={'FINISH_LATITUDE': 'lat', 'FINISH_LONGITUDE': 'lon'})
    )


def adjust_bounds_for_points(folium_map: folium.Map, points: pd.DataFrame) -> folium.Map:
    """Adjusts map bounds to fit all points."""
    lat_min, lon_min = points.min()
    lat_max, lon_max = points.max()
    folium_map.fit_bounds(((lat_min, lon_min), (lat_max, lon_max)))
    return folium_map


def map_for_points(points: pd.DataFrame, zoom_start: int = 8) -> folium.Map:
    """Creates a folium map centered and zoomed to given point bounds."""
    lat_min, lon_min = points.min()
    lat_max, lon_max = points.max()
    center = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    fmap = folium.Map(location=center, zoom_start=zoom_start)
    fmap.fit_bounds(((lat_min, lon_min), (lat_max, lon_max)))
    return fmap


def default_marker(lat: float, lon: float, **kwargs) -> folium.CircleMarker:
    return folium.CircleMarker(location=(lat, lon), **kwargs)


def default_describer(lat: float, lon: float) -> str:
    return f'({lat:.5f}, {lon:.5f})'


def mark_points_on_map(
    points: pd.DataFrame,
    marker=default_marker,
    describe=default_describer,
    **kwargs
) -> folium.Map:
    """Adds a marker for each point on a folium map."""
    fmap = map_for_points(points)
    for lat, lon in points.values:
        marker(lat, lon, popup=describe(lat, lon), **kwargs).add_to(fmap)
    return fmap


def mark_route_through_points_on_map(
    points: pd.DataFrame,
    folium_map: folium.Map = None,
    describe=default_describer,
    color: str = 'blue',
    weight: int = 3,
    opacity: float = 1.0,
    icon_shape: str = 'marker',
    background_color: str = 'red',
    border_width: int = 1,
    inner_icon_style: str = 'font-size:10px',
    **kwargs
) -> folium.Map:
    """
    Draws a sequential route through the points, with numbered markers and a polyline.
    """
    if folium_map is None:
        folium_map = map_for_points(points)

    stops = [tuple(row) for row in points.drop_duplicates().dropna().values]
    for i, (lat, lon) in enumerate(stops):
        folium.Marker(
            location=(lat, lon),
            icon=BeautifyIcon(
                number=i,
                icon_shape=icon_shape,
                background_color=background_color,
                border_width=border_width,
                inner_icon_style=inner_icon_style,
                **kwargs
            ),
            popup=describe(lat, lon)
        ).add_to(folium_map)

    folium.PolyLine(stops, color=color, weight=weight, opacity=opacity, **kwargs).add_to(folium_map)
    return folium_map


def folium_to_png(folium_map: folium.Map, file_name: str, rendering_seconds: int = 5) -> None:
    """Renders a Folium map to PNG using its built-in _to_png method (requires a headless renderer)."""
    png_bytes = folium_map._to_png(rendering_seconds)
    image = Image.open(io.BytesIO(png_bytes))
    image.save(f'{file_name}.png')


def visualize_or_add_pop_poi_connection(
    row_idx,
    all_distances,
    population,
    points_of_interest,
    network,
    m: folium.Map = None,
    metric_crs='EPSG:28992',
    pop_id_col=None,
    poi_id_col=None,
    color='red',
    weight=3,
    index_label=None
) -> folium.Map:
    """
    Visualizes or adds a population–POI shortest path (and snapping lines) to a folium map.

    Args:
        row_idx: Index in `all_distances`.
        all_distances: DataFrame with 'pop_node_id' and 'poi_node_id' columns.
        population: GeoDataFrame of population locations.
        points_of_interest: GeoDataFrame of POIs.
        network: Pandana network object.
        m: Existing folium.Map to update (if None, a new one is created centered on this route).
        metric_crs: Projected CRS used for distance computations.
        pop_id_col: Population ID column.
        poi_id_col: POI ID column.
        color: Line color for the shortest path.
        weight: Line weight (thickness).
        index_label: Optional label to show in tooltips.

    Returns:
        folium.Map with the route added.
    """
    row = all_distances.loc[row_idx]

    if pop_id_col is None:
        pop_id_col = next((c for c in population.columns if c in row and 'pop' in c), 'idx')
    if poi_id_col is None:
        poi_id_col = next((c for c in points_of_interest.columns if c in row and 'poi' in c), 'idx')

    pop_idx = row.get(pop_id_col)
    poi_idx = row.get(poi_id_col)
    from_node = row['pop_node_id']
    to_node = row['poi_node_id']

    path_nodes = network.shortest_path(from_node, to_node)
    if path_nodes is None or len(path_nodes) < 2:
        return m  # skip trivial path

    node_coords = network.nodes_df.loc[path_nodes, ['x', 'y']].to_numpy()
    path_line_proj = LineString(node_coords)
    path_length_m = path_line_proj.length

    transformer = Transformer.from_crs(metric_crs, 'EPSG:4326', always_xy=True)
    lon_lat = [transformer.transform(x, y) for x, y in node_coords]
    path_line = LineString(lon_lat)

    pop_row = population.loc[population[pop_id_col] == pop_idx].iloc[0]
    poi_row = points_of_interest.loc[points_of_interest[poi_id_col] == poi_idx].iloc[0]
    pop_geom = pop_row.geometry
    poi_geom = poi_row.geometry
    pop_lonlat = transformer.transform(pop_geom.x, pop_geom.y)
    poi_lonlat = transformer.transform(poi_geom.x, poi_geom.y)

    snap_pop_xy = network.nodes_df.loc[from_node, ['x', 'y']]
    snap_poi_xy = network.nodes_df.loc[to_node, ['x', 'y']]
    snap_pop_lonlat = transformer.transform(snap_pop_xy['x'], snap_pop_xy['y'])
    snap_poi_lonlat = transformer.transform(snap_poi_xy['x'], snap_poi_xy['y'])

    snap_line_pop = LineString([pop_geom.coords[0], (snap_pop_xy['x'], snap_pop_xy['y'])])
    snap_line_poi = LineString([poi_geom.coords[0], (snap_poi_xy['x'], snap_poi_xy['y'])])

    # Create new map if needed
    if m is None:
        m = folium.Map(location=[(pop_lonlat[1] + poi_lonlat[1]) / 2, (pop_lonlat[0] + poi_lonlat[0]) / 2], zoom_start=13)

    # Add markers
    folium.Marker(
        location=(pop_lonlat[1], pop_lonlat[0]),
        tooltip=f"Population: {pop_row.get('Population', 'N/A')}",
        icon=folium.Icon(color='blue', icon='user')
    ).add_to(m)

    folium.Marker(
        location=(poi_lonlat[1], poi_lonlat[0]),
        tooltip=f"POI: {poi_row.get('FullAddress', 'N/A')}",
        icon=folium.Icon(color='green', icon='flag')
    ).add_to(m)

    folium.CircleMarker(
        location=(snap_pop_lonlat[1], snap_pop_lonlat[0]),
        radius=4,
        color='blue',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

    folium.CircleMarker(
        location=(snap_poi_lonlat[1], snap_poi_lonlat[0]),
        radius=4,
        color='green',
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

    folium.PolyLine(
        locations=[(pop_lonlat[1], pop_lonlat[0]), (snap_pop_lonlat[1], snap_pop_lonlat[0])],
        color='black',
        dash_array='3',
        tooltip=f'Snap line (pop): {snap_line_pop.length:.1f} m'
    ).add_to(m)

    folium.PolyLine(
        locations=[(poi_lonlat[1], poi_lonlat[0]), (snap_poi_lonlat[1], snap_poi_lonlat[0])],
        color='black',
        dash_array='3',
        tooltip=f'Snap line (POI): {snap_line_poi.length:.1f} m'
    ).add_to(m)

    # Add path line
    tooltip = f"Shortest path: {path_length_m:.1f} m"
    if index_label is not None:
        tooltip = f"#{index_label}: {path_length_m:.1f} m"

    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in lon_lat],
        color=color,
        weight=weight,
        tooltip=tooltip
    ).add_to(m)

    return m
