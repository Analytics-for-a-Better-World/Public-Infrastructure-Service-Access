from pathlib import Path
from time import perf_counter as pc

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import box


def estimate_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    '''Estimate a suitable UTM EPSG code from the GeoDataFrame extent.'''
    if gdf.empty:
        raise ValueError('Input GeoDataFrame is empty')
    if gdf.crs is None:
        raise ValueError('Input GeoDataFrame has no CRS')

    gdf_ll = gdf.to_crs(epsg=4326)
    min_lon, min_lat, max_lon, max_lat = gdf_ll.total_bounds

    lon = (min_lon + max_lon) / 2
    lat = (min_lat + max_lat) / 2
    zone = int((lon + 180) // 6) + 1

    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def describe_extent(
    gdf: gpd.GeoDataFrame,
    metric_epsg: int | None = None,
    label: str = 'Layer',
) -> None:
    '''Print lon lat bounds and estimated width and height in km.

    Width and height are computed in a metric projected CRS. If none is
    provided, a suitable UTM CRS is estimated from the layer extent.
    '''
    if gdf.empty:
        raise ValueError('Input GeoDataFrame is empty')
    if gdf.crs is None:
        raise ValueError('Input GeoDataFrame has no CRS')

    gdf_ll = gdf.to_crs(epsg=4326)
    min_lon, min_lat, max_lon, max_lat = gdf_ll.total_bounds

    if metric_epsg is None:
        metric_epsg = estimate_utm_epsg(gdf)

    gdf_metric = gdf.to_crs(epsg=metric_epsg)
    minx, miny, maxx, maxy = gdf_metric.total_bounds

    width_km = (maxx - minx) / 1000
    height_km = (maxy - miny) / 1000

    print(f'{label} longitude range: {min_lon:.4f} to {max_lon:.4f}')
    print(f'{label} latitude range: {min_lat:.4f} to {max_lat:.4f}')
    print(f'{label} metric CRS: EPSG:{metric_epsg}')
    print(f'{label} estimated width: {width_km:,.1f} km')
    print(f'{label} estimated height: {height_km:,.1f} km')


def to_point_geometries(
    gdf: gpd.GeoDataFrame,
    projected_epsg: int = 32751,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Convert non point geometries to centroids in a projected CRS.'''
    t0 = pc()

    if gdf.crs is None:
        raise ValueError('Input GeoDataFrame has no CRS')

    result = gdf.copy()
    non_points = ~result.geometry.geom_type.isin(['Point', 'MultiPoint'])

    if non_points.any():
        original_crs = result.crs
        projected = result.loc[non_points].to_crs(epsg=projected_epsg).copy()
        projected['geometry'] = projected.geometry.centroid
        result.loc[non_points, 'geometry'] = projected.to_crs(original_crs).geometry.values

    if verbose:
        print(
            f'Converted geometries to points in {pc() - t0:.2f} seconds, '
            f'{int(non_points.sum()):,} non point geometries converted'
        )

    return result


def classify_roads(
    edges: gpd.GeoDataFrame,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    '''Normalize road classes for plotting.'''
    t0 = pc()

    if 'highway' not in edges.columns:
        raise ValueError("Column 'highway' not found in edges GeoDataFrame")

    road_class_map = {
        'motorway': 'motorway',
        'motorway_link': 'motorway',
        'trunk': 'trunk',
        'trunk_link': 'trunk',
        'primary': 'primary',
        'primary_link': 'primary',
        'secondary': 'secondary',
        'secondary_link': 'secondary',
        'tertiary': 'tertiary',
        'tertiary_link': 'tertiary',
        'residential': 'residential',
        'unclassified': 'unclassified',
        'living_street': 'living_street',
        'service': 'service',
        'road': 'road',
    }

    roads = edges.loc[edges['highway'].isin(road_class_map)].copy()

    if 'highway' in roads.columns:
        roads['highway'] = roads['highway'].astype('category')

    roads['road_class'] = pd.Categorical(
        roads['highway'].map(road_class_map),
        categories=[
            'service',
            'living_street',
            'road',
            'unclassified',
            'residential',
            'tertiary',
            'secondary',
            'primary',
            'trunk',
            'motorway',
        ],
        ordered=True,
    )

    if roads.empty:
        raise ValueError('No supported road classes found')

    if verbose:
        dropped_mask = ~edges['highway'].isin(road_class_map)
        dropped_counts = edges.loc[dropped_mask, 'highway'].value_counts().head(10)
        dropped_summary = 'none'
        if not dropped_counts.empty:
            dropped_summary = ', '.join(f'{cls}: {cnt:,}' for cls, cnt in dropped_counts.items())

        print(
            f'Classified roads in {pc() - t0:.2f} seconds, '
            f'{len(roads):,} retained while {len(edges) - len(roads):,} dropped'
        )
        print(f'Top dropped classes: {dropped_summary}')
        print(roads['road_class'].value_counts().to_string())

    return roads


def _log_step(message: str, *, start_time: float | None = None) -> None:
    '''Print a verbose pipeline message with optional elapsed time.'''
    if start_time is None:
        print(message)
        return

    print(f'{message} in {pc() - start_time:.2f} seconds')


def plot_context_map(
    roads: gpd.GeoDataFrame,
    population_points: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    title: str | None = None,
    legend_loc: str = 'lower left',
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_title: str | None = 'Layers',
    show_legend: bool = True,
    road_colors: dict[str, str] | None = None,
    road_widths: dict[str, float] | None = None,
    road_order: list[str] | None = None,
    basemap_provider: object | None = None,
    basemap_zoom: int = 8,
    basemap_alpha: float = 0.28,
    population_max_marker_size: float = 7.0,
    population_alpha: float = 0.035,
    facility_marker_size: float = 24.0,
    candidate_marker_size: float | None = None,
    output_path: Path | None = None,
    dpi: int = 300,
    show: bool = True,
    verbose: bool = True,
) -> None:
    '''
    Plot roads, population points, existing amenities, and optional candidate sites.

    The road network is emphasized using a strong white casing and thicker colored
    overlays. Population points are intentionally subdued so they do not overpower
    the transport structure.

    Parameters
    ----------
    title
        Optional map title. If None or empty, no title is shown.
    legend_loc
        Matplotlib legend location, for example 'lower left', 'upper right',
        'center left', or 'best'.
    legend_bbox_to_anchor
        Optional anchor for precise legend placement, for example (1.02, 0.5)
        with legend_loc='center left' to place the legend outside the plot.
    legend_title
        Optional legend title. Use None for no legend title.
    show_legend
        Whether to show the legend.
    '''
    t0 = pc()

    if verbose:
        print('Starting context map plotting')
        print(f'Road rows: {len(roads):,}')
        print(f'Population point rows: {len(population_points):,}')
        print(f'Facility rows: {len(facilities):,}')
        print(f'Output path: {output_path}')
        print(f'DPI: {dpi}')
        print(f'Show figure: {show}')

    if 'road_class' not in roads.columns:
        raise ValueError("Column 'road_class' not found in roads GeoDataFrame")
    if 'population' not in population_points.columns:
        raise ValueError("Column 'population' not found in population_points")

    if roads.crs is None:
        raise ValueError('roads has no CRS')
    if population_points.crs is None:
        raise ValueError('population_points has no CRS')
    if facilities.crs is None:
        raise ValueError('facilities has no CRS')

    if verbose:
        print(f'Roads CRS: {roads.crs}')
        print(f'Population CRS: {population_points.crs}')
        print(f'Facilities CRS: {facilities.crs}')
        print(f'Total population represented: {population_points["population"].sum():,.0f}')

    if road_colors is None:
        road_colors = {
            'motorway': '#b30000',
            'trunk': '#e34a33',
            'primary': '#fdbb84',
            'secondary': '#6baed6',
            'tertiary': '#225ea8',
            'residential': '#4292c6',
            'unclassified': '#9ecae1',
            'living_street': '#8c8c8c',
            'service': '#6f6f6f',
            'road': '#b3b3b3',
        }

    if road_widths is None:
        road_widths = {
            'motorway': 3.6,
            'trunk': 3.1,
            'primary': 2.6,
            'secondary': 2.1,
            'tertiary': 1.7,
            'residential': 1.3,
            'unclassified': 1.1,
            'living_street': 1.0,
            'service': 0.95,
            'road': 0.95,
        }

    if road_order is None:
        road_order = [
            'service',
            'living_street',
            'road',
            'unclassified',
            'residential',
            'tertiary',
            'secondary',
            'primary',
            'trunk',
            'motorway',
        ]

    if basemap_provider is None:
        basemap_provider = cx.providers.CartoDB.PositronNoLabels

    if verbose:
        counts = roads['road_class'].value_counts()

        width = max(len(k) for k in counts.index)

        print('Road class counts:')
        for k, v in counts.items():
            print(f'{k:<{width}}  {v:>12,}')

    t_project = pc()
    if verbose:
        print('Projecting layers to EPSG:3857')

    roads_3857 = roads.to_crs(epsg=3857)
    if verbose:
        _log_step('Projected roads', start_time=t_project)

    t_project_pop = pc()
    pop_3857 = population_points.to_crs(epsg=3857)
    if verbose:
        _log_step('Projected population points', start_time=t_project_pop)

    t_project_facilities = pc()
    facilities_3857 = facilities.to_crs(epsg=3857)
    if verbose:
        _log_step('Projected facilities', start_time=t_project_facilities)
        _log_step('Projection completed', start_time=t_project)

        describe_extent(roads, label='Roads')
        describe_extent(population_points, label='Population points')
        describe_extent(facilities, label='Facilities')

    if candidate_marker_size is None:
        candidate_marker_size = facility_marker_size * 0.80

    if 'source_type' in facilities_3857.columns:
        existing = facilities_3857.loc[facilities_3857['source_type'] != 'candidate']
        candidates = facilities_3857.loc[facilities_3857['source_type'] == 'candidate']
    else:
        existing = facilities_3857
        candidates = facilities_3857.iloc[0:0].copy()

    if verbose:
        print(f'Existing facilities: {len(existing):,}')
        print(f'Candidate sites: {len(candidates):,}')

    t_figure = pc()
    if verbose:
        print('Creating figure')

    fig, ax = plt.subplots(figsize=(15, 15))

    if verbose:
        _log_step('Figure created', start_time=t_figure)

    t_roads = pc()
    if verbose:
        print('Plotting roads')

    for road_class in road_order:
        t_class = pc()
        subset = roads_3857.loc[roads_3857['road_class'] == road_class]

        if subset.empty:
            if verbose:
                print(f'  Skipping {road_class}, no rows')
            continue

        if verbose:
            print(f'  Plotting {road_class}: {len(subset):,} segments')

        subset.plot(
            ax=ax,
            color='white',
            linewidth=road_widths[road_class] * 2.8,
            alpha=1.0,
            zorder=1,
        )
        subset.plot(
            ax=ax,
            color=road_colors[road_class],
            linewidth=road_widths[road_class] * 1.25,
            alpha=1.0,
            zorder=2,
        )

        if verbose:
            _log_step(f'  Finished {road_class}', start_time=t_class)

    if verbose:
        _log_step('Road plotting completed', start_time=t_roads)

    t_population = pc()
    if verbose:
        print('Computing population marker sizes')

    pop_sizes = np.clip(
        np.sqrt(pop_3857['population'].to_numpy(dtype='float64')) * 0.35,
        0.6,
        population_max_marker_size,
    )

    if verbose:
        print(f'Population marker size min: {pop_sizes.min():.2f}')
        print(f'Population marker size max: {pop_sizes.max():.2f}')
        print('Plotting population points')

    pop_3857.plot(
        ax=ax,
        markersize=pop_sizes,
        alpha=population_alpha,
        color='#2f2f2f',
        edgecolor='none',
        zorder=3,
    )

    if verbose:
        _log_step('Population plotting completed', start_time=t_population)

    t_facilities = pc()
    if verbose:
        print('Plotting facilities and candidate sites')

    if not existing.empty:
        existing.plot(
            ax=ax,
            markersize=facility_marker_size,
            marker='^',
            alpha=0.95,
            color='magenta',
            zorder=4,
        )

    if not candidates.empty:
        candidates.plot(
            ax=ax,
            markersize=candidate_marker_size,
            marker='o',
            alpha=0.80,
            color='cyan',
            edgecolor='black',
            linewidth=0.35,
            zorder=5,
        )

    if verbose:
        _log_step('Facility and candidate plotting completed', start_time=t_facilities)

    if verbose:
        print(f'Axis extent before basemap: xlim={ax.get_xlim()}, ylim={ax.get_ylim()}')

        minx, miny, maxx, maxy = roads_3857.total_bounds
        width_km = (maxx - minx) / 1000
        height_km = (maxy - miny) / 1000
        bbox = box(minx, miny, maxx, maxy)
        bbox_ll = gpd.GeoDataFrame(geometry=[bbox], crs=3857).to_crs(4326)
        min_lon, min_lat, max_lon, max_lat = bbox_ll.total_bounds

        print(f'Roads longitude range in plotted CRS check: {min_lon:.4f} to {max_lon:.4f}')
        print(f'Roads latitude range in plotted CRS check: {min_lat:.4f} to {max_lat:.4f}')
        print(f'Roads width from EPSG:3857 bounds: {width_km:,.1f} km')
        print(f'Roads height from EPSG:3857 bounds: {height_km:,.1f} km')

    t_basemap = pc()
    if verbose:
        print(f'Adding basemap, zoom={basemap_zoom}, alpha={basemap_alpha}')

    cx.add_basemap(
        ax,
        source=basemap_provider,
        zoom=basemap_zoom,
        alpha=basemap_alpha,
    )

    if verbose:
        _log_step('Basemap added', start_time=t_basemap)

    t_legend = pc()
    if verbose:
        print('Building legend')

    legend_handles: list[Line2D] = []
    for road_class in road_order:
        if (roads['road_class'] == road_class).any():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=road_colors[road_class],
                    lw=road_widths[road_class] * 1.8,
                    label=road_class.replace('_', ' ').title(),
                )
            )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='#2f2f2f',
            markerfacecolor='#2f2f2f',
            markersize=5,
            linestyle='None',
            alpha=0.25,
            label='Population points',
        )
    )

    if (
        'source_type' in facilities.columns
        and (facilities['source_type'] == 'candidate').any()
    ):
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker='^',
                    color='magenta',
                    markerfacecolor='magenta',
                    markersize=8,
                    linestyle='None',
                    label='Existing amenities',
                ),
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='black',
                    markerfacecolor='cyan',
                    markersize=8,
                    linestyle='None',
                    label='Candidate sites',
                ),
            ]
        )
    else:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker='^',
                color='magenta',
                markerfacecolor='magenta',
                markersize=8,
                linestyle='None',
                label='Facilities',
            )
        )

    if show_legend:
        legend_kwargs = {
            'handles': legend_handles,
            'title': legend_title,
            'loc': legend_loc,
            'frameon': True,
            'facecolor': 'white',
            'framealpha': 0.95,
        }

        if legend_bbox_to_anchor is not None:
            legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor

        ax.legend(**legend_kwargs)

    if verbose:
        _log_step('Legend built', start_time=t_legend)

    t_format = pc()
    if verbose:
        print('Formatting figure')

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()

    if verbose:
        _log_step('Figure formatting completed', start_time=t_format)

    if output_path is not None:
        t_save = pc()
        if verbose:
            print(f'Saving context map to {output_path}')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

        if verbose:
            _log_step('Figure saved', start_time=t_save)

    if show:
        t_show = pc()
        if verbose:
            print('Showing figure, this may take time in notebooks')

        plt.show()

        if verbose:
            _log_step('Figure shown', start_time=t_show)
    else:
        if verbose:
            print('Closing figure because show=False')
        plt.close(fig)

    if verbose:
        _log_step('Total plotting completed', start_time=t0)
