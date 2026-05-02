from pathlib import Path
from time import perf_counter as pc
from typing import Any

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


def selected_sources_from_model(model: Any, tol: float = 0.5) -> list[Any]:
    '''Return selected source IDs from a solved binary location model.'''
    selected: list[Any] = []

    for source_id in model.J:
        value = getattr(model.x[source_id], 'value', None)
        if value is not None and value > tol:
            selected.append(source_id)

    return selected


def read_point_layer(
    path: str | Path,
    x_candidates: tuple[str, ...] = ('longitude', 'Longitude', 'xcoord', 'lon'),
    y_candidates: tuple[str, ...] = ('latitude', 'Latitude', 'ycoord', 'lat'),
    crs: str = 'EPSG:4326',
) -> gpd.GeoDataFrame:
    '''Read a point layer from GeoParquet or plain Parquet.'''
    try:
        return gpd.read_parquet(path)
    except ValueError as error:
        if 'Missing geo metadata' not in str(error):
            raise

    data = pd.read_parquet(path)

    x_col = next((col for col in x_candidates if col in data.columns), None)
    y_col = next((col for col in y_candidates if col in data.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(
            'Could not infer coordinate columns. '
            f'Available columns are: {list(data.columns)}'
        )

    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data[x_col], data[y_col]),
        crs=crs,
    )


def build_solution_layers(
    model: Any,
    matrix_path: str | Path,
    population_path: str | Path,
    sources_path: str | Path,
    max_cover_dist_m: float,
    matrix: pd.DataFrame | None = None,
    target_col: str = 'target_id',
    source_col: str = 'source_id',
    distance_col: str = 'total_dist',
    population_id_col: str = 'ID',
    source_id_col: str = 'ID',
) -> dict[str, Any]:
    '''Build GeoDataFrame layers for plotting a maximum-covering solution.'''
    selected_sources = selected_sources_from_model(model)

    if matrix is None:
        matrix = pd.read_parquet(
            matrix_path,
            columns=[target_col, source_col, distance_col],
        )
    population = read_point_layer(population_path)
    sources = read_point_layer(sources_path)

    covered_pairs = matrix.loc[
        (matrix[source_col].isin(selected_sources))
        & (matrix[distance_col] <= max_cover_dist_m),
        [target_col, source_col, distance_col],
    ].copy()

    assignment = (
        covered_pairs
        .sort_values(distance_col)
        .drop_duplicates(target_col, keep='first')
    )

    covered_target_ids = set(assignment[target_col])

    population = population.copy()
    population['covered'] = population[population_id_col].isin(covered_target_ids)

    covered_population = population.loc[population['covered']].copy()

    opened_sources = sources.loc[
        sources[source_id_col].isin(selected_sources)
    ].copy()

    assignment = assignment.rename(
        columns={
            target_col: population_id_col,
            source_col: 'assigned_source_id',
        }
    )

    covered_population = covered_population.merge(
        assignment[[population_id_col, 'assigned_source_id', distance_col]],
        on=population_id_col,
        how='left',
    )

    return {
        'selected_source_ids': selected_sources,
        'population': population,
        'covered_population': covered_population,
        'opened_sources': opened_sources,
        'assignment': assignment,
    }


def plot_max_cover_solution(
    solution_layers: dict[str, Any],
    title: str | None = 'Maximum covering solution',
    roads: gpd.GeoDataFrame | None = None,
    figsize: tuple[float, float] = (10, 10),
    output_path: str | Path | None = None,
    dpi: int = 300,
    legend_loc: str = 'upper center',
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, -0.06),
    legend_title: str | None = None,
    legend_ncol: int = 3,
    add_basemap: bool = True,
    basemap_source: object | None = None,
    basemap_alpha: float = 0.85,
    uncovered_marker_size: float = 5.0,
    covered_marker_size: float = 5.0,
    opened_marker_size: float = 120,
    show_facility_callouts: bool = True,
    source_id_col: str = 'ID',
    population_col: str = 'population',
    callout_fontsize: float = 9.0,
) -> tuple[plt.Figure, plt.Axes]:
    '''Plot uncovered points, covered points, and opened sources.'''
    if basemap_source is None:
        basemap_source = cx.providers.CartoDB.Positron

    display_crs = 'EPSG:3857'
    population = solution_layers['population'].to_crs(display_crs)
    covered_population = solution_layers['covered_population'].to_crs(display_crs)
    opened_sources = solution_layers['opened_sources'].to_crs(display_crs)

    fig, ax = plt.subplots(figsize=figsize)

    if roads is not None:
        roads.to_crs(display_crs).plot(
            ax=ax,
            color='lightgray',
            linewidth=0.3,
            alpha=0.6,
        )

    population.loc[~population['covered']].plot(
        ax=ax,
        color='#BDBDBD',
        markersize=uncovered_marker_size,
        alpha=0.70,
    )

    if not covered_population.empty:
        covered_population.plot(
            ax=ax,
            color='#2CA25F',
            markersize=covered_marker_size,
            alpha=0.90,
        )

    if not opened_sources.empty:
        opened_sources.plot(
            ax=ax,
            color='#D62728',
            edgecolor='black',
            markersize=opened_marker_size,
            marker='*',
            linewidth=0.9,
            zorder=7,
        )

    if add_basemap:
        cx.add_basemap(
            ax,
            source=basemap_source,
            crs=display_crs,
            alpha=basemap_alpha,
        )

    if (
        show_facility_callouts
        and not opened_sources.empty
        and not covered_population.empty
        and source_id_col in opened_sources.columns
        and 'assigned_source_id' in covered_population.columns
        and population_col in covered_population.columns
        and population_col in population.columns
    ):
        served_by_source = (
            covered_population
            .assign(_source_key=lambda df: df['assigned_source_id'].astype(str))
            .groupby('_source_key')[population_col]
            .sum()
        )
        total_population = population[population_col].sum()

        if total_population > 0:
            center = opened_sources.unary_union.centroid
            fallback_offsets = [
                (34, 34),
                (34, -34),
                (-34, 34),
                (-34, -34),
                (50, 8),
                (-50, 8),
            ]

            for idx, (_, row) in enumerate(opened_sources.iterrows()):
                source_key = str(row[source_id_col])
                served_population = served_by_source.get(source_key, 0.0)

                if served_population <= 0:
                    continue

                point = row.geometry
                label = f'{served_population / total_population:.1%}'

                dx = point.x - center.x
                dy = point.y - center.y
                norm = float(np.hypot(dx, dy))
                if norm > 0:
                    offset = (int(42 * dx / norm), int(42 * dy / norm))
                    if abs(offset[0]) < 16:
                        offset = (16 if offset[0] >= 0 else -16, offset[1])
                    if abs(offset[1]) < 12:
                        offset = (offset[0], 12 if offset[1] >= 0 else -12)
                else:
                    offset = fallback_offsets[idx % len(fallback_offsets)]

                ax.annotate(
                    label,
                    xy=(point.x, point.y),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=callout_fontsize,
                    ha='left' if offset[0] >= 0 else 'right',
                    va='bottom' if offset[1] >= 0 else 'top',
                    color='black',
                    bbox={
                        'boxstyle': 'round,pad=0.2',
                        'facecolor': 'white',
                        'edgecolor': 'black',
                        'alpha': 0.95,
                        'linewidth': 0.7,
                    },
                    arrowprops={
                        'arrowstyle': '-',
                        'color': 'black',
                        'linewidth': 0.7,
                    },
                    zorder=8,
                )

    if title:
        ax.set_title(title)

    ax.set_axis_off()
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor='#BDBDBD',
            markeredgecolor='#BDBDBD',
            markersize=7,
            linestyle='None',
            label='Uncovered population points',
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor='#2CA25F',
            markeredgecolor='#2CA25F',
            markersize=7,
            linestyle='None',
            label='Covered population points',
        ),
        Line2D(
            [0],
            [0],
            marker='*',
            color='black',
            markerfacecolor='#D62728',
            markeredgecolor='black',
            markersize=11,
            linestyle='None',
            label='Opened facilities (percentage of total population served)',
        ),
    ]
    legend_kwargs = {
        'handles': legend_handles,
        'loc': legend_loc,
        'frameon': False,
        'title': legend_title,
        'ncol': legend_ncol,
    }

    if legend_bbox_to_anchor is not None:
        legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor

    ax.legend(**legend_kwargs)
    fig.subplots_adjust(bottom=0.14)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

    return fig, ax
