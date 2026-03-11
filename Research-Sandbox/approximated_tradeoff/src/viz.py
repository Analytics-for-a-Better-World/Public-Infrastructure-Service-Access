import math

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx

def plot_points_with_basemap_grid(
    gdfs: list[gpd.GeoDataFrame],
    titles: list[str] | None = None,
    *,
    input_crs: str = 'EPSG:4326',
    nrows: int | None = 1,
    ncols: int | None = None,
    provider: object = cx.providers.CartoDB.Positron,
    markersize: float = 2.0,
    alpha: float = 0.7,
    pad_frac: float = 0.02,
    cell_height_in: float = 6.0,
    max_fig_width_in: float = 22.0,
    filename: str | None = None,
    dpi: int = 150
) -> None:
    """
    Plot any number of point GeoDataFrames on Contextily tiles in a grid.

    Features
    - Reprojects to EPSG:3857 for tile alignment
    - Shared extent across all panels, optional padding
    - Control rows and columns
    - Figure sizing adapts to the map aspect ratio (useful for tall shapes)

    Parameters
    ----------
    gdfs
        List of GeoDataFrames, each with point geometries.
    titles
        Optional titles, same length as gdfs.
    input_crs
        CRS assumed when a GeoDataFrame has no CRS.
    nrows, ncols
        Layout control.
        If nrows is provided and ncols is None, ncols is computed.
        If ncols is provided and nrows is None, nrows is computed.
        If both are None, a near square layout is chosen.
    provider
        Contextily tile provider.
    markersize, alpha
        Marker styling for scatter.
    pad_frac
        Padding fraction applied to shared bounds.
    cell_height_in
        Height in inches of each subplot cell, width is derived from map aspect.
    max_fig_width_in
        Clamp overall figure width to avoid extreme wide figures.
    dpi
        Figure DPI.
    """
    if not gdfs:
        raise ValueError('gdfs must not be empty.')

    n = len(gdfs)
    if titles is not None and len(titles) != n:
        raise ValueError('titles must have the same length as gdfs.')

    if nrows is None and ncols is None:
        ncols = max(1, math.ceil(math.sqrt(n)))
        nrows = math.ceil(n / ncols)
    elif nrows is None and ncols is not None:
        nrows = math.ceil(n / ncols)
    elif nrows is not None and ncols is None:
        ncols = math.ceil(n / nrows)
    else:
        assert nrows is not None and ncols is not None
        if nrows * ncols < n:
            raise ValueError('nrows * ncols must be at least the number of plots.')

    gdfs_3857: list[gpd.GeoDataFrame] = []
    for gdf in gdfs:
        if gdf.crs is None:
            gdf = gdf.set_crs(input_crs)
        gdfs_3857.append(gdf.to_crs(epsg=3857))

    bounds = [gdf.total_bounds for gdf in gdfs_3857]
    minx = min(b[0] for b in bounds)
    miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds)
    maxy = max(b[3] for b in bounds)

    dx = maxx - minx
    dy = maxy - miny
    if dx <= 0 or dy <= 0:
        raise ValueError('Degenerate bounds, check geometries.')

    minx -= dx * pad_frac
    maxx += dx * pad_frac
    miny -= dy * pad_frac
    maxy += dy * pad_frac

    map_aspect = dx / dy
    cell_width_in = max(2.0, cell_height_in * map_aspect)

    fig_w = min(max_fig_width_in, cell_width_in * ncols)
    if fig_w < cell_width_in * ncols:
        scale = fig_w / (cell_width_in * ncols)
        cell_height_eff = cell_height_in * scale
    else:
        cell_height_eff = cell_height_in

    fig_h = cell_height_eff * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes_list = list(axes.ravel()) if hasattr(axes, 'ravel') else [axes]

    for i, ax in enumerate(axes_list):
        if i >= n:
            ax.axis('off')
            continue

        gdf = gdfs_3857[i]
        ax.scatter(
            gdf.geometry.x.to_numpy(),
            gdf.geometry.y.to_numpy(),
            s=markersize,
            alpha=alpha
        )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal', adjustable='box')

        cx.add_basemap(ax, source=provider)
        ax.set_axis_off()

        if titles is not None:
            ax.set_title(titles[i])

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()


def _as_xy(xy: np.ndarray) -> np.ndarray:
    """
    Validate and return the first two columns of xy as a float array.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape (n, 2) or (n, >=2).

    Returns
    -------
    np.ndarray
        Float array of shape (n, 2), finite rows only.

    Raises
    ------
    ValueError
        If xy is not a 2D array with at least 2 columns.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError('xy must be a 2D array with at least 2 columns')
    pts = xy[:, :2]
    mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    return pts[mask]


def _guess_order_global(xy: np.ndarray) -> str:
    """
    Guess coordinate order as 'lonlat' or 'latlon' using global validity bounds.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape (n, 2) in either [lon, lat] or [lat, lon].

    Returns
    -------
    str
        'lonlat' or 'latlon'.
    """
    pts = _as_xy(xy)
    c0, c1 = pts[:, 0], pts[:, 1]

    lonlat_ok = (np.abs(c0) <= 180.0) & (np.abs(c1) <= 90.0)
    latlon_ok = (np.abs(c0) <= 90.0) & (np.abs(c1) <= 180.0)

    lonlat_score = int(lonlat_ok.sum())
    latlon_score = int(latlon_ok.sum())

    if lonlat_score > latlon_score:
        return 'lonlat'
    if latlon_score > lonlat_score:
        return 'latlon'
    return 'lonlat'


def _to_gdf_wgs84(xy: np.ndarray, order: str) -> gpd.GeoDataFrame:
    """
    Convert xy to a WGS84 GeoDataFrame.

    Parameters
    ----------
    xy : np.ndarray
        Array with coordinates.
    order : str
        'lonlat' or 'latlon'.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with EPSG:4326 CRS.

    Raises
    ------
    ValueError
        If order is not 'lonlat' or 'latlon'.
    """
    pts = _as_xy(xy)

    if order == 'lonlat':
        lon, lat = pts[:, 0], pts[:, 1]
    elif order == 'latlon':
        lat, lon = pts[:, 0], pts[:, 1]
    else:
        raise ValueError("order must be 'lonlat' or 'latlon'")

    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat),
        crs='EPSG:4326',
    )


def plot_points_with_tiles_generic(
    layers: list[dict],
    *,
    provider: object = cx.providers.CartoDB.Positron,
    figsize: tuple[float, float] = (9.0, 9.0),
    pad_frac: float = 0.04,
    rasterize_threshold: int = 200_000,
    axis_off: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more point layers with a contextily basemap, anywhere on Earth.

    Each layer is a dict with:
      - 'xy': np.ndarray of shape (n, >=2) containing coords
      - optional 'order': 'auto' (default), 'lonlat', or 'latlon'
      - optional 'style': kwargs for GeoDataFrame.plot (color, markersize, alpha, etc)
      - optional 'label': legend label

    Parameters
    ----------
    layers : list[dict]
        List of layer dicts.
    provider : object
        Basemap provider for contextily.
    figsize : tuple[float, float]
        Size in inches when creating a new figure. Ignored if ax is provided.
    pad_frac : float
        Fractional padding around the combined bounds.
    rasterize_threshold : int
        Rasterize layers with at least this many points to speed up rendering.
    axis_off : bool
        If True, hide axes.
    ax : matplotlib.axes.Axes | None
        If provided, draw on this axis (enables subplots). If None, a new figure
        and axis are created.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis.

    Raises
    ------
    ValueError
        If layers is empty or no valid layers remain after cleaning.
    """
    if not layers:
        raise ValueError('layers must be a non empty list')

    prepared: list[tuple[gpd.GeoDataFrame, dict, str | None]] = []
    bad_layers: list[str] = []

    for i, layer in enumerate(layers):
        if 'xy' not in layer:
            raise ValueError("each layer dict must contain key 'xy'")

        xy = layer['xy']
        order = layer.get('order', 'auto')
        style = dict(layer.get('style', {}))
        label = layer.get('label')

        if order == 'auto':
            order = _guess_order_global(xy)
        if order not in {'lonlat', 'latlon'}:
            raise ValueError("order must be 'auto', 'lonlat', or 'latlon'")

        gdf = _to_gdf_wgs84(xy, order=order)

        if gdf.empty:
            bad_layers.append(f'layer {i} is empty after cleaning')
            continue

        gdf_3857 = gdf.to_crs(epsg=3857)

        if gdf_3857.empty:
            bad_layers.append(f'layer {i} is empty after reprojection')
            continue

        b = gdf_3857.total_bounds
        if np.isnan(b).any() or ~np.isfinite(b).all():
            bad_layers.append(f'layer {i} has invalid bounds {b}')
            continue

        if len(gdf_3857) >= rasterize_threshold:
            style.setdefault('rasterized', True)

        prepared.append((gdf_3857, style, label))

    if not prepared:
        msg = 'No valid layers to plot.'
        if bad_layers:
            msg += ' Problems: ' + '; '.join(bad_layers)
        raise ValueError(msg)

    bounds = np.array([g.total_bounds for g, _, _ in prepared])
    minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
    maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

    dx = maxx - minx
    dy = maxy - miny

    if not np.isfinite(dx) or dx <= 0:
        dx = 1.0
    if not np.isfinite(dy) or dy <= 0:
        dy = 1.0

    minx -= dx * pad_frac
    maxx += dx * pad_frac
    miny -= dy * pad_frac
    maxy += dy * pad_frac

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for gdf_3857, style, label in prepared:
        if isinstance(label, str) and label:
            style.setdefault('label', label)
        gdf_3857.plot(ax=ax, **style)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    cx.add_basemap(ax, source=provider, crs=prepared[0][0].crs)

    if axis_off:
        ax.set_axis_off()

    if any(isinstance(label, str) and label for _, _, label in prepared):
        ax.legend()

    return fig, ax