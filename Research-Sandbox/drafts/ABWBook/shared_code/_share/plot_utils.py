import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from time import perf_counter as pc
import colorcet as cc
import fast_histogram
import pandas as pd
import geopandas as gpd

# plot_utils.py

def show_nodes_colored_per_component_with_basemap(
    nodes: gpd.GeoDataFrame,
    width: int = 800,
    height: int = 600,
    file_name: str | None = None
) -> None:
    """
    Plots nodes colored by component on a background basemap.
    The largest component is shown in light gray; others use distinct colors.

    Args:
        nodes: GeoDataFrame with geometry and 'component' column.
        width: Width of the figure in pixels.
        height: Height of the figure in pixels.
        file_name: Optional path to save the figure.
    """
    t0 = pc()
    print('⏳ Starting component coloring with basemap...')

    # ── Reproject if necessary ─────────────────────────────────────────────
    if nodes.crs != 'EPSG:3857':
        t_reproj = pc()
        nodes = nodes.to_crs('EPSG:3857')
        print(f'📐 Reprojected to EPSG:3857 in {pc() - t_reproj:.2f}s')

    # ── Component and color assignment ─────────────────────────────────────
    t_comp = pc()
    component_sizes = nodes['component'].value_counts()
    n_components = len(component_sizes)
    largest_group = component_sizes.idxmax()
    largest_size = component_sizes.max()
    print(f'✔️ Found {n_components} components (largest at index {largest_group} with size {largest_size:,} nodes) in {pc() - t_comp:.2f}s')

    t_color = pc()
    color_key = {largest_group: '#E5E5E5'}  # light gray background
    color_idx = 0
    for group_id in component_sizes.index:
        if group_id == largest_group:
            continue
        r, g, b = cc.glasbey_hv[color_idx % len(cc.glasbey_hv)]
        color_key[group_id] = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
        color_idx += 1
    nodes['color'] = nodes['component'].map(color_key)
    print(f'🎨 Assigned colors in {pc() - t_color:.2f}s')

    # ── Scatter plot drawing (fast, single call) ─────────────────────────────
    t_plot = pc()
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Set marker size and alpha based on whether it's the largest component
    is_largest = nodes['component'] == largest_group
    marker_size = is_largest.map({True: 0.1, False: 1.0})
    alpha = is_largest.map({True: 0.3, False: 1.0})

    ax.scatter(
        nodes.geometry.x,
        nodes.geometry.y,
        c=nodes['color'],
        s=marker_size,
        alpha=alpha,
        marker='.',
        linewidths=0
    )
    print(f'🖌️ Plotted {len(nodes):,} nodes in {pc() - t_plot:.2f}s')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    # ── Add basemap ────────────────────────────────────────────────────────
    t_base = pc()
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, attribution_size=6)
    print(f'🗺️ Added basemap in {pc() - t_base:.2f}s')

    # ── Save to file if requested ──────────────────────────────────────────
    if file_name:
        t_save = pc()
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f'💾 Saved plot to {file_name} in {pc() - t_save:.2f}s')

    print(f'✅ Total visualization completed in {pc() - t0:.2f}s')
    plt.show()


def plot_fast_histogram(
    data: np.ndarray,
    bins: int = 20,
    title: str | None = None,
    ax=None
) -> plt.Axes:
    """
    Plots a histogram using fast_histogram. Creates a new Axes if none is provided.

    Args:
        data: 1D NumPy array of numeric values to plot.
        bins: Number of histogram bins (default: 20).
        title: Optional title for the plot.
        ax: Optional matplotlib Axes. If None, a new figure and axes are created.

    Returns:
        The matplotlib Axes object used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if data.size == 0:
        ax.set_title(f'{title or "Empty data"} (no data)')
        return ax

    vmin, vmax = data.min(), data.max()
    if vmin == vmax:
        ax.set_title(f'{title or "Constant data"} (single value)')
        ax.text(0.5, 0.5, f'{vmin:.2f}', ha='center', va='center', transform=ax.transAxes)
        return ax

    counts = fast_histogram.histogram1d(data, bins=bins, range=(vmin, vmax))
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(centers, counts, width=(vmax - vmin) / bins, edgecolor='black')
    if title:
        ax.set_title(title)

    return ax