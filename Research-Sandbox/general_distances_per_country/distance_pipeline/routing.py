"""Routing utilities for sparse road-network experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter as pc
import itertools as it
import re
from typing import Hashable, Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


DEFAULT_SPEEDS_KPH: dict[str, float] = {
    'motorway': 100.0,
    'motorway_link': 60.0,
    'trunk': 80.0,
    'trunk_link': 50.0,
    'primary': 70.0,
    'primary_link': 45.0,
    'secondary': 60.0,
    'secondary_link': 40.0,
    'tertiary': 50.0,
    'tertiary_link': 35.0,
    'unclassified': 40.0,
    'residential': 30.0,
    'living_street': 10.0,
    'service': 20.0,
    'road': 30.0,
}


@dataclass(frozen=True, slots=True)
class TSPResult:
    """Result of a sparse TSP solve."""

    tour: list[Hashable]
    objective: float
    runtime_s: float
    status: str


def _first_value(value: object) -> object:
    """Return the first scalar when OSM tags arrive as list-like objects."""
    if isinstance(value, list | tuple | set | np.ndarray):
        return next(iter(value), None)
    return value


def normalize_highway(value: object) -> str:
    """Normalize an OSM highway tag to one representative string."""
    value = _first_value(value)
    if value is None or pd.isna(value):
        return 'road'
    return str(value).strip()


def parse_maxspeed_kph(value: object) -> float | None:
    """Parse a numeric speed in km/h from an OSM maxspeed-like value."""
    value = _first_value(value)
    if value is None or pd.isna(value):
        return None

    text = str(value).casefold().strip()
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match is None:
        return None

    speed = float(match.group(1))
    if 'mph' in text:
        speed *= 1.609344
    return speed


def _reverse_geometry(geometry: object) -> object:
    """Reverse simple line geometry when adding synthetic reverse edges."""
    if isinstance(geometry, LineString):
        return LineString(list(geometry.coords)[::-1])
    return geometry


def add_edge_speeds(
    edges: gpd.GeoDataFrame,
    *,
    default_speeds_kph: dict[str, float] | None = None,
    fallback_speed_kph: float = 30.0,
    speed_col: str = 'speed_kph',
    time_col: str = 'travel_time_s',
) -> gpd.GeoDataFrame:
    """Add speed and travel-time columns to an OSM edge layer.

    Existing ``maxspeed`` tags are used when parseable. Missing speeds fall
    back to the normalized ``highway`` class and then to ``fallback_speed_kph``.
    Edge length is interpreted as meters, matching pyrosm network output.
    """
    if 'length' not in edges.columns:
        raise KeyError("edges must contain a 'length' column in meters")

    speeds = DEFAULT_SPEEDS_KPH if default_speeds_kph is None else default_speeds_kph
    result = edges.copy()

    highway = result['highway'].map(normalize_highway) if 'highway' in result.columns else 'road'
    highway_speed = pd.Series(highway, index=result.index).map(speeds).fillna(
        fallback_speed_kph
    )

    if 'maxspeed' in result.columns:
        parsed = result['maxspeed'].map(parse_maxspeed_kph)
        result[speed_col] = pd.to_numeric(parsed, errors='coerce').fillna(highway_speed)
    else:
        result[speed_col] = highway_speed

    result[speed_col] = pd.to_numeric(result[speed_col], errors='raise').astype(
        'float64'
    )
    result['length'] = pd.to_numeric(result['length'], errors='raise').astype(
        'float64'
    )
    result[time_col] = result['length'] / (result[speed_col] * 1000.0 / 3600.0)
    return result


def build_networkx_graph(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    weight_col: str = 'length',
    bidirectional: bool = False,
) -> nx.DiGraph:
    """Build a directed NetworkX graph from pyrosm-style nodes and edges."""
    required_edge_cols = {'u', 'v', weight_col}
    missing = required_edge_cols.difference(edges.columns)
    if missing:
        raise KeyError(f'Missing edge columns: {sorted(missing)}')

    if 'id' not in nodes.columns:
        raise KeyError("nodes must contain an 'id' column")

    graph = nx.DiGraph()

    for row in nodes.itertuples(index=False):
        node_id = int(getattr(row, 'id'))
        graph.add_node(
            node_id,
            x=float(getattr(row, 'lon')),
            y=float(getattr(row, 'lat')),
            geometry=getattr(row, 'geometry', None),
        )

    def add_or_replace_edge(
        u: int,
        v: int,
        weight: float,
        geometry: object,
        length: float,
    ) -> None:
        attrs = {
            'weight': weight,
            weight_col: weight,
            'length': length,
            'geometry': geometry,
        }

        if graph.has_edge(u, v) and graph[u][v]['weight'] <= weight:
            return
        graph.add_edge(u, v, **attrs)

    for row in edges.itertuples(index=False):
        u = int(getattr(row, 'u'))
        v = int(getattr(row, 'v'))
        weight = float(getattr(row, weight_col))
        geometry = getattr(row, 'geometry', None)
        length = float(getattr(row, 'length', weight))
        add_or_replace_edge(u, v, weight, geometry, length)
        if bidirectional:
            add_or_replace_edge(v, u, weight, _reverse_geometry(geometry), length)

    return graph


def shortest_path_nodes(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    *,
    weight: str = 'weight',
) -> list[int]:
    """Return shortest-path node IDs between two snapped network nodes."""
    return [int(node) for node in nx.shortest_path(graph, origin, destination, weight=weight)]


def route_geometry_from_nodes(
    graph: nx.DiGraph,
    path_nodes: Iterable[int],
) -> LineString:
    """Build an approximate route geometry from a sequence of path nodes."""
    nodes = list(path_nodes)
    if len(nodes) < 2:
        raise ValueError('path_nodes must contain at least two nodes')

    pieces = []
    fallback_coords = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        data = graph[u][v]
        geometry = data.get('geometry')
        if geometry is not None:
            pieces.append(geometry)
        else:
            ux, uy = graph.nodes[u]['x'], graph.nodes[u]['y']
            vx, vy = graph.nodes[v]['x'], graph.nodes[v]['y']
            pieces.append(LineString([(ux, uy), (vx, vy)]))
        fallback_coords.append((graph.nodes[u]['x'], graph.nodes[u]['y']))

    fallback_coords.append((graph.nodes[nodes[-1]]['x'], graph.nodes[nodes[-1]]['y']))
    unioned = unary_union(pieces)
    if isinstance(unioned, LineString):
        return unioned
    merged = linemerge(unioned)
    if isinstance(merged, LineString):
        return merged
    return LineString(fallback_coords)


def route_between_nodes(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    *,
    weight: str = 'weight',
) -> tuple[list[int], LineString]:
    """Return shortest path node IDs and route geometry."""
    path = shortest_path_nodes(graph, origin, destination, weight=weight)
    return path, route_geometry_from_nodes(graph, path)


def sparse_costs_from_matrix(
    matrix: pd.DataFrame,
    *,
    origin_col: str = 'source_id',
    destination_col: str = 'target_id',
    cost_col: str = 'total_dist',
) -> dict[tuple[Hashable, Hashable], float]:
    """Convert a sparse distance table to a symmetric edge-cost dictionary."""
    required = {origin_col, destination_col, cost_col}
    missing = required.difference(matrix.columns)
    if missing:
        raise KeyError(f'Missing matrix columns: {sorted(missing)}')

    costs: dict[tuple[Hashable, Hashable], float] = {}
    for origin, destination, cost in matrix[[origin_col, destination_col, cost_col]].itertuples(index=False):
        if origin == destination:
            continue
        edge = tuple(sorted((origin, destination), key=str))
        value = float(cost)
        if edge not in costs or value < costs[edge]:
            costs[edge] = value
    return costs


def directed_costs_from_matrix(
    matrix: pd.DataFrame,
    *,
    origin_col: str = 'source_id',
    destination_col: str = 'target_id',
    cost_col: str = 'total_dist',
) -> dict[tuple[Hashable, Hashable], float]:
    """Convert a sparse distance table to an ordered arc-cost dictionary."""
    required = {origin_col, destination_col, cost_col}
    missing = required.difference(matrix.columns)
    if missing:
        raise KeyError(f'Missing matrix columns: {sorted(missing)}')

    costs: dict[tuple[Hashable, Hashable], float] = {}
    for origin, destination, cost in matrix[[origin_col, destination_col, cost_col]].itertuples(index=False):
        if origin == destination:
            continue
        arc = (origin, destination)
        value = float(cost)
        if arc not in costs or value < costs[arc]:
            costs[arc] = value
    return costs


def _selected_edges(values: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    return [edge for edge, value in values.items() if value > 0.5]


def _adjacency(selected: list[tuple[int, int]], nodes: range) -> dict[int, list[int]]:
    adjacent = {i: [] for i in nodes}
    for i, j in selected:
        adjacent[i].append(j)
        adjacent[j].append(i)
    return adjacent


def _subtours(adjacent: dict[int, list[int]]) -> list[list[int]]:
    unseen = set(adjacent)
    tours: list[list[int]] = []

    while unseen:
        start = unseen.pop()
        stack = [start]
        tour = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacent[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    tour.append(neighbor)
        tours.append(tour)

    return tours


def _directed_cycles(successor: dict[int, int], nodes: Iterable[int]) -> list[list[int]]:
    unseen = set(nodes)
    cycles: list[list[int]] = []

    while unseen:
        start = unseen.pop()
        cycle = [start]
        current = start
        while True:
            nxt = successor.get(current)
            if nxt is None or nxt == start:
                break
            if nxt not in unseen:
                cycle.append(nxt)
                break
            unseen.remove(nxt)
            cycle.append(nxt)
            current = nxt
        cycles.append(cycle)

    return cycles


def symmetric_tsp_via_gurobi_sparse(
    costs: dict[tuple[Hashable, Hashable], float],
    *,
    nodes: Iterable[Hashable] | None = None,
    trace: bool = False,
) -> TSPResult:
    """Solve a symmetric TSP over the available sparse edge dictionary.

    The formulation follows the lazy subtour-elimination approach used in the
    MO-book TSP notebook, adapted so variables are created only for edges
    present in ``costs`` rather than for every pair in a dense matrix.
    """
    import gurobipy as gp
    from gurobipy import GRB

    t0 = pc()

    labels = sorted(set(nodes or []), key=str)
    for i, j in costs:
        labels.extend([i, j])
    labels = sorted(set(labels), key=str)

    if len(labels) < 2:
        raise ValueError('At least two nodes are required for TSP')

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    edge_costs: dict[tuple[int, int], float] = {}
    for i, j in costs:
        if i == j:
            continue
        edge = tuple(sorted((label_to_idx[i], label_to_idx[j])))
        value = float(costs[(i, j)])
        if edge not in edge_costs or value < edge_costs[edge]:
            edge_costs[edge] = value

    model = gp.Model()
    model.Params.OutputFlag = int(trace)
    model.Params.LazyConstraints = 1

    all_nodes = range(len(labels))
    x = model.addVars(edge_costs.keys(), obj=edge_costs, vtype=GRB.BINARY, name='x')
    model.addConstrs(x.sum(i, '*') + x.sum('*', i) == 2 for i in all_nodes)

    def subtour_elimination(model_: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        solution = model_.cbGetSolution(x)
        selected = _selected_edges(solution)
        for tour in _subtours(_adjacency(selected, all_nodes)):
            if len(tour) < len(labels):
                model_.cbLazy(
                    gp.quicksum(x[edge] for edge in it.combinations(sorted(tour), 2) if edge in x)
                    <= len(tour) - 1
                )

    model.optimize(subtour_elimination)

    status = model.Status
    if status != GRB.OPTIMAL:
        return TSPResult(
            tour=[],
            objective=float('nan'),
            runtime_s=pc() - t0,
            status=str(status),
        )

    selected = _selected_edges(model.getAttr('X', x))
    tours = _subtours(_adjacency(selected, all_nodes))
    if len(tours) != 1:
        raise RuntimeError(f'Expected one tour, found {tours}')

    adjacent = _adjacency(selected, all_nodes)
    tour = [tours[0][0]]
    previous = None
    while True:
        current = tour[-1]
        candidates = [node for node in adjacent[current] if node != previous]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == tour[0]:
            break
        tour.append(nxt)
        previous = current

    tour.append(tour[0])
    return TSPResult(
        tour=[idx_to_label[idx] for idx in tour],
        objective=float(model.ObjVal),
        runtime_s=pc() - t0,
        status='optimal',
    )


def directed_tsp_via_gurobi_sparse(
    costs: dict[tuple[Hashable, Hashable], float],
    *,
    nodes: Iterable[Hashable] | None = None,
    trace: bool = False,
) -> TSPResult:
    """Solve a directed TSP over the available sparse arc dictionary."""
    import gurobipy as gp
    from gurobipy import GRB

    t0 = pc()

    labels = sorted(set(nodes or []), key=str)
    for i, j in costs:
        labels.extend([i, j])
    labels = sorted(set(labels), key=str)

    if len(labels) < 2:
        raise ValueError('At least two nodes are required for TSP')

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    arc_costs: dict[tuple[int, int], float] = {}
    for i, j in costs:
        if i == j:
            continue
        arc = (label_to_idx[i], label_to_idx[j])
        value = float(costs[(i, j)])
        if arc not in arc_costs or value < arc_costs[arc]:
            arc_costs[arc] = value

    model = gp.Model()
    model.Params.OutputFlag = int(trace)
    model.Params.LazyConstraints = 1

    all_nodes = range(len(labels))
    x = model.addVars(arc_costs.keys(), obj=arc_costs, vtype=GRB.BINARY, name='x')
    model.addConstrs(x.sum(i, '*') == 1 for i in all_nodes)
    model.addConstrs(x.sum('*', i) == 1 for i in all_nodes)

    def subtour_elimination(model_: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        solution = model_.cbGetSolution(x)
        selected = {i: j for (i, j), value in solution.items() if value > 0.5}
        for tour in _directed_cycles(selected, all_nodes):
            if len(tour) < len(labels):
                model_.cbLazy(
                    gp.quicksum(
                        x[i, j]
                        for i in tour
                        for j in tour
                        if i != j and (i, j) in x
                    )
                    <= len(tour) - 1
                )

    model.optimize(subtour_elimination)

    status = model.Status
    if status != GRB.OPTIMAL:
        return TSPResult(
            tour=[],
            objective=float('nan'),
            runtime_s=pc() - t0,
            status=str(status),
        )

    selected = {i: j for (i, j), value in model.getAttr('X', x).items() if value > 0.5}
    cycles = _directed_cycles(selected, all_nodes)
    if len(cycles) != 1 or len(cycles[0]) != len(labels):
        raise RuntimeError(f'Expected one directed tour, found {cycles}')

    tour = cycles[0]
    tour.append(tour[0])
    return TSPResult(
        tour=[idx_to_label[idx] for idx in tour],
        objective=float(model.ObjVal),
        runtime_s=pc() - t0,
        status='optimal',
    )
