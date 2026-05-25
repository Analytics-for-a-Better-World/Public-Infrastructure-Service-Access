from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from scalable_distances.routing.base import NetworkData, RouterCapabilities


def _nearest_nodes_bruteforce(points: pd.DataFrame, nodes: pd.DataFrame) -> pd.Series:
    import numpy as np

    px = points["lon"].to_numpy(dtype=float)
    py = points["lat"].to_numpy(dtype=float)
    nx = nodes["lon"].to_numpy(dtype=float)
    ny = nodes["lat"].to_numpy(dtype=float)
    node_ids = nodes["node_id"].to_numpy()
    nearest: list[Any] = []
    for lon, lat in zip(px, py):
        distances = (nx - lon) ** 2 + (ny - lat) ** 2
        nearest.append(node_ids[int(np.argmin(distances))])
    return pd.Series(nearest, index=points.index, name="nearest_node")


@dataclass
class NetworkXRouter:
    """Shortest-path router backed by NetworkX, imported only when selected."""

    name: str = "networkx"
    capabilities: RouterCapabilities = field(default_factory=RouterCapabilities)
    contract_version: str = "routing.v1"
    _graph: Any = field(default=None, init=False, repr=False)
    _nodes: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _weight_col: str = field(default="length_m", init=False, repr=False)

    def prepare(self, network: NetworkData, config: dict[str, Any] | None = None) -> None:
        import networkx as nx

        self._weight_col = network.weight_col
        nodes = network.nodes.copy()
        edges = network.edges.copy()
        nodes = nodes.rename(columns={network.node_id_col: "node_id", network.x_col: "lon", network.y_col: "lat"})
        edges = edges.rename(columns={network.source_col: "u", network.target_col: "v", network.weight_col: "length_m"})
        graph = nx.DiGraph()
        for row in nodes.itertuples(index=False):
            graph.add_node(getattr(row, "node_id"), lon=float(getattr(row, "lon")), lat=float(getattr(row, "lat")))
        for row in edges.itertuples(index=False):
            graph.add_edge(getattr(row, "u"), getattr(row, "v"), length_m=float(getattr(row, "length_m")))
        self._graph = graph
        self._nodes = nodes[["node_id", "lon", "lat"]]

    def snap(self, points: pd.DataFrame) -> pd.DataFrame:
        if self._nodes is None:
            raise RuntimeError("Router must be prepared before snapping points.")
        result = points.copy()
        result["nearest_node"] = _nearest_nodes_bruteforce(result, self._nodes)
        return result

    def route_many(self, origins: pd.DataFrame, destinations: pd.DataFrame) -> pd.DataFrame:
        if self._graph is None:
            raise RuntimeError("Router must be prepared before routing.")
        import networkx as nx

        rows: list[dict[str, Any]] = []
        for origin in origins.itertuples(index=False):
            origin_node = getattr(origin, "nearest_node")
            lengths = nx.single_source_dijkstra_path_length(
                self._graph,
                origin_node,
                weight="length_m",
            )
            for destination in destinations.itertuples(index=False):
                destination_node = getattr(destination, "nearest_node")
                distance = lengths.get(destination_node)
                if distance is None:
                    continue
                rows.append(
                    {
                        "source_id": getattr(origin, "source_id"),
                        "target_id": getattr(destination, "target_id"),
                        "source_type": getattr(origin, "source_type", "source"),
                        "target_type": getattr(destination, "target_type", "target"),
                        "source_node": origin_node,
                        "target_node": destination_node,
                        "network_dist": float(distance),
                        "total_dist": float(distance),
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class PandanaRouter:
    """Pandana router imported only when the Pandana strategy is selected."""

    name: str = "pandana"
    capabilities: RouterCapabilities = field(default_factory=RouterCapabilities)
    contract_version: str = "routing.v1"
    _network: Any = field(default=None, init=False, repr=False)
    _nodes: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def prepare(self, network: NetworkData, config: dict[str, Any] | None = None) -> None:
        import pandana as pdna

        nodes = network.nodes.copy()
        edges = network.edges.copy()
        nodes = nodes.rename(columns={network.node_id_col: "node_id", network.x_col: "lon", network.y_col: "lat"})
        edges = edges.rename(columns={network.source_col: "u", network.target_col: "v", network.weight_col: "length_m"})
        self._network = pdna.Network(
            node_x=nodes.set_index("node_id")["lon"],
            node_y=nodes.set_index("node_id")["lat"],
            edge_from=edges["u"],
            edge_to=edges["v"],
            edge_weights=edges[["length_m"]],
        )
        self._nodes = nodes[["node_id", "lon", "lat"]]

    def snap(self, points: pd.DataFrame) -> pd.DataFrame:
        if self._network is None:
            raise RuntimeError("Router must be prepared before snapping points.")
        result = points.copy()
        result["nearest_node"] = self._network.get_node_ids(result["lon"], result["lat"])
        return result

    def route_many(self, origins: pd.DataFrame, destinations: pd.DataFrame) -> pd.DataFrame:
        if self._network is None:
            raise RuntimeError("Router must be prepared before routing.")
        rows: list[dict[str, Any]] = []
        for origin in origins.itertuples(index=False):
            origin_node = getattr(origin, "nearest_node")
            distances = self._network.shortest_path_lengths(
                [origin_node] * len(destinations),
                destinations["nearest_node"].tolist(),
                imp_name="length_m",
            )
            for destination, distance in zip(destinations.itertuples(index=False), distances):
                if distance < 0:
                    continue
                rows.append(
                    {
                        "source_id": getattr(origin, "source_id"),
                        "target_id": getattr(destination, "target_id"),
                        "source_type": getattr(origin, "source_type", "source"),
                        "target_type": getattr(destination, "target_type", "target"),
                        "source_node": origin_node,
                        "target_node": getattr(destination, "nearest_node"),
                        "network_dist": float(distance),
                        "total_dist": float(distance),
                    }
                )
        return pd.DataFrame(rows)


class R5Router:
    name = "r5"
    contract_version = "routing.v1"
    capabilities = RouterCapabilities(distance=True, travel_time=True, multimodal=True, isochrones=True)

    def prepare(self, network: Any, config: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("R5 routing is a future optional adapter.")

    def snap(self, points: Any) -> Any:
        raise NotImplementedError("R5 routing is a future optional adapter.")

    def route_many(self, origins: Any, destinations: Any) -> Any:
        raise NotImplementedError("R5 routing is a future optional adapter.")
