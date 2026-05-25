from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RouterCapabilities:
    distance: bool = True
    travel_time: bool = False
    multimodal: bool = False
    isochrones: bool = False


class RouterStrategy(Protocol):
    name: str
    contract_version: str
    capabilities: RouterCapabilities

    def prepare(self, network: Any, config: dict[str, Any]) -> None: ...

    def snap(self, points: Any) -> Any: ...

    def route_many(self, origins: Any, destinations: Any) -> Any: ...


@dataclass(frozen=True)
class NetworkData:
    """Router-neutral road network tables."""

    nodes: Any
    edges: Any
    node_id_col: str = "node_id"
    source_col: str = "u"
    target_col: str = "v"
    weight_col: str = "length_m"
    x_col: str = "lon"
    y_col: str = "lat"
