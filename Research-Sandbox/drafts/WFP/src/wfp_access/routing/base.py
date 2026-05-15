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
