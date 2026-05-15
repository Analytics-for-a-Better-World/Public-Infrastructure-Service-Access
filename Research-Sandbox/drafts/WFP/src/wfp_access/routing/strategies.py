from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wfp_access.routing.base import RouterCapabilities


@dataclass
class PlaceholderRouter:
    """Base placeholder for future optional routing adapters."""

    name: str
    capabilities: RouterCapabilities = RouterCapabilities()
    contract_version: str = "routing.v1"

    def prepare(self, network: Any, config: dict[str, Any]) -> None:
        raise NotImplementedError(f"{self.name} adapter is not implemented yet")

    def snap(self, points: Any) -> Any:
        raise NotImplementedError(f"{self.name} adapter is not implemented yet")

    def route_many(self, origins: Any, destinations: Any) -> Any:
        raise NotImplementedError(f"{self.name} adapter is not implemented yet")


class PandanaRouter(PlaceholderRouter):
    def __init__(self) -> None:
        super().__init__(name="pandana")


class NetworkXRouter(PlaceholderRouter):
    def __init__(self) -> None:
        super().__init__(name="networkx")


class R5Router(PlaceholderRouter):
    def __init__(self) -> None:
        super().__init__(
            name="r5",
            capabilities=RouterCapabilities(distance=True, travel_time=True, multimodal=True, isochrones=True),
        )
