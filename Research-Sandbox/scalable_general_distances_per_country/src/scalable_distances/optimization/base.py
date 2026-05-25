from __future__ import annotations

from typing import Any, Protocol


class LocationModel(Protocol):
    name: str
    contract_version: str

    def solve(self, demand: Any, candidates: Any, distances: Any, config: dict[str, Any]) -> Any: ...
