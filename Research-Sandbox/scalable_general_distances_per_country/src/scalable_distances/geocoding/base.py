from __future__ import annotations

from typing import Any, Protocol


class GeocoderStage(Protocol):
    name: str
    contract_version: str

    def run(self, records: Any, context: Any) -> Any: ...
