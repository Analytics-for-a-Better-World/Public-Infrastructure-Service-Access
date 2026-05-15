from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wfp_access.core.registry import Registry
from wfp_access.geocoding.base import GeocoderStage


class GeocoderRegistry(Registry[GeocoderStage]):
    """Registry for geocoding stage classes."""


@dataclass
class GeocodingPipeline:
    stages: list[GeocoderStage]

    def run(self, records: Any, context: Any) -> Any:
        current = records
        for stage in self.stages:
            current = stage.run(current, context)
        return current
