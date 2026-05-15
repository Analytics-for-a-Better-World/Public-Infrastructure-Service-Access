from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PlaceholderGeocoderStage:
    name: str
    contract_version: str = "geocoding_stage.v1"

    def run(self, records: Any, context: Any) -> Any:
        raise NotImplementedError(f"{self.name} stage is not implemented yet")


class NormalizeNamesStage(PlaceholderGeocoderStage):
    def __init__(self) -> None:
        super().__init__("normalize_names")


class OsmAmenityMatchStage(PlaceholderGeocoderStage):
    def __init__(self) -> None:
        super().__init__("osm_amenity_match")


class NominatimStage(PlaceholderGeocoderStage):
    def __init__(self) -> None:
        super().__init__("nominatim")


class ReferenceDatasetStage(PlaceholderGeocoderStage):
    def __init__(self) -> None:
        super().__init__("reference_dataset")


class ValidationStage(PlaceholderGeocoderStage):
    def __init__(self) -> None:
        super().__init__("validation")


class ConfidenceStage(PlaceholderGeocoderStage):
    def __init__(self) -> None:
        super().__init__("confidence")
