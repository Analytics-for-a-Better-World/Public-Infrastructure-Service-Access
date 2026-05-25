from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    required: bool = True


@dataclass(frozen=True)
class Schema:
    name: str
    version: str
    fields: tuple[Field, ...]

    def required_names(self) -> set[str]:
        return {field.name for field in self.fields if field.required}


PointSchema = Schema(
    name="point",
    version="v1",
    fields=(
        Field("point_id", "string"),
        Field("lat", "float"),
        Field("lon", "float"),
        Field("crs", "string", required=False),
    ),
)

FacilitySchema = Schema(
    name="facility",
    version="v1",
    fields=(
        Field("facility_id", "string"),
        Field("lat", "float"),
        Field("lon", "float"),
        Field("facility_type", "string"),
        Field("source_method", "string", required=False),
        Field("confidence", "float", required=False),
    ),
)

DistanceMatrixSchema = Schema(
    name="distance_matrix",
    version="v1",
    fields=(
        Field("origin_id", "string"),
        Field("destination_id", "string"),
        Field("distance_m", "float"),
        Field("travel_time_s", "float", required=False),
        Field("router", "string"),
    ),
)

GeocodeCandidateSchema = Schema(
    name="geocode_candidate",
    version="v1",
    fields=(
        Field("input_id", "string"),
        Field("candidate_id", "string"),
        Field("lat", "float"),
        Field("lon", "float"),
        Field("source_method", "string"),
        Field("confidence", "float"),
        Field("validation_status", "string", required=False),
    ),
)
