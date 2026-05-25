"""API contracts for the scalable general-distance pipeline."""

from scalable_distances.api import (
    create_context,
    describe_backends,
    describe_country_sources,
    write_distance_matrix,
)
from scalable_distances.core.context import DataContext

__all__ = [
    "DataContext",
    "create_context",
    "describe_backends",
    "describe_country_sources",
    "write_distance_matrix",
]
