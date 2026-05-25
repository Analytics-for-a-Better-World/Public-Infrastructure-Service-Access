"""API contracts for the scalable general-distance pipeline."""

from scalable_distances.api import (
    create_context,
    describe_backends,
    describe_country_sources,
    run_production_country,
    write_distance_matrix,
)
from scalable_distances.core.context import DataContext
from scalable_distances.pipeline import ProductionRunConfig, ProductionRunResult

__all__ = [
    "DataContext",
    "create_context",
    "describe_backends",
    "describe_country_sources",
    "ProductionRunConfig",
    "ProductionRunResult",
    "run_production_country",
    "write_distance_matrix",
]
