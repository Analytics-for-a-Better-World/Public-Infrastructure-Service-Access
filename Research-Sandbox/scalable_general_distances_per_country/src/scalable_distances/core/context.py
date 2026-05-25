from __future__ import annotations

from dataclasses import dataclass, field

from scalable_distances.artifacts.runner import ArtifactRunner
from scalable_distances.data.registry import DatasetRegistry
from scalable_distances.geocoding.pipeline import GeocoderRegistry
from scalable_distances.routing.registry import RouterRegistry
from scalable_distances.storage.repository import Repository


@dataclass
class DataContext:
    """Scoped runtime context for one reproducible pipeline run."""

    run_id: str
    repository: Repository
    data: DatasetRegistry = field(init=False)
    runner: ArtifactRunner = field(init=False)
    routers: RouterRegistry = field(default_factory=RouterRegistry)
    geocoders: GeocoderRegistry = field(default_factory=GeocoderRegistry)

    def __post_init__(self) -> None:
        self.data = DatasetRegistry(repository=self.repository)
        self.runner = ArtifactRunner(repository=self.repository, registry=self.data)
