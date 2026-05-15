from __future__ import annotations

from dataclasses import dataclass, field

from wfp_access.artifacts.runner import ArtifactRunner
from wfp_access.data.registry import DatasetRegistry
from wfp_access.geocoding.pipeline import GeocoderRegistry
from wfp_access.routing.registry import RouterRegistry
from wfp_access.storage.repository import Repository


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
