from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scalable_distances.data.schemas import Schema

if TYPE_CHECKING:
    from scalable_distances.storage.repository import Repository


@dataclass(frozen=True)
class DatasetKey:
    """Stable identity for a dataset or view within a run."""

    name: str
    uri: str | None = None
    schema: str | None = None
    filters: tuple[tuple[str, str], ...] = ()
    crs: str | None = None
    backend: str = "auto"


@dataclass
class DatasetHandle:
    """Light wrapper around a loaded or lazy dataframe."""

    key: DatasetKey
    data: Any
    schema: Schema | None = None
    fingerprint: str | None = None

    def frame(self) -> Any:
        return self.data

    def __repr__(self) -> str:
        schema_name = self.schema.name if self.schema is not None else None
        return (
            "DatasetHandle("
            f"key={self.key!r}, "
            f"schema={schema_name!r}, "
            f"fingerprint={self.fingerprint!r}"
            ")"
        )


@dataclass
class DatasetRegistry:
    """Loads each dataset key once and returns the same handle by identity."""

    repository: Repository
    _cache: dict[DatasetKey, DatasetHandle] = field(default_factory=dict)

    def get(self, key: DatasetKey) -> DatasetHandle:
        if key not in self._cache:
            if not key.uri:
                raise ValueError(f"Dataset {key.name!r} has no URI and is not registered")
            self._cache[key] = self.repository.read_dataset(key)
        return self._cache[key]

    def register(self, handle: DatasetHandle) -> DatasetHandle:
        existing = self._cache.get(handle.key)
        if existing is not None:
            return existing
        self._cache[handle.key] = handle
        return handle
