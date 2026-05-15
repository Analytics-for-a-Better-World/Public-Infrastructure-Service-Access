from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wfp_access.artifacts.fingerprint import stable_fingerprint
from wfp_access.artifacts.spec import ArtifactSpec, CachePolicy
from wfp_access.data.registry import DatasetHandle, DatasetKey, DatasetRegistry
from wfp_access.storage.repository import Repository


@dataclass
class ArtifactRunner:
    """Interprets artifact metadata around otherwise plain functions."""

    repository: Repository
    registry: DatasetRegistry

    def run(self, func: Callable[..., Any], *args, **kwargs) -> DatasetHandle:
        spec: ArtifactSpec | None = getattr(func, "__artifact_spec__", None)
        if spec is None:
            result = func(*args, **kwargs)
            key = DatasetKey(name=func.__name__)
            return self.registry.register(DatasetHandle(key=key, data=result))

        fingerprint = stable_fingerprint(
            {
                "function": f"{func.__module__}.{func.__name__}",
                "artifact": spec.name,
                "args": args,
                "kwargs": kwargs,
                "schema": spec.schema,
                "contract": spec.contract_version,
            }
        )
        key = DatasetKey(name=spec.name, uri=self.repository.artifact_uri(spec, fingerprint))

        if spec.cache_policy != CachePolicy.NEVER and self.repository.exists(key.uri):
            return self.registry.register(self.repository.read_dataset(key, schema=spec.schema))

        result = func(*args, **kwargs)
        handle = DatasetHandle(key=key, data=result, schema=spec.schema, fingerprint=fingerprint)
        self.repository.write_dataset(handle, spec)
        return self.registry.register(handle)
