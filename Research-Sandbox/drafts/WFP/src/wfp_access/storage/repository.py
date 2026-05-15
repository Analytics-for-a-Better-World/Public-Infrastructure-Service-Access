from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wfp_access.artifacts.spec import ArtifactSpec
from wfp_access.data.registry import DatasetHandle, DatasetKey
from wfp_access.data.schemas import Schema
from wfp_access.storage.codecs import StorageCodec


@dataclass
class Repository:
    """Physical storage boundary for all reads and writes."""

    root: Path
    default_format: str = "parquet"
    codecs: dict[str, StorageCodec] = field(default_factory=dict)

    def register_codec(self, codec: StorageCodec) -> None:
        for fmt in codec.formats:
            self.codecs[fmt] = codec

    def artifact_uri(self, spec: ArtifactSpec, fingerprint: str) -> str:
        suffix = self.default_format if spec.format_role in {"table", "geotable"} else spec.format_role
        return str(self.root / "artifacts" / spec.name / f"{fingerprint}.{suffix}")

    def exists(self, uri: str | None) -> bool:
        return bool(uri) and Path(uri).exists()

    def read_dataset(self, key: DatasetKey, schema: Schema | None = None) -> DatasetHandle:
        if not key.uri:
            raise ValueError(f"Cannot read dataset without URI: {key.name}")
        path = Path(key.uri)
        fmt = path.suffix.removeprefix(".")
        codec = self.codecs.get(fmt)
        if codec is None:
            raise NotImplementedError(f"No storage codec registered for {fmt!r}")
        return DatasetHandle(key=key, data=codec.read(path), schema=schema)

    def write_dataset(self, handle: DatasetHandle, spec: ArtifactSpec) -> None:
        if not handle.key.uri:
            raise ValueError(f"Cannot write dataset without URI: {handle.key.name}")
        path = Path(handle.key.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = path.suffix.removeprefix(".")
        codec = self.codecs.get(fmt)
        if codec is None:
            raise NotImplementedError(f"No storage codec registered for {fmt!r}")
        codec.write(handle.data, path)
