from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class StorageCodec(Protocol):
    formats: set[str]

    def read(self, path: Path, **kwargs) -> Any: ...

    def write(self, data: Any, path: Path, **kwargs) -> None: ...


class UnsupportedCodec:
    """Placeholder used until optional backend codecs are registered."""

    formats: set[str] = set()

    def read(self, path: Path, **kwargs) -> Any:
        raise NotImplementedError(f"No codec registered for reading {path.suffix}")

    def write(self, data: Any, path: Path, **kwargs) -> None:
        raise NotImplementedError(f"No codec registered for writing {path.suffix}")
